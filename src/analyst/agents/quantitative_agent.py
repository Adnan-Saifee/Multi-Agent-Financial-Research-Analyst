# 1. Read the prompt and decide what filters to use.
# 2. Filters would ideally say: Ticker, Form name (based on keywords: yearly or monthly)
# 3. And then go through each accession_number folder and look at the different summaries?
# 4. Once found what table to use, find the table number using the block_index.
# 5. Copy the entire block (heading, table) and add as context.

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

import re
from typing import Optional
from pathlib import Path
import json

from analyst.config import settings
from analyst.log import print_verbose
from analyst.ingestion.table_indexer import _split_into_blocks, _parse_block

SELECTION_SYSTEM_PROMPT = """You are a financial data retrieval router. You will be given a \
question and an index of available financial tables across several SEC filings.
 
Each index entry describes one table: which company (ticker), which filing, and what \
metrics/time periods it covers.
 
Your job: identify which table(s) — if any — are needed to answer the question. You must \
figure out which company/ticker the question refers to yourself from the question text.
 
Respond ONLY with a valid JSON object, no markdown fences, no explanation:
{
  "selections": [
    {"accession_number": "...", "table_id": "table_000"},
    {"accession_number": "...", "table_id": "table_002"}
  ]
}
 
If no available table can answer the question, respond with: {"selections": []}
Select the minimum set of tables needed. Do not select tables "just in case"."""
 
ANSWER_SYSTEM_PROMPT = """You are a financial research analyst. Answer the question using \
ONLY the table data provided below. Do not use outside knowledge.
 
Rules:
- After every claim, add an inline citation in this exact format:
  [TICKER | FILING_TYPE | DATE | table]
- All figures are in millions of USD unless the table context states otherwise.
- If the provided tables do not contain enough information to answer, say exactly:
  "The available filings do not contain sufficient information to answer this question."
- Do not guess, infer beyond what the table states, or fabricate figures.
- Be precise with numbers — copy them exactly as they appear in the table.
 
Tables:
{context}"""

def _format_indexes_for_selection(indexes: list[dict]):
    # Returns the tables metadata formatted properly.
    lines = []
    for idx in indexes:
        for table in idx["tables"]:
            lines.append(
                f"- accession_number: {idx['accession_number']} | "
                f"table_id: {table['table_id']} | "
                f"ticker: {idx['ticker']} | "
                f"filing_type: {idx['filing_type']} | "
                f"filing_date: {idx['filing_date']} | "
                f"summary: {table['summary']} | "
                f"metrics: {', '.join(table['metrics_covered'])} | "
                f"periods: {', '.join(table['time_periods'])}"
            )
    return "\n".join(lines)

def _strip_json_fences(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)
    return raw

class QuantitativeAgent:

    def __init__(self, verbose: bool = False, raw_dir: Optional[Path] = None, model: str = settings.GROQ_MODEL_LARGE):
        self._raw_dir = Path(raw_dir or settings.RAW_DATA_DIR)
        self._filings_root = self._raw_dir / "sec-edgar-filings"
        self.verbose = verbose

        self._llm = init_chat_model(
            model = model,
            model_provider="groq"
        )

    def _load_all_table_indexes(self) -> list[dict]:
        # Returns all table_index.json files appended as dicts in a list

        indexes = []
        for accession_dir in sorted(self._filings_root.glob("*/*/*")):
            index_file = accession_dir / "table_index.json"

            if not index_file:
                continue

            data = json.loads(index_file.read_text(encoding="utf-8"))

            if not data.get("tables"):
                # No indexable tables in this file -> Skip to next accession directory
                continue

            data["accession_dir"] = accession_dir

            indexes.append(data)
        
        return indexes
    
    def _prefilter_indexes(self, question: str, indexes: list[dict]) -> list[dict]:
        """
        Cheap pre-filter before the LLM selection call.
        Reduces the index menu to only plausibly relevant filings.
        """
        question_lower = question.lower()

        # Ticker name -> ticker symbol mapping
        ticker_hints = {
            "apple": "AAPL", "aapl": "AAPL",
            "microsoft": "MSFT", "msft": "MSFT",
            "nvidia": "NVDA", "nvda": "NVDA",
        }

        # Filing type hints
        annual_hints = ["annual", "yearly", "10-k", "fiscal year"]
        quarterly_hints = ["quarter", "quarterly", "10-q", "q1", "q2", "q3", "q4"]

        # Detect mentioned tickers
        mentioned_tickers = {
            symbol for name, symbol in ticker_hints.items()
            if name in question_lower
        }

        # Detect filing type preference
        wants_annual = any(h in question_lower for h in annual_hints)
        wants_quarterly = any(h in question_lower for h in quarterly_hints)

        filtered = []
        for idx in indexes:
            # Filter by ticker if question mentions specific companies
            if mentioned_tickers and idx["ticker"] not in mentioned_tickers:
                continue

            # Filter by filing type if question is specific
            if wants_annual and not wants_quarterly and idx["filing_type"] != "10-K":
                continue
            if wants_quarterly and not wants_annual and idx["filing_type"] != "10-Q":
                continue

            filtered.append(idx)

        # Safety fallback — if pre-filter was too aggressive, return everything
        return filtered if filtered else indexes

    def _select_relevant_tables(self, question: str, indexes: list[dict]) -> list[dict]:
        # Returns a list dicts of selected tables.
        filtered_indexes = self._prefilter_indexes(question, indexes)
        formatted_indexes = _format_indexes_for_selection(indexes=filtered_indexes)

        messages = [
            SystemMessage(content=SELECTION_SYSTEM_PROMPT),
            HumanMessage(content=f"Question: {question}\n\nAvailable tables: {formatted_indexes}")
        ]

        response = self._llm.invoke(messages)

        try:
            parsed = json.loads(_strip_json_fences(response.content))
            return parsed.get("selections", [])
        except json.JSONDecodeError:
            print_verbose(self, error=True, message=f"Failed to parse table selection response: {response.content[:200]}")
            return []

    
    def _fetch_table_markdown(self, accession_dir: Path, table_id: str, indexes: list[dict]) -> Optional[str]:

        index_file = accession_dir / "table_index.json"
        index_data = json.loads(index_file.read_text(encoding="utf-8"))

        table_entry = next(
            (table for table in index_data["tables"] if table["table_id"]==table_id),
            None
        )

        if table_entry is None:
            print_verbose(self,error=True, message=f"table_id {table_id} not found in {index_file}")
        
        tables_md = (accession_dir / "mda_tables.md").read_text(encoding="utf-8")
        blocks = _split_into_blocks(tables_md=tables_md)

        block_index = table_entry["block_index"]
        if block_index >= len(blocks):
            print_verbose(self, error=True, message=f"Block index: {block_index} is out of range.")
        
        context_text, table_markdown = _parse_block(blocks[block_index])
        
        return f"{context_text}\n\n{table_markdown}"

    def answer(self, question: str):

        all_indexes = self._load_all_table_indexes()
        if not all_indexes:
            print_verbose(self, error=True, message="No table_index.json files found — has build_table_index.py run?")
            return {
                "answer": "No indexed financial tables are available.",
                "tables_used": [],
                "found_data": False,
            }

        selections = self._select_relevant_tables(question, all_indexes)

        if not selections or len(selections) <= 0:
            print_verbose(self, error=True, message="[ERROR] No tables selections were received.")
            return {
                "answer": "The available filings do not contain sufficient information to answer this question.",
                "tables_used": [],
                "found_data": False,
            }
        
        accession_lookup = {idx["accession_number"]: idx["accession_dir"] for idx in all_indexes}
        meta_lookup = {
            idx["accession_number"]: (idx["ticker"], idx["filing_type"], idx["filing_date"])
            for idx in all_indexes
        }
        context_blocks = []
        tables_used = []
        for sel in selections:
            accession_number = sel.get("accession_number")
            accession_dir = accession_lookup.get(accession_number)
            table_id = sel.get("table_id")

            if accession_dir is None:
                print_verbose(self, error=True, message=f"Selected accession_number {accession_number} not found in index.")
                continue

            table_markdown = self._fetch_table_markdown(accession_dir, table_id, all_indexes)

            ticker, filing_type, filing_date = meta_lookup[accession_number]
            header = f"[{ticker} | {filing_type} | {filing_date} | {table_id}]"
            context_blocks.append(f"{header}\n{table_markdown}")
            tables_used.append(sel)

        if not context_blocks:
            print_verbose(self, error=True, message="[ERROR] No table context provided.")
            return {
                "answer": "The available filings do not contain sufficient information to answer this question.",
                "tables_used": [],
                "found_data": False,
            }

        context = "\n\n---\n\n".join(context_blocks)

        messages = [
            SystemMessage(content=ANSWER_SYSTEM_PROMPT.format(context=context)),
            HumanMessage(content=question)
        ]

        response = self._llm.invoke(messages)

        return {
            "answer": response.content,
            "tables_used": tables_used,
            "found_data": True
        }


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    agent = QuantitativeAgent(model="meta-llama/llama-4-scout-17b-16e-instruct")
    response = agent.answer("Can you help provide information on Apple's operating expenses for periods ending June 2024 and 2025?")

    print(response["answer"])
    print("="*60)
    print(f"TABLES USED: {len(response["tables_used"])}")
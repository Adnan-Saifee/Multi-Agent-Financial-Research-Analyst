import json
import re
import time
from pathlib import Path
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from analyst.config import settings
from analyst.log import print_verbose

# Separator that marks the start of each table block in mda_tables.md
TABLE_BLOCK_SEPARATOR = "### Table Context / Title:"

# Minimum number of numeric cells a table must have to be considered
MIN_NUMERIC_CELLS = 3

# How many pipe rows a table must have to be worth indexing.
MIN_TABLE_ROWS = 2

# Groq rate limit free tier allows ~30 requests/minute.
# Sleep briefly between LLM calls to avoid hitting it.
RATE_LIMIT_SLEEP_SECONDS = 2.5

SUMMARIZE_SYSTEM_PROMPT = """You are a financial data indexer. Your job is to write a precise, 
searchable summary of a financial table from a SEC filing.

You will be given:
- Filing metadata (company, filing type, date)
- The context text above the table (heading, units, explanatory sentence)
- The raw markdown table itself

Important notes:
- All figures are in millions of USD unless the context explicitly states otherwise
- If units are stated in the context (e.g. "In millions, except percentages"), include that
- Be specific about metric names, time periods, and breakdowns — this summary will be 
  used to decide whether to retrieve this table for a given question
- If the table contains no numeric financial data (e.g. it is a glossary of metric 
  definitions or a signature block), set is_numeric to false

Respond ONLY with a valid JSON object — no markdown fences, no preamble, no explanation.
Use this exact schema:
{
  "summary": "2-3 sentence description of what this table contains",
  "metrics_covered": ["metric1", "metric2"],
  "time_periods": ["2024", "2025"],
  "is_numeric": true
}"""


def _build_summarize_prompt(
    ticker: str,
    filing_type: str,
    filing_date: str,
    table_context: str,
    table_markdown: str,
) -> str:
    return f"""Filing: {ticker} {filing_type} dated {filing_date}
All figures in millions USD unless stated otherwise in the context below.

Table context (heading + surrounding text):
{table_context.strip()}

Table markdown:
{table_markdown.strip()}"""


# Table block parsing 

def _split_into_blocks(tables_md: str) -> list[str]:
    # Split mda_tables.md content into individual table blocks.
    
    raw_blocks = tables_md.split(TABLE_BLOCK_SEPARATOR)
    # First split is everything before the first marker — discard it
    blocks = [b.strip() for b in raw_blocks[1:] if b.strip()]
    return blocks


def _parse_block(block: str) -> tuple[str, str]:
    # Split a single table block into (context_text, table_markdown).

    lines = block.splitlines()
    context_lines = []
    table_lines = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        if not in_table and (stripped.startswith('|') or re.match(r'^[\s|:-]+$', stripped)):
            in_table = True
        if in_table:
            table_lines.append(line)
        else:
            context_lines.append(line)

    return "\n".join(context_lines).strip(), "\n".join(table_lines).strip()


def _count_numeric_cells(table_markdown: str) -> int:
    count = 0
    for line in table_markdown.splitlines():
        if not line.strip().startswith('|'):
            continue
        cells = line.split('|')
        for cell in cells:
            cleaned = re.sub(r'[$,%\s()\-]', '', cell.strip())
            if re.match(r'^\d+\.?\d*$', cleaned) and cleaned:
                count += 1
    return count


def _count_table_rows(table_markdown: str) -> int:
    
    return sum(
        1 for line in table_markdown.splitlines()
        if line.strip().startswith('|') and not re.match(r'^[\s|:-]+$', line.strip())
    )


def _is_indexable(table_markdown: str) -> bool:

    if not table_markdown:
        return False
    if _count_table_rows(table_markdown) < MIN_TABLE_ROWS:
        return False
    if _count_numeric_cells(table_markdown) < MIN_NUMERIC_CELLS:
        return False
    return True


# ── LLM summarization ──────────────────────────────────────────────────────

def _summarize_table(
    llm: ChatGroq,
    ticker: str,
    filing_type: str,
    filing_date: str,
    context_text: str,
    table_markdown: str,
) -> Optional[dict]:
    
    user_content = _build_summarize_prompt(
        ticker, filing_type, filing_date, context_text, table_markdown
    )

    messages = [
        SystemMessage(content=SUMMARIZE_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()

        # Strip markdown fences if the model added them despite instructions
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)

        parsed = json.loads(raw)

        required = {"summary", "metrics_covered", "time_periods", "is_numeric"}
        if not required.issubset(parsed.keys()):
            return None

        return parsed

    except json.JSONDecodeError:
        return None
    except Exception:
        return None


class TableIndexer:

    def __init__(self, raw_dir: Optional[Path] = None, verbose: bool = False):
        self.verbose = verbose
        self._raw_dir = Path(raw_dir or settings.RAW_DATA_DIR)
        self._filings_root = self._raw_dir / "sec-edgar-filings"

        self._llm = ChatGroq(
            model=settings.GROQ_MODEL_LARGE,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
        )

    # ── Public interface ───────────────────────────────────────────────

    def index_all(self, force: bool = False) -> None:
        # force = True will reindex every table even if they already exist.

        accession_dirs = sorted(self._filings_root.glob("*/*/*"))
        accession_dirs = [d for d in accession_dirs if d.is_dir()]

        print_verbose(self, f"Found {len(accession_dirs)} accession folder(s).")

        indexed = 0
        skipped = 0

        for accession_dir in accession_dirs:
            index_file = accession_dir / "table_index.json"

            if index_file.exists() and not force:
                print_verbose(self, f"[{accession_dir.name}] Already indexed — skipping.")
                skipped += 1
                continue

            if not (accession_dir / "mda_tables.md").exists():
                print_verbose(self, f"[{accession_dir.name}] No mda_tables.md — skipping.", error=True)
                skipped += 1
                continue

            if not (accession_dir / "metadata.json").exists():
                print_verbose(self, f"[{accession_dir.name}] No metadata.json — skipping.", error=True)
                skipped += 1
                continue

            self.index_accession(accession_dir)
            indexed += 1

        print_verbose(
            self,
            f"Done. {indexed} accession(s) indexed, {skipped} skipped.",
            local_verbose=True,
        )

    def index_accession(self, accession_dir: Path) -> Optional[Path]:
        
        metadata    = json.loads((accession_dir / "metadata.json").read_text())
        ticker      = metadata.get("ticker", "UNKNOWN")
        filing_type = metadata.get("filing_type", "UNKNOWN")
        filing_date = metadata.get("filing_date", "UNKNOWN")

        print_verbose(self, f"Indexing [{ticker}] {filing_type} {filing_date} ...")

        tables_md = (accession_dir / "mda_tables.md").read_text(encoding="utf-8")
        blocks = _split_into_blocks(tables_md)

        if not blocks:
            print_verbose(self, f"[{ticker} {filing_type}] No table blocks found.", error=True)
            return None

        print_verbose(self, f"[{ticker}] Found {len(blocks)} block(s), filtering...")

        table_entries = []
        table_index = 0

        for block_index, block in enumerate(blocks):
            context_text, table_markdown = _parse_block(block)

            if not _is_indexable(table_markdown):
                print_verbose(
                    self,
                    f"  Block {block_index}: skipped "
                    f"({_count_numeric_cells(table_markdown)} numeric cells, "
                    f"{_count_table_rows(table_markdown)} rows)."
                )
                continue

            print_verbose(self, f"  Summarizing table {table_index} (block {block_index})...")

            if table_index > 0:
                time.sleep(RATE_LIMIT_SLEEP_SECONDS)

            summary_data = _summarize_table(
                llm=self._llm,
                ticker=ticker,
                filing_type=filing_type,
                filing_date=filing_date,
                context_text=context_text,
                table_markdown=table_markdown,
            )

            if summary_data is None:
                print_verbose(self, f"  Table {table_index}: summarization failed — skipping.", error=True)
                continue

            if not summary_data.get("is_numeric", True):
                print_verbose(self, f"  Table {table_index}: LLM flagged as non-numeric — skipping.")
                continue

            table_entries.append({
                "table_id": f"table_{table_index:03d}",
                "block_index": block_index,
                "summary": summary_data["summary"],
                "metrics_covered": summary_data["metrics_covered"],
                "time_periods": summary_data["time_periods"],
                "source_file": "mda_tables.md",
            })
            table_index += 1

        index_doc = {
            "ticker":           ticker,
            "filing_type":      filing_type,
            "filing_date":      filing_date,
            "accession_number": metadata.get("accession_number", accession_dir.name),
            "source_file":      "mda_tables.md",
            "tables":           table_entries,
        }

        out_path = accession_dir / "table_index.json"
        out_path.write_text(json.dumps(index_doc, indent=2), encoding="utf-8")

        print_verbose(
            self,
            f"[{ticker} {filing_type} {filing_date}] "
            f"Indexed {len(table_entries)} table(s) → {out_path.name}",
            local_verbose=True,
        )

        return out_path

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    indexer = TableIndexer(verbose=True)
    indexer.index_all()
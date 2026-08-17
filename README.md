# Multi-Agent Financial Research Analyst

A production-style multi-agent AI system that answers natural language questions over SEC filings (10-K and 10-Q) for Apple (AAPL), Microsoft (MSFT), and Nvidia (NVDA). The system retrieves, verifies, and synthesizes cited research memos by orchestrating specialized agents through a conditional LangGraph execution graph.

---

## The Problem

Fundamental equity research requires analysts to manually read hundreds of pages of SEC filings to extract facts, figures, and management commentary. This system automates that retrieval, answering questions like:

> *"How did Apple's gross margin change in fiscal year 2025, and what did management attribute it to?"*

by pulling exact figures from financial tables and narrative explanations from filing text simultaneously, then verifying every claim against its source before delivering a unified, cited answer.

---

## Architecture

```
User Question
      ↓
  [Planner]  ←  Router classifies question, extracts tickers + time scope
      ↓
  ┌───────────────────────────┐
  │  Parallel Fan-Out         │
  │  [Qualitative Agent]      │  ← Tool-calling ReAct agent over ChromaDB
  │  [Quantitative Agent]     │  ← Vectorless RAG over structured table index
  └───────────────────────────┘
      ↓
  [Critic]  ←  Re-fetches sources, verifies every cited claim
      ↓
  conditional routing:
    approved  →  [Synthesizer]
    rejected  →  loop back to failed agent(s) with feedback
      ↓
  [Synthesizer]  ←  Merges verified answers into one cited research memo
      ↓
  Final Answer
```

### Key Design Decisions

**Dual Retrieval Architecture**

The system uses two fundamentally different retrieval paths depending on the question type:

- **Narrative path (Qualitative Agent):** Semantic vector search over ChromaDB, indexing MD&A and Risk Factor sections from all filings. Handles questions or parts of questions that require a narrative answer.
- **Structured path (Quantitative Agent):** Vectorless RAG — an LLM reasons over a pre-built index of table summaries to identify which specific financial table answers the question, then fetches the raw markdown table directly. No embedding similarity is used. This eliminates the numeric hallucination that occurs when financial tables are embedded and retrieved by cosine similarity.

**Tool-Calling Qualitative Agent**

The Qualitative Agent is a genuine ReAct tool-calling agent built with LangChain's `create_agent`. It dynamically decides which tools to call and how many times, adapting its retrieval strategy to the question at runtime. Tools available:

- `create_search_query` — isolates the qualitative portion of a mixed question
- `search_filings` — semantic search with optional ticker and section filters
- `grade_chunks` — LLM-based relevancy grading of retrieved context
- `escalate_to_quantitative` — signals mid-retrieval that the question needs the numeric path (qualitative-only mode only)

**Independent Critic Verification**

The Critic re-fetches source documents using the citation tags embedded in each answer — it does not trust what the agents passed through state. For each claim it asks: "does this source actually say this?" Claims that fail verification are sent back to the originating agent with specific feedback. Separate retry counters per agent ensure the Critic loop can't run indefinitely.

**Adversarial Evaluation**

A deliberate wrong figure is injected into a test document to verify the Critic catches it. This is the most important validation test — it proves the verification layer works rather than just runs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent Orchestration | LangGraph (StateGraph, conditional edges, parallel fan-out) |
| Agent Framework | LangChain (`create_agent`, tool-calling, ReAct) |
| LLM Provider | Groq (llama / openai-oss models) |
| Vector Store | ChromaDB with HuggingFace embeddings (`all-MiniLM-L6-v2`) |
| Filing Data | SEC EDGAR via edgartools (markdown extraction) |
| Evaluation | LangSmith (trajectory, faithfulness, context precision) |
| Configuration | pydantic-settings |
| Language | Python 3.11+ |

---

## Setup

**1. Clone and install**
```bash
git clone https://github.com/your-username/financial-research-analyst
cd financial-research-analyst
pip install -e .
```

**2. Configure environment**
```bash
cp .env.example .env
# Fill in: GROQ_API_KEY, SEC_USER_AGENT_NAME, SEC_USER_AGENT_EMAIL
# Optional: LANGSMITH_API_KEY, LANGSMITH_PROJECT
```

**3. Download filings**
```bash
python scripts/download_filings.py
```

**4. Build narrative index**
```bash
python scripts/build_index.py
```

**5. Build table index**
```bash
python scripts/build_table_index.py
```

**6. Run**
```bash
python scripts/run.py "What are Apple's main risk factors and how did they affect gross margin in fiscal 2025?"
```

---

## Example Questions

**Narrative only**
```
What did Microsoft's management say caused the decline in More Personal Computing revenue?
What supply chain risks does Apple consistently highlight across its filings?
```

**Quantitative only**
```
What was Nvidia's total revenue in fiscal year 2024?
How did Apple's Services gross margin percentage change from 2023 to 2025?
```

**Both paths (parallel)**
```
How did Microsoft's cloud revenue grow in 2025 and what drove it?
What are Apple's main risk factors and how have they affected gross margin?
```

---

## How Retrieval Works

### Narrative Path (ChromaDB + MMR)
Filings are parsed into clean markdown, split into semantically coherent chunks at heading boundaries, and embedded using `all-MiniLM-L6-v2`. At query time, Maximal Marginal Relevance (MMR) retrieval balances relevance with diversity, preventing duplicate boilerplate chunks (common in SEC filings) from dominating results.

### Structured Path (Vectorless RAG)
During ingestion, an LLM generates a structured summary for each financial table in each filing, saved as `table_index.json` per accession folder. At query time:
1. The Quantitative Agent pre-filters indexes by ticker and filing type
2. An LLM selects which specific table(s) answer the question by reading the summaries
3. The raw markdown table is fetched directly from `mda_tables.md` using the stored `block_index`
4. The LLM answers from the raw table with exact figures

No embeddings are used anywhere in this path.

---

## Evaluation

The system is evaluated using LangSmith across three metrics:

- **Faithfulness** — are all claims grounded in retrieved source documents?
- **Context Precision** — did retrieval return relevant chunks?
- **Trajectory Accuracy** — did agents call the right tools in the right sequence?

The adversarial test deliberately injects an incorrect figure into a test document and verifies the Critic flags it before the answer reaches the Synthesizer.

---

## Notes

- All figures in financial tables are in millions of USD unless stated otherwise in the table context
- The system only answers from indexed filings — no real-time data, no predictions
- Adding new companies requires updating `TICKERS` in `edgar_client.py` and re-running the ingestion pipeline
- Groq free tier limits apply — the table indexer includes rate limiting between LLM calls

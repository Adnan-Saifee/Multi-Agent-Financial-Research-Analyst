# 1. Call edgar_client (Optional)
# 2. Call Chunker
# 3. Call VectorStore
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analyst.ingestion.edgar_client import EdgarClient
from analyst.ingestion.chunking import MarkdownChunker
from analyst.retrieval.vector_store import VectorStore
from analyst.log import print_verbose
from analyst.config import settings
import json

SECTIONS = {
    "mda": "mda_text.md",
    "risk_factors": "risk_factors_text.md",
}

def main(clean: bool = False, fetch_data: bool = False) -> None:

    filings_root = Path(settings.RAW_DATA_DIR) / "sec-edgar-filings"
    store = VectorStore()
    chunker = MarkdownChunker()

    if clean:
        store.delete_collection()

    if fetch_data:
        client = EdgarClient(verbose=False, tickers=settings.TICKERS, max_10k=3, max_10q=4)
        client.fetch_all()
    
    for accession_dir in sorted(filings_root.glob("*/*/*")):
        if not accession_dir.is_dir():
            continue

        metadata_file = accession_dir / "metadata.json"
        if not metadata_file.exists():
            continue
        
        metadata = json.loads(metadata_file.read_text())
        sections = { 
            name: (accession_dir / filename).read_text(encoding="utf-8")
            for name, filename in SECTIONS.items()
            if (accession_dir / filename).exists()
        }

        if not sections:
            continue

        chunks = chunker.chunk_all_sections(sections=sections, metadata=metadata)
        store.upsert_chunks(chunks=chunks)

if __name__ == "__main__":
    # main()
    store = VectorStore()
    retriever = store.as_retriever()
    chunks = retriever.invoke("Commercial paper program", filter={"ticker": "AAPL"})
    
    from pprint import pprint as pp

    pp(chunks)
import re
import logging
from dataclasses import dataclass, field
from typing import Optional
from analyst.log import print_verbose
from langchain_text_splitters import RecursiveCharacterTextSplitter

@dataclass
class Chunk: # Custom dataclass to ensure strict metadata structure in the future when using Document()
    chunk_id: str
    text: str
    heading: str # nearest markdown heading above this chunk
    token_count: int # approximate, based on whitespace split

    # Metadata from metadata.json
    ticker: str
    filing_type: str 
    filing_date: str # "YYYY-MM-DD"
    accession_number: str
    section: str

    def to_chroma_metadata(self) -> dict:
        # A dict that will be appended to langchain's Document objects when we convert Chunk -> Document
        return {
            "chunk_id": self.chunk_id,
            "heading": self.heading,
            "token_count": self.token_count,
            "ticker": self.ticker,
            "filing_type": self.filing_type,
            "filing_date": self.filing_date,
            "accession_number": self.accession_number,
            "section": self.section
        }

# Matches any markdown heading line: #, ##, ###, ####, etc.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

def _approximate_token_count(text: str) -> int:
    return len(text.split())

def _split_by_headings(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    last_end = 0
    last_heading = ""

    for match in _HEADING_RE.finditer(text):
        # Save the body of the previous heading
        body = text[last_end:match.start()].strip()
        if body:
            sections.append((last_heading, body))

        last_heading = match.group(2).strip()
        last_end = match.end()

    # The final section after the last heading
    body = text[last_end:].strip()
    if body:
        sections.append((last_heading, body))

    return sections # [(heading, section), ...]

# The actual Chunker classs
class MarkdownChunker:

    def __init__(
        self,
        max_chunk_tokens: int = 400,
        overlap_tokens: int = 50,
        min_chunk_tokens: int = 50,
        verbose: bool = False
    ):
        self.verbose = verbose
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

        chars_per_token = 4
        self._fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chunk_tokens * chars_per_token,
            chunk_overlap=overlap_tokens * chars_per_token,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def chunk_filing(
        self,
        text: str,
        metadata: dict,
        section: str,
    ) -> list[Chunk]:
        
        if not text:
            print_verbose(self, error=True, message=f"[{metadata.get('ticker', '?')}] Empty text passed to chunker for section '{section}'.")
            return []

        heading_sections = _split_by_headings(text)
        if not heading_sections:
            print_verbose(self, error=True, message=f"[{metadata.get('ticker', '?')}] No content found after heading split for '{section}'.")
            return []

        # heading_sections but with some of the tuples split into multiple due to large token counts
        raw_chunks: list[tuple[str, str]] = []  # (heading, chunk_text)

        for heading, body in heading_sections:
            token_count = _approximate_token_count(body)

            if token_count <= self.max_chunk_tokens:
                # Section fits in one chunk — keep it whole.
                raw_chunks.append((heading, body))
            else:
                # Section is too large — subdivide with the fallback splitter.
                print_verbose(
                    self,
                    f"Section '{heading}' has ~{token_count} tokens —> subdividing with RecursiveCharacterTextSplitter."
                )
                sub_texts = self._fallback_splitter.split_text(body)
                for sub in sub_texts:
                    raw_chunks.append((heading, sub))

        # Build Chunk objects, filtering out anything too small
        chunks: list[Chunk] = []
        chunk_index = 0

        ticker = metadata.get("ticker", "UNKNOWN")
        filing_type = metadata.get("filing_type", "UNKNOWN").replace("/", "-")
        filing_date = metadata.get("filing_date", "UNKNOWN")
        accession = metadata.get("accession_number", "UNKNOWN")

        for heading, chunk_text in raw_chunks:
            chunk_text = chunk_text.strip()
            token_count = _approximate_token_count(chunk_text)

            if token_count < self.min_chunk_tokens:
                print_verbose(
                    self,
                    f"Discarding short chunk under heading '{heading}'; (~{token_count} tokens < min {self.min_chunk_tokens})."
                )
                continue

            chunk_id = (
                f"{ticker}_{filing_type}_{filing_date}_{section}_{chunk_index:03d}"
            )

            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                heading=heading,
                token_count=token_count,
                ticker=ticker,
                filing_type=filing_type,
                filing_date=filing_date,
                accession_number=accession,
                section=section
            ))

            chunk_index += 1

        print_verbose(
            self,
            f"[{ticker} {filing_type} {filing_date}] Section '{section}': {len(chunks)} chunks produced."
        )
        return chunks

    def chunk_all_sections(
        self,
        sections: dict[str, str], # {"mda": "all the text for mda", "risk_factors": "text for risk"}
        metadata: dict,
    ) -> list[Chunk]:
        
        all_chunks: list[Chunk] = []
        for section_name, text in sections.items():
            all_chunks.extend(
                self.chunk_filing(text=text, metadata=metadata, section=section_name)
            )
        return all_chunks

if __name__ == "__main__":

    import json
    from pprint import pprint
    from pathlib import Path

    chunker = MarkdownChunker()
    accession_dir = Path("C:\\Users\\adnan\\Rutgers 2024\\Projects\\VSCode\\Multi-Agent Financial Research Analyst\\data\\raw\\sec-edgar-filings\\AAPL\\10-K\\0000320193-23-000106")
    metadata_file = accession_dir / "metadata.json"
    # metadata_file = "C:\Users\\adnan\Rutgers 2024\Projects\VSCode\Multi-Agent Financial Research Analyst\data\\raw\sec-edgar-filings\AAPL\\10-K\\0000320193-23-000106\\metadata.json"
    
    with open(metadata_file, 'r') as file:
        metadata = json.load(file)
    
    sections = {
        "mda": (accession_dir / "mda_text.md").read_text(encoding="utf-8"),
        "risk_factors": (accession_dir / "risk_factors_text.md").read_text(encoding="utf-8"),
    }

    chunks_list = chunker.chunk_all_sections(sections, metadata)
    pprint(chunks_list[:15])
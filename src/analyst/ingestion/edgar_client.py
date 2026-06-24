"""
analyst/ingestion/edgar_client.py

Wraps the `sec-edgar-downloader` package to pull 10-K and 10-Q filings
for a configured list of tickers, with local caching so re-running the
script doesn't re-hit SEC EDGAR for filings we already have on disk.
"""
# Doc string above helps with hover info

import logging
from pathlib import Path
from typing import List, Literal, Optional, Dict, Any
from edgar import set_identity, Company
import json
from pathlib import Path
from sec_edgar_downloader import Downloader

from analyst.config import settings
from analyst.log import print_verbose

class EdgarClient():
    def __init__(self, verbose: bool, tickers: List[str], max_10k: int, max_10q: int) -> None:
        self.verbose = verbose
        self.TICKERS = tickers

        # Tracks how many latest 10-K and 10-Q forms we want.
        self.MAX_10K = max_10k
        self.MAX_10Q = max_10q

        set_identity("Adnan Saifee adnan.saifee2006@gmail.com")


    def _get_filings(self, ticker: str, form: Literal["10-K", "10-Q"], num_filings: int):
        """
        Returns latest num_filings filings for a given ticker and form.
        """

        company = Company(ticker)
        filings = company.get_filings(form=form).latest(num_filings)

        return filings

    def _already_downloaded(self, raw_dir: Path, ticker: str, form: str) -> int:
        """
        Return how many filings of `form` type are already cached locally
        for `ticker`, by counting accession-number subfolders.

        sec-edgar-downloader saves filings to:
            {raw_dir}/sec-edgar-filings/{ticker}/{form}/{accession_number}/
        so counting subdirectories tells us how many filings we already have
        without needing to call the SEC API at all.
        """
        filing_dir = raw_dir / "sec-edgar-filings" / ticker / form
        if not filing_dir.exists():
            return 0
        return sum(1 for p in filing_dir.iterdir() if p.is_dir())

    def save_raw_sec_data(
        self,
        ticker: str, 
        form_name: str, 
        accession_no: str, 
        content: str,
        file_name: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:

        """
        Saves `content` into a text file into the raw data directory, and also saves a
        `metadata.json` file if a `metadata` python dictionary was provided.
        """
        base_dir = settings.RAW_DATA_DIR / "sec-edgar-filings"
        target_dir = base_dir / ticker.upper() / form_name.upper() / accession_no
        file_path = target_dir / file_name
        metadata_path = target_dir / "metadata.json"
        
        # Track if any new files are actually written
        action_taken = False

        # Handle text content saving safely
        if not file_path.exists():
            # Safely create nested directory structure if missing
            target_dir.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")

            print_verbose(self, f"Successfully saved new text filing: {file_path}")
            action_taken = True
        else:
            print_verbose(self, f" -> Text file already exists for Acc: {accession_no}. Skipping text write.")

        # Handle metadata saving safely if dictionary is provided
        if metadata is not None:
            if not metadata_path.exists():
                # Ensure directory exists (in case text saving was skipped but metadata is new)
                target_dir.mkdir(parents=True, exist_ok=True)
                
                # Write dictionary to a formatted JSON file
                with open(metadata_path, "w", encoding="utf-8") as json_file:
                    json.dump(metadata, json_file, indent=4, ensure_ascii=False)
                print_verbose(self, f"Successfully saved metadata: {metadata_path}")
                action_taken = True
            else:
                print_verbose(self, f" -> Metadata file already exists for Acc: {accession_no}. Skipping JSON write.")

        return action_taken

    def fetch_filings_for_ticker(self, ticker: str) -> None:
        """
        Pull the most recent 10-K and 10-Q filings
        for a single ticker, skipping anything already cached locally.
        """
        raw_dir = Path(settings.RAW_DATA_DIR)

        # --- 10-K: latest self.MAX_10K only ---
        existing_10k = self._already_downloaded(raw_dir, ticker, "10-K")
        if existing_10k >= self.MAX_10K:
            print_verbose(self, f"[{ticker}] 10-K already cached — skipping fetch.")
        else:
            print_verbose(self, f"[{ticker}] Fetching latest 10-K...")
            
            try:
                # First index: Latest filing
                filings = self._get_filings(form="10-K", ticker=ticker, num_filings=self.MAX_10K)

                # Will save each accession as raw text with a metadata.json file with the following details:
                # filing.obj() -> saved to \raw\... as text
                # metadata.json -> {"accession": 1141231-14314-xxx, "fiscal": 2015, "form": "10-K"}
                for filing in filings:
                    object = filing.obj()
                    text_item_1A = object["Item 1A"]      
                    text_item_7 = object["Item 7"]

                    metadata_dict = {
                        "ticker": ticker,
                        "filing_type": "10-K",
                        "fiscal_year": int(filing.period_of_report[:4]),
                        "accession_number": filing.accession_no
                    }

                    self.save_raw_sec_data(ticker=ticker, form_name="10-K", accession_no=filing.accession_no, content=text_item_1A, file_name="risk_factors.txt")
                    self.save_raw_sec_data(ticker=ticker, form_name="10-K", accession_no=filing.accession_no, content=text_item_7, file_name="mda.txt", metadata=metadata_dict)
            
                print_verbose(self, f"[{ticker}] Downloaded {len(filings)} 10-K filing(s).")
            except Exception as e:
                print_verbose(self, f"\nRecieved ERROR! Exception: {e}\n", True)

            
        # --- 10-Q: latest self.MAX_10Q filings ---
        existing_10q = self._already_downloaded(raw_dir, ticker, "10-Q")
        if existing_10q >= self.MAX_10Q:
            print_verbose(self, f"[{ticker}] {existing_10q} 10-Q filing(s) already cached — skipping fetch.")
        else:
            print_verbose(self, f"[{ticker}] Fetching latest 10-Q...")
            
            try:
                # First index: Latest filing
                filings = self._get_filings(form="10-Q", ticker=ticker, num_filings=self.MAX_10Q)

                for filing in filings:
                    object = filing.obj()
                    text_item_1A = object["Item 1A"] # Risking Factors  
                    text_item_2 = object["Item 2"] # MDA

                    metadata_dict = {
                        "ticker": ticker,
                        "filing_type": "10-K",
                        "fiscal_year": int(filing.period_of_report[:4]),
                        "accession_number": filing.accession_no
                    }

                    self.save_raw_sec_data(ticker=ticker, form_name="10-Q", accession_no=filing.accession_no, content=text_item_1A, file_name="risk_factors.txt")
                    self.save_raw_sec_data(ticker=ticker, form_name="10-Q", accession_no=filing.accession_no, content=text_item_2, file_name="mda.txt", metadata=metadata_dict)
            
                print_verbose(self, f"[{ticker}] Downloaded {len(filings)} 10-Q filing(s).")
            except Exception as e:
                print_verbose(self, f"\n[ERROR] {e}\n", True, error=True)


    def fetch_all(self) -> None:
        """
        Fetch 10-K + 10-Q filings for every configured ticker.
        Safe to call repeatedly — already-cached filings are skipped.
        """
        
        for ticker in self.TICKERS:
            self.fetch_filings_for_ticker(ticker)


if __name__ == "__main__":
    # Companies to ingest for this project.
    TICKERS: List[str] = ["AAPL", "MSFT", "NVDA"]

    # How many quarters of 10-Q history to pull per company
    client = EdgarClient(verbose=True, tickers=TICKERS, max_10k=3, max_10q=4)
    client.fetch_all()
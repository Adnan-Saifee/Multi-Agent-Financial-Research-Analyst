
import re
from pathlib import Path
from edgar import Company
from analyst.log import print_verbose

class FilingParser:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.full_markdown = ""
        self.current_accession_no = None  # Track the active filing ID

    def _load_filing_markdown(self, filing) -> str:
        # Check if this is a brand new filing document by comparing accession numbers
        if self.current_accession_no != filing.accession_no:
            print_verbose(self, f"New filing detected ({filing.accession_no}). Clearing cache...")
            self.full_markdown = ""
            self.current_accession_no = filing.accession_no

        # Now the cache only populates ONCE per unique filing document
        if not self.full_markdown:    
            print_verbose(self, "Converting full filing to Markdown strings...")
            self.full_markdown = filing.markdown()

        return self.full_markdown

    def extract_section(self, filing, start_item: str, end_item: str) -> str:
        """
        Slices out a specific chunk of the document between two item headings.
        Example: start_item="Item 7", end_item="Item 7A" -> This returns the Item 7 section.
        """
        markdown_data = self._load_filing_markdown(filing)
        
        print_verbose(self, f"Slicing boundaries from {start_item} to {end_item}...")
        # Strips out spacing and builds a resilient regex pattern for Markdown headers
        # pattern = re.compile(
        #     rf'(##\s*{re.escape(start_item)}\b.*?)(##\s*{re.escape(end_item)}\b)', 
        #     re.DOTALL | re.IGNORECASE
        # )

        pattern = re.compile(
            rf'(##\s*{re.escape(start_item)}[\s.,-].*?)(##\s*{re.escape(end_item)}[\s.,-]|^\s*{re.escape(end_item)}[\s.,-])', 
            re.DOTALL | re.IGNORECASE | re.MULTILINE
        )
        
        match = pattern.search(markdown_data)
        if not match:
            # Broader fallback if the document layout skips standard Markdown header tags
            print_verbose(self, error=True, message="Falling back to broader regex pattern matching", local_verbose=True)
            # pattern = re.compile(
            #     rf'({re.escape(start_item)}\b.*?)({re.escape(end_item)}\b)', 
            #     re.DOTALL | re.IGNORECASE
            # )

            pattern = re.compile(
                rf'(^\s*{re.escape(start_item)}[\s.,-].*?)(^\s*{re.escape(end_item)}[\s.,-])', 
                re.DOTALL | re.IGNORECASE | re.MULTILINE
            )

            match = pattern.search(markdown_data)

        if not match:
            raise ValueError(f"Could not isolate section boundaries between {start_item} and {end_item}.")
            
        return match.group(1)

    def parse_text_and_tables(self, section_markdown: str):
        """
        Processes a raw section block to extract text and tables (along with their titles/headings and sub-headings) separately.
        `Returns` .md (markdown) content strings for the text and tables separately.
        """
        lines = section_markdown.splitlines()
        
        text_content = []
        table_content = []
        lines_to_remove_from_text = set()
        in_table = False
        
        print_verbose(self, "Parsing chunk lines: separating markdown rows and context titles...")
        for idx, line in enumerate(lines):
            stripped = line.strip()
            
            # CRITICAL FIX: Skip and completely delete any lines containing HTML fragments/page breaks
            if "<div" in stripped:
                continue
            
            # Detect Markdown table rows bounded by pipes
            if stripped.startswith('|') and stripped.endswith('|'):
                
                # Look-back engine to capture table titles right as a new table block is entered
                if not in_table:
                    in_table = True
                    title_lines = []
                    look_back = idx - 1
                    blank_count = 0

                    passed_text_lines = 0  # track whether we've seen real text since starting

                    while look_back >= 0 and len(title_lines) < 6:
                        prev_line = lines[look_back].strip()

                        if prev_line.endswith('|') or re.match(r'^[\s|:-]+$', prev_line):
                            if passed_text_lines > 0:
                                # We already grabbed real text lines above, now hitting pipes again
                                # — this is a previous table's boundary, hard stop
                                break
                            else:
                                # Still in the header rows of the current table, skip past them
                                look_back -= 1
                                continue

                        if prev_line == "":
                            blank_count += 1
                            if blank_count > 2:
                                break
                            look_back -= 1
                            continue

                        blank_count = 0
                        passed_text_lines += 1
                        title_lines.insert(0, lines[look_back])
                        look_back -= 1
                    
                    if title_lines:
                        table_content.append("\n### Table Context / Title:")
                        for t_line in title_lines:
                            table_content.append(t_line)
                            lines_to_remove_from_text.add(t_line)
                        table_content.append("") # Margin space before pipe syntax elements
                
                table_content.append(line)
                
            elif in_table and stripped == "":
                table_content.append(line)
            else:
                in_table = False
                # Filter structural syntax artifacts from standard text arrays
                if not re.match(r'^[\s|:-]+$', stripped):
                    text_content.append(line)

        # Scrub title strings from the text file array
        clean_text_content = [line for line in text_content if line not in lines_to_remove_from_text]

        # Joining them to return them as strings
        clean_text_content = "\n".join(clean_text_content)
        table_content = "\n".join(table_content)
    
        return clean_text_content, table_content


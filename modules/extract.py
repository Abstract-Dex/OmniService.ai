"""
PDF text & table extraction using pdfplumber.

Takes a PDF file path and returns a JSON string containing:
  - source metadata (file path, total pages)
  - per-page data: raw_text, tables (list of dicts), has_tables flag

This is the first step in the ingestion pipeline:
    extract (this) -> vector_ingest -> llm_parser -> graph_store
"""

import pdfplumber
import json
import logging

# Suppress noisy pdfminer debug logs (scoped to pdfminer only, not the root logger)
logging.getLogger("pdfminer").setLevel(logging.ERROR)


def extract_pdf_contents(pdf_path: str) -> str:
    """
    Extract every page of a PDF into a structured JSON string.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A JSON string with the structure:
        {
            "source": {"path": "...", "total_pages": N},
            "pages": [
                {
                    "page_number": 1,
                    "raw_text": "...",
                    "tables": [{"table_index": 0, "table_data": [[...]]}],
                    "has_tables": true/false
                }, ...
            ]
        }
    """
    with pdfplumber.open(pdf_path) as pdf:
        pages_list = []

        for page in pdf.pages:
            # Extract plain text from the page
            page_text = page.extract_text() or ""

            # Extract any tables; returns list of lists (rows x cols)
            page_tables = page.extract_tables() or []

            # Wrap each table with its index for easier reference
            table_objects = [
                {
                    "table_index": idx,
                    "table_data": table
                }
                for idx, table in enumerate(page_tables)
            ]

            pages_list.append({
                "page_number": page.page_number,
                "raw_text": page_text,
                "tables": table_objects,
                "has_tables": len(page_tables) > 0
            })

        page_obj = {
            "source": {
                "path": pdf_path,
                "total_pages": len(pdf.pages),
            },
            "pages": pages_list
        }

        # Return as JSON string (consumed by vector_ingest and llm_parser)
        return json.dumps(page_obj, indent=2)

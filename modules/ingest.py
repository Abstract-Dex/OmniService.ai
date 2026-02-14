"""
Main ingestion pipeline — orchestrates the full flow from PDF to knowledge base.

For each PDF manual in the given directory:
  1. extract.py      — Extract raw text + tables from the PDF (pdfplumber)
  2. vector_ingest.py — Embed pages via Cohere and store in Weaviate (vector DB)
  3. llm_parser.py   — Send raw text to Claude Haiku, get structured UnifiedModel
  4. graph_store.py   — Ingest the structured model into Neo4j (graph DB)

Usage:
    python -m modules.ingest -i Data/
"""

from modules.extract import extract_pdf_contents
from modules.vector_ingest import ingest_to_weaviate
from modules.graph_store import ingest_sbom, is_model_in_graph
from modules.llm_parser import parse_manual
import os
from dotenv import load_dotenv

load_dotenv()


def ingest_data(data_path: str):
    """
    Run the full ingestion pipeline on every PDF in data_path.

    Args:
        data_path: Directory containing PDF manuals to ingest.
    """
    # Discover all PDF files in the directory
    pdf_files = [f for f in os.listdir(data_path) if f.endswith(".pdf")]

    print(f"\n{'=' * 60}")
    print(f"Knowledge Base Ingestion Pipeline")
    print(f"  Found {len(pdf_files)} PDF(s) in {data_path}")
    print(f"{'=' * 60}\n")

    vector_ok, graph_ok = 0, 0

    for i, file in enumerate(sorted(pdf_files), 1):
        model_name = file.replace(".pdf", "")
        print(f"[{i}/{len(pdf_files)}] {model_name}")

        # Step 1: Extract raw pages from the PDF
        page_obj = extract_pdf_contents(os.path.join(data_path, file))

        # Step 2: Embed and ingest pages into Weaviate (vector store)
        if ingest_to_weaviate(page_obj):
            vector_ok += 1

        # Step 3: Check if the model already exists in the graph DB
        # Derive expected model_id from filename (e.g. "Model-CPB050JC-S-0-EV" -> "CPB050JC-S-0-EV")
        expected_model_id = model_name.replace("Model-", "", 1)
        if is_model_in_graph(expected_model_id):
            print(f"  [=] {expected_model_id} already in Neo4j — skipping LLM parse")
            graph_ok += 1
            print()
            continue

        # Step 4: Parse raw text into a structured UnifiedModel via Claude Haiku
        parsed_manual = parse_manual(page_obj)
        if parsed_manual is None:
            print(f"  [x] LLM parsing failed — skipping graph ingestion")
            print()
            continue

        # Step 5: Ingest the structured model into Neo4j (graph store)
        if ingest_sbom(parsed_manual):
            graph_ok += 1

        print()

    # Summary
    print(f"{'=' * 60}")
    print(
        f"  Done: {vector_ok}/{len(pdf_files)} vector | {graph_ok}/{len(pdf_files)} graph")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest PDF manuals into the knowledge base (Weaviate + Neo4j)")
    parser.add_argument("--data_path", "-i", type=str, required=True,
                        help="Path to directory containing PDF manuals")
    args = parser.parse_args()

    ingest_data(args.data_path)

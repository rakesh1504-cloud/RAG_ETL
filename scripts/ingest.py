#!/usr/bin/env python3
"""
CLI script to ingest a file or directory into the vector store.

Usage:
    python scripts/ingest.py --path data/raw/my_doc.pdf
    python scripts/ingest.py --path data/raw/ --dir
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into the RAG vector store.")
    parser.add_argument("--path", required=True, help="File or directory path to ingest")
    parser.add_argument("--dir", action="store_true", help="Ingest an entire directory")
    args = parser.parse_args()

    pipeline = RAGPipeline()

    if args.dir:
        count = pipeline.ingest_directory(args.path)
    else:
        count = pipeline.ingest_file(args.path)

    print(f"\nDone. Total chunks in store: {pipeline.stats()['total_chunks']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CLI script to query the RAG pipeline interactively.

Usage:
    python scripts/query.py
    python scripts/query.py --question "What is the refund policy?"
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.llm.rag_pipeline import RAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Query the RAG pipeline.")
    parser.add_argument("--question", help="Question to ask (omit for interactive mode)")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    print(f"Vector store contains {pipeline.stats()['total_chunks']} chunks.\n")

    if args.question:
        answer = pipeline.query(args.question)
    else:
        while True:
            question = input("Ask a question (or 'exit'): ").strip()
            if question.lower() in ("exit", "quit", "q"):
                break
            if question:
                pipeline.query(question)
                print()


if __name__ == "__main__":
    main()

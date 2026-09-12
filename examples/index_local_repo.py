#!/usr/bin/env python3
"""Example: Index a local repository.

This example demonstrates how to:
1. Index local documentation files
2. Optionally index code files
3. Search across indexed content

Usage:
    python examples/index_local_repo.py /path/to/repo
"""

import argparse
import sys
from pathlib import Path

from docugraph.ingestion.local_files import LocalFileIndexer, LocalFileConfig
from docugraph.ingestion.chunker import Chunker
from docugraph.storage.vector_store import VectorStore


def main():
    """Index a local repository and demonstrate search."""
    parser = argparse.ArgumentParser(description="Index a local repository")
    parser.add_argument("path", help="Path to repository or directory")
    parser.add_argument("--include-code", action="store_true", help="Include code files")
    parser.add_argument("--query", help="Optional search query after indexing")

    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    print("DocuGraph AI - Index Local Repository")
    print("=" * 50)

    # Configure indexer
    config = LocalFileConfig()
    if args.include_code:
        config.include_patterns = [
            "*.md",
            "*.rst",
            "*.txt",
            "*.py",
            "*.js",
            "*.ts",
        ]
        print("\nIncluding code files in index")
    else:
        print("\nIndexing documentation files only")

    # Initialize components
    indexer = LocalFileIndexer(config=config)
    chunker = Chunker()
    vector_store = VectorStore()

    # 1. Index files
    print(f"\n1. Scanning: {path}")
    documents = indexer.index_to_documents(path, recursive=True)
    print(f"   Found {len(documents)} files")

    if not documents:
        print("   No matching files found")
        sys.exit(0)

    # Show files found
    print("\n   Files:")
    for doc in documents[:10]:
        name = doc.metadata.get("file_name", "unknown")
        ftype = doc.metadata.get("file_type", "unknown")
        print(f"   - {name} ({ftype})")
    if len(documents) > 10:
        print(f"   ... and {len(documents) - 10} more")

    # 2. Chunk documents
    print("\n2. Chunking documents...")
    all_chunks = []
    for doc in documents:
        chunks = chunker.chunk_document(doc)
        all_chunks.extend(chunks)
    print(f"   Created {len(all_chunks)} chunks")

    # 3. Index chunks
    print("\n3. Indexing chunks...")
    count = vector_store.add_chunks(all_chunks)
    print(f"   Indexed {count} chunks")
    print(f"   Total chunks in store: {vector_store.count()}")

    # 4. Optional search
    if args.query:
        print(f"\n4. Searching for: '{args.query}'")
        results = vector_store.search(args.query, top_k=5)

        if results:
            for i, result in enumerate(results, 1):
                chunk = result.chunk
                source = chunk.metadata.get("source_path", "unknown")
                score = result.score
                preview = chunk.content[:150].replace("\n", " ")

                print(f"\n   [{i}] Score: {score:.4f}")
                print(f"       Source: {source}")
                print(f"       {preview}...")
        else:
            print("   No results found")

    print("\nDone!")
    print(f"\nYou can now search with: docugraph search '<your query>'")


if __name__ == "__main__":
    main()

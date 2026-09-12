#!/usr/bin/env python3
"""Example: Crawl a documentation site and search it.

This example demonstrates the basic workflow:
1. Crawl a documentation page
2. Chunk and index the content
3. Search the indexed content

Usage:
    python examples/crawl_and_search.py
"""

import asyncio

from docugraph.ingestion.crawler import DocCrawler
from docugraph.ingestion.chunker import Chunker
from docugraph.storage.vector_store import VectorStore


async def main():
    """Crawl FastAPI docs and search for path operations."""
    print("DocuGraph AI - Crawl and Search Example")
    print("=" * 50)

    # Initialize components
    crawler = DocCrawler()
    chunker = Chunker()
    vector_store = VectorStore()

    # 1. Crawl a documentation page
    url = "https://fastapi.tiangolo.com/tutorial/first-steps/"
    print(f"\n1. Crawling: {url}")

    try:
        document = await crawler.crawl_single(url)
        print(f"   Crawled: {document.metadata.get('title', 'Untitled')}")
        print(f"   Content length: {len(document.content)} characters")
    except Exception as e:
        print(f"   Error crawling: {e}")
        print("   Using sample content instead...")
        from docugraph.core.models import Document

        document = Document(
            content="""# First Steps

## Create a FastAPI Application

The simplest FastAPI file could look like this:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}
```

## Path Operations

A path operation is a combination of a path and an HTTP method.

You can use the decorators:
- @app.get()
- @app.post()
- @app.put()
- @app.delete()

Each decorator creates a path operation.
""",
            content_type="markdown",
            source_url=url,
            metadata={"title": "First Steps - FastAPI"},
        )

    # 2. Chunk the document
    print("\n2. Chunking document...")
    chunks = chunker.chunk_document(document)
    print(f"   Created {len(chunks)} chunks")

    # 3. Index the chunks
    print("\n3. Indexing chunks...")
    count = vector_store.add_chunks(chunks)
    print(f"   Indexed {count} chunks")
    print(f"   Total chunks in store: {vector_store.count()}")

    # 4. Search
    queries = [
        "How do I create a path operation?",
        "What decorators are available?",
        "Hello World example",
    ]

    print("\n4. Searching...")
    for query in queries:
        print(f"\n   Query: '{query}'")
        results = vector_store.search(query, top_k=2)

        if results:
            for i, result in enumerate(results, 1):
                score = result.score
                preview = result.chunk.content[:100].replace("\n", " ")
                print(f"   [{i}] Score: {score:.4f}")
                print(f"       {preview}...")
        else:
            print("   No results found")

    # Cleanup (optional)
    print("\n5. Cleaning up...")
    vector_store.clear()
    print("   Index cleared")

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())

---
name: docs
description: Search indexed documentation using natural language queries
allowed-tools: Read
---

# Search Documentation

Search DocuGraph's indexed documentation for relevant information. Uses hybrid search combining vector similarity, keyword matching, and knowledge graph traversal.

## Usage

```
/docs how to handle async errors
/docs authentication patterns in FastAPI
/docs "exact phrase to find"
/docs database connection pooling --top 10
```

## Instructions

1. **Extract the search query** from $ARGUMENTS
   - The entire argument string is the search query
   - Extract `--top N` or `-k N` flag if present (default: 5)

2. **Perform hybrid search:**
   - Use the `hybrid_search` MCP tool from docugraph
   - Set `query` to the extracted search query
   - Set `top_k` to the requested number of results
   - Set `mode` to "hybrid" for best results
   - Set `include_graph` to `true` to include entity relationships

3. **Present results clearly:**
   - Show each result with its relevance score
   - Include the source URL or file path
   - Show a preview of the matching content
   - Include section/title information when available

4. **If no results found:**
   - Suggest the user may need to index relevant documentation first
   - Recommend using `/index` to add documentation sources
   - Offer to try a different search query

5. **For follow-up questions:**
   - If the user asks about something mentioned in results, provide more detail
   - Offer to search for related topics

## Output Format

```
## Search Results for: "your query"

### 1. [Title or Section Name] (score: 0.85)
**Source:** https://example.com/docs/page
> Preview of the matching content showing relevant information...

### 2. [Another Result] (score: 0.72)
**Source:** /path/to/local/file.md
> Another preview showing why this matched...

---
Found 5 results. Use `/docs query --top 10` for more results.
```

## Tips

- Use natural language queries for best results
- Wrap exact phrases in quotes
- The search combines semantic understanding with keyword matching
- Graph search finds related entities even if not directly mentioned

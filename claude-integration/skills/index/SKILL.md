---
name: index
description: Index documentation from URLs, git repos, or local files into DocuGraph
allowed-tools: Bash, Read, Glob
---

# Index Documentation

Index documentation into DocuGraph for later searching. Supports multiple source types.

## Usage

```
/index https://docs.example.com              # Crawl and index a URL
/index https://docs.example.com --pages 10   # Crawl up to 10 pages
/index git:https://github.com/org/repo       # Index a git repository
/index git:https://github.com/org/repo#main  # Index specific branch
/index ./docs                                # Index local directory
/index ./docs --pattern "*.md"               # Index with file pattern
```

## Instructions

1. **Detect the source type** from $ARGUMENTS:
   - URLs starting with `http://` or `https://` (without `git:` prefix) → Web crawl
   - URLs starting with `git:` → Git repository
   - Local paths (starting with `./`, `/`, or `~`) → Local files

2. **For Web URLs:**
   - Use the `crawl_url` MCP tool from docugraph
   - Extract `--pages N` or `--max-pages N` flag if present (default: 1)
   - Example: `crawl_url` with `url` and `max_pages` parameters

3. **For Git repositories:**
   - Remove the `git:` prefix
   - Extract branch/ref after `#` if present
   - Use the `index_git` MCP tool with `url` and optionally `ref`
   - Include `--include-code` flag maps to `include_code: true`

4. **For Local paths:**
   - Use Bash to run: `docugraph index local <path>`
   - Include `--pattern` flag if specified
   - Include `--recursive` by default

5. **Report results:**
   - Show number of documents/pages processed
   - Show number of chunks created
   - Show total chunks now in the index

## Examples

**Index FastAPI docs:**
```
/index https://fastapi.tiangolo.com/tutorial/
```

**Index a GitHub repository:**
```
/index git:https://github.com/tiangolo/fastapi --include-code
```

**Index local documentation:**
```
/index ./docs --pattern "*.md"
```

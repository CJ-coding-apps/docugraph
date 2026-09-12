"""Command-line interface for DocuGraph AI."""

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="docugraph")
def main() -> None:
    """DocuGraph AI - Intelligent documentation RAG for coding agents."""
    pass


@main.command()
@click.argument("url")
@click.option(
    "--max-pages",
    "-n",
    default=1,
    help="Maximum number of pages to crawl (default: 1 for single page)",
)
@click.option(
    "--pattern",
    "-p",
    default=None,
    help="URL pattern to filter links (regex)",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable caching",
)
def crawl(url: str, max_pages: int, pattern: str | None, no_cache: bool) -> None:
    """Crawl a URL and index its content.

    Examples:
        docugraph crawl https://fastapi.tiangolo.com/tutorial/
        docugraph crawl https://docs.python.org --max-pages 50
    """
    from docugraph.ingestion.chunker import Chunker
    from docugraph.ingestion.crawler import DocCrawler
    from docugraph.storage.vector_store import VectorStore

    async def _crawl() -> None:
        crawler = DocCrawler()
        chunker = Chunker()
        vector_store = VectorStore()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            # Crawl
            if max_pages == 1:
                task = progress.add_task(f"Crawling {url}...", total=None)
                documents = [await crawler.crawl_single(url, use_cache=not no_cache)]
            else:
                task = progress.add_task(
                    f"Crawling up to {max_pages} pages from {url}...", total=None
                )
                documents = await crawler.crawl_site(
                    url,
                    max_pages=max_pages,
                    url_pattern=pattern,
                    use_cache=not no_cache,
                )

            progress.update(task, description=f"Crawled {len(documents)} pages")

            # Chunk
            progress.update(task, description="Chunking documents...")
            all_chunks = []
            for doc in documents:
                chunks = chunker.chunk_document(doc)
                all_chunks.extend(chunks)

            progress.update(
                task, description=f"Created {len(all_chunks)} chunks from {len(documents)} pages"
            )

            # Index
            progress.update(task, description="Indexing chunks (generating embeddings)...")
            count = vector_store.add_chunks(all_chunks)

            progress.update(task, description="Done!")

        console.print(f"\n[green]Successfully indexed {count} chunks from {len(documents)} pages[/green]")
        console.print(f"[dim]Total chunks in store: {vector_store.count()}[/dim]")

    asyncio.run(_crawl())


@main.command()
@click.argument("query")
@click.option(
    "--top-k",
    "-k",
    default=5,
    help="Number of results to return (default: 5)",
)
@click.option(
    "--show-content",
    "-c",
    is_flag=True,
    help="Show full chunk content",
)
def search(query: str, top_k: int, show_content: bool) -> None:
    """Search indexed documentation.

    Examples:
        docugraph search "how to create a FastAPI endpoint"
        docugraph search "async error handling" --top-k 10
    """
    from docugraph.storage.vector_store import VectorStore

    vector_store = VectorStore()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=None)
        results = vector_store.search(query, top_k=top_k)
        progress.update(task, description="Done!")

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    console.print(f"\n[bold]Found {len(results)} results for:[/bold] {query}\n")

    for i, result in enumerate(results, 1):
        chunk = result.chunk
        score = result.score

        # Get metadata
        source = chunk.metadata.get("source_url") or chunk.metadata.get("source_path", "Unknown")
        title = chunk.metadata.get("title", "Untitled")
        section = chunk.metadata.get("section_title", "")

        # Create display
        console.print(f"[bold cyan]{i}. {title}[/bold cyan]")
        if section:
            console.print(f"   [dim]Section: {section}[/dim]")
        console.print(f"   [dim]Source: {source}[/dim]")
        console.print(f"   [dim]Score: {score:.4f}[/dim]")

        if show_content:
            # Truncate content for display
            content = chunk.content
            if len(content) > 500:
                content = content[:500] + "..."
            console.print(f"\n   [italic]{content}[/italic]")

        console.print()


@main.command()
def stats() -> None:
    """Show statistics about indexed content."""
    from docugraph.core.config import get_config
    from docugraph.storage.vector_store import VectorStore

    config = get_config()
    vector_store = VectorStore()

    table = Table(title="DocuGraph Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Chunks", str(vector_store.count()))
    table.add_row("Data Directory", str(config.storage.data_dir))
    table.add_row("Embedding Model", config.embeddings.model)
    table.add_row("Embedding Provider", config.embeddings.provider.value)

    console.print(table)


@main.command()
@click.confirmation_option(prompt="Are you sure you want to clear all indexed data?")
def clear() -> None:
    """Clear all indexed content."""
    from docugraph.storage.vector_store import VectorStore

    vector_store = VectorStore()
    vector_store.clear()

    console.print("[green]All indexed content has been cleared.[/green]")


@main.group()
def config() -> None:
    """Manage configuration."""
    pass


@config.command("show")
def config_show() -> None:
    """Show current configuration."""
    from docugraph.core.config import get_config

    cfg = get_config()

    console.print("[bold]Current Configuration[/bold]\n")
    console.print(cfg.model_dump_json(indent=2))


@config.command("init")
@click.option(
    "--path",
    "-p",
    default=None,
    help="Path for config file (default: ~/.docugraph/config.yaml)",
)
def config_init(path: str | None) -> None:
    """Initialize a configuration file."""
    from docugraph.core.config import Config

    config = Config()
    save_path = Path(path) if path else None
    config.save(save_path)

    actual_path = save_path or Path.home() / ".docugraph" / "config.yaml"
    console.print(f"[green]Configuration saved to {actual_path}[/green]")


@main.command()
@click.option("--port", "-p", default=8000, help="Port to run the server on")
@click.option("--host", "-h", default="0.0.0.0", help="Host to bind to")
def serve(port: int, host: str) -> None:
    """Start the REST API server."""
    import uvicorn

    from docugraph.interfaces.api.main import app

    console.print(f"[green]Starting DocuGraph API server on {host}:{port}[/green]")
    uvicorn.run(app, host=host, port=port)


@main.command("mcp-server")
def mcp_server() -> None:
    """Start the MCP server for Claude Code integration."""
    from docugraph.interfaces.mcp_server import run_server

    console.print("[green]Starting DocuGraph MCP server...[/green]")
    run_server()


# =============================================================================
# Memory Commands
# =============================================================================


@main.group()
def memory() -> None:
    """Manage agent memory (key-value store with repo/branch awareness)."""
    pass


@memory.command("set")
@click.argument("key")
@click.argument("value")
@click.option("--repo", "-r", default="default", help="Repository scope")
@click.option("--branch", "-b", default="main", help="Branch scope")
def memory_set(key: str, value: str, repo: str, branch: str) -> None:
    """Store a value in memory.

    Examples:
        docugraph memory set auth_method "JWT tokens"
        docugraph memory set db_choice "PostgreSQL" --repo myapp --branch feature/db
    """
    import json

    from docugraph.storage.memory_store import MemoryStore

    store = MemoryStore()

    # Try to parse as JSON, otherwise store as string
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value

    entry = store.set(key, parsed_value, repository=repo, branch=branch)
    console.print(f"[green]Stored:[/green] {key} = {value}")
    console.print(f"[dim]Scope: {repo}/{branch}[/dim]")


@memory.command("get")
@click.argument("key")
@click.option("--repo", "-r", default="default", help="Repository scope")
@click.option("--branch", "-b", default="main", help="Branch scope")
def memory_get(key: str, repo: str, branch: str) -> None:
    """Retrieve a value from memory.

    Examples:
        docugraph memory get auth_method
        docugraph memory get db_choice --repo myapp --branch feature/db
    """
    import json

    from docugraph.storage.memory_store import MemoryStore

    store = MemoryStore()
    value = store.get(key, repository=repo, branch=branch)

    if value is None:
        console.print(f"[yellow]No value found for key: {key}[/yellow]")
        console.print(f"[dim]Scope: {repo}/{branch}[/dim]")
        return

    # Format output
    if isinstance(value, (dict, list)):
        console.print(f"[bold]{key}[/bold] =")
        console.print(json.dumps(value, indent=2))
    else:
        console.print(f"[bold]{key}[/bold] = {value}")

    console.print(f"[dim]Scope: {repo}/{branch}[/dim]")


@memory.command("list")
@click.option("--repo", "-r", default="default", help="Repository scope")
@click.option("--branch", "-b", default="main", help="Branch scope")
@click.option("--pattern", "-p", default="%", help="Key pattern (SQL LIKE, use % for wildcard)")
def memory_list(repo: str, branch: str, pattern: str) -> None:
    """List stored memory keys.

    Examples:
        docugraph memory list
        docugraph memory list --repo myapp
        docugraph memory list --pattern "auth%"
    """
    from docugraph.storage.memory_store import MemoryStore

    store = MemoryStore()

    if pattern == "%":
        keys = store.list_keys(repository=repo, branch=branch)
    else:
        entries = store.search(pattern, repository=repo, branch=branch)
        keys = [e.key for e in entries]

    if not keys:
        console.print("[yellow]No memories found.[/yellow]")
        console.print(f"[dim]Scope: {repo}/{branch}[/dim]")
        return

    table = Table(title=f"Memories ({repo}/{branch})")
    table.add_column("Key", style="cyan")

    for key in keys:
        table.add_row(key)

    console.print(table)
    console.print(f"[dim]Total: {len(keys)} entries[/dim]")


@memory.command("delete")
@click.argument("key")
@click.option("--repo", "-r", default="default", help="Repository scope")
@click.option("--branch", "-b", default="main", help="Branch scope")
def memory_delete(key: str, repo: str, branch: str) -> None:
    """Delete a memory entry.

    Examples:
        docugraph memory delete auth_method
        docugraph memory delete db_choice --repo myapp
    """
    from docugraph.storage.memory_store import MemoryStore

    store = MemoryStore()
    deleted = store.delete(key, repository=repo, branch=branch)

    if deleted:
        console.print(f"[green]Deleted: {key}[/green]")
    else:
        console.print(f"[yellow]Key not found: {key}[/yellow]")

    console.print(f"[dim]Scope: {repo}/{branch}[/dim]")


@memory.command("clear")
@click.option("--repo", "-r", default=None, help="Repository scope (omit for all)")
@click.option("--branch", "-b", default=None, help="Branch scope")
@click.confirmation_option(prompt="Are you sure you want to clear memories?")
def memory_clear(repo: str | None, branch: str | None) -> None:
    """Clear memory entries.

    Examples:
        docugraph memory clear                    # Clear all
        docugraph memory clear --repo myapp       # Clear repo
        docugraph memory clear --repo myapp --branch main  # Clear branch
    """
    from docugraph.storage.memory_store import MemoryStore

    store = MemoryStore()
    count = store.clear(repository=repo, branch=branch)

    scope = "all repositories"
    if repo and branch:
        scope = f"{repo}/{branch}"
    elif repo:
        scope = f"repository {repo}"

    console.print(f"[green]Cleared {count} entries from {scope}[/green]")


@memory.command("show")
@click.option("--repo", "-r", default="default", help="Repository scope")
@click.option("--branch", "-b", default="main", help="Branch scope")
def memory_show(repo: str, branch: str) -> None:
    """Show all memories as key-value pairs.

    Examples:
        docugraph memory show
        docugraph memory show --repo myapp --branch feature/auth
    """
    import json

    from docugraph.storage.memory_store import MemoryStore

    store = MemoryStore()
    memories = store.get_all(repository=repo, branch=branch)

    if not memories:
        console.print("[yellow]No memories found.[/yellow]")
        console.print(f"[dim]Scope: {repo}/{branch}[/dim]")
        return

    console.print(f"[bold]Memories ({repo}/{branch})[/bold]\n")
    console.print(json.dumps(memories, indent=2))


@memory.command("repos")
def memory_repos() -> None:
    """List all repositories with stored memories."""
    from docugraph.storage.memory_store import MemoryStore

    store = MemoryStore()
    repos = store.list_repositories()

    if not repos:
        console.print("[yellow]No repositories found.[/yellow]")
        return

    table = Table(title="Repositories with Memories")
    table.add_column("Repository", style="cyan")
    table.add_column("Branches", style="green")

    for repo in repos:
        branches = store.list_branches(repo)
        table.add_row(repo, ", ".join(branches))

    console.print(table)


# =============================================================================
# Graph Commands
# =============================================================================


@main.group()
def graph() -> None:
    """Manage knowledge graph (entities and relationships)."""
    pass


@graph.command("add")
@click.argument("content")
@click.option("--name", "-n", default=None, help="Episode name")
@click.option("--group", "-g", default="default", help="Group ID for scoping searches")
@click.option("--type", "-t", "source_type", default="text", help="Content type (text/json)")
def graph_add(content: str, name: str | None, group: str, source_type: str) -> None:
    """Add content to extract entities and relationships.

    Graphiti automatically extracts entities and relationships from text.
    Requires OPENAI_API_KEY to be set for LLM-based entity extraction.

    Examples:
        docugraph graph add "FastAPI is a Python web framework. It uses Starlette for routing."
        docugraph graph add '{"name": "Alice", "role": "Developer"}' --type json
    """
    from docugraph.storage.graph_store import GraphStore

    async def _add() -> None:
        store = GraphStore()
        try:
            episode_name = name or f"episode_{hash(content) % 10000}"
            result = await store.add_episode(
                name=episode_name,
                content=content,
                source_type=source_type,
                group_id=group,
            )

            console.print(f"[green]Added episode: {result['episode_uuid']}[/green]")
            console.print(f"[dim]Extracted {result['entities_count']} entities[/dim]")
            console.print(f"[dim]Extracted {result['relationships_count']} relationships[/dim]")

            if result["entities"]:
                console.print("\n[bold]Entities:[/bold]")
                for e in result["entities"][:5]:
                    console.print(f"  - {e['name']}")
                if len(result["entities"]) > 5:
                    console.print(f"  ... and {len(result['entities']) - 5} more")

        except RuntimeError as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("\n[bold]To use the knowledge graph, you need an LLM:[/bold]")
            console.print("  [cyan]Local (recommended):[/cyan] Install Ollama: https://ollama.ai")
            console.print("    ollama pull llama3.2 && ollama pull nomic-embed-text && ollama serve")
            console.print("  [cyan]Cloud:[/cyan] Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
            console.print("\n[dim]For local-only vector search (no LLM), use 'docugraph search'[/dim]")
        finally:
            await store.close()

    asyncio.run(_add())


@graph.command("search")
@click.argument("query")
@click.option("--limit", "-l", default=10, help="Maximum results")
@click.option("--group", "-g", default=None, help="Group ID to filter by")
def graph_search(query: str, limit: int, group: str | None) -> None:
    """Search the knowledge graph.

    Requires OPENAI_API_KEY for embedding-based search.

    Examples:
        docugraph graph search "Python web frameworks"
        docugraph graph search "authentication" --group security
    """
    from docugraph.storage.graph_store import GraphStore

    async def _search() -> None:
        store = GraphStore()
        try:
            group_ids = [group] if group else None
            results = await store.search(query, group_ids=group_ids, num_results=limit)

            if not results:
                console.print("[yellow]No results found.[/yellow]")
                return

            console.print(f"\n[bold]Found {len(results)} facts for:[/bold] {query}\n")

            for i, r in enumerate(results, 1):
                console.print(f"[cyan]{i}.[/cyan] {r['fact']}")
                if r.get("valid_at"):
                    console.print(f"   [dim]Valid: {r['valid_at']}[/dim]")
                console.print()

        except RuntimeError as e:
            console.print(f"[red]Error: {e}[/red]")
            console.print("\n[bold]To use the knowledge graph, you need an LLM:[/bold]")
            console.print("  [cyan]Local (recommended):[/cyan] Install Ollama: https://ollama.ai")
            console.print("    ollama pull llama3.2 && ollama pull nomic-embed-text && ollama serve")
            console.print("  [cyan]Cloud:[/cyan] Set OPENAI_API_KEY or ANTHROPIC_API_KEY")
            console.print("\n[dim]For local-only vector search (no LLM), use 'docugraph search'[/dim]")
        finally:
            await store.close()

    asyncio.run(_search())


@graph.command("clear")
@click.confirmation_option(prompt="Are you sure you want to clear the knowledge graph?")
def graph_clear() -> None:
    """Clear all data from the knowledge graph."""
    from docugraph.storage.graph_store import GraphStore

    async def _clear() -> None:
        store = GraphStore()
        await store.clear()
        console.print("[green]Knowledge graph has been cleared.[/green]")

    asyncio.run(_clear())


# =============================================================================
# Index Commands (Local Files and Git Repositories)
# =============================================================================


@main.group()
def index() -> None:
    """Index local files and git repositories."""
    pass


@index.command("local")
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option(
    "--recursive/--no-recursive",
    "-r/-R",
    default=True,
    help="Recursively index subdirectories (default: yes)",
)
@click.option(
    "--pattern",
    "-p",
    multiple=True,
    help="Include patterns (e.g., '*.md', '*.rst'). Can be specified multiple times.",
)
@click.option(
    "--exclude",
    "-e",
    multiple=True,
    help="Exclude patterns. Can be specified multiple times.",
)
@click.option(
    "--include-hidden",
    is_flag=True,
    help="Include hidden files (starting with .)",
)
@click.option(
    "--stats-only",
    is_flag=True,
    help="Only show statistics, don't index",
)
def index_local(
    path: str,
    recursive: bool,
    pattern: tuple[str, ...],
    exclude: tuple[str, ...],
    include_hidden: bool,
    stats_only: bool,
) -> None:
    """Index local documentation files (Markdown, RST, text).

    Examples:
        docugraph index local ./docs
        docugraph index local ./my-project --pattern "*.md" --pattern "README*"
        docugraph index local ./docs --exclude "*.draft.md"
        docugraph index local ./project --stats-only
    """
    from docugraph.ingestion.chunker import Chunker
    from docugraph.ingestion.local_files import LocalFileConfig, LocalFileIndexer
    from docugraph.storage.vector_store import VectorStore

    # Build config
    config = LocalFileConfig(include_hidden=include_hidden)

    if pattern:
        config.include_patterns = list(pattern)

    if exclude:
        config.exclude_patterns = list(config.exclude_patterns) + list(exclude)

    indexer = LocalFileIndexer(config=config)

    # Stats only mode
    if stats_only:
        stats = indexer.get_stats(path, recursive=recursive)

        table = Table(title=f"Index Statistics: {path}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Indexable Files", str(stats["total_files"]))
        table.add_row("Total Size", f"{stats['total_size'] / 1024:.1f} KB")
        table.add_row("Skipped Files", str(stats["skipped"]))

        for file_type, count in stats.get("by_type", {}).items():
            table.add_row(f"  {file_type}", str(count))

        console.print(table)
        return

    # Index and store
    chunker = Chunker()
    vector_store = VectorStore()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Indexing {path}...", total=None)

        # Index files
        documents = indexer.index_to_documents(path, recursive=recursive)
        progress.update(task, description=f"Found {len(documents)} files")

        # Chunk documents
        progress.update(task, description="Chunking documents...")
        all_chunks = []
        for doc in documents:
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        progress.update(task, description=f"Created {len(all_chunks)} chunks")

        # Index chunks
        progress.update(task, description="Indexing chunks (generating embeddings)...")
        count = vector_store.add_chunks(all_chunks)

        progress.update(task, description="Done!")

    console.print(
        f"\n[green]Successfully indexed {count} chunks from {len(documents)} files[/green]"
    )
    console.print(f"[dim]Total chunks in store: {vector_store.count()}[/dim]")


@index.command("git")
@click.argument("url")
@click.option(
    "--name",
    "-n",
    default=None,
    help="Custom name for the repository",
)
@click.option(
    "--ref",
    "-r",
    default=None,
    help="Branch, tag, or commit to checkout",
)
@click.option(
    "--shallow",
    is_flag=True,
    help="Shallow clone (depth=1) for faster cloning",
)
@click.option(
    "--include-code",
    is_flag=True,
    help="Include code files (not just documentation)",
)
@click.option(
    "--cleanup",
    is_flag=True,
    help="Remove cloned repository after indexing",
)
@click.option(
    "--stats-only",
    is_flag=True,
    help="Only show statistics, don't index",
)
def index_git(
    url: str,
    name: str | None,
    ref: str | None,
    shallow: bool,
    include_code: bool,
    cleanup: bool,
    stats_only: bool,
) -> None:
    """Clone and index a git repository.

    Examples:
        docugraph index git https://github.com/fastapi/fastapi
        docugraph index git https://github.com/user/repo --ref main --shallow
        docugraph index git git@github.com:user/repo.git --include-code
        docugraph index git https://github.com/user/repo --stats-only
    """
    from docugraph.ingestion.chunker import Chunker
    from docugraph.ingestion.git_indexer import GitIndexer, GitIndexerConfig
    from docugraph.storage.vector_store import VectorStore

    config = GitIndexerConfig(
        clone_depth=1 if shallow else None,
        ref=ref,
        include_code=include_code,
        cleanup_after_index=cleanup,
    )

    indexer = GitIndexer(config=config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Cloning {url}...", total=None)

        try:
            # Clone repository
            repo_info = indexer.clone(url, name=name, ref=ref)
            progress.update(
                task, description=f"Cloned to {repo_info.local_path.name}"
            )

            # Stats only mode
            if stats_only:
                progress.update(task, description="Analyzing repository...")
                stats = indexer.get_repo_stats(repo_info.local_path)
                progress.update(task, description="Done!")

                table = Table(title=f"Repository: {repo_info.name}")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("URL", repo_info.url)
                table.add_row("Branch", repo_info.branch or "N/A")
                table.add_row("Commit", (repo_info.commit_hash or "N/A")[:12])
                table.add_row("Indexable Files", str(stats["total_files"]))
                table.add_row("Total Size", f"{stats['total_size'] / 1024:.1f} KB")

                for file_type, count in stats.get("by_type", {}).items():
                    table.add_row(f"  {file_type}", str(count))

                console.print(table)

                if cleanup:
                    indexer.cleanup(repo_info.local_path)
                    console.print("[dim]Repository cleaned up.[/dim]")
                return

            # Index repository
            progress.update(task, description="Indexing files...")
            documents = []
            for indexed_file in indexer.index_repo(repo_info.local_path, repo_info):
                documents.append(indexed_file.to_document())

            progress.update(task, description=f"Found {len(documents)} files")

            # Chunk documents
            progress.update(task, description="Chunking documents...")
            chunker = Chunker()
            all_chunks = []
            for doc in documents:
                chunks = chunker.chunk_document(doc)
                all_chunks.extend(chunks)

            progress.update(task, description=f"Created {len(all_chunks)} chunks")

            # Index chunks
            progress.update(task, description="Indexing chunks (generating embeddings)...")
            vector_store = VectorStore()
            count = vector_store.add_chunks(all_chunks)

            progress.update(task, description="Done!")

            # Cleanup if requested
            if cleanup:
                indexer.cleanup(repo_info.local_path)

        except RuntimeError as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    console.print(
        f"\n[green]Successfully indexed {count} chunks from {len(documents)} files[/green]"
    )
    console.print(f"[dim]Repository: {repo_info.name} @ {repo_info.branch or 'HEAD'}[/dim]")
    console.print(f"[dim]Commit: {(repo_info.commit_hash or 'N/A')[:12]}[/dim]")
    console.print(f"[dim]Total chunks in store: {vector_store.count()}[/dim]")


@index.command("list")
def index_list() -> None:
    """List all cloned repositories."""
    from docugraph.ingestion.git_indexer import GitIndexer

    indexer = GitIndexer()
    repos = indexer.list_repos()

    if not repos:
        console.print("[yellow]No cloned repositories found.[/yellow]")
        return

    table = Table(title="Cloned Repositories")
    table.add_column("Name", style="cyan")
    table.add_column("Branch", style="green")
    table.add_column("Commit", style="dim")
    table.add_column("URL")

    for repo in repos:
        table.add_row(
            repo.name,
            repo.branch or "N/A",
            (repo.commit_hash or "N/A")[:12],
            repo.url[:50] + "..." if len(repo.url) > 50 else repo.url,
        )

    console.print(table)


@index.command("update")
@click.argument("name")
def index_update(name: str) -> None:
    """Update a cloned repository (git pull).

    Examples:
        docugraph index update fastapi
    """
    from docugraph.ingestion.git_indexer import GitIndexer

    indexer = GitIndexer()
    repos = indexer.list_repos()

    # Find repo by name
    repo_path = None
    for repo in repos:
        if repo.name == name:
            repo_path = repo.local_path
            break

    if repo_path is None:
        console.print(f"[red]Repository not found: {name}[/red]")
        console.print("[dim]Use 'docugraph index list' to see cloned repositories.[/dim]")
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Updating {name}...", total=None)

        try:
            repo_info = indexer.update(repo_path)
            progress.update(task, description="Done!")

            console.print(f"[green]Updated repository: {name}[/green]")
            console.print(f"[dim]Branch: {repo_info.branch}[/dim]")
            console.print(f"[dim]Commit: {(repo_info.commit_hash or 'N/A')[:12]}[/dim]")

        except RuntimeError as e:
            console.print(f"[red]Error: {e}[/red]")


@index.command("remove")
@click.argument("name")
@click.confirmation_option(prompt="Are you sure you want to remove this repository?")
def index_remove(name: str) -> None:
    """Remove a cloned repository.

    Examples:
        docugraph index remove fastapi
    """
    from docugraph.ingestion.git_indexer import GitIndexer

    indexer = GitIndexer()
    repos = indexer.list_repos()

    # Find repo by name
    repo_path = None
    for repo in repos:
        if repo.name == name:
            repo_path = repo.local_path
            break

    if repo_path is None:
        console.print(f"[red]Repository not found: {name}[/red]")
        return

    if indexer.cleanup(repo_path):
        console.print(f"[green]Removed repository: {name}[/green]")
    else:
        console.print(f"[red]Failed to remove repository: {name}[/red]")


if __name__ == "__main__":
    main()

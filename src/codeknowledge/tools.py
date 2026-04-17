"""MCP tool implementations for CodeKnowledge."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

from pydantic import Field

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy globals — initialized by lifespan
# ---------------------------------------------------------------------------

_repo_root: Path | None = None
_config = None
_call_graph = None


def init(repo_root: Path) -> None:
    """Initialize tool globals from a repo root."""
    global _repo_root, _config, _call_graph

    from .config import Config
    from .graph import CallGraph

    _repo_root = repo_root
    ck_dir = repo_root / ".codeknowledge"
    _config = Config.load(ck_dir)

    graph_dir = ck_dir / "graph"
    if graph_dir.is_dir():
        _call_graph = CallGraph.load(graph_dir)
        log.info("Call graph loaded: %d files", len(_call_graph.files))


def _ck_dir() -> Path:
    if _repo_root is None:
        raise RuntimeError("No project loaded. Call open_project first.")
    return _repo_root / ".codeknowledge"


# ---------------------------------------------------------------------------
# search_codebase
# ---------------------------------------------------------------------------

async def search_codebase(
    query: Annotated[str, Field(description="Natural language search query about the codebase")],
    limit: Annotated[int, Field(description="Maximum number of results to return")] = 5,
    source: Annotated[str, Field(description="Which index to search: 'all' (default), 'docs' (descriptions and articles only), or 'code' (raw source code only)")] = "all",
) -> dict:
    """Semantic search over codebase knowledge articles, file descriptions, and source code.

    Returns ranked results with relevance scores, document paths, heading
    context, and content snippets. Use this to find information about how
    the codebase works, what specific functions do, or architectural patterns.
    Use source='code' to search raw source code, or source='docs' for
    descriptions and architecture articles only.
    """
    from .index import search_index
    from .embeddings import reset_embedder

    db_path = _ck_dir() / "codeknowledge.db"
    if not db_path.exists():
        return {"error": "Search index not found. Run 'codeknowledge index' first."}

    emb_cfg = _config.embedding if _config else None

    # Check for code index
    code_db_path = _ck_dir() / "codeknowledge-code.db"
    code_emb_cfg = None
    if _config and _config.code_embedding and code_db_path.exists():
        code_emb_cfg = _config.code_embedding
    else:
        code_db_path = None

    results = search_index(
        query=query, db_path=db_path, top_k=limit, embedding_config=emb_cfg,
        code_db_path=code_db_path, code_embedding_config=code_emb_cfg,
        source=source,
    )

    if not results:
        return {"results": [], "message": "No results found."}

    formatted = []
    for r in results:
        entry: dict = {
            "score": round(r["score"], 4),
            "document": r["document"],
            "type": r["doc_type"],
        }
        if r["heading"]:
            entry["heading"] = r["heading"]
        if r["title"]:
            entry["title"] = r["title"]
        entry["content"] = r["content"]
        formatted.append(entry)

    return {"results": formatted}


# ---------------------------------------------------------------------------
# get_symbol_context
# ---------------------------------------------------------------------------

async def get_symbol_context(
    file_path: Annotated[str, Field(description="Relative path to the source file (e.g. 'src/codeknowledge/graph.py')")],
    symbol: Annotated[str, Field(description="Function or class name to look up (e.g. 'build_graph' or 'CallGraph.load')")] = "",
) -> dict:
    """Get description and call graph context for a file or symbol.

    Returns the LLM-generated description of the file/symbol plus
    incoming callers and outgoing calls from the static call graph.
    Use this to understand what a function does and how it connects
    to the rest of the codebase.
    """
    result: dict = {"file": file_path}

    # Load file description if available
    desc_path = _ck_dir() / "descriptions" / (file_path + ".md")
    if desc_path.exists():
        content = desc_path.read_text()
        # Strip frontmatter
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                content = content[end + 3:].strip()
        if symbol:
            # Extract just the symbol's section
            section = _extract_section(content, symbol)
            if section:
                result["description"] = section
            else:
                result["description"] = f"Symbol '{symbol}' not found in description."
        else:
            result["description"] = content
    else:
        result["description"] = "No description available. Run 'codeknowledge describe' first."

    # Add call graph context
    if _call_graph:
        if symbol:
            callers = _call_graph.get_callers(file_path, symbol)
            callees = _call_graph.get_callees(file_path, symbol)
            if callers:
                result["callers"] = [
                    {"file": cf, "function": cn} for cf, cn in callers
                ]
            if callees:
                result["calls"] = [
                    {"name": c.name, "resolved": c.resolved}
                    for c in callees
                ]
        else:
            # Return all functions with their caller/callee counts
            funcs = _call_graph.get_file_functions(file_path)
            if funcs:
                result["functions"] = [
                    {
                        "name": fn.qualified_name,
                        "calls_count": len(fn.calls),
                        "callers_count": len(fn.callers),
                    }
                    for fn in funcs
                ]

    return result


def _extract_section(markdown: str, symbol: str) -> str | None:
    """Extract a section from markdown by heading matching a symbol name."""
    lines = markdown.split("\n")
    collecting = False
    collected: list[str] = []
    target_level = 0

    for line in lines:
        if line.startswith("#"):
            # Count heading level
            level = 0
            for ch in line:
                if ch == "#":
                    level += 1
                else:
                    break
            heading_text = line[level:].strip().strip("`").strip()

            if collecting:
                # Stop at same or higher level heading
                if level <= target_level:
                    break
            elif symbol in heading_text:
                collecting = True
                target_level = level
                collected.append(line)
                continue

        if collecting:
            collected.append(line)

    return "\n".join(collected).strip() if collected else None


# ---------------------------------------------------------------------------
# get_architecture
# ---------------------------------------------------------------------------

async def get_architecture() -> dict:
    """Get the synthesized architecture overview of the codebase.

    Returns the high-level architecture article that describes the overall
    structure, key modules, data flow, and design patterns. This is the
    best starting point for understanding an unfamiliar codebase.
    """
    articles_dir = _ck_dir() / "articles"
    arch_file = articles_dir / "architecture-overview.md"

    if not arch_file.exists():
        return {"error": "No architecture overview. Run 'codeknowledge synthesize' first."}

    content = arch_file.read_text()
    # Strip frontmatter
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            content = content[end + 3:].strip()

    return {"content": content}


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------

async def list_files() -> dict:
    """List all source files known to the knowledge base.

    Returns files that have been extracted and described, with their
    call graph summary (number of functions, callers, callees).
    """
    desc_dir = _ck_dir() / "descriptions"
    if not desc_dir.is_dir():
        return {"error": "No descriptions found. Run 'codeknowledge describe' first."}

    files: list[dict] = []
    for md in sorted(desc_dir.rglob("*.md")):
        rel = str(md.relative_to(desc_dir))
        # Strip the .md suffix to get source path
        if rel.endswith(".md"):
            source_path = rel[:-3]
        else:
            source_path = rel

        entry: dict = {"path": source_path}

        # Add call graph stats if available
        if _call_graph:
            funcs = _call_graph.get_file_functions(source_path)
            if funcs:
                entry["functions"] = len(funcs)
                entry["total_callers"] = sum(len(fn.callers) for fn in funcs)
                entry["total_calls"] = sum(len(fn.calls) for fn in funcs)

        files.append(entry)

    return {"files": files}

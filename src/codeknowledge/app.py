"""FastMCP server definition for CodeKnowledge."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastmcp import FastMCP

from . import tools


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Initialize knowledge base on startup."""
    repo_root = _find_repo_root()
    tools.init(repo_root)
    yield


def _find_repo_root() -> Path:
    """Walk up from cwd to find .codeknowledge/ directory."""
    current = Path.cwd().resolve()
    for parent in [current, *current.parents]:
        if (parent / ".codeknowledge").is_dir():
            return parent
    raise FileNotFoundError(
        "No .codeknowledge/ directory found. "
        "Run 'codeknowledge init' in your project root first."
    )


def create_mcp_server() -> FastMCP:
    """Build the CodeKnowledge MCP server."""
    mcp = FastMCP(
        name="codeknowledge",
        instructions=(
            "CodeKnowledge provides structured understanding of a codebase "
            "through LLM-generated descriptions, synthesized architecture "
            "articles, and a static call graph. "
            "Start with get_architecture for a high-level overview, "
            "use search_codebase to find specific information, "
            "and get_symbol_context for detailed function/class context."
        ),
        lifespan=lifespan,
    )

    mcp.add_tool(tools.search_codebase)
    mcp.add_tool(tools.get_symbol_context)
    mcp.add_tool(tools.get_architecture)
    mcp.add_tool(tools.list_files)

    return mcp

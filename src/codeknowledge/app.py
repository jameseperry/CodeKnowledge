"""FastMCP server definition for CodeKnowledge."""

from __future__ import annotations

import fnmatch
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field

from . import tools

log = logging.getLogger(__name__)

# Set by create_mcp_server(), read by open_project tool.
_allow_patterns: list[str] = []


def _path_allowed(path: Path) -> bool:
    """Check whether *path* matches any of the configured allow patterns."""
    if not _allow_patterns:
        return True  # no whitelist → allow everything
    resolved = str(path.resolve())
    return any(fnmatch.fnmatch(resolved, pat) for pat in _allow_patterns)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[None]:
    """Server lifespan — project is loaded lazily via open_project."""
    yield


def create_mcp_server(allow: list[str] | None = None) -> FastMCP:
    """Build the CodeKnowledge MCP server.

    Parameters
    ----------
    allow:
        Glob patterns for allowed project paths.  If empty, all paths
        are permitted.  Example: ``["/home/james/dev/**"]``
    """
    global _allow_patterns
    _allow_patterns = list(allow or [])

    mcp = FastMCP(
        name="codeknowledge",
        instructions=(
            "CodeKnowledge provides structured understanding of a codebase "
            "through LLM-generated descriptions, synthesized architecture "
            "articles, and a static call graph.\n"
            "Call open_project first with the repo root path to load a "
            "knowledge base. Then use get_architecture for a high-level "
            "overview, search_codebase to find specific information, "
            "and get_symbol_context for detailed function/class context."
        ),
        lifespan=lifespan,
    )

    mcp.add_tool(open_project)
    mcp.add_tool(tools.search_codebase)
    mcp.add_tool(tools.get_symbol_context)
    mcp.add_tool(tools.get_architecture)
    mcp.add_tool(tools.list_files)

    return mcp


# ---------------------------------------------------------------------------
# open_project tool
# ---------------------------------------------------------------------------

async def open_project(
    repo_root: Annotated[
        str,
        Field(description="Absolute path to the project root containing a .codeknowledge/ directory"),
    ],
) -> dict:
    """Open a CodeKnowledge project for this session.

    Loads the knowledge base from the given repo root. The path must
    contain a .codeknowledge/ directory and match the server's allow list.
    Call this before using any other tools.
    """
    path = Path(repo_root).resolve()

    if not _path_allowed(path):
        return {"error": f"Path not allowed by server whitelist: {path}"}

    ck_dir = path / ".codeknowledge"
    if not ck_dir.is_dir():
        return {"error": f"No .codeknowledge/ directory found at {path}"}

    tools.init(path)
    log.info("Opened project: %s", path)
    return {"status": "ok", "project": str(path)}

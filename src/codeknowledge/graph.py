"""Call-graph data model, disk I/O, and querying.

Handles reading/writing per-file graph YAML, and provides an in-memory
CallGraph that holds both outgoing calls (from YAML) and incoming calls
(synthesized by inverting edges after loading all files).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .extractors.python.calls import extract_file_graph

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CallInfo:
    """A single call made by a function."""
    name: str
    resolved: str | None  # "local", "self", "<module>", or None


@dataclass
class FunctionNode:
    """A function/method in the call graph."""
    name: str
    file_path: str
    scope: str | None  # enclosing class name, if a method
    calls: list[CallInfo] = field(default_factory=list)
    callers: list[tuple[str, str]] = field(default_factory=list)  # (file_path, qualified_name)

    @property
    def qualified_name(self) -> str:
        return f"{self.scope}.{self.name}" if self.scope else self.name


@dataclass
class FileGraph:
    """Graph data for a single source file."""
    path: str
    source_hash: str
    imports: list[dict]
    functions: list[FunctionNode]


# ---------------------------------------------------------------------------
# In-memory call graph with inverted edges
# ---------------------------------------------------------------------------

class CallGraph:
    """In-memory call graph loaded from .codeknowledge/graph/ YAML files.

    Provides both outgoing (calls) and incoming (callers) lookups.
    """

    def __init__(self) -> None:
        self.files: dict[str, FileGraph] = {}  # rel_path → FileGraph
        # Inverted index: (file, func_name) → list of (caller_file, caller_func)
        self._callers: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)

    @classmethod
    def load(cls, graph_dir: Path) -> CallGraph:
        """Load all graph YAML files from a directory and build inverted edges."""
        cg = cls()

        for yaml_path in sorted(graph_dir.rglob("*.yaml")):
            try:
                data = yaml.safe_load(yaml_path.read_text())
            except Exception:
                log.warning("Failed to parse %s, skipping", yaml_path)
                continue

            if not data or "path" not in data:
                continue

            rel_path = data["path"]
            functions: list[FunctionNode] = []

            for fn_data in data.get("functions", []):
                calls = [
                    CallInfo(name=c["name"], resolved=c.get("resolved"))
                    for c in fn_data.get("calls", [])
                ]
                functions.append(FunctionNode(
                    name=fn_data["name"],
                    file_path=rel_path,
                    scope=fn_data.get("scope"),
                    calls=calls,
                ))

            cg.files[rel_path] = FileGraph(
                path=rel_path,
                source_hash=data.get("source_hash", ""),
                imports=data.get("imports", []),
                functions=functions,
            )

        cg._build_inverted_index()
        return cg

    def _build_inverted_index(self) -> None:
        """Walk all outgoing calls and build the callers (incoming) index.

        Only resolves "local" calls (same-file) and "self" calls for now.
        Cross-file resolution requires matching import paths to file paths,
        which we can add later.
        """
        self._callers.clear()

        for rel_path, file_graph in self.files.items():
            for fn in file_graph.functions:
                caller_key = (rel_path, fn.qualified_name)
                for call in fn.calls:
                    if call.resolved == "local" or call.resolved == "self":
                        callee_key = (rel_path, call.name)
                        self._callers[callee_key].append(caller_key)

        # Wire callers back into FunctionNodes
        for file_graph in self.files.values():
            for fn in file_graph.functions:
                key = (fn.file_path, fn.qualified_name)
                fn.callers = self._callers.get(key, [])

    def get_callers(self, file_path: str, qualified_name: str) -> list[tuple[str, str]]:
        """Get list of (file_path, qualified_name) that call the given function."""
        return self._callers.get((file_path, qualified_name), [])

    def get_callees(self, file_path: str, qualified_name: str) -> list[CallInfo]:
        """Get list of calls made by the given function."""
        fg = self.files.get(file_path)
        if not fg:
            return []
        for fn in fg.functions:
            if fn.qualified_name == qualified_name:
                return fn.calls
        return []

    def get_file_functions(self, file_path: str) -> list[FunctionNode]:
        """Get all functions defined in a file."""
        fg = self.files.get(file_path)
        return fg.functions if fg else []

    def get_file_callers(self, file_path: str) -> dict[str, list[tuple[str, str]]]:
        """Get all callers for all functions in a file.

        Returns: {qualified_name: [(caller_file, caller_qualified_name), ...]}
        Only includes functions that have at least one caller.
        """
        result: dict[str, list[tuple[str, str]]] = {}
        for fn in self.get_file_functions(file_path):
            if fn.callers:
                result[fn.qualified_name] = fn.callers
        return result


# ---------------------------------------------------------------------------
# Disk I/O: write graph YAML files
# ---------------------------------------------------------------------------

def build_graph(
    files: list[tuple[Path, str]],
    output_dir: Path,
) -> int:
    """Extract call graphs for all files and write YAML to output_dir.

    Args:
        files: List of (absolute_path, relative_path) tuples.
        output_dir: Directory to write graph YAML files to.

    Returns:
        Number of files processed.
    """
    count = 0
    for abs_path, rel_path in files:
        source = abs_path.read_bytes()
        graph = extract_file_graph(source, rel_path)

        out_path = output_dir / (rel_path + ".yaml")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(yaml.dump(graph, default_flow_style=False, sort_keys=False))
        count += 1

    return count

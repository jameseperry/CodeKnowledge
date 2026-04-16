"""Python call-graph extraction using tree-sitter.

Parses a Python source file and extracts per-function call information
with best-effort resolution of call targets to local definitions,
imports, or builtins.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Python builtins to tag and skip
# ---------------------------------------------------------------------------

PYTHON_BUILTINS = frozenset({
    "abs", "all", "any", "ascii", "bin", "bool", "breakpoint", "bytearray",
    "bytes", "callable", "chr", "classmethod", "compile", "complex",
    "delattr", "dict", "dir", "divmod", "enumerate", "eval", "exec",
    "filter", "float", "format", "frozenset", "getattr", "globals",
    "hasattr", "hash", "help", "hex", "id", "input", "int", "isinstance",
    "issubclass", "iter", "len", "list", "locals", "map", "max",
    "memoryview", "min", "next", "object", "oct", "open", "ord", "pow",
    "print", "property", "range", "repr", "reversed", "round", "set",
    "setattr", "slice", "sorted", "staticmethod", "str", "sum", "super",
    "tuple", "type", "vars", "zip",
})


# ---------------------------------------------------------------------------
# Tree-sitter helpers
# ---------------------------------------------------------------------------

def _node_text(node: Node) -> str:
    return node.text.decode("utf-8")


def _get_parser() -> Parser:
    lang = Language(tspython.language())
    return Parser(lang)


# ---------------------------------------------------------------------------
# Import resolution
# ---------------------------------------------------------------------------

def _parse_imports(root: Node) -> list[dict]:
    """Extract imports as structured dicts.

    Returns list of:
        {"module": "os.path", "names": ["join", "exists"]}
        {"module": "os.path", "names": []}  # import os.path
    """
    imports: list[dict] = []

    for child in root.children:
        if child.type == "import_statement":
            # import foo, bar.baz, baz as b
            for named in child.named_children:
                if named.type == "dotted_name":
                    imports.append({"module": _node_text(named), "names": []})
                elif named.type == "aliased_import":
                    dotted = named.child_by_field_name("name")
                    alias_node = named.child_by_field_name("alias")
                    if dotted:
                        rec = {"module": _node_text(dotted), "names": []}
                        if alias_node:
                            rec["alias"] = _node_text(alias_node)
                        imports.append(rec)

        elif child.type == "import_from_statement":
            # from foo import bar, baz
            module_node = child.child_by_field_name("module_name")
            module_name = _node_text(module_node) if module_node else ""

            names: list[str] = []
            for named in child.named_children:
                if named.type == "dotted_name" and named != module_node:
                    names.append(_node_text(named))
                elif named.type == "aliased_import":
                    name_node = named.child_by_field_name("name")
                    if name_node:
                        names.append(_node_text(name_node))
                elif named.type == "identifier" and named != module_node:
                    names.append(_node_text(named))

            # Handle wildcard: from foo import *
            if not names:
                for ch in child.children:
                    if ch.type == "wildcard_import":
                        names.append("*")
                        break

            imports.append({"module": module_name, "names": names})

    return imports


def _build_import_lookup(imports: list[dict]) -> dict[str, str]:
    """Build a name -> module mapping from parsed imports.

    E.g. from os.path import join -> {"join": "os.path"}
         import click -> {"click": "click"}
         import tree_sitter_python as tspython -> {"tspython": "tree_sitter_python"}
    """
    lookup: dict[str, str] = {}
    for imp in imports:
        module = imp["module"]
        names = imp["names"]
        if names:
            for n in names:
                if n != "*":
                    lookup[n] = module
        else:
            # Aliased: import foo as bar -> "bar" -> "foo"
            alias = imp.get("alias")
            if alias:
                lookup[alias] = module
            else:
                # import foo.bar -> "foo" resolves to "foo"
                top = module.split(".")[0]
                lookup[top] = module
    return lookup


# ---------------------------------------------------------------------------
# Call extraction
# ---------------------------------------------------------------------------

def _resolve_call_name(func_node: Node) -> str | None:
    """Extract a clean call name from a call's function node.

    Handles:
      - Simple: foo -> "foo"
      - Attribute: os.path.join -> "os.path.join"
      - Chained: foo().bar -> None (skip)
      - self.method -> "self.method"
      - String literal method: "".join -> None (skip)
    """
    if func_node.type == "identifier":
        return _node_text(func_node)

    if func_node.type == "attribute":
        attr_node = func_node.child_by_field_name("attribute")
        obj_node = func_node.child_by_field_name("object")
        if not attr_node or not obj_node:
            return None

        attr_name = _node_text(attr_node)

        parts = _collect_dotted_parts(obj_node)
        if parts is None:
            return None
        parts.append(attr_name)
        return ".".join(parts)

    return None


def _collect_dotted_parts(node: Node) -> list[str] | None:
    """Collect parts of a dotted name (a.b.c), returning None if
    the expression contains calls, subscripts, string literals, etc."""
    if node.type == "identifier":
        return [_node_text(node)]
    if node.type == "attribute":
        obj = node.child_by_field_name("object")
        attr = node.child_by_field_name("attribute")
        if not obj or not attr:
            return None
        parts = _collect_dotted_parts(obj)
        if parts is None:
            return None
        parts.append(_node_text(attr))
        return parts
    return None


def _extract_calls(body_node: Node) -> list[str]:
    """Extract all call expression names from a function body."""
    calls: list[str] = []

    def _walk(node: Node) -> None:
        if node.type == "call":
            func_node = node.child_by_field_name("function")
            if func_node:
                name = _resolve_call_name(func_node)
                if name:
                    calls.append(name)
            args_node = node.child_by_field_name("arguments")
            if args_node:
                for child in args_node.children:
                    _walk(child)
            return
        if node.type in ("function_definition", "class_definition"):
            return
        for child in node.children:
            _walk(child)

    _walk(body_node)
    return calls


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Function/class definition extraction
# ---------------------------------------------------------------------------

def _collect_definitions(root: Node) -> dict[str, str]:
    """Collect all top-level function and class names -> kind mapping."""
    defs: dict[str, str] = {}

    def _scan(node: Node, class_name: str | None = None) -> None:
        for child in node.children:
            actual = child
            if child.type == "decorated_definition":
                inner = child.named_children[-1] if child.named_children else None
                if inner:
                    actual = inner
                else:
                    continue

            if actual.type == "function_definition":
                name_node = actual.child_by_field_name("name")
                if name_node:
                    name = _node_text(name_node)
                    if class_name:
                        defs[f"{class_name}.{name}"] = "method"
                        defs[name] = "method"
                    else:
                        defs[name] = "function"

            elif actual.type == "class_definition":
                name_node = actual.child_by_field_name("name")
                if name_node:
                    name = _node_text(name_node)
                    defs[name] = "class"
                    body = actual.child_by_field_name("body")
                    if body:
                        _scan(body, class_name=name)

    _scan(root)
    return defs


# ---------------------------------------------------------------------------
# Call resolution
# ---------------------------------------------------------------------------

def _resolve_call(
    raw_call: str,
    local_defs: dict[str, str],
    import_lookup: dict[str, str],
    class_name: str | None = None,
) -> dict:
    """Resolve a raw call string to a structured call record.

    Returns:
        {"name": str, "resolved": str | None}

    resolved values:
        "local"         - defined in the same file
        "builtin"       - Python builtin
        "<module>"      - resolved via import
        "self"          - method on the enclosing class
        null            - couldn't resolve
    """
    if raw_call.startswith("self."):
        method = raw_call.split(".", 1)[1]
        if class_name:
            return {"name": f"{class_name}.{method}", "resolved": "self"}
        return {"name": raw_call, "resolved": "self"}

    if raw_call.startswith("super()."):
        return {"name": raw_call, "resolved": "super"}

    if "." in raw_call:
        parts = raw_call.split(".")
        top = parts[0]
        if top in import_lookup:
            return {"name": raw_call, "resolved": import_lookup[top]}
        return {"name": raw_call, "resolved": None}

    if raw_call in PYTHON_BUILTINS:
        return {"name": raw_call, "resolved": "builtin"}

    if raw_call in local_defs:
        return {"name": raw_call, "resolved": "local"}

    if raw_call in import_lookup:
        return {"name": raw_call, "resolved": import_lookup[raw_call]}

    return {"name": raw_call, "resolved": None}


# ---------------------------------------------------------------------------
# Per-function extraction
# ---------------------------------------------------------------------------

def _extract_function_graph(
    node: Node,
    local_defs: dict[str, str],
    import_lookup: dict[str, str],
    class_name: str | None = None,
) -> dict | None:
    """Extract call graph info for a single function/method node."""
    name_node = node.child_by_field_name("name")
    if not name_node:
        return None

    name = _node_text(name_node)
    body_node = node.child_by_field_name("body")
    if not body_node:
        return {"name": name, "calls": []}

    raw_calls = _extract_calls(body_node)
    raw_calls = _dedupe_preserve_order(raw_calls)

    calls: list[dict] = []
    for raw in raw_calls:
        resolved = _resolve_call(raw, local_defs, import_lookup, class_name)
        if resolved["resolved"] == "builtin":
            continue
        calls.append(resolved)

    result: dict = {"name": name}
    if class_name:
        result["scope"] = class_name
    result["calls"] = calls
    return result


# ---------------------------------------------------------------------------
# File-level extraction (public API)
# ---------------------------------------------------------------------------

def extract_file_graph(source: bytes, rel_path: str) -> dict:
    """Extract the call graph for a single Python file.

    Returns a dict suitable for YAML serialization:
        path: str
        source_hash: str
        imports: [{module, names}, ...]
        functions: [{name, scope?, calls: [{name, resolved}]}, ...]
    """
    parser = _get_parser()
    tree = parser.parse(source)
    root = tree.root_node

    imports = _parse_imports(root)
    import_lookup = _build_import_lookup(imports)
    local_defs = _collect_definitions(root)

    functions: list[dict] = []

    def _scan(node: Node, class_name: str | None = None) -> None:
        for child in node.children:
            actual = child
            if child.type == "decorated_definition":
                inner = child.named_children[-1] if child.named_children else None
                if inner:
                    actual = inner
                else:
                    continue

            if actual.type == "function_definition":
                info = _extract_function_graph(
                    actual, local_defs, import_lookup, class_name
                )
                if info:
                    functions.append(info)

            elif actual.type == "class_definition":
                cname_node = actual.child_by_field_name("name")
                if cname_node:
                    cname = _node_text(cname_node)
                    body = actual.child_by_field_name("body")
                    if body:
                        _scan(body, class_name=cname)

    _scan(root)

    return {
        "path": rel_path,
        "source_hash": hashlib.sha256(source).hexdigest()[:12],
        "imports": imports,
        "functions": functions,
    }

"""Python structural extractor using tree-sitter."""

from __future__ import annotations

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

from ...model import ElementKind, FileStructure, Import, StructuralElement
from .. import Extractor, register_extractor


class PythonExtractor(Extractor):

    def __init__(self) -> None:
        lang = Language(tspython.language())
        self._parser = Parser(lang)

    def language(self) -> str:
        return "python"

    def extensions(self) -> set[str]:
        return {".py"}

    def extract(self, source: bytes, rel_path: str) -> FileStructure:
        tree = self._parser.parse(source)
        imports = _extract_imports(tree.root_node)
        elements = _extract_elements(tree.root_node, source, scope_path="")
        return FileStructure(
            path=rel_path,
            language="python",
            imports=imports,
            elements=elements,
        )


def _node_text(node: Node) -> str:
    return node.text.decode("utf-8")


def _extract_imports(root: Node) -> list[Import]:
    imports: list[Import] = []
    for child in root.children:
        if child.type in ("import_statement", "import_from_statement"):
            imports.append(Import(raw=_node_text(child)))
    return imports


def _extract_docstring(body_node: Node) -> str | None:
    """Extract docstring from the first statement of a block."""
    if body_node.type != "block" or body_node.child_count == 0:
        return None
    first = body_node.children[0]
    if first.type == "expression_statement" and first.child_count == 1:
        expr = first.children[0]
        if expr.type == "string":
            text = _node_text(expr)
            # Strip triple quotes
            for q in ('"""', "'''"):
                if text.startswith(q) and text.endswith(q):
                    return text[3:-3].strip()
            return text.strip('"').strip("'")
    return None


def _build_signature(node: Node) -> str:
    """Build a human-readable signature for a function/method."""
    name_node = node.child_by_field_name("name")
    params_node = node.child_by_field_name("parameters")
    return_node = node.child_by_field_name("return_type")

    parts = ["def "]
    parts.append(_node_text(name_node) if name_node else "?")
    parts.append(_node_text(params_node) if params_node else "()")
    if return_node:
        parts.append(f" -> {_node_text(return_node)}")
    return "".join(parts)


def _extract_decorators(node: Node) -> list[str]:
    """Extract decorator strings from a decorated_definition or function/class with decorators."""
    decorators: list[str] = []
    if node.type == "decorated_definition":
        for child in node.children:
            if child.type == "decorator":
                decorators.append(_node_text(child))
    return decorators


def _extract_class_bases(node: Node) -> str | None:
    """Extract base classes from a class definition."""
    arg_list = node.child_by_field_name("superclasses")
    if arg_list:
        return _node_text(arg_list)
    return None


def _build_class_signature(node: Node) -> str:
    name_node = node.child_by_field_name("name")
    name = _node_text(name_node) if name_node else "?"
    bases = _extract_class_bases(node)
    if bases:
        return f"class {name}{bases}"
    return f"class {name}"


def _extract_elements(
    parent: Node,
    source: bytes,
    scope_path: str,
) -> list[StructuralElement]:
    """Recursively extract structural elements from a node's children."""
    elements: list[StructuralElement] = []

    for child in parent.children:
        if child.type == "function_definition":
            elements.append(
                _extract_function(child, source, scope_path, decorators=[])
            )
        elif child.type == "class_definition":
            elements.append(
                _extract_class(child, source, scope_path, decorators=[])
            )
        elif child.type == "decorated_definition":
            decorators = _extract_decorators(child)
            # The actual definition is the last named child
            inner = child.named_children[-1] if child.named_children else None
            if inner and inner.type == "function_definition":
                elements.append(
                    _extract_function(inner, source, scope_path, decorators)
                )
            elif inner and inner.type == "class_definition":
                elements.append(
                    _extract_class(inner, source, scope_path, decorators)
                )
        elif child.type == "expression_statement":
            elem = _try_extract_constant(child, source, scope_path)
            if elem:
                elements.append(elem)

    return elements


def _extract_function(
    node: Node,
    source: bytes,
    scope_path: str,
    decorators: list[str],
) -> StructuralElement:
    name_node = node.child_by_field_name("name")
    name = _node_text(name_node) if name_node else "?"
    full_scope = f"{scope_path} > {name}" if scope_path else name

    # Determine if this is a method (inside a class scope) or a function
    kind = ElementKind.METHOD if scope_path else ElementKind.FUNCTION

    body_node = node.child_by_field_name("body")
    docstring = _extract_docstring(body_node) if body_node else None

    return StructuralElement(
        kind=kind,
        name=name,
        scope_path=full_scope,
        signature=_build_signature(node),
        decorators=decorators,
        docstring=docstring,
        body=_node_text(node),
        line_start=node.start_point.row + 1,
        line_end=node.end_point.row + 1,
    )


def _extract_class(
    node: Node,
    source: bytes,
    scope_path: str,
    decorators: list[str],
) -> StructuralElement:
    name_node = node.child_by_field_name("name")
    name = _node_text(name_node) if name_node else "?"
    full_scope = f"{scope_path} > {name}" if scope_path else name

    body_node = node.child_by_field_name("body")
    docstring = _extract_docstring(body_node) if body_node else None

    # Recursively extract methods and nested classes
    children: list[StructuralElement] = []
    if body_node:
        children = _extract_elements(body_node, source, scope_path=full_scope)

    return StructuralElement(
        kind=ElementKind.CLASS,
        name=name,
        scope_path=full_scope,
        signature=_build_class_signature(node),
        decorators=decorators,
        docstring=docstring,
        body=_node_text(node),
        line_start=node.start_point.row + 1,
        line_end=node.end_point.row + 1,
        children=children,
    )


def _try_extract_constant(
    node: Node,
    source: bytes,
    scope_path: str,
) -> StructuralElement | None:
    """Try to extract a module-level constant assignment (NAME = value)."""
    if node.child_count != 1:
        return None
    expr = node.children[0]
    if expr.type != "assignment":
        return None
    left = expr.child_by_field_name("left")
    if not left or left.type != "identifier":
        return None
    name = _node_text(left)
    # Only treat UPPER_CASE names as constants
    if not name.isupper() and not name.upper() == name:
        return None
    full_scope = f"{scope_path} > {name}" if scope_path else name
    return StructuralElement(
        kind=ElementKind.CONSTANT,
        name=name,
        scope_path=full_scope,
        signature=_node_text(expr),
        body=_node_text(node),
        line_start=node.start_point.row + 1,
        line_end=node.end_point.row + 1,
    )


# Auto-register on import
register_extractor(PythonExtractor())

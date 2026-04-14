"""Language-agnostic data model for structural code extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ElementKind(str, Enum):
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    STRUCT = "struct"
    NAMESPACE = "namespace"
    CONSTANT = "constant"


@dataclass
class Import:
    """A single import/include statement."""
    raw: str                        # Original source text: "from foo import bar", "#include <vector>"
    resolved_path: str | None = None  # Resolved to a project file, if possible


@dataclass
class StructuralElement:
    """A single structural element extracted from source code."""
    kind: ElementKind
    name: str
    scope_path: str                 # "Class > method" or "namespace > class > method"
    signature: str | None = None    # Human-readable: "def foo(x: int) -> str"
    decorators: list[str] = field(default_factory=list)
    docstring: str | None = None
    body: str = ""                  # Raw source of the element
    line_start: int = 0
    line_end: int = 0
    children: list[StructuralElement] = field(default_factory=list)


@dataclass
class FileStructure:
    """Complete structural extraction for a single source file."""
    path: str                       # Relative path from repo root
    language: str                   # "python", "cpp"
    imports: list[Import] = field(default_factory=list)
    elements: list[StructuralElement] = field(default_factory=list)

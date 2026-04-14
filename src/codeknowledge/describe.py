"""LLM-based code description generation.

Takes structural extraction output and produces natural language descriptions
of what each file and symbol does.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .model import FileStructure, StructuralElement

log = logging.getLogger(__name__)


@dataclass
class SymbolDescription:
    """Description of a single structural element."""
    scope_path: str
    kind: str
    signature: str | None
    description: str
    line_start: int
    line_end: int


@dataclass
class FileDescription:
    """Complete description output for a single source file."""
    path: str
    language: str
    summary: str
    symbols: list[SymbolDescription] = field(default_factory=list)


def build_prompt(
    structure: FileStructure,
    source: str,
    neighbor_context: dict[str, str] | None = None,
    architecture_context: str | None = None,
    project_name: str = "",
) -> str:
    """Build the LLM prompt for describing a file.

    Args:
        structure: Tree-sitter extraction output for the file.
        source: Raw source code of the file.
        neighbor_context: {rel_path: source_or_summary} for neighboring files.
        architecture_context: Architecture overview text to inform descriptions.
        project_name: Name of the project for context.
    """
    parts: list[str] = []

    # Header
    proj = f" from the project `{project_name}`" if project_name else ""
    parts.append(f"Here is the file `{structure.path}`{proj}.\n")

    # Architecture context — gives the model the big picture
    if architecture_context:
        parts.append(f"\n## Architecture context\n\n{architecture_context}\n\n")

    # Neighbor context
    if neighbor_context:
        parts.append("## Other files in the same directory\n")
        for path, content in neighbor_context.items():
            parts.append(f"### {path}\n```\n{content}\n```\n")

    # The source file itself
    parts.append(f"### {structure.path}\n```{structure.language}\n{source}\n```\n")

    # Structural skeleton — tell the model what elements to describe
    parts.append("## Structural elements to describe\n")
    parts.append(
        "Produce a description for each of the following elements. "
        "For each one, explain what it does, its key inputs and outputs, "
        "and any notable implementation details.\n"
    )
    _append_element_list(parts, structure.elements, indent=0)

    # Output format instructions
    parts.append("""
## Output format

Respond with ONLY the following structure (no code fences around the whole response):

Start with a file summary paragraph — what this module's responsibility is and how it fits in the project.

Then for each structural element, use a markdown heading with the scope path and write a description paragraph:

### `scope_path`

Description of what this element does...

Use the exact scope paths listed above as heading text. Do not add elements that aren't listed. Do not include the raw source code in your response.
""")

    return "".join(parts)


def _append_element_list(
    parts: list[str],
    elements: list[StructuralElement],
    indent: int,
) -> None:
    prefix = "  " * indent
    for elem in elements:
        sig = f" — `{elem.signature}`" if elem.signature else ""
        parts.append(f"{prefix}- **{elem.scope_path}** ({elem.kind.value}){sig}\n")
        if elem.children:
            _append_element_list(parts, elem.children, indent + 1)


def parse_response(
    response_text: str,
    structure: FileStructure,
) -> FileDescription:
    """Parse the LLM response into a structured FileDescription.

    Extracts the file summary (text before first ###) and per-symbol
    descriptions (text under each ### heading matched to scope paths).
    """
    lines = response_text.split("\n")
    summary_lines: list[str] = []
    current_scope: str | None = None
    current_lines: list[str] = []
    symbols: list[SymbolDescription] = []

    # Build a lookup for structural elements by scope path
    element_map = _build_element_map(structure.elements)

    for line in lines:
        if line.startswith("### "):
            # Flush previous section
            if current_scope is not None:
                _flush_symbol(symbols, current_scope, current_lines, element_map)
            elif summary_lines or current_lines:
                summary_lines = summary_lines or current_lines

            # Parse heading — strip backticks and whitespace
            heading = line[4:].strip().strip("`").strip()
            current_scope = heading
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    if current_scope is not None:
        _flush_symbol(symbols, current_scope, current_lines, element_map)
    elif current_lines:
        summary_lines = current_lines

    summary = "\n".join(summary_lines).strip()

    return FileDescription(
        path=structure.path,
        language=structure.language,
        summary=summary,
        symbols=symbols,
    )


def _build_element_map(
    elements: list[StructuralElement],
) -> dict[str, StructuralElement]:
    result: dict[str, StructuralElement] = {}
    for elem in elements:
        result[elem.scope_path] = elem
        if elem.children:
            result.update(_build_element_map(elem.children))
    return result


def _flush_symbol(
    symbols: list[SymbolDescription],
    scope_path: str,
    lines: list[str],
    element_map: dict[str, StructuralElement],
) -> None:
    description = "\n".join(lines).strip()
    if not description:
        return

    elem = element_map.get(scope_path)
    symbols.append(SymbolDescription(
        scope_path=scope_path,
        kind=elem.kind.value if elem else "unknown",
        signature=elem.signature if elem else None,
        description=description,
        line_start=elem.line_start if elem else 0,
        line_end=elem.line_end if elem else 0,
    ))


def render_description_markdown(
    desc: FileDescription,
    source_hash: str = "",
    commit: str = "",
    model: str = "",
) -> str:
    """Render a FileDescription as a markdown file with YAML frontmatter."""
    lines: list[str] = []

    # Frontmatter
    lines.append("---")
    lines.append(f"source: {desc.path}")
    if source_hash:
        lines.append(f"source_hash: {source_hash}")
    if commit:
        lines.append(f"commit: {commit}")
    if model:
        lines.append(f"model: {model}")
    # Symbol table in frontmatter
    if desc.symbols:
        lines.append("symbols:")
        for sym in desc.symbols:
            lines.append(f"  - scope_path: \"{sym.scope_path}\"")
            lines.append(f"    kind: {sym.kind}")
            lines.append(f"    lines: {sym.line_start}-{sym.line_end}")
    lines.append("---")
    lines.append("")

    # File summary
    lines.append(f"# {desc.path}")
    lines.append("")
    lines.append(desc.summary)
    lines.append("")

    # Symbol descriptions
    for sym in desc.symbols:
        sig = f" — `{sym.signature}`" if sym.signature else ""
        lines.append(f"## `{sym.scope_path}`{sig}")
        lines.append("")
        lines.append(sym.description)
        lines.append("")

    return "\n".join(lines)

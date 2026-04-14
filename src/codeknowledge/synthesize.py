"""Top-down architectural synthesis.

Reads raw source code and structural skeletons to produce architecture
overviews, flow documents, and other high-level understanding.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .model import FileStructure, StructuralElement

log = logging.getLogger(__name__)


def _format_skeleton(fs: FileStructure) -> str:
    """Format a FileStructure as a concise structural skeleton."""
    lines: list[str] = []
    if fs.imports:
        for imp in fs.imports:
            lines.append(imp.raw)
        lines.append("")
    _format_elements(fs.elements, lines, indent=0)
    return "\n".join(lines)


def _format_elements(elements: list[StructuralElement], lines: list[str], indent: int) -> None:
    prefix = "  " * indent
    for elem in elements:
        sig = elem.signature or elem.name
        doc = f'  """{elem.docstring}"""' if elem.docstring else ""
        lines.append(f"{prefix}{sig}{doc}")
        if elem.children:
            _format_elements(elem.children, lines, indent + 1)


# ---------------------------------------------------------------------------
# Architecture overview
# ---------------------------------------------------------------------------

def build_architecture_prompt(
    files: list[tuple[FileStructure, str]],
    project_name: str = "",
) -> str:
    """Build a prompt for a project-level architecture overview.

    Args:
        files: List of (FileStructure, raw_source) tuples for all files.
        project_name: Project name for context.
    """
    parts: list[str] = []

    proj = project_name or "this project"
    parts.append(f"Analyze the complete source code of `{proj}` and produce an architecture document.\n\n")

    for fs, source in files:
        lang = fs.language
        parts.append(f"### {fs.path} — Structural skeleton\n```\n{_format_skeleton(fs)}\n```\n\n")
        parts.append(f"### {fs.path} — Source\n```{lang}\n{source}\n```\n\n")

    parts.append("""Produce a project-level architecture document that covers:

1. **Project purpose**: What does this system do? What problem does it solve?
2. **Architecture overview**: How is the system organized? What are the major components and their responsibilities?
3. **Data flow**: How does data flow through the system? What are the key pipelines or processing paths?
4. **Key design decisions**: What architectural choices were made and why are they significant?
5. **Entry points**: How is the system invoked? What are the main user-facing interfaces?

Write in clear prose with markdown headings for each section. Be specific and concrete — reference actual file names, function names, data structures, and constants. Do not generalize when you can be precise.
""")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Key flow documents
# ---------------------------------------------------------------------------

def build_flow_prompt(
    flow_name: str,
    flow_description: str,
    files: list[tuple[FileStructure, str]],
    architecture: str,
    project_name: str = "",
) -> str:
    """Build a prompt for tracing a specific execution flow.

    Args:
        flow_name: Short name for the flow (e.g., "Search", "Indexing").
        flow_description: Brief description of what to trace.
        files: List of (FileStructure, raw_source) tuples.
        architecture: The architecture overview (from the previous synthesis step).
        project_name: Project name.
    """
    parts: list[str] = []

    proj = f" in `{project_name}`" if project_name else ""
    parts.append(f"Trace the **{flow_name}** flow{proj}.\n\n")
    parts.append(f"{flow_description}\n\n")

    parts.append(f"## Architecture context\n\n{architecture}\n\n")

    parts.append("## Source code\n\n")
    for fs, source in files:
        lang = fs.language
        parts.append(f"### {fs.path}\n```{lang}\n{source}\n```\n\n")

    parts.append(f"""Produce a document tracing the {flow_name} flow end-to-end:

1. Start with where the flow is initiated (entry point, user action, CLI command).
2. Trace through each function/module involved, in execution order.
3. Note what data is transformed at each step.
4. Identify the key decision points and branching logic.
5. End with the final output or side effect.

Use markdown headings for major phases. Reference specific files and functions by name. Be precise — use actual parameter names, return types, and data structures. Write to help someone understand how this feature actually works.
""")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Flow identification
# ---------------------------------------------------------------------------

def build_flow_identification_prompt(
    architecture: str,
    project_name: str = "",
) -> str:
    """Build a prompt to identify key execution flows from an architecture overview."""
    proj = project_name or "this project"
    return (
        f"Based on this architecture overview of `{proj}`:\n\n"
        f"{architecture}\n\n"
        "List the 3-5 most important execution flows a developer would need "
        "to understand. For each flow, give:\n"
        "1. A short name (2-4 words)\n"
        "2. A one-sentence description of what to trace\n\n"
        "Format each as:\n"
        "FLOW: <name>\n"
        "TRACE: <description>\n"
    )


def parse_flows(response: str) -> list[tuple[str, str]]:
    """Parse flow identification response into (name, description) pairs."""
    flows: list[tuple[str, str]] = []
    current_name: str | None = None

    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("FLOW:"):
            current_name = line[5:].strip()
        elif line.startswith("TRACE:") and current_name:
            flows.append((current_name, line[6:].strip()))
            current_name = None

    return flows


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_article(
    title: str,
    content: str,
    article_type: str = "synthesis",
    sources: list[str] | None = None,
    model: str = "",
) -> str:
    """Render a synthesis article as markdown with YAML frontmatter."""
    lines: list[str] = []
    lines.append("---")
    lines.append(f"title: \"{title}\"")
    lines.append(f"type: {article_type}")
    if model:
        lines.append(f"model: {model}")
    if sources:
        lines.append("sources:")
        for s in sources:
            lines.append(f"  - \"{s}\"")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append("")
    lines.append(content)
    lines.append("")
    return "\n".join(lines)

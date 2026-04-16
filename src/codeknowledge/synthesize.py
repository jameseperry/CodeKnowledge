"""Top-down architectural synthesis.

Reads raw source code and structural skeletons to produce architecture
overviews, flow documents, and other high-level understanding.

Supports two modes:
- **Single-pass**: all source fits in context (small projects).
- **Batched**: project skeleton is always present, source is sent
  one directory at a time, then architecture is built from module summaries.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .model import FileStructure, StructuralElement

log = logging.getLogger(__name__)

# Rough char budget for source in a single prompt.  Sonnet has ~200K context
# but we want headroom for skeleton + instructions + output.
SOURCE_CHAR_BUDGET = 400_000


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


def build_full_skeleton(files: list[tuple[FileStructure, str]]) -> str:
    """Build a full project skeleton from all files (no source bodies)."""
    parts: list[str] = []
    for fs, _ in files:
        parts.append(f"### {fs.path}\n```\n{_format_skeleton(fs)}\n```\n")
    return "\n".join(parts)


def total_source_chars(files: list[tuple[FileStructure, str]]) -> int:
    """Total chars of raw source across all files."""
    return sum(len(source) for _, source in files)


def group_by_directory(
    files: list[tuple[FileStructure, str]],
) -> dict[str, list[tuple[FileStructure, str]]]:
    """Group files by their parent directory path."""
    groups: dict[str, list[tuple[FileStructure, str]]] = {}
    for fs, source in files:
        dir_path = str(Path(fs.path).parent)
        groups.setdefault(dir_path, []).append((fs, source))
    return groups


def needs_batching(files: list[tuple[FileStructure, str]]) -> bool:
    """Check whether the project is too large for single-pass synthesis."""
    return total_source_chars(files) > SOURCE_CHAR_BUDGET


# Char limit for source within a single module-summary prompt.
# Skeleton is ~270K for TensileLite, so source budget per prompt is tighter.
MODULE_SOURCE_LIMIT = 300_000


def split_module_batches(
    module_files: list[tuple[FileStructure, str]],
) -> list[list[tuple[FileStructure, str]]]:
    """Split a module's files into sub-batches that each fit under MODULE_SOURCE_LIMIT."""
    total = sum(len(s) for _, s in module_files)
    if total <= MODULE_SOURCE_LIMIT:
        return [module_files]

    batches: list[list[tuple[FileStructure, str]]] = []
    current: list[tuple[FileStructure, str]] = []
    current_chars = 0

    for fs, source in module_files:
        n = len(source)
        if current and current_chars + n > MODULE_SOURCE_LIMIT:
            batches.append(current)
            current = []
            current_chars = 0
        current.append((fs, source))
        current_chars += n

    if current:
        batches.append(current)

    return batches


def build_merge_summaries_prompt(
    module_path: str,
    partial_summaries: list[str],
    project_name: str = "",
) -> str:
    """Merge multiple partial summaries of the same module into one."""
    proj = project_name or "this project"
    parts: list[str] = []
    parts.append(
        f"The `{module_path}` module of `{proj}` was analyzed in "
        f"{len(partial_summaries)} batches. Merge the partial summaries below "
        "into a single coherent module summary.\n\n"
    )
    for i, summary in enumerate(partial_summaries, 1):
        parts.append(f"## Batch {i}\n{summary}\n\n")

    parts.append(
        "Produce a single unified module summary covering purpose, key abstractions, "
        "internal structure, external interfaces, and notable patterns. "
        "Deduplicate and reconcile any overlapping information."
    )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Module summaries (batched mode)
# ---------------------------------------------------------------------------

# Max chars of source to include per file.  Files over this get truncated.
# The skeleton is always present for structural context.
FILE_SOURCE_CAP = 150_000


def build_module_summary_prompt(
    module_path: str,
    module_files: list[tuple[FileStructure, str]],
    project_skeleton: str,
    project_name: str = "",
) -> str:
    """Build a prompt to summarize one module (directory).

    Args:
        module_path: Relative path of the directory being summarized.
        module_files: (FileStructure, raw_source) for files in this directory.
        project_skeleton: Full project skeleton (all files, no bodies).
        project_name: Project name for context.
    """
    parts: list[str] = []
    proj = project_name or "this project"

    parts.append(
        f"You are analyzing the `{module_path}` module of `{proj}`.\n\n"
        "Below is the full project skeleton (structural outlines of every file) "
        "so you understand where this module fits in the larger codebase.\n\n"
    )

    parts.append(f"## Full project skeleton\n\n{project_skeleton}\n\n")

    parts.append(f"## Source code for `{module_path}`\n\n")
    for fs, source in module_files:
        lang = fs.language
        if len(source) > FILE_SOURCE_CAP:
            truncated = source[:FILE_SOURCE_CAP]
            parts.append(
                f"### {fs.path} (TRUNCATED — {len(source):,} chars, showing first {FILE_SOURCE_CAP:,})\n"
                f"```{lang}\n{truncated}\n```\n\n"
                f"*Note: This file is very large. The structural skeleton above has the full outline.*\n\n"
            )
        else:
            parts.append(f"### {fs.path}\n```{lang}\n{source}\n```\n\n")

    parts.append(f"""Produce a module summary for `{module_path}` covering:

1. **Purpose**: What does this module do? What responsibility does it own?
2. **Key abstractions**: What are the important classes, functions, and data structures? What do they represent?
3. **Internal structure**: How do the files within this module relate to each other?
4. **External interfaces**: What does this module export or provide to the rest of the project? What does it depend on from other modules?
5. **Notable patterns**: Any important design patterns, algorithms, or non-obvious behaviors.

Be specific — reference file names, class names, function signatures. Keep the summary concise but complete enough that someone could understand this module's role without reading its source.
""")

    return "".join(parts)


# ---------------------------------------------------------------------------
# Architecture overview
# ---------------------------------------------------------------------------

def build_architecture_prompt(
    files: list[tuple[FileStructure, str]],
    project_name: str = "",
) -> str:
    """Build a prompt for single-pass architecture overview (small projects).

    Sends all source + skeletons. Use only when `needs_batching()` is False.
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


def build_architecture_from_summaries_prompt(
    project_skeleton: str,
    module_summaries: dict[str, str],
    project_name: str = "",
) -> str:
    """Build a prompt for architecture overview from module summaries (batched mode).

    Args:
        project_skeleton: Full project skeleton (all files, no bodies).
        module_summaries: {dir_path: summary} for all modules.
        project_name: Project name for context.
    """
    parts: list[str] = []
    proj = project_name or "this project"

    parts.append(
        f"Synthesize an architecture document for `{proj}` from the structural "
        "skeleton and per-module summaries below.\n\n"
    )

    parts.append(f"## Full project skeleton\n\n{project_skeleton}\n\n")

    parts.append("## Module summaries\n\n")
    for dir_path, summary in module_summaries.items():
        parts.append(f"### {dir_path}\n{summary}\n\n")

    parts.append("""Produce a project-level architecture document that covers:

1. **Project purpose**: What does this system do? What problem does it solve?
2. **Architecture overview**: How is the system organized? What are the major components and their responsibilities?
3. **Data flow**: How does data flow through the system? What are the key pipelines or processing paths?
4. **Key design decisions**: What architectural choices were made and why are they significant?
5. **Entry points**: How is the system invoked? What are the main user-facing interfaces?
6. **Module relationships**: How do the modules depend on and interact with each other?

Write in clear prose with markdown headings for each section. Be specific and concrete — reference actual file names, function names, data structures, and constants from the skeleton here. You have structural detail for every file; use it. Do not generalize when you can be precise.
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


def build_flow_prompt_from_skeleton(
    flow_name: str,
    flow_description: str,
    project_skeleton: str,
    module_summaries: dict[str, str],
    architecture: str,
    project_name: str = "",
) -> str:
    """Build a flow-tracing prompt for batched mode (no raw source).

    Uses the architecture overview, full skeleton, and module summaries
    instead of raw source code.
    """
    parts: list[str] = []

    proj = f" in `{project_name}`" if project_name else ""
    parts.append(f"Trace the **{flow_name}** flow{proj}.\n\n")
    parts.append(f"{flow_description}\n\n")

    parts.append(f"## Architecture context\n\n{architecture}\n\n")

    parts.append(f"## Full project skeleton\n\n{project_skeleton}\n\n")

    parts.append("## Module summaries\n\n")
    for dir_path, summary in module_summaries.items():
        parts.append(f"### {dir_path}\n{summary}\n\n")

    parts.append(f"""Produce a document tracing the {flow_name} flow end-to-end:

1. Start with where the flow is initiated (entry point, user action, CLI command).
2. Trace through each function/module involved, in execution order.
3. Note what data is transformed at each step.
4. Identify the key decision points and branching logic.
5. End with the final output or side effect.

Use markdown headings for major phases. Reference specific files and functions by name. Be precise — use actual parameter names, return types, and data structures from the skeleton. Write to help someone understand how this feature actually works.
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
    """Parse flow identification response into (name, description) pairs.

    Tolerates markdown formatting: **FLOW:**, `FLOW:`, numbered prefixes, etc.
    """
    import re

    flows: list[tuple[str, str]] = []
    current_name: str | None = None

    # Strip markdown bold/backtick/italic around labels
    def _clean(line: str) -> str:
        return re.sub(r"[*_`#\d.)\-]+\s*", "", line, count=1).strip()

    for raw_line in response.split("\n"):
        line = raw_line.strip()
        # Try to match FLOW: with optional markdown cruft before it
        cleaned = _clean(line)
        if cleaned.upper().startswith("FLOW:"):
            current_name = cleaned[5:].strip().strip("*_`")
        elif cleaned.upper().startswith("TRACE:") and current_name:
            flows.append((current_name, cleaned[6:].strip().strip("*_`")))
            current_name = None

    return flows


# ---------------------------------------------------------------------------
# Significance evaluation (incremental synthesis)
# ---------------------------------------------------------------------------

# Cap the diff text sent to the evaluator to keep the prompt small.
DIFF_CAP = 30_000


def build_significance_prompt(
    article_title: str,
    article_content: str,
    diff_text: str,
) -> str:
    """Build a prompt asking whether a git diff warrants re-synthesizing an article.

    Returns a prompt for a cheap LLM call (haiku-tier).
    """
    # Truncate the diff if it's huge — if the diff itself is massive,
    # that's a strong signal that regeneration is needed.
    if len(diff_text) > DIFF_CAP:
        truncated = diff_text[:DIFF_CAP]
        diff_section = (
            f"```diff\n{truncated}\n```\n\n"
            f"*(Diff truncated — {len(diff_text):,} chars total, showing first {DIFF_CAP:,})*\n"
        )
    else:
        diff_section = f"```diff\n{diff_text}\n```\n"

    return (
        "You are evaluating whether a code change is significant enough to warrant "
        "re-generating a synthesis article.\n\n"
        f"## Existing article: {article_title}\n\n"
        f"{article_content}\n\n"
        f"## Code changes since this article was generated\n\n"
        f"{diff_section}\n"
        "Does this diff change the architectural understanding captured in the article above? "
        "Consider: new/removed modules, changed APIs, altered data flows, modified key algorithms, "
        "renamed core abstractions. Ignore: whitespace, comments, minor refactors, test-only changes, "
        "formatting, version bumps.\n\n"
        "Answer with exactly one line:\n"
        "VERDICT: YES or NO\n"
        "Then optionally a one-sentence reason.\n"
    )


def parse_significance_verdict(response: str) -> bool:
    """Parse the evaluator response. Returns True if regeneration is needed."""
    for line in response.strip().split("\n"):
        line = line.strip().upper()
        if line.startswith("VERDICT:"):
            return "YES" in line
    # If we can't parse, regenerate to be safe
    return True


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

def parse_article_frontmatter(text: str) -> dict[str, str]:
    """Parse simple YAML frontmatter from an article, returning flat key-value pairs.

    Only handles top-level scalar values (not nested sources mapping).
    """
    result: dict[str, str] = {}
    if not text.startswith("---"):
        return result
    end = text.find("---", 3)
    if end < 0:
        return result
    for line in text[3:end].strip().split("\n"):
        line = line.strip()
        if ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip().strip('"')
    return result


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_article(
    title: str,
    content: str,
    article_type: str = "synthesis",
    sources: dict[str, str] | None = None,
    model: str = "",
    commit: str | None = None,
    commit_dirty: bool = False,
) -> str:
    """Render a synthesis article as markdown with YAML frontmatter.

    Args:
        sources: Mapping of relative path → content hash.
        commit: Git commit SHA at generation time.
        commit_dirty: True if working tree had uncommitted changes.
    """
    lines: list[str] = []
    lines.append("---")
    lines.append(f"title: \"{title}\"")
    lines.append(f"type: {article_type}")
    if model:
        lines.append(f"model: {model}")
    if commit:
        lines.append(f"commit: {commit}")
    if commit_dirty:
        lines.append("commit_dirty: true")
    if sources:
        lines.append("sources:")
        for path, hash_val in sources.items():
            lines.append(f"  {path}: {hash_val}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {title}")
    lines.append("")
    lines.append(content)
    lines.append("")
    return "\n".join(lines)

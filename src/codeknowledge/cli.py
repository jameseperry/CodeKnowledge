"""Click CLI for CodeKnowledge operations."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import click

from .model import FileStructure
from .extractors import get_extractor
from .extractors import python as _  # noqa: F401 — register extractor


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _extract_file(file_path: Path, repo_root: Path) -> FileStructure | None:
    extractor = get_extractor(file_path)
    if not extractor:
        return None
    source = file_path.read_bytes()
    rel_path = str(file_path.relative_to(repo_root))
    return extractor.extract(source, rel_path)


def _collect_files(path: Path, repo_root: Path) -> list[tuple[Path, FileStructure]]:
    """Collect all extractable files under a path."""
    results = []
    if path.is_file():
        fs = _extract_file(path, repo_root)
        if fs:
            results.append((path, fs))
    elif path.is_dir():
        for file_path in sorted(path.rglob("*")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(repo_root)
            if any(p.startswith(".") for p in rel.parts):
                continue
            fs = _extract_file(file_path, repo_root)
            if fs:
                results.append((file_path, fs))
    return results


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# CLI root
# ---------------------------------------------------------------------------

@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """CodeKnowledge — build persistent understanding of codebases."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# ---------------------------------------------------------------------------
# extract — structural extraction (no LLM)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--repo-root", type=click.Path(exists=True), default=None,
              help="Repository root for relative paths.")
@click.option("--body", is_flag=True, help="Show element bodies")
@click.pass_context
def extract(ctx: click.Context, path: str, repo_root: str | None, body: bool) -> None:
    """Extract structural elements from source files."""
    target = Path(path).resolve()
    root = Path(repo_root).resolve() if repo_root else (target if target.is_dir() else target.parent)

    files = _collect_files(target, root)
    if not files:
        click.echo("No extractable files found.", err=True)
        sys.exit(1)

    for _, fs in files:
        _print_structure(fs, show_body=body)

    click.echo(f"\nExtracted structure from {len(files)} files.")


def _print_structure(fs: FileStructure, show_body: bool = False) -> None:
    click.echo(f"{'=' * 60}")
    click.echo(f"File: {fs.path}  (language: {fs.language})")
    click.echo(f"{'=' * 60}")

    if fs.imports:
        click.echo(f"\nImports ({len(fs.imports)}):")
        for imp in fs.imports:
            click.echo(f"  {imp.raw}")

    if fs.elements:
        click.echo(f"\nElements ({len(fs.elements)}):")
        _print_elements(fs.elements, indent=1, show_body=show_body)

    click.echo()


def _print_elements(elements, indent: int, show_body: bool) -> None:
    prefix = "  " * indent
    for elem in elements:
        click.echo(f"{prefix}[{elem.kind.value}] {elem.scope_path}")
        click.echo(f"{prefix}  signature: {elem.signature}")
        click.echo(f"{prefix}  lines: {elem.line_start}-{elem.line_end}")
        if elem.decorators:
            click.echo(f"{prefix}  decorators: {', '.join(elem.decorators)}")
        if elem.docstring:
            doc_preview = elem.docstring[:80].replace("\n", " ")
            click.echo(f"{prefix}  docstring: {doc_preview}")
        if show_body:
            click.echo(f"{prefix}  body:")
            for line in elem.body.splitlines():
                click.echo(f"{prefix}    {line}")
        if elem.children:
            _print_elements(elem.children, indent + 1, show_body=show_body)


# ---------------------------------------------------------------------------
# describe — LLM description generation (with architecture context)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--repo-root", type=click.Path(exists=True), default=None,
              help="Repository root.")
@click.option("--output-dir", type=click.Path(), default=None,
              help="Output directory. Defaults to .codeknowledge/descriptions/ under repo root.")
@click.option("--articles-dir", type=click.Path(exists=True), default=None,
              help="Directory with synthesis articles (architecture-overview.md). "
                   "Defaults to .codeknowledge/articles/ under repo root.")
@click.option("--model", default="sonnet", help="Model tier: haiku, sonnet, opus, or a model name.")
@click.option("--dry-run", is_flag=True, help="Print the prompt without calling the LLM.")
@click.option("--file-filter", default=None, help="Only process files matching this substring.")
@click.pass_context
def describe(
    ctx: click.Context,
    path: str,
    repo_root: str | None,
    output_dir: str | None,
    articles_dir: str | None,
    model: str,
    dry_run: bool,
    file_filter: str | None,
) -> None:
    """Generate LLM descriptions for source files.

    Extracts structure via tree-sitter, sends each file to an LLM with
    sibling-file context and architecture context, and writes description
    markdown files.
    """
    from .describe import build_prompt, parse_response, render_description_markdown
    from .llm import describe_file

    target = Path(path).resolve()
    root = Path(repo_root).resolve() if repo_root else (target if target.is_dir() else target.parent)
    out = Path(output_dir) if output_dir else root / ".codeknowledge" / "descriptions"

    # Load architecture context if available
    arch_dir = Path(articles_dir).resolve() if articles_dir else root / ".codeknowledge" / "articles"
    architecture_context: str | None = None
    arch_file = arch_dir / "architecture-overview.md"
    if arch_file.is_file():
        # Strip YAML frontmatter, just use the content
        raw = arch_file.read_text()
        if raw.startswith("---"):
            end = raw.find("---", 3)
            if end != -1:
                architecture_context = raw[end + 3:].strip()
            else:
                architecture_context = raw
        else:
            architecture_context = raw
        click.echo(f"Loaded architecture context from {arch_file.name} ({len(architecture_context)} chars)")
    else:
        click.echo("No architecture context found. Run 'synthesize' first for better descriptions.")

    all_files = _collect_files(target, root)

    # Group ALL files by directory for neighbor context (before filtering)
    dir_groups: dict[Path, list[tuple[Path, FileStructure]]] = {}
    for file_path, fs in all_files:
        dir_groups.setdefault(file_path.parent, []).append((file_path, fs))

    # Now filter for processing
    files = all_files
    if file_filter:
        files = [(p, fs) for p, fs in files if file_filter in str(p)]

    if not files:
        click.echo("No extractable files found.", err=True)
        sys.exit(1)

    click.echo(f"Found {len(files)} files to describe.")

    for i, (file_path, fs) in enumerate(files):
        rel = fs.path
        click.echo(f"\n[{i+1}/{len(files)}] {rel}")

        source = file_path.read_text()

        # Neighbor context: other files in same directory
        siblings = dir_groups.get(file_path.parent, [])
        neighbor_ctx: dict[str, str] = {}
        neighbor_chars = 0
        max_neighbor_chars = 50_000
        for sib_path, sib_fs in siblings:
            if sib_path == file_path:
                continue
            sib_source = sib_path.read_text()
            if neighbor_chars + len(sib_source) > max_neighbor_chars:
                continue
            neighbor_ctx[sib_fs.path] = sib_source
            neighbor_chars += len(sib_source)

        prompt = build_prompt(
            structure=fs,
            source=source,
            neighbor_context=neighbor_ctx if neighbor_ctx else None,
            architecture_context=architecture_context,
            project_name=root.name,
        )

        if dry_run:
            click.echo(f"--- Prompt ({len(prompt)} chars) ---")
            click.echo(prompt)
            click.echo("--- End prompt ---")
            continue

        response_text = describe_file(prompt, model_tier=model)
        desc = parse_response(response_text, fs)
        md = render_description_markdown(
            desc,
            source_hash=_file_hash(file_path),
            model=model,
        )

        out_path = out / (rel + ".md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        try:
            display_path = out_path.relative_to(root)
        except ValueError:
            display_path = out_path
        click.echo(f"  -> {display_path}")
        click.echo(f"     {len(desc.symbols)} symbols described")


# ---------------------------------------------------------------------------
# init — initialize .codeknowledge directory
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.pass_context
def init(ctx: click.Context, path: str) -> None:
    """Initialize a .codeknowledge directory in a repository."""
    root = Path(path).resolve()
    ck_dir = root / ".codeknowledge"

    if ck_dir.exists():
        click.echo(f".codeknowledge already exists at {ck_dir}")
        return

    ck_dir.mkdir()
    (ck_dir / "descriptions").mkdir()
    (ck_dir / "articles").mkdir()

    manifest = ck_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "repo_root": str(root),
        "created_at": _now_iso(),
    }, indent=2) + "\n")

    # Add .codeknowledge to parent .gitignore
    gitignore = root / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        if ".codeknowledge/" not in content:
            with gitignore.open("a") as f:
                f.write("\n# CodeKnowledge data (separate repo)\n.codeknowledge/\n")
            click.echo("Added .codeknowledge/ to .gitignore")
    else:
        gitignore.write_text("# CodeKnowledge data (separate repo)\n.codeknowledge/\n")
        click.echo("Created .gitignore with .codeknowledge/ entry")

    click.echo(f"Initialized {ck_dir}")


# ---------------------------------------------------------------------------
# synthesize — top-down architecture synthesis
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--repo-root", type=click.Path(exists=True), default=None,
              help="Repository root for relative paths.")
@click.option("--output-dir", type=click.Path(), default=None,
              help="Output directory for articles. Defaults to .codeknowledge/articles/ under repo root.")
@click.option("--model", default="sonnet", help="Model tier: haiku, sonnet, opus, or a model name.")
@click.option("--project-name", default=None, help="Project name for context.")
@click.option("--dry-run", is_flag=True, help="Print prompts without calling the LLM.")
@click.option("--skip-flows", is_flag=True, help="Skip flow document generation.")
@click.pass_context
def synthesize(
    ctx: click.Context,
    path: str,
    repo_root: str | None,
    output_dir: str | None,
    model: str,
    project_name: str | None,
    dry_run: bool,
    skip_flows: bool,
) -> None:
    """Synthesize architecture articles from source code.

    Extracts structure via tree-sitter, reads raw source, and produces
    an architecture overview and key flow documents.
    """
    from .synthesize import (
        build_architecture_prompt,
        build_flow_identification_prompt,
        build_flow_prompt,
        parse_flows,
        render_article,
    )
    from .llm import describe_file

    target = Path(path).resolve()
    root = Path(repo_root).resolve() if repo_root else (target if target.is_dir() else target.parent)
    out = Path(output_dir).resolve() if output_dir else root / ".codeknowledge" / "articles"
    out.mkdir(parents=True, exist_ok=True)

    proj = project_name or root.name

    # Collect all files via tree-sitter extraction + raw source
    extracted = _collect_files(target, root)
    if not extracted:
        click.echo("No extractable files found.", err=True)
        sys.exit(1)

    files: list[tuple[FileStructure, str]] = []
    for file_path, fs in extracted:
        source = file_path.read_text()
        files.append((fs, source))

    click.echo(f"Collected {len(files)} source files.")

    # ---- Step 1: Architecture overview ----
    click.echo("\n--- Architecture overview ---")

    arch_prompt = build_architecture_prompt(files=files, project_name=proj)

    if dry_run:
        click.echo(f"  Prompt: {len(arch_prompt)} chars")
        click.echo(arch_prompt[:2000])
        click.echo("...")
        return

    architecture = describe_file(arch_prompt, model_tier=model, max_tokens=8192)

    article_path = out / "architecture-overview.md"
    sources = [fs.path for fs, _ in files]
    article = render_article(
        title="Architecture Overview",
        content=architecture,
        article_type="architecture",
        sources=sources,
        model=model,
    )
    article_path.write_text(article)
    click.echo(f"  -> {article_path.name}")

    # ---- Step 2: Key flow documents ----
    if not skip_flows:
        click.echo("\n--- Identifying key flows ---")

        flow_id_prompt = build_flow_identification_prompt(
            architecture=architecture,
            project_name=proj,
        )
        flow_response = describe_file(flow_id_prompt, model_tier=model, max_tokens=2048)
        flows = parse_flows(flow_response)

        click.echo(f"  Identified {len(flows)} flows.")
        for name, desc in flows:
            click.echo(f"    - {name}: {desc}")

        for name, description in flows:
            click.echo(f"\n  Flow: {name}")

            prompt = build_flow_prompt(
                flow_name=name,
                flow_description=description,
                files=files,
                architecture=architecture,
                project_name=proj,
            )

            response = describe_file(prompt, model_tier=model, max_tokens=8192)

            safe_name = name.lower().replace(" ", "-")
            article_path = out / f"flow-{safe_name}.md"
            article = render_article(
                title=f"Flow: {name}",
                content=response,
                article_type="flow",
                sources=sources,
                model=model,
            )
            article_path.write_text(article)
            click.echo(f"  -> {article_path.name}")

    click.echo("\nSynthesis complete.")


if __name__ == "__main__":
    cli()

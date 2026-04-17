"""Click CLI for CodeKnowledge operations."""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import sys
from pathlib import Path

import click

from .config import Config
from .model import FileStructure
from .extractors import get_extractor
from .extractors import python as _  # noqa: F401 — register extractor


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    # Only make our own loggers verbose
    logging.getLogger("codeknowledge").setLevel(level)


def _extract_file(file_path: Path, repo_root: Path) -> FileStructure | None:
    extractor = get_extractor(file_path)
    if not extractor:
        return None
    source = file_path.read_bytes()
    rel_path = str(file_path.relative_to(repo_root))
    return extractor.extract(source, rel_path)


def _collect_files(
    path: Path,
    repo_root: Path,
    exclude: list[str] | None = None,
) -> list[tuple[Path, FileStructure]]:
    """Collect all extractable files under a path.

    Args:
        path: File or directory to scan.
        repo_root: Repository root for relative path computation.
        exclude: Glob patterns to exclude (matched against relative paths).
    """
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
            rel_str = str(rel)
            if any(p.startswith(".") for p in rel.parts):
                continue
            if exclude and any(fnmatch.fnmatch(rel_str, pat) for pat in exclude):
                continue
            fs = _extract_file(file_path, repo_root)
            if fs:
                results.append((file_path, fs))
    return results


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _find_repo_root(start: Path) -> Path | None:
    """Walk up from *start* looking for a directory that contains .codeknowledge/.

    Returns the repo root Path, or None if not found.
    """
    cur = start if start.is_dir() else start.parent
    for _ in range(20):  # safety bound
        if (cur / ".codeknowledge").is_dir():
            return cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent
    return None


def _resolve_root(repo_root_arg: str | None, target: Path) -> Path:
    """Resolve the repository root, preferring explicit --repo-root, then
    .codeknowledge/ detection, then falling back to the target directory."""
    if repo_root_arg:
        return Path(repo_root_arg).resolve()
    found = _find_repo_root(target)
    if found:
        return found
    return target if target.is_dir() else target.parent


def _load_config(repo_root: Path) -> Config | None:
    """Load config if .codeknowledge/ exists under repo_root."""
    ck = repo_root / ".codeknowledge"
    if ck.is_dir():
        return Config.load(ck)
    return None


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
@click.argument("path", type=click.Path(exists=True), required=False, default=None)
@click.option("--repo-root", type=click.Path(exists=True), default=None,
              help="Repository root.")
@click.option("--output-dir", type=click.Path(), default=None,
              help="Output directory. Defaults to .codeknowledge/descriptions/ under repo root.")
@click.option("--articles-dir", type=click.Path(exists=True), default=None,
              help="Directory with synthesis articles (architecture-overview.md). "
                   "Defaults to .codeknowledge/articles/ under repo root.")
@click.option("--model", default=None, help="Model tier: haiku, sonnet, opus, or a model name.")
@click.option("--dry-run", is_flag=True, help="Print the prompt without calling the LLM.")
@click.option("--force", is_flag=True, help="Regenerate descriptions even if source is unchanged.")
@click.option("--file-filter", default=None, help="Only process files matching this substring.")
@click.pass_context
def describe(
    ctx: click.Context,
    path: str | None,
    repo_root: str | None,
    output_dir: str | None,
    articles_dir: str | None,
    model: str | None,
    dry_run: bool,
    force: bool,
    file_filter: str | None,
) -> None:
    """Generate LLM descriptions for source files.

    Extracts structure via tree-sitter, sends each file to an LLM with
    sibling-file context and architecture context, and writes description
    markdown files.
    """
    from .describe import build_prompt, parse_response, render_description_markdown, count_elements, find_missing_elements, build_continuation_prompt
    from .graph import CallGraph
    from .llm import describe_file

    target = Path(path).resolve() if path else Path.cwd().resolve()
    root = _resolve_root(repo_root, target)
    cfg = _load_config(root)

    # Resolve source targets from config if no path given
    if path is None and cfg and cfg.source_dirs:
        targets = cfg.resolved_source_dirs()
    else:
        targets = [target]

    model_tier = model or (cfg.model if cfg else "sonnet")
    exclude = cfg.exclude if cfg else None
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

    # Load call graph for caller context
    graph_dir = root / ".codeknowledge" / "graph"
    call_graph: CallGraph | None = None
    if graph_dir.is_dir():
        call_graph = CallGraph.load(graph_dir)
        click.echo(f"Loaded call graph: {len(call_graph.files)} files, "
                   f"{sum(len(v) for v in call_graph._callers.values())} caller edges")
    else:
        click.echo("No call graph found. Run 'graph' first for caller context in descriptions.")

    all_files: list[tuple[Path, FileStructure]] = []
    for t in targets:
        all_files.extend(_collect_files(t, root, exclude=exclude))

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

    skipped = 0
    for i, (file_path, fs) in enumerate(files):
        rel = fs.path
        out_path = out / (rel + ".md")

        # Skip if source hasn't changed since last description
        current_hash = _file_hash(file_path)
        if not force and out_path.exists():
            existing = out_path.read_text()
            if f"source_hash: {current_hash}" in existing.split("---")[1] if existing.startswith("---") else "":
                skipped += 1
                continue

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

        # Caller context from call graph
        file_callers: dict[str, list[tuple[str, str]]] | None = None
        if call_graph:
            file_callers = call_graph.get_file_callers(rel) or None

        prompt = build_prompt(
            structure=fs,
            source=source,
            neighbor_context=neighbor_ctx if neighbor_ctx else None,
            architecture_context=architecture_context,
            project_name=cfg.project_name if cfg else root.name,
            callers=file_callers,
        )

        if dry_run:
            click.echo(f"--- Prompt ({len(prompt)} chars) ---")
            click.echo(prompt)
            click.echo("--- End prompt ---")
            continue

        # Scale max_tokens to the number of elements (~100 tokens per element)
        n_elements = count_elements(fs)
        tokens = min(max(n_elements * 100, 4096), 16384)

        response_text = describe_file(prompt, model_tier=model_tier, max_tokens=tokens)
        desc = parse_response(response_text, fs)

        # Follow-up for missing elements (up to 2 continuation rounds)
        for round_num in range(2):
            missing = find_missing_elements(fs, desc)
            if not missing:
                break
            click.echo(f"     {len(missing)} elements missing, continuation {round_num + 1}...")
            cont_prompt = build_continuation_prompt(
                structure=fs,
                source=source,
                missing=missing,
                project_name=cfg.project_name if cfg else root.name,
            )
            cont_tokens = min(max(len(missing) * 100, 4096), 16384)
            cont_text = describe_file(cont_prompt, model_tier=model_tier, max_tokens=cont_tokens)
            cont_desc = parse_response(cont_text, fs)
            desc.symbols.extend(cont_desc.symbols)
        md = render_description_markdown(
            desc,
            source_hash=current_hash,
            model=model_tier,
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        try:
            display_path = out_path.relative_to(root)
        except ValueError:
            display_path = out_path
        click.echo(f"  -> {display_path}")
        click.echo(f"     {len(desc.symbols)} symbols described")

    if skipped:
        click.echo(f"\nSkipped {skipped} unchanged files (use --force to regenerate).")


# ---------------------------------------------------------------------------
# init — initialize .codeknowledge directory
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("path", type=click.Path(exists=True), default=".")
@click.option("--project-name", default=None, help="Project name (stored in config).")
@click.option("--source-dir", "source_dirs", multiple=True,
              help="Source directories to index (relative to repo root). Repeatable.")
@click.pass_context
def init(ctx: click.Context, path: str, project_name: str | None, source_dirs: tuple[str, ...]) -> None:
    """Initialize a .codeknowledge directory in a repository."""
    from .config import DEFAULT_EXCLUDE

    root = Path(path).resolve()
    ck_dir = root / ".codeknowledge"

    if ck_dir.exists():
        click.echo(f".codeknowledge already exists at {ck_dir}")
        return

    ck_dir.mkdir()
    (ck_dir / "descriptions").mkdir()
    (ck_dir / "articles").mkdir()

    # Create .gitignore inside .codeknowledge to exclude the binary index
    ck_gitignore = ck_dir / ".gitignore"
    ck_gitignore.write_text("# Binary embedding indexes (regenerate with 'codeknowledge index')\ncodeknowledge.db\ncodeknowledge-code.db\n")
    click.echo("Created .codeknowledge/.gitignore (excludes codeknowledge.db)")

    # Create default config
    cfg = Config(
        project_name=project_name or root.name,
        source_dirs=list(source_dirs),
        exclude=list(DEFAULT_EXCLUDE),
        repo_root=root,
        ck_dir=ck_dir,
    )
    cfg_path = cfg.save()
    click.echo(f"Created {cfg_path.name} (edit to configure source dirs and exclusions)")

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
@click.argument("path", type=click.Path(exists=True), required=False, default=None)
@click.option("--repo-root", type=click.Path(exists=True), default=None,
              help="Repository root for relative paths.")
@click.option("--output-dir", type=click.Path(), default=None,
              help="Output directory for articles. Defaults to .codeknowledge/articles/ under repo root.")
@click.option("--model", default=None, help="Model tier: haiku, sonnet, opus, or a model name.")
@click.option("--project-name", default=None, help="Project name for context.")
@click.option("--dry-run", is_flag=True, help="Print prompts without calling the LLM.")
@click.option("--skip-flows", is_flag=True, help="Skip flow document generation.")
@click.option("--force", is_flag=True, help="Regenerate all articles, ignoring commit cache.")
@click.pass_context
def synthesize(
    ctx: click.Context,
    path: str | None,
    repo_root: str | None,
    output_dir: str | None,
    model: str | None,
    project_name: str | None,
    dry_run: bool,
    skip_flows: bool,
    force: bool,
) -> None:
    """Synthesize architecture articles from source code.

    Extracts structure via tree-sitter, reads raw source, and produces
    an architecture overview and key flow documents.

    For large projects, uses batched mode: generates per-module summaries
    first (one directory at a time), then synthesizes architecture from
    those summaries.
    """
    from .synthesize import (
        build_architecture_prompt,
        build_architecture_from_summaries_prompt,
        build_flow_identification_prompt,
        build_flow_prompt,
        build_flow_prompt_from_skeleton,
        build_full_skeleton,
        build_merge_summaries_prompt,
        build_module_summary_prompt,
        group_by_directory,
        needs_batching,
        parse_article_frontmatter,
        parse_article_sources,
        parse_flows,
        render_article,
        split_module_batches,
    )
    from .llm import describe_file
    from .git import get_head_commit, has_uncommitted_changes, is_git_repo

    target = Path(path).resolve() if path else Path.cwd().resolve()
    root = _resolve_root(repo_root, target)
    cfg = _load_config(root)

    out = Path(output_dir).resolve() if output_dir else root / ".codeknowledge" / "articles"
    out.mkdir(parents=True, exist_ok=True)

    model_tier = model or (cfg.model if cfg else "sonnet")
    proj = project_name or (cfg.project_name if cfg else root.name)
    exclude = cfg.exclude if cfg else None

    # Resolve source targets from config if no path given
    if path is None and cfg and cfg.source_dirs:
        targets = cfg.resolved_source_dirs()
    else:
        targets = [target]

    # Collect all files via tree-sitter extraction + raw source
    extracted: list[tuple[Path, FileStructure]] = []
    for t in targets:
        extracted.extend(_collect_files(t, root, exclude=exclude))
    if not extracted:
        click.echo("No extractable files found.", err=True)
        sys.exit(1)

    files: list[tuple[FileStructure, str]] = []
    for file_path, fs in extracted:
        source = file_path.read_text()
        files.append((fs, source))

    # Build source hash map for provenance
    source_hashes: dict[str, str] = {}
    for file_path, fs in extracted:
        source_hashes[fs.path] = _file_hash(file_path)

    click.echo(f"Collected {len(files)} source files.")

    # ---- Git commit tracking ----
    git_available = is_git_repo(root)
    head_commit: str | None = None
    dirty = False
    source_paths = [fs.path for fs, _ in files]

    if git_available:
        head_commit = get_head_commit(root)
        dirty = has_uncommitted_changes(root, source_paths)
        if head_commit:
            click.echo(f"Git commit: {head_commit}" + (" (dirty)" if dirty else ""))
        if dirty:
            click.echo("Warning: uncommitted changes detected — commit info will not be recorded.")
    else:
        click.echo("Not a git repository — commit tracking disabled.")

    # Commit to record: only if clean
    record_commit = head_commit if (head_commit and not dirty) else None
    skipped_articles = 0

    def _should_regenerate(article_path: Path, relevant_hashes: dict[str, str] | None = None) -> bool:
        """Check if an existing article should be regenerated based on source hashes."""
        nonlocal skipped_articles
        if force:
            return True
        if not article_path.exists():
            return True
        existing = article_path.read_text()
        stored_sources = parse_article_sources(existing)
        if not stored_sources:
            return True  # no hashes stored → must regenerate

        check_against = relevant_hashes if relevant_hashes is not None else source_hashes
        for path, stored_hash in stored_sources.items():
            current = check_against.get(path)
            if current is None:
                return True
            if current != stored_hash:
                return True

        if relevant_hashes is not None:
            for path in relevant_hashes:
                if path not in stored_sources:
                    return True

        skipped_articles += 1
        return False

    batched = needs_batching(files)
    if batched:
        click.echo(f"Large project detected — using batched synthesis.\n")

    # ---- Step 1: Architecture overview ----
    if batched:
        # --- Batched mode: module summaries → architecture ---
        skeleton = build_full_skeleton(files)
        click.echo(f"Project skeleton: {len(skeleton)} chars")

        groups = group_by_directory(files)
        click.echo(f"Modules: {len(groups)} directories\n")

        # Generate module summaries
        module_summaries: dict[str, str] = {}
        summaries_dir = out / "module-summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        any_summary_regenerated = False

        for i, (dir_path, module_files) in enumerate(sorted(groups.items()), 1):
            n_files = len(module_files)
            n_chars = sum(len(s) for _, s in module_files)
            click.echo(f"  [{i}/{len(groups)}] {dir_path} ({n_files} files, {n_chars:,} chars)")

            # Incremental check for module summary
            safe_name = dir_path.replace("/", "_").replace("\\", "_") or "_root"
            summary_path = summaries_dir / f"{safe_name}.md"
            module_source_paths = [fs.path for fs, _ in module_files]
            module_hashes = {p: source_hashes[p] for p in module_source_paths if p in source_hashes}

            if not _should_regenerate(summary_path, relevant_hashes=module_hashes):
                click.echo(f"    Unchanged — reusing cached summary.")
                existing = summary_path.read_text()
                content_start = existing.find("---", 3)
                if content_start >= 0:
                    # Skip frontmatter and "# Module: ..." heading
                    body = existing[content_start + 3:].strip()
                    heading_end = body.find("\n")
                    if heading_end >= 0 and body.startswith("# "):
                        body = body[heading_end:].strip()
                    module_summaries[dir_path] = body
                else:
                    module_summaries[dir_path] = existing
                continue

            batches = split_module_batches(module_files)

            if len(batches) == 1:
                # Single batch — normal path
                prompt = build_module_summary_prompt(
                    module_path=dir_path,
                    module_files=module_files,
                    project_skeleton=skeleton,
                    project_name=proj,
                )

                if dry_run:
                    click.echo(f"    Prompt: {len(prompt)} chars")
                    continue

                summary = describe_file(prompt, model_tier=model_tier, max_tokens=4096)
            else:
                # Multiple sub-batches — summarize each, then merge
                click.echo(f"    Split into {len(batches)} sub-batches")
                partial_summaries: list[str] = []

                for j, batch in enumerate(batches, 1):
                    batch_chars = sum(len(s) for _, s in batch)
                    batch_files = len(batch)
                    click.echo(f"    Batch {j}/{len(batches)}: {batch_files} files, {batch_chars:,} chars")

                    prompt = build_module_summary_prompt(
                        module_path=dir_path,
                        module_files=batch,
                        project_skeleton=skeleton,
                        project_name=proj,
                    )

                    if dry_run:
                        click.echo(f"      Prompt: {len(prompt)} chars")
                        continue

                    partial = describe_file(prompt, model_tier=model_tier, max_tokens=4096)
                    partial_summaries.append(partial)

                if dry_run:
                    continue

                # Merge partial summaries
                click.echo(f"    Merging {len(partial_summaries)} partial summaries...")
                merge_prompt = build_merge_summaries_prompt(
                    module_path=dir_path,
                    partial_summaries=partial_summaries,
                    project_name=proj,
                )
                summary = describe_file(merge_prompt, model_tier=model_tier, max_tokens=4096)

            module_summaries[dir_path] = summary
            any_summary_regenerated = True

            # Save module summary
            safe_name = dir_path.replace("/", "_").replace("\\", "_") or "_root"
            summary_path = summaries_dir / f"{safe_name}.md"
            article = render_article(
                title=f"Module: {dir_path}",
                content=summary,
                article_type="module-summary",
                sources={fs.path: source_hashes[fs.path] for fs, _ in module_files},
                model=model_tier,
                commit=record_commit,
                commit_dirty=dirty,
            )
            summary_path.write_text(article)
            click.echo(f"    -> {summary_path.relative_to(out)}")

        if dry_run:
            return

        click.echo(f"\n--- Architecture overview (from {len(module_summaries)} module summaries) ---")

        # In batched mode, skip architecture if no summaries were regenerated
        arch_path = out / "architecture-overview.md"
        if not any_summary_regenerated and arch_path.exists() and not force:
            click.echo("  No module summaries changed — checking architecture...")
            if not _should_regenerate(arch_path, relevant_hashes=source_hashes):
                click.echo("  Architecture unchanged — skipping.")
                # Load existing architecture for flow generation
                existing = arch_path.read_text()
                content_start = existing.find("---", 3)
                if content_start >= 0:
                    body = existing[content_start + 3:].strip()
                    heading_end = body.find("\n")
                    if heading_end >= 0 and body.startswith("# "):
                        body = body[heading_end:].strip()
                    architecture = body
                else:
                    architecture = existing
                arch_regenerated = False
            else:
                arch_prompt = build_architecture_from_summaries_prompt(
                    project_skeleton=skeleton,
                    module_summaries=module_summaries,
                    project_name=proj,
                )
                arch_regenerated = True
        else:
            arch_prompt = build_architecture_from_summaries_prompt(
                project_skeleton=skeleton,
                module_summaries=module_summaries,
                project_name=proj,
            )
            arch_regenerated = True
    else:
        # --- Single-pass mode ---
        click.echo("\n--- Architecture overview ---")
        arch_path = out / "architecture-overview.md"
        if not _should_regenerate(arch_path, relevant_hashes=source_hashes):
            click.echo("  Architecture unchanged — skipping.")
            existing = arch_path.read_text()
            content_start = existing.find("---", 3)
            if content_start >= 0:
                body = existing[content_start + 3:].strip()
                heading_end = body.find("\n")
                if heading_end >= 0 and body.startswith("# "):
                    body = body[heading_end:].strip()
                architecture = body
            else:
                architecture = existing
            arch_regenerated = False
        else:
            arch_prompt = build_architecture_prompt(files=files, project_name=proj)
            arch_regenerated = True

    if dry_run:
        if arch_regenerated:
            click.echo(f"  Prompt: {len(arch_prompt)} chars")
            click.echo(arch_prompt[:2000])
            click.echo("...")
        return

    if arch_regenerated:
        architecture = describe_file(arch_prompt, model_tier=model_tier, max_tokens=8192)

        article_path = out / "architecture-overview.md"
        article = render_article(
            title="Architecture Overview",
            content=architecture,
            article_type="architecture",
            sources=source_hashes,
            model=model_tier,
            commit=record_commit,
            commit_dirty=dirty,
        )
        article_path.write_text(article)
        click.echo(f"  -> {article_path.name}")

    # ---- Step 2: Key flow documents ----
    if not skip_flows:
        # If architecture wasn't regenerated, flows are likely still valid
        if not arch_regenerated:
            click.echo("\n--- Checking flow documents ---")
        else:
            click.echo("\n--- Identifying key flows ---")

        flow_id_prompt = build_flow_identification_prompt(
            architecture=architecture,
            project_name=proj,
        )
        flow_response = describe_file(flow_id_prompt, model_tier=model_tier, max_tokens=2048)
        flows = parse_flows(flow_response)

        click.echo(f"  Identified {len(flows)} flows.")
        for name, desc in flows:
            click.echo(f"    - {name}: {desc}")

        for name, description in flows:
            safe_flow_name = name.lower().replace(" ", "-")
            flow_path = out / f"flow-{safe_flow_name}.md"
            click.echo(f"\n  Flow: {name}")

            if not _should_regenerate(flow_path, relevant_hashes=source_hashes):
                click.echo(f"  Unchanged — skipping.")
                continue

            if batched:
                prompt = build_flow_prompt_from_skeleton(
                    flow_name=name,
                    flow_description=description,
                    project_skeleton=skeleton,
                    module_summaries=module_summaries,
                    architecture=architecture,
                    project_name=proj,
                )
            else:
                prompt = build_flow_prompt(
                    flow_name=name,
                    flow_description=description,
                    files=files,
                    architecture=architecture,
                    project_name=proj,
                )

            response = describe_file(prompt, model_tier=model_tier, max_tokens=8192)

            article = render_article(
                title=f"Flow: {name}",
                content=response,
                article_type="flow",
                sources=source_hashes,
                model=model_tier,
                commit=record_commit,
                commit_dirty=dirty,
            )
            flow_path.write_text(article)
            click.echo(f"  -> {flow_path.name}")

    if skipped_articles:
        click.echo(f"\nSkipped {skipped_articles} unchanged articles (use --force to regenerate).")
    click.echo("\nSynthesis complete.")


# ---------------------------------------------------------------------------
# graph — static call-graph extraction
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("path", type=click.Path(exists=True), required=False, default=None)
@click.option("--repo-root", type=click.Path(exists=True), default=None,
              help="Repository root for relative paths.")
@click.option("--output-dir", type=click.Path(), default=None,
              help="Output directory. Defaults to .codeknowledge/graph/ under repo root.")
@click.pass_context
def graph(
    ctx: click.Context,
    path: str | None,
    repo_root: str | None,
    output_dir: str | None,
) -> None:
    """Extract static call graphs from source files.

    Produces per-file YAML files listing imports, functions, and the
    calls each function makes, with best-effort resolution.
    No LLM calls — purely tree-sitter based.
    """
    from .graph import build_graph

    target = Path(path).resolve() if path else Path.cwd().resolve()
    root = _resolve_root(repo_root, target)
    cfg = _load_config(root)

    exclude = cfg.exclude if cfg else None

    # Resolve source targets
    if path is None and cfg and cfg.source_dirs:
        targets = cfg.resolved_source_dirs()
    else:
        targets = [target]

    out = Path(output_dir).resolve() if output_dir else root / ".codeknowledge" / "graph"
    out.mkdir(parents=True, exist_ok=True)

    # Collect files
    extracted: list[tuple[Path, str]] = []
    for t in targets:
        for file_path, fs in _collect_files(t, root, exclude=exclude):
            extracted.append((file_path, fs.path))

    if not extracted:
        click.echo("No extractable files found.", err=True)
        sys.exit(1)

    click.echo(f"Extracting call graphs from {len(extracted)} files...")
    count = build_graph(extracted, out)
    click.echo(f"Graph extracted: {count} files -> {out}")


# ---------------------------------------------------------------------------
# index — build embedding index from articles/descriptions
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--articles-dir", type=click.Path(exists=True), default=None,
              help="Directory with article markdown files.")
@click.option("--descriptions-dir", type=click.Path(exists=True), default=None,
              help="Directory with description markdown files.")
@click.option("--db", "db_path", type=click.Path(), default=None,
              help="Output database path. Defaults to codeknowledge.db next to articles.")
@click.option("--model", "embed_model", default=None,
              help="Embedding model name. Default: nomic-ai/nomic-embed-text-v1.5.")
@click.option("--repo-root", type=click.Path(exists=True), default=None,
              help="Repository root (looks for .codeknowledge/ layout).")
@click.pass_context
def index(
    ctx: click.Context,
    articles_dir: str | None,
    descriptions_dir: str | None,
    db_path: str | None,
    embed_model: str | None,
    repo_root: str | None,
) -> None:
    """Build the embedding search index from articles and descriptions.

    If --repo-root is given (or run from a repo with .codeknowledge/),
    automatically finds articles/ and descriptions/ directories.
    Otherwise, specify --articles-dir and/or --descriptions-dir explicitly.
    """
    from .index import build_index

    # Resolve directories
    a_dir: Path | None = Path(articles_dir).resolve() if articles_dir else None
    d_dir: Path | None = Path(descriptions_dir).resolve() if descriptions_dir else None
    db: Path | None = Path(db_path).resolve() if db_path else None

    # Auto-detect from repo root
    cfg = None
    if a_dir is None and d_dir is None:
        root = _resolve_root(repo_root, Path.cwd())
        cfg = _load_config(root)
        ck = root / ".codeknowledge"
        if ck.is_dir():
            a_candidate = ck / "articles"
            d_candidate = ck / "descriptions"
            if a_candidate.is_dir():
                a_dir = a_candidate
            if d_candidate.is_dir():
                d_dir = d_candidate
            if db is None:
                db = ck / "codeknowledge.db"

    if a_dir is None and d_dir is None:
        click.echo(
            "No documents found. Specify --articles-dir / --descriptions-dir, "
            "or run from a repo with .codeknowledge/.",
            err=True,
        )
        sys.exit(1)

    # Build embedding config: CLI --model overrides config.yaml
    from .config import EmbeddingConfig
    if cfg:
        emb_cfg = cfg.embedding
    else:
        emb_cfg = EmbeddingConfig()
    if embed_model:
        emb_cfg = EmbeddingConfig(
            backend=emb_cfg.backend,
            model_name=embed_model,
            dimensions=emb_cfg.dimensions,
            api_url=emb_cfg.api_url,
            batch_size=emb_cfg.batch_size,
            max_concurrent=emb_cfg.max_concurrent,
        )

    # Count documents
    n_articles = len(list(a_dir.rglob("*.md"))) if a_dir else 0
    n_descs = len(list(d_dir.rglob("*.md"))) if d_dir else 0
    click.echo(f"Indexing: {n_articles} articles, {n_descs} descriptions")
    click.echo(f"Embedding model: {emb_cfg.model_name} ({emb_cfg.backend})")

    result_path = build_index(
        articles_dir=a_dir,
        descriptions_dir=d_dir,
        db_path=db,
        embedding_config=emb_cfg,
    )

    click.echo(f"\nIndex built: {result_path}")

    # Print stats
    import sqlite3
    conn = sqlite3.connect(str(result_path))
    n_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    dim = conn.execute("SELECT value FROM meta WHERE key='embedding_dim'").fetchone()
    conn.close()
    click.echo(f"  {n_docs} documents, {n_chunks} chunks, {dim[0] if dim else '?'}d embeddings")

    # Build code index if configured
    if cfg and cfg.code_embedding:
        from .index import build_code_index

        click.echo(f"\nCode embedding model: {cfg.code_embedding.model_name} ({cfg.code_embedding.backend})")
        source_targets = cfg.resolved_source_dirs() if cfg.source_dirs else [root]
        extracted = []
        for t in source_targets:
            extracted.extend(_collect_files(t, root, exclude=cfg.exclude))

        if extracted:
            file_structures = [(fs, fp.read_text()) for fp, fs in extracted]
            code_db = ck / "codeknowledge-code.db"
            code_result = build_code_index(
                file_structures=file_structures,
                db_path=code_db,
                embedding_config=cfg.code_embedding,
            )
            conn = sqlite3.connect(str(code_result))
            n_code_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            code_dim = conn.execute("SELECT value FROM meta WHERE key='embedding_dim'").fetchone()
            conn.close()
            click.echo(f"Code index built: {code_result}")
            click.echo(f"  {n_code_chunks} chunks, {code_dim[0] if code_dim else '?'}d embeddings")


# ---------------------------------------------------------------------------
# search — semantic search over the index
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("query")
@click.option("--db", "db_path", type=click.Path(exists=True), default=None,
              help="Index database path.")
@click.option("--repo-root", type=click.Path(exists=True), default=None,
              help="Repository root (looks for .codeknowledge/codeknowledge.db).")
@click.option("--top-k", default=5, type=int, help="Number of results to return.")
@click.option("--source", "source", type=click.Choice(["all", "docs", "code"]),
              default="all", help="Which index to search (default: all).")
@click.option("--verbose", "-v", "show_content", is_flag=True,
              help="Show full chunk content in results.")
@click.pass_context
def search(
    ctx: click.Context,
    query: str,
    db_path: str | None,
    repo_root: str | None,
    top_k: int,
    source: str,
    show_content: bool,
) -> None:
    """Search the embedding index with a natural language query."""
    from .index import search_index

    # Resolve DB path
    db: Path | None = Path(db_path).resolve() if db_path else None
    cfg = None

    if db is None:
        root = _resolve_root(repo_root, Path.cwd())
        cfg = _load_config(root)
        candidate = root / ".codeknowledge" / "codeknowledge.db"
        if candidate.exists():
            db = candidate

    if db is None or not db.exists():
        click.echo(
            "Index not found. Run 'codeknowledge index' first, "
            "or specify --db path.",
            err=True,
        )
        sys.exit(1)

    emb_cfg = cfg.embedding if cfg else None

    # Check for code index
    code_db: Path | None = None
    code_emb_cfg = None
    if cfg and cfg.code_embedding and db:
        candidate_code = db.parent / "codeknowledge-code.db"
        if candidate_code.exists():
            code_db = candidate_code
            code_emb_cfg = cfg.code_embedding

    results = search_index(
        query=query, db_path=db, top_k=top_k, embedding_config=emb_cfg,
        code_db_path=code_db, code_embedding_config=code_emb_cfg,
        source=source,
    )

    if not results:
        click.echo("No results found.")
        return

    for i, r in enumerate(results, 1):
        score = r["score"]
        doc = r["document"]
        heading = r["heading"]
        doc_type = r["doc_type"]

        # Header line
        click.echo(f"\n{'─' * 60}")
        click.echo(f"  [{i}] {score:.4f}  {doc}")
        if heading:
            click.echo(f"      § {heading}")
        click.echo(f"      type: {doc_type}")

        if show_content:
            click.echo(f"{'─' * 60}")
            # Truncate very long content for display
            content = r["content"]
            if len(content) > 1000:
                content = content[:1000] + "\n... (truncated)"
            click.echo(content)

        elif r["content"]:
            # Show first line as preview
            preview = r["content"].split("\n")[0][:120]
            click.echo(f"      {preview}")


# ---------------------------------------------------------------------------
# update — orchestrated full-pipeline run
# ---------------------------------------------------------------------------

def _box(text: str, width: int = 40) -> str:
    """Draw a Unicode box around text."""
    inner = f"  {text}  "
    pad = width - 2
    if len(inner) < pad:
        inner = inner + " " * (pad - len(inner))
    return (
        f"╭{'─' * pad}╮\n"
        f"│{inner}│\n"
        f"╰{'─' * pad}╯"
    )


def _step_line(step: int, total: int, name: str, status: str, detail: str = "") -> str:
    """Format a step progress line."""
    bar_width = 18
    if status == "done":
        bar = "━" * bar_width
        icon = "✓"
    elif status == "running":
        bar = "━" * (bar_width // 2) + "╸" + " " * (bar_width // 2 - 1)
        icon = "…"
    elif status == "skip":
        bar = "─" * bar_width
        icon = "–"
    else:
        bar = " " * bar_width
        icon = " "
    suffix = f"  {detail}" if detail else ""
    return f"[{step}/{total}] {name:<12s} {bar} {icon}{suffix}"


@cli.command()
@click.argument("path", type=click.Path(exists=True), required=False, default=None)
@click.option("--repo-root", type=click.Path(exists=True), default=None,
              help="Repository root for relative paths.")
@click.option("--model", default=None, help="Model tier: haiku, sonnet, opus, or a model name.")
@click.option("--force", is_flag=True, help="Regenerate all content, ignoring caches.")
@click.option("--skip-flows", is_flag=True, help="Skip flow document generation.")
@click.option("--skip-step", multiple=True, type=click.Choice(
    ["extract", "graph", "describe", "synthesize", "index"]),
    help="Skip a step. Repeatable.")
@click.pass_context
def update(
    ctx: click.Context,
    path: str | None,
    repo_root: str | None,
    model: str | None,
    force: bool,
    skip_flows: bool,
    skip_step: tuple[str, ...],
) -> None:
    """Run the full knowledge-building pipeline.

    Sequentially runs: extract → graph → synthesize → describe → index.
    Each step reuses incremental caching where possible.
    """
    import sqlite3
    import time

    from .describe import (
        build_prompt, parse_response, render_description_markdown,
        count_elements, find_missing_elements, build_continuation_prompt,
    )
    from .graph import build_graph, CallGraph
    from .index import build_index
    from .llm import describe_file
    from .synthesize import (
        build_architecture_prompt,
        build_architecture_from_summaries_prompt,
        build_flow_identification_prompt,
        build_flow_prompt,
        build_flow_prompt_from_skeleton,
        build_full_skeleton,
        build_merge_summaries_prompt,
        build_module_summary_prompt,
        group_by_directory,
        needs_batching,
        parse_article_frontmatter,
        parse_article_sources,
        parse_flows,
        render_article,
        split_module_batches,
    )
    from .git import get_head_commit, has_uncommitted_changes, is_git_repo
    from .config import EmbeddingConfig

    t_start = time.monotonic()

    # ---- Setup ----
    target = Path(path).resolve() if path else Path.cwd().resolve()
    root = _resolve_root(repo_root, target)
    cfg = _load_config(root)
    ck_dir = root / ".codeknowledge"

    if not ck_dir.is_dir():
        click.echo("No .codeknowledge/ found. Run 'codeknowledge init' first.", err=True)
        sys.exit(1)

    model_tier = model or (cfg.model if cfg else "sonnet")
    proj = cfg.project_name if cfg else root.name
    exclude = cfg.exclude if cfg else None
    skip = set(skip_step)

    if path is None and cfg and cfg.source_dirs:
        targets = cfg.resolved_source_dirs()
    else:
        targets = [target]

    steps = ["Extract", "Graph", "Synthesize", "Describe", "Index"]
    n_steps = len(steps)

    click.echo(_box(f"CodeKnowledge · {proj}"))
    click.echo()

    # Git info
    git_available = is_git_repo(root)
    head_commit: str | None = None
    dirty = False
    if git_available:
        head_commit = get_head_commit(root)
        dirty = has_uncommitted_changes(root)
        if head_commit:
            tag = " (dirty)" if dirty else ""
            click.echo(f"  commit: {head_commit}{tag}")
    record_commit = head_commit if (head_commit and not dirty) else None
    click.echo()

    # ---- Step 1: Extract ----
    step = 1
    if "extract" in skip:
        click.echo(_step_line(step, n_steps, "Extract", "skip", "skipped"))
    else:
        click.echo(_step_line(step, n_steps, "Extract", "running"), nl=False)
        t0 = time.monotonic()

        extracted: list[tuple[Path, FileStructure]] = []
        for t in targets:
            extracted.extend(_collect_files(t, root, exclude=exclude))

        if not extracted:
            click.echo("\r" + _step_line(step, n_steps, "Extract", "done", "0 files"))
            click.echo("No extractable files found.", err=True)
            sys.exit(1)

        files: list[tuple[FileStructure, str]] = []
        for file_path, fs in extracted:
            source = file_path.read_text()
            files.append((fs, source))

        dt = time.monotonic() - t0
        click.echo("\r" + _step_line(step, n_steps, "Extract", "done",
                                      f"{len(files)} files ({dt:.1f}s)"))

    # ---- Step 2: Graph ----
    step = 2
    graph_dir = ck_dir / "graph"
    if "graph" in skip:
        click.echo(_step_line(step, n_steps, "Graph", "skip", "skipped"))
    else:
        click.echo(_step_line(step, n_steps, "Graph", "running"), nl=False)
        t0 = time.monotonic()

        graph_dir.mkdir(parents=True, exist_ok=True)
        graph_files = [(fp, fs.path) for fp, fs in extracted]
        n_graph = build_graph(graph_files, graph_dir)

        # Count edges for display
        call_graph = CallGraph.load(graph_dir)
        n_edges = sum(len(v) for v in call_graph._callers.values())

        dt = time.monotonic() - t0
        click.echo("\r" + _step_line(step, n_steps, "Graph", "done",
                                      f"{n_graph} files, {n_edges} edges ({dt:.1f}s)"))

    # ---- Step 3: Synthesize ----
    step = 3
    articles_dir = ck_dir / "articles"
    articles_dir.mkdir(parents=True, exist_ok=True)
    if "synthesize" in skip:
        click.echo(_step_line(step, n_steps, "Synthesize", "skip", "skipped"))
    else:
        click.echo(_step_line(step, n_steps, "Synthesize", "running"), nl=False)
        t0 = time.monotonic()

        source_hashes: dict[str, str] = {}
        for file_path, fs in extracted:
            source_hashes[fs.path] = _file_hash(file_path)

        source_paths = [fs.path for fs, _ in files]
        synth_generated = 0
        synth_cached = 0

        def _synth_should_regen(article_path: Path, relevant_hashes: dict[str, str] | None = None) -> bool:
            """Check if a synthesis article needs regeneration.

            Compares stored source hashes against current hashes.  If all
            relevant source files are unchanged, skip.
            """
            nonlocal synth_cached
            if force:
                return True
            if not article_path.exists():
                return True

            existing = article_path.read_text()
            stored_sources = parse_article_sources(existing)
            if not stored_sources:
                return True  # no hashes stored → must regenerate

            # Compare against the relevant subset of current hashes
            check_against = relevant_hashes if relevant_hashes is not None else source_hashes
            for path, stored_hash in stored_sources.items():
                current = check_against.get(path)
                if current is None:
                    # File was removed or renamed
                    return True
                if current != stored_hash:
                    return True

            # Check if new files were added that this article should cover
            if relevant_hashes is not None:
                for path in relevant_hashes:
                    if path not in stored_sources:
                        return True

            synth_cached += 1
            return False

        batched = needs_batching(files)

        if batched:
            skeleton = build_full_skeleton(files)
            groups = group_by_directory(files)
            module_summaries: dict[str, str] = {}
            summaries_dir = articles_dir / "module-summaries"
            summaries_dir.mkdir(parents=True, exist_ok=True)
            any_summary_regenerated = False

            for dir_path, module_files in sorted(groups.items()):
                safe_name = dir_path.replace("/", "_").replace("\\", "_") or "_root"
                summary_path = summaries_dir / f"{safe_name}.md"
                module_source_paths = [fs.path for fs, _ in module_files]
                module_hashes = {p: source_hashes[p] for p in module_source_paths if p in source_hashes}

                if not _synth_should_regen(summary_path, relevant_hashes=module_hashes):
                    existing = summary_path.read_text()
                    content_start = existing.find("---", 3)
                    if content_start >= 0:
                        body = existing[content_start + 3:].strip()
                        heading_end = body.find("\n")
                        if heading_end >= 0 and body.startswith("# "):
                            body = body[heading_end:].strip()
                        module_summaries[dir_path] = body
                    else:
                        module_summaries[dir_path] = existing
                    continue

                any_summary_regenerated = True
                batches = split_module_batches(module_files)

                if len(batches) == 1:
                    prompt = build_module_summary_prompt(
                        module_path=dir_path,
                        module_files=module_files,
                        project_skeleton=skeleton,
                        project_name=proj,
                    )
                    summary = describe_file(prompt, model_tier=model_tier, max_tokens=4096)
                else:
                    partial_summaries: list[str] = []
                    for batch in batches:
                        prompt = build_module_summary_prompt(
                            module_path=dir_path,
                            module_files=batch,
                            project_skeleton=skeleton,
                            project_name=proj,
                        )
                        partial_summaries.append(
                            describe_file(prompt, model_tier=model_tier, max_tokens=4096)
                        )
                    merge_prompt = build_merge_summaries_prompt(
                        module_path=dir_path,
                        partial_summaries=partial_summaries,
                        project_name=proj,
                    )
                    summary = describe_file(merge_prompt, model_tier=model_tier, max_tokens=4096)

                module_summaries[dir_path] = summary
                synth_generated += 1
                article = render_article(
                    title=f"Module: {dir_path}", content=summary,
                    article_type="module-summary",
                    sources={fs.path: source_hashes[fs.path] for fs, _ in module_files},
                    model=model_tier, commit=record_commit, commit_dirty=dirty,
                )
                summary_path.write_text(article)

            # Architecture
            arch_path = articles_dir / "architecture-overview.md"
            arch_regenerated = False
            if not any_summary_regenerated and not _synth_should_regen(arch_path, relevant_hashes=source_hashes):
                existing = arch_path.read_text()
                content_start = existing.find("---", 3)
                if content_start >= 0:
                    body = existing[content_start + 3:].strip()
                    heading_end = body.find("\n")
                    if heading_end >= 0 and body.startswith("# "):
                        body = body[heading_end:].strip()
                    architecture = body
                else:
                    architecture = existing
            else:
                arch_prompt = build_architecture_from_summaries_prompt(
                    project_skeleton=skeleton,
                    module_summaries=module_summaries,
                    project_name=proj,
                )
                architecture = describe_file(arch_prompt, model_tier=model_tier, max_tokens=8192)
                arch_regenerated = True
                synth_generated += 1
                article = render_article(
                    title="Architecture Overview", content=architecture,
                    article_type="architecture", sources=source_hashes,
                    model=model_tier, commit=record_commit, commit_dirty=dirty,
                )
                arch_path.write_text(article)
        else:
            # Single-pass
            arch_path = articles_dir / "architecture-overview.md"
            arch_regenerated = False
            if not _synth_should_regen(arch_path, relevant_hashes=source_hashes):
                existing = arch_path.read_text()
                content_start = existing.find("---", 3)
                if content_start >= 0:
                    body = existing[content_start + 3:].strip()
                    heading_end = body.find("\n")
                    if heading_end >= 0 and body.startswith("# "):
                        body = body[heading_end:].strip()
                    architecture = body
                else:
                    architecture = existing
            else:
                arch_prompt = build_architecture_prompt(files=files, project_name=proj)
                architecture = describe_file(arch_prompt, model_tier=model_tier, max_tokens=8192)
                arch_regenerated = True
                synth_generated += 1
                article = render_article(
                    title="Architecture Overview", content=architecture,
                    article_type="architecture", sources=source_hashes,
                    model=model_tier, commit=record_commit, commit_dirty=dirty,
                )
                arch_path.write_text(article)

        # Flows
        if not skip_flows:
            # Check if any existing flow articles need regeneration before
            # calling the LLM to identify flows (which is expensive).
            existing_flows = sorted(articles_dir.glob("flow-*.md"))
            any_flow_stale = arch_regenerated  # new arch → always re-identify

            if not any_flow_stale and not existing_flows:
                any_flow_stale = True  # no flows yet → need to identify

            if not any_flow_stale:
                for fp in existing_flows:
                    if _synth_should_regen(fp, relevant_hashes=source_hashes):
                        any_flow_stale = True
                        break

            if any_flow_stale:
                flow_id_prompt = build_flow_identification_prompt(
                    architecture=architecture, project_name=proj,
                )
                flow_response = describe_file(flow_id_prompt, model_tier=model_tier, max_tokens=2048)
                flow_list = parse_flows(flow_response)

                for name, description in flow_list:
                    safe_flow_name = name.lower().replace(" ", "-")
                    flow_path = articles_dir / f"flow-{safe_flow_name}.md"

                    if not _synth_should_regen(flow_path, relevant_hashes=source_hashes):
                        continue

                    if batched:
                        prompt = build_flow_prompt_from_skeleton(
                            flow_name=name, flow_description=description,
                            project_skeleton=skeleton, module_summaries=module_summaries,
                            architecture=architecture, project_name=proj,
                        )
                    else:
                        prompt = build_flow_prompt(
                            flow_name=name, flow_description=description,
                            files=files, architecture=architecture, project_name=proj,
                        )

                    response = describe_file(prompt, model_tier=model_tier, max_tokens=8192)
                    synth_generated += 1
                    article = render_article(
                        title=f"Flow: {name}", content=response,
                        article_type="flow", sources=source_hashes,
                        model=model_tier, commit=record_commit, commit_dirty=dirty,
                    )
                    flow_path.write_text(article)

        dt = time.monotonic() - t0
        parts = []
        if synth_generated:
            parts.append(f"{synth_generated} generated")
        if synth_cached:
            parts.append(f"{synth_cached} cached")
        if not parts:
            parts.append("up to date")
        click.echo("\r" + _step_line(step, n_steps, "Synthesize", "done",
                                      f"{', '.join(parts)} ({dt:.1f}s)"))

    # ---- Step 4: Describe ----
    step = 4
    desc_dir = ck_dir / "descriptions"
    desc_dir.mkdir(parents=True, exist_ok=True)
    if "describe" in skip:
        click.echo(_step_line(step, n_steps, "Describe", "skip", "skipped"))
    else:
        click.echo(_step_line(step, n_steps, "Describe", "running"), nl=False)
        t0 = time.monotonic()

        # Load architecture context (now available from synthesize step)
        arch_dir = ck_dir / "articles"
        architecture_context: str | None = None
        arch_file = arch_dir / "architecture-overview.md"
        if arch_file.is_file():
            raw = arch_file.read_text()
            if raw.startswith("---"):
                end = raw.find("---", 3)
                if end != -1:
                    architecture_context = raw[end + 3:].strip()

        # Load call graph
        if not ("graph" in skip) and graph_dir.is_dir():
            if "call_graph" not in dir():
                call_graph = CallGraph.load(graph_dir)
        elif graph_dir.is_dir():
            call_graph = CallGraph.load(graph_dir)
        else:
            call_graph = None

        # Group files by directory for neighbor context
        dir_groups: dict[Path, list[tuple[Path, FileStructure]]] = {}
        for file_path, fs in extracted:
            dir_groups.setdefault(file_path.parent, []).append((file_path, fs))

        desc_generated = 0
        desc_cached = 0

        for file_path, fs in extracted:
            rel = fs.path
            out_path = desc_dir / (rel + ".md")
            current_hash = _file_hash(file_path)

            # Skip if unchanged
            if not force and out_path.exists():
                existing = out_path.read_text()
                if existing.startswith("---"):
                    fm_section = existing.split("---")[1] if "---" in existing[3:] else ""
                    if f"source_hash: {current_hash}" in fm_section:
                        desc_cached += 1
                        continue

            source = file_path.read_text()

            # Neighbor context
            siblings = dir_groups.get(file_path.parent, [])
            neighbor_ctx: dict[str, str] = {}
            neighbor_chars = 0
            for sib_path, sib_fs in siblings:
                if sib_path == file_path:
                    continue
                sib_source = sib_path.read_text()
                if neighbor_chars + len(sib_source) > 50_000:
                    continue
                neighbor_ctx[sib_fs.path] = sib_source
                neighbor_chars += len(sib_source)

            file_callers = None
            if call_graph:
                file_callers = call_graph.get_file_callers(rel) or None

            prompt = build_prompt(
                structure=fs,
                source=source,
                neighbor_context=neighbor_ctx if neighbor_ctx else None,
                architecture_context=architecture_context,
                project_name=proj,
                callers=file_callers,
            )

            n_elements = count_elements(fs)
            tokens = min(max(n_elements * 100, 4096), 16384)
            response_text = describe_file(prompt, model_tier=model_tier, max_tokens=tokens)
            desc = parse_response(response_text, fs)

            # Continuation for missing elements
            for _ in range(2):
                missing = find_missing_elements(fs, desc)
                if not missing:
                    break
                cont_prompt = build_continuation_prompt(
                    structure=fs, source=source,
                    missing=missing, project_name=proj,
                )
                cont_tokens = min(max(len(missing) * 100, 4096), 16384)
                cont_text = describe_file(cont_prompt, model_tier=model_tier, max_tokens=cont_tokens)
                cont_desc = parse_response(cont_text, fs)
                desc.symbols.extend(cont_desc.symbols)

            md = render_description_markdown(desc, source_hash=current_hash, model=model_tier)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(md)
            desc_generated += 1

        dt = time.monotonic() - t0
        parts = [f"{len(extracted)} files"]
        if desc_cached:
            parts.append(f"{desc_cached} cached")
        if desc_generated:
            parts.append(f"{desc_generated} generated")
        click.echo("\r" + _step_line(step, n_steps, "Describe", "done",
                                      f"{', '.join(parts)} ({dt:.1f}s)"))

    # ---- Step 5: Index ----
    step = 5
    if "index" in skip:
        click.echo(_step_line(step, n_steps, "Index", "skip", "skipped"))
    else:
        click.echo(_step_line(step, n_steps, "Index", "running"), nl=False)
        t0 = time.monotonic()

        emb_cfg = cfg.embedding if cfg else EmbeddingConfig()
        a_dir = ck_dir / "articles" if (ck_dir / "articles").is_dir() else None
        d_dir = ck_dir / "descriptions" if (ck_dir / "descriptions").is_dir() else None
        db = ck_dir / "codeknowledge.db"

        result_path = build_index(
            articles_dir=a_dir,
            descriptions_dir=d_dir,
            db_path=db,
            embedding_config=emb_cfg,
        )

        conn = sqlite3.connect(str(result_path))
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        dim_row = conn.execute("SELECT value FROM meta WHERE key='embedding_dim'").fetchone()
        conn.close()
        dim = dim_row[0] if dim_row else "?"

        detail_parts = [f"{n_chunks} chunks · {dim}d"]

        # Build code index if configured
        if cfg and cfg.code_embedding:
            from .index import build_code_index

            code_db = ck_dir / "codeknowledge-code.db"
            code_result = build_code_index(
                file_structures=files,
                db_path=code_db,
                embedding_config=cfg.code_embedding,
            )
            conn = sqlite3.connect(str(code_result))
            n_code = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            code_dim_row = conn.execute("SELECT value FROM meta WHERE key='embedding_dim'").fetchone()
            conn.close()
            code_dim = code_dim_row[0] if code_dim_row else "?"
            detail_parts.append(f"+ {n_code} code · {code_dim}d")

        dt = time.monotonic() - t0
        click.echo("\r" + _step_line(step, n_steps, "Index", "done",
                                      f"{' '.join(detail_parts)} ({dt:.1f}s)"))

    # ---- Summary ----
    total_time = time.monotonic() - t_start
    minutes = int(total_time // 60)
    seconds = total_time % 60
    if minutes:
        click.echo(f"\nDone in {minutes}m {seconds:.0f}s")
    else:
        click.echo(f"\nDone in {seconds:.1f}s")


if __name__ == "__main__":
    cli()

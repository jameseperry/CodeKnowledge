# CodeKnowledge MCP — Design

## What This Is

An MCP server that provides persistent, structured understanding of codebases. Rather than embedding raw code, it uses LLMs to analyze code and produce natural language descriptions, then indexes those descriptions for semantic search. The result is a system where a model can query against pre-built understanding rather than re-reading and re-reasoning over source code every session.

## Core Bet

Embedding LLM-written descriptions of code is more effective for retrieval than embedding the code itself. Code embedding models (voyage-code, etc.) optimize for code-to-code similarity, but the actual query pattern is natural language → meaning ("what handles authentication?" not "find code structurally similar to this code"). An LLM reading code can reason about what it does — see that a loop over a stride pattern is a transpose, that a generic helper is actually a critical path — and bake that reasoning into a description. A standard text embedding model then indexes the description effectively because the hard work is already done.

## Architecture Overview

### Repository Structure

The CodeKnowledge data lives in a `.codeknowledge/` directory inside the target repository. This directory is its own git repo, gitignored by the parent. The two repos are fully decoupled — no submodules, no pointer synchronization.

```
target-project/                  # parent git repo
  .gitignore                     # contains: .codeknowledge/
  src/
    module_a/
      foo.py
      bar.py
    module_b/
      baz.py
  .codeknowledge/                # its own independent git repo
    .git/
    manifest.json                # provenance: which source commit was indexed
    descriptions/                # Pass 1 output: mirrors source tree
      src/
        module_a/
          foo.py.md              # file summary + symbol descriptions
          bar.py.md
        module_b/
          baz.py.md
    articles/                    # Pass 2 output: curated understanding
      architecture-overview.md
      indexing-pipeline-flow.md
      authentication-system.md
    codeknowledge.db             # embeddings + structural index (regenerable)
```

### Why This Layout

- **Markdown on disk, index in SQLite.** The expensive artifacts (LLM-generated descriptions and articles) are stored as plain markdown files — the most durable, portable, human-readable format. The SQLite database holds what's cheap to regenerate: embeddings, structural metadata, content hashes. Understanding is more durable than the index.
- **Descriptions mirror the source tree.** When `src/module_a/foo.py` changes, regenerate `descriptions/src/module_a/foo.py.md`. The 1:1 correspondence makes the relationship obvious and avoids any naming scheme complexity.
- **Articles are flat/lightly organized.** They cut across the source tree (an architecture overview references many files), so no mirroring.
- **Separate git repo in a gitignored directory.** The source repo stays clean — no documentation churn in its history. The understanding repo has its own history showing how the understanding evolved. The manifest links them.

### Provenance Tracking

Each description file includes frontmatter tracking its source:

```yaml
---
source: src/knowledge_mcp/indexer.py
source_hash: a3f8b2c...
commit: e7d1f4a
indexed_at: 2026-04-13T10:30:00Z
model: claude-3-haiku
---
```

A top-level manifest records the overall index state:

```json
{
  "last_full_index": "e7d1f4a",
  "indexed_at": "2026-04-13T10:30:00Z",
  "repo_root": "/path/to/target-project"
}
```

This enables `codeknowledge status` to diff the current source tree against the manifest and report exactly which descriptions are stale and by how many commits.

## Three Tiers of Understanding

1. **Symbol descriptions** (auto-generated, per-file) — What does this function/class do? Produced by Pass 1. Stored in `descriptions/`. Cheap to refresh when source changes.

2. **File and module summaries** (auto-generated, per-file and per-directory) — What is this module's responsibility? Included in description files (file summary) and as index-level articles (module summaries). Generated in Pass 1 with file context, refined in Pass 2 with module context.

3. **Architecture articles** (synthesized, curated) — How does the system work? What are the key flows? Why was it built this way? Produced by Pass 2. Stored in `articles/`. The highest-value output — captures the understanding a senior engineer carries but that currently evaporates at the end of every session.

## Tool Interface (MCP)

| Tool | Purpose |
|------|---------|
| `browse` | Navigate repo structure — list files and directories |
| `outline` | File's structural elements — functions, classes, with signatures and descriptions |
| `read` | Read a description file, article, or raw source section |
| `search` | Semantic search over description and article embeddings |
| `grep` | Exact text/regex search across descriptions and articles |
| `dependencies` | Import-level dependency queries — what does X depend on? what depends on X? |
| `write_article` | Create or update a curated architecture article |
| `delete_article` | Remove an article |
| `set_metadata` | Annotate files/symbols with additional context without regenerating |
| `index_status` | Health check — indexed files, stale descriptions, coverage stats |

## Key Design Decisions

1. **Embed descriptions, not code.** The fundamental approach. LLMs reason about code meaning; embedding models compress. Do the reasoning first, then embed the result.

2. **Source files are never modified.** All understanding lives in `.codeknowledge/`. The target codebase is read-only input.

3. **Understanding is more durable than the index.** Markdown files survive database corruption. The SQLite DB (embeddings, structural index) is regenerable from the markdown; the markdown requires the full LLM pipeline to regenerate.

4. **Bidirectional indexing — bottom-up then top-down.** Pass 1 produces accurate but context-free descriptions. Pass 2 reads those descriptions to build architectural understanding, then revises the descriptions with that context. This mirrors how a human expert actually understands code.

5. **Tree-sitter for structural extraction.** Language-agnostic, offline, no running server. Each language needs query patterns to extract functions, classes, imports. Start with Python and C++.

6. **Staleness is tracked explicitly.** Each description records the content hash and commit of the source it was generated from. The system knows what's stale rather than silently returning outdated descriptions.

7. **Separate git repos.** Source repo stays clean. Understanding repo has its own history. Manifest links them. No submodule complexity.

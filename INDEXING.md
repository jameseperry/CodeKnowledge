# CodeKnowledge MCP — Indexing Pipeline

## Overview

The indexing pipeline transforms a codebase into searchable understanding through two passes: a bottom-up structural analysis, then a top-down architectural synthesis that revises and enriches the first pass.

## Prerequisites

- **Tree-sitter** with language grammars for supported languages (Python, C++ initially)
- **LLM access** — cheap model for Pass 1 (Haiku-class), capable model for Pass 2 (Sonnet/Opus-class)
- **Embedding model** — standard text embeddings (nomic-embed-text-v1.5 or similar), NOT code-specific
- **SQLite + sqlite-vec** for the vector index

## Pass 1: Bottom-Up Structural Analysis

### Step 1: Tree-Sitter Parse

For each source file:
1. Detect language from file extension
2. Parse with tree-sitter to extract structural elements:
   - **Functions/methods**: name, signature (parameters + return type), decorators, docstring, body, line range
   - **Classes**: name, bases, decorators, docstring, methods (as children), line range
   - **Module-level**: imports, constants, top-level statements
3. Build a scope path for each element (like heading_path in Knowledge MCP):
   - `"MyClass"`, `"MyClass > process_data"`, `"MyClass > process_data > _helper"`
4. Compute content hash (SHA-256) for each file

**Output:** A structural skeleton for each file — list of elements with scope paths, signatures, line ranges, and the raw body text.

### Step 2: File-Level LLM Description

For each file, send to an LLM (Haiku-class):

**Input context:**
- The full file content
- Adjacent files in the same directory (as much as fits in context) — import targets, sibling modules
- The structural skeleton from step 1 (so the model knows what elements to describe)

**Prompt structure:**
```
Here is the file `{rel_path}` from the project `{project_name}`.
Here are other files in the same directory for context: [neighboring file contents or summaries]

Produce:
1. A file-level summary: what this module's responsibility is
2. For each of these structural elements, a description of what it does,
   its key inputs/outputs, and notable implementation details:
   [list of elements from tree-sitter with signatures]
```

**Output format:** A markdown file with YAML frontmatter:

```markdown
---
source: src/knowledge_mcp/indexer.py
source_hash: a3f8b2c...
commit: e7d1f4a
indexed_at: 2026-04-13T10:30:00Z
model: claude-3-haiku
symbols:
  - name: index_all
    scope_path: "index_all"
    kind: function
    line_start: 45
    line_end: 120
  - name: chunk_markdown
    scope_path: "chunk_markdown"
    kind: function
    line_start: 122
    line_end: 200
---

# src/knowledge_mcp/indexer.py

Orchestrates the full document ingestion pipeline — file scanning, change
detection, markdown chunking, and embedding coordination. This is the core
module that transforms raw documentation files into searchable content.

## `index_all(db, settings, kb_path)`

Runs a full index of a knowledge base directory. Walks the filesystem,
detects changed files by SHA-256 hash comparison, chunks new/modified files,
and coordinates bulk embedding across all perspectives. Handles stale file
cleanup for deleted sources.

## `chunk_markdown(text, max_chars)`

Splits markdown text into chunks along heading boundaries while maintaining
a heading breadcrumb stack. Each chunk knows its structural location
(e.g., "Features > WMMA > Matrix Layouts"). Oversized chunks are
sub-split by paragraph.

...
```

### Step 3: Embed Descriptions

For each description file:
1. Split into chunks — one per symbol description, plus the file summary
2. Embed each chunk with the text embedding model (prefixed for asymmetric search)
3. Store embeddings in `codeknowledge.db` with references to the description file and symbol

### Step 4: Extract Import Graph

From the tree-sitter parse:
1. Extract all import statements per file
2. Resolve imports to files within the project where possible
3. Store as directed edges in the database: `file_a imports file_b`
4. This is a static, cheap analysis — no LLM needed

**Pass 1 is parallelizable.** Each file can be parsed, described, and embedded independently. The only shared resource is the database (serialize writes).

### Cost Estimate for Pass 1

For a 10K-line codebase (~100 files):
- ~10K input tokens + ~2K output tokens per file
- ~1.2M tokens total with Haiku at ~$0.30
- Embedding cost negligible for local model

For rocm-libraries scoped to relevant directories: scale linearly, still single-digit dollars.

---

## Pass 2: Top-Down Architectural Synthesis

### Step 5: Module-Level Summaries

For each directory (bottom-up through the directory tree):
1. Gather the file summaries from Pass 1 for all files in the directory
2. Send to a capable LLM (Sonnet-class):

**Input:**
- All file summaries in the directory
- The directory's position in the project structure
- Parent directory summary (if already generated)

**Prompt:**
```
Here are the files in `{dir_path}` and what each one does:
[file summaries from Pass 1]

Describe this module's overall responsibility, how the files work together,
and what its external interface is.
```

**Output:** A module-level article in `articles/`, e.g., `articles/module-src-knowledge-mcp.md`

### Step 6: Architecture and Flow Articles

With all file and module summaries available:

1. **Project overview:** Feed all module summaries → produce a top-level architecture description
2. **Key flows:** Identify entry points (from config, CLI definitions, tool handlers), then trace execution paths through the described code. For each major flow, produce a document explaining the end-to-end path.
3. **Design decisions and patterns:** Identify non-obvious patterns, invariants, architectural choices that aren't apparent from any single file.

These are the highest-value outputs. They capture understanding that cuts across the codebase.

### Step 7: Description Revision

With the architectural context from steps 5-6:
1. Identify descriptions that are now known to be incomplete or misleading
2. Re-generate those specific descriptions with the architectural context included as input
3. Re-embed the revised descriptions

This is selective — not every description gets revised. The top-down pass reveals which bottom-up descriptions were contextually wrong.

### Step 8: Embed Articles

Same as step 3 but for the architecture articles. These get chunked by heading and embedded into the same vector index.

---

## Incremental Updates

When source files change:

1. **Detect changes:** Hash each source file, compare against `source_hash` in description frontmatter
2. **Re-run Pass 1** on changed files only — re-parse, re-describe, re-embed
3. **Flag staleness:** Mark module summaries and architecture articles that reference changed files as potentially stale
4. **Selective Pass 2 re-run:** On demand or when staleness exceeds a threshold, re-run the top-down pass for affected modules

The key cost property: Pass 1 is incremental by file. Pass 2 is incremental by module/article. A single file change in a large codebase triggers one file re-description (cheap) and flags a handful of articles for review (deferred).

---

## CLI Interface

```
codeknowledge init                          # initialize .codeknowledge/ in current repo
codeknowledge index                         # full pipeline: parse → describe → embed → synthesize
codeknowledge index --pass1-only            # just bottom-up: parse → describe → embed
codeknowledge index --file src/foo.py       # re-index a single file (Pass 1 only)
codeknowledge status                        # show stale descriptions, coverage stats
codeknowledge search "query text"           # semantic search (for testing/debugging)
```

The MCP server is separate:
```
codeknowledge-mcp                           # run the MCP server (stdio)
```

---

## Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Structural parsing | tree-sitter | Language-agnostic, offline, no running server needed |
| Pass 1 LLM | Haiku-class | Cheap enough for per-file bulk processing |
| Pass 2 LLM | Sonnet/Opus-class | Needs stronger reasoning for architectural synthesis |
| Embeddings | nomic-embed-text-v1.5 (local) | Standard text model — we're embedding descriptions, not code |
| Vector store | SQLite + sqlite-vec | Proven pattern from Knowledge MCP, single-file, portable |
| Config | Pydantic + JSON | Same pattern as Knowledge MCP |
| Server | FastMCP | Same pattern as Knowledge MCP |

## Language Support

Each language requires tree-sitter query patterns to extract:
- Function/method definitions (name, signature, body, decorators)
- Class definitions (name, bases, body)
- Import statements (what's imported, from where)

**v0.1:** Python, C++
**Future:** TypeScript/JavaScript, Rust, Go, Java

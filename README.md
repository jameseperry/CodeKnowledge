# CodeKnowledge

Build persistent, searchable understanding of a codebase using LLM-generated descriptions, architecture synthesis, and semantic search — served as an MCP server for AI coding assistants.

## How it works

CodeKnowledge processes a codebase in five steps:

1. **Extract** — Parse source files with tree-sitter to extract structural elements (classes, functions, signatures, docstrings)
2. **Graph** — Build a static call graph from the extracted structure
3. **Synthesize** — Generate architecture overviews, module summaries, and key flow documents using an LLM
4. **Describe** — Generate per-file, per-symbol descriptions using an LLM (with architecture context from step 3)
5. **Index** — Embed all generated content into a SQLite vector database for semantic search

The results are served via an MCP server that provides tools for searching the knowledge base, retrieving symbol context, and browsing the architecture.

## Installation

```bash
pip install -e .
```

### Requirements

- Python 3.10+
- An Anthropic API key (or compatible proxy)
- The embedding model is downloaded automatically on first run (~500MB)

## Quick start

### 1. Initialize a project

```bash
cd /path/to/your/project
codeknowledge init --project-name "MyProject" --source-dir src
```

This creates a `.codeknowledge/` directory with a `config.yaml`. Edit the config to adjust source directories, exclusion patterns, and model settings.

### 2. Build the knowledge base

```bash
codeknowledge update
```

This runs the full pipeline (extract → graph → synthesize → describe → index). On subsequent runs, it uses incremental caching:

- **Descriptions** are skipped for files whose source hash hasn't changed
- **Synthesis articles** are skipped when the git diff since last generation isn't architecturally significant (evaluated by a cheap LLM call)
- Use `--force` to regenerate everything

Output:
```
╭──────────────────────────────────────╮
│  CodeKnowledge · MyProject           │
╰──────────────────────────────────────╯

  commit: a1b2c3d

[1/5] Extract      ━━━━━━━━━━━━━━━━━━ ✓  42 files (0.1s)
[2/5] Graph        ━━━━━━━━━━━━━━━━━━ ✓  42 files, 380 edges (0.5s)
[3/5] Synthesize   ━━━━━━━━━━━━━━━━━━ ✓  8 generated (2m 15s)
[4/5] Describe     ━━━━━━━━━━━━━━━━━━ ✓  42 files, 40 cached, 2 generated (45s)
[5/5] Index        ━━━━━━━━━━━━━━━━━━ ✓  650 chunks · 768d (12s)

Done in 3m 14s
```

You can also run individual steps:
```bash
codeknowledge extract src/          # Parse structure only
codeknowledge graph                 # Build call graph
codeknowledge synthesize            # Generate architecture articles
codeknowledge describe              # Generate per-file descriptions
codeknowledge index                 # Build embedding index
```

### 3. Search the knowledge base

```bash
codeknowledge search "how does the caching layer work"
```

## MCP server setup

The MCP server exposes the knowledge base to AI coding assistants (e.g., Claude in VS Code, Cursor).

### Single project

Add to your MCP config (e.g., `.vscode/mcp.json` or `claude_desktop_config.json`):

```json
{
  "servers": {
    "codeknowledge": {
      "command": "codeknowledge-mcp",
      "args": ["--allow", "/path/to/your/project"]
    }
  }
}
```

### Multiple projects

Use `--allow` patterns to whitelist directories. The model calls `open_project` to switch between them:

```json
{
  "servers": {
    "codeknowledge": {
      "command": "codeknowledge-mcp",
      "args": [
        "--allow", "/home/user/dev/project-a",
        "--allow", "/home/user/dev/project-b"
      ]
    }
  }
}
```

You can also use glob patterns:
```bash
codeknowledge-mcp --allow '/home/user/dev/**'
```

### Server options

```
codeknowledge-mcp --help

  --transport {stdio,sse,http}  MCP transport (default: stdio)
  --host HOST                   HTTP host (default: 127.0.0.1)
  --port PORT                   HTTP port (default: 8767)
  --allow PATTERN               Glob pattern for allowed paths (repeatable)
  -v, --verbose                 Enable debug logging
```

### Available tools

Once connected, the MCP server provides:

| Tool | Description |
|------|-------------|
| `open_project` | Load a project's knowledge base (call first) |
| `search_codebase` | Semantic search across all descriptions and articles |
| `get_symbol_context` | Get detailed context for a function, class, or method |
| `get_architecture` | Retrieve the architecture overview document |
| `list_files` | List all files in the knowledge base |

## Configuration

The `config.yaml` in `.codeknowledge/` controls project settings:

```yaml
project_name: MyProject
source_dirs:
  - src
  - lib
exclude:
  - "**/tests/**"
  - "**/__pycache__/**"
  - "**/build/**"
model: sonnet
embedding:
  backend: local
  model_name: nomic-ai/nomic-embed-text-v1.5
  dimensions: 768
```

## LLM configuration

CodeKnowledge uses the Anthropic API. Set these environment variables:

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

To use a proxy or custom endpoint:
```bash
export ANTHROPIC_BASE_URL="https://your-proxy.example.com"
export ANTHROPIC_CUSTOM_HEADERS="Ocp-Apim-Subscription-Key: your-key"
```

To override default model names:
```bash
export ANTHROPIC_DEFAULT_HAIKU_MODEL="claude-haiku-4"
export ANTHROPIC_DEFAULT_SONNET_MODEL="claude-sonnet-4-20250514"
```

## Project structure

```
.codeknowledge/
├── config.yaml              # Project configuration
├── .gitignore               # Excludes codeknowledge.db
├── articles/                # Synthesized architecture documents
│   ├── architecture-overview.md
│   ├── flow-*.md
│   └── module-summaries/
├── descriptions/            # Per-file LLM descriptions
│   └── <mirrored source tree>.md
├── graph/                   # Static call graph YAML
│   └── <mirrored source tree>.yaml
└── codeknowledge.db         # SQLite vector index (not committed)
```

## CLI reference

```
codeknowledge --help

Commands:
  init        Initialize a .codeknowledge directory
  update      Run the full pipeline (extract → graph → synthesize → describe → index)
  extract     Parse structural elements from source files
  graph       Extract static call graphs
  synthesize  Generate architecture articles from source code
  describe    Generate LLM descriptions for source files
  index       Build the embedding search index
  search      Search the index with a natural language query
```

"""Embedding index and semantic search over CodeKnowledge articles.

Chunks markdown documents (articles + descriptions), embeds them with
a sentence-transformer model, and stores everything in a SQLite database
for retrieval via cosine similarity.
"""

from __future__ import annotations

import hashlib
import logging
import re
import sqlite3
import struct
from pathlib import Path

import numpy as np
import yaml

from .config import EmbeddingConfig
from .embeddings import get_embedder

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "nomic-ai/nomic-embed-text-v1.5"
# Target chunk size in chars.  Headings act as natural split points; chunks
# larger than this get split further by paragraph.
CHUNK_TARGET = 1500
CHUNK_MAX = 3000

# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    doc_type TEXT NOT NULL,
    title TEXT,
    content_hash TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    doc_id INTEGER NOT NULL REFERENCES documents(id),
    heading TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    char_start INTEGER NOT NULL,
    char_end INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id),
    embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
"""


def _init_db(db_path: Path) -> sqlite3.Connection:
    """Open (or create) the index database and ensure schema exists."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.executescript(SCHEMA_SQL)
    return conn


# ---------------------------------------------------------------------------
# Frontmatter parsing
# ---------------------------------------------------------------------------

def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split YAML frontmatter from markdown content.

    Returns (frontmatter_dict, body) where body has the frontmatter
    stripped.  If no frontmatter is found, returns ({}, text).
    """
    if not text.startswith("---"):
        return {}, text
    end = text.find("---", 3)
    if end == -1:
        return {}, text
    raw_yaml = text[3:end].strip()
    body = text[end + 3:].strip()
    try:
        fm = yaml.safe_load(raw_yaml) or {}
    except yaml.YAMLError:
        fm = {}
    return fm, body


# ---------------------------------------------------------------------------
# Markdown chunking
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def chunk_document(
    rel_path: str,
    text: str,
    doc_type: str = "article",
) -> list[dict]:
    """Split a markdown document into chunks by heading structure.

    Each chunk dict has:
        heading: str  — heading hierarchy ("# Title > ## Section")
        content: str  — the text of this chunk
        char_start: int
        char_end: int
    """
    fm, body = parse_frontmatter(text)
    title = fm.get("title", "") or ""

    # Find all headings and their positions
    headings: list[tuple[int, int, str]] = []  # (pos, level, text)
    for m in _HEADING_RE.finditer(body):
        level = len(m.group(1))
        headings.append((m.start(), level, m.group(2).strip()))

    if not headings:
        # No headings — treat the whole body as one chunk
        content = body.strip()
        if not content:
            return []
        return [_make_chunk(title, "", content, 0, len(body))]

    chunks: list[dict] = []
    heading_stack: list[str] = []  # Current heading hierarchy
    heading_levels: list[int] = []

    for i, (pos, level, heading_text) in enumerate(headings):
        # Determine the end of this section
        if i + 1 < len(headings):
            end_pos = headings[i + 1][0]
        else:
            end_pos = len(body)

        # Update heading stack
        while heading_levels and heading_levels[-1] >= level:
            heading_levels.pop()
            heading_stack.pop()
        heading_stack.append(heading_text)
        heading_levels.append(level)

        heading_path = " > ".join(heading_stack)

        # Extract content (excluding the heading line itself)
        heading_line_end = body.find("\n", pos)
        if heading_line_end == -1:
            heading_line_end = len(body)
        section_content = body[heading_line_end + 1:end_pos].strip()

        if not section_content:
            continue

        # Split large sections by paragraph
        if len(section_content) > CHUNK_MAX:
            sub_chunks = _split_by_paragraph(section_content, CHUNK_TARGET)
            for j, sub in enumerate(sub_chunks):
                label = f"{heading_path} (part {j+1})" if len(sub_chunks) > 1 else heading_path
                c_start = body.find(sub[:50], pos)
                if c_start == -1:
                    c_start = pos
                chunks.append(_make_chunk(title, label, sub, c_start, c_start + len(sub)))
        else:
            chunks.append(_make_chunk(title, heading_path, section_content, heading_line_end + 1, end_pos))

    # Content before first heading (if any)
    if headings[0][0] > 0:
        pre = body[:headings[0][0]].strip()
        if pre:
            chunks.insert(0, _make_chunk(title, "", pre, 0, headings[0][0]))

    return chunks


def _make_chunk(title: str, heading: str, content: str, char_start: int, char_end: int) -> dict:
    return {
        "heading": heading,
        "content": content,
        "char_start": char_start,
        "char_end": char_end,
        "embed_text": _build_embed_text(title, heading, content),
    }


def _build_embed_text(title: str, heading: str, content: str) -> str:
    """Build the text that gets embedded — includes title + heading for context."""
    parts: list[str] = []
    if title:
        parts.append(title)
    if heading:
        parts.append(heading)
    parts.append(content)
    return "\n".join(parts)


def _split_by_paragraph(text: str, target: int) -> list[str]:
    """Split text into chunks at paragraph boundaries, targeting `target` chars."""
    paragraphs = re.split(r"\n\n+", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if current and current_len + len(para) > target:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _serialize_embedding(vec: np.ndarray) -> bytes:
    """Serialize a float32 vector to bytes."""
    return vec.astype(np.float32).tobytes()


def _deserialize_embedding(blob: bytes) -> np.ndarray:
    """Deserialize bytes back to a float32 vector."""
    return np.frombuffer(blob, dtype=np.float32)


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _collect_documents(
    articles_dir: Path | None,
    descriptions_dir: Path | None,
) -> list[tuple[str, str, str]]:
    """Collect all markdown files to index.

    Returns list of (rel_path, content, doc_type).
    """
    docs: list[tuple[str, str, str]] = []

    if articles_dir and articles_dir.is_dir():
        for md in sorted(articles_dir.rglob("*.md")):
            rel = str(md.relative_to(articles_dir))
            docs.append((rel, md.read_text(), "article"))

    if descriptions_dir and descriptions_dir.is_dir():
        for md in sorted(descriptions_dir.rglob("*.md")):
            rel = str(md.relative_to(descriptions_dir))
            docs.append((rel, md.read_text(), "description"))

    return docs


def build_index(
    articles_dir: Path | None = None,
    descriptions_dir: Path | None = None,
    db_path: Path | None = None,
    embedding_config: EmbeddingConfig | None = None,
) -> Path:
    """Build (or rebuild) the embedding index from markdown files.

    Args:
        articles_dir: Directory containing article markdown files.
        descriptions_dir: Directory containing description markdown files.
        db_path: Path for the SQLite database.
        embedding_config: Embedding configuration. Uses defaults if not provided.

    Returns:
        Path to the created database.
    """
    if db_path is None:
        if articles_dir:
            db_path = articles_dir.parent / "codeknowledge.db"
        else:
            db_path = Path("codeknowledge.db")

    docs = _collect_documents(articles_dir, descriptions_dir)
    if not docs:
        raise ValueError("No documents found to index.")

    log.info("Indexing %d documents into %s", len(docs), db_path)

    # Wipe and rebuild — index is cheap to regenerate
    if db_path.exists():
        db_path.unlink()

    conn = _init_db(db_path)

    embedder = get_embedder(embedding_config)
    model_name = embedding_config.model_name if embedding_config else DEFAULT_MODEL

    # Store metadata
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        ("embedding_model", model_name),
    )

    # Chunk all documents
    all_chunks: list[tuple[int, dict]] = []  # (doc_id, chunk_dict)

    for rel_path, content, doc_type in docs:
        fm, _ = parse_frontmatter(content)
        title = fm.get("title", "")
        chash = _content_hash(content)

        conn.execute(
            "INSERT INTO documents (path, doc_type, title, content_hash) VALUES (?, ?, ?, ?)",
            (rel_path, doc_type, title, chash),
        )
        doc_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        chunks = chunk_document(rel_path, content, doc_type)
        for chunk in chunks:
            conn.execute(
                "INSERT INTO chunks (doc_id, heading, content, char_start, char_end) "
                "VALUES (?, ?, ?, ?, ?)",
                (doc_id, chunk["heading"], chunk["content"],
                 chunk["char_start"], chunk["char_end"]),
            )
            chunk_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            chunk["chunk_id"] = chunk_id
            all_chunks.append((doc_id, chunk))

    log.info("Chunked into %d chunks across %d documents", len(all_chunks), len(docs))

    if not all_chunks:
        conn.commit()
        conn.close()
        return db_path

    # Embed all chunks in one batch
    embed_texts_list = [chunk["embed_text"] for _, chunk in all_chunks]
    log.info("Embedding %d chunks...", len(embed_texts_list))
    embeddings = embedder.embed_documents(embed_texts_list)

    # Store embeddings
    for i, (_, chunk) in enumerate(all_chunks):
        blob = _serialize_embedding(embeddings[i])
        conn.execute(
            "INSERT INTO embeddings (chunk_id, embedding) VALUES (?, ?)",
            (chunk["chunk_id"], blob),
        )

    # Store embedding dimension for validation
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        ("embedding_dim", str(embeddings.shape[1])),
    )

    conn.commit()
    conn.close()

    log.info("Index built: %d chunks, %d dimensions", len(all_chunks), embeddings.shape[1])
    return db_path


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_index(
    query: str,
    db_path: Path,
    top_k: int = 10,
    embedding_config: EmbeddingConfig | None = None,
) -> list[dict]:
    """Search the embedding index for chunks matching a query.

    Args:
        query: Natural language search query.
        db_path: Path to the SQLite index database.
        top_k: Number of results to return.
        embedding_config: Embedding configuration. Uses defaults if not provided.

    Returns:
        List of result dicts with score, document, doc_type, title,
        heading, and content.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"Index not found: {db_path}")

    conn = sqlite3.connect(str(db_path))

    embedder = get_embedder(embedding_config)

    # Embed the query
    query_vec = embedder.embed_query(query)
    query_norm = query_vec / (np.linalg.norm(query_vec) or 1.0)

    # Load all embeddings
    rows = conn.execute(
        "SELECT e.chunk_id, e.embedding, c.heading, c.content, "
        "       d.path, d.doc_type, d.title "
        "FROM embeddings e "
        "JOIN chunks c ON c.id = e.chunk_id "
        "JOIN documents d ON d.id = c.doc_id"
    ).fetchall()

    if not rows:
        conn.close()
        return []

    # Compute cosine similarities
    chunk_ids = []
    chunk_meta = []
    vectors = []

    for chunk_id, blob, heading, content, doc_path, doc_type, title in rows:
        chunk_ids.append(chunk_id)
        chunk_meta.append({
            "document": doc_path,
            "doc_type": doc_type,
            "title": title or "",
            "heading": heading,
            "content": content,
        })
        vectors.append(_deserialize_embedding(blob))

    matrix = np.stack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    matrix_normed = matrix / norms

    scores = matrix_normed @ query_norm

    # Rank and return top-k
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        result = dict(chunk_meta[idx])
        result["score"] = float(scores[idx])
        results.append(result)

    conn.close()
    return results

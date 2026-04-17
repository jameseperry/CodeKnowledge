"""Subprocess entry point for building the code index.

Run as:  python -m codeknowledge._build_code_index <repo_root> <ck_dir>

Designed to be invoked from the main CLI in a separate process so that the
embedding model's memory is fully reclaimed after the text index is built.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python -m codeknowledge._build_code_index <repo_root> <ck_dir>",
              file=sys.stderr)
        return 1

    repo_root = Path(sys.argv[1])
    ck_dir = Path(sys.argv[2])

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from .config import Config
    from .cli import _collect_files
    from .index import build_code_index

    cfg = Config.load(ck_dir)

    source_targets = cfg.resolved_source_dirs() if cfg.source_dirs else [repo_root]
    extracted = []
    for t in source_targets:
        extracted.extend(_collect_files(t, repo_root, exclude=cfg.exclude))

    if not extracted:
        print("No source files found for code index.")
        return 0

    file_structures = [(fs, fp.read_text()) for fp, fs in extracted]
    code_db = ck_dir / "codeknowledge-code.db"

    code_result = build_code_index(
        file_structures=file_structures,
        db_path=code_db,
        embedding_config=cfg.code_embedding,
    )

    conn = sqlite3.connect(str(code_result))
    n_code = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    dim_row = conn.execute("SELECT value FROM meta WHERE key='embedding_dim'").fetchone()
    conn.close()
    dim = dim_row[0] if dim_row else "?"

    print(f"Code index built: {code_result}")
    print(f"  {n_code} chunks, {dim}d embeddings")
    return 0


if __name__ == "__main__":
    sys.exit(main())

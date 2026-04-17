"""Configuration for a .codeknowledge project."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


DEFAULT_CONFIG_NAME = "config.yaml"

DEFAULT_EXCLUDE = [
    "**/tests/**",
    "**/test/**",
    "**/__pycache__/**",
    "**/node_modules/**",
    "**/build/**",
    "**/dist/**",
]


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    backend: str = "local"  # "local" or "remote"
    model_name: str = "nomic-ai/nomic-embed-text-v1.5"
    dimensions: int = 768
    api_url: str = ""  # only needed for remote
    batch_size: int = 128
    max_concurrent: int = 4
    doc_prefix: str = "search_document: "
    query_prefix: str = "search_query: "


def _default_code_embedding() -> EmbeddingConfig:
    return EmbeddingConfig()


@dataclass
class Config:
    """Project-level configuration stored in .codeknowledge/config.yaml."""

    project_name: str = ""
    source_dirs: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    model: str = "sonnet"
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    code_embedding: EmbeddingConfig = field(default_factory=_default_code_embedding)

    # Paths resolved at load time (not serialized)
    repo_root: Path = field(default_factory=lambda: Path("."), repr=False)
    ck_dir: Path = field(default_factory=lambda: Path(".codeknowledge"), repr=False)

    @classmethod
    def load(cls, ck_dir: Path) -> Config:
        """Load config from a .codeknowledge directory.

        Returns a default Config if the file doesn't exist.
        """
        config_path = ck_dir / DEFAULT_CONFIG_NAME
        repo_root = ck_dir.parent

        if config_path.is_file():
            raw = yaml.safe_load(config_path.read_text()) or {}
        else:
            raw = {}

        emb_raw = raw.get("embedding", {})

        code_emb_raw = raw.get("code_embedding", {})
        code_emb = EmbeddingConfig(
            backend=code_emb_raw.get("backend", "local"),
            model_name=code_emb_raw.get("model_name", "nomic-ai/nomic-embed-text-v1.5"),
            dimensions=code_emb_raw.get("dimensions", 768),
            api_url=code_emb_raw.get("api_url", ""),
            batch_size=code_emb_raw.get("batch_size", 128),
            max_concurrent=code_emb_raw.get("max_concurrent", 4),
            doc_prefix=code_emb_raw.get("doc_prefix", "search_document: "),
            query_prefix=code_emb_raw.get("query_prefix", "search_query: "),
        )

        return cls(
            project_name=raw.get("project_name", repo_root.name),
            source_dirs=raw.get("source_dirs", []),
            exclude=raw.get("exclude", list(DEFAULT_EXCLUDE)),
            model=raw.get("model", "sonnet"),
            embedding=EmbeddingConfig(
                backend=emb_raw.get("backend", "local"),
                model_name=emb_raw.get("model_name", "nomic-ai/nomic-embed-text-v1.5"),
                dimensions=emb_raw.get("dimensions", 768),
                api_url=emb_raw.get("api_url", ""),
                batch_size=emb_raw.get("batch_size", 128),
                max_concurrent=emb_raw.get("max_concurrent", 4),
                doc_prefix=emb_raw.get("doc_prefix", "search_document: "),
                query_prefix=emb_raw.get("query_prefix", "search_query: "),
            ),
            code_embedding=code_emb,
            repo_root=repo_root,
            ck_dir=ck_dir,
        )

    def save(self, ck_dir: Path | None = None) -> Path:
        """Write config to disk. Returns the config file path."""
        target = ck_dir or self.ck_dir
        config_path = target / DEFAULT_CONFIG_NAME

        data: dict = {}
        if self.project_name:
            data["project_name"] = self.project_name
        if self.source_dirs:
            data["source_dirs"] = self.source_dirs
        data["exclude"] = self.exclude
        data["model"] = self.model

        # Always serialize embedding sections
        emb = self.embedding
        data["embedding"] = {
            "backend": emb.backend,
            "model_name": emb.model_name,
        }
        if emb.dimensions != 768:
            data["embedding"]["dimensions"] = emb.dimensions
        if emb.api_url:
            data["embedding"]["api_url"] = emb.api_url
        if emb.doc_prefix != "search_document: ":
            data["embedding"]["doc_prefix"] = emb.doc_prefix
        if emb.query_prefix != "search_query: ":
            data["embedding"]["query_prefix"] = emb.query_prefix

        ce = self.code_embedding
        data["code_embedding"] = {
            "backend": ce.backend,
            "model_name": ce.model_name,
        }
        if ce.dimensions != 768:
            data["code_embedding"]["dimensions"] = ce.dimensions
        if ce.api_url:
            data["code_embedding"]["api_url"] = ce.api_url
        if ce.doc_prefix != "search_document: ":
            data["code_embedding"]["doc_prefix"] = ce.doc_prefix
        if ce.query_prefix != "search_query: ":
            data["code_embedding"]["query_prefix"] = ce.query_prefix

        config_path.write_text(yaml.dump(data, default_flow_style=False, sort_keys=False))
        return config_path

    @property
    def articles_dir(self) -> Path:
        return self.ck_dir / "articles"

    @property
    def descriptions_dir(self) -> Path:
        return self.ck_dir / "descriptions"

    @property
    def db_path(self) -> Path:
        return self.ck_dir / "codeknowledge.db"

    def resolved_source_dirs(self) -> list[Path]:
        """Resolve source_dirs relative to repo_root.

        If source_dirs is empty, returns [repo_root] (index everything).
        """
        if not self.source_dirs:
            return [self.repo_root]
        return [(self.repo_root / d).resolve() for d in self.source_dirs]

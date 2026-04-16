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
class Config:
    """Project-level configuration stored in .codeknowledge/config.yaml."""

    project_name: str = ""
    source_dirs: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=lambda: list(DEFAULT_EXCLUDE))
    model: str = "sonnet"

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

        return cls(
            project_name=raw.get("project_name", repo_root.name),
            source_dirs=raw.get("source_dirs", []),
            exclude=raw.get("exclude", list(DEFAULT_EXCLUDE)),
            model=raw.get("model", "sonnet"),
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

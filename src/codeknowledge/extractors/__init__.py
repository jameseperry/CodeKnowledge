"""Base interface for language-specific structural extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ..model import FileStructure


class Extractor(ABC):
    """Base class for language-specific code extractors."""

    @abstractmethod
    def language(self) -> str:
        """Return the language name (e.g., 'python', 'cpp')."""
        ...

    @abstractmethod
    def extensions(self) -> set[str]:
        """Return supported file extensions (e.g., {'.py'})."""
        ...

    @abstractmethod
    def extract(self, source: bytes, rel_path: str) -> FileStructure:
        """Extract structural elements from source code."""
        ...


# Registry of extractors by file extension
_registry: dict[str, Extractor] = {}


def register_extractor(extractor: Extractor) -> None:
    for ext in extractor.extensions():
        _registry[ext] = extractor


def get_extractor(path: str | Path) -> Extractor | None:
    """Get the appropriate extractor for a file path, or None if unsupported."""
    suffix = Path(path).suffix
    return _registry.get(suffix)

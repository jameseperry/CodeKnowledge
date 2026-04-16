"""Git utilities for commit tracking and diff computation."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


def get_head_commit(repo_root: Path) -> str | None:
    """Return the short SHA of HEAD, or None if not a git repo / no commits."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def has_uncommitted_changes(repo_root: Path, paths: list[str] | None = None) -> bool:
    """Check if working tree has uncommitted changes in the given paths.

    If *paths* is None, checks the entire repo.
    """
    cmd = ["git", "status", "--porcelain"]
    if paths:
        cmd.append("--")
        cmd.extend(paths)
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False


def get_diff(repo_root: Path, from_commit: str, paths: list[str] | None = None) -> str | None:
    """Get the diff from *from_commit* to the current working tree.

    Returns the diff text, empty string if no changes, or None on error.
    """
    cmd = ["git", "diff", from_commit, "--"]
    if paths:
        cmd.extend(paths)
    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def is_git_repo(repo_root: Path) -> bool:
    """Check whether *repo_root* is inside a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

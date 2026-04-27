from __future__ import annotations

from pathlib import Path


def resolve_project_path(path_str: str, project_root: Path) -> Path:
    """Resolve ``path_str`` against ``project_root`` with a datasets fallback.

    Absolute paths are returned as-is. Relative paths are first looked up under
    ``<project_root>/datasets/``; if that does not exist, the path is resolved
    relative to ``project_root`` itself.
    """

    path = Path(path_str)
    if path.is_absolute():
        return path
    dataset_candidate = (project_root / "datasets" / path).resolve()
    if dataset_candidate.exists():
        return dataset_candidate
    return (project_root / path).resolve()

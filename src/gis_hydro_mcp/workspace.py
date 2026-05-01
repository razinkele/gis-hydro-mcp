"""Workspace path validation, mirrors gdal-mcp's GDAL_MCP_WORKSPACES.

Reads ``GIS_HYDRO_WORKSPACES`` (semicolon- or comma-separated absolute paths).
Any input/output path passed to a tool must resolve inside one of these roots.
If the env var is unset, no restriction is applied (development mode).
"""

from __future__ import annotations

import os
from pathlib import Path


def _roots() -> list[Path]:
    raw = os.environ.get("GIS_HYDRO_WORKSPACES", "").strip()
    if not raw:
        return []
    parts = [p for chunk in raw.split(";") for p in chunk.split(",")]
    return [Path(p).expanduser().resolve() for p in parts if p.strip()]


def validate_path(path: str | os.PathLike[str], *, must_exist: bool = False) -> Path:
    """Return the resolved Path if it lies inside an allowed workspace root.

    Raises ``PermissionError`` on escape attempts and ``FileNotFoundError`` if
    ``must_exist`` and the file is missing.
    """
    p = Path(path).expanduser().resolve()
    roots = _roots()
    if roots and not any(_is_within(p, r) for r in roots):
        raise PermissionError(
            f"Path {p} is outside the allowed GIS_HYDRO_WORKSPACES roots: "
            + ", ".join(str(r) for r in roots)
        )
    if must_exist and not p.exists():
        raise FileNotFoundError(p)
    return p


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False

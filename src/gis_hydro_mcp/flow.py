"""D8 flow direction, accumulation, stream extraction, watershed.

Pure-numpy implementations with strict invariants:

* NoData (NaN) cells emit no flow and receive no accumulation.
* Edge cells flow off-raster (treated as outlets) — their downstream is None.
* D8 encoding follows ESRI:

    32  64  128
    16   0    1
     8   4    2

* ``0`` denotes an unresolved sink (no lower neighbour after fill).
* Ties are broken deterministically by the neighbour ordering above
  (E, SE, S, SW, W, NW, N, NE = 1, 2, 4, 8, 16, 32, 64, 128).
* ``flow_accumulation`` validates input encoding and detects cycles via
  Kahn's algorithm; an unresolved remainder raises ``RuntimeError``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

# (code, dr, dc, distance_factor)
_D8 = [
    (1,    0,  1, 1.0),
    (2,    1,  1, np.sqrt(2)),
    (4,    1,  0, 1.0),
    (8,    1, -1, np.sqrt(2)),
    (16,   0, -1, 1.0),
    (32,  -1, -1, np.sqrt(2)),
    (64,  -1,  0, 1.0),
    (128, -1,  1, np.sqrt(2)),
]
_VALID_CODES = {0, 1, 2, 4, 8, 16, 32, 64, 128}


def flow_direction(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Return D8 flow direction raster (uint8). Sinks encoded as 0.

    NaN propagates as 255 (sentinel for NoData; consumers should mask).
    """
    h, w = dem.shape
    out = np.zeros((h, w), dtype=np.uint8)
    out[np.isnan(dem)] = 255

    pixel_dist = np.array(
        [
            dx,
            np.hypot(dx, dy),
            dy,
            np.hypot(dx, dy),
            dx,
            np.hypot(dx, dy),
            dy,
            np.hypot(dx, dy),
        ]
    )

    best_drop = np.full((h, w), -np.inf, dtype="float64")
    for idx, (code, dr, dc, _) in enumerate(_D8):
        z = dem
        z_n = np.full_like(z, np.nan)
        r0 = max(0, -dr); r1 = h - max(0, dr)
        c0 = max(0, -dc); c1 = w - max(0, dc)
        z_n[r0:r1, c0:c1] = z[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        drop = (z - z_n) / pixel_dist[idx]
        better = ~np.isnan(drop) & (drop > best_drop) & (drop > 0)
        best_drop = np.where(better, drop, best_drop)
        out = np.where(better & ~np.isnan(z), code, out)

    out[np.isnan(dem)] = 255
    return out


@dataclass
class AccumulationResult:
    accumulation: np.ndarray  # float64
    sinks: int
    nodata_cells: int
    cycles_detected: int


def flow_accumulation(flow_dir: np.ndarray, *, weights: np.ndarray | None = None) -> AccumulationResult:
    """Topological-order accumulation. Each valid cell starts with weight 1.0
    (or ``weights`` if provided). NoData cells (255) contribute nothing.
    """
    h, w = flow_dir.shape
    fd = flow_dir
    valid = fd != 255
    nodata_cells = int((~valid).sum())

    # Validate encoding
    invalid_mask = valid & ~np.isin(fd, list(_VALID_CODES))
    if invalid_mask.any():
        bad = np.unique(fd[invalid_mask])
        raise ValueError(f"flow_dir contains invalid D8 codes: {bad.tolist()}")

    if weights is None:
        acc = np.where(valid, 1.0, 0.0)
    else:
        if weights.shape != fd.shape:
            raise ValueError("weights shape must match flow_dir shape")
        acc = np.where(valid, weights.astype("float64"), 0.0)

    # Build downstream lookup as flat index → flat index (-1 = outlet).
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    downstream = np.full(h * w, -1, dtype=np.int64)
    in_degree = np.zeros(h * w, dtype=np.int32)

    for code, dr, dc, _ in _D8:
        m = valid & (fd == code)
        if not m.any():
            continue
        nr = rr[m] + dr
        nc = cc[m] + dc
        within = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
        src = (rr[m] * w + cc[m])[within]
        nrw = nr[within]; ncw = nc[within]
        # downstream must itself be valid (not NoData, not sink-only)
        downstream_valid = valid[nrw, ncw]
        src = src[downstream_valid]
        dst = (nrw * w + ncw)[downstream_valid]
        downstream[src] = dst
        np.add.at(in_degree, dst, 1)

    sinks = int(((fd == 0) & valid).sum())

    # Kahn's algorithm
    q: deque[int] = deque(int(i) for i in np.flatnonzero((in_degree == 0) & valid.ravel()))
    acc_flat = acc.ravel()
    processed = 0
    while q:
        i = q.popleft()
        processed += 1
        d = downstream[i]
        if d >= 0:
            acc_flat[d] += acc_flat[i]
            in_degree[d] -= 1
            if in_degree[d] == 0:
                q.append(int(d))

    valid_count = int(valid.sum())
    cycles = max(0, valid_count - processed)
    if cycles:
        raise RuntimeError(
            f"flow_accumulation: {cycles} cells unprocessed (cycle in flow_dir?). "
            "Run dem_fill_sinks before flow_direction."
        )

    return AccumulationResult(
        accumulation=acc_flat.reshape(h, w),
        sinks=sinks,
        nodata_cells=nodata_cells,
        cycles_detected=cycles,
    )


def stream_mask(accumulation: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean mask of cells with accumulation ≥ threshold."""
    return accumulation >= threshold


def watershed(flow_dir: np.ndarray, outlet_row: int, outlet_col: int) -> np.ndarray:
    """Boolean mask of all cells that drain to ``(outlet_row, outlet_col)``."""
    h, w = flow_dir.shape
    fd = flow_dir
    valid = fd != 255

    # Build reverse adjacency on the fly via BFS from the outlet, walking
    # upstream: a cell is upstream if its downstream == current cell.
    upstream_lookup: dict[int, list[int]] = {}
    rr, cc = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    for code, dr, dc, _ in _D8:
        m = valid & (fd == code)
        if not m.any():
            continue
        nr = rr[m] + dr
        nc = cc[m] + dc
        within = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
        src_r = rr[m][within]; src_c = cc[m][within]
        dst_r = nr[within]; dst_c = nc[within]
        for sr, sc, drr, dcc in zip(src_r, src_c, dst_r, dst_c):
            upstream_lookup.setdefault(int(drr) * w + int(dcc), []).append(int(sr) * w + int(sc))

    mask = np.zeros((h, w), dtype=bool)
    start = outlet_row * w + outlet_col
    stack = [start]
    mask[outlet_row, outlet_col] = True
    while stack:
        i = stack.pop()
        for u in upstream_lookup.get(i, ()):
            ur, uc = divmod(u, w)
            if not mask[ur, uc]:
                mask[ur, uc] = True
                stack.append(u)
    return mask

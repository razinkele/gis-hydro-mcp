"""Topographic surface analysis: slope, aspect, hillshade, sink fill.

All functions accept a 2-D ``numpy.float64`` DEM (NaN = NoData) and pixel
sizes ``dx`` and ``dy``, and return a 2-D ``numpy.float64`` array.
"""

from __future__ import annotations

import numpy as np

_PAD = ((1, 1), (1, 1))


def _pad_nan(arr: np.ndarray) -> np.ndarray:
    return np.pad(arr, _PAD, mode="edge")


def _horn_gradients(dem: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (dz/dx, dz/dy) using Horn's 3x3 finite-difference stencil."""
    z = _pad_nan(dem)
    a = z[0:-2, 0:-2]; b = z[0:-2, 1:-1]; c = z[0:-2, 2:]
    d = z[1:-1, 0:-2];                      f = z[1:-1, 2:]
    g = z[2:,   0:-2]; h = z[2:,   1:-1]; i = z[2:,   2:]
    dzdx = ((c + 2 * f + i) - (a + 2 * d + g)) / (8.0 * dx)
    dzdy = ((g + 2 * h + i) - (a + 2 * b + c)) / (8.0 * dy)
    return dzdx, dzdy


def slope(dem: np.ndarray, dx: float, dy: float, *, units: str = "degrees") -> np.ndarray:
    """Slope angle. ``units`` ∈ {"degrees", "percent", "radians"}."""
    dzdx, dzdy = _horn_gradients(dem, dx, dy)
    rise = np.hypot(dzdx, dzdy)
    if units == "radians":
        return np.arctan(rise)
    if units == "percent":
        return rise * 100.0
    return np.degrees(np.arctan(rise))


def aspect(dem: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Aspect azimuth in degrees clockwise from north. -1 on flat cells."""
    dzdx, dzdy = _horn_gradients(dem, dx, dy)
    flat = (dzdx == 0) & (dzdy == 0)
    asp = np.degrees(np.arctan2(dzdy, -dzdx))
    asp = np.where(asp < 0, 90.0 - asp, np.where(asp > 90.0, 360.0 - asp + 90.0, 90.0 - asp))
    asp = np.where(flat, -1.0, asp)
    return asp


def hillshade(
    dem: np.ndarray,
    dx: float,
    dy: float,
    *,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> np.ndarray:
    """ESRI-style hillshade in [0, 255]. NaN propagates."""
    dzdx, dzdy = _horn_gradients(dem * z_factor, dx, dy)
    slope_rad = np.arctan(np.hypot(dzdx, dzdy))
    aspect_rad = np.arctan2(dzdy, -dzdx)
    aspect_rad = np.where(aspect_rad < 0, 2 * np.pi + aspect_rad, aspect_rad)
    az_rad = np.radians(360.0 - azimuth + 90.0)
    az_rad = np.where(az_rad >= 2 * np.pi, az_rad - 2 * np.pi, az_rad)
    zen_rad = np.radians(90.0 - altitude)
    hs = (
        np.cos(zen_rad) * np.cos(slope_rad)
        + np.sin(zen_rad) * np.sin(slope_rad) * np.cos(az_rad - aspect_rad)
    )
    hs = np.clip(hs * 255.0, 0, 255)
    return hs


def fill_sinks(dem: np.ndarray, *, epsilon: float = 1e-3) -> np.ndarray:
    """Planchon–Darboux iterative depression fill.

    Returns a hydrologically corrected DEM where every non-edge cell has at
    least one neighbour with strictly lower elevation. ``epsilon`` is the
    minimum drop enforced between adjacent filled cells.

    For very large DEMs this is O(n^2) in the worst case; fine for the
    workspace-sized rasters this MCP targets. NaN cells are treated as
    "open" boundaries (water can drain through them).
    """
    dem = dem.astype("float64", copy=True)
    valid = ~np.isnan(dem)
    if not valid.any():
        return dem

    INF = np.float64(np.nanmax(dem) + 1e6)
    w = np.where(valid, INF, np.nan)

    # Edge / NoData-adjacent cells keep their original elevation.
    h, wd = dem.shape
    edge = np.zeros_like(valid, dtype=bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True
    # cells next to NaN are also open boundaries
    pad = np.pad(~valid, 1, constant_values=True)
    nan_neigh = (
        pad[0:-2, 1:-1] | pad[2:, 1:-1] | pad[1:-1, 0:-2] | pad[1:-1, 2:]
        | pad[0:-2, 0:-2] | pad[0:-2, 2:] | pad[2:, 0:-2] | pad[2:, 2:]
    )
    boundary = valid & (edge | nan_neigh)
    w[boundary] = dem[boundary]

    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    # Worst-case propagation distance is the DEM diameter; one iteration advances
    # the front by 1 cell per offset direction, so cap generously by dimensions.
    max_iters = max(2 * (h + wd), 200)
    changed = True
    it = 0
    while changed and it < max_iters:
        changed = False
        it += 1
        for dr, dc in offsets:
            r0 = max(0, dr); r1 = h + min(0, dr)
            c0 = max(0, dc); c1 = wd + min(0, dc)
            nr0 = r0 - dr; nr1 = r1 - dr
            nc0 = c0 - dc; nc1 = c1 - dc

            target = w[r0:r1, c0:c1]
            neigh = w[nr0:nr1, nc0:nc1]
            z = dem[r0:r1, c0:c1]

            cond = ~np.isnan(target) & ~np.isnan(neigh) & (target > z)
            new1 = np.where(z >= neigh + epsilon, z, np.nan)
            new2 = np.where(neigh + epsilon > z, neigh + epsilon, np.nan)
            new = np.where(~np.isnan(new1) & cond, new1, np.where(cond, new2, target))
            update = cond & (new < target)
            if update.any():
                target_view = w[r0:r1, c0:c1]
                target_view[update] = new[update]
                changed = True
    # Safety: any cell still sitting at the INF sentinel did not converge —
    # revert it to the original DEM elevation rather than leaking 1e6 m artifacts.
    unconverged = valid & (w >= INF - 1.0)
    if unconverged.any():
        w[unconverged] = dem[unconverged]
    return w

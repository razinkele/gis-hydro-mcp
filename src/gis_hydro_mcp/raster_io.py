"""Thin rasterio helpers shared by topo/flow modules."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import Affine


@dataclass
class RasterData:
    array: np.ndarray  # 2D, float64 with NaN for nodata
    transform: Affine
    crs: rasterio.crs.CRS | None
    nodata: float | None
    width: int
    height: int
    dx: float
    dy: float  # absolute pixel size (positive)


def read_dem(path: str | Path, band: int = 1) -> RasterData:
    """Read a single-band raster as float64 with NaN for nodata.

    Returns the array along with the geospatial metadata needed by all
    downstream functions.
    """
    with rasterio.open(path) as ds:
        arr = ds.read(band).astype("float64")
        nodata = ds.nodata
        if nodata is not None:
            arr[arr == nodata] = np.nan
        return RasterData(
            array=arr,
            transform=ds.transform,
            crs=ds.crs,
            nodata=nodata,
            width=ds.width,
            height=ds.height,
            dx=abs(ds.transform.a),
            dy=abs(ds.transform.e),
        )


def write_like(
    path: str | Path,
    array: np.ndarray,
    template: RasterData,
    *,
    dtype: str | None = None,
    nodata: float | int | None = None,
) -> Path:
    """Write ``array`` using ``template``'s CRS/transform/shape."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if dtype is None:
        dtype = "float32" if array.dtype.kind == "f" else str(array.dtype)
    if nodata is None and array.dtype.kind == "f":
        nodata = float("nan")
    profile = {
        "driver": "GTiff",
        "height": template.height,
        "width": template.width,
        "count": 1,
        "dtype": dtype,
        "crs": template.crs,
        "transform": template.transform,
        "compress": "deflate",
        "tiled": True,
    }
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(out_path, "w", **profile) as dst:
        out_arr = array.astype(dtype, copy=False)
        if dtype.startswith("float") and nodata is not None and np.isnan(nodata):
            dst.write(np.where(np.isnan(out_arr), np.nan, out_arr).astype(dtype), 1)
        else:
            dst.write(out_arr, 1)
    return out_path

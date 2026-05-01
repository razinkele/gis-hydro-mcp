"""FastMCP server exposing topo + flow + network tools."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import rasterio
from fastmcp import FastMCP
from pydantic import BaseModel, Field
from rasterio import features
from shapely.geometry import shape

from . import flow, network, raster_io, topo
from .workspace import validate_path

mcp = FastMCP("gis-hydro")


# ---------------------------------------------------------------------------
# Topographic tools
# ---------------------------------------------------------------------------


class DemIO(BaseModel):
    input: str = Field(description="Path to input DEM raster (single band).")
    output: str = Field(description="Path to write output GeoTIFF.")


class SlopeArgs(DemIO):
    units: Literal["degrees", "percent", "radians"] = "degrees"


@mcp.tool
def dem_slope(args: SlopeArgs) -> dict:
    """Compute slope from a DEM (Horn's 3x3 method)."""
    src = validate_path(args.input, must_exist=True)
    dst = validate_path(args.output)
    dem = raster_io.read_dem(src)
    out = topo.slope(dem.array, dem.dx, dem.dy, units=args.units)
    raster_io.write_like(dst, out, dem)
    return {"output": str(dst), "units": args.units, "min": float(np.nanmin(out)), "max": float(np.nanmax(out))}


@mcp.tool
def dem_aspect(args: DemIO) -> dict:
    """Compute aspect azimuth (degrees clockwise from north; -1 on flats)."""
    src = validate_path(args.input, must_exist=True)
    dst = validate_path(args.output)
    dem = raster_io.read_dem(src)
    out = topo.aspect(dem.array, dem.dx, dem.dy)
    raster_io.write_like(dst, out, dem)
    return {"output": str(dst)}


class HillshadeArgs(DemIO):
    azimuth: float = Field(315.0, ge=0, le=360, description="Sun azimuth in degrees")
    altitude: float = Field(45.0, ge=0, le=90, description="Sun altitude in degrees")
    z_factor: float = Field(1.0, gt=0, description="Vertical exaggeration")


@mcp.tool
def dem_hillshade(args: HillshadeArgs) -> dict:
    """Compute hillshade (0–255)."""
    src = validate_path(args.input, must_exist=True)
    dst = validate_path(args.output)
    dem = raster_io.read_dem(src)
    out = topo.hillshade(dem.array, dem.dx, dem.dy, azimuth=args.azimuth, altitude=args.altitude, z_factor=args.z_factor)
    raster_io.write_like(dst, out, dem, dtype="uint8", nodata=0)
    return {"output": str(dst)}


@mcp.tool
def dem_fill_sinks(args: DemIO) -> dict:
    """Fill depressions (Planchon–Darboux). Run before flow_direction."""
    src = validate_path(args.input, must_exist=True)
    dst = validate_path(args.output)
    dem = raster_io.read_dem(src)
    out = topo.fill_sinks(dem.array)
    diff = out - dem.array
    raster_io.write_like(dst, out, dem)
    return {
        "output": str(dst),
        "cells_raised": int(np.nansum(diff > 0)),
        "max_fill": float(np.nanmax(diff)) if np.any(diff > 0) else 0.0,
    }


# ---------------------------------------------------------------------------
# Flow / hydrology tools
# ---------------------------------------------------------------------------


@mcp.tool
def dem_flow_direction(args: DemIO) -> dict:
    """D8 flow direction. ESRI encoding (1/2/4/8/16/32/64/128); 0=sink, 255=NoData."""
    src = validate_path(args.input, must_exist=True)
    dst = validate_path(args.output)
    dem = raster_io.read_dem(src)
    fd = flow.flow_direction(dem.array, dem.dx, dem.dy)
    raster_io.write_like(dst, fd, dem, dtype="uint8", nodata=255)
    sinks = int(((fd == 0) & (~np.isnan(dem.array))).sum())
    return {"output": str(dst), "sinks": sinks, "encoding": "ESRI D8 (1,2,4,8,16,32,64,128); 0=sink; 255=NoData"}


class FlowAccArgs(BaseModel):
    flow_dir: str = Field(description="Path to D8 flow-direction raster.")
    output: str = Field(description="Path to write accumulation GeoTIFF.")


@mcp.tool
def dem_flow_accumulation(args: FlowAccArgs) -> dict:
    """Accumulation (cell counts) over a D8 flow-direction raster."""
    src = validate_path(args.flow_dir, must_exist=True)
    dst = validate_path(args.output)
    with rasterio.open(src) as ds:
        fd = ds.read(1)
        template = raster_io.RasterData(
            array=fd.astype("float64"),
            transform=ds.transform,
            crs=ds.crs,
            nodata=ds.nodata,
            width=ds.width,
            height=ds.height,
            dx=abs(ds.transform.a),
            dy=abs(ds.transform.e),
        )
    res = flow.flow_accumulation(fd)
    raster_io.write_like(dst, res.accumulation, template, dtype="float32")
    return {
        "output": str(dst),
        "max_accumulation": float(res.accumulation.max()),
        "sinks": res.sinks,
        "nodata_cells": res.nodata_cells,
    }


class StreamArgs(BaseModel):
    accumulation: str = Field(description="Path to flow-accumulation raster.")
    output: str = Field(description="Output GeoPackage for stream lines.")
    threshold: float = Field(description="Min accumulation (cell count) to be a stream.")


@mcp.tool
def dem_stream_network(args: StreamArgs) -> dict:
    """Vectorize cells with accumulation ≥ threshold into lines (GeoPackage)."""
    src = validate_path(args.accumulation, must_exist=True)
    dst = validate_path(args.output)
    with rasterio.open(src) as ds:
        acc = ds.read(1)
        transform = ds.transform
        crs = ds.crs
    mask = flow.stream_mask(acc, args.threshold).astype("uint8")
    shapes = features.shapes(mask, mask=mask.astype(bool), transform=transform)
    polys = [shape(geom) for geom, val in shapes if val == 1]
    if not polys:
        gpd.GeoDataFrame({"geometry": []}, crs=crs).to_file(dst, driver="GPKG")
        return {"output": str(dst), "features": 0}
    # Skeletonise via centerline approximation: take polygon as raster mask and
    # emit cell-sized line segments along true axis. For an MVP we just emit
    # polygons; consumers can post-process. Future: centerlines via rasterio
    # transform + numpy.
    gdf = gpd.GeoDataFrame({"geometry": polys}, crs=crs)
    gdf.to_file(dst, driver="GPKG", layer="streams")
    return {"output": str(dst), "features": len(gdf), "note": "MVP: polygons of stream cells; centerline extraction TBD"}


class WatershedArgs(BaseModel):
    flow_dir: str = Field(description="Path to D8 flow-direction raster.")
    outlet_x: float = Field(description="Outlet X coordinate (in raster CRS).")
    outlet_y: float = Field(description="Outlet Y coordinate (in raster CRS).")
    output_raster: str = Field(description="Output GeoTIFF (1=in basin, 0=out).")
    output_polygon: str = Field(description="Output GeoPackage of basin polygon.")


@mcp.tool
def dem_watershed(args: WatershedArgs) -> dict:
    """Delineate the upstream catchment for an outlet point."""
    src = validate_path(args.flow_dir, must_exist=True)
    dst_r = validate_path(args.output_raster)
    dst_v = validate_path(args.output_polygon)
    with rasterio.open(src) as ds:
        fd = ds.read(1)
        transform = ds.transform
        crs = ds.crs
        col, row = ~transform * (args.outlet_x, args.outlet_y)
        row, col = int(row), int(col)
        template = raster_io.RasterData(
            array=fd.astype("float64"), transform=transform, crs=crs,
            nodata=ds.nodata, width=ds.width, height=ds.height,
            dx=abs(transform.a), dy=abs(transform.e),
        )
    if not (0 <= row < fd.shape[0] and 0 <= col < fd.shape[1]):
        raise ValueError("Outlet coordinates fall outside the raster extent.")
    mask = flow.watershed(fd, row, col)
    raster_io.write_like(dst_r, mask.astype("uint8"), template, dtype="uint8", nodata=0)
    polys = [shape(g) for g, v in features.shapes(mask.astype("uint8"), mask=mask, transform=transform) if v == 1]
    if polys:
        merged = polys[0] if len(polys) == 1 else polys[0].union(*polys[1:])
        gpd.GeoDataFrame({"geometry": [merged]}, crs=crs).to_file(dst_v, driver="GPKG", layer="watershed")
    return {
        "output_raster": str(dst_r),
        "output_polygon": str(dst_v),
        "area_cells": int(mask.sum()),
        "area_units2": float(mask.sum() * abs(transform.a) * abs(transform.e)),
    }


# ---------------------------------------------------------------------------
# Vector network tools
# ---------------------------------------------------------------------------


class NetworkBuildArgs(BaseModel):
    input: str = Field(description="Input line layer (any OGR-readable; for GPKG/GDB use 'path|layer=name').")
    output: str = Field(description="Output GeoPackage (will contain 'nodes' and 'edges' layers).")
    snap_tolerance: float = Field(description="Endpoint snap distance in CRS units.")
    dem: str | None = Field(None, description="Optional DEM for direction inference.")
    min_dz: float = Field(0.0, description="Min |Δz| to commit to a DEM-inferred direction.")


def _read_vector(spec: str) -> gpd.GeoDataFrame:
    if "|layer=" in spec:
        path, _, layer = spec.partition("|layer=")
        return gpd.read_file(validate_path(path, must_exist=True), layer=layer)
    return gpd.read_file(validate_path(spec, must_exist=True))


@mcp.tool
def network_build(args: NetworkBuildArgs) -> dict:
    """Build a directed stream-network graph from a line layer.

    Snaps endpoints, nodes interior tributary junctions, and (optionally)
    infers flow direction from a DEM. Writes 'nodes' and 'edges' layers to a
    GeoPackage and returns diagnostics.
    """
    out = validate_path(args.output)
    dem_path = str(validate_path(args.dem, must_exist=True)) if args.dem else None
    lines = _read_vector(args.input)
    res = network.build_network(lines, snap_tolerance=args.snap_tolerance, dem_path=dem_path, min_dz=args.min_dz)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.nodes.to_file(out, driver="GPKG", layer="nodes")
    res.edges.to_file(out, driver="GPKG", layer="edges")
    return {"output": str(out), **res.diagnostics}


class NetworkOpArgs(BaseModel):
    network: str = Field(description="GeoPackage produced by network_build (must contain 'nodes' and 'edges').")
    output: str = Field(description="Output GeoPackage.")


@mcp.tool
def network_connected_components(args: NetworkOpArgs) -> dict:
    """Label connected components on the edges layer."""
    src = validate_path(args.network, must_exist=True)
    dst = validate_path(args.output)
    edges = gpd.read_file(src, layer="edges")
    out = network.connected_components(edges)
    out.to_file(dst, driver="GPKG", layer="edges")
    return {"output": str(dst), "components": int(out["component_id"].nunique())}


class TraceArgs(NetworkOpArgs):
    start_edge_id: int = Field(description="edge_id from which to start tracing.")
    direction: Literal["upstream", "downstream"] = "downstream"


@mcp.tool
def network_trace(args: TraceArgs) -> dict:
    """Trace upstream or downstream from a given edge."""
    src = validate_path(args.network, must_exist=True)
    dst = validate_path(args.output)
    edges = gpd.read_file(src, layer="edges")
    out = network.trace(edges, args.start_edge_id, direction=args.direction)
    out.to_file(dst, driver="GPKG", layer="trace")
    return {"output": str(dst), "edges": len(out), "direction": args.direction}


@mcp.tool
def network_strahler(args: NetworkOpArgs) -> dict:
    """Compute Strahler stream order per edge."""
    src = validate_path(args.network, must_exist=True)
    dst = validate_path(args.output)
    edges = gpd.read_file(src, layer="edges")
    out = network.strahler(edges)
    out.to_file(dst, driver="GPKG", layer="edges")
    return {"output": str(dst), "max_order": int(out["strahler"].max())}


@mcp.tool
def network_confluences(args: NetworkOpArgs) -> dict:
    """Extract confluence points (in-degree ≥ 2) from a built network."""
    src = validate_path(args.network, must_exist=True)
    dst = validate_path(args.output)
    edges = gpd.read_file(src, layer="edges")
    nodes = gpd.read_file(src, layer="nodes")
    out = network.confluences(edges, nodes)
    out.to_file(dst, driver="GPKG", layer="confluences")
    return {"output": str(dst), "confluences": len(out)}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="gis-hydro")
    parser.add_argument("--transport", default="stdio", choices=["stdio"])
    args = parser.parse_args(argv)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()

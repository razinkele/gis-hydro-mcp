# gis-hydro-mcp

Topographic and hydrological-connectivity MCP server. A pure-Python companion
to [`gdal-mcp`](https://github.com/JordanGunn/gdal-mcp): it adds DEM-based
terrain + flow tools and vector stream-network topology tools that gdal-mcp
does not yet provide.

Designed so that `topo.py`, `flow.py`, and `network.py` are plain-Python
modules with no MCP coupling — they can be lifted directly into a future
`gdal-mcp` PR.

## Tools

### Topographic (DEM raster → raster)
- `dem_slope` — Horn's 3×3 slope (degrees or percent)
- `dem_aspect` — aspect azimuth (−1 for flat)
- `dem_hillshade` — shaded relief
- `dem_fill_sinks` — Planchon–Darboux depression fill

### Hydrological (DEM raster)
- `dem_flow_direction` — D8 (ESRI encoding 1/2/4/8/16/32/64/128; 0 = sink)
- `dem_flow_accumulation` — topological-sort accumulation (cycle-checked)
- `dem_stream_network` — vectorize streams above an accumulation threshold
- `dem_watershed` — upstream catchment (raster + polygon) for an outlet point

### Vector network (river/stream lines)
- `network_build` — snap endpoints, **node interior tributary junctions**, build
  a directed graph; optional DEM-based direction inference with confidence
  fields. Returns diagnostics (counts of dangles, junctions, ambiguous edges,
  components).
- `network_connected_components` — adds `component_id` per edge
- `network_trace` — upstream/downstream trace from an edge
- `network_strahler` — Strahler stream order
- `network_confluences` — points with in-degree ≥ 2

## Conventions

- All paths must resolve under `GIS_HYDRO_WORKSPACES` (semicolon- or
  comma-separated list of allowed roots; mirrors `GDAL_MCP_WORKSPACES`).
- Raster outputs preserve input CRS, transform, and nodata.
- Vector outputs default to GeoPackage.
- D8 encoding is ESRI standard:

  ```
  32  64  128
  16   0    1
   8   4    2
  ```

  `0` = unresolved sink; NoData stays NoData.

## Run

```
uvx --from ./gis-hydro-mcp gis-hydro --transport stdio
```

with `GIS_HYDRO_WORKSPACES` pointing at the data root.

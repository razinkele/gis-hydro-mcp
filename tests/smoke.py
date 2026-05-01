"""Smoke test: exercise topo + flow + network modules on synthetic data.

Avoids importing rasterio (slow/hung in this env). Validates the actual
algorithm modules end-to-end.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(ROOT))

# ---- topo + flow on a synthetic DEM ----
from gis_hydro_mcp import topo, flow  # noqa: E402

# Conical hill 50x50, summit at center, gentle slope
y, x = np.meshgrid(np.arange(50), np.arange(50), indexing="ij")
dem = 100.0 - np.hypot(y - 25, x - 25) * 1.5
dem[0, 0] = np.nan  # introduce a NoData cell

slope_deg = topo.slope(dem, dx=10.0, dy=10.0)
asp = topo.aspect(dem, dx=10.0, dy=10.0)
hs = topo.hillshade(dem, dx=10.0, dy=10.0)
print(f"slope:  min={np.nanmin(slope_deg):.2f}  max={np.nanmax(slope_deg):.2f} deg")
print(f"aspect: min={np.nanmin(asp):.2f}  max={np.nanmax(asp):.2f}")
print(f"hillshade: min={np.nanmin(hs):.0f}  max={np.nanmax(hs):.0f}")

filled = topo.fill_sinks(dem)
print(f"fill_sinks: cells_raised={int(np.nansum(filled - dem > 1e-9))}")

fd = flow.flow_direction(filled, dx=10.0, dy=10.0)
unique_codes = sorted(set(np.unique(fd).tolist()))
print(f"flow_direction unique codes: {unique_codes}")

acc = flow.flow_accumulation(fd)
print(
    f"flow_accumulation: max={acc.accumulation.max():.0f}  sinks={acc.sinks}  "
    f"nodata={acc.nodata_cells}  cycles={acc.cycles_detected}"
)

# Watershed: pick the cell with max accumulation as outlet
r, c = np.unravel_index(np.nanargmax(acc.accumulation), acc.accumulation.shape)
ws = flow.watershed(fd, int(r), int(c))
print(f"watershed at ({r},{c}): area_cells={int(ws.sum())}")

# ---- network on a tiny synthetic line layer ----
import geopandas as gpd  # noqa: E402
from shapely.geometry import LineString  # noqa: E402

from gis_hydro_mcp import network  # noqa: E402

# Two tributaries joining a mainstem at an interior junction
mainstem = LineString([(0, 0), (10, 0), (20, 0)])  # west→east
trib_a = LineString([(10, 5), (10, 0.05)])         # north→ near (10,0)
trib_b = LineString([(15, -5), (15, 0)])           # south→ near (15,0); needs noding
gdf = gpd.GeoDataFrame({"name": ["main", "tribA", "tribB"]},
                        geometry=[mainstem, trib_a, trib_b],
                        crs="EPSG:3346")

res = network.build_network(gdf, snap_tolerance=0.5)
print("network diagnostics:", {k: v for k, v in res.diagnostics.items()})
print(f"  edges: {len(res.edges)}  nodes: {len(res.nodes)}")

cc = network.connected_components(res.edges)
print(f"  components: {cc['component_id'].nunique()}")

confl = network.confluences(res.edges, res.nodes)
print(f"  confluences: {len(confl)}")

st = network.strahler(res.edges)
print(f"  strahler max order: {int(st['strahler'].max())}")

print("\nALL SMOKE TESTS PASSED")

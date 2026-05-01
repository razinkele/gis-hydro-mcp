"""Vector stream-network topology.

Builds a directed graph from a line layer, including:

* endpoint snapping within tolerance,
* **interior junction noding** (split a line where another line's endpoint
  touches its interior within tolerance),
* optional DEM-based direction inference with confidence diagnostics,
* connected components, upstream/downstream tracing, Strahler order,
  confluence detection.

Returns a GeoDataFrame of edges and nodes plus a diagnostics dict. Designed
to operate without any MCP / I/O dependencies — pass in a GeoDataFrame, get
GeoDataFrames back. The MCP server layer wraps this with file I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import geopandas as gpd
import networkx as nx
import numpy as np
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import linemerge, snap, split, unary_union
from shapely.strtree import STRtree


@dataclass
class NetworkBuildResult:
    nodes: gpd.GeoDataFrame
    edges: gpd.GeoDataFrame
    graph: nx.MultiDiGraph
    diagnostics: dict = field(default_factory=dict)


def _explode_lines(lines: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Explode MultiLineStrings to LineStrings; drop empties."""
    exploded = lines.explode(index_parts=False, ignore_index=True)
    exploded = exploded[~exploded.geometry.is_empty & exploded.geometry.notna()]
    exploded = exploded[exploded.geometry.geom_type == "LineString"].reset_index(drop=True)
    return exploded


def _node_interior_junctions(
    lines: gpd.GeoDataFrame, tolerance: float
) -> tuple[gpd.GeoDataFrame, int]:
    """For every line endpoint that lies within ``tolerance`` of another line's
    interior (not its endpoints), split that other line at the projected point.
    """
    geoms = list(lines.geometry)
    tree = STRtree(geoms)
    splits: dict[int, list[Point]] = {}
    junction_count = 0

    endpoints: list[tuple[int, Point]] = []
    for i, g in enumerate(geoms):
        endpoints.append((i, Point(g.coords[0])))
        endpoints.append((i, Point(g.coords[-1])))

    for src_idx, ep in endpoints:
        candidate_idx = tree.query(ep.buffer(tolerance))
        for j in candidate_idx:
            j = int(j)
            if j == src_idx:
                continue
            target = geoms[j]
            d = target.distance(ep)
            if d > tolerance:
                continue
            # Skip if endpoint coincides with target's endpoints (within tol).
            if (
                Point(target.coords[0]).distance(ep) <= tolerance
                or Point(target.coords[-1]).distance(ep) <= tolerance
            ):
                continue
            proj_dist = target.project(ep)
            split_pt = target.interpolate(proj_dist)
            splits.setdefault(j, []).append(split_pt)

    if not splits:
        return lines.copy(), 0

    new_rows = []
    keep_mask = np.ones(len(geoms), dtype=bool)
    for j, pts in splits.items():
        keep_mask[j] = False
        target = geoms[j]
        # Use a small MultiPoint to split; sort by distance along the line.
        pts_sorted = sorted(pts, key=lambda p: target.project(p))
        try:
            pieces = split(snap(target, unary_union(pts_sorted), tolerance), unary_union(pts_sorted))
        except Exception:
            pieces = [target]
        attrs = lines.iloc[j].drop(labels="geometry").to_dict()
        for piece in getattr(pieces, "geoms", [pieces]):
            if piece.is_empty or piece.geom_type != "LineString":
                continue
            row = dict(attrs)
            row["geometry"] = piece
            new_rows.append(row)
            junction_count += 1

    surviving = lines[keep_mask].to_dict("records")
    surviving.extend(new_rows)
    out = gpd.GeoDataFrame(surviving, geometry="geometry", crs=lines.crs)
    return out, len(splits)


def _snap_endpoints(lines: gpd.GeoDataFrame, tolerance: float) -> tuple[gpd.GeoDataFrame, dict]:
    """Cluster endpoints within tolerance to a common coordinate.

    Replaces the first/last vertex of each LineString with the cluster centroid.
    Returns the modified gdf plus per-vertex snap counts.
    """
    coords = []
    for g in lines.geometry:
        coords.append(g.coords[0])
        coords.append(g.coords[-1])
    coords_arr = np.array(coords)

    # Greedy clustering via spatial hashing on a tolerance grid.
    if tolerance <= 0:
        return lines.copy(), {"snapped_pairs": 0}

    keys = np.floor(coords_arr / tolerance).astype(np.int64)
    cluster_for: dict[tuple[int, int], int] = {}
    cluster_centroid: dict[int, list[float]] = {}
    cluster_count: dict[int, int] = {}
    point_cluster = np.empty(len(coords_arr), dtype=np.int64)
    next_cid = 0

    for i, (kx, ky) in enumerate(map(tuple, keys)):
        cid = None
        # check this cell + 8 neighbours
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (kx + dx, ky + dy)
                if key in cluster_for:
                    candidate = cluster_for[key]
                    cx, cy = cluster_centroid[candidate]
                    if (cx - coords_arr[i, 0]) ** 2 + (cy - coords_arr[i, 1]) ** 2 <= tolerance ** 2:
                        cid = candidate
                        break
            if cid is not None:
                break
        if cid is None:
            cid = next_cid
            next_cid += 1
            cluster_for[(kx, ky)] = cid
            cluster_centroid[cid] = [coords_arr[i, 0], coords_arr[i, 1]]
            cluster_count[cid] = 1
        else:
            n = cluster_count[cid]
            cx, cy = cluster_centroid[cid]
            cluster_centroid[cid] = [
                (cx * n + coords_arr[i, 0]) / (n + 1),
                (cy * n + coords_arr[i, 1]) / (n + 1),
            ]
            cluster_count[cid] = n + 1
            cluster_for[(kx, ky)] = cid
        point_cluster[i] = cid

    new_geoms = []
    snapped_pairs = 0
    for k, g in enumerate(lines.geometry):
        coords_list = list(g.coords)
        cid_a = point_cluster[2 * k]
        cid_b = point_cluster[2 * k + 1]
        new_a = tuple(cluster_centroid[cid_a])
        new_b = tuple(cluster_centroid[cid_b])
        if new_a != tuple(coords_list[0]):
            coords_list[0] = new_a
            snapped_pairs += 1
        if new_b != tuple(coords_list[-1]):
            coords_list[-1] = new_b
            snapped_pairs += 1
        new_geoms.append(LineString(coords_list))

    out = lines.copy()
    out["geometry"] = new_geoms
    return out, {"snapped_pairs": snapped_pairs, "node_clusters": next_cid}


def _sample_dem(coords: Iterable[tuple[float, float]], dem_path: str | None) -> list[float]:
    if dem_path is None:
        return [float("nan")] * len(list(coords))
    import rasterio  # local import: optional dep

    coords = list(coords)
    with rasterio.open(dem_path) as ds:
        return [float(v[0]) if v[0] is not None else float("nan") for v in ds.sample(coords)]


def build_network(
    lines: gpd.GeoDataFrame,
    *,
    snap_tolerance: float,
    dem_path: str | None = None,
    min_dz: float = 0.0,
) -> NetworkBuildResult:
    """Build a directed stream graph from a line layer."""
    if lines.crs is None:
        raise ValueError("input lines must have a CRS")

    diagnostics: dict = {"input_features": len(lines)}

    work = _explode_lines(lines)
    diagnostics["after_explode"] = len(work)

    work, n_split = _node_interior_junctions(work, snap_tolerance)
    diagnostics["interior_junctions_created"] = n_split
    diagnostics["after_noding"] = len(work)

    work, snap_diag = _snap_endpoints(work, snap_tolerance)
    diagnostics.update({f"snap_{k}": v for k, v in snap_diag.items()})

    # Drop near-zero-length edges produced by snapping
    lengths = work.geometry.length
    keep = lengths > snap_tolerance / 10.0
    diagnostics["zero_length_edges_dropped"] = int((~keep).sum())
    work = work[keep].reset_index(drop=True)

    # Build node table from unique endpoint coordinates (post-snap they match exactly).
    coords_all: list[tuple[float, float]] = []
    edge_endpoints: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for g in work.geometry:
        a = tuple(g.coords[0])
        b = tuple(g.coords[-1])
        coords_all.append(a); coords_all.append(b)
        edge_endpoints.append((a, b))

    unique_coords = list({c: None for c in coords_all}.keys())
    coord_to_node = {c: i for i, c in enumerate(unique_coords)}

    # DEM elevations at nodes
    z_values = _sample_dem(unique_coords, dem_path)
    nodes_gdf = gpd.GeoDataFrame(
        {
            "node_id": list(range(len(unique_coords))),
            "z": z_values,
            "geometry": [Point(c) for c in unique_coords],
        },
        crs=lines.crs,
    )

    # Build edges with direction inference
    g_nx = nx.MultiDiGraph()
    for nid, z in zip(nodes_gdf["node_id"], nodes_gdf["z"]):
        g_nx.add_node(int(nid), z=float(z) if z == z else float("nan"))

    edge_records = []
    ambiguous = 0
    reversed_edges = 0
    for idx, (a, b) in enumerate(edge_endpoints):
        na = coord_to_node[a]; nb = coord_to_node[b]
        za = nodes_gdf.iloc[na]["z"]; zb = nodes_gdf.iloc[nb]["z"]
        direction_source = "digitized"
        reversed_flag = False
        if dem_path is not None and not (np.isnan(za) or np.isnan(zb)):
            dz = float(za) - float(zb)
            if abs(dz) < min_dz:
                direction_source = "ambiguous"
                ambiguous += 1
            elif dz < 0:
                # start lower than end → flow goes b→a, reverse
                na, nb = nb, na
                direction_source = "dem"
                reversed_flag = True
                reversed_edges += 1
            else:
                direction_source = "dem"
        attrs = work.iloc[idx].drop(labels="geometry").to_dict()
        attrs.update(
            {
                "edge_id": idx,
                "from_node": int(na),
                "to_node": int(nb),
                "length": float(work.iloc[idx].geometry.length),
                "direction_source": direction_source,
                "reversed_from_input": bool(reversed_flag),
                "z_start": float(nodes_gdf.iloc[na]["z"]),
                "z_end": float(nodes_gdf.iloc[nb]["z"]),
                "geometry": work.iloc[idx].geometry,
            }
        )
        edge_records.append(attrs)
        g_nx.add_edge(int(na), int(nb), key=int(idx), edge_id=int(idx))

    edges_gdf = gpd.GeoDataFrame(edge_records, geometry="geometry", crs=lines.crs)

    # Diagnostics
    diagnostics["nodes"] = len(nodes_gdf)
    diagnostics["edges"] = len(edges_gdf)
    diagnostics["dangling_nodes"] = int(((edges_gdf["from_node"].value_counts().add(
        edges_gdf["to_node"].value_counts(), fill_value=0
    ) == 1).sum()))
    diagnostics["self_loops"] = int((edges_gdf["from_node"] == edges_gdf["to_node"]).sum())
    diagnostics["ambiguous_direction_edges"] = ambiguous
    diagnostics["reversed_edges"] = reversed_edges
    diagnostics["weakly_connected_components"] = nx.number_weakly_connected_components(g_nx)

    return NetworkBuildResult(nodes=nodes_gdf, edges=edges_gdf, graph=g_nx, diagnostics=diagnostics)


# ---------------------------------------------------------------------------
# Topology helpers operating on (nodes, edges) GeoDataFrames.
# ---------------------------------------------------------------------------


def _graph_from_edges(edges: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for row in edges.itertuples(index=False):
        g.add_edge(int(row.from_node), int(row.to_node), key=int(row.edge_id), edge_id=int(row.edge_id))
    return g


def connected_components(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = _graph_from_edges(edges).to_undirected()
    comp_for_node: dict[int, int] = {}
    for cid, comp in enumerate(nx.connected_components(g)):
        for n in comp:
            comp_for_node[n] = cid
    out = edges.copy()
    out["component_id"] = out["from_node"].map(comp_for_node).astype("int64")
    return out


def trace(
    edges: gpd.GeoDataFrame,
    start_edge_id: int,
    direction: Literal["upstream", "downstream"] = "downstream",
) -> gpd.GeoDataFrame:
    g = _graph_from_edges(edges)
    row = edges.loc[edges["edge_id"] == start_edge_id].iloc[0]
    start_node = int(row["to_node"]) if direction == "downstream" else int(row["from_node"])
    if direction == "downstream":
        reachable_nodes = nx.descendants(g, start_node) | {start_node}
    else:
        reachable_nodes = nx.ancestors(g, start_node) | {start_node}
    mask = edges["from_node"].isin(reachable_nodes) & edges["to_node"].isin(reachable_nodes)
    return edges[mask].copy()


def strahler(edges: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Compute Strahler stream order per edge (DAG required)."""
    g = _graph_from_edges(edges)
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("strahler requires a DAG; resolve cycles first")
    order_for_edge: dict[int, int] = {}
    # process nodes in topological order
    for node in nx.topological_sort(g):
        in_edges = list(g.in_edges(node, keys=True))
        if not in_edges:
            continue  # source node, contributes order 1 to its out-edges below
        # actually order is computed on out-edges; but in trees, an edge's
        # order is determined by the upstream sub-tree feeding its from_node.
    # Simpler approach: recursive memoised
    edge_lookup = {int(r.edge_id): (int(r.from_node), int(r.to_node)) for r in edges.itertuples(index=False)}
    upstream_edges_at_node: dict[int, list[int]] = {}
    for eid, (a, b) in edge_lookup.items():
        upstream_edges_at_node.setdefault(b, []).append(eid)

    memo: dict[int, int] = {}

    def order_of(eid: int) -> int:
        if eid in memo:
            return memo[eid]
        a, _b = edge_lookup[eid]
        ups = upstream_edges_at_node.get(a, [])
        if not ups:
            memo[eid] = 1
            return 1
        orders = sorted((order_of(u) for u in ups), reverse=True)
        if len(orders) == 1 or orders[0] > orders[1]:
            o = orders[0]
        else:
            o = orders[0] + 1
        memo[eid] = o
        return o

    out = edges.copy()
    out["strahler"] = out["edge_id"].map(lambda e: order_of(int(e))).astype("int64")
    return out


def confluences(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    in_counts = edges["to_node"].value_counts()
    confl_node_ids = in_counts[in_counts >= 2].index.tolist()
    out = nodes[nodes["node_id"].isin(confl_node_ids)].copy()
    out["in_degree"] = out["node_id"].map(in_counts).astype("int64")
    return out

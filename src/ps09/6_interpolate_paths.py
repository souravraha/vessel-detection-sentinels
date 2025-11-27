#!/usr/bin/env python3
"""
Rule-based AIS imputation with Natural Earth 10m coastline land-mask.

Core features:
- Uses a coastline mask to avoid routing imputed paths across land.
- Fills missing lat/lon using:
  • Constant-velocity geodesic interpolation across gaps with known endpoints.
  • Water-only path bridging via grid A* if the straight line would cross land.
  • Dead-reckoning from SOG/COG when only one endpoint is known.
- Fills missing SOG/COG from positions via geodesic differencing with smoothing.
- Detects anchoring windows (near-zero SOG and tight position cluster).
- Emits an impute_flag per row for auditability.

Assumptions:
- path_id groups independent trajectories.
- point_id is a per-path time-sorted integer starting at 1.
- time_stamp is time-of-day "HH:MM" or "HH:MM:SS" (midnight wrap handled per path).

Note:
- The input CSV may contain unnamed columns (e.g., 'Unnamed: 8'); these are dropped on read
  and will not be reproduced in the output.

Dependencies: pandas, numpy, shapely>=2, fiona, pyproj
"""
from __future__ import annotations

import math
import argparse
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import fiona
from shapely.geometry import LineString, shape
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree
from pyproj import Geod

# -----------------------------
# Configuration and parameters
# -----------------------------

# Geodesic calculator (WGS84 ellipsoid)
GEOD = Geod(ellps="WGS84")

# Rule thresholds (tune to your fleet)
SPEED_CAP_KTS = 45.0          # plausible vessel speed cap (knots)
LOW_SPEED_KTS = 1.0           # below this, COG is unreliable
SHORT_GAP_SECONDS = 30 * 60   # short-gap threshold (seconds) for interpolation
ANCHOR_WINDOW_MIN = 15        # minutes window for anchoring detection
ANCHOR_RADIUS_M = 200         # max radius for anchor cluster (meters)
ANCHOR_SOG_KTS = 0.8          # SOG threshold to consider anchor (knots)

# Water-path A* grid parameters
GRID_STEP_M = 800.0           # grid step (meters)
GRID_MARGIN_M = 2000.0        # margin around bbox for routing (meters)
MAX_ASTAR_NODES = 200000      # safety cap to avoid runaway

# Smoothing window sizes
SOG_SMOOTH_WINDOW = 5         # rolling window (points) for SOG smoothing
COG_SMOOTH_WINDOW = 7         # rolling window for heading vector smoothing

# Confidence flags
CONF_OBS = 0                    # observed
CONF_INTERP_CV = 1              # constant-velocity interpolation
CONF_DEADRECKON = 3             # dead-reckoned from SOG/COG
CONF_WATER_BRIDGE = 4           # water-only path bridge (A*)
CONF_EXTRAP_LOWCONF = 5         # extrapolated / low confidence
CONF_ANCHORED = 6               # anchored

# -----------------------------
# Utilities: geodesy and angles
# -----------------------------

def kts_to_mps(kts: float) -> float:
    return kts * 0.514444

def mps_to_kts(mps: float) -> float:
    return mps / 0.514444

def geodesic_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    _, _, dist_m = GEOD.inv(lon1, lat1, lon2, lat2)
    return dist_m

def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    fwd_az, _, _ = GEOD.inv(lon1, lat1, lon2, lat2)
    return (fwd_az + 360.0) % 360.0

def forward_point(lat: float, lon: float, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    lon2, lat2, _ = GEOD.fwd(lon, lat, bearing_deg, distance_m)
    return lat2, lon2

def wrap_angle_deg(angle: float) -> float:
    return (angle % 360.0 + 360.0) % 360.0

def avg_heading_deg(angles_deg: np.ndarray) -> float:
    valid = ~np.isnan(angles_deg)
    if not np.any(valid):
        return np.nan
    ang_rad = np.deg2rad(angles_deg[valid])
    x = np.cos(ang_rad).mean()
    y = np.sin(ang_rad).mean()
    if x == 0 and y == 0:
        return np.nan
    return wrap_angle_deg(np.rad2deg(math.atan2(y, x)))

# -----------------------------
# Land-mask: coastline handling
# -----------------------------

@dataclass
class CoastlineIndex:
    """Spatial index over coastline LineStrings for fast intersection tests."""
    lines: List[LineString]
    tree: STRtree

    @staticmethod
    def _to_lines(geom: BaseGeometry) -> List[LineString]:
        """Normalize an input geometry into a list of non-empty LineStrings."""
        out: List[LineString] = []
        if geom is None or not isinstance(geom, BaseGeometry) or geom.is_empty:
            return out
        gt = geom.geom_type
        if gt == "LineString":
            out.append(geom)
        elif gt == "MultiLineString":
            out.extend([ls for ls in geom.geoms if not ls.is_empty])
        elif gt == "Polygon":
            b = geom.boundary
            if b.geom_type == "LinearRing":
                out.append(LineString(b.coords))
            elif b.geom_type == "MultiLineString":
                out.extend([ls for ls in b.geoms if not ls.is_empty])
        elif gt == "MultiPolygon":
            for pg in geom.geoms:
                if not pg.is_empty:
                    b = pg.boundary
                    if b.geom_type == "LinearRing":
                        out.append(LineString(b.coords))
                    elif b.geom_type == "MultiLineString":
                        out.extend([ls for ls in b.geoms if not ls.is_empty])
        return out

    @classmethod
    def from_shapefile(cls, shp_path: str) -> "CoastlineIndex":
        logging.info(f"Loading coastline shapefile: {shp_path}")
        lines: List[LineString] = []
        with fiona.open(shp_path, "r") as src:
            for feat in src:
                geom = shape(feat["geometry"]) if feat and feat.get("geometry") else None
                for ls in cls._to_lines(geom):
                    if isinstance(ls, BaseGeometry) and not ls.is_empty:
                        lines.append(ls)
        logging.info(f"Loaded {len(lines)} coastline line segments (non-empty).")
        tree = STRtree(lines)
        return cls(lines=lines, tree=tree)

    def segment_crosses_coastline(self, lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
        """True if the segment between (lat1,lon1)->(lat2,lon2) intersects any coastline polyline(s)."""
        seg = LineString([(lon1, lat1), (lon2, lat2)])
        if seg.is_empty:
            return False
        try:
            candidates = self.tree.query(seg, predicate="intersects")
        except TypeError:
            candidates = self.tree.query(seg)
        for cand in candidates:
            if not isinstance(cand, BaseGeometry) or cand.is_empty:
                continue
            try:
                if seg.intersects(cand):
                    return True
            except TypeError:
                continue
        return False

# -----------------------------
# A*: water-only path bridging
# -----------------------------

@dataclass(frozen=True)
class Node:
    """Grid node in lat/lon degrees."""
    lat: float
    lon: float

def approx_deg_per_meter_lat(_: float) -> float:
    return 1.0 / 111320.0

def approx_deg_per_meter_lon(lat: float) -> float:
    c = max(1e-6, math.cos(math.radians(lat)))
    return 1.0 / (40075000.0 * c / 360.0)

def build_grid_bbox(lat1: float, lon1: float, lat2: float, lon2: float, margin_m: float) -> Tuple[float, float, float, float]:
    min_lat = min(lat1, lat2)
    max_lat = max(lat1, lat2)
    min_lon = min(lon1, lon2)
    max_lon = max(lon1, lon2)
    mid_lat = 0.5 * (lat1 + lat2)
    dlat = margin_m * approx_deg_per_meter_lat(mid_lat)
    dlon = margin_m * approx_deg_per_meter_lon(mid_lat)
    return min_lat - dlat, min_lon - dlon, max_lat + dlat, max_lon + dlon

def neighbors(node: Node, step_m: float) -> List[Node]:
    lat = node.lat
    dlat = step_m * approx_deg_per_meter_lat(lat)
    dlon = step_m * approx_deg_per_meter_lon(lat)
    nbrs = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di == 0 and dj == 0:
                continue
            nbrs.append(Node(lat=lat + di * dlat, lon=node.lon + dj * dlon))
    return nbrs

def is_in_bbox(node: Node, bbox: Tuple[float, float, float, float]) -> bool:
    min_lat, min_lon, max_lat, max_lon = bbox
    return (min_lat <= node.lat <= max_lat) and (min_lon <= node.lon <= max_lon)

def heuristic_distance(a: Node, b: Node) -> float:
    return geodesic_distance_m(a.lat, a.lon, b.lat, b.lon)

def reconstruct_path(came_from: Dict[Node, Node], start: Node, goal: Node) -> List[Node]:
    path = [goal]
    cur = goal
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path

def astar_water_path(start: Tuple[float, float], goal: Tuple[float, float],
                     coastline: CoastlineIndex,
                     step_m: float,
                     margin_m: float,
                     max_nodes: int = MAX_ASTAR_NODES) -> Optional[List[Tuple[float, float]]]:
    """
    A* on a lat/lon grid where edges are disallowed if they intersect coastline.
    Returns a sequence of (lat, lon) nodes from start to goal, or None if not found.
    """
    start_node = Node(lat=start[0], lon=start[1])
    goal_node = Node(lat=goal[0], lon=goal[1])
    bbox = build_grid_bbox(start_node.lat, start_node.lon, goal_node.lat, goal_node.lon, margin_m)

    import heapq
    open_heap = []
    heapq.heappush(open_heap, (0.0, start_node))
    came_from: Dict[Node, Node] = {}
    g_score: Dict[Node, float] = {start_node: 0.0}
    visited = 0

    while open_heap:
        _, current = heapq.heappop(open_heap)
        visited += 1
        if visited > max_nodes:
            logging.warning("A* node expansion exceeded cap; aborting search.")
            return None

        if heuristic_distance(current, goal_node) <= step_m:
            if not coastline.segment_crosses_coastline(current.lat, current.lon, goal_node.lat, goal_node.lon):
                came_from[goal_node] = current
                path = reconstruct_path(came_from, start_node, goal_node)
                return [(n.lat, n.lon) for n in path]

        for nbr in neighbors(current, step_m):
            if not is_in_bbox(nbr, bbox):
                continue
            if coastline.segment_crosses_coastline(current.lat, current.lon, nbr.lat, nbr.lon):
                continue
            tentative_g = g_score[current] + heuristic_distance(current, nbr)
            if tentative_g < g_score.get(nbr, math.inf):
                came_from[nbr] = current
                g_score[nbr] = tentative_g
                f_score = tentative_g + heuristic_distance(nbr, goal_node)
                import heapq as _hq
                _hq.heappush(open_heap, (f_score, nbr))
    logging.warning("A* failed to find a water-only path.")
    return None

# -----------------------------
# Time handling
# -----------------------------

def _parse_hms_to_seconds(x: str) -> float:
    x = str(x).strip()
    parts = x.split(":")
    if len(parts) not in (2, 3):
        return float("nan")
    try:
        h = int(parts[0]); m = int(parts[1]); s = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        return float("nan")
    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
        return float("nan")
    return float(h * 3600 + m * 60 + s)

def add_t_seconds_from_point_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 't_seconds' per path_id from time_stamp strings 'HH:MM[:SS]',
    ordered by point_id (assumed sorted within each path).
    Handles midnight wraps by adding 86400 at each decrease in time-of-day.
    """
    df = df.copy()
    if "path_id" not in df.columns or "point_id" not in df.columns:
        raise ValueError("Expected 'path_id' and 'point_id' in columns.")
    tsec = np.full(len(df), np.nan, dtype=float)
    for pid, grp in df.groupby("path_id", sort=False):
        g = grp.sort_values("point_id")
        secs = g["time_stamp"].apply(_parse_hms_to_seconds).astype(float).values
        if np.isnan(secs).any():
            raise ValueError(f"path_id={pid}: time_stamp must be HH:MM or HH:MM:SS")
        offset = 0.0
        for i in range(len(secs)):
            if i > 0 and secs[i] < secs[i-1]:
                offset += 86400.0
            secs[i] += offset
        tsec[df.index.get_indexer(g.index)] = secs
    df["t_seconds"] = tsec
    return df

# -----------------------------
# AIS processing core
# -----------------------------

def compute_sog_cog_from_positions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SOG and COG for rows with positions present, using neighbor differencing.
    Leaves existing SOG/COG if present; fills missing entries, then smooths.
    Uses df['t_seconds'] for time deltas.
    """
    lat = df["point_latitude"].values
    lon = df["point_longitude"].values
    t = df["t_seconds"].values.astype(float)
    n = len(df)

    sog_est = np.full(n, np.nan)
    cog_est = np.full(n, np.nan)

    for i in range(n):
        if i > 0 and i < n - 1 and not np.isnan(lat[i-1]) and not np.isnan(lat[i+1]) and not np.isnan(lat[i]):
            dt = (t[i+1] - t[i-1])
            if dt > 0:
                d1 = geodesic_distance_m(lat[i-1], lon[i-1], lat[i], lon[i])
                d2 = geodesic_distance_m(lat[i], lon[i], lat[i+1], lon[i+1])
                sog_est[i] = mps_to_kts((d1 + d2) / dt)
                cog_est[i] = initial_bearing_deg(lat[i-1], lon[i-1], lat[i+1], lon[i+1])
        else:
            if i > 0 and not np.isnan(lat[i-1]) and not np.isnan(lat[i]):
                dt = (t[i] - t[i-1])
                if dt > 0:
                    d = geodesic_distance_m(lat[i-1], lon[i-1], lat[i], lon[i])
                    sog_est[i] = mps_to_kts(d / dt)
                    cog_est[i] = initial_bearing_deg(lat[i-1], lon[i-1], lat[i], lon[i])
            elif i < n - 1 and not np.isnan(lat[i+1]) and not np.isnan(lat[i]):
                dt = (t[i+1] - t[i])
                if dt > 0:
                    d = geodesic_distance_m(lat[i], lon[i], lat[i+1], lon[i+1])
                    sog_est[i] = mps_to_kts(d / dt)
                    cog_est[i] = initial_bearing_deg(lat[i], lon[i], lat[i+1], lon[i+1])

    sog_out = np.where(np.isnan(df["speed_on_ground"].values), sog_est, df["speed_on_ground"].values)
    cog_out = np.where(np.isnan(df["course_on_ground"].values), cog_est, df["course_on_ground"].values)

    sog_series = pd.Series(sog_out)
    sog_out = sog_series.rolling(SOG_SMOOTH_WINDOW, min_periods=1, center=True).median().rolling(
        SOG_SMOOTH_WINDOW, min_periods=1, center=True).mean().values

    cog_out_sm = np.full(n, np.nan)
    for i in range(n):
        lo = max(0, i - COG_SMOOTH_WINDOW // 2)
        hi = min(n, i + COG_SMOOTH_WINDOW // 2 + 1)
        window = np.array(cog_out[lo:hi], dtype=float)
        sogw = np.array(sog_out[lo:hi], dtype=float)
        window = np.where(sogw < LOW_SPEED_KTS, np.nan, window)
        cog_out_sm[i] = avg_heading_deg(window)

    df["speed_on_ground"] = sog_out
    df["course_on_ground"] = cog_out_sm
    return df

def _find_missing_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return list of (start_idx, end_idx) inclusive runs where mask is True."""
    runs: List[Tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while j < n and mask[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs

def interpolate_cv_between(df: pd.DataFrame, idx_start: int, idx_end: int,
                           coastline: CoastlineIndex) -> Tuple[pd.DataFrame, int]:
    """
    Constant-velocity geodesic interpolation between known endpoints [idx_start, idx_end].
    If the straight line crosses coastline, uses water-only A* bridge if possible.
    Sets impute_flag for interior points.
    """
    lat1, lon1 = df.at[idx_start, "point_latitude"], df.at[idx_start, "point_longitude"]
    lat2, lon2 = df.at[idx_end, "point_latitude"], df.at[idx_end, "point_longitude"]
    if np.isnan(lat1) or np.isnan(lat2):
        return df, CONF_EXTRAP_LOWCONF

    crosses = coastline.segment_crosses_coastline(lat1, lon1, lat2, lon2)

    if not crosses:
        latlon_path = [(lat1, lon1), (lat2, lon2)]
        confidence = CONF_INTERP_CV
    else:
        logging.info(f"Straight line crosses coastline between rows {idx_start}->{idx_end}; running A* water-bridge.")
        path_nodes = astar_water_path((lat1, lon1), (lat2, lon2), coastline, GRID_STEP_M, GRID_MARGIN_M)
        if path_nodes is None or len(path_nodes) < 2:
            logging.warning("A* failed; falling back to straight geodesic interpolation with low confidence.")
            latlon_path = [(lat1, lon1), (lat2, lon2)]
            confidence = CONF_EXTRAP_LOWCONF
        else:
            latlon_path = path_nodes
            confidence = CONF_WATER_BRIDGE

    t_start = float(df.at[idx_start, "t_seconds"])
    t_end = float(df.at[idx_end, "t_seconds"])
    total_dt = t_end - t_start
    if total_dt <= 0:
        return df, CONF_EXTRAP_LOWCONF

    cum_dists = [0.0]
    for i in range(1, len(latlon_path)):
        d = geodesic_distance_m(latlon_path[i-1][0], latlon_path[i-1][1], latlon_path[i][0], latlon_path[i][1])
        cum_dists.append(cum_dists[-1] + d)

    for idx in range(idx_start + 1, idx_end):
        ti = float(df.at[idx, "t_seconds"])
        frac = (ti - t_start) / total_dt
        target_dist = frac * cum_dists[-1]
        seg_idx = np.searchsorted(cum_dists, target_dist, side="right") - 1
        seg_idx = max(0, min(seg_idx, len(latlon_path) - 2))
        d_into = target_dist - cum_dists[seg_idx]
        seg_total = cum_dists[seg_idx + 1] - cum_dists[seg_idx]
        if seg_total <= 0:
            lat_i, lon_i = latlon_path[seg_idx]
        else:
            lat_a, lon_a = latlon_path[seg_idx]
            lat_b, lon_b = latlon_path[seg_idx + 1]
            bearing = initial_bearing_deg(lat_a, lon_a, lat_b, lon_b)
            lat_i, lon_i = forward_point(lat_a, lon_a, bearing, d_into)
        df.at[idx, "point_latitude"] = lat_i
        df.at[idx, "point_longitude"] = lon_i
        df.at[idx, "impute_flag"] = confidence

    segment = df.loc[idx_start:idx_end].copy()
    segment = compute_sog_cog_from_positions(segment)
    df.loc[idx_start:idx_end, ["speed_on_ground", "course_on_ground"]] = segment[["speed_on_ground", "course_on_ground"]]
    return df, confidence

def dead_reckon(df: pd.DataFrame, idx_from: int, idx_to: int) -> pd.DataFrame:
    """
    Fill positions between idx_from and idx_to (exclusive) by dead-reckoning using SOG/COG
    where available, starting from the known endpoint closest to the fill direction.
    """
    use_forward = not np.isnan(df.at[idx_from, "point_latitude"])
    use_backward = not np.isnan(df.at[idx_to, "point_latitude"]) and not use_forward

    if use_forward:
        lat_cur = df.at[idx_from, "point_latitude"]
        lon_cur = df.at[idx_from, "point_longitude"]
        t_prev = float(df.at[idx_from, "t_seconds"])
        for idx in range(idx_from + 1, idx_to):
            t = float(df.at[idx, "t_seconds"])
            dt = t - t_prev
            sog = df.at[idx, "speed_on_ground"]
            cog = df.at[idx, "course_on_ground"]
            if dt <= 0 or np.isnan(sog) or sog < LOW_SPEED_KTS or np.isnan(cog):
                df.at[idx, "point_latitude"] = lat_cur
                df.at[idx, "point_longitude"] = lon_cur
                df.at[idx, "impute_flag"] = CONF_EXTRAP_LOWCONF
            else:
                dist = kts_to_mps(min(sog, SPEED_CAP_KTS)) * dt
                lat_cur, lon_cur = forward_point(lat_cur, lon_cur, cog, dist)
                df.at[idx, "point_latitude"] = lat_cur
                df.at[idx, "point_longitude"] = lon_cur
                df.at[idx, "impute_flag"] = CONF_DEADRECKON
            t_prev = t
    elif use_backward:
        lat_cur = df.at[idx_to, "point_latitude"]
        lon_cur = df.at[idx_to, "point_longitude"]
        t_next = float(df.at[idx_to, "t_seconds"])
        for idx in range(idx_to - 1, idx_from, -1):
            t = float(df.at[idx, "t_seconds"])
            dt = t_next - t
            sog = df.at[idx, "speed_on_ground"]
            cog = df.at[idx, "course_on_ground"]
            if dt <= 0 or np.isnan(sog) or sog < LOW_SPEED_KTS or np.isnan(cog):
                df.at[idx, "point_latitude"] = lat_cur
                df.at[idx, "point_longitude"] = lon_cur
                df.at[idx, "impute_flag"] = CONF_EXTRAP_LOWCONF
            else:
                dist = kts_to_mps(min(sog, SPEED_CAP_KTS)) * dt
                rev_bearing = wrap_angle_deg(cog + 180.0)
                lat_cur, lon_cur = forward_point(lat_cur, lon_cur, rev_bearing, dist)
                df.at[idx, "point_latitude"] = lat_cur
                df.at[idx, "point_longitude"] = lon_cur
                df.at[idx, "impute_flag"] = CONF_DEADRECKON
            t_next = t
    else:
        for idx in range(idx_from + 1, idx_to):
            df.at[idx, "impute_flag"] = CONF_EXTRAP_LOWCONF
    return df

def detect_anchor_windows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mark rows likely at anchor: low SOG sustained and positions clustered in small radius.
    """
    times_sec = df["t_seconds"].values.astype(float)
    sog = df["speed_on_ground"].values
    lat = df["point_latitude"].values
    lon = df["point_longitude"].values
    n = len(df)

    window_sec = ANCHOR_WINDOW_MIN * 60
    flags = np.zeros(n, dtype=bool)
    for i in range(n):
        t0 = times_sec[i] - window_sec / 2.0
        t1 = times_sec[i] + window_sec / 2.0
        mask = (times_sec >= t0) & (times_sec <= t1)
        if mask.sum() < 5:
            continue
        sog_med = np.nanmedian(sog[mask])
        if sog_med > ANCHOR_SOG_KTS:
            continue
        latw = lat[mask]
        lonw = lon[mask]
        if np.any(np.isnan(latw)) or np.any(np.isnan(lonw)):
            continue
        latc = np.nanmean(latw)
        lonc = np.nanmean(lonw)
        max_r = 0.0
        for la, lo in zip(latw, lonw):
            max_r = max(max_r, geodesic_distance_m(latc, lonc, la, lo))
        if max_r <= ANCHOR_RADIUS_M:
            flags[i] = True

    df["anchored"] = flags
    df.loc[df["anchored"], "impute_flag"] = CONF_ANCHORED
    df.loc[df["anchored"], "course_on_ground"] = np.nan
    df.loc[df["anchored"] & df["speed_on_ground"].isna(), "speed_on_ground"] = 0.0
    return df

def process_path(df_path: pd.DataFrame, coastline: CoastlineIndex) -> pd.DataFrame:
    """
    Process one path_id group: sort by t_seconds, fill missing values with rule-based methods.
    """
    df = df_path.sort_values("t_seconds").reset_index(drop=True).copy()
    if "impute_flag" not in df.columns:
        df["impute_flag"] = np.nan
    obs_mask = df["point_latitude"].notna() & df["point_longitude"].notna()
    df.loc[obs_mask, "impute_flag"] = CONF_OBS

    df = compute_sog_cog_from_positions(df)

    mask = df["point_latitude"].isna().values | df["point_longitude"].isna().values
    runs = _find_missing_runs(mask)
    n = len(df)

    for a, b in runs:
        left = a - 1 if a - 1 >= 0 else None
        right = b + 1 if b + 1 < n else None

        if left is not None and right is not None:
            df, _ = interpolate_cv_between(df, left, right, coastline)
        elif left is not None and right is None:
            df = dead_reckon(df, left, b)
        elif left is None and right is not None:
            df = dead_reckon(df, a, right)
        else:
            df.loc[a:b, "impute_flag"] = CONF_EXTRAP_LOWCONF

    df = detect_anchor_windows(df)

    df.loc[df["speed_on_ground"] > SPEED_CAP_KTS, "speed_on_ground"] = SPEED_CAP_KTS
    low_speed_mask = df["speed_on_ground"] < LOW_SPEED_KTS
    df.loc[low_speed_mask, "course_on_ground"] = np.nan

    return df

# -----------------------------
# I/O and CLI
# -----------------------------

def process_csv(input_csv: str, output_csv: str, coastline_shp: str) -> None:
    logging.info(f"Loading AIS CSV: {input_csv}")
    df = pd.read_csv(input_csv)

    # Drop unnamed columns (e.g., 'Unnamed: 8', '', etc.) and do not reproduce them
    unnamed_cols = [c for c in df.columns if str(c).strip() == "" or str(c).lower().startswith("unnamed")]
    if unnamed_cols:
        logging.info(f"Dropping unnamed columns: {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    required = ["sl.no.", "time_stamp", "path_id", "point_id",
                "point_latitude", "point_longitude", "speed_on_ground", "course_on_ground"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = add_t_seconds_from_point_id(df)

    coastline = CoastlineIndex.from_shapefile(coastline_shp)

    out_frames = []
    for pid, grp in df.groupby("path_id", sort=False):
        logging.info(f"Processing path_id={pid} with {len(grp)} rows.")
        out = process_path(grp, coastline)
        out["path_id"] = pid
        out_frames.append(out)

    out_df = pd.concat(out_frames, ignore_index=True)
    out_df = out_df.sort_values(["path_id", "t_seconds"]).reset_index(drop=True)

    logging.info(f"Writing output CSV: {output_csv}")
    out_df.to_csv(output_csv, index=False)
    logging.info("Done.")

def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

def main():
    global GRID_STEP_M, GRID_MARGIN_M, SHORT_GAP_SECONDS, SPEED_CAP_KTS, LOW_SPEED_KTS
    parser = argparse.ArgumentParser(description="Rule-based AIS imputation with Natural Earth coastline mask.")
    parser.add_argument("--input", required=True, help="Input AIS CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--coastline", required=True, help="Natural Earth 10m coastline shapefile (e.g., ne_10m_coastline.shp).")
    parser.add_argument("--grid-step-m", type=float, default=GRID_STEP_M, help="Grid step (meters) for A* water-path.")
    parser.add_argument("--grid-margin-m", type=float, default=GRID_MARGIN_M, help="BBox margin (meters) for A* search.")
    parser.add_argument("--short-gap-seconds", type=float, default=SHORT_GAP_SECONDS, help="Short-gap threshold (seconds).")
    parser.add_argument("--speed-cap-kts", type=float, default=SPEED_CAP_KTS, help="Speed cap (knots).")
    parser.add_argument("--low-speed-kts", type=float, default=LOW_SPEED_KTS, help="Below this, COG is unreliable.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args()

    GRID_STEP_M = args.grid_step_m
    GRID_MARGIN_M = args.grid_margin_m
    SHORT_GAP_SECONDS = args.short_gap_seconds
    SPEED_CAP_KTS = args.speed_cap_kts
    LOW_SPEED_KTS = args.low_speed_kts

    setup_logging(args.verbose)
    try:
        process_csv(args.input, args.output, args.coastline)
    except Exception as e:
        logging.exception(f"Processing failed: {e}")
        raise

if __name__ == "__main__":
    main()
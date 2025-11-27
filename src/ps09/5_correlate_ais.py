#!/usr/bin/env python3
"""
Assign MMSIs to vessel detections using AIS with a unified ±6-hour window and integer AIS Status.

Key updates based on your specs:
  - Uses a default time window of ±6 hours from the image timestamp for all sensors.
  - Treats AIS Status as an integer and considers {1 (anchored), 5 (moored), 7 (aground)} as stationary.
  - Restricts correlation to selection rows marked for AIS correlations (Remarks contains "AIS correlations") [1].
  - Emits MMSI '0' when no AIS co-relation is found, as required [1].

Inputs:
  - selection CSV with columns:
      S.No.(ID), time_stamp, image_name, image_centre_latitude, image_centre_longitude, Remarks
  - detections CSV with columns (subset used):
      scene_id, lat, lon, score, vessel_length_m, vessel_width_m,
      vessel_speed_k, orientation, meters_per_pixel, heading_bucket_0..15
  - directory of cleaned AIS CSVs with columns:
      MMSI, BaseDateTime, LON, LAT, SOG, COG, Heading, VesselName, IMO, CallSign, VesselType,
      Status (integer), Length, Width, Draft, Cargo, TransceiverClass, segment_id, pos_speed_kn

Output:
  - CSV with columns: sl.no., time_stamp, image_name, vessel_latitude, vessel_longitude, mmsi

Usage:
  python correlate_ais.py \
    --selection path/to/selection.csv \
    --detections path/to/detections.csv \
    --ais-dir path/to/ais_dir \
    --output matched.csv

Notes:
  - Times are treated as UTC if timezone info is missing (override with --assume-utc false).
  - Requires: pandas, numpy. Optional: scipy (Hungarian assignment). Falls back to greedy if SciPy is unavailable.
"""

import argparse
import glob
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.optimize import linear_sum_assignment
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def parse_args():
    p = argparse.ArgumentParser(description="Assign MMSIs to vessel detections using AIS")
    p.add_argument("--selection", required=True, help="Path to selection CSV")
    p.add_argument("--detections", required=True, help="Path to detections CSV")
    p.add_argument("--ais-dir", required=True, help="Directory containing AIS CSV files")
    p.add_argument("--output", default="matched_output.csv", help="Path to output CSV")
    p.add_argument("--score-threshold", type=float, default=0.5, help="Min detection score to consider")
    p.add_argument("--time-window-hours", type=float, default=6.0, help="Temporal window (hours) around image time")
    p.add_argument("--spatial-buffer-km", type=float, default=250.0, help="Scene extent buffer for AIS prefiltering")
    p.add_argument("--gate-max-m", type=float, default=5000.0, help="Max spatial gate for candidate filtering (meters)")
    p.add_argument("--assume-utc", type=lambda x: str(x).lower() not in {"false", "0", "no"}, default=True,
                   help="Treat naive datetimes as UTC (default: true)")
    p.add_argument("--restrict-to-remarks", type=lambda x: str(x).lower() not in {"false", "0", "no"}, default=True,
                   help="Only correlate scenes whose selection Remarks mention AIS correlations (default: true)")
    p.add_argument("--verbose", action="store_true", help="Verbose logging")
    return p.parse_args()


def to_utc_datetime(series: pd.Series, assume_utc: bool) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce", utc=False)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert("UTC")
    else:
        if assume_utc:
            dt = dt.dt.tz_localize("UTC")
        else:
            dt = dt.dt.tz_localize("UTC")
            print("Warning: Datetimes were naive and --assume-utc=false. Localizing to UTC anyway; verify your times.",
                  file=sys.stderr)
    return dt


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    R = 6371000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def forward_point(lat_deg, lon_deg, distance_m, bearing_deg) -> Tuple[float, float]:
    R = 6371000.0
    br = math.radians(bearing_deg % 360.0)
    d_over_r = distance_m / R
    lat1 = math.radians(lat_deg)
    lon1 = math.radians(lon_deg)
    lat2 = math.asin(math.sin(lat1) + math.cos(lat1) * d_over_r * math.cos(br))
    lon2 = lon1 + math.atan2(math.sin(br) * d_over_r * math.cos(lat1),
                             1 - (math.sin(lat1) * math.sin(lat2)))
    return math.degrees(lat2), (math.degrees(lon2) + 540.0) % 360.0 - 180.0


def bearing_between(lat1, lon1, lat2, lon2) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    brng = math.degrees(math.atan2(y, x))
    return (brng + 360.0) % 360.0


def circ_angle_diff_deg(a, b) -> float:
    d = abs((a - b) % 360.0)
    return d if d <= 180.0 else 360.0 - d


def most_likely_heading(row: pd.Series) -> Optional[float]:
    bucket_cols = [c for c in row.index if c.startswith("heading_bucket_")]
    if not bucket_cols:
        return None
    probs = row[bucket_cols].astype(float).values
    if np.all(np.isnan(probs)):
        return None
    idx = int(np.nanargmax(probs))
    nb = len(bucket_cols)
    return ((idx + 0.5) * (360.0 / nb)) % 360.0


def is_stationary_status(status_val: Optional[float], sog_kn: Optional[float]) -> bool:
    """
    AIS integer NavStatus: we treat 1 (anchored), 5 (moored), 7 (aground) as stationary.
    Also consider SOG < 0.5 kn stationary if status is missing.
    """
    try:
        s = int(status_val) if status_val is not None and not (pd.isna(status_val)) else None
    except Exception:
        s = None
    if s in {1, 5, 7}:
        return True
    try:
        if sog_kn is not None and float(sog_kn) < 0.5:
            return True
    except Exception:
        pass
    return False


def predict_ais_position(track_df: pd.DataFrame, t_img: pd.Timestamp) -> Optional[Dict]:
    if track_df.empty:
        return None
    before = track_df[track_df["BaseDateTime"] <= t_img]
    after = track_df[track_df["BaseDateTime"] >= t_img]
    row_b = before.iloc[-1] if not before.empty else None
    row_a = after.iloc[0] if not after.empty else None

    interp_max = pd.Timedelta(minutes=20)

    if row_b is not None and row_a is not None and (row_a["BaseDateTime"] - row_b["BaseDateTime"]) <= interp_max:
        tb = row_b["BaseDateTime"]
        ta = row_a["BaseDateTime"]
        total = (ta - tb).total_seconds()
        alpha = 0.0 if total == 0 else (t_img - tb).total_seconds() / total
        lat = (1 - alpha) * float(row_b["LAT"]) + alpha * float(row_a["LAT"])
        lon = (1 - alpha) * float(row_b["LON"]) + alpha * float(row_a["LON"])
        cog = bearing_between(float(row_b["LAT"]), float(row_b["LON"]), float(row_a["LAT"]), float(row_a["LON"]))
        sog = np.nan
        dt_used = min(abs((t_img - tb)), abs((ta - t_img))).total_seconds()
        status = row_b.get("Status", np.nan) if "Status" in row_b else np.nan
        if pd.isna(status) and "Status" in row_a:
            status = row_a.get("Status", np.nan)
        return {"lat": lat, "lon": lon, "cog": cog, "sog_kn": sog, "time_gap_s": dt_used, "status": status}

    candidates = []
    if row_b is not None:
        candidates.append(("before", row_b, (t_img - row_b["BaseDateTime"]).total_seconds()))
    if row_a is not None:
        candidates.append(("after", row_a, (t_img - row_a["BaseDateTime"]).total_seconds()))
    if not candidates:
        return None

    which, row, dt = min(candidates, key=lambda x: abs(x[2]))
    lat0 = float(row["LAT"])
    lon0 = float(row["LON"])
    sog_kn = row["SOG"] if "SOG" in row and pd.notna(row["SOG"]) else (row["pos_speed_kn"] if "pos_speed_kn" in row and pd.notna(row["pos_speed_kn"]) else np.nan)
    cog = row["COG"] if "COG" in row and pd.notna(row["COG"]) else np.nan
    status = row["Status"] if "Status" in row else np.nan

    if is_stationary_status(status, sog_kn):
        return {"lat": lat0, "lon": lon0, "cog": float(cog) if pd.notna(cog) else np.nan,
                "sog_kn": float(sog_kn) if pd.notna(sog_kn) else 0.0,
                "time_gap_s": abs(dt), "status": status}

    if pd.notna(sog_kn) and pd.notna(cog):
        sog_mps = float(sog_kn) * 0.514444
        distance = sog_mps * float(dt)
        lat_pred, lon_pred = forward_point(lat0, lon0, distance, float(cog))
        return {"lat": lat_pred, "lon": lon_pred, "cog": float(cog), "sog_kn": float(sog_kn),
                "time_gap_s": abs(dt), "status": status}

    return {"lat": lat0, "lon": lon0, "cog": float(cog) if pd.notna(cog) else np.nan,
            "sog_kn": float(sog_kn) if pd.notna(sog_kn) else np.nan,
            "time_gap_s": abs(dt), "status": status}


def build_candidates_for_scene(det_df: pd.DataFrame,
                               ais_df_scene: pd.DataFrame,
                               t_img: pd.Timestamp,
                               buffer_km: float,
                               gate_max_m: float,
                               verbose: bool = False) -> Tuple[List[Tuple[int, str, float]], Dict[Tuple[int, str], float]]:
    """
    Build candidate detection–MMSI pairs with costs and distances.
    Returns:
      - list of (det_row_index, mmsi, cost)
      - dict mapping (det_row_index, mmsi) -> distance_m
    """
    if det_df.empty or ais_df_scene.empty:
        return [], {}

    min_lat = det_df["lat"].min()
    max_lat = det_df["lat"].max()
    min_lon = det_df["lon"].min()
    max_lon = det_df["lon"].max()
    lat0 = float(det_df["lat"].mean())
    deg_lat_buf = buffer_km / 111.32
    deg_lon_buf = buffer_km / (111.32 * max(0.2, math.cos(math.radians(lat0))))

    ais_pref = ais_df_scene[
        (ais_df_scene["LAT"] >= (min_lat - deg_lat_buf)) &
        (ais_df_scene["LAT"] <= (max_lat + deg_lat_buf)) &
        (ais_df_scene["LON"] >= (min_lon - deg_lon_buf)) &
        (ais_df_scene["LON"] <= (max_lon + deg_lon_buf))
    ].copy()

    if ais_pref.empty:
        if verbose:
            print("  No AIS in spatially buffered scene extent.", file=sys.stderr)
        return [], {}

    ais_pref = ais_pref.sort_values(["MMSI", "BaseDateTime"])
    groups = list(ais_pref.groupby("MMSI", sort=False))

    pred_by_mmsi: Dict[str, Dict] = {}
    for mmsi, g in groups:
        pred = predict_ais_position(g, t_img)
        if pred is not None and pd.notna(pred["lat"]) and pd.notna(pred["lon"]):
            pred_by_mmsi[str(mmsi)] = pred

    if not pred_by_mmsi:
        return [], {}

    candidates: List[Tuple[int, str, float]] = []
    dist_lookup: Dict[Tuple[int, str], float] = {}

    # Precompute median AIS length per MMSI in subset
    ais_len_by_mmsi: Dict[str, float] = {}
    if "Length" in ais_pref.columns:
        for mmsi, g in ais_pref.groupby(ais_pref["MMSI"].astype(str)):
            vals = g["Length"].dropna()
            if not vals.empty:
                ais_len_by_mmsi[mmsi] = float(vals.median())

    for det_idx, row in det_df.iterrows():
        det_lat = float(row["lat"])
        det_lon = float(row["lon"])
        det_len = float(row["vessel_length_m"]) if pd.notna(row.get("vessel_length_m", np.nan)) else np.nan
        det_speed_kn = float(row["vessel_speed_k"]) if pd.notna(row.get("vessel_speed_k", np.nan)) else np.nan
        det_heading = most_likely_heading(row)

        for mmsi, pred in pred_by_mmsi.items():
            ais_lat = float(pred["lat"])
            ais_lon = float(pred["lon"])
            ais_cog = float(pred["cog"]) if pd.notna(pred.get("cog", np.nan)) else np.nan
            ais_sog_kn = float(pred["sog_kn"]) if pd.notna(pred.get("sog_kn", np.nan)) else np.nan
            ais_status = pred.get("status", np.nan)
            time_gap_s = float(pred.get("time_gap_s", 0.0))

            # Dynamic spatial gate, capped at gate_max_m
            if is_stationary_status(ais_status, ais_sog_kn):
                gate_m = min(gate_max_m, 300.0)
            else:
                sog_mps = (ais_sog_kn * 0.514444) if pd.notna(ais_sog_kn) else 1.543  # ~3 kn default
                gate_m = min(gate_max_m, max(300.0, 0.5 * sog_mps * abs(time_gap_s)))

            distance_m = haversine_m(det_lat, det_lon, ais_lat, ais_lon)
            if distance_m > gate_m:
                continue

            # Heading gate for non-stationary targets
            if det_heading is not None and pd.notna(ais_cog) and not is_stationary_status(ais_status, ais_sog_kn):
                angle_diff = circ_angle_diff_deg(det_heading, ais_cog)
                if angle_diff > 60.0:
                    continue
            else:
                angle_diff = np.nan

            # Size gate (soft)
            length_penalty = 0.0
            ais_len = ais_len_by_mmsi.get(mmsi, np.nan)
            if pd.notna(det_len) and pd.notna(ais_len) and ais_len > 0:
                rel_diff = abs(det_len - ais_len) / ais_len
                if rel_diff > 0.5:
                    continue
                length_penalty = 300.0 * rel_diff

            # Speed penalty
            speed_penalty = 0.0
            if pd.notna(det_speed_kn) and pd.notna(ais_sog_kn):
                if max(det_speed_kn, ais_sog_kn) > 5.0 and abs(det_speed_kn - ais_sog_kn) > 3.0:
                    continue
                speed_penalty = 30.0 * abs(det_speed_kn - ais_sog_kn)

            angle_penalty = 0.0 if np.isnan(angle_diff) else (5.0 * angle_diff)

            cost = float(distance_m) + angle_penalty + length_penalty + speed_penalty + 0.05 * abs(time_gap_s)
            candidates.append((det_idx, mmsi, cost))
            dist_lookup[(det_idx, mmsi)] = float(distance_m)

    return candidates, dist_lookup


def assign_pairs(detections_idx: List[int],
                 mmsis: List[str],
                 candidates: List[Tuple[int, str, float]],
                 max_cost: float,
                 use_hungarian: bool = True) -> Dict[int, Optional[str]]:
    det_to_idx = {d: i for i, d in enumerate(detections_idx)}
    mmsi_to_idx = {m: j for j, m in enumerate(mmsis)}
    nD = len(detections_idx)
    nM = len(mmsis)
    C = np.full((nD, nM), 1e9, dtype=float)
    for d_idx, mmsi, cost in candidates:
        i = det_to_idx[d_idx]
        j = mmsi_to_idx[mmsi]
        C[i, j] = min(C[i, j], cost)

    assigned: Dict[int, Optional[str]] = {d: None for d in detections_idx}

    if use_hungarian and HAVE_SCIPY and nD > 0 and nM > 0:
        row_ind, col_ind = linear_sum_assignment(C)
        for i, j in zip(row_ind, col_ind):
            d = detections_idx[i]
            m = mmsis[j]
            cost = C[i, j]
            if cost <= max_cost:
                assigned[d] = m
    else:
        flat = []
        for i, d in enumerate(detections_idx):
            for j, m in enumerate(mmsis):
                cost = C[i, j]
                if cost < 1e9:
                    flat.append((cost, d, m))
        flat.sort(key=lambda x: x[0])
        used_d = set()
        used_m = set()
        for cost, d, m in flat:
            if cost > max_cost:
                break
            if d in used_d or m in used_m:
                continue
            assigned[d] = m
            used_d.add(d)
            used_m.add(m)

    return assigned


def main():
    args = parse_args()
    verbose = args.verbose

    sel = pd.read_csv(args.selection)
    det = pd.read_csv(args.detections)

    required_sel_cols = {"S.No.(ID)", "time_stamp", "image_name", "image_centre_latitude", "image_centre_longitude", "Remarks"}
    missing = required_sel_cols - set(sel.columns)
    if missing:
        print(f"Error: selection CSV missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    required_det_cols = {"scene_id", "lat", "lon", "score"}
    missing = required_det_cols - set(det.columns)
    if missing:
        print(f"Error: detections CSV missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    # Optionally restrict to scenes flagged for AIS correlation [1]
    if args.restrict_to_remarks:
        mask = sel["Remarks"].fillna("").str.lower().str.contains("ais correlation")
        sel = sel[mask].copy()
        if sel.empty:
            print("Warning: No selection rows flagged for AIS correlation; proceeding with all scenes.",
                  file=sys.stderr)

    # Parse selection timestamps
    sel["image_time"] = to_utc_datetime(sel["time_stamp"], assume_utc=args.assume_utc)

    # Filter detections by selected scenes and score threshold
    det = det[det["scene_id"].isin(sel["image_name"])].copy()
    det = det[det["score"] >= args.score_threshold].copy()

    if det.empty:
        print("No detections after filtering by selection and score.", file=sys.stderr)
        out = pd.DataFrame(columns=["sl.no.", "time_stamp", "image_name", "vessel_latitude", "vessel_longitude", "mmsi"])
        out.to_csv(args.output, index=False)
        print(f"Wrote empty output to {args.output}")
        return

    # Merge selection info into detections
    det = det.merge(
        sel[["S.No.(ID)", "image_name", "image_time", "image_centre_latitude", "image_centre_longitude", "time_stamp"]],
        left_on="scene_id",
        right_on="image_name",
        how="left",
        suffixes=("", "_sel")
    )

    # Load AIS CSVs
    ais_files = sorted(glob.glob(os.path.join(args.ais_dir, "noaa_ais_cleaned_*.csv")))
    if not ais_files:
        print(f"Error: no AIS CSVs found in {args.ais_dir}", file=sys.stderr)
        sys.exit(1)

    ais_list = []
    usecols = ["MMSI", "BaseDateTime", "LON", "LAT", "SOG", "COG", "Status", "Length", "Width", "pos_speed_kn", "segment_id"]
    for f in ais_files:
        try:
            header_cols = pd.read_csv(f, nrows=0).columns
            a = pd.read_csv(f, usecols=[c for c in usecols if c in header_cols])
        except Exception as e:
            print(f"Warning: skipping AIS file {f}: {e}", file=sys.stderr)
            continue
        req = {"MMSI", "BaseDateTime", "LAT", "LON"}
        if not req.issubset(set(a.columns)):
            print(f"Warning: AIS file {f} missing required columns; skipping.", file=sys.stderr)
            continue
        a["BaseDateTime"] = to_utc_datetime(a["BaseDateTime"], assume_utc=args.assume_utc)
        # Status is integer per your cleaned AIS
        if "Status" in a.columns:
            a["Status"] = pd.to_numeric(a["Status"], errors="coerce").astype("Int64")
        ais_list.append(a)

    if not ais_list:
        print("Error: no valid AIS data loaded.", file=sys.stderr)
        sys.exit(1)

    ais = pd.concat(ais_list, ignore_index=True)
    ais = ais.dropna(subset=["MMSI", "BaseDateTime", "LAT", "LON"])
    ais["MMSI"] = ais["MMSI"].astype(str)

    out_rows = []

    # Process per scene
    for scene_id, group in det.groupby("scene_id"):
        scene_sel = group.iloc[0]
        t_img = scene_sel["image_time"]
        if pd.isna(t_img):
            if verbose:
                print(f"Scene {scene_id}: missing time_stamp; skipping.", file=sys.stderr)
            continue

        win = pd.Timedelta(hours=float(args.time_window_hours))
        t0 = t_img - win
        t1 = t_img + win

        ais_scene = ais[(ais["BaseDateTime"] >= t0) & (ais["BaseDateTime"] <= t1)].copy()
        if ais_scene.empty:
            if verbose:
                print(f"Scene {scene_id}: no AIS in ±{args.time_window_hours} hours.", file=sys.stderr)
            for det_idx, row in group.iterrows():
                out_rows.append({
                    # "sl.no.": scene_sel["S.No.(ID)"],
                    "time_stamp": scene_sel["time_stamp"],
                    "image_name": row["scene_id"],
                    "vessel_latitude": row["lat"],
                    "vessel_longitude": row["lon"],
                    "mmsi": "0"  # per spec [1]
                })
            continue

        candidates, dist_lookup = build_candidates_for_scene(
            group, ais_scene, t_img, buffer_km=args.spatial_buffer_km, gate_max_m=args.gate_max_m, verbose=verbose
        )

        if not candidates:
            if verbose:
                print(f"Scene {scene_id}: no gated candidates.", file=sys.stderr)
            for det_idx, row in group.iterrows():
                out_rows.append({
                    # "sl.no.": scene_sel["S.No.(ID)"],
                    "time_stamp": scene_sel["time_stamp"],
                    "image_name": row["scene_id"],
                    "vessel_latitude": row["lat"],
                    "vessel_longitude": row["lon"],
                    "mmsi": "0"  # per spec [1]
                })
            continue

        det_indices = list(group.index)
        mmsis = sorted(list({m for (_, m, _) in candidates}))

        assigned = assign_pairs(det_indices, mmsis, candidates, max_cost=args.gate_max_m, use_hungarian=True)

        for det_idx, row in group.iterrows():
            mmsi = assigned.get(det_idx)
            d = dist_lookup.get((det_idx, mmsi), 1e9) if mmsi is not None else 1e9
            if mmsi is None or d > args.gate_max_m:
                mmsi_out = "0"  # per spec [1]
            else:
                mmsi_out = mmsi
            out_rows.append({
                # "sl.no.": scene_sel["S.No.(ID)"],
                "time_stamp": scene_sel["time_stamp"],
                "image_name": row["scene_id"],
                "vessel_latitude": row["lat"],
                "vessel_longitude": row["lon"],
                "mmsi": mmsi_out
            })

        if verbose:
            n_assigned = sum(1 for v in assigned.values() if v is not None)
            print(f"Scene {scene_id}: detections={len(group)} assigned={n_assigned}", file=sys.stderr)

    out_df = pd.DataFrame(out_rows, columns=["time_stamp", "image_name", "vessel_latitude", "vessel_longitude", "mmsi"])
    out_df.insert(0, "sl.no.", range(1, len(out_df) + 1))
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
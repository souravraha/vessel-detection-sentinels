#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path
import re
import math

import numpy as np
import pandas as pd
from shapely.geometry import Polygon, mapping

# Optional deps only for export & crop visualization (no rasterio/GDAL):
import fiona
from fiona.crs import from_string as fiona_from_string
from PIL import Image, ImageDraw

# ---------------------------
# Helpers
# ---------------------------

def meters_per_deg_lat(lat_rad: float) -> float:
    # Snyder-derived approximation (meters per degree latitude)
    # https://en.wikipedia.org/wiki/Geographic_coordinate_system#Length_of_a_degree
    # with higher-order terms for accuracy
    return (111132.92
            - 559.82 * math.cos(2 * lat_rad)
            + 1.175 * math.cos(4 * lat_rad)
            - 0.0023 * math.cos(6 * lat_rad))

def meters_per_deg_lon(lat_rad: float) -> float:
    # meters per degree longitude (varies with latitude)
    return (111412.84 * math.cos(lat_rad)
            - 93.5 * math.cos(3 * lat_rad)
            + 0.118 * math.cos(5 * lat_rad))

def heading_from_buckets(row: pd.Series) -> float:
    """Return heading in degrees (clockwise from True North) using bucket center of argmax."""
    probs = []
    for i in range(16):
        k = f"heading_bucket_{i}"
        probs.append(float(row.get(k, 0.0)) if pd.notna(row.get(k, np.nan)) else 0.0)
    if sum(probs) <= 0:
        return 0.0
    i = int(np.argmax(probs))
    return i * 22.5 + 11.25  # bucket center

def oriented_corners_enu(L_m: float, W_m: float, heading_deg: float) -> np.ndarray:
    """
    Build 4 corners of the oriented rectangle in local ENU (east, north) meters,
    centered at (0,0). Heading is clockwise from North.
    Long axis aligned with heading; short axis is 90° clockwise from long axis.
    """
    a = max(0.5, L_m / 2.0)  # half-length (m)
    b = max(0.5, W_m / 2.0)  # half-width  (m)
    φ = math.radians(heading_deg)
    # unit vectors in ENU
    u_long  = np.array([math.sin(φ),  math.cos(φ)])   # (east, north)
    u_short = np.array([math.cos(φ), -math.sin(φ)])   # 90° clockwise from u_long
    C1 =  a*u_long + b*u_short
    C2 =  a*u_long - b*u_short
    C3 = -a*u_long - b*u_short
    C4 = -a*u_long + b*u_short
    return np.stack([C1, C2, C3, C4], axis=0)  # shape (4, 2) as (E, N)

def enu_to_latlon(center_lon: float, center_lat: float, enu_pts: np.ndarray) -> np.ndarray:
    """
    Convert small ENU offsets (meters) to lon/lat using local meters/deg at the center.
    Returns array of shape (N, 2) as (lon, lat).
    """
    lat_rad = math.radians(center_lat)
    m_per_deg_lat = meters_per_deg_lat(lat_rad)
    m_per_deg_lon = meters_per_deg_lon(lat_rad)
    # enu_pts[:, 0] = east, enu_pts[:, 1] = north
    dlon = enu_pts[:, 0] / m_per_deg_lon
    dlat = enu_pts[:, 1] / m_per_deg_lat
    lons = center_lon + dlon
    lats = center_lat + dlat
    return np.stack([lons, lats], axis=1)

def enclosing_lonlat_aabb(corners_ll: np.ndarray) -> np.ndarray:
    """Return axis-aligned bbox ring from lon/lat corners (5 points closed)."""
    lons = corners_ll[:, 0]
    lats = corners_ll[:, 1]
    min_lon, max_lon = float(np.min(lons)), float(np.max(lons))
    min_lat, max_lat = float(np.min(lats)), float(np.max(lats))
    ring = np.array([
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ], dtype=float)
    return ring

def find_predictions_csvs(root: Path):
    return [p for p in root.rglob("*predictions.csv") if p.is_file()]

def guess_crop_for_detect(crops_root: Path, detect_id: str):
    """
    Try to find a crop PNG for a given detect_id. Prefers S2 TCI or any S1 channel.
    Filenames look like: {scene_id}_{detection_idx}_{suffix}.png
    detect_id is usually "{scene_id}_{idx}".
    """
    # match any file that starts with detect_id + '_' and ends with .png
    pattern = f"{detect_id}_*.png"
    cands = list(crops_root.rglob(pattern))
    # prefer tci, then vv, vh, then anything
    def score(p: Path):
        n = p.name.lower()
        if "tci" in n: return 0
        if re.search(r"(^|[_\-])(vv|vh)([_\-]|\.png$)", n): return 1
        return 2
    cands.sort(key=score)
    return cands[0] if cands else None

def draw_on_crop(crop_path: Path, out_path: Path, L_m: float, W_m: float, heading_deg: float,
                 orientation_deg: float, mpp: float):
    """
    Draw oriented rectangle on a 128x128 crop centered at (64,64).
    Pixel heading = orientation + heading (clockwise from 'up').
    """
    if not crop_path or not crop_path.exists():
        return
    im = Image.open(crop_path).convert("RGB")
    w, h = im.size
    cx, cy = w / 2.0, h / 2.0  # typically 64, 64

    L_px = max(1.0, L_m / mpp)
    W_px = max(1.0, W_m / mpp)
    φ = math.radians((orientation_deg or 0.0) + (heading_deg or 0.0))
    # image coords: +x right, +y down, "up" is negative y
    u_long  = np.array([ math.sin(φ), -math.cos(φ) ])
    u_short = np.array([ math.cos(φ),  math.sin(φ) ])
    a = L_px / 2.0
    b = W_px / 2.0
    C1 = np.array([cx, cy]) + a*u_long + b*u_short
    C2 = np.array([cx, cy]) + a*u_long - b*u_short
    C3 = np.array([cx, cy]) - a*u_long - b*u_short
    C4 = np.array([cx, cy]) - a*u_long + b*u_short
    ring = [tuple(C1), tuple(C2), tuple(C3), tuple(C4), tuple(C1)]

    drw = ImageDraw.Draw(im)
    drw.line(ring, width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path)

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Compute lat-lon AABB for vessels (no rasterio), export GeoJSON/Shapefiles, and draw on crops.")
    ap.add_argument("--predictions_dir", required=True, type=Path, help="Root containing predictions.csv (searched recursively).")
    ap.add_argument("--crops_dir", type=Path, default=None, help="Root of cropped PNGs (optional, for visualization).")
    ap.add_argument("--output_dir", required=True, type=Path, help="Output folder for GeoJSON, shapefiles, and previews.")
    ap.add_argument("--min_score", type=float, default=0.0, help="Filter detections below this confidence.")
    ap.add_argument("--participant_name", type=str, default="participant", help="Name to record in GeoJSON info.predicted_by.")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = args.output_dir / "viz_crops"
    shp_root = args.output_dir / "shapefiles"
    shp_root.mkdir(parents=True, exist_ok=True)

    csvs = find_predictions_csvs(args.predictions_dir)
    if not csvs:
        print("No predictions.csv found.")
        return

    dfs = []
    for c in csvs:
        try:
            df = pd.read_csv(c)
            dfs.append(df)
        except Exception as e:
            print(f"[WARN] Skipping {c}: {e}")

    if not dfs:
        print("No readable predictions.")
        return

    pred = pd.concat(dfs, ignore_index=True)
    if "score" in pred.columns:
        pred = pred[pred["score"].fillna(0) >= args.min_score].copy()

    # Ensure required fields exist
    for col in ["lat","lon","vessel_length_m","vessel_width_m","scene_id","detect_id","meters_per_pixel","orientation"]:
        if col not in pred.columns:
            pred[col] = np.nan

    # GeoJSON skeleton (PS-09 style)
    gj = {
        "info": {
            "description": "Grand challenge MDA - Detection",
            "version": "1.0",
            "year": 2025,
            "predicted_by": args.participant_name,
        },
        "licenses": [],
        "images": [],
        "categories": [{"id": 1, "name": "ship"}],
        "annotations": []
    }

    # We'll emit one shapefile per scene_id with EPSG:4326 polygons
    scenes = sorted(pred["scene_id"].dropna().unique().tolist())
    image_id_map = {sid: i+1 for i, sid in enumerate(scenes)}
    for sid in scenes:
        gj["images"].append({
            "id": image_id_map[sid],
            "file_name": sid,
            "width": 0,     # not known / not needed
            "height": 0,    # not known / not needed
            "date_captured": ""
        })

    # Create shapefiles per scene
    shp_handles = {}
    schema = {
        "geometry": "Polygon",
        "properties": {
            "image_id": "int",
            "category_id": "int",
            "score": "float",
            "detect_id": "str"
        }
    }
    crs = fiona_from_string("EPSG:4326")
    for sid in scenes:
        out_dir = shp_root / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        shp_path = out_dir / f"{sid}.shp"
        shp_handles[sid] = fiona.open(shp_path, mode="w",
                                      driver="ESRI Shapefile",
                                      schema=schema, crs=crs, encoding="utf-8")

    annot_id = 1

    for idx, r in pred.iterrows():
        try:
            center_lon = float(r["lon"])
            center_lat = float(r["lat"])
            L_m = float(r["vessel_length_m"])
            W_m = float(r["vessel_width_m"])
            score = float(r.get("score", np.nan)) if pd.notna(r.get("score", np.nan)) else None
            sid = str(r.get("scene_id", ""))
            detect_id = str(r.get("detect_id", f"{sid}_{idx}"))
            mpp = float(r.get("meters_per_pixel", np.nan))
            orientation = float(r.get("orientation", 0.0)) if pd.notna(r.get("orientation", np.nan)) else 0.0

            if not (np.isfinite(center_lon) and np.isfinite(center_lat) and
                    np.isfinite(L_m) and np.isfinite(W_m)):
                continue

            # Heading from buckets
            H = heading_from_buckets(r)

            # Oriented corners in ENU meters (centered)
            corners_enu = oriented_corners_enu(L_m, W_m, H)  # shape (4,2): (E,N)

            # Convert corners to lon/lat around center
            corners_ll = enu_to_latlon(center_lon, center_lat, corners_enu)  # (4,2)

            # Enclosing axis-aligned bbox ring in lon/lat
            ring = enclosing_lonlat_aabb(corners_ll)  # (5,2)

            # Append to GeoJSON
            gj["annotations"].append({
                "image_id": image_id_map.get(sid, 0),
                "category_id": 1,
                "bbox": ring.tolist(),  # rectangular ring in lon/lat
                "score": None if score is None else float(score),
                "id": annot_id
            })
            annot_id += 1

            # Write to the per-scene shapefile
            shp = shp_handles.get(sid)
            if shp is not None:
                poly = Polygon(ring)
                shp.write({
                    "geometry": mapping(poly),
                    "properties": {
                        "image_id": image_id_map.get(sid, 0),
                        "category_id": 1,
                        "score": None if score is None else float(score),
                        "detect_id": detect_id
                    }
                })

            # Optional: draw on crop (oriented rectangle in pixel coords)
            if args.crops_dir and np.isfinite(mpp):
                crop_path = guess_crop_for_detect(args.crops_dir, detect_id)
                if crop_path:
                    out_png = (Path(viz_dir) / f"{detect_id}.png")
                    draw_on_crop(crop_path, out_png,
                                 L_m=L_m, W_m=W_m,
                                 heading_deg=H,
                                 orientation_deg=orientation,
                                 mpp=mpp)

        except Exception as e:
            print(f"[WARN] row {idx} skipped: {e}")

    # Close shapefiles
    for sid, shp in shp_handles.items():
        try:
            shp.close()
        except Exception:
            pass

    # Write consolidated GeoJSON
    out_geojson = args.output_dir / "ps09_detections.geojson"
    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(gj, f, ensure_ascii=False, indent=2)

    print(f"[OK] GeoJSON: {out_geojson}")
    print(f"[OK] Shapefiles: {shp_root}")
    if args.crops_dir:
        print(f"[OK] Crop overlays: {viz_dir}")

if __name__ == "__main__":
    main()

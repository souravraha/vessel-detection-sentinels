#!/usr/bin/env python3
import argparse, os, re, glob, json
from datetime import datetime, timedelta
import math
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform_bounds
import xml.etree.ElementTree as ET

def find_s1_kml(safe_dir):
    # Typical location: preview/map-overlay.kml
    kml = _first(os.path.join(safe_dir, "preview", "map-overlay.kml"))
    if not kml:
        # Some packages use 'quick-look' or different name
        kml = _first(os.path.join(safe_dir, "preview", "*.kml"))
    return kml

def footprint_from_kml(kml_path):
    """
    Parse S1 map-overlay.kml and return polygon coords in lon/lat.
    Returns: list[(lon, lat)], closed ring.
    """
    tree = ET.parse(kml_path); root = tree.getroot()
    # KML namespaces
    ns = {"kml": "http://www.opengis.net/kml/2.2"}
    # Look for <coordinates> under Polygon/outerBoundaryIs/LinearRing
    coords_nodes = root.findall(".//kml:coordinates", ns)
    for node in coords_nodes:
        txt = (node.text or "").strip()
        if not txt:
            continue
        pts = []
        for triplet in txt.replace("\n", " ").split():
            parts = triplet.split(",")
            if len(parts) >= 2:
                lon, lat = float(parts[0]), float(parts[1])
                pts.append((lon, lat))
        if len(pts) >= 4:
            # Ensure closed ring
            if pts[0] != pts[-1]:
                pts.append(pts[0])
            return pts
    raise RuntimeError("No coordinates found in KML")

# ----------------- helpers -----------------
def _first(pattern):
    xs = sorted(glob.glob(pattern, recursive=True))
    return xs[0] if xs else None

def parse_s2_time_from_name(safe_name):
    m = re.search(r"_([0-9]{8}T[0-9]{6})_", safe_name)
    return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S") if m else None

def parse_s1_times_from_name(safe_name):
    m = re.search(r"_([0-9]{8}T[0-9]{6})_([0-9]{8}T[0-9]{6})_", safe_name)
    if not m: return None, None
    return (datetime.strptime(m.group(1), "%Y%m%dT%H%M%S"),
            datetime.strptime(m.group(2), "%Y%m%dT%H%M%S"))

def km_to_deg_buffer(lat_deg, buffer_km):
    dlat = buffer_km / 110.574
    dlon = buffer_km / (111.320 * max(0.01, math.cos(math.radians(lat_deg))))
    return dlat, dlon

def detect_safe_type(safe_path):
    n = os.path.basename(safe_path).upper()
    if n.startswith(("S2A_", "S2B_")): return "S2"
    if n.startswith(("S1A_", "S1B_")): return "S1"
    if glob.glob(os.path.join(safe_path, "GRANULE", "*", "IMG_DATA", "*_B02*.jp2"), recursive=True): return "S2"
    return "S1"

# ----- find an anchor band to read geotransform quickly -----
def find_s2_anchor_band(safe_dir):
    gran = _first(os.path.join(safe_dir, "GRANULE", "*"))
    if not gran: return None
    img_root = os.path.join(gran, "IMG_DATA")
    return (_first(os.path.join(img_root, "**", "*_B03_10m.[jJ][pP]2")) or
            _first(os.path.join(img_root, "**", "*_B03.[jJ][pP]2")) or
            _first(os.path.join(img_root, "R10m", "*_B03.[jJ][pP]2")))

def find_s1_anchor_band(safe_dir):
    tiff = _first(os.path.join(safe_dir, "measurement", "*.tif*"))
    return tiff or _first(os.path.join(safe_dir, "**", "*.tif*"))

def get_footprint_or_bounds_wgs84(raster_path, safe_dir):
    """
    Try raster bounds (fast). If CRS is None, fall back to SAFE preview KML footprint.
    Returns: (lon_min, lat_min, lon_max, lat_max), and optionally a polygon for debugging.
    """
    try:
        with rasterio.open(raster_path) as src:
            if src.crs:
                b = src.bounds
                lon_min, lat_min, lon_max, lat_max = transform_bounds(
                    src.crs, "EPSG:4326", b.left, b.bottom, b.right, b.top, densify_pts=21
                )
                lon_min, lon_max = sorted([lon_min, lon_max])
                lat_min, lat_max = sorted([lat_min, lat_max])
                return (lon_min, lat_min, lon_max, lat_max), None
            # If we’re here, CRS is None -> fall back to KML
    except Exception:
        # Couldn’t open the raster; try KML anyway
        pass

    kml = find_s1_kml(safe_dir)
    if not kml:
        raise RuntimeError(f"CRS is invalid: None and no KML footprint found in {os.path.basename(safe_dir)}")
    ring = footprint_from_kml(kml)  # list of (lon,lat)
    lons = [p[0] for p in ring]; lats = [p[1] for p in ring]
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    return (lon_min, lat_min, lon_max, lat_max), ring


def pad_bbox(lon_min, lat_min, lon_max, lat_max, buffer_km):
    lat_c = (lat_min + lat_max)/2.0
    dlat, dlon = km_to_deg_buffer(lat_c, buffer_km)
    return lon_min - dlon, lat_min - dlat, lon_max + dlon, lat_max + dlat

def extract_extent_for_safe(safe_dir, time_pad_min=15, bbox_buffer_km=5.0):
    safe_name = os.path.basename(safe_dir)
    sensor = detect_safe_type(safe_dir)

    if sensor == "S2":
        t0 = parse_s2_time_from_name(safe_name)
        if t0 is None:
            raise RuntimeError(f"Cannot parse S2 timestamp from name: {safe_name}")
        start = t0 - timedelta(minutes=time_pad_min)
        end   = t0 + timedelta(minutes=time_pad_min)
        anchor = find_s2_anchor_band(safe_dir)
    else:
        t1, t2 = parse_s1_times_from_name(safe_name)
        if t1 is None or t2 is None:
            raise RuntimeError(f"Cannot parse S1 time range from name: {safe_name}")
        start, end = t1, t2
        anchor = find_s1_anchor_band(safe_dir)

    if not anchor:
        raise RuntimeError(f"No georeferenced raster found inside: {safe_name}")

    (bounds, ring) = get_footprint_or_bounds_wgs84(anchor, safe_dir)
    lon_min, lat_min, lon_max, lat_max = bounds
    if bbox_buffer_km and bbox_buffer_km > 0:
        lon_min, lat_min, lon_max, lat_max = pad_bbox(lon_min, lat_min, lon_max, lat_max, bbox_buffer_km)

    return {
        "safe_name": safe_name,
        "sensor": sensor,
        "start_time_utc": start.isoformat() + "Z",
        "end_time_utc": end.isoformat() + "Z",
        "lon_min": lon_min, "lat_min": lat_min, "lon_max": lon_max, "lat_max": lat_max,
        "center_lat": (lat_min + lat_max)/2.0,
        "center_lon": (lon_min + lon_max)/2.0,
        "ais_months_hint": sorted({start.strftime("%Y-%m"), end.strftime("%Y-%m")})
    }

# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser(description="Build AIS query extents for selected SAFE folders (from CSV).")
    ap.add_argument("--safe_root", required=True, help="Directory containing *.SAFE folders")
    ap.add_argument("--list_csv", required=True, help="CSV listing SAFE folder names to process")
    ap.add_argument("--safe_col", default=None, help="Column name with SAFE folder names (if omitted, tries to auto-detect)")
    ap.add_argument("--out_csv", default="ais_queries.csv")
    ap.add_argument("--out_json", default="ais_queries.json")
    ap.add_argument("--time_pad_min", type=int, default=15, help="Pad S2 single timestamp by +/- minutes")
    ap.add_argument("--bbox_buffer_km", type=float, default=5.0, help="Grow bbox by this many km on all sides")
    args = ap.parse_args()

    # load list
    df = pd.read_csv(args.list_csv)
    cols_lower = {c.lower(): c for c in df.columns}
    safe_col = args.safe_col or cols_lower.get("safe_name") or cols_lower.get("image_name") or list(df.columns)[0]
    safes_from_csv = df[safe_col].astype(str).str.strip()

    # normalize names (ensure .SAFE suffix; allow paths)
    target_names = []
    for s in safes_from_csv:
        base = os.path.basename(s)
        if not base.upper().endswith(".SAFE"):
            base += ".SAFE"
        target_names.append(base)

    # match to directories under safe_root (case-insensitive)
    disk_dirs = {d.lower(): d for d in glob.glob(os.path.join(args.safe_root, "*.SAFE")) if os.path.isdir(d)}
    rows, not_found, errors = [], [], []
    for name in target_names:
        # try exact, then case-insensitive
        cand = os.path.join(args.safe_root, name)
        if os.path.isdir(cand):
            safe_dir = cand
        else:
            # case-insensitive lookup
            safe_dir = disk_dirs.get(cand.lower())
            if not safe_dir:
                not_found.append(name); continue
        try:
            info = extract_extent_for_safe(safe_dir, time_pad_min=args.time_pad_min, bbox_buffer_km=args.bbox_buffer_km)
            rows.append(info)
        except Exception as e:
            errors.append((name, str(e)))

    if rows:
        df_out = pd.DataFrame(rows, columns=[
            "safe_name","sensor","start_time_utc","end_time_utc",
            "lon_min","lat_min","lon_max","lat_max","center_lat","center_lon","ais_months_hint"
        ])
        df_out.to_csv(args.out_csv, index=False)
        with open(args.out_json, "w") as f: json.dump(rows, f, indent=2)
        print(f"Wrote {args.out_csv} and {args.out_json} with {len(rows)} entries.")
    else:
        print("No extents produced (no valid SAFE folders matched).")

    if not_found:
        print("\nNot found under safe_root:")
        for n in not_found: print("  -", n)
    if errors:
        print("\nErrors:")
        for n, msg in errors: print(f"  - {n}: {msg}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Update a template JSON's annotations with minimal lat–lon–aligned bboxes
and write ESRI Shapefiles per scene_id. The rest of the template stays intact.

Inputs:
  - input.csv: detections with required columns:
      lat, lon, vessel_length_m, vessel_width_m,
      heading_bucket_0 ... heading_bucket_15,
      scene_id (must match an images[].file_name in the template to associate image_id)
  - template.json: JSON containing "images" (with {id, file_name, ...}) and other fields
  - output.json: path to write the updated template with new "annotations"
  - output_shp_dir: directory to write shapefiles grouped by scene_id

Usage:
  python update_template_annotations_and_shp.py input.csv template.json output.json output_shp_dir
"""

import sys
import os
import math
import json
import re
from typing import List, Dict, Tuple, Optional

try:
    import pandas as pd
except ImportError:
    print("Please install pandas: pip install pandas", file=sys.stderr)
    sys.exit(1)

try:
    from pyproj import Geod
except ImportError:
    print("Please install pyproj: pip install pyproj", file=sys.stderr)
    sys.exit(1)

try:
    import geopandas as gpd
except ImportError:
    print("Please install geopandas: pip install geopandas", file=sys.stderr)
    sys.exit(1)

try:
    from shapely.geometry import Polygon, MultiPolygon
except ImportError:
    print("Please install shapely: pip install shapely", file=sys.stderr)
    sys.exit(1)


WGS84_GEOD = Geod(ellps="WGS84")


def parse_heading_buckets(row: pd.Series, prefix: str = "heading_bucket_") -> Optional[float]:
    # Weighted circular mean of 16 bucket centers (each 22.5° wide)
    items = []
    for col in row.index:
        if col.startswith(prefix):
            try:
                i = int(col[len(prefix):])
            except ValueError:
                continue
            w = row[col]
            if pd.notna(w):
                items.append((i, float(w)))
    if not items:
        return None

    items.sort(key=lambda x: x[0])
    sin_sum = 0.0
    cos_sum = 0.0
    total_w = 0.0
    for i, w in items:
        phi_deg = (i + 0.5) * (360.0 / 16.0)
        phi_rad = math.radians(phi_deg)
        sin_sum += w * math.sin(phi_rad)
        cos_sum += w * math.cos(phi_rad)
        total_w += w

    if total_w == 0.0 or (abs(sin_sum) < 1e-15 and abs(cos_sum) < 1e-15):
        i_max = max(items, key=lambda x: x[1])[0]
        return ((i_max + 0.5) * (360.0 / 16.0)) % 360.0

    theta_rad = math.atan2(sin_sum, cos_sum)
    return (math.degrees(theta_rad)) % 360.0


def rect_corners_en(lat: float, lon: float, a: float, b: float, theta_deg: float) -> List[Tuple[float, float]]:
    # 4 geographic corners (lon, lat) of oriented rectangle centered at (lat, lon)
    th = math.radians(theta_deg)
    u_long = (math.sin(th), math.cos(th))      # East, North per +1 m along vessel axis
    u_width = (math.cos(th), -math.sin(th))    # East, North per +1 m to starboard
    corners = []
    for s in (+a, -a):
        for t in (+b, -b):
            dx = s * u_long[0] + t * u_width[0]
            dy = s * u_long[1] + t * u_width[1]
            rho = math.hypot(dx, dy)
            az_deg = 0.0 if rho == 0.0 else (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
            lon_i, lat_i, _ = WGS84_GEOD.fwd(lon, lat, az_deg, rho)
            corners.append((lon_i, lat_i))
    return corners


def minimal_lon_interval(lons_deg: List[float]) -> Tuple[float, float, bool]:
    # Minimal-width longitude interval on the circle; flags antimeridian crossing
    if not lons_deg:
        return -180.0, 180.0, False
    L = [((lon + 360.0) % 360.0) for lon in lons_deg]
    L.sort()
    L2 = L + [x + 360.0 for x in L]
    n = len(L)
    best_width = float("inf")
    best_start = L2[0]
    best_end = L2[0]
    for i in range(n):
        start = L2[i]
        end = L2[i + n - 1]
        width = end - start
        if width < best_width:
            best_width = width
            best_start = start
            best_end = end
    eps = 1e-12
    if best_end <= 180.0 + eps:
        lon_min = best_start
        lon_max = best_end
        crosses = False
    elif best_start >= 180.0 - eps:
        lon_min = best_start - 360.0
        lon_max = best_end - 360.0
        crosses = False
    else:
        lon_min = best_start           # (0, 180)
        lon_max = best_end - 360.0     # (-180, 0)
        crosses = True

    def clamp180(x):
        if x > 180.0: return 180.0
        if x < -180.0: return -180.0
        return x

    return clamp180(lon_min), clamp180(lon_max), crosses


def bbox_coords(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> List[Tuple[float, float]]:
    return [
        (lon_min, lat_min),
        (lon_max, lat_min),
        (lon_max, lat_max),
        (lon_min, lat_max),
        (lon_min, lat_min),
    ]


def bbox_wkt_polygon(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> str:
    coords = bbox_coords(lat_min, lat_max, lon_min, lon_max)
    def fmt(x: float) -> str:
        s = f"{x:.13f}"
        return s.rstrip('0').rstrip('.') if '.' in s else s
    pairs = ", ".join(f"{fmt(lat)} {fmt(lon)}" for lon, lat in coords)
    return f"POLYGON(({pairs}))"


def bbox_wkt_multipolygon_antimeridian(lat_min: float, lat_max: float, lon_min_wrapped: float, lon_max_wrapped: float) -> str:
    ring_a = bbox_coords(lat_min, lat_max, lon_min_wrapped, 180.0)
    ring_b = bbox_coords(lat_min, lat_max, -180.0, lon_max_wrapped)
    def fmt(x: float) -> str:
        s = f"{x:.13f}"
        return s.rstrip('0').rstrip('.') if '.' in s else s
    part_a = ", ".join(f"{fmt(lat)} {fmt(lon)}" for lon, lat in ring_a)
    part_b = ", ".join(f"{fmt(lat)} {fmt(lon)}" for lon, lat in ring_b)
    return f"MULTIPOLYGON((({part_a})),(({part_b})))"


def shapely_bbox_geometry(lat_min: float, lat_max: float, lon_min: float, lon_max: float, crosses: bool):
    if not crosses:
        return Polygon(bbox_coords(lat_min, lat_max, lon_min, lon_max))
    poly_a = Polygon(bbox_coords(lat_min, lat_max, lon_min, 180.0))
    poly_b = Polygon(bbox_coords(lat_min, lat_max, -180.0, lon_max))
    return MultiPolygon([poly_a, poly_b])


def sanitize_scene_name(name: str) -> str:
    if name is None or (isinstance(name, float) and math.isnan(name)):
        return "scene_unknown"
    safe = re.sub(r'[^A-Za-z0-9._-]+', '_', str(name))
    safe = re.sub(r'_+', '_', safe).strip('_')
    return safe[:200] if len(safe) > 200 else safe


def normalize_for_match(name: str) -> str:
    """
    Normalize scene/file name for matching:
      - strip whitespace
      - lower-case
      - remove trailing ".safe"
    Does not alter other parts; aims to match typical Sentinel product names.
    """
    if name is None:
        return ""
    s = str(name).strip().lower()
    if s.endswith(".safe"):
        s = s[:-5]
    return s


def main():
    if len(sys.argv) != 5:
        print("Usage: python update_template_annotations_and_shp.py input.csv template.json output.json output_shp_dir", file=sys.stderr)
        sys.exit(2)

    input_csv = sys.argv[1]
    template_json = sys.argv[2]
    output_json = sys.argv[3]
    output_shp_dir = sys.argv[4]

    # Load inputs
    df = pd.read_csv(input_csv)

    with open(template_json, "r", encoding="utf-8") as f:
        template = json.load(f)

    # Build mapping from template images to ids
    images = template.get("images", [])
    file_to_id_exact = {img.get("file_name"): img.get("id") for img in images if "file_name" in img and "id" in img}
    file_to_id_norm = {normalize_for_match(img.get("file_name")): img.get("id") for img in images if "file_name" in img and "id" in img}

    # Create annotations list
    annotations = []
    shp_records_per_scene: Dict[str, List[Dict]] = {}
    skipped = 0
    ann_id = 1

    for _, row in df.iterrows():
        # Scene association: exact match first, then normalized match
        scene = row["scene_id"] if "scene_id" in row else None
        image_id = None
        if scene in file_to_id_exact:
            image_id = file_to_id_exact[scene]
        else:
            norm_scene = normalize_for_match(scene)
            image_id = file_to_id_norm.get(norm_scene, None)

        if image_id is None:
            # No matching image in template -> skip to keep template intact
            skipped += 1
            print(f"This scene_id has no matching image in template, skipping: {scene}", file=sys.stderr)
            continue

        # Required geometry inputs
        try:
            lat = float(row["lat"])
            lon = float(row["lon"])
            length_m = float(row["vessel_length_m"])
            width_m = float(row["vessel_width_m"])
        except Exception:
            skipped += 1
            print(f"Invalid geometry data, skipping row for scene_id: {scene}", file=sys.stderr)
            continue

        if not (math.isfinite(length_m) and math.isfinite(width_m) and length_m > 0 and width_m > 0):
            skipped += 1
            print(f"Non-positive or non-finite vessel dimensions, skipping row for scene_id: {scene}", file=sys.stderr)
            continue

        theta_deg = parse_heading_buckets(row)
        if theta_deg is None:
            skipped += 1
            print(f"Could not parse heading buckets, skipping row for scene_id: {scene}", file=sys.stderr)
            continue

        a = 0.5 * length_m
        b = 0.5 * width_m

        # Oriented rectangle corners, then minimal lat/lon-aligned bbox
        corners = rect_corners_en(lat, lon, a, b, theta_deg)
        lats = [c[1] for c in corners]
        lons = [c[0] for c in corners]
        lat_min = min(lats)
        lat_max = max(lats)
        lon_min, lon_max, crosses = minimal_lon_interval(lons)

        # WKT for JSON annotation
        wkt = bbox_wkt_polygon(lat_min, lat_max, lon_min, lon_max) if not crosses \
            else bbox_wkt_multipolygon_antimeridian(lat_min, lat_max, lon_min, lon_max)

        # Score (optional)
        score = None
        if "score" in row and pd.notna(row["score"]):
            try:
                score = float(row["score"])
            except Exception:
                score = None

        annotations.append({
            "id": ann_id,
            "image_id": int(image_id),
            "category_id": 1,
            "bbox": wkt,
            "score": score if score is not None else None
        })

        # Shapefile record grouped by scene_id
        safe_scene = sanitize_scene_name(scene)
        geom = shapely_bbox_geometry(lat_min, lat_max, lon_min, lon_max, crosses)
        shp_records_per_scene.setdefault(safe_scene, []).append({
            "geometry": geom,
            "id": ann_id,
            "image_id": int(image_id),
            "score": score if score is not None else None,
        })

        ann_id += 1

    # Update template's annotations and write output JSON
    template_out = dict(template)  # shallow copy is fine; we only replace 'annotations'
    template_out["annotations"] = annotations
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(template_out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(annotations)} annotations to {output_json}. Skipped {skipped} rows (no image match or insufficient data).", file=sys.stderr)

    # Write shapefiles per scene (includes scenes even if not in template's images)
    os.makedirs(output_shp_dir, exist_ok=True)
    for safe_scene, recs in shp_records_per_scene.items():
        gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:4326")
        scene_dir = os.path.join(output_shp_dir, safe_scene)
        os.makedirs(scene_dir, exist_ok=True)
        shp_path = os.path.join(scene_dir, f"{safe_scene}.shp")
        gdf.to_file(shp_path, driver="ESRI Shapefile")
        print(f"Wrote shapefile: {shp_path} ({len(gdf)} features)", file=sys.stderr)


if __name__ == "__main__":
    main()
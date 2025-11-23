#!/usr/bin/env python3
import argparse, os, glob, json, hashlib
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import rasterio, cv2
from rasterio.enums import Resampling
from rasterio.warp import reproject
import fiona

MANIFEST_NAME = "manifest.json"
PER_SAFE_GEOJSON_SUFFIX = "_detections.geojson"
COMBINED_NAME = "detections_combined.geojson"
LOCKFILE = ".lock"

# ---------------- Common helpers ----------------

def _first(pattern):
    # recursive-aware glob: supports ** patterns
    x = sorted(glob.glob(pattern, recursive=True))
    return x[0] if x else None

def load_manifest(out_dir):
    p = os.path.join(out_dir, MANIFEST_NAME)
    if os.path.exists(p):
        with open(p, "r") as f:
            return json.load(f)
    return {"version": 2, "safes": {}}

def save_manifest(out_dir, m):
    with open(os.path.join(out_dir, MANIFEST_NAME), "w") as f:
        json.dump(m, f, indent=2)

def lock(out_dir):
    lf = os.path.join(out_dir, LOCKFILE)
    if os.path.exists(lf):
        raise RuntimeError(f"Another run is in progress (found {LOCKFILE}). If not, delete it and retry.")
    open(lf, "w").close()
    return lf

def file_sig(path):
    st = os.stat(path)
    return (os.path.basename(path), int(st.st_mtime), int(st.st_size))

def hash_sig(tuples):
    h = hashlib.md5()
    for t in tuples:
        h.update(str(t).encode("utf-8"))
    return h.hexdigest()

# ---------------- SAFE discovery / loading ----------------

def detect_safe_type(safe_dir):
    name = os.path.basename(safe_dir).upper()
    if name.startswith(("S2A_", "S2B_")): return "S2"
    if name.startswith(("S1A_", "S1B_")): return "S1"
    if glob.glob(os.path.join(safe_dir, "GRANULE", "*", "IMG_DATA", "*_B02*.jp2"), recursive=True): return "S2"
    return "S1"

def core_files_s1(safe_dir):
    tiff = _first(os.path.join(safe_dir, "measurement", "*.tif*"))
    if not tiff:
        cands = sorted(glob.glob(os.path.join(safe_dir, "**", "*.tif*"), recursive=True))
        tiff = cands[0] if cands else None
    return [tiff] if tiff else []

def core_files_s2(safe_dir):
    gran = _first(os.path.join(safe_dir, "GRANULE", "*"))
    if not gran:
        return []
    img_root = os.path.join(gran, "IMG_DATA")

    def find_any(tag):
        # works for compact L1C (no _10m suffix) and classic R10m layouts
        return (
            _first(os.path.join(img_root, "**", f"*_{tag}_10m.[jJ][pP]2")) or
            _first(os.path.join(img_root, "**", f"*_{tag}.[jJ][pP]2")) or
            _first(os.path.join(img_root, "R10m", f"*_{tag}.[jJ][pP]2"))
        )

    b02 = find_any("B02")
    b03 = find_any("B03")
    b04 = find_any("B04")
    b08 = find_any("B08")
    # optional SWIR (20m) and SCL (20m)
    b11 = (
        _first(os.path.join(img_root, "**", f"*_B11_20m.[jJ][pP]2")) or
        _first(os.path.join(img_root, "**", f"*_B11.[jJ][pP]2")) or
        _first(os.path.join(img_root, "R20m", f"*_B11.[jJ][pP]2"))
    )
    scl = (
        _first(os.path.join(img_root, "**", f"*_SCL_20m.[jJ][pP]2")) or
        _first(os.path.join(img_root, "R20m", f"*_SCL_20m.[jJ][pP]2"))
    )

    return [p for p in [b02, b03, b04, b08, b11, scl] if p]

def load_s1_safe(safe_dir, gdal_threads=None):
    tiff = core_files_s1(safe_dir)[0]
    env = rasterio.Env(GDAL_NUM_THREADS=gdal_threads, NUM_THREADS=gdal_threads) if gdal_threads else rasterio.Env()
    with env:
        with rasterio.open(tiff) as src:
            arr = src.read(1).astype(np.float32)
            crs = src.crs; transform = src.transform
            h, w = arr.shape
    p1, p99 = np.percentile(arr,1), np.percentile(arr,99)
    arr = np.clip((arr - p1) / (p99 - p1 + 1e-6), 0, 1)
    return arr[None, ...], crs, transform, w, h

def load_s2_safe_fast(safe_dir, use_bands=("B03","B08"), want_b11=True, want_scl=False, gdal_threads=None):
    """
    Returns tuple:
      stack: np.float32 with bands in order [Green(10m), NIR(10m), (SWIR10m if available)], shape (C,H,W)
      aux: dict possibly containing {'SWIR20': swir20, 'SCL20': scl20, 'crs', 'transform'}
    """
    gran = _first(os.path.join(safe_dir, "GRANULE", "*"))
    if not gran:
        raise RuntimeError(f"No GRANULE in {safe_dir}")
    img_root = os.path.join(gran, "IMG_DATA")

    def find_any(tag):
        return (
            _first(os.path.join(img_root, "**", f"*_{tag}_10m.[jJ][pP]2")) or
            _first(os.path.join(img_root, "**", f"*_{tag}.[jJ][pP]2")) or
            _first(os.path.join(img_root, "R10m", f"*_{tag}.[jJ][pP]2"))
        )
    def open_band(p):
        with rasterio.open(p) as s: return s.read(1).astype(np.float32), s.crs, s.transform

    env = rasterio.Env(GDAL_NUM_THREADS=gdal_threads, NUM_THREADS=gdal_threads) if gdal_threads else rasterio.Env()
    with env:
        # read 10m anchors first to set shape/geo
        b03 = find_any("B03"); b08 = find_any("B08")
        if not b03 or not b08:
            raise RuntimeError(f"Missing B03/B08 in {safe_dir}")
        g, crs, transform = open_band(b03)
        n, _, _ = open_band(b08)
        h, w = g.shape

        bands = [g, n]  # [Green, NIR]
        aux = {"crs": crs, "transform": transform}

        if want_b11:
            b11 = (
                _first(os.path.join(img_root, "**", f"*_B11_20m.[jJ][pP]2")) or
                _first(os.path.join(img_root, "R20m", f"*_B11.[jJ][pP]2")) or
                _first(os.path.join(img_root, "**", f"*_B11.[jJ][pP]2"))
            )
            if b11:
                with rasterio.open(b11) as s:
                    swir20 = s.read(1).astype(np.float32); t20 = s.transform; crs20 = s.crs
                # resample SWIR 20m to 10m (fast bilinear)
                swir10 = np.empty((h, w), dtype=np.float32)
                reproject(swir20, swir10, src_transform=t20, src_crs=crs20,
                          dst_transform=transform, dst_crs=crs, resampling=Resampling.bilinear)
                bands.append(swir10)
                aux["SWIR20"] = swir20  # keep if caller wants 20 m mask path

        if want_scl:
            scl = (
                _first(os.path.join(img_root, "**", f"*_SCL_20m.[jJ][pP]2")) or
                _first(os.path.join(img_root, "R20m", f"*_SCL_20m.[jJ][pP]2"))
            )
            if scl:
                with rasterio.open(scl) as s:
                    scl20 = s.read(1).astype(np.uint16)  # classification labels
                aux["SCL20"] = scl20

    stack = np.stack(bands, axis=0)
    return stack, aux

# ---------------- Detection kernels ----------------

def ndwi(g, n): return (g - n) / (g + n + 1e-6)
def mndwi(g, s): return (g - s) / (g + s + 1e-6)

def lee_filter(img, size=5):
    k = cv2.blur(img, (size, size)); k2 = cv2.blur(img*img, (size, size))
    var = k2 - k*k; noise = np.median(var)
    w = var / (var + noise + 1e-6)
    return k + w*(img - k)

def cfar(img, ksize=25, k=3.0):
    mu = cv2.blur(img, (ksize, ksize)); mu2 = cv2.blur(img*img, (ksize, ksize))
    sigma = np.sqrt(np.maximum(mu2 - mu*mu, 0))
    return (img > (mu + k*sigma)).astype(np.uint8)

def percentile_thr(vals, p50=50, p99=99, alpha=0.25):
    if vals.size == 0: return 1.0
    q50, q99 = np.percentile(vals, [p50, p99])
    return q99 - alpha * (q99 - q50)

def CC_labels(mask, min_area, max_area):
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = []
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            ys, xs = np.where(lab==i)
            comps.append((ys, xs))
    return comps

def pxbbox_to_lonlat(ys, xs, H, W, transform):
    y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
    pad=2; x1=max(0,x1-pad); y1=max(0,y1-pad); x2=min(W-1,x2+pad); y2=min(H-1,y2+pad)
    xs_=[x1,x2,x2,x1,x1]; ys_=[y1,y1,y2,y2,y1]
    coords=[]
    for X,Y in zip(xs_, ys_):
        lon, lat = rasterio.transform.xy(transform, Y, X)
        coords.append((lon, lat))
    return coords

def detect_S1(stack, transform, min_area, max_area, cfar_k):
    I = stack[0]
    I_f = lee_filter(I, size=5)
    mask = cfar(I_f, ksize=25, k=cfar_k)
    comps = CC_labels(mask, min_area, max_area)
    H, W = I.shape
    polys = [({"type":"Polygon","coordinates":[pxbbox_to_lonlat(ys,xs,H,W,transform)]},
              float(np.mean(I_f[ys, xs]))) for (ys,xs) in comps]
    return polys

def build_water_mask_EO(G, N, SWIR10=None, SCL20=None, transform=None, use_scl=False):
    if use_scl and SCL20 is not None:
        # SCL classes: 6=Water (L2A); include only water class
        # Upsample to 10 m
        h10, w10 = G.shape
        scl_bin = (SCL20 == 6).astype(np.uint8)
        mask10 = cv2.resize(scl_bin, (w10, h10), interpolation=cv2.INTER_NEAREST)
        return mask10
    if SWIR10 is not None:
        return (mndwi(G, SWIR10) > 0.05).astype(np.uint8)
    return (ndwi(G, N) > 0.0).astype(np.uint8)

def detect_S2_fast(stack, aux, min_area, max_area, global_thr, tile, overlap):
    """
    stack: [Green(10m), NIR(10m), (optional SWIR10m)]
    """
    G = stack[0]; N = stack[1]
    SWIR10 = stack[2] if stack.shape[0] >= 3 else None
    wmask = build_water_mask_EO(G, N, SWIR10=SWIR10,
                                SCL20=aux.get("SCL20"), transform=aux["transform"],
                                use_scl=("SCL20" in aux))

    # grayscale proxy (fast): just Green
    Gray = G

    def tile_func(g, wm):
        vals = g[wm==1]
        if global_thr:
            thr = percentile_thr(vals, 50, 99, 0.25)
            m = ((g > thr).astype(np.uint8)) * wm
        else:
            # fallback: simple Otsu on water pixels (still fast)
            if vals.size > 0:
                thr, _ = cv2.threshold((vals*255).astype(np.uint8), 0, 255, cv2.THRESH_OTSU)
                thr = thr/255.0
            else:
                thr = 1.0
            m = ((g > thr).astype(np.uint8)) * wm
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        return m

    if tile > 0:
        H, W = Gray.shape
        out = np.zeros((H, W), dtype=np.uint8)
        step = max(64, tile - overlap)
        for y in range(0, H, step):
            for x in range(0, W, step):
                y2 = min(H, y + tile); x2 = min(W, x + tile)
                g_tile = Gray[y:y2, x:x2]; wm_tile = wmask[y:y2, x:x2]
                m_tile = tile_func(g_tile, wm_tile)
                oy1 = y + (overlap//2 if y>0 else 0)
                ox1 = x + (overlap//2 if x>0 else 0)
                out[oy1:y2, ox1:x2] = m_tile[(oy1-y):(y2-y), (ox1-x):(x2-x)]
        mask = out
    else:
        mask = tile_func(Gray, wmask)

    comps = CC_labels(mask, min_area, max_area)
    H, W = Gray.shape
    polys = [({"type":"Polygon","coordinates":[pxbbox_to_lonlat(ys,xs,H,W,aux["transform"])]},
              float(np.mean(Gray[ys, xs]))) for (ys,xs) in comps]
    return polys

# ---------------- Outputs ----------------

def write_per_safe_outputs(out_dir, safe_name, crs, polys):
    # Shapefile
    schema = {"geometry":"Polygon","properties":{"score":"float"}}
    shp_path = os.path.join(out_dir, f"{safe_name}_detections.shp")
    with fiona.open(shp_path, "w", driver="ESRI Shapefile", crs=crs, schema=schema) as shp:
        for poly, sc in polys:
            shp.write({"geometry": poly, "properties":{"score": sc}})
    # Per-SAFE geojson
    gpath = os.path.join(out_dir, f"{safe_name}{PER_SAFE_GEOJSON_SUFFIX}")
    data = {"images":[{"id":1,"file_name":f"{safe_name}.SAFE","width":0,"height":0}],"annotations":[]}
    for i,(poly, sc) in enumerate(polys, start=1):
        coords = [[list(pt) for pt in poly["coordinates"][0]]]
        data["annotations"].append({"image_id":1,"category_id":1,
                                    "bbox":{"type":"Polygon","coordinates":coords},
                                    "score":float(sc),"id":i})
    with open(gpath, "w") as f: json.dump(data, f)
    return shp_path, gpath

def rebuild_combined(out_dir):
    per = sorted(glob.glob(os.path.join(out_dir, f"*{PER_SAFE_GEOJSON_SUFFIX}")))
    combined = {
        "info":{"description":"Grand challenge MDA","version":"1.0"},
        "licenses":[],
        "images":[],
        "categories":[{"id":1,"name":"ship"}],
        "annotations":[]
    }
    img_id = 1; ann_id = 1
    for p in per:
        with open(p) as f: d = json.load(f)
        safe_name = os.path.basename(p).replace(PER_SAFE_GEOJSON_SUFFIX,"")
        combined["images"].append({"id":img_id,"file_name":f"{safe_name}.SAFE","width":0,"height":0})
        for a in d.get("annotations", []):
            combined["annotations"].append({
                "image_id": img_id, "category_id": 1,
                "bbox": a["bbox"], "score": float(a.get("score",0.0)), "id": ann_id
            })
            ann_id += 1
        img_id += 1
    with open(os.path.join(out_dir, COMBINED_NAME), "w") as f:
        json.dump(combined, f)
    return len(per), ann_id-1

# ---------------- Worker (per SAFE) ----------------

def process_safe(sd, args):
    sname = os.path.basename(sd)
    stype = detect_safe_type(sd)
    # compute signature for incremental behavior
    files = core_files_s2(sd) if stype=="S2" else core_files_s1(sd)
    if not files:
        return (sname, stype, "skip", 0, "no core files")

    sig = hash_sig([file_sig(f) for f in files])
    # skip unchanged unless --force
    if not args.force:
        entry = args.manifest["safes"].get(sname)
        if entry and entry.get("signature") == sig and entry.get("status","ok").startswith("ok"):
            return (sname, stype, "unchanged", entry.get("detections",0), "")

    try:
        if stype == "S1":
            stack, crs, transform, W, H = load_s1_safe(sd, gdal_threads=args.gdal_threads)
            polys = detect_S1(stack, transform, args.min_area, args.max_area, args.cfar_k)
        else:
            stack, aux = load_s2_safe_fast(
                sd,
                use_bands=("B03","B08"),
                want_b11=args.eo_fast,            # read SWIR only if eo_fast true (for better water mask)
                want_scl=args.use_scl,           # try to load SCL if requested
                gdal_threads=args.gdal_threads
            )
            aux["transform"] = aux["transform"]; crs = aux["crs"]
            polys = detect_S2_fast(
                stack, aux,
                args.min_area, args.max_area,
                global_thr=args.global_thr,
                tile=args.tile, overlap=args.overlap
            )
        shp_path, gjson_path = write_per_safe_outputs(args.out_dir, sname, crs, polys)
        args.manifest["safes"][sname] = {
            "type": stype, "signature": sig,
            "detections": len(polys),
            "shapefile": os.path.basename(shp_path),
            "geojson": os.path.basename(gjson_path),
            "last_run": datetime.utcnow().isoformat()+"Z",
            "status": "ok"
        }
        return (sname, stype, "ok", len(polys), "")
    except Exception as e:
        args.manifest["safes"][sname] = {
            "type": stype, "signature": sig,
            "detections": 0,
            "last_run": datetime.utcnow().isoformat()+"Z",
            "status": f"error: {e}"
        }
        return (sname, stype, "error", 0, str(e))

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(description="Incremental PS-09 SAFE detector (S1/S2) — fast options")
    ap.add_argument("--safe_root", required=True, help="Folder containing *.SAFE directories")
    ap.add_argument("--out_dir", required=True, help="Where outputs & manifest live")
    ap.add_argument("--only", default="", help="Process only SAFE names containing this substring (optional)")
    ap.add_argument("--min_area", type=int, default=5)
    ap.add_argument("--max_area", type=int, default=600)
    ap.add_argument("--cfar_k", type=float, default=3.0)
    ap.add_argument("--force", action="store_true", help="Reprocess even if unchanged")
    ap.add_argument("--dry_run", action="store_true", help="Show what would be processed, do nothing")

    # Speed toggles
    ap.add_argument("--workers", type=int, default=1, help="Parallel SAFEs (default 1). Try CPU count for throughput.")
    ap.add_argument("--gdal_threads", default=None, help="Set GDAL_NUM_THREADS/NUM_THREADS (e.g., ALL_CPUS or 8)")
    ap.add_argument("--eo_fast", action="store_true", help="Use fewer bands; SWIR only if available; water-first pipeline")
    ap.add_argument("--global_thr", action="store_true", help="Use fast percentile threshold instead of adaptive")
    ap.add_argument("--tile", type=int, default=0, help="Tile size for windowed processing (0=off). Try 1024.")
    ap.add_argument("--overlap", type=int, default=32, help="Overlap for tiling (ignored if tile=0)")
    ap.add_argument("--use_scl", action="store_true", help="If S2 L2A SCL is present, use class 6 (water) mask")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Lock to avoid concurrent runs clobbering manifest
    lf = lock(args.out_dir)
    try:
        manifest = load_manifest(args.out_dir)
        args.manifest = manifest  # pass into worker (read/write after)

        # discover safes
        safes = sorted([p for p in glob.glob(os.path.join(args.safe_root, "*.SAFE")) if os.path.isdir(p)])
        if args.only:
            safes = [p for p in safes if args.only in os.path.basename(p)]

        if args.dry_run:
            # Determine what would process vs skip
            plan = []
            for sd in safes:
                sname = os.path.basename(sd)
                stype = detect_safe_type(sd)
                files = core_files_s2(sd) if stype=="S2" else core_files_s1(sd)
                if not files:
                    plan.append((sname, "skip: no core files"))
                    continue
                sig = hash_sig([file_sig(f) for f in files])
                entry = manifest["safes"].get(sname)
                if args.force or entry is None or entry.get("signature") != sig or not entry.get("status","ok").startswith("ok"):
                    plan.append((sname, "process"))
                else:
                    plan.append((sname, "unchanged"))
            print("Dry run plan:")
            for n, st in plan: print(f"  {n}: {st}")
            return

        # Parallel (per SAFE) if requested
        if args.workers and args.workers > 1:
            work = partial(process_safe, args=args)
            with Pool(processes=args.workers) as pool:
                for sname, stype, status, n, msg in pool.imap_unordered(work, safes):
                    if status == "ok":
                        print(f"[OK] {sname} ({stype}) — {n} detections")
                    elif status == "unchanged":
                        print(f"[=]  {sname} unchanged — {n} detections (skipped)")
                    elif status == "skip":
                        print(f"[!]  {sname} skipped: {msg}")
                    else:
                        print(f"[X]  {sname} error: {msg}")
                    save_manifest(args.out_dir, manifest)
        else:
            for sd in safes:
                sname, stype, status, n, msg = process_safe(sd, args)
                if status == "ok":
                    print(f"[OK] {sname} ({stype}) — {n} detections")
                elif status == "unchanged":
                    print(f"[=]  {sname} unchanged — {n} detections (skipped)")
                elif status == "skip":
                    print(f"[!]  {sname} skipped: {msg}")
                else:
                    print(f"[X]  {sname} error: {msg}")
                save_manifest(args.out_dir, manifest)

        # Rebuild combined every run
        n_imgs, n_anns = rebuild_combined(args.out_dir)
        print(f"Combined GeoJSON updated: {COMBINED_NAME} (images={n_imgs}, detections={n_anns})")

    finally:
        try: os.remove(lf)
        except: pass

if __name__ == "__main__":
    main()

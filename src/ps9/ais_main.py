# main.py
# Orchestrator: parse satellite CSV, select AIS dates (±6h), download/extract, run full cleaning, and save per-day files
import argparse
from pathlib import Path
import pandas as pd

from download_extract import build_noaa_ais_urls, prepare_inputs
from cleaning import CleanConfig, AISCleaner

def parse_image_csv(csv_path, only_ais_correlations=True, ts_col_candidates=("time_stamp","timestamp"),
                    remarks_col="remarks"):
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    # Pick a timestamp column
    ts_col = None
    for c in ts_col_candidates:
        if c.lower() in df.columns:
            ts_col = c.lower(); break
    if ts_col is None:
        raise ValueError(f"No timestamp column found among {ts_col_candidates}")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col])

    if only_ais_correlations and remarks_col.lower() in df.columns:
        # Restrict to images explicitly flagged for AIS correlation [2]
        mask = df[remarks_col.lower()].fillna("").str.contains("For AIS correlations", case=False, regex=False)
        df = df.loc[mask]
    return df[ts_col].tolist()

def dates_from_timestamps(timestamps, window_hours=6):
    dates = set()
    for t in timestamps:
        start = t - pd.Timedelta(hours=window_hours)
        end   = t + pd.Timedelta(hours=window_hours)
        for d in pd.date_range(start.floor("D"), end.floor("D"), freq="D"):
            dates.add(d.date())
    return sorted(dates)

def main():
    ap = argparse.ArgumentParser(description="NOAA AIS download + full clean for satellite image correlation (±6h).")
    ap.add_argument("--images_csv", required=True, help="CSV of satellite images (must include time_stamp and remarks).")
    ap.add_argument("--window_hours", type=int, default=6, help="Half-window in hours around image timestamps.")
    ap.add_argument("--work_dir", default="data/working", help="Download staging dir.")
    ap.add_argument("--extract_dir", default="data/extracted", help="Extraction staging dir.")
    ap.add_argument("--out_dir", default="data/cleaned", help="Output dir for cleaned per-day files.")
    ap.add_argument("--max_workers", type=int, default=4, help="Parallel workers for download/extract.")
    ap.add_argument("--save_format", choices=["parquet","csv"], default="parquet")
    ap.add_argument("--context", choices=["port","offshore"], default="port")
    ap.add_argument("--aoi_shp", default=None, help="Optional AOI shapefile path (WGS84 or will be reprojected).")
    ap.add_argument("--landmask_shp", default=None, help="Optional landmask shapefile path.")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 1) Parse satellite CSV; by default only rows marked for AIS correlation [2]
    timestamps = parse_image_csv(args.images_csv, only_ais_correlations=True)

    if not timestamps:
        print("No timestamps found to process (check the CSV and remarks filter).")
        return

    # 2) Compute AIS days across ±window_hours and build NOAA URLs
    dates = dates_from_timestamps(timestamps, window_hours=args.window_hours)
    urls = build_noaa_ais_urls(dates)

    # 3) Download and extract raw AIS
    data_files = prepare_inputs(urls, work_dir=args.work_dir, extract_dir=args.extract_dir, max_workers=args.max_workers)
    if not data_files:
        print("No data files were staged. Exiting.")
        return

    # 4) Group staged files by day (from filename) and clean per-day
    import re
    date_pat = re.compile(r"(20\d{2})[-_](\d{2})[-_](\d{2})")

    def date_from_name(p):
        m = date_pat.search(Path(p).name.lower())
        if not m: return None
        y, mth, dd = map(int, m.groups())
        return pd.Timestamp(year=y, month=mth, day=dd, tz="UTC").date()

    files_by_date = {}
    for f in data_files:
        d = date_from_name(f)
        if d is not None:
            files_by_date.setdefault(d, []).append(f)

    cfg = CleanConfig(context=args.context,
                      aoi_polygon=args.aoi_shp,
                      landmask_polygon=args.landmask_shp,
                      enforce_domains=True,
                      round_to_resolution=True)

    cleaner = AISCleaner(cfg)

    outputs = []
    for d in sorted(files_by_date):
        paths = files_by_date[d]
        cleaned_df, audit = cleaner.run(paths)
        out_stem = f"noaa_ais_cleaned_{d.strftime('%Y%m%d')}"
        out_path = Path(args.out_dir) / (out_stem + (".parquet" if args.save_format == "parquet" else ".csv"))
        if args.save_format == "parquet":
            cleaned_df.to_parquet(out_path, index=False)
        else:
            cleaned_df.to_csv(out_path, index=False)
        outputs.append(str(out_path))
        print(f"Saved {out_path} with {len(cleaned_df):,} rows.")

    print("Done. Outputs:")
    for p in outputs:
        print(" -", p)

if __name__ == "__main__":
    main()
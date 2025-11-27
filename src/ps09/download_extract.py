# download_extract.py
# NOAA AIS downloader and extractor for 2018–2024 .zip and 2025+ .csv.zst
import os, time, gzip, shutil, zipfile, subprocess
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

def build_noaa_ais_urls(dates):
    """
    dates: iterable of datetime.date
    2024 and earlier: .../AIS_YYYY_MM_DD.zip
    2025 and later:   .../ais-YYYY-MM-DD.csv.zst
    """
    base = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler"
    urls = []
    for d in dates:
        y, m, dd = d.year, d.month, d.day
        if y >= 2025:
            urls.append(f"{base}/{y}/ais-{y}-{m:02d}-{dd:02d}.csv.zst")
        else:
            urls.append(f"{base}/{y}/AIS_{y}_{m:02d}_{dd:02d}.zip")
    return urls

def is_url(s):
    try:
        return urlparse(str(s)).scheme in ("http","https")
    except Exception:
        return False

def safe_filename_from_url(url):
    name = os.path.basename(urlparse(url).path)
    return name or f"d_{int(time.time()*1000)}"


def find_expected_extracted_files(archive_path):
    """
    Given an archive filename, predict what extracted file(s) should be present.
    Used to decide whether we can skip download & extraction.
    """
    p = Path(archive_path)
    suf = p.suffix.lower()
    suffixes = "".join(p.suffixes).lower()

    if suf == ".zip":
        # NOAA zip ALWAYS contains one CSV per day
        return [p.with_suffix("").name + ".csv"]
    elif suf == ".gz":
        return [p.with_suffix("").name]
    elif suf == ".zst" or suffixes.endswith(".csv.zst"):
        # zst decompresses to .csv
        return [p.with_suffix("").name]
    else:
        return [p.name]


def extracted_already_exists(path, extract_dir):
    """
    Check if *all* expected extracted files exist in extract_dir.
    """
    expected = find_expected_extracted_files(path)
    return all((Path(extract_dir) / e).exists() for e in expected)


def download_file(url, dest_dir, timeout=120, retries=3, chunk=1 << 20):
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    out = dest_dir / safe_filename_from_url(url)

    # If archive already downloaded → skip
    if out.exists() and out.stat().st_size > 0:
        return str(out)

    last = None
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(out, "wb") as f:
                    for b in r.iter_content(chunk_size=chunk):
                        if b:
                            f.write(b)
            return str(out)
        except Exception as e:
            last = e
            time.sleep(2.0 * (attempt + 1))

    raise RuntimeError(f"Download failed {url}: {last}")


def _decompress_zst(src_path, dst_path):
    try:
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        with open(src_path, "rb") as fin, open(dst_path, "wb") as fout:
            dctx.copy_stream(fin, fout)
    except ImportError:
        if shutil.which("zstd") is None:
            raise RuntimeError("Install 'zstandard' (pip) or provide 'zstd' CLI in PATH.")
        subprocess.run(["zstd","-d","-f","-o",str(dst_path),str(src_path)], check=True)

def extract_archive(path, extract_dir):
    extract_dir = Path(extract_dir); extract_dir.mkdir(parents=True, exist_ok=True)
    p = Path(path)

    # Skip extraction if already exists
    if extracted_already_exists(p, extract_dir):
        expected = find_expected_extracted_files(p)
        return [str(extract_dir / e) for e in expected]

    out_paths = []
    suf = p.suffix.lower()
    suffixes = "".join(p.suffixes).lower()

    if suf == ".zip":
        with zipfile.ZipFile(p, "r") as z:
            for m in z.infolist():
                if m.is_dir():
                    continue
                name = Path(m.filename).name
                target = extract_dir / name
                if not target.exists():
                    z.extract(m, extract_dir)
                out_paths.append(str(target))

    elif suf == ".gz":
        target = extract_dir / p.with_suffix("").name
        if not target.exists():
            with gzip.open(p, "rb") as fin, open(target, "wb") as fout:
                shutil.copyfileobj(fin, fout)
        out_paths.append(str(target))

    elif suf == ".zst" or suffixes.endswith(".csv.zst"):
        target = extract_dir / p.with_suffix("").name
        if not target.exists():
            _decompress_zst(str(p), str(target))
        out_paths.append(str(target))

    else:
        out_paths.append(str(p))

    return out_paths


def prepare_inputs(paths_or_urls, work_dir="data/working", extract_dir="data/extracted", max_workers=4):
    items = paths_or_urls if isinstance(paths_or_urls, (list, tuple)) else [paths_or_urls]
    work_dir = Path(work_dir)
    extract_dir = Path(extract_dir)

    work_dir.mkdir(parents=True, exist_ok=True)
    extract_dir.mkdir(parents=True, exist_ok=True)

    staged = []

    # ---- DOWNLOAD ONLY WHAT IS NEEDED ----
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = []
        for u in items:
            if is_url(u):
                # Predict future local archive name
                filename = safe_filename_from_url(u)
                local_archive_path = work_dir / filename

                # If extracted exists, skip downloading
                if extracted_already_exists(local_archive_path, extract_dir):
                    print(f"[SKIP DOWNLOAD] {u} (already extracted)")
                    staged.append(str(local_archive_path))  # pretend we "downloaded"
                else:
                    futs.append(pool.submit(download_file, u, work_dir))

        for f in as_completed(futs):
            staged.append(f.result())

    # Include local paths too
    for p in items:
        if not is_url(p):
            staged.append(str(p))

    # Extraction
    extracted = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = [pool.submit(extract_archive, p, extract_dir) for p in staged]
        for f in as_completed(futs):
            extracted.extend(f.result())

    # Keep likely tabular data
    keep_ext = {".csv",".txt",".parquet"}
    data_files = [p for p in extracted if Path(p).suffix.lower() in keep_ext]
    return data_files
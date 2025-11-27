#!/usr/bin/env python3
"""
cdse_download.py

Batch-download Sentinel products from Copernicus Data Space Ecosystem (CDSE)
given a TXT/CSV file of product SAFE names (one per line).

Features:
- Interactive / env-var credentials (no secrets on CLI).
- Parallel downloads with bounded concurrency (configurable).
- Robust retry/backoff for 429/5xx; token refresh.
- Graceful handling of duplicates, blanks, and comments in list file.
- Simple resume: if final file exists with the right size, skip.

Usage:
  python cdse_download_batch.py --in products.txt --out ./downloads --workers 6
  # or
  python cdse_download_batch.py --in products.csv --out ./downloads --workers 8

Env (optional):
  export CDSE_USERNAME="your_user"
  export CDSE_PASSWORD="your_pass"
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time
import math
import random
from pathlib import Path
from typing import Optional, Iterable, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter, Retry
from getpass import getpass

IDENTITY_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
ODATA_BASE = "https://catalogue.dataspace.copernicus.eu/odata/v1"
DOWNLOAD_BASE = "https://download.dataspace.copernicus.eu/odata/v1"

# -------- Credentials --------

def prompt_credentials() -> tuple[str, str]:
    user = os.environ.get("CDSE_USERNAME") or input("CDSE username: ").strip()
    pwd = os.environ.get("CDSE_PASSWORD") or getpass("CDSE password: ")
    if not user or not pwd:
        print("Username and password are required.", file=sys.stderr)
        sys.exit(2)
    return user, pwd

# -------- Auth & sessions --------

def get_token(username: str, password: str, client_id: str = "cdse-public") -> dict:
    data = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": client_id,
    }
    r = requests.post(IDENTITY_URL, data=data, timeout=30)
    r.raise_for_status()
    return r.json()

def refresh_token(refresh_token: str, client_id: str = "cdse-public") -> dict:
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
    }
    r = requests.post(IDENTITY_URL, data=data, timeout=30)
    r.raise_for_status()
    return r.json()

def make_retrying_session(access_token: str) -> requests.Session:
    """
    Session with generous retries and backoff for CDSE (429/5xx),
    plus a modest per-request timeout.
    """
    retry = Retry(
        total=10,
        connect=5,
        read=5,
        backoff_factor=0.8,  # exponential backoff with jitter (we add jitter manually, too)
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
        raise_on_status=False,
    )
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50))
    s.headers.update({"Authorization": f"Bearer {access_token}"})
    return s

# -------- Input parsing --------

def load_product_names(path: Path) -> List[str]:
    """
    Accepts a csv file to pick up entries under the header "image_name"
    Lines starting with '#' are ignored. Empty lines are skipped.
    """
    names: List[str] = []
    ext = path.suffix.lower()

    if ext == ".csv":
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if "image_name" not in reader.fieldnames:
                print(f"CSV input must have 'image_name' column.", file=sys.stderr)
                sys.exit(2)
            for row in reader:
                if not row:
                    continue
                name = (row.get("image_name") or "").strip()
                if not name or name.startswith("#"):
                    continue
                names.append(name)
    
    if not names:
        print("No product names found in CSV 'image_name' column.", file=sys.stderr)
        sys.exit(2)
        
    # de-duplicate while preserving order
    seen = set()
    unique = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique

# -------- OData helpers --------

def random_jitter(min_ms=100, max_ms=400):
    """Small sleep to reduce burstiness across many workers."""
    time.sleep(random.uniform(min_ms/1000.0, max_ms/1000.0))

def find_product_id_by_name(
    sess: requests.Session,
    product_name: str,
    token_refresher=None,
    refresh_token_value: Optional[str] = None,
    max_retries: int = 6,
) -> Optional[str]:
    """
    Query OData for exact Name match and return the product UUID (Id).
    Retries/backoff on 429/5xx/403 and refreshes token once on 401/403.
    """
    name_escaped = product_name.replace("'", "''")
    url = f"{ODATA_BASE}/Products?$filter=Name eq '{name_escaped}'&$select=Id,Name,ContentLength"

    attempt = 0
    did_refresh = False
    while attempt <= max_retries:
        # small jitter to spread concurrent calls
        time.sleep(random.uniform(0.08, 0.25))
        try:
            r = sess.get(url, timeout=60)
            if r.status_code in (401, 403) and token_refresher and refresh_token_value and not did_refresh:
                # try one token refresh
                tok = token_refresher(refresh_token_value)
                sess.headers.update({"Authorization": f"Bearer {tok['access_token']}"})
                refresh_token_value = tok.get("refresh_token", refresh_token_value)
                did_refresh = True
                attempt += 1
                continue
            if r.status_code in (429, 500, 502, 503, 504, 403):
                # treat 403 as throttling/WAF sometimes; backoff & retry
                back = min(30, (2 ** attempt) * 0.5 + random.uniform(0.1, 0.6))
                time.sleep(back)
                attempt += 1
                continue

            r.raise_for_status()
            items = r.json().get("value", [])
            if not items:
                return None
            return items[0]["Id"]

        except requests.RequestException:
            back = min(30, (2 ** attempt) * 0.5 + random.uniform(0.1, 0.6))
            time.sleep(back)
            attempt += 1

    raise requests.HTTPError(f"Catalogue lookup failed after retries for {product_name}")

def head_product_size(sess: requests.Session, product_id: str) -> Optional[int]:
    """
    HEAD to learn expected size for skip/resume checks. Some servers may not support HEAD;
    if so, return None and rely on GET Content-Length.
    """
    url = f"{DOWNLOAD_BASE}/Products({product_id})/$value"
    try:
        r = sess.head(url, timeout=60, allow_redirects=True)
        if r.status_code == 200 and "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except requests.RequestException:
        pass
    return None

# -------- Download --------

def infer_filename_from_headers(resp: requests.Response, fallback: str) -> str:
    cd = resp.headers.get("Content-Disposition", "")
    if "filename=" in cd:
        fname = cd.split("filename=", 1)[1].strip().strip('"')
        return fname or fallback
    return fallback

def stream_download(
    sess: requests.Session,
    product_id: str,
    out_dir: Path,
    token_refresher,
    refresh_token_value: Optional[str],
    max_single_file_retries: int = 4,
) -> Path:
    """
    Streams the ZIP using /Products(<id>)/$value with retries and token refresh fallback.
    Simple resume: if final file exists with exact size → skip. Partial ".part" files will be re-fetched from scratch
    (Range resume is attempted but gracefully falls back if unsupported).
    """
    url = f"{DOWNLOAD_BASE}/Products({product_id})/$value"

    # Attempt to know final size (helps skip already-complete files)
    expected_size = head_product_size(sess, product_id)

    # Prepare names
    fallback_name = f"{product_id}.zip"
    # We'll determine 'filename' on first successful GET
    out_path = out_dir / fallback_name
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    # If a previous file exists with expected size, skip
    if expected_size and out_path.exists() and out_path.stat().st_size == expected_size:
        return out_path

    retries = 0
    while retries <= max_single_file_retries:
        try:
            headers = {}
            # Try resuming from .part if server supports ranges
            if tmp_path.exists():
                offset = tmp_path.stat().st_size
                if offset > 0:
                    headers["Range"] = f"bytes={offset}-"

            with sess.get(url, stream=True, timeout=120, allow_redirects=True, headers=headers) as r:
                # Handle auth expiry → one refresh chance
                if r.status_code in (401, 403) and refresh_token_value:
                    tok = token_refresher(refresh_token_value)
                    new_access = tok["access_token"]
                    # Update session header
                    sess.headers.update({"Authorization": f"Bearer {new_access}"})
                    # retry this loop iteration
                    retries += 1
                    continue

                # If resume not supported and we sent Range, start from scratch
                if r.status_code == 416:  # Range not satisfiable
                    tmp_path.unlink(missing_ok=True)
                    retries += 1
                    continue

                r.raise_for_status()

                # Determine final filename from headers (could be SAFE.zip)
                filename = infer_filename_from_headers(r, fallback_name)
                out_path = out_dir / filename
                tmp_path = out_path.with_suffix(out_path.suffix + ".part")

                # Skip if already complete (server gave length)
                total = int(r.headers.get("Content-Length", "0"))
                if expected_size is None and total > 0:
                    expected_size = total
                if out_path.exists() and expected_size and out_path.stat().st_size == expected_size:
                    return out_path

                # Open with append if we’re resuming
                mode = "ab" if "Range" in headers else "wb"
                downloaded = tmp_path.stat().st_size if tmp_path.exists() and mode == "ab" else 0

                with open(tmp_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                # Finalize
                # If we know expected size, verify before renaming
                if expected_size and downloaded != expected_size:
                    # Server might have given compressed-on-the-fly size; accept mismatch if no length available
                    # If mismatch and we have retries left, try again (start fresh)
                    retries += 1
                    continue

                tmp_path.replace(out_path)
                return out_path

        except requests.HTTPError as e:
            # Backoff on server pushback
            sleep_s = min(60, (2 ** retries) + random.uniform(0.1, 0.9))
            print(f"[warn] HTTP error for Id={product_id}: {e}. Retrying in {sleep_s:.1f}s...", file=sys.stderr)
            time.sleep(sleep_s)
            retries += 1
        except requests.RequestException as e:
            sleep_s = min(60, (2 ** retries) + random.uniform(0.1, 0.9))
            print(f"[warn] Network error for Id={product_id}: {e}. Retrying in {sleep_s:.1f}s...", file=sys.stderr)
            time.sleep(sleep_s)
            retries += 1

    raise RuntimeError(f"Exceeded retries for product {product_id}")

# -------- Orchestrator --------

def download_one(
    name: str,
    sess: requests.Session,
    out_dir: Path,
    token_refresher,
    refresh_token_value: Optional[str],
) -> tuple[str, str]:
    """
    Returns (name, status_message)
    """
    # In download_one(...)
    try:
        pid = find_product_id_by_name(sess, name, token_refresher=refresh_token, refresh_token_value=refresh_token_value)
    except requests.HTTPError as e:
        return name, f"[err] OData query failed: {e}"

    if pid is None:
        return name, "[skip] Not found in CDSE"

    try:
        path = stream_download(sess, pid, out_dir, token_refresher, refresh_token_value)
        return name, f"[ok] -> {path.name}"
    except Exception as e:
        return name, f"[err] Download failed (Id={pid}): {e}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True, help="TXT/CSV with one product name per line (CSV: first column)")
    ap.add_argument("--out", default="./cdse_downloads", help="Output directory")
    ap.add_argument("--workers", type=int, default=6, help="Concurrent downloads (suggest 4–10).")
    ap.add_argument("--max-catalog-qps", type=float, default=3.0, help="Soft cap on catalog queries per second (per process).")
    args = ap.parse_args()

    in_path = Path(args.infile)
    if not in_path.exists():
        print(f"Input file not found: {in_path}", file=sys.stderr)
        sys.exit(2)

    names = load_product_names(in_path)
    if not names:
        print("No product names found in input.", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auth
    username, password = prompt_credentials()
    tok = get_token(username, password)
    access_token = tok["access_token"]
    refresh_token_value = tok.get("refresh_token")

    sess = make_retrying_session(access_token)

    # --- Server-friendly knobs ---
    # 1) Bounded thread pool (workers)
    # 2) Jitter + Retry on 429/5xx inside requests session
    # 3) Light throttle between OData calls via jitter already applied
    # If you want a hard QPS cap, you could add a token-bucket here. For most batches,
    # the above is enough to avoid pushback, especially with workers in 4–10 range.

    print(f"Starting downloads: {len(names)} products, workers={args.workers}")
    results: List[Tuple[str, str]] = []

    # We reuse ONE session across threads safely for GETs — requests.Session is generally thread-safe for reads.
    # If you prefer per-thread sessions, you can instantiate inside the worker.
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(download_one, name, sess, out_dir, refresh_token, refresh_token_value)
            for name in names
        ]
        for fut in as_completed(futures):
            name, status = fut.result()
            print(f"{name}: {status}")
            results.append((name, status))

    # Summary
    ok = sum(1 for _, s in results if s.startswith("[ok]"))
    skip = sum(1 for _, s in results if s.startswith("[skip]"))
    err = sum(1 for _, s in results if s.startswith("[err]"))
    print(f"\nDone. ok={ok}, skip={skip}, err={err} / total={len(results)}")

if __name__ == "__main__":
    main()

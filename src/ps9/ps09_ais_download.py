#!/usr/bin/env python3

# Given a csv with time stamp and a remarks column
# download AIS files from Marine Cadestre. Filter 
# time stamps using the remarks column

import os, csv
import sys
import requests
import zipfile
import logging
from pathlib import Path
from typing import List
from datetime import datetime


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)    

URL_TEMPLATE_1 = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/ais-{year}-{month:02d}-{day:02d}.csv.zst" 
URL_TEMPLATE_2 = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{year}_{month:02d}_{day:02d}.zip"


def download_ais_data(dates: List[datetime], output_dir: Path) -> None:
    """
    Download AIS data files for the given dates.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for date in dates:
        url = URL_TEMPLATE_1.format(year=date.year, month=date.month, day=date.day) if date.year >= 2025 else URL_TEMPLATE_2.format(year=date.year, month=date.month, day=date.day)
        filename = url.split("/")[-1]
        output_path = output_dir / filename
        if output_path.exists():
            logger.info(f"File {filename} already exists. Skipping download.")
            continue
        try:
            logger.info(f"Downloading {url}")
            response = requests.get(url)
            response.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(response.content)
            logger.info(f"Downloaded and saved to {output_path}")
        except requests.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")

        # Extract archived files if necessary
        if filename.endswith(".zip"):
            with zipfile.ZipFile(output_path, 'r') as zip_ref:
                zip_ref.extractall(output_dir)
            os.remove(output_path)  # Remove the zip file after extraction
            logger.info(f"Extracted {filename} and removed the archive.")
        elif filename.endswith(".zst"):
            import zstandard as zstd
            dctx = zstd.ZstdDecompressor()
            with open(output_path, 'rb') as compressed:
                with open(output_path.with_suffix('.csv'), 'wb') as decompressed:
                    dctx.copy_stream(compressed, decompressed)
            os.remove(output_path)  # Remove the zst file after extraction
            logger.info(f"Decompressed {filename} and removed the archive.")

def main(input_csv: Path, output_dir: Path) -> None:
    with open(input_csv, 'r') as f:
        reader = csv.DictReader(f)
        if 'time_stamp' not in reader.fieldnames or 'Remarks' not in reader.fieldnames:
            logger.error("Input CSV must contain 'time_stamp' and 'Remarks' columns.")
            return
        
        ais_dates = []
        for row in reader:
            if 'For AIS correlations' in row['Remarks']:
                try:
                    date = datetime.fromisoformat(row['time_stamp'])
                    ais_dates.append(date)
                except ValueError as e:
                    logger.error(f"Invalid date format in row {row}: {e}")
                    

    if not ais_dates:
        logger.info("No dates found for AIS correlation.")
        return

    # Remove duplicates and sort
    ais_dates = sorted(set(ais_dates))
    download_ais_data(ais_dates, output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cdse_download.py <input_csv> <output_dir>")
        sys.exit(1)
    input_csv = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    main(input_csv, output_dir)
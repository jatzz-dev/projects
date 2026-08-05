"""Download the ERA5 climate input used by the Mumbai model.

Credentials are read from CDS_API_KEY and never written to the project. The
current build requests a point time series at the ERA5 grid point nearest Mumbai,
then converts the hourly values to daily and monthly features with
process_era5_timeseries.py.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cdsapi

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
DATASET = "reanalysis-era5-single-levels-timeseries"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    key = os.environ.get("CDS_API_KEY")
    if not key:
        raise SystemExit("Set CDS_API_KEY in the shell; it is intentionally not stored in this project.")
    if args.end_year < args.start_year:
        raise SystemExit("--end-year must be >= --start-year")

    target = RAW / f"era5_timeseries_{args.start_year}_{args.end_year}.zip"
    if target.exists() and not args.force:
        print(f"Skip existing {target.name}")
        return
    client = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key=key,
        quiet=True,
        progress=False,
    )
    request = {
        "variable": ["2m_temperature", "2m_dewpoint_temperature", "total_precipitation"],
        "location": {"latitude": 19.076, "longitude": 72.878},
        "date": [f"{args.start_year}-01-01/{args.end_year}-12-31"],
        "data_format": "csv",
    }
    print(f"Requesting ERA5 time series for {args.start_year}-{args.end_year} ...")
    client.retrieve(DATASET, request, str(target))
    print(f"Saved {target} ({target.stat().st_size:,} bytes)")
    print("Next: python scripts/process_era5_timeseries.py")


if __name__ == "__main__":
    main()

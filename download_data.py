"""Download public non-ERA5 layers for the Mumbai prototype.

The attached Praja CSV is not downloaded here; place it at data/raw/praja_raw.csv.
ERA5 is downloaded separately by download_era5.py because it requires a Copernicus
CDS Personal Access Token in the CDS_API_KEY environment variable.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)


def download(url: str, path: Path, params: dict | None = None) -> None:
    print(f"GET {url} -> {path.name}")
    r = requests.get(url, params=params, timeout=180)
    r.raise_for_status()
    path.write_bytes(r.content)
    print(f"  {len(r.content):,} bytes")


def main() -> None:
    download(
        "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt",
        RAW / "noaa_nino34_monthly.ascii.txt",
    )
    download(
        "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt",
        RAW / "noaa_oni.ascii.txt",
    )
    download(
        "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/master/Mumbai/BMC_Wards.geojson",
        RAW / "BMC_Wards.geojson",
    )
    download(
        "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/master/Mumbai/slumClusters.geojson",
        RAW / "slumClusters.geojson",
    )
    download(
        "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/master/Mumbai/Readme.md",
        RAW / "Mumbai_Readme.md",
    )
    # Elevation is a static vulnerability attribute, not a climate predictor.
    ward_obj = json.loads((RAW / "BMC_Wards.geojson").read_text())
    points = [shape(f["geometry"]).representative_point() for f in ward_obj["features"]]
    elev = requests.get(
        "https://api.open-meteo.com/v1/elevation",
        params={"latitude": ",".join(f"{p.y:.6f}" for p in points), "longitude": ",".join(f"{p.x:.6f}" for p in points)},
        timeout=60,
    )
    elev.raise_for_status()
    (RAW / "ward_elevation_response.json").write_bytes(elev.content)
    print("GET static ward elevation -> ward_elevation_response.json")
    download(
        "https://praja.org/praja_docs/praja_downloads/Mumbai%20Health%20White%20Paper%202022_Final.pdf",
        RAW / "praja_mumbai_health_2022.pdf",
    )
    download(
        "https://www.praja.org/praja_docs/praja_downloads/Mumbai%20Health%20White%20Paper%202024.pdf",
        RAW / "praja_mumbai_health_2024.pdf",
    )
    print("\nDone. For the climate layer, run:")
    print("  export CDS_API_KEY=YOUR_CDS_PERSONAL_ACCESS_TOKEN")
    print("  python scripts/download_era5.py --start-year 2008 --end-year 2018")
    print("  python scripts/process_era5_timeseries.py")
    print("  python scripts/prepare_data.py")
    print("  python -m src.forecast")


if __name__ == "__main__":
    main()

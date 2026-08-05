"""Prepare the Mumbai diarrhoea early-warning data package.

This script deliberately keeps the raw sources intact and writes derived, versionable
CSV/GeoJSON files under data/processed. It can be re-run after replacing
 data/raw/praja_raw.csv with a newer Praja/BMC extract.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Values transcribed from Praja Foundation, State of Health in Mumbai (July 2022),
# Table 4. The table is also retained in data/raw/praja_mumbai_health_2022.pdf.
# Slum share is unavailable for C in that source and is intentionally kept NaN.
WARD_META: dict[str, dict[str, Any]] = {
    "A": {"population_2019": 191450, "slum_pct": 34.0, "government_hospitals": 4, "dispensaries_2021": 7, "dispensary_norm_15000": 13},
    "B": {"population_2019": 131718, "slum_pct": 11.0, "government_hospitals": 0, "dispensaries_2021": 5, "dispensary_norm_15000": 9},
    "C": {"population_2019": 171941, "slum_pct": np.nan, "government_hospitals": 0, "dispensaries_2021": 5, "dispensary_norm_15000": 11},
    "D": {"population_2019": 358933, "slum_pct": 10.0, "government_hospitals": 0, "dispensaries_2021": 8, "dispensary_norm_15000": 24},
    "E": {"population_2019": 406967, "slum_pct": 20.0, "government_hospitals": 6, "dispensaries_2021": 12, "dispensary_norm_15000": 27},
    "F/N": {"population_2019": 547438, "slum_pct": 58.0, "government_hospitals": 2, "dispensaries_2021": 8, "dispensary_norm_15000": 36},
    "F/S": {"population_2019": 373529, "slum_pct": 26.0, "government_hospitals": 5, "dispensaries_2021": 10, "dispensary_norm_15000": 25},
    "G/N": {"population_2019": 619878, "slum_pct": 32.0, "government_hospitals": 0, "dispensaries_2021": 10, "dispensary_norm_15000": 41},
    "G/S": {"population_2019": 390890, "slum_pct": 21.0, "government_hospitals": 1, "dispensaries_2021": 14, "dispensary_norm_15000": 26},
    "H/E": {"population_2019": 576624, "slum_pct": 42.0, "government_hospitals": 1, "dispensaries_2021": 8, "dispensary_norm_15000": 38},
    "H/W": {"population_2019": 318281, "slum_pct": 39.0, "government_hospitals": 1, "dispensaries_2021": 5, "dispensary_norm_15000": 21},
    "K/E": {"population_2019": 852546, "slum_pct": 49.0, "government_hospitals": 2, "dispensaries_2021": 13, "dispensary_norm_15000": 57},
    "K/W": {"population_2019": 774733, "slum_pct": 15.0, "government_hospitals": 1, "dispensaries_2021": 7, "dispensary_norm_15000": 52},
    "L": {"population_2019": 933611, "slum_pct": 54.0, "government_hospitals": 1, "dispensaries_2021": 16, "dispensary_norm_15000": 62},
    "M/E": {"population_2019": 835819, "slum_pct": 30.0, "government_hospitals": 1, "dispensaries_2021": 11, "dispensary_norm_15000": 56},
    "M/W": {"population_2019": 426222, "slum_pct": 53.0, "government_hospitals": 1, "dispensaries_2021": 6, "dispensary_norm_15000": 28},
    "N": {"population_2019": 644521, "slum_pct": 62.0, "government_hospitals": 2, "dispensaries_2021": 9, "dispensary_norm_15000": 43},
    "P/N": {"population_2019": 974114, "slum_pct": 54.0, "government_hospitals": 3, "dispensaries_2021": 12, "dispensary_norm_15000": 65},
    "P/S": {"population_2019": 479631, "slum_pct": 57.0, "government_hospitals": 1, "dispensaries_2021": 3, "dispensary_norm_15000": 32},
    "R/C": {"population_2019": 581718, "slum_pct": 19.0, "government_hospitals": 2, "dispensaries_2021": 7, "dispensary_norm_15000": 39},
    "R/N": {"population_2019": 446374, "slum_pct": 51.0, "government_hospitals": 0, "dispensaries_2021": 5, "dispensary_norm_15000": 30},
    "R/S": {"population_2019": 715275, "slum_pct": 58.0, "government_hospitals": 2, "dispensaries_2021": 7, "dispensary_norm_15000": 48},
    "S": {"population_2019": 769657, "slum_pct": 72.0, "government_hospitals": 1, "dispensaries_2021": 8, "dispensary_norm_15000": 51},
    "T": {"population_2019": 353343, "slum_pct": 33.0, "government_hospitals": 3, "dispensaries_2021": 3, "dispensary_norm_15000": 24},
}

# Ward-level BMC dispensary diarrhoea cases transcribed from Praja Foundation,
# State of Health in Mumbai (2024), Table 19. This extends the raw monthly file
# with annual totals through 2023. Values are reported cases, not model estimates.
DIARRHOEA_2019_2023: dict[str, list[int]] = {
    "A": [1526, 976, 1136, 1030, 1296],
    "B": [1132, 696, 651, 1042, 1175],
    "C": [2318, 1612, 2010, 1761, 344],
    "D": [99, 91, 141, 10, 0],
    "E": [5561, 4087, 5118, 5207, 5365],
    "F/N": [2012, 1240, 1874, 3071, 1048],
    "F/S": [3391, 2323, 1717, 2247, 483],
    "G/N": [5856, 3843, 3250, 3580, 2854],
    "G/S": [8529, 6370, 5017, 7950, 7839],
    "H/E": [2797, 1613, 1572, 3247, 2261],
    "H/W": [934, 787, 880, 885, 1278],
    "K/E": [3951, 2988, 2700, 2424, 3092],
    "K/W": [2492, 2431, 2365, 2343, 1941],
    "L": [10915, 7530, 6125, 5966, 5538],
    "M/E": [5806, 4466, 3967, 4673, 4504],
    "M/W": [1203, 970, 571, 2580, 1811],
    "N": [7374, 5293, 4444, 2306, 1221],
    "P/N": [4305, 4135, 3888, 3006, 2697],
    "P/S": [1160, 897, 799, 610, 431],
    "R/C": [4093, 2892, 2340, 2428, 2428],
    "R/N": [1253, 777, 548, 1410, 1622],
    "R/S": [2189, 948, 963, 969, 456],
    "S": [5638, 3434, 1685, 2060, 2206],
    "T": [1328, 978, 845, 939, 977],
}


def _zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    return (s - s.mean()) / (std if std else 1.0)


def prepare_disease() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    p = RAW / "praja_raw.csv"
    raw = pd.read_csv(p)
    raw["date"] = pd.to_datetime(raw["Month"], utc=True).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
    raw["Disease_norm"] = raw["Disease"].astype(str).str.strip().str.casefold()
    dia = raw.loc[raw["Disease_norm"].eq("diarrhoea")].copy()
    wards = sorted(WARD_META)
    months = pd.date_range("2008-01-01", "2018-12-01", freq="MS")

    agg = (
        dia.groupby(["Ward", "date"], as_index=False)
        .agg(cases=("Occurrence", "sum"), reporting_dispensaries=("Dispensary", "nunique"))
    )
    grid = pd.MultiIndex.from_product([wards, months], names=["ward", "date"]).to_frame(index=False)
    monthly = grid.merge(agg.rename(columns={"Ward": "ward"}), on=["ward", "date"], how="left")
    monthly["cases"] = monthly["cases"].fillna(0).astype(float)
    monthly["reporting_dispensaries"] = monthly["reporting_dispensaries"].fillna(0).astype(int)
    monthly["source"] = "Praja raw (monthly BMC dispensary records)"
    monthly["year"] = monthly["date"].dt.year
    monthly["month"] = monthly["date"].dt.month
    monthly = monthly.sort_values(["date", "ward"]).reset_index(drop=True)
    monthly.to_csv(OUT / "ward_monthly_diarrhoea.csv", index=False)

    annual_raw = (
        monthly.groupby(["ward", "year"], as_index=False)
        .agg(cases=("cases", "sum"), reporting_months=("date", "nunique"))
    )
    annual_extra = []
    for ward, vals in DIARRHOEA_2019_2023.items():
        for year, value in zip(range(2019, 2024), vals):
            annual_extra.append({"ward": ward, "year": year, "cases": value, "reporting_months": np.nan, "source": "Praja 2024 Table 19 (annual BMC dispensary cases)"})
    annual = annual_raw.assign(source="Praja raw (monthly BMC dispensary records)")
    annual = pd.concat([annual, pd.DataFrame(annual_extra)], ignore_index=True)
    annual = annual.sort_values(["year", "ward"]).reset_index(drop=True)
    annual.to_csv(OUT / "ward_annual_diarrhoea.csv", index=False)

    # Citywide annual totals are useful for checking the extraction.
    city = annual.groupby(["year", "source"], as_index=False).agg(cases=("cases", "sum"))
    city.to_csv(OUT / "city_annual_diarrhoea_check.csv", index=False)
    return raw, monthly, annual


def prepare_climate() -> tuple[pd.DataFrame, pd.DataFrame]:
    # ERA5 is mandatory for the current build. This prevents an accidental
    # fallback to the old Open-Meteo prototype.
    era5_monthly_path = OUT / "era5_climate_monthly.csv"
    if not era5_monthly_path.exists():
        raise FileNotFoundError(
            "ERA5 climate data are missing. Run `python scripts/download_era5.py` "
            "with CDS_API_KEY, then `python scripts/process_era5_timeseries.py`."
        )
    monthly = pd.read_csv(era5_monthly_path, parse_dates=["date"])
    normal_path = OUT / "era5_climate_monthly_normals.csv"
    if normal_path.exists():
        normal = pd.read_csv(normal_path)
    else:
        train_clim = monthly.loc[monthly["date"].between("2008-01-01", "2018-12-01")]
        normal = (
            train_clim.groupby("month", as_index=False)
            .agg(
                normal_rain_mm=("rain_mm", "mean"),
                normal_rain_mm_p25=("rain_mm", lambda s: s.quantile(0.25)),
                normal_rain_mm_p75=("rain_mm", lambda s: s.quantile(0.75)),
                normal_rainy_days=("rainy_days", "mean"),
                normal_heavy_rain_days=("heavy_rain_days", "mean"),
                normal_tmax=("tmax_mean", "mean"),
                normal_tmin=("tmin_mean", "mean"),
                normal_humidity=("humidity_mean", "mean"),
            )
        )
    monthly.to_csv(OUT / "climate_monthly_observed.csv", index=False)
    normal.to_csv(OUT / "climate_monthly_normals_2008_2018.csv", index=False)
    daily_path = OUT / "era5_daily_2008_2018.csv"
    daily = pd.read_csv(daily_path, parse_dates=["date"]) if daily_path.exists() else pd.DataFrame()
    return daily, monthly

def prepare_oni() -> pd.DataFrame:
    path = RAW / "noaa_nino34_monthly.ascii.txt"
    rows = []
    pattern = re.compile(r"^\s*(\d{4})\s+(\d{1,2})\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*$")
    for line in path.read_text(errors="ignore").splitlines():
        m = pattern.match(line)
        if m:
            year, month, total, climatology, anom = m.groups()
            rows.append({"year": int(year), "month": int(month), "nino34_anomaly": float(anom)})
    oni = pd.DataFrame(rows)
    oni["date"] = pd.to_datetime(dict(year=oni["year"], month=oni["month"], day=1))
    # A trailing 3-month mean avoids looking ahead when this index is used for training.
    oni = oni.sort_values("date")
    oni["oni_trailing_3m"] = oni["nino34_anomaly"].rolling(3, min_periods=1).mean()
    oni["enso_phase"] = np.select(
        [oni["oni_trailing_3m"] >= 0.5, oni["oni_trailing_3m"] <= -0.5],
        ["El Niño-like", "La Niña-like"],
        default="Neutral",
    )
    oni.to_csv(OUT / "noaa_oni_monthly.csv", index=False)
    return oni


def prepare_wards() -> pd.DataFrame:
    wards_path = RAW / "BMC_Wards.geojson"
    obj = json.loads(wards_path.read_text())
    elevation_obj = {}
    elev_path = RAW / "ward_elevation_response.json"
    if elev_path.exists():
        elevation_obj = json.loads(elev_path.read_text())
    elevations = elevation_obj.get("elevation", [])
    ward_rows = []
    for i, feature in enumerate(obj["features"]):
        ward = feature["properties"]["name"]
        geom = shape(feature["geometry"])
        pt = geom.representative_point()
        meta = dict(WARD_META.get(ward, {}))
        meta.update({"ward": ward, "centroid_lon": pt.x, "centroid_lat": pt.y, "area_deg2": geom.area})
        if i < len(elevations):
            meta["elevation_m"] = elevations[i]
        else:
            meta["elevation_m"] = np.nan
        ward_rows.append(meta)
    meta_df = pd.DataFrame(ward_rows)
    meta_df["area_km2_approx"] = meta_df["area_deg2"] * (111.32**2) * np.cos(np.deg2rad(meta_df["centroid_lat"]))
    meta_df["population_density_per_km2"] = meta_df["population_2019"] / meta_df["area_km2_approx"]
    meta_df["slum_pct_imputed"] = meta_df["slum_pct"].fillna(meta_df["slum_pct"].median())
    meta_df["dispensary_gap_to_15k"] = (meta_df["dispensary_norm_15000"] - meta_df["dispensaries_2021"]).clip(lower=0)
    # An interpretable vulnerability composite for prioritisation only; it is not a causal estimate.
    risk_raw = (
        0.50 * _zscore(meta_df["slum_pct_imputed"])
        + 0.30 * _zscore(meta_df["population_density_per_km2"])
        + 0.20 * _zscore(-meta_df["elevation_m"])
    )
    meta_df["vulnerability_index"] = 100 * (risk_raw - risk_raw.min()) / (risk_raw.max() - risk_raw.min())
    meta_df.sort_values("ward").to_csv(OUT / "ward_vulnerability.csv", index=False)

    lookup = meta_df.set_index("ward").to_dict(orient="index")
    for feature in obj["features"]:
        ward = feature["properties"]["name"]
        p = lookup.get(ward, {})
        props = {"ward": ward, "gid": feature["properties"].get("gid")}
        for key in ["population_2019", "slum_pct", "elevation_m", "vulnerability_index", "dispensaries_2021", "government_hospitals"]:
            value = p.get(key)
            if isinstance(value, float) and not math.isfinite(value):
                value = None
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            props[key] = value
        feature["properties"] = props
    (OUT / "mumbai_wards_enriched.geojson").write_text(json.dumps(obj))
    return meta_df


def main() -> None:
    _, monthly, annual = prepare_disease()
    _, climate_monthly = prepare_climate()
    oni = prepare_oni()
    meta = prepare_wards()
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "target": "BMC dispensary diarrhoea cases aggregated to 24 administrative wards",
        "raw_disease_coverage": "2008-01 through 2018-12 monthly",
        "annual_disease_extension": "2019 through 2023 from Praja Foundation 2024 Table 19",
        "weather_coverage": f"{climate_monthly.date.min().date()} through {climate_monthly.date.max().date()}",
        "ward_count": int(meta.shape[0]),
        "sources": {
            "praja_raw": "User-provided Praja/BMC CSV",
            "praja_health_2022": "https://praja.org/praja_docs/praja_downloads/Mumbai%20Health%20White%20Paper%202022_Final.pdf",
            "praja_health_2024": "https://www.praja.org/praja_docs/praja_downloads/Mumbai%20Health%20White%20Paper%202024.pdf",
            "era5_cds": "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=download",
            "noaa_nino34": "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt",
            "ward_boundaries": "https://github.com/datameet/Municipal_Spatial_Data/tree/master/Mumbai",
        },
        "api_keys_required": True,
        "api_key_note": "Copernicus CDS Personal Access Token is required for ERA5 download and is not stored in this project.",
    }
    (OUT / "data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print("Prepared", len(monthly), "monthly ward rows and", len(annual), "annual ward rows")


if __name__ == "__main__":
    main()

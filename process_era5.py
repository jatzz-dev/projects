"""Convert CDS ERA5 daily-statistics ZIP files to model-ready climate tables."""
from __future__ import annotations

import math
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def _open_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory(prefix="era5_") as td:
        nc_names = [n for n in zf.namelist() if n.lower().endswith((".nc", ".nc4"))]
        if not nc_names:
            raise RuntimeError(f"No NetCDF file found inside {path}")
        frames = []
        for name in nc_names:
            extracted = Path(zf.extract(name, td))
            ds = xr.open_dataset(extracted)
            # New CDS files sometimes contain expver or another non-spatial
            # auxiliary dimension. Average it before turning the data into a table.
            time_dim = "valid_time" if "valid_time" in ds.dims else "time"
            spatial_dims = [d for d in ds.dims if d in {"latitude", "longitude", "lat", "lon"}]
            other_dims = [d for d in ds.dims if d not in spatial_dims and d != time_dim]
            if spatial_dims:
                ds = ds.mean(dim=spatial_dims, skipna=True)
            if other_dims:
                ds = ds.mean(dim=other_dims, skipna=True)
            frame = ds.to_dataframe().reset_index()
            if time_dim not in frame.columns:
                raise RuntimeError(f"Could not find time dimension in {name}: {frame.columns.tolist()}")
            frame["date"] = pd.to_datetime(frame[time_dim], utc=True).dt.tz_localize(None).dt.normalize()
            keep = ["date"] + [c for c in ["t2m", "d2m", "tp"] if c in frame.columns]
            frames.append(frame[keep])
            ds.close()
        return pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")


def _load(label: str, start_year: int = 2008, end_year: int = 2018) -> pd.DataFrame:
    paths = []
    for path in sorted(RAW.glob(f"era5_daily_{label}_*.zip")):
        try:
            year = int(path.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        if start_year <= year <= end_year:
            paths.append(path)
    if not paths:
        raise FileNotFoundError(f"No ERA5 {label} archives for {start_year}-{end_year}")
    frames = [_open_zip(path) for path in paths]
    # The mean request includes both t2m and d2m; the others contain t2m or tp.
    return pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date")


def saturation_vapour_pressure(temp_c: pd.Series) -> pd.Series:
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def main() -> None:
    # Current build downloads the historical period matching the monthly disease
    # series. The same function supports a different range when files are present.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2018)
    args = parser.parse_args()

    mx = _load("max", args.start_year, args.end_year).rename(columns={"t2m": "tmax_k"})
    mn = _load("min", args.start_year, args.end_year).rename(columns={"t2m": "tmin_k"})
    mean = _load("mean", args.start_year, args.end_year).rename(columns={"t2m": "tmean_k", "d2m": "dewpoint_k"})
    precip = _load("precip", args.start_year, args.end_year).rename(columns={"tp": "precip_m"})
    daily = mx.merge(mn, on="date", how="outer").merge(mean, on="date", how="outer").merge(precip, on="date", how="outer").sort_values("date")
    for c in ["tmax_k", "tmin_k", "tmean_k", "dewpoint_k", "precip_m"]:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")
    daily["tmax_c"] = daily["tmax_k"] - 273.15
    daily["tmin_c"] = daily["tmin_k"] - 273.15
    daily["tmean_c"] = daily["tmean_k"] - 273.15
    daily["dewpoint_c"] = daily["dewpoint_k"] - 273.15
    # Approximate RH from daily mean temperature/dewpoint using the Magnus equation.
    daily["relative_humidity"] = 100 * saturation_vapour_pressure(daily["dewpoint_c"]) / saturation_vapour_pressure(daily["tmean_c"])
    daily["relative_humidity"] = daily["relative_humidity"].clip(0, 100)
    daily["rain_mm"] = daily["precip_m"].clip(lower=0) * 1000
    daily["rain_day"] = (daily["rain_mm"] >= 1).astype(int)
    daily["heavy_rain_day"] = (daily["rain_mm"] >= 50).astype(int)
    daily = daily[["date", "tmax_c", "tmin_c", "tmean_c", "dewpoint_c", "relative_humidity", "rain_mm", "rain_day", "heavy_rain_day"]]
    daily.to_csv(OUT / f"era5_daily_{args.start_year}_{args.end_year}.csv", index=False)

    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    monthly = (
        daily.groupby(["year", "month"], as_index=False)
        .agg(
            rain_mm=("rain_mm", "sum"),
            rainy_days=("rain_day", "sum"),
            heavy_rain_days=("heavy_rain_day", "sum"),
            tmax_mean=("tmax_c", "mean"),
            tmin_mean=("tmin_c", "mean"),
            humidity_mean=("relative_humidity", "mean"),
            days_observed=("date", "nunique"),
        )
    )
    monthly["temp_mean"] = (monthly["tmax_mean"] + monthly["tmin_mean"]) / 2
    monthly["date"] = pd.to_datetime(dict(year=monthly["year"], month=monthly["month"], day=1))
    monthly = monthly.sort_values("date")
    monthly.to_csv(OUT / "era5_climate_monthly.csv", index=False)
    normal = (
        monthly.groupby("month", as_index=False)
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
    normal.to_csv(OUT / "era5_climate_monthly_normals.csv", index=False)
    print("Saved", len(daily), "ERA5 daily rows and", len(monthly), "monthly rows")
    print(monthly.head().to_string(index=False))


if __name__ == "__main__":
    main()

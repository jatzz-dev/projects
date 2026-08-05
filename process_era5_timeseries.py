"""Process a CDS ERA5 hourly time-series ZIP into daily/monthly Mumbai climate data."""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def saturation_vapour_pressure(temp_c: pd.Series) -> pd.Series:
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def load_hourly(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No CSV found inside {path}")
        df = pd.read_csv(zf.open(names[0]))
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    # Convert UTC ERA5 time to Mumbai local date before calculating daily statistics.
    df["local_time"] = df["valid_time"].dt.tz_convert("Asia/Kolkata")
    df["date"] = df["local_time"].dt.tz_localize(None).dt.normalize()
    for c in ["t2m", "d2m", "tp"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/era5_timeseries_2008_2018.zip")
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2018)
    args = parser.parse_args()
    hourly = load_hourly(ROOT / args.input)
    daily = (
        hourly.groupby("date", as_index=False)
        .agg(
            tmax_k=("t2m", "max"),
            tmin_k=("t2m", "min"),
            tmean_k=("t2m", "mean"),
            dewpoint_k=("d2m", "mean"),
            precip_m=("tp", "sum"),
            hours_observed=("valid_time", "nunique"),
        )
    )
    daily["tmax_c"] = daily["tmax_k"] - 273.15
    daily["tmin_c"] = daily["tmin_k"] - 273.15
    daily["tmean_c"] = daily["tmean_k"] - 273.15
    daily["dewpoint_c"] = daily["dewpoint_k"] - 273.15
    daily["relative_humidity"] = 100 * saturation_vapour_pressure(daily["dewpoint_c"]) / saturation_vapour_pressure(daily["tmean_c"])
    daily["relative_humidity"] = daily["relative_humidity"].clip(0, 100)
    daily["rain_mm"] = daily["precip_m"].clip(lower=0) * 1000
    daily["rain_day"] = (daily["rain_mm"] >= 1).astype(int)
    daily["heavy_rain_day"] = (daily["rain_mm"] >= 50).astype(int)
    daily = daily[["date", "tmax_c", "tmin_c", "tmean_c", "dewpoint_c", "relative_humidity", "rain_mm", "rain_day", "heavy_rain_day", "hours_observed"]]
    daily = daily.sort_values("date")
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
        monthly.loc[monthly["date"].between("2008-01-01", "2018-12-01")]
        .groupby("month", as_index=False)
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
    print(f"Saved {len(daily):,} ERA5 daily rows and {len(monthly):,} monthly rows")
    print(monthly.head(12).to_string(index=False))
    print("Annual precipitation (mm):")
    print(monthly.groupby("year")["rain_mm"].sum().round(1).to_string())


if __name__ == "__main__":
    main()

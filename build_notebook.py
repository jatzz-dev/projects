from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(r'''# Mumbai diarrhoea early-warning model — ERA5 version

This notebook is a **standalone VS Code/Jupyter workflow** for the Mumbai ward-level diarrhoea early-warning model.

It follows the proposed pipeline:

**Praja/BMC disease data + Copernicus ERA5 + NOAA ENSO + ward vulnerability → leakage-safe features → time-based model selection → 2026 planning scenario → spatial heat map.**

## Before running

1. Put `praja_raw.csv` in the same folder as this notebook, or in `data/raw/praja_raw.csv`.
2. Accept the relevant ERA5 licence in the Copernicus Climate Data Store: <https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=download>
3. Set the CDS Personal Access Token in the VS Code terminal, **not in this notebook**:

```bash
# macOS/Linux
export CDS_API_KEY="YOUR_CDS_PERSONAL_ACCESS_TOKEN"

# Windows PowerShell
$env:CDS_API_KEY="YOUR_CDS_PERSONAL_ACCESS_TOKEN"
```

The notebook will use an existing `data/raw/era5_timeseries_2008_2018.zip` if present. Otherwise it downloads the ERA5 time series. The token is used only in memory and is never written to a file.

> The attached monthly disease data ends in December 2018. The 2026 output is therefore a planning scenario calibrated with annual Praja ward data through 2023; it is not a validated 2026 nowcast.'''))

cells.append(nbf.v4.new_code_cell(r'''# 1. Install dependencies in the active VS Code/Jupyter kernel.
# If VS Code asks to restart the kernel after this cell, do so and continue.
%pip install -q pandas numpy scikit-learn joblib requests shapely plotly cdsapi'''))

cells.append(nbf.v4.new_code_cell(r'''# 2. Imports and project paths
from __future__ import annotations

import getpass
import json
import math
import os
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from shapely.geometry import shape
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Run the notebook from its project folder in VS Code.
ROOT = Path.cwd()
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# Locate the user-provided file in common layouts.
candidates = [
    ROOT / "praja_raw.csv",
    RAW / "praja_raw.csv",
    ROOT / "uploads" / "praja_raw.csv",
]
PRAJA_CSV = next((p for p in candidates if p.exists()), None)
if PRAJA_CSV is None:
    raise FileNotFoundError("Place praja_raw.csv beside this notebook or under data/raw/praja_raw.csv")

print("Project:", ROOT)
print("Praja file:", PRAJA_CSV)
print("ERA5 raw file:", RAW / "era5_timeseries_2008_2018.zip")'''))

cells.append(nbf.v4.new_markdown_cell(r'''## 3. Download ERA5 and public supporting data

The climate predictors below come from the **Copernicus ERA5 hourly time-series** at the nearest ERA5 grid point to Mumbai. We will convert hourly temperature, dew point and precipitation to Mumbai-local daily and monthly predictors.

The NOAA ENSO index and BMC ward boundaries are public downloads. The elevation values used below are static vulnerability attributes; they are not climate predictors.'''))

cells.append(nbf.v4.new_code_cell(r'''# 3A. Download or reuse ERA5, NOAA and ward boundaries.

def download_if_missing(url: str, path: Path) -> Path:
    if path.exists():
        print("Using existing", path.name)
        return path
    print("Downloading", url)
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    path.write_bytes(response.content)
    print("Saved", path, f"({len(response.content):,} bytes)")
    return path

ERA5_ZIP = RAW / "era5_timeseries_2008_2018.zip"
if not ERA5_ZIP.exists():
    cds_key = os.environ.get("CDS_API_KEY")
    if not cds_key:
        cds_key = getpass.getpass("Enter CDS Personal Access Token (not saved): ")
    import cdsapi
    client = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key=cds_key,
        quiet=True,
        progress=False,
    )
    request = {
        "variable": ["2m_temperature", "2m_dewpoint_temperature", "total_precipitation"],
        "location": {"latitude": 19.076, "longitude": 72.878},
        "date": ["2008-01-01/2018-12-31"],
        "data_format": "csv",
    }
    print("Requesting ERA5. This may take a few minutes...")
    client.retrieve("reanalysis-era5-single-levels-timeseries", request, str(ERA5_ZIP))
    del cds_key
    print("Saved", ERA5_ZIP)
else:
    print("Using existing ERA5 archive", ERA5_ZIP)

NOAA_FILE = download_if_missing(
    "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt",
    RAW / "noaa_nino34_monthly.ascii.txt",
)
WARD_FILE = download_if_missing(
    "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/master/Mumbai/BMC_Wards.geojson",
    RAW / "BMC_Wards.geojson",
)
print("No secret has been written by this notebook.")'''))

cells.append(nbf.v4.new_code_cell(r'''# 3B. Process ERA5 hourly data into daily and monthly Mumbai climate predictors.
with zipfile.ZipFile(ERA5_ZIP) as zf:
    csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if not csv_names:
        raise RuntimeError("No CSV found inside ERA5 archive")
    hourly = pd.read_csv(zf.open(csv_names[0]))

hourly["valid_time"] = pd.to_datetime(hourly["valid_time"], utc=True)
hourly["local_time"] = hourly["valid_time"].dt.tz_convert("Asia/Kolkata")
hourly["date"] = hourly["local_time"].dt.tz_localize(None).dt.normalize()
for col in ["t2m", "d2m", "tp"]:
    hourly[col] = pd.to_numeric(hourly[col], errors="coerce")

# Convert to daily local-time statistics. ERA5 total precipitation is in metres.
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
daily = daily.loc[daily["date"].between("2008-01-01", "2018-12-31")].copy()
daily["tmax_c"] = daily["tmax_k"] - 273.15
daily["tmin_c"] = daily["tmin_k"] - 273.15
daily["tmean_c"] = daily["tmean_k"] - 273.15
daily["dewpoint_c"] = daily["dewpoint_k"] - 273.15

def saturation_vapour_pressure(temp_c):
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))

daily["relative_humidity"] = 100 * saturation_vapour_pressure(daily["dewpoint_c"]) / saturation_vapour_pressure(daily["tmean_c"])
daily["relative_humidity"] = daily["relative_humidity"].clip(0, 100)
daily["rain_mm"] = daily["precip_m"].clip(lower=0) * 1000
daily["rain_day"] = (daily["rain_mm"] >= 1).astype(int)
daily["heavy_rain_day"] = (daily["rain_mm"] >= 50).astype(int)

daily["year"] = daily["date"].dt.year
daily["month"] = daily["date"].dt.month
climate_monthly = (
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
climate_monthly["temp_mean"] = (climate_monthly["tmax_mean"] + climate_monthly["tmin_mean"]) / 2
climate_monthly["date"] = pd.to_datetime(dict(year=climate_monthly["year"], month=climate_monthly["month"], day=1))
climate_monthly = climate_monthly.sort_values("date")

climate_normals = (
    climate_monthly.groupby("month", as_index=False)
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

daily.to_csv(PROCESSED / "era5_daily_2008_2018.csv", index=False)
climate_monthly.to_csv(PROCESSED / "era5_climate_monthly.csv", index=False)
climate_normals.to_csv(PROCESSED / "era5_climate_monthly_normals.csv", index=False)

print("ERA5 hourly rows:", len(hourly))
print("ERA5 daily rows:", len(daily))
print("ERA5 monthly rows:", len(climate_monthly))
print("ERA5 monthly date range:", climate_monthly.date.min().date(), "to", climate_monthly.date.max().date())
print("Annual rainfall (mm):")
display(climate_monthly.groupby("year")["rain_mm"].sum().round(1).to_frame())'''))

cells.append(nbf.v4.new_code_cell(r'''# 4. NOAA Niño 3.4 / ENSO monthly index
rows = []
pattern = re.compile(r"^\s*(\d{4})\s+(\d{1,2})\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s*$")
for line in NOAA_FILE.read_text(errors="ignore").splitlines():
    match = pattern.match(line)
    if match:
        year, month, total, climatology, anomaly = match.groups()
        rows.append({"year": int(year), "month": int(month), "nino34_anomaly": float(anomaly)})
oni = pd.DataFrame(rows)
oni["date"] = pd.to_datetime(dict(year=oni["year"], month=oni["month"], day=1))
oni = oni.sort_values("date")
# Trailing mean avoids future information leakage.
oni["oni_trailing_3m"] = oni["nino34_anomaly"].rolling(3, min_periods=1).mean()
oni["enso_phase"] = np.select(
    [oni["oni_trailing_3m"] >= 0.5, oni["oni_trailing_3m"] <= -0.5],
    ["El Niño-like", "La Niña-like"],
    default="Neutral",
)
oni.to_csv(PROCESSED / "noaa_oni_monthly.csv", index=False)
display(oni.tail())'''))

cells.append(nbf.v4.new_markdown_cell(r'''## 5. Prepare Praja/BMC disease outcomes and vulnerability inputs

The outcome is **reported BMC dispensary diarrhoea cases**, aggregated from dispensary to administrative ward and month.

The raw monthly dataset ends in 2018. The annual 2019–2023 extension below comes from Praja's ward-level table and is used only to calibrate the 2026 planning level.'''))

cells.append(nbf.v4.new_code_cell(r'''# 5A. Load and aggregate the user-provided Praja raw CSV.
raw = pd.read_csv(PRAJA_CSV)
raw["date"] = pd.to_datetime(raw["Month"], utc=True).dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
raw["Disease_norm"] = raw["Disease"].astype(str).str.strip().str.casefold()
dia = raw.loc[raw["Disease_norm"].eq("diarrhoea")].copy()

wards = sorted(dia["Ward"].unique())
months = pd.date_range("2008-01-01", "2018-12-01", freq="MS")
agg = (
    dia.groupby(["Ward", "date"], as_index=False)
    .agg(cases=("Occurrence", "sum"), reporting_dispensaries=("Dispensary", "nunique"))
    .rename(columns={"Ward": "ward"})
)
grid = pd.MultiIndex.from_product([wards, months], names=["ward", "date"]).to_frame(index=False)
monthly = grid.merge(agg, on=["ward", "date"], how="left")
monthly["cases"] = monthly["cases"].fillna(0).astype(float)
monthly["reporting_dispensaries"] = monthly["reporting_dispensaries"].fillna(0).astype(int)
monthly["year"] = monthly["date"].dt.year
monthly["month"] = monthly["date"].dt.month
monthly = monthly.sort_values(["date", "ward"]).reset_index(drop=True)

annual = monthly.groupby(["ward", "year"], as_index=False).agg(cases=("cases", "sum"), reporting_months=("date", "nunique"))
print("Raw rows:", raw.shape)
print("Monthly ward rows:", monthly.shape)
print("Wards:", len(wards))
print("Disease coverage:", monthly.date.min().date(), "to", monthly.date.max().date())
print("2018 diarrhoea cases:", int(monthly.loc[monthly.year.eq(2018), "cases"].sum()))'''))

cells.append(nbf.v4.new_code_cell(r'''# 5B. Annual ward-level extension from Praja Health White Paper 2024, Table 19.
# Values are reported BMC dispensary cases for 2019, 2020, 2021, 2022 and 2023.
DIARRHOEA_2019_2023 = {
    "A": [1526, 976, 1136, 1030, 1296], "B": [1132, 696, 651, 1042, 1175],
    "C": [2318, 1612, 2010, 1761, 344], "D": [99, 91, 141, 10, 0],
    "E": [5561, 4087, 5118, 5207, 5365], "F/N": [2012, 1240, 1874, 3071, 1048],
    "F/S": [3391, 2323, 1717, 2247, 483], "G/N": [5856, 3843, 3250, 3580, 2854],
    "G/S": [8529, 6370, 5017, 7950, 7839], "H/E": [2797, 1613, 1572, 3247, 2261],
    "H/W": [934, 787, 880, 885, 1278], "K/E": [3951, 2988, 2700, 2424, 3092],
    "K/W": [2492, 2431, 2365, 2343, 1941], "L": [10915, 7530, 6125, 5966, 5538],
    "M/E": [5806, 4466, 3967, 4673, 4504], "M/W": [1203, 970, 571, 2580, 1811],
    "N": [7374, 5293, 4444, 2306, 1221], "P/N": [4305, 4135, 3888, 3006, 2697],
    "P/S": [1160, 897, 799, 610, 431], "R/C": [4093, 2892, 2340, 2428, 2428],
    "R/N": [1253, 777, 548, 1410, 1622], "R/S": [2189, 948, 963, 969, 456],
    "S": [5638, 3434, 1685, 2060, 2206], "T": [1328, 978, 845, 939, 977],
}
extra = []
for ward, values in DIARRHOEA_2019_2023.items():
    for year, value in zip(range(2019, 2024), values):
        extra.append({"ward": ward, "year": year, "cases": value, "reporting_months": np.nan, "source": "Praja 2024 Table 19"})
annual["source"] = "Praja raw monthly"
annual = pd.concat([annual, pd.DataFrame(extra)], ignore_index=True).sort_values(["year", "ward"])
annual.to_csv(PROCESSED / "ward_annual_diarrhoea.csv", index=False)
print("Annual 2023 BMC dispensary total:", int(annual.loc[annual.year.eq(2023), "cases"].sum()))'''))

cells.append(nbf.v4.new_code_cell(r'''# 5C. Ward vulnerability metadata.
# Population, slum share and facility fields are from Praja's ward table.
WARD_META = {
    "A": [191450, 34, 4, 7, 13], "B": [131718, 11, 0, 5, 9], "C": [171941, np.nan, 0, 5, 11],
    "D": [358933, 10, 0, 8, 24], "E": [406967, 20, 6, 12, 27], "F/N": [547438, 58, 2, 8, 36],
    "F/S": [373529, 26, 5, 10, 25], "G/N": [619878, 32, 0, 10, 41], "G/S": [390890, 21, 1, 14, 26],
    "H/E": [576624, 42, 1, 8, 38], "H/W": [318281, 39, 1, 5, 21], "K/E": [852546, 49, 2, 13, 57],
    "K/W": [774733, 15, 1, 7, 52], "L": [933611, 54, 1, 16, 62], "M/E": [835819, 30, 1, 11, 56],
    "M/W": [426222, 53, 1, 6, 28], "N": [644521, 62, 2, 9, 43], "P/N": [974114, 54, 3, 12, 65],
    "P/S": [479631, 57, 1, 3, 32], "R/C": [581718, 19, 2, 7, 39], "R/N": [446374, 51, 0, 5, 30],
    "R/S": [715275, 58, 2, 7, 48], "S": [769657, 72, 1, 8, 51], "T": [353343, 33, 3, 3, 24],
}
# Static representative-point elevations used only in the vulnerability layer.
ELEVATION_M = {
    "A": 0, "B": 8, "C": 15, "D": 17, "E": 11, "F/S": 29, "G/S": 7, "F/N": 11,
    "G/N": 13, "N": 4, "R/C": 11, "S": 18, "T": 97, "K/W": 4, "R/N": 11, "M/E": 29,
    "M/W": 8, "H/E": 5, "K/E": 32, "P/S": 13, "P/N": 21, "R/S": 23, "H/W": 16, "L": 14,
}

ward_obj = json.loads(WARD_FILE.read_text())
meta_rows = []
for feature in ward_obj["features"]:
    ward = feature["properties"]["name"]
    geom = shape(feature["geometry"])
    point = geom.representative_point()
    population, slum_pct, hospitals, dispensaries, norm = WARD_META[ward]
    area_km2 = geom.area * (111.32 ** 2) * np.cos(np.deg2rad(point.y))
    meta_rows.append({
        "ward": ward, "population_2019": population, "slum_pct": slum_pct,
        "government_hospitals": hospitals, "dispensaries_2021": dispensaries,
        "dispensary_norm_15000": norm, "centroid_lon": point.x, "centroid_lat": point.y,
        "elevation_m": ELEVATION_M.get(ward, 0), "area_km2_approx": area_km2,
    })
meta = pd.DataFrame(meta_rows)
meta["slum_pct_imputed"] = meta["slum_pct"].fillna(meta["slum_pct"].median())
meta["population_density_per_km2"] = meta["population_2019"] / meta["area_km2_approx"]
meta["dispensary_gap_to_15k"] = (meta["dispensary_norm_15000"] - meta["dispensaries_2021"]).clip(lower=0)

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    return (s - s.mean()) / (s.std(ddof=0) or 1)

vulnerability_raw = (
    0.50 * zscore(meta["slum_pct_imputed"])
    + 0.30 * zscore(meta["population_density_per_km2"])
    + 0.20 * zscore(-meta["elevation_m"])
)
meta["vulnerability_index"] = 100 * (vulnerability_raw - vulnerability_raw.min()) / (vulnerability_raw.max() - vulnerability_raw.min())
meta.to_csv(PROCESSED / "ward_vulnerability.csv", index=False)
print("Vulnerability rows:", len(meta))
display(meta.sort_values("vulnerability_index", ascending=False).head(10))'''))

cells.append(nbf.v4.new_code_cell(r'''# 6. Leakage-safe feature engineering
NUMERIC_FEATURES = [
    "cases_lag_1", "cases_lag_2", "cases_lag_3", "cases_lag_12",
    "cases_roll_3", "cases_roll_6", "rain_lag_1", "rain_3m_lag",
    "rainy_days_lag_1", "heavy_rain_days_lag_1", "tmax_lag_1", "tmin_lag_1",
    "humidity_lag_1", "oni_lag_1", "month_sin", "month_cos", "slum_pct_imputed",
    "population_density_per_km2", "elevation_m", "vulnerability_index", "population_2019",
]
FEATURE_COLUMNS = ["ward"] + NUMERIC_FEATURES

def build_feature_frame(monthly, climate, oni, meta):
    d = monthly.copy().sort_values(["ward", "date"]).reset_index(drop=True)
    group = d.groupby("ward", sort=False)
    for lag in [1, 2, 3, 12]:
        d[f"cases_lag_{lag}"] = group["cases"].shift(lag)
    d["cases_roll_3"] = d.groupby("ward")["cases_lag_1"].transform(lambda s: s.rolling(3, min_periods=3).mean())
    d["cases_roll_6"] = d.groupby("ward")["cases_lag_1"].transform(lambda s: s.rolling(6, min_periods=6).mean())

    c = climate.copy().sort_values("date").drop_duplicates("date")
    c["rain_lag_1"] = c["rain_mm"].shift(1)
    c["rain_3m_lag"] = c["rain_mm"].shift(1).rolling(3, min_periods=3).sum()
    c["rainy_days_lag_1"] = c["rainy_days"].shift(1)
    c["heavy_rain_days_lag_1"] = c["heavy_rain_days"].shift(1)
    c["tmax_lag_1"] = c["tmax_mean"].shift(1)
    c["tmin_lag_1"] = c["tmin_mean"].shift(1)
    c["humidity_lag_1"] = c["humidity_mean"].shift(1)
    ccols = ["date", "rain_lag_1", "rain_3m_lag", "rainy_days_lag_1", "heavy_rain_days_lag_1", "tmax_lag_1", "tmin_lag_1", "humidity_lag_1"]
    d = d.merge(c[ccols], on="date", how="left", validate="many_to_one")

    o = oni[["date", "oni_trailing_3m"]].copy().sort_values("date").drop_duplicates("date")
    o["oni_lag_1"] = o["oni_trailing_3m"].shift(1)
    d = d.merge(o[["date", "oni_lag_1"]], on="date", how="left", validate="many_to_one")
    d = d.merge(meta[["ward", "slum_pct_imputed", "population_density_per_km2", "elevation_m", "vulnerability_index", "population_2019"]], on="ward", how="left", validate="many_to_one")
    d["month_sin"] = np.sin(2 * np.pi * d["date"].dt.month / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["date"].dt.month / 12)
    d = d.dropna(subset=NUMERIC_FEATURES).copy()
    d["target_log"] = np.log1p(d["cases"].clip(lower=0))
    return d

features = build_feature_frame(monthly, climate_monthly, oni, meta)
print("Feature rows:", features.shape)
print("Feature date range:", features.date.min().date(), "to", features.date.max().date())'''))

cells.append(nbf.v4.new_code_cell(r'''# 7. Time-based model selection: Random Forest vs Gradient Boosting

def make_preprocessor():
    return ColumnTransformer(
        [("ward_onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["ward"])],
        remainder="passthrough",
    )

def make_pipeline(model_name, final=False):
    if model_name == "random_forest":
        reg = RandomForestRegressor(
            n_estimators=300 if final else 250,
            min_samples_leaf=2,
            max_features=0.80,
            random_state=42,
            n_jobs=-1,
        )
    elif model_name == "gradient_boosting":
        reg = GradientBoostingRegressor(
            n_estimators=300 if final else 220,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=4,
            loss="huber",
            random_state=42,
        )
    else:
        raise ValueError(model_name)
    return Pipeline([("preprocess", make_preprocessor()), ("regressor", reg)])

def metrics(y_true, pred):
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    return {
        "mae_cases": float(mean_absolute_error(y, p)),
        "rmse_cases": float(mean_squared_error(y, p) ** 0.5),
        "correlation": float(np.corrcoef(y, p)[0, 1]) if np.std(y) and np.std(p) else 0.0,
        "within_25pct": float(np.mean(np.abs(y - p) <= 0.25 * (y + 1))),
    }

def fit_predict(model_name, train_frame, test_frame):
    pipe = make_pipeline(model_name, final=False)
    pipe.fit(train_frame[FEATURE_COLUMNS], train_frame["target_log"])
    pred = np.clip(np.expm1(pipe.predict(test_frame[FEATURE_COLUMNS])), 0, None)
    return pipe, pred

folds = [("2015", "2015-01-01", "2015-12-01"), ("2016", "2016-01-01", "2016-12-01"), ("2017", "2017-01-01", "2017-12-01"), ("2018", "2018-01-01", "2018-12-01")]
selection = {}
for model_name in ["random_forest", "gradient_boosting"]:
    fold_scores = []
    for label, start, end in folds:
        train_fold = features.loc[features.date < pd.Timestamp(start)]
        test_fold = features.loc[features.date.between(start, end)]
        _, pred = fit_predict(model_name, train_fold, test_fold)
        fold_scores.append({"fold": label, **metrics(test_fold.cases, pred)})
    selection[model_name] = {
        "folds": fold_scores,
        "mean_mae_cases": float(np.mean([x["mae_cases"] for x in fold_scores])),
        "mean_rmse_cases": float(np.mean([x["rmse_cases"] for x in fold_scores])),
    }
chosen_model = min(selection, key=lambda k: selection[k]["mean_mae_cases"])
print("Selected:", chosen_model)
display(pd.DataFrame([
    {"model": k, "mean_MAE": v["mean_mae_cases"], "mean_RMSE": v["mean_rmse_cases"]}
    for k, v in selection.items()
]).round(2))'''))

cells.append(nbf.v4.new_code_cell(r'''# 8. Honest 2017–2018 holdout and final model fit
train_holdout = features.loc[features.date < pd.Timestamp("2017-01-01")]
holdout = features.loc[features.date >= pd.Timestamp("2017-01-01")]
holdout_model, holdout_pred = fit_predict(chosen_model, train_holdout, holdout)
holdout_metrics = metrics(holdout.cases, holdout_pred)
holdout_metrics["seasonal_naive_baseline"] = metrics(holdout.cases, holdout.cases_lag_12.to_numpy())

# Ward-specific risk-label accuracy using thresholds learned only from pre-2017 data.
thresholds = train_holdout.groupby("ward")["cases"].quantile([0.50, 0.75]).unstack()
actual_labels, pred_labels = [], []
for (_, row), prediction in zip(holdout.iterrows(), holdout_pred):
    q50 = thresholds.loc[row["ward"], 0.50]
    q75 = thresholds.loc[row["ward"], 0.75]
    actual_labels.append("High" if row["cases"] >= q75 else ("Medium" if row["cases"] >= q50 else "Low"))
    pred_labels.append("High" if prediction >= q75 else ("Medium" if prediction >= q50 else "Low"))
holdout_metrics["risk_label_accuracy"] = float(np.mean(np.asarray(actual_labels) == np.asarray(pred_labels)))
holdout_metrics["n_rows"] = int(len(holdout))

final_model = make_pipeline(chosen_model, final=True)
final_model.fit(features[FEATURE_COLUMNS], features["target_log"])

print("2017–2018 holdout metrics:")
print(json.dumps(holdout_metrics, indent=2))'''))

cells.append(nbf.v4.new_code_cell(r'''# 9. 2026 planning forecast functions

def tree_interval(pipe, x):
    transformed = pipe.named_steps["preprocess"].transform(x[FEATURE_COLUMNS])
    reg = pipe.named_steps["regressor"]
    if hasattr(reg, "estimators_"):
        tree_log = np.vstack([tree.predict(transformed) for tree in reg.estimators_])
        tree_cases = np.clip(np.expm1(tree_log), 0, None)
        return np.median(tree_cases, axis=0), np.percentile(tree_cases, 10, axis=0), np.percentile(tree_cases, 90, axis=0)
    point = np.clip(np.expm1(pipe.predict(x[FEATURE_COLUMNS])), 0, None)
    return point, point * 0.75, point * 1.25

def seasonal_anchor():
    month_means = (
        monthly.loc[monthly.year.between(2014, 2018)]
        .assign(month=lambda x: x.date.dt.month)
        .groupby(["ward", "month"], as_index=False)["cases"]
        .mean()
        .rename(columns={"cases": "seasonal_mean"})
    )
    month_means["share"] = month_means["seasonal_mean"] / month_means.groupby("ward")["seasonal_mean"].transform("sum")
    latest = annual.loc[annual.year.eq(2023), ["ward", "cases"]].rename(columns={"cases": "latest_annual_cases"})
    anchor = month_means.merge(latest, on="ward", how="left")
    anchor["anchor_cases"] = anchor["share"] * anchor["latest_annual_cases"]
    return anchor, latest

def scenario_climate(scenario="typical", enso_value=0.0):
    dates = pd.date_range("2025-10-01", "2026-12-01", freq="MS")
    out = pd.DataFrame({"date": dates, "month": dates.month}).merge(climate_normals, on="month", how="left")
    out["rain_mm"] = out["normal_rain_mm"].astype(float)
    out["rainy_days"] = out["normal_rainy_days"].astype(float)
    out["heavy_rain_days"] = out["normal_heavy_rain_days"].astype(float)
    out["tmax_mean"] = out["normal_tmax"].astype(float)
    out["tmin_mean"] = out["normal_tmin"].astype(float)
    out["humidity_mean"] = out["normal_humidity"].astype(float)
    monsoon = out.month.isin([6, 7, 8, 9, 10])
    if scenario == "wetter_monsoon":
        out.loc[monsoon, ["rain_mm", "rainy_days", "heavy_rain_days"]] *= 1.30
        out.loc[~monsoon, ["rain_mm", "rainy_days", "heavy_rain_days"]] *= 1.10
        out[["tmax_mean", "tmin_mean"]] -= 0.2
    elif scenario == "drier_hotter":
        out.loc[monsoon, ["rain_mm", "rainy_days", "heavy_rain_days"]] *= 0.70
        out.loc[~monsoon, ["rain_mm", "rainy_days", "heavy_rain_days"]] *= 0.90
        out[["tmax_mean", "tmin_mean"]] += 0.8
        out["humidity_mean"] -= 3.0
    out["oni_trailing_3m"] = float(enso_value)
    return out

def forecast_feature_rows(anchor, climate_scenario, target_year=2026):
    dates = pd.date_range(f"{target_year}-01-01", f"{target_year}-12-01", freq="MS")
    clim = climate_scenario.set_index("date")
    anchor_idx = anchor.set_index(["ward", "month"])["anchor_cases"]
    rows = []
    for ward in sorted(meta.ward.unique()):
        m = meta.loc[meta.ward.eq(ward)].iloc[0]
        for date in dates:
            month_num = int(date.month)
            def a(delta):
                mm = (month_num - 1 - delta) % 12 + 1
                return float(anchor_idx.loc[(ward, mm)])
            prev1, prev2, prev3 = a(1), a(2), a(3)
            c1 = clim.loc[date - pd.offsets.MonthBegin(1)]
            c2 = clim.loc[date - pd.offsets.MonthBegin(2)]
            c3 = clim.loc[date - pd.offsets.MonthBegin(3)]
            rows.append({
                "ward": ward, "date": date,
                "cases_lag_1": prev1, "cases_lag_2": prev2, "cases_lag_3": prev3, "cases_lag_12": a(12),
                "cases_roll_3": np.mean([prev1, prev2, prev3]), "cases_roll_6": np.mean([a(i) for i in range(1, 7)]),
                "rain_lag_1": float(c1.rain_mm), "rain_3m_lag": float(c1.rain_mm + c2.rain_mm + c3.rain_mm),
                "rainy_days_lag_1": float(c1.rainy_days), "heavy_rain_days_lag_1": float(c1.heavy_rain_days),
                "tmax_lag_1": float(c1.tmax_mean), "tmin_lag_1": float(c1.tmin_mean),
                "humidity_lag_1": float(c1.humidity_mean), "oni_lag_1": float(c1.oni_trailing_3m),
                "month_sin": np.sin(2 * np.pi * month_num / 12), "month_cos": np.cos(2 * np.pi * month_num / 12),
                "slum_pct_imputed": float(m.slum_pct_imputed), "population_density_per_km2": float(m.population_density_per_km2),
                "elevation_m": float(m.elevation_m), "vulnerability_index": float(m.vulnerability_index), "population_2019": float(m.population_2019),
            })
    return pd.DataFrame(rows)

def add_risk_outputs(pred):
    thresholds_hist = monthly.groupby("ward")["cases"].quantile([0.50, 0.75, 0.90]).unstack().rename(columns={0.5: "q50", 0.75: "q75", 0.9: "q90"}).reset_index()
    history = {ward: group.cases.to_numpy() for ward, group in monthly.groupby("ward")}
    out = pred.merge(thresholds_hist, on="ward", how="left").merge(meta[["ward", "slum_pct"]], on="ward", how="left")
    out["risk_score"] = [100 * np.mean(history[w] <= v) for w, v in zip(out.ward, out.predicted_cases)]
    out["risk_level"] = np.select([out.risk_score >= 75, out.risk_score >= 50], ["High", "Medium"], default="Low")
    out["priority_score"] = 0.75 * out.risk_score + 0.25 * out.vulnerability_index
    out["predicted_cases_per_100k"] = out.predicted_cases / out.population_2019 * 100000
    return out

def forecast_2026(climate_scenario_name="typical", enso_value=0.0):
    anchor, latest = seasonal_anchor()
    scenario = scenario_climate(climate_scenario_name, enso_value)
    x = forecast_feature_rows(anchor, scenario)
    point, low, high = tree_interval(final_model, x)
    x["predicted_cases"], x["lower_80"], x["upper_80"] = point, low, high
    typical_x = forecast_feature_rows(anchor, scenario_climate("typical", 0.0))
    typical_point, _, _ = tree_interval(final_model, typical_x)
    scales = pd.DataFrame({"ward": typical_x.ward, "typical_point": typical_point}).groupby("ward", as_index=False).sum()
    scales = scales.merge(latest, on="ward", how="left")
    scales["calibration_scale"] = (scales.latest_annual_cases / scales.typical_point.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.35, 3.0)
    x = x.merge(scales[["ward", "latest_annual_cases", "calibration_scale"]], on="ward", how="left")
    for col in ["predicted_cases", "lower_80", "upper_80"]:
        x[col] = np.clip(x[col] * x.calibration_scale, 0, None)
    x["scenario"] = climate_scenario_name
    x["enso_value"] = enso_value
    x["climate_rain_mm_lag1"] = x.rain_lag_1
    x["climate_temp_lag1"] = x.tmax_lag_1
    return add_risk_outputs(x).sort_values(["date", "priority_score"], ascending=[True, False]).reset_index(drop=True)

forecast = forecast_2026("typical", 0.0)
forecast.to_csv(PROCESSED / "forecast_2026_era5_typical_neutral.csv", index=False)
print("2026 typical/neutral scenario total:", round(forecast.predicted_cases.sum()))
print("August 2026 scenario total:", round(forecast.loc[forecast.date.eq("2026-08-01"), "predicted_cases"].sum()))
display(forecast.loc[forecast.date.eq("2026-08-01"), ["ward", "predicted_cases", "lower_80", "upper_80", "risk_level", "risk_score", "priority_score"]].sort_values("priority_score", ascending=False).head(10))'''))

cells.append(nbf.v4.new_code_cell(r'''# 10. Spatial heat map and historical trend
# Enrich ward GeoJSON with forecast values for the selected month.
with open(WARD_FILE) as f:
    geojson = json.load(f)
map_df = forecast.loc[forecast.date.eq("2026-08-01")].copy()
for feature in geojson["features"]:
    ward = feature["properties"]["name"]
    row = map_df.loc[map_df.ward.eq(ward)].iloc[0]
    feature["properties"].update({
        "predicted_cases": float(row.predicted_cases),
        "risk_score": float(row.risk_score),
        "priority_score": float(row.priority_score),
        "risk_level": row.risk_level,
    })

fig = px.choropleth(
    map_df,
    geojson=geojson,
    locations="ward",
    featureidkey="properties.name",
    color="priority_score",
    color_continuous_scale="YlOrRd",
    hover_name="ward",
    hover_data={"predicted_cases": ":,.0f", "risk_score": ":.0f", "priority_score": ":.0f", "risk_level": True},
    title="Mumbai ward priority heat map — August 2026 ERA5 scenario",
)
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(height=600, margin={"l": 0, "r": 0, "t": 50, "b": 0})
fig.show()

annual_plot = annual.groupby("year", as_index=False).agg(cases=("cases", "sum"))
fig2 = px.line(annual_plot, x="year", y="cases", markers=True, title="Observed BMC dispensary diarrhoea cases")
fig2.update_layout(yaxis_title="Cases", xaxis_title="Year")
fig2.show()'''))

cells.append(nbf.v4.new_code_cell(r'''# 11. Save model, metrics and a data manifest for reuse outside the notebook.
feature_importance = {}
reg = final_model.named_steps["regressor"]
if hasattr(reg, "feature_importances_"):
    names = final_model.named_steps["preprocess"].get_feature_names_out()
    importance = pd.DataFrame({"feature": names, "importance": reg.feature_importances_}).sort_values("importance", ascending=False)
    feature_importance = {row.feature: float(row.importance) for row in importance.head(20).itertuples()}

metrics_report = {
    "chosen_model": chosen_model,
    "training_rows": int(len(features)),
    "training_date_start": str(features.date.min().date()),
    "training_date_end": str(features.date.max().date()),
    "walk_forward_selection": selection,
    "holdout_2017_2018": holdout_metrics,
    "top_feature_importance": feature_importance,
    "climate_source": "Copernicus ERA5 hourly time-series at nearest Mumbai grid point, aggregated to Mumbai-local daily/monthly features",
    "target_note": "BMC dispensary reported diarrhoea cases; zero-filled ward-months may include incomplete reporting.",
}

bundle = {
    "model": final_model,
    "metrics": metrics_report,
    "feature_columns": FEATURE_COLUMNS,
    "numeric_features": NUMERIC_FEATURES,
}
joblib.dump(bundle, MODELS / "model_bundle_era5.joblib")
(PROCESSED / "model_metrics_era5.json").write_text(json.dumps(metrics_report, indent=2))
(PROCESSED / "forecast_2026_era5_typical_neutral.csv").write_text(forecast.to_csv(index=False))
(PROCESSED / "era5_data_manifest.json").write_text(json.dumps({
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "climate_source": "Copernicus Climate Data Store ERA5 hourly time-series",
    "climate_archive": str(ERA5_ZIP),
    "climate_coverage": "2008-01-01 through 2018-12-31",
    "disease_source": str(PRAJA_CSV),
    "model": chosen_model,
    "api_key_saved": False,
}, indent=2))

print("Saved:")
print(" -", MODELS / "model_bundle_era5.joblib")
print(" -", PROCESSED / "model_metrics_era5.json")
print(" -", PROCESSED / "forecast_2026_era5_typical_neutral.csv")'''))

cells.append(nbf.v4.new_markdown_cell(r'''## Interpretation

The notebook produces a working model and a spatial planning dashboard artifact, but it cannot prove 2026 accuracy because monthly ward-level disease data after 2018 is not available in the supplied file.

For operational deployment, add monthly BMC/Praja data through 2025/2026, then perform a prospective 4-, 8- and 12-week backtest before using the alerts for public-health decisions.'''))

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

out = Path("mumbai_diarrhoea_era5_model.ipynb")
nbf.write(nb, out)
print(out, len(nb.cells), "cells")

# Mumbai diarrhoea early-warning system

A runnable prototype of the pipeline in the attached diagram: Praja/BMC disease data + climate + ENSO + vulnerability → leakage-safe features → time-based model selection → ward-level 2026 planning scenarios → spatial heat map.

## What is included

- `app.py` — Streamlit dashboard with an interactive 24-ward GeoJSON choropleth, priority queue, trends, model validation and data/method notes.
- `scripts/download_data.py` — downloads public ENSO, ward-boundary, elevation and Praja report sources. **The user-provided Praja CSV must be placed at `data/raw/praja_raw.csv`.**
- `scripts/download_era5.py` — downloads Copernicus ERA5 hourly time-series data using `CDS_API_KEY` without storing the token.
- `scripts/process_era5_timeseries.py` — converts the ERA5 time series to Mumbai-local daily and monthly climate features.
- `scripts/prepare_data.py` — aggregates the raw CSV and builds monthly/annual disease tables, climate normals, vulnerability attributes and an enriched ward GeoJSON.
- `src/forecast.py` — feature engineering, walk-forward model selection, 2026 scenario forecasting and risk scoring.
- `models/model_bundle.joblib` — trained model used by the dashboard.
- `models/model_metrics.json` — validation metrics and feature importance.
- `data/processed/forecast_2026_typical_neutral.csv` — default forecast artifact.

## Current data coverage and the 2026 limitation

The attached `praja_raw.csv` contains 80,000 rows, 13 diseases, 24 administrative wards and monthly records from **January 2008 through December 2018**. The monthly diarrhoea target is aggregated from dispensary to ward and month.

The model also uses the ward-level diarrhoea table in Praja Foundation's 2024 *State of Health in Mumbai* report, which extends annual **BMC-dispensary** diarrhoea counts through 2023. That table is used only to calibrate the latest ward level and is retained separately from the monthly training series.

Therefore, the 2026 output is a **planning scenario**, not a validated 2026 nowcast: there is no monthly ward outcome data after 2018 in the attached file. Before operational use, replace the raw extract with newer monthly BMC/Praja data, rerun the pipeline and re-evaluate the 4/8/12-week skill. The dashboard labels this limitation directly.

The product target is **reported BMC dispensary cases**, not all Mumbai government-hospital cases. This distinction matters: the Praja 2024 ward table is BMC dispensary data, while its citywide disease table also includes hospitals and other government facilities.

## Quick start

```bash
# From the project root
python -m pip install -r requirements.txt

# The attached CSV has already been staged in the build at data/raw/praja_raw.csv.
# If starting from a clean clone, copy it there first.

# Download the public non-climate layers.
python scripts/download_data.py

# ERA5 requires the Copernicus CDS Personal Access Token. Do not commit it.
export CDS_API_KEY="YOUR_CDS_PERSONAL_ACCESS_TOKEN"
python scripts/download_era5.py --start-year 2008 --end-year 2018
python scripts/process_era5_timeseries.py

python scripts/prepare_data.py
python -m src.forecast
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Open the printed local/preview URL. The dashboard sidebar controls:

- selected 2026 month and a 1–3 month period (approximately 4–12 weeks),
- typical, wetter-monsoon or drier/hotter climate scenario,
- neutral, latest downloaded NOAA ENSO, El Niño-like or La Niña-like assumption,
- map layer: forecast cases, risk score, cases per 100k, slum share or vulnerability.

## Data sources downloaded for this build

| Layer | Source | Coverage / use |
|---|---|---|
| Disease | User-provided Praja/BMC CSV | 2008–2018 monthly dispensary records; primary modelling target |
| Annual disease extension | [Praja Health White Paper 2024](https://www.praja.org/praja_docs/praja_downloads/Mumbai%20Health%20White%20Paper%202024.pdf), Table 19 | Ward-level BMC-dispensary diarrhoea totals 2014–2023 |
| Ward vulnerability | [Praja Health White Paper 2022](https://praja.org/praja_docs/praja_downloads/Mumbai%20Health%20White%20Paper%202022_Final.pdf), Table 4 | 2019 ward population, slum share, government facilities and 2021 dispensaries |
| Weather | [Copernicus ERA5 hourly time-series](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries?tab=download) | ERA5 hourly temperature, dew point and precipitation at the nearest Mumbai grid point, aggregated to Mumbai-local daily/monthly rainfall, rainy days, heavy rain, Tmax/Tmin and humidity |
| ENSO | [NOAA CPC Niño 3.4 monthly anomaly](https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/detrend.nino34.ascii.txt) | Trailing 3-month ENSO feature; latest downloaded values are shown in the dashboard |
| Spatial boundaries | [DataMeet Municipal Spatial Data — Mumbai](https://github.com/datameet/Municipal_Spatial_Data/tree/master/Mumbai) | 24 BMC administrative ward polygons; license/attribution is retained in `data/raw/Mumbai_Readme.md` |
| Elevation | Static ward representative-point elevation lookup | Ward elevation, used as a static vulnerability signal |

The ERA5 download requires a Copernicus CDS Personal Access Token and acceptance of the applicable ERA5 licence. The token is read from `CDS_API_KEY` and is not stored in this project. NOAA, GitHub and the public Praja report endpoints do not require a key. Rate limits and terms should still be checked before production automation.

## Modelling approach

1. **Aggregation:** filter `Disease == Diarrhoea`; sum `Occurrence` by ward and month. Missing ward-month combinations are zero-filled so the panel is regular, but a reporting-dispensary count is retained. A missing row may be no reported case or incomplete reporting, so this is not treated as a perfect surveillance denominator.
2. **Leakage-safe features:** disease lags 1/2/3/12 months, prior 3/6-month means, lagged rainfall and rainfall accumulation, lagged rainy/heavy-rain days, lagged temperature/humidity, lagged ENSO, month sine/cosine, population density, slum share, elevation and a vulnerability index. Weather and ENSO variables are lagged by at least one month.
3. **Model selection:** Random Forest and Gradient Boosting are compared with contiguous one-year forward folds for 2015, 2016, 2017 and 2018. The selected model is refit on the full 2009–2018 feature panel. The current build selects Random Forest and exposes its tree-ensemble 10th/90th percentile interval. On a pre-2017-trained 2017–2018 holdout, the Random Forest MAE is about 47.5 cases/ward-month versus about 102.4 for a seasonal-naive baseline; the historical risk-label accuracy is about 77%. These are retrospective skill measures, not 2026 validation.
4. **2026 scenario:** historical monthly seasonality is anchored to each ward's latest observed 2023 annual BMC-dispensary count. Weather beyond short-range observations is represented with 2008–2018 monthly normals under the chosen scenario. Future ENSO is an explicit assumption.
5. **Risk / priority:** risk is the empirical percentile of predicted monthly cases relative to each ward's 2008–2018 reported monthly distribution. High = ≥75th percentile, Medium = 50th–75th, Low <50th. Priority = 75% risk score + 25% static vulnerability index.

## Interpretation and production next steps

This is a decision-support prototype, not a diagnostic or causal model. It is intended to help a public-health team decide where to inspect reporting, water/sanitation complaints, stock ORS/IV fluids, and prepare outreach. It should not trigger an outbreak declaration automatically.

For an operational S2S system, the next high-value additions are:

- monthly BMC/Praja ward and dispensary records through 2025/2026, including reporting completeness and facility closures;
- a real-time BMC water-contamination / CCRS complaint feed and rainfall gauge or IMD station data;
- ward-level age structure, population denominator, water/sanitation access and updated slum data;
- explicit 4/8/12-week targets, alert thresholds agreed with BMC Public Health, and calibration/coverage tests;
- prospective backtesting after each new month, drift checks and human review before dissemination.

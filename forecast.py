"""Train and use the Mumbai ward-level diarrhoea forecasting model.

The outcome is reported BMC dispensary cases. The 2008-2018 monthly Praja extract
is the modelling series; Praja's 2024 health paper supplies annual ward totals through
2023 for a transparent level calibration. The 2026 output is a planning scenario,
not an observed 2026 case count, because no monthly ward outcome data after 2018 was
provided in the attached extract.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)

NUMERIC_FEATURES = [
    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_3",
    "cases_lag_12",
    "cases_roll_3",
    "cases_roll_6",
    "rain_lag_1",
    "rain_3m_lag",
    "rainy_days_lag_1",
    "heavy_rain_days_lag_1",
    "tmax_lag_1",
    "tmin_lag_1",
    "humidity_lag_1",
    "oni_lag_1",
    "month_sin",
    "month_cos",
    "slum_pct_imputed",
    "population_density_per_km2",
    "elevation_m",
    "vulnerability_index",
    "population_2019",
]
FEATURE_COLUMNS = ["ward"] + NUMERIC_FEATURES


def load_processed() -> dict[str, pd.DataFrame]:
    return {
        "monthly": pd.read_csv(PROCESSED / "ward_monthly_diarrhoea.csv", parse_dates=["date"]),
        "annual": pd.read_csv(PROCESSED / "ward_annual_diarrhoea.csv"),
        "climate": pd.read_csv(PROCESSED / "climate_monthly_observed.csv", parse_dates=["date"]),
        "normals": pd.read_csv(PROCESSED / "climate_monthly_normals_2008_2018.csv"),
        "oni": pd.read_csv(PROCESSED / "noaa_oni_monthly.csv", parse_dates=["date"]),
        "meta": pd.read_csv(PROCESSED / "ward_vulnerability.csv"),
    }


def _make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("ward_onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["ward"]),
        ],
        remainder="passthrough",
    )


def _make_pipeline(model_name: str, final: bool = False) -> Pipeline:
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
        raise ValueError(f"Unknown model {model_name}")
    return Pipeline([("preprocess", _make_preprocessor()), ("regressor", reg)])


def build_feature_frame(
    monthly: pd.DataFrame,
    climate: pd.DataFrame,
    oni: pd.DataFrame,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """Create leakage-safe monthly features.

    Every weather/ENSO variable is lagged by at least one month. Disease lags and
    rolling features also use only values strictly before the target month. This is
    stricter than using same-month weather and makes the feature table suitable for
    a lead-time early warning model.
    """
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
    c_cols = [
        "date",
        "rain_lag_1",
        "rain_3m_lag",
        "rainy_days_lag_1",
        "heavy_rain_days_lag_1",
        "tmax_lag_1",
        "tmin_lag_1",
        "humidity_lag_1",
    ]
    d = d.merge(c[c_cols], on="date", how="left", validate="many_to_one")

    o = oni[["date", "oni_trailing_3m"]].copy().sort_values("date").drop_duplicates("date")
    o["oni_lag_1"] = o["oni_trailing_3m"].shift(1)
    d = d.merge(o[["date", "oni_lag_1"]], on="date", how="left", validate="many_to_one")

    mcols = [
        "ward",
        "slum_pct_imputed",
        "population_density_per_km2",
        "elevation_m",
        "vulnerability_index",
        "population_2019",
    ]
    d = d.merge(meta[mcols], on="ward", how="left", validate="many_to_one")
    d["month_sin"] = np.sin(2 * np.pi * d["date"].dt.month / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["date"].dt.month / 12)
    d = d.dropna(subset=NUMERIC_FEATURES).copy()
    d["target_log"] = np.log1p(d["cases"].clip(lower=0))
    return d


def _metrics(y_true: pd.Series, pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(pred, dtype=float)
    return {
        "mae_cases": float(mean_absolute_error(y, p)),
        "rmse_cases": float(mean_squared_error(y, p) ** 0.5),
        "correlation": float(np.corrcoef(y, p)[0, 1]) if np.std(p) and np.std(y) else 0.0,
        "within_25pct": float(np.mean(np.abs(y - p) <= 0.25 * (y + 1))),
    }


def _fit_predict(model_name: str, train: pd.DataFrame, test: pd.DataFrame) -> tuple[Pipeline, np.ndarray]:
    pipe = _make_pipeline(model_name, final=False)
    pipe.fit(train[FEATURE_COLUMNS], train["target_log"])
    pred = np.clip(np.expm1(pipe.predict(test[FEATURE_COLUMNS])), 0, None)
    return pipe, pred


def walk_forward_model_selection(frame: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    """Compare ML candidates using contiguous, one-year forward validation folds."""
    folds = [
        ("2015", "2015-01-01", "2015-12-01"),
        ("2016", "2016-01-01", "2016-12-01"),
        ("2017", "2017-01-01", "2017-12-01"),
        ("2018", "2018-01-01", "2018-12-01"),
    ]
    scores: dict[str, Any] = {}
    for model_name in ["random_forest", "gradient_boosting"]:
        fold_scores = []
        for label, start, end in folds:
            train = frame.loc[frame["date"] < pd.Timestamp(start)]
            test = frame.loc[frame["date"].between(start, end)].copy()
            # A fold is only valid when it contains a full ward-month panel.
            if train.empty or test.empty:
                continue
            _, pred = _fit_predict(model_name, train, test)
            fold_scores.append({"fold": label, **_metrics(test["cases"], pred)})
        scores[model_name] = {
            "folds": fold_scores,
            "mean_mae_cases": float(np.mean([x["mae_cases"] for x in fold_scores])),
            "mean_rmse_cases": float(np.mean([x["rmse_cases"] for x in fold_scores])),
        }
    chosen = min(scores, key=lambda k: scores[k]["mean_mae_cases"])
    return chosen, scores


def _fit_final(frame: pd.DataFrame, model_name: str) -> Pipeline:
    pipe = _make_pipeline(model_name, final=True)
    pipe.fit(frame[FEATURE_COLUMNS], frame["target_log"])
    return pipe


def _tree_interval(pipe: Pipeline, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point and 10/90 percentile intervals for the random forest."""
    transformed = pipe.named_steps["preprocess"].transform(x[FEATURE_COLUMNS])
    reg = pipe.named_steps["regressor"]
    if hasattr(reg, "estimators_"):
        tree_log = np.vstack([tree.predict(transformed) for tree in reg.estimators_])
        trees = np.clip(np.expm1(tree_log), 0, None)
        return (
            np.median(trees, axis=0),
            np.percentile(trees, 10, axis=0),
            np.percentile(trees, 90, axis=0),
        )
    point = np.clip(np.expm1(pipe.predict(x[FEATURE_COLUMNS])), 0, None)
    return point, np.maximum(0, point * 0.75), point * 1.25


def _seasonal_anchor(data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly = data["monthly"].copy()
    annual = data["annual"].copy()
    month_means = (
        monthly.loc[monthly["year"].between(2014, 2018)]
        .assign(month=lambda x: x["date"].dt.month)
        .groupby(["ward", "month"], as_index=False)["cases"]
        .mean()
        .rename(columns={"cases": "seasonal_mean"})
    )
    month_means["share"] = month_means["seasonal_mean"] / month_means.groupby("ward")["seasonal_mean"].transform("sum")
    latest = annual.loc[annual["year"].eq(2023), ["ward", "cases"]].rename(columns={"cases": "latest_annual_cases"})
    anchor = month_means.merge(latest, on="ward", how="left")
    anchor["anchor_cases"] = anchor["share"] * anchor["latest_annual_cases"]
    annual_hist = (
        annual.loc[annual["year"].between(2014, 2018)]
        .groupby("ward", as_index=False)["cases"]
        .mean()
        .rename(columns={"cases": "historical_annual_mean_2014_2018"})
    )
    return anchor, latest, annual_hist


def _scenario_climate(normals: pd.DataFrame, climate_scenario: str, enso_value: float) -> pd.DataFrame:
    dates = pd.date_range("2025-10-01", "2026-12-01", freq="MS")
    out = pd.DataFrame({"date": dates, "month": dates.month}).merge(normals, on="month", how="left")
    out["rain_mm"] = out["normal_rain_mm"].astype(float)
    out["rainy_days"] = out["normal_rainy_days"].astype(float)
    out["heavy_rain_days"] = out["normal_heavy_rain_days"].astype(float)
    out["tmax_mean"] = out["normal_tmax"].astype(float)
    out["tmin_mean"] = out["normal_tmin"].astype(float)
    out["humidity_mean"] = out["normal_humidity"].astype(float)
    monsoon = out["month"].isin([6, 7, 8, 9, 10])
    if climate_scenario == "wetter_monsoon":
        out.loc[monsoon, ["rain_mm", "rainy_days", "heavy_rain_days"]] *= 1.30
        out.loc[~monsoon, ["rain_mm", "rainy_days", "heavy_rain_days"]] *= 1.10
        out[["tmax_mean", "tmin_mean"]] -= 0.2
    elif climate_scenario == "drier_hotter":
        out.loc[monsoon, ["rain_mm", "rainy_days", "heavy_rain_days"]] *= 0.70
        out.loc[~monsoon, ["rain_mm", "rainy_days", "heavy_rain_days"]] *= 0.90
        out[["tmax_mean", "tmin_mean"]] += 0.8
        out["humidity_mean"] -= 3.0
    out["oni_trailing_3m"] = float(enso_value)
    return out


def _feature_rows_for_2026(
    data: dict[str, pd.DataFrame],
    anchor: pd.DataFrame,
    scenario_climate: pd.DataFrame,
    target_year: int = 2026,
) -> pd.DataFrame:
    meta = data["meta"].copy()
    wards = sorted(meta["ward"].unique())
    target_dates = pd.date_range(f"{target_year}-01-01", f"{target_year}-12-01", freq="MS")
    clim = scenario_climate.set_index("date")
    anchor_idx = anchor.set_index(["ward", "month"])["anchor_cases"]
    rows = []
    for ward in wards:
        m = meta.loc[meta["ward"].eq(ward)].iloc[0]
        for date in target_dates:
            month = int(date.month)
            def a(delta: int) -> float:
                mm = (month - 1 - delta) % 12 + 1
                return float(anchor_idx.loc[(ward, mm)])
            prev1 = a(1)
            prev2 = a(2)
            prev3 = a(3)
            c1 = clim.loc[date - pd.offsets.MonthBegin(1)]
            c2 = clim.loc[date - pd.offsets.MonthBegin(2)]
            c3 = clim.loc[date - pd.offsets.MonthBegin(3)]
            rows.append(
                {
                    "ward": ward,
                    "date": date,
                    "cases_lag_1": prev1,
                    "cases_lag_2": prev2,
                    "cases_lag_3": prev3,
                    "cases_lag_12": a(12),
                    "cases_roll_3": np.mean([prev1, prev2, prev3]),
                    "cases_roll_6": np.mean([a(1), a(2), a(3), a(4), a(5), a(6)]),
                    "rain_lag_1": float(c1["rain_mm"]),
                    "rain_3m_lag": float(c1["rain_mm"] + c2["rain_mm"] + c3["rain_mm"]),
                    "rainy_days_lag_1": float(c1["rainy_days"]),
                    "heavy_rain_days_lag_1": float(c1["heavy_rain_days"]),
                    "tmax_lag_1": float(c1["tmax_mean"]),
                    "tmin_lag_1": float(c1["tmin_mean"]),
                    "humidity_lag_1": float(c1["humidity_mean"]),
                    "oni_lag_1": float(c1["oni_trailing_3m"]),
                    "month_sin": np.sin(2 * np.pi * month / 12),
                    "month_cos": np.cos(2 * np.pi * month / 12),
                    "slum_pct_imputed": float(m["slum_pct_imputed"]),
                    "population_density_per_km2": float(m["population_density_per_km2"]),
                    "elevation_m": float(m["elevation_m"]),
                    "vulnerability_index": float(m["vulnerability_index"]),
                    "population_2019": float(m["population_2019"]),
                }
            )
    return pd.DataFrame(rows)


def _add_risk_outputs(pred: pd.DataFrame, monthly: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    q = monthly.groupby("ward")["cases"].quantile([0.50, 0.75, 0.90]).unstack().rename(columns={0.5: "q50", 0.75: "q75", 0.9: "q90"}).reset_index()
    rows = []
    for ward, group in monthly.groupby("ward"):
        values = group["cases"].to_numpy()
        rows.append(pd.DataFrame({"ward": [ward], "p_hist": [values]}))
    hist = pd.concat(rows, ignore_index=True)
    out = pred.merge(q, on="ward", how="left").merge(meta[["ward", "slum_pct"]], on="ward", how="left")
    # Empirical percentile is more stable across wards with different reporting volumes.
    scores = []
    for _, row in out.iterrows():
        values = hist.loc[hist["ward"].eq(row["ward"]), "p_hist"].iloc[0]
        scores.append(100 * float(np.mean(values <= row["predicted_cases"])))
    out["risk_score"] = np.clip(scores, 0, 100)
    out["risk_level"] = np.select(
        [out["risk_score"] >= 75, out["risk_score"] >= 50],
        ["High", "Medium"],
        default="Low",
    )
    out["priority_score"] = 0.75 * out["risk_score"] + 0.25 * out["vulnerability_index"]
    out["predicted_cases_per_100k"] = out["predicted_cases"] / out["population_2019"] * 100000
    return out.drop(columns=["p_hist"] if "p_hist" in out.columns else [])


def forecast_2026(
    bundle: dict[str, Any],
    climate_scenario: str = "typical",
    enso_value: float = 0.0,
) -> pd.DataFrame:
    """Return a 12-month 2026 scenario forecast for all 24 administrative wards."""
    data = load_processed()
    anchor, latest, _ = _seasonal_anchor(data)
    scenario = _scenario_climate(data["normals"], climate_scenario, enso_value)
    x = _feature_rows_for_2026(data, anchor, scenario)
    point, low, high = _tree_interval(bundle["model"], x)

    # Calibrate the typical scenario to the latest observed ward annual level (2023).
    # The same calibration factor is reused for alternative climate/ENSO scenarios so
    # scenario differences are not washed out.
    typical_climate = _scenario_climate(data["normals"], "typical", 0.0)
    x_typical = _feature_rows_for_2026(data, anchor, typical_climate)
    typical_point, typical_low, typical_high = _tree_interval(bundle["model"], x_typical)
    x["predicted_cases"] = point
    x["lower_80"] = low
    x["upper_80"] = high
    x["scenario"] = climate_scenario
    x["enso_value"] = enso_value
    x["climate_rain_mm_lag1"] = x["rain_lag_1"]
    x["climate_temp_lag1"] = x["tmax_lag_1"]
    typical_df = pd.DataFrame({"ward": x_typical["ward"], "typical_point": typical_point})
    scales = typical_df.groupby("ward")["typical_point"].sum().reset_index()
    scales = scales.merge(latest, on="ward", how="left")
    scales["calibration_scale"] = scales["latest_annual_cases"] / scales["typical_point"].replace(0, np.nan)
    scales["calibration_scale"] = scales["calibration_scale"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.35, 3.0)
    x = x.merge(scales[["ward", "latest_annual_cases", "calibration_scale"]], on="ward", how="left")
    for col in ["predicted_cases", "lower_80", "upper_80"]:
        x[col] = np.clip(x[col] * x["calibration_scale"], 0, None)
    x["lower_80"] = np.minimum(x["lower_80"], x["predicted_cases"])
    x["upper_80"] = np.maximum(x["upper_80"], x["predicted_cases"])
    out = _add_risk_outputs(x, data["monthly"], data["meta"])
    return out.sort_values(["date", "priority_score"], ascending=[True, False]).reset_index(drop=True)


def train_and_save() -> dict[str, Any]:
    data = load_processed()
    frame = build_feature_frame(data["monthly"], data["climate"], data["oni"], data["meta"])
    chosen, selection = walk_forward_model_selection(frame)
    final_model = _fit_final(frame, chosen)

    # A final 2017-2018 holdout is shown in the dashboard as the primary validation view.
    # It is scored using a model fit only on pre-2017 rows; the production model below
    # is then re-fit on all available monthly data.
    train = frame.loc[frame["date"] < pd.Timestamp("2017-01-01")]
    holdout = frame.loc[frame["date"] >= pd.Timestamp("2017-01-01")]
    _, holdout_pred = _fit_predict(chosen, train, holdout)
    holdout_metrics = _metrics(holdout["cases"], holdout_pred)
    holdout_metrics["seasonal_naive_baseline"] = _metrics(holdout["cases"], holdout["cases_lag_12"].to_numpy())
    # Risk-label accuracy uses thresholds estimated only from pre-2017 data.
    thresholds = train.groupby("ward")["cases"].quantile([0.50, 0.75]).unstack()
    actual_labels = []
    pred_labels = []
    for (_, row), prediction in zip(holdout.iterrows(), holdout_pred):
        q50 = thresholds.loc[row["ward"], 0.50]
        q75 = thresholds.loc[row["ward"], 0.75]
        actual_labels.append("High" if row["cases"] >= q75 else ("Medium" if row["cases"] >= q50 else "Low"))
        pred_labels.append("High" if prediction >= q75 else ("Medium" if prediction >= q50 else "Low"))
    holdout_metrics["risk_label_accuracy"] = float(np.mean(np.asarray(actual_labels) == np.asarray(pred_labels)))
    holdout_metrics["n_rows"] = int(len(holdout))
    holdout_metrics["date_start"] = str(holdout["date"].min().date())
    holdout_metrics["date_end"] = str(holdout["date"].max().date())

    reg = final_model.named_steps["regressor"]
    feature_importance = {}
    if hasattr(reg, "feature_importances_"):
        names = final_model.named_steps["preprocess"].get_feature_names_out()
        imp = pd.DataFrame({"feature": names, "importance": reg.feature_importances_}).sort_values("importance", ascending=False)
        feature_importance = {str(row.feature): float(row.importance) for row in imp.head(20).itertuples()}

    metrics = {
        "chosen_model": chosen,
        "training_rows": int(len(frame)),
        "training_date_start": str(frame["date"].min().date()),
        "training_date_end": str(frame["date"].max().date()),
        "walk_forward_selection": selection,
        "holdout_2017_2018": holdout_metrics,
        "top_feature_importance": feature_importance,
        "target_note": "Monthly BMC dispensary reported diarrhoea cases; zero-filled ward-months may reflect missing reporting as well as zero cases.",
    }
    (MODELS / "model_metrics.json").write_text(json.dumps(metrics, indent=2))
    bundle = {"model": final_model, "metrics": metrics, "feature_columns": FEATURE_COLUMNS, "numeric_features": NUMERIC_FEATURES}
    joblib.dump(bundle, MODELS / "model_bundle.joblib")

    # Save the default typical/neutral forecast as a portable artifact for non-Streamlit use.
    forecast = forecast_2026(bundle, "typical", 0.0)
    forecast.to_csv(PROCESSED / "forecast_2026_typical_neutral.csv", index=False)
    return metrics


if __name__ == "__main__":
    result = train_and_save()
    print(json.dumps(result, indent=2))

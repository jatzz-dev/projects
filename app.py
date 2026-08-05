from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.forecast import forecast_2026, load_processed

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

st.set_page_config(
    page_title="Mumbai diarrhoea early warning",
    page_icon="◒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --navy:#132b52; --ink:#1d2b3d; --muted:#66758b; --blue:#2e5d99; --pale:#edf3fb; --sand:#f5efe4; --amber:#d69a29; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1,h2,h3 { font-family:'Space Grotesk', sans-serif !important; color:var(--navy); letter-spacing:-.02em; }
    h1 { font-size:2.2rem !important; margin-bottom:.15rem !important; }
    .hero { background:linear-gradient(120deg,#132b52 0%,#234f83 68%,#2e7090 100%); color:#fff; border-radius:18px; padding:27px 31px 25px; margin:4px 0 22px; box-shadow:0 10px 30px rgba(19,43,82,.16); }
    .hero h1 { color:#fff !important; font-size:2.25rem !important; margin:0 0 6px !important; }
    .hero p { color:#dbe8f7; margin:0; font-size:1rem; max-width:920px; line-height:1.5; }
    .eyebrow { text-transform:uppercase; letter-spacing:.16em; font-size:.70rem; color:#c4d9f2; font-weight:700; margin-bottom:7px; }
    .pill { display:inline-block; background:#dcecff; color:#244a7c; border-radius:999px; padding:4px 10px; font-size:.72rem; font-weight:700; margin:2px 4px 3px 0; }
    .note { background:#fff9ec; border-left:4px solid #d69a29; border-radius:8px; padding:11px 14px; color:#614c25; font-size:.88rem; line-height:1.45; }
    .source { color:#68768a; font-size:.76rem; line-height:1.45; }
    [data-testid="stMetric"] { background:#fff; border:1px solid #e4eaf1; border-radius:14px; padding:12px 14px; box-shadow:0 5px 16px rgba(26,51,87,.05); }
    [data-testid="stMetricLabel"] { color:#6e7c8e; font-size:.75rem; }
    [data-testid="stMetricValue"] { color:var(--navy); font-family:'Space Grotesk',sans-serif; }
    section[data-testid="stSidebar"] { background:#f6f8fb; border-right:1px solid #e7edf4; }
    div[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
    .smallcaps { text-transform:uppercase; font-size:.68rem; letter-spacing:.13em; font-weight:700; color:#738197; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_bundle():
    return joblib.load(MODELS / "model_bundle.joblib")


@st.cache_data(show_spinner=False)
def load_data():
    return load_processed()


@st.cache_data(show_spinner=False)
def get_forecast(climate_scenario: str, enso_value: float) -> pd.DataFrame:
    return forecast_2026(load_bundle(), climate_scenario=climate_scenario, enso_value=enso_value)


@st.cache_data(show_spinner=False)
def load_geojson():
    return json.loads((PROCESSED / "mumbai_wards_enriched.geojson").read_text())


def fmt_num(v: float, decimals: int = 0) -> str:
    if pd.isna(v):
        return "—"
    return f"{v:,.{decimals}f}"


def risk_color(level: str) -> str:
    return {"High": "#c94b4b", "Medium": "#d69a29", "Low": "#4e8b70"}.get(level, "#6e7c8e")


def make_map(map_df: pd.DataFrame, value_col: str, title: str, colorscale: str = "YlOrRd"):
    geojson = load_geojson()
    fig = px.choropleth(
        map_df,
        geojson=geojson,
        locations="ward",
        featureidkey="properties.ward",
        color=value_col,
        color_continuous_scale=colorscale,
        hover_name="ward",
        hover_data={
            "predicted_cases": ":,.0f",
            "lower_80": ":,.0f",
            "upper_80": ":,.0f",
            "risk_level": True,
            "risk_score": ":.0f",
            "slum_pct": ":.0f",
            "priority_score": ":.0f",
            value_col: False,
        },
    )
    meta = load_data()["meta"]
    fig.add_trace(
        go.Scattergeo(
            lon=meta["centroid_lon"],
            lat=meta["centroid_lat"],
            text=meta["ward"],
            mode="text",
            textfont={"size": 9, "color": "#19375f", "family": "DM Sans"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        projection_type="mercator",
        lataxis_range=[18.85, 19.35],
        lonaxis_range=[72.75, 73.05],
    )
    fig.update_layout(
        title={"text": title, "font": {"family": "Space Grotesk", "size": 17, "color": "#132b52"}, "x": 0.02},
        height=570,
        margin={"l": 0, "r": 0, "t": 52, "b": 0},
        paper_bgcolor="white",
        plot_bgcolor="white",
        coloraxis_colorbar={"title": "", "thickness": 13, "len": .65, "tickfont": {"size": 10}},
    )
    return fig


def make_rank_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["priority_score", "predicted_cases"], ascending=False)
    out["risk"] = out["risk_level"].map(lambda x: f"{x}")
    out = out.rename(
        columns={
            "ward": "Ward",
            "predicted_cases": "Forecast cases",
            "lower_80": "80% lower",
            "upper_80": "80% upper",
            "risk": "Risk",
            "risk_score": "Risk score",
            "priority_score": "Priority",
            "predicted_cases_per_100k": "Cases / 100k",
            "slum_pct": "Slum %",
        }
    )
    cols = ["Ward", "Forecast cases", "80% lower", "80% upper", "Risk", "Risk score", "Priority", "Cases / 100k", "Slum %"]
    out = out[cols]
    for c in ["Forecast cases", "80% lower", "80% upper"]:
        out[c] = out[c].round(0).astype(int)
    for c in ["Risk score", "Priority", "Cases / 100k", "Slum %"]:
        out[c] = out[c].round(1)
    return out


def main():
    bundle = load_bundle()
    data = load_data()
    metrics = bundle["metrics"]

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Mumbai · BMC Public Health · 2026 planning view</div>
          <h1>Diarrhoea early-warning system</h1>
          <p>Ward-level spatial intelligence for resource planning and outbreak preparedness. The model joins Praja/BMC disease counts to lagged weather, NOAA ENSO, and a vulnerability layer.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Forecast controls")
        st.caption("Select a 4–12 week planning window. Values are scenario forecasts, not a replacement for surveillance.")
        target_month = st.selectbox("Map period starts", list(range(1, 13)), index=7, format_func=lambda m: pd.Timestamp(2026, m, 1).strftime("%b 2026"))
        horizon = st.select_slider("Planning horizon", options=[1, 2, 3], value=1, format_func=lambda n: f"{n} month" + ("s" if n > 1 else ""))
        climate_label = st.selectbox(
            "Climate scenario",
            ["Typical climatology", "Wetter monsoon", "Drier / hotter"],
            help="2026 weather beyond the short-range forecast is unknown. These are planning scenarios built from 2008–2018 monthly normals.",
        )
        climate_key = {"Typical climatology": "typical", "Wetter monsoon": "wetter_monsoon", "Drier / hotter": "drier_hotter"}[climate_label]
        oni_latest = float(data["oni"].dropna(subset=["oni_trailing_3m"]).iloc[-1]["oni_trailing_3m"])
        enso_label = st.selectbox(
            "ENSO assumption",
            ["Neutral (0.0)", "Latest NOAA value", "El Niño-like (+0.8)", "La Niña-like (-0.8)"],
            help="NOAA Niño 3.4 observations are available through the latest downloaded month; future ENSO values are uncertain.",
        )
        enso_value = {"Neutral (0.0)": 0.0, "Latest NOAA value": oni_latest, "El Niño-like (+0.8)": 0.8, "La Niña-like (-0.8)": -0.8}[enso_label]
        map_metric_label = st.selectbox(
            "Map layer",
            ["Priority score", "Forecasted cases", "Risk score", "Forecast cases / 100k", "Slum share", "Vulnerability index"],
        )
        map_metric = {
            "Priority score": "priority_score",
            "Forecasted cases": "predicted_cases",
            "Risk score": "risk_score",
            "Forecast cases / 100k": "predicted_cases_per_100k",
            "Slum share": "slum_pct",
            "Vulnerability index": "vulnerability_index",
        }[map_metric_label]
        st.divider()
        st.markdown("**Status**")
        st.markdown('<span class="pill">ERA5 / CDS</span><span class="pill">24 wards</span><span class="pill">Spatial GeoJSON</span>', unsafe_allow_html=True)
        st.caption(f"Latest downloaded NOAA 3-month ENSO value: {oni_latest:+.2f}")

    fcast = get_forecast(climate_key, float(enso_value))
    start_date = pd.Timestamp(2026, target_month, 1)
    end_date = start_date + pd.offsets.MonthBegin(horizon)
    period = fcast.loc[fcast["date"].between(start_date, end_date - pd.offsets.MonthBegin(1))].copy()
    if period.empty:
        period = fcast.loc[fcast["date"].eq(start_date)].copy()
    map_df = (
        period.groupby("ward", as_index=False)
        .agg(
            predicted_cases=("predicted_cases", "sum"),
            lower_80=("lower_80", "sum"),
            upper_80=("upper_80", "sum"),
            risk_score=("risk_score", "max"),
            priority_score=("priority_score", "max"),
            predicted_cases_per_100k=("predicted_cases_per_100k", "sum"),
            risk_level=("risk_level", lambda s: "High" if "High" in set(s) else ("Medium" if "Medium" in set(s) else "Low")),
            slum_pct=("slum_pct", "first"),
            vulnerability_index=("vulnerability_index", "first"),
            latest_annual_cases=("latest_annual_cases", "first"),
        )
    )
    map_df["predicted_cases_per_100k"] = map_df["predicted_cases"] / map_df["ward"].map(data["meta"].set_index("ward")["population_2019"]) * 100000
    period_label = f"{start_date.strftime('%b')}–{(end_date - pd.offsets.MonthBegin(1)).strftime('%b %Y')}" if horizon > 1 else start_date.strftime("%b %Y")
    total_cases = float(map_df["predicted_cases"].sum())
    high_priority = int((map_df["priority_score"] >= 70).sum())
    peak = fcast.groupby("date", as_index=False)["predicted_cases"].sum().sort_values("predicted_cases", ascending=False).iloc[0]
    holdout = metrics["holdout_2017_2018"]

    st.markdown(f"<div class='smallcaps'>Selected planning window · {period_label} · {climate_label} · {enso_label}</div>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Expected BMC dispensary cases", fmt_num(total_cases))
    k2.metric("High-priority wards", f"{high_priority} / 24")
    k3.metric("Peak scenario month", pd.Timestamp(peak["date"]).strftime("%b"), f"{fmt_num(peak['predicted_cases'])} cases")
    k4.metric("Holdout MAE", f"{fmt_num(holdout['mae_cases'], 1)}", f"risk-label accuracy {holdout['risk_label_accuracy'] * 100:.0f}%")

    st.markdown(
        f'<div class="note"><b>Interpretation:</b> “High” means the period forecast is at or above the ward’s historical 75th percentile of reported monthly BMC dispensary cases. Priority combines forecast risk (75%) with vulnerability (25%). The interval is a model ensemble interval, not a formal public-health confidence interval.</div>',
        unsafe_allow_html=True,
    )
    st.write("")

    tab_map, tab_trend, tab_methods = st.tabs(["Spatial early warning", "Trends & validation", "Data & methods"])

    with tab_map:
        left, right = st.columns([1.7, 1])
        with left:
            st.plotly_chart(make_map(map_df, map_metric, f"Mumbai ward heat map · {map_metric_label} · {period_label}"), use_container_width=True, config={"displayModeBar": False})
            st.markdown('<div class="source">Boundaries: DataMeet / Municipal Spatial Data, Mumbai. Map polygons are administrative wards, not 227 electoral wards.</div>', unsafe_allow_html=True)
        with right:
            st.markdown("#### Response priority queue")
            st.dataframe(
                make_rank_table(map_df).head(12),
                use_container_width=True,
                hide_index=True,
                height=485,
                column_config={
                    "Risk": st.column_config.TextColumn(width="small"),
                    "Forecast cases": st.column_config.NumberColumn(format="%d"),
                    "80% lower": st.column_config.NumberColumn(format="%d"),
                    "80% upper": st.column_config.NumberColumn(format="%d"),
                },
            )
            st.download_button(
                "Download selected ward forecast (CSV)",
                data=make_rank_table(map_df).to_csv(index=False).encode("utf-8"),
                file_name=f"mumbai_diarrhoea_{start_date.strftime('%Y_%m')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("#### Vulnerability context")
        vm = data["meta"].copy().sort_values("vulnerability_index", ascending=False)
        vm["slum_pct"] = vm["slum_pct"].round(1)
        vm["vulnerability_index"] = vm["vulnerability_index"].round(1)
        vm["population_density_per_km2"] = vm["population_density_per_km2"].round(0)
        st.dataframe(
            vm[["ward", "vulnerability_index", "slum_pct", "population_2019", "population_density_per_km2", "elevation_m", "dispensaries_2021"]].rename(columns={"ward": "Ward", "vulnerability_index": "Vulnerability", "slum_pct": "Slum %", "population_2019": "Population (2019)", "population_density_per_km2": "Density / km²", "elevation_m": "Elevation (m)", "dispensaries_2021": "BMC dispensaries (2021)"}),
            use_container_width=True,
            hide_index=True,
            height=250,
        )

    with tab_trend:
        a1, a2 = st.columns([1.5, 1])
        with a1:
            annual = data["annual"].groupby("year", as_index=False).agg(cases=("cases", "sum"))
            annual["year"] = annual["year"].astype(str)
            fc_annual = fcast.assign(year_num=fcast["date"].dt.year).groupby("year_num", as_index=False).agg(cases=("predicted_cases", "sum"))
            fc_annual["year"] = fc_annual["year_num"].astype(str)
            fc_annual = fc_annual.drop(columns=["year_num"])
            # Use a compact observed + scenario timeline.
            trend = pd.concat([annual.assign(series="Observed"), fc_annual.assign(series="2026 planning scenario")], ignore_index=True)
            fig = px.line(trend, x="year", y="cases", color="series", markers=True, title="Annual BMC dispensary diarrhoea cases")
            fig.update_layout(height=390, margin={"l": 0, "r": 10, "t": 50, "b": 0}, legend_title_text="", yaxis_title="Reported / forecast cases", xaxis_title="Year")
            fig.update_traces(line={"width": 3})
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        with a2:
            st.markdown("#### Walk-forward validation")
            rows = []
            for name, info in metrics["walk_forward_selection"].items():
                rows.append({"Model": name.replace("_", " ").title(), "Mean MAE": info["mean_mae_cases"], "Mean RMSE": info["mean_rmse_cases"]})
            st.dataframe(pd.DataFrame(rows).round(1), hide_index=True, use_container_width=True)
            naive = metrics["holdout_2017_2018"]["seasonal_naive_baseline"]
            st.markdown(f"<div class='note'>Selected model: <b>{metrics['chosen_model'].replace('_', ' ').title()}</b>. The score is based on contiguous one-year forward folds, not a random split. On the 2017–18 holdout, its MAE was <b>{holdout['mae_cases']:.1f}</b> versus <b>{naive['mae_cases']:.1f}</b> for a seasonal-naive baseline. This is evidence of historical skill, not proof of 2026 accuracy.</div>", unsafe_allow_html=True)
            st.write("")
            imp = pd.DataFrame([{"Feature": k.replace("remainder__", "").replace("ward_onehot__", "Ward: "), "Importance": v} for k, v in metrics["top_feature_importance"].items()]).head(8).sort_values("Importance")
            figi = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Top model features", color="Importance", color_continuous_scale="Blues")
            figi.update_layout(height=300, margin={"l": 0, "r": 0, "t": 45, "b": 0}, coloraxis_showscale=False, xaxis_title="Relative importance", yaxis_title="")
            st.plotly_chart(figi, use_container_width=True, config={"displayModeBar": False})

        st.markdown("#### 2026 scenario climate")
        normals = data["normals"].copy()
        normal_fig = go.Figure()
        normal_fig.add_trace(go.Bar(x=[pd.Timestamp(2026, m, 1).strftime("%b") for m in normals["month"]], y=normals["normal_rain_mm"], name="Normal rainfall (mm)", marker_color="#4f86c6", yaxis="y"))
        normal_fig.add_trace(go.Scatter(x=[pd.Timestamp(2026, m, 1).strftime("%b") for m in normals["month"]], y=normals["normal_tmax"], name="Normal Tmax (°C)", line={"color": "#d16b42", "width": 3}, yaxis="y2"))
        normal_fig.update_layout(height=300, margin={"l": 0, "r": 0, "t": 20, "b": 0}, yaxis={"title": "Rain (mm)"}, yaxis2={"title": "Tmax (°C)", "overlaying": "y", "side": "right"}, legend={"orientation": "h", "y": 1.15}, xaxis_title="Month")
        st.plotly_chart(normal_fig, use_container_width=True, config={"displayModeBar": False})

    with tab_methods:
        st.markdown("### How this prototype follows the proposed pipeline")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**1 · Inputs & feature engineering**")
            st.markdown("- Praja/BMC diarrhoea occurrences are aggregated from dispensary → ward → month.\n- Weather is from the Copernicus ERA5 hourly time-series at the nearest Mumbai grid point; rainfall, rainy days, heavy-rain days, Tmax/Tmin and humidity are aggregated monthly.\n- ENSO is the NOAA Niño 3.4 anomaly with a trailing three-month mean.\n- Static vulnerability uses Praja ward population/slum share and ward elevation.")
            st.markdown("**2 · Leakage control**")
            st.markdown("Every weather/ENSO variable is lagged one month; disease lags and rolling features are strictly prior to the target month. Models are evaluated with forward, time-based folds.")
        with c2:
            st.markdown("**3 · Model, validation & output**")
            st.markdown("A global ward model compares Random Forest and Gradient Boosting on log-transformed cases. The selected Random Forest supplies an ensemble interval. Risk is a ward-specific empirical percentile and priority blends risk with vulnerability.")
            st.markdown("**4 · Important limitation**")
            st.markdown("The attached monthly Praja file ends in December 2018. Praja’s 2024 health paper adds annual ward-level BMC dispensary counts through 2023; the 2026 output uses those 2023 ward levels plus historical monthly seasonality and climate scenarios. It is therefore a planning scenario, not a validated 2026 nowcast. Upload newer monthly BMC/Praja data and re-run the pipeline before operational use.")
        st.markdown("### Data coverage")
        coverage = pd.DataFrame(
            [
                ["Praja raw CSV", "Disease / dispensary / ward / month", "2008–2018 monthly", "User-provided"],
                ["Praja Health White Paper 2024, Table 19", "Ward-level diarrhoea", "2014–2023 annual", "Downloaded public PDF"],
                ["Copernicus ERA5 time-series", "Rain, Tmax/Tmin, humidity", "2008–2018 hourly → daily", "Downloaded with CDS token"],
                ["NOAA CPC Niño 3.4", "ENSO anomaly", "1950–Jun 2026 monthly", "Downloaded; no key"],
                ["DataMeet Mumbai", "24 BMC ward polygons", "Administrative boundaries", "Downloaded GeoJSON"],
            ],
            columns=["Source", "Layer", "Coverage", "Status"],
        )
        st.dataframe(coverage, use_container_width=True, hide_index=True)
        st.markdown('<div class="source">Climate predictors are derived from the Copernicus ERA5 time-series and downloaded with a CDS Personal Access Token; the token is not stored in this project. NOAA CPC supplies the ENSO index. See README.md for URLs, attribution and re-run commands.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

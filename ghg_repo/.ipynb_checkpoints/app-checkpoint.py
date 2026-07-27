"""
GHG Emissions Analysis — Streamlit Dashboard
Week 6 deliverable.

Data expected in ./data/:
    owid-co2-data.csv        (raw OWID download)
    ghg_features.csv         (Week 2 export)
    scenario_projections.csv (Week 5 export, optional — Scenario Comparison tab
                               is skipped gracefully if this file is missing)

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

PROJECT_COUNTRIES = [
    "China", "United States", "India", "Russia", "Japan",
    "Germany", "South Africa", "Canada", "Brazil", "United Kingdom",
]

GAS_COLUMN_MAP = {"CO2": "co2", "Methane": "methane", "Nitrous Oxide": "nitrous_oxide"}

AGGREGATES_TO_EXCLUDE = [
    "World", "Asia", "Europe", "Africa", "North America", "South America", "Oceania",
    "European Union (27)", "European Union (28)",
    "High-income countries", "Low-income countries",
    "Upper-middle-income countries", "Lower-middle-income countries",
    "Asia (excl. China and India)", "Europe (excl. EU-27)", "Europe (excl. EU-28)",
    "North America (excl. USA)",
    "International transport", "International aviation", "International shipping",
    "Kuwaiti Oil Fires", "Kuwaiti Oil Fires (GCP)",
    "OECD (GCP)", "OECD (Jones et al.)", "Non-OECD (GCP)", "G20", "G7",
    "Africa (GCP)", "Asia (GCP)", "Europe (GCP)", "Middle East (GCP)",
    "North America (GCP)", "Oceania (GCP)", "South America (GCP)",
    "Central America (GCP)", "Ryukyu Islands (GCP)",
]

FORECAST_HORIZON_END = 2040
SCENARIO_COLORS = {"BAU": "#457B9D", "Moderate Mitigation": "#F4A261", "Aggressive Mitigation": "#2A9D8F"}


# ────────────────────────────────────────────────────────────────────────────
# Data loading — pure functions, independently testable outside Streamlit
# ────────────────────────────────────────────────────────────────────────────

def build_filtered_dataset(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Week 1 filtering logic: year >= 1990, exclude aggregate/non-sovereign entities."""
    return df_raw[
        (df_raw["year"] >= 1990) & (~df_raw["country"].isin(AGGREGATES_TO_EXCLUDE))
    ].copy()


def load_all_data(data_dir: str = "data"):
    df_raw = pd.read_csv(f"{data_dir}/owid-co2-data.csv")
    df_filtered = build_filtered_dataset(df_raw)
    df_features = pd.read_csv(f"{data_dir}/ghg_features.csv")

    try:
        df_scenarios = pd.read_csv(f"{data_dir}/scenario_projections.csv")
    except FileNotFoundError:
        df_scenarios = None

    return df_filtered, df_features, df_scenarios


def fit_ets_forecast(df_filtered: pd.DataFrame, country: str, horizon_end: int = FORECAST_HORIZON_END):
    """Fit ETS(A,Ad,N) on 1990-2018 for `country`, forecast through `horizon_end` with 95% CI."""
    train_co2 = (
        df_filtered[(df_filtered["country"] == country) & (df_filtered["year"].between(1990, 2018))]
        .sort_values("year")
        .set_index("year")["co2"]
        .dropna()
    )
    model = ExponentialSmoothing(train_co2, trend="add", damped_trend=True, seasonal=None)
    fit = model.fit(optimized=True)

    steps = horizon_end - 2018
    idx = pd.RangeIndex(2019, horizon_end + 1)
    try:
        fc_result = fit.get_forecast(steps)
        mean_fc = pd.Series(fc_result.predicted_mean.values, index=idx)
        ci = fc_result.conf_int(alpha=0.05)
        lower = pd.Series(ci.iloc[:, 0].values, index=idx)
        upper = pd.Series(ci.iloc[:, 1].values, index=idx)
    except Exception:
        mean_fc = pd.Series(fit.forecast(steps).values, index=idx)
        sims = fit.simulate(steps, repetitions=1000, error="add", random_state=42)
        sims.index = idx
        lower = sims.quantile(0.025, axis=1)
        upper = sims.quantile(0.975, axis=1)

    return train_co2, mean_fc, lower, upper, fit


# ────────────────────────────────────────────────────────────────────────────
# Streamlit app (only runs when invoked via `streamlit run app.py`)
# ────────────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="GHG Emissions Dashboard", layout="wide")

    df_filtered, df_features, df_scenarios = load_all_data()

    tabs = st.tabs([
        "Overview", "Historical Trends", "Country Profile",
        "Forecasts", "Scenario Comparison", "About",
    ])

    # ── Overview ──────────────────────────────────────────────────────────
    with tabs[0]:
        st.title("Global GHG Emissions Analysis Dashboard")
        st.markdown(
            "An analysis of CO₂ and GHG emissions trends (1990–2024) and ETS(A,Ad,N) "
            f"forecasts to {FORECAST_HORIZON_END} for 10 countries: "
            + ", ".join(PROJECT_COUNTRIES) + "."
        )

        latest_year = df_filtered["year"].max()
        global_latest = df_filtered[df_filtered["year"] == latest_year]["co2"].sum()
        global_1990 = df_filtered[df_filtered["year"] == 1990]["co2"].sum()
        pct_change = (global_latest - global_1990) / global_1990 * 100

        col1, col2, col3 = st.columns(3)
        col1.metric(f"Total Global CO₂ ({int(latest_year)})", f"{global_latest:,.0f} Mt")
        col2.metric("% Change Since 1990", f"{pct_change:+.1f}%")
        col3.metric("Countries Analysed (project focus)", len(PROJECT_COUNTRIES))
        st.caption("Global totals are summed across all sovereign countries in the filtered dataset, "
                   "not just the 10 project countries.")

    # ── Historical Trends ────────────────────────────────────────────────
    with tabs[1]:
        st.header("Historical Trends")

        selected_countries = st.multiselect(
            "Select countries to compare", options=PROJECT_COUNTRIES,
            default=PROJECT_COUNTRIES[:5],
        )
        if selected_countries:
            trend_data = df_filtered[
                df_filtered["country"].isin(selected_countries) & df_filtered["year"].between(1990, 2024)
            ]
            fig = px.line(
                trend_data, x="year", y="co2", color="country",
                title="CO₂ Emissions Over Time — Selected Countries",
                labels={"co2": "CO₂ (Mt)", "year": "Year"},
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Select at least one country above to see the comparison chart.")

        st.subheader("GHG Share by Gas Type (Global, per Decade)")
        selected_gases = st.multiselect(
            "Select gas types to include", options=list(GAS_COLUMN_MAP.keys()),
            default=list(GAS_COLUMN_MAP.keys()),
        )
        if selected_gases:
            gas_cols = [GAS_COLUMN_MAP[g] for g in selected_gases]
            gas_df = df_filtered.groupby("year")[gas_cols].sum().reset_index()
            gas_df["decade"] = (gas_df["year"] // 10 * 10).astype(str) + "s"
            decade_avg = gas_df.groupby("decade")[gas_cols].mean().reset_index()
            decade_long = decade_avg.melt(id_vars="decade", var_name="gas", value_name="emissions")
            inverse_map = {v: k for k, v in GAS_COLUMN_MAP.items()}
            decade_long["gas"] = decade_long["gas"].map(inverse_map)

            fig2 = px.area(
                decade_long, x="decade", y="emissions", color="gas",
                title="Global GHG Emissions Share by Gas Type per Decade",
                labels={"emissions": "Avg Annual Emissions (Mt CO₂-eq)", "decade": "Decade"},
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Select at least one gas type above.")

    # ── Country Profile ──────────────────────────────────────────────────
    with tabs[2]:
        st.header("Country Profile")
        profile_country = st.selectbox("Select a country", options=PROJECT_COUNTRIES, key="profile_country")

        hist = df_filtered[
            (df_filtered["country"] == profile_country) & (df_filtered["year"].between(1990, 2024))
        ].sort_values("year")
        feat = df_features[df_features["country"] == profile_country].sort_values("year")

        col1, col2 = st.columns(2)
        with col1:
            fig_emissions = px.line(hist, x="year", y="co2", title=f"{profile_country}: CO₂ Emissions Trend",
                                     labels={"co2": "CO₂ (Mt)", "year": "Year"})
            st.plotly_chart(fig_emissions, use_container_width=True)
        with col2:
            fig_percap = px.line(hist, x="year", y="co2_per_capita", title=f"{profile_country}: CO₂ Per Capita Trend",
                                  labels={"co2_per_capita": "CO₂ per capita (t)", "year": "Year"})
            st.plotly_chart(fig_percap, use_container_width=True)

        if not feat.empty and "co2_yoy_pct_change" in feat.columns:
            fig_yoy = px.bar(feat, x="year", y="co2_yoy_pct_change",
                              title=f"{profile_country}: Year-on-Year CO₂ % Change",
                              labels={"co2_yoy_pct_change": "YoY % Change", "year": "Year"})
            st.plotly_chart(fig_yoy, use_container_width=True)

        st.subheader("Key Stats")
        if not hist.empty:
            key_stats = pd.DataFrame({
                "Metric": ["Latest Year CO₂ (Mt)", "1990 CO₂ (Mt)", "% Change Since 1990",
                           "Latest Per-Capita CO₂ (t)", "Peak Year CO₂ (Mt)"],
                "Value": [
                    f"{hist['co2'].iloc[-1]:,.1f}",
                    f"{hist['co2'].iloc[0]:,.1f}",
                    f"{(hist['co2'].iloc[-1] - hist['co2'].iloc[0]) / hist['co2'].iloc[0] * 100:+.1f}%",
                    f"{hist['co2_per_capita'].iloc[-1]:,.2f}" if "co2_per_capita" in hist.columns else "N/A",
                    f"{hist['co2'].max():,.1f}",
                ],
            })
            st.table(key_stats)

    # ── Forecasts ─────────────────────────────────────────────────────────
    with tabs[3]:
        st.header("Forecasts — ETS(A,Ad,N)")
        forecast_country = st.selectbox("Select a country for forecast", options=PROJECT_COUNTRIES, key="forecast_country")

        with st.spinner("Fitting ETS(A,Ad,N) model..."):
            train_co2, mean_fc, lower, upper, fit = fit_ets_forecast(df_filtered, forecast_country)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train_co2.index, y=train_co2.values, mode="lines",
                                  name="Historical Actual (1990–2018)", line=dict(color="#1d3557")))
        fig.add_trace(go.Scatter(x=mean_fc.index, y=mean_fc.values, mode="lines",
                                  name=f"Forecast to {FORECAST_HORIZON_END}", line=dict(color="#e63946")))
        fig.add_trace(go.Scatter(
            x=list(mean_fc.index) + list(mean_fc.index[::-1]),
            y=list(upper.values) + list(lower.values[::-1]),
            fill="toself", fillcolor="rgba(230,57,70,0.15)", line=dict(color="rgba(255,255,255,0)"),
            name="95% CI", hoverinfo="skip",
        ))
        fig.update_layout(title=f"{forecast_country}: ETS(A,Ad,N) Forecast with 95% CI",
                           xaxis_title="Year", yaxis_title="CO₂ (Mt)")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Forecast Summary")
        summary_years = [2025, 2030, 2035, 2040]
        summary_table = pd.DataFrame({
            "Year": summary_years,
            "Forecast CO₂ (Mt)": [round(mean_fc.get(y, np.nan), 1) for y in summary_years],
            "Lower 95% CI": [round(lower.get(y, np.nan), 1) for y in summary_years],
            "Upper 95% CI": [round(upper.get(y, np.nan), 1) for y in summary_years],
        })
        st.table(summary_table)
        st.caption(f"φ (damping) = {fit.params['damping_trend']:.4f} — values closer to 1 imply the "
                   "current trend persists longer into the forecast; lower values imply faster flattening.")

    # ── Scenario Comparison ──────────────────────────────────────────────
    with tabs[4]:
        st.header("Scenario Comparison")
        if df_scenarios is None:
            st.info("`scenario_projections.csv` not found in the data/ folder — "
                     "complete Week 5 and add that file to enable this tab.")
        else:
            scenario_country = st.selectbox("Select a country", options=PROJECT_COUNTRIES, key="scenario_country")
            highlight_scenario = st.radio(
                "Highlight scenario for avoided-emissions comparison",
                options=["Moderate Mitigation", "Aggressive Mitigation"],
            )

            country_scenarios = df_scenarios[df_scenarios["country"] == scenario_country]
            fig = px.line(
                country_scenarios, x="year", y="co2_projected", color="scenario",
                title=f"{scenario_country}: Scenario Overlay (2025–{FORECAST_HORIZON_END})",
                color_discrete_map=SCENARIO_COLORS,
                labels={"co2_projected": "CO₂ (Mt)", "year": "Year"},
            )
            st.plotly_chart(fig, use_container_width=True)

            bau_total = country_scenarios[country_scenarios["scenario"] == "BAU"]["co2_projected"].sum()
            highlight_total = country_scenarios[country_scenarios["scenario"] == highlight_scenario]["co2_projected"].sum()
            avoided = bau_total - highlight_total
            st.metric(f"Cumulative CO₂ Avoided (2025–{FORECAST_HORIZON_END}) — {highlight_scenario} vs BAU",
                      f"{avoided:,.0f} Mt")

            st.subheader("Cumulative Emissions by Country and Scenario")
            cumulative = df_scenarios.groupby(["country", "scenario"])["co2_projected"].sum().reset_index()
            fig2 = px.bar(
                cumulative, x="country", y="co2_projected", color="scenario", barmode="group",
                color_discrete_map=SCENARIO_COLORS,
                title="Cumulative Emissions (2025–2040) by Country and Scenario",
                labels={"co2_projected": "Cumulative CO₂ (Mt)", "country": "Country"},
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── About ─────────────────────────────────────────────────────────────
    with tabs[5]:
        st.header("About This Project")
        st.markdown(f"""
**Data source:** [Our World in Data — CO₂ and Greenhouse Gas Emissions dataset](https://github.com/owid/co2-data),
covering {int(df_filtered['year'].min())}–{int(df_filtered['year'].max())}.

**Methodology summary:**
- Week 1: data acquisition, profiling, and filtering (aggregate/non-sovereign entities excluded).
- Week 2: feature engineering (lags, rolling means, YoY growth, GHG intensity).
- Week 3: baseline regression models (Naive, Linear Regression, Random Forest), temporal train/test split.
- Week 4: ETS(A,Ad,N) exponential smoothing forecasts to 2043 with simulation-based confidence intervals.
- Week 5 *(optional)*: scenario analysis — Business as Usual, Moderate Mitigation (2%/yr), Aggressive Mitigation (5%/yr).
- Week 6: this dashboard.

**Team / attribution:** *(replace with your name(s), course/internship name, and mentor, as applicable)*.
""")


if __name__ == "__main__":
    main()

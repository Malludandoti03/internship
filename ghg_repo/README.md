# GHG Emissions Forecasting & Scenario Analysis

A data science project analysing historical greenhouse gas emissions (1990–2024) and
producing statistical forecasts and policy scenario projections through 2040–2043,
for 10 countries: Brazil, Canada, China, Germany, India, Japan, Russia, South Africa,
United Kingdom, and United States.

## Project Description

This project walks through a full analysis pipeline: data acquisition and cleaning,
feature engineering, baseline machine-learning regression models, time-series
forecasting with exponential smoothing, illustrative policy scenario modelling, and
an interactive dashboard for exploring the results.

## Repository Structure

```
├── notebook/
│   └── ghg_analysis.ipynb        # Complete Jupyter Notebook (Weeks 1–5)
├── app.py                        # Streamlit dashboard (Week 6)
├── data/
│   ├── owid-co2-data.csv         # Raw OWID CO2 & GHG dataset
│   ├── ghg_features.csv          # Engineered feature set (Week 2 output)
│   └── scenario_projections.csv  # Scenario analysis output (Week 5 output)
├── requirements.txt
└── README.md
```

## Methodology Summary

- **Week 1 — Data Acquisition & EDA:** Loaded the OWID CO₂ dataset, profiled null
  values and coverage, filtered to 1990 onward and excluded aggregate/non-sovereign
  entities (World, continents, income groups, etc.), and produced initial trend charts.
- **Week 2 — Feature Engineering:** Built lag features (`co2_lag1/2/3`), a 5-year
  rolling mean, year-on-year growth rates, and GHG intensity (`total_ghg / gdp`).
- **Week 3 — Baseline ML Models:** Trained per-country Naive Baseline, Linear
  Regression, and Random Forest models using a temporal (not random) 1990–2018 /
  2019–2023 train-test split, and compared MAE/RMSE across models.
- **Week 4 — ETS(A,Ad,N) Forecasting:** Fit Holt's Damped Trend exponential
  smoothing per country and generated forecasts to 2043 with 95% confidence
  intervals (simulation-based where `get_forecast`/`conf_int` were unavailable).
- **Week 5 — Scenario Analysis (optional):** Modelled three illustrative policy
  scenarios — Business as Usual, Moderate Mitigation (2%/yr compounding
  reduction), and Aggressive Mitigation (5%/yr compounding reduction) — from
  2025–2040.
- **Week 6 — Dashboard:** An interactive Streamlit app (`app.py`) presenting
  overview KPIs, historical trends, per-country profiles, ETS forecasts, and
  scenario comparisons.

## Data Sources

- [Our World in Data — CO₂ and Greenhouse Gas Emissions dataset](https://github.com/owid/co2-data)
  (`owid-co2-data.csv`), covering 1750–2024.

## How to Run

### Jupyter Notebook

1. Install dependencies: `pip install -r requirements.txt`
2. Ensure `owid-co2-data.csv` is present in `data/`.
3. Open `notebook/ghg_analysis.ipynb` in Jupyter and run all cells top to bottom
   (Kernel → Restart & Run All).
4. Running the notebook regenerates `ghg_features.csv` and (if Week 5 is
   included) `scenario_projections.csv` into the working directory — copy these
   into `data/` for the dashboard to use them.

### Streamlit Dashboard

1. Install dependencies: `pip install -r requirements.txt`
2. Ensure `data/owid-co2-data.csv` and `data/ghg_features.csv` are present
   (and `data/scenario_projections.csv`, if you want the Scenario Comparison tab).
3. From the repository root, run:
   ```
   streamlit run app.py
   ```
4. The app opens in your browser, by default at `http://localhost:8501`.

## Team / Attribution

*(Replace with your name(s), course or internship program name, and mentor/supervisor, as applicable.)*

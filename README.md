# Pearls AQI Predictor

An end-to-end, serverless machine learning system that forecasts the Air Quality Index (AQI) for Islamabad, Pakistan up to 3 days in advance. Built as a data science internship project.

**Live dashboard:** https://danishwajid999-aqi-predict-app-atjs2t.streamlit.app/

**[Read the full project report](./Pearls_AQI_Predictor_Final_Report.docx)** for architecture details, model evaluation, EDA findings, and a full account of the engineering challenges solved along the way.

## What it does

- Collects live AQI + weather data every few hours from the AQICN API
- Backfilled with ~90 days of historical data from Open-Meteo to bootstrap training
- Trains 3 separate models (one per forecast day: 24h / 48h / 72h ahead) using Ridge Regression and Random Forest
- Serves live 3-day forecasts through an interactive Streamlit dashboard, with SHAP explanations and hazard alerts for unhealthy AQI levels
- Fully automated via GitHub Actions (feature collection + model retraining)

## Architecture

```
AQICN / Open-Meteo APIs
        |
        v
Feature Pipeline (feature_pipeline.py) ---> Hopsworks Feature Store
        ^                                          |
        |                                          v
Backfill (backfill_pipeline.py)          Training Pipeline (training_pipeline.py)
  [one-time historical seed]                       |
                                                     v
                                          Hopsworks Model Registry
                                                     |
                                                     v
                                        Streamlit Dashboard (app.py)
                                     [predictions + SHAP + alerts]
```

Both the feature pipeline and training pipeline run automatically on a schedule via GitHub Actions (see `.github/workflows/`).

> **Note on automation status:** Both workflows have been fully built and verified working on their own schedules (see the Actions run history in this repo, and Section 8 of the [final report](./Pearls_AQI_Predictor_Final_Report.docx)). They are intentionally left disabled between demos to stay within Hopsworks' free-tier compute budget, and can be re-enabled with a single click from the repo's Actions tab. The trained models and dashboard remain fully functional while automation is paused; only the live "current conditions" reading may not reflect the most recent hour.

## Project structure

| File | Purpose |
|---|---|
| `feature_pipeline.py` | Fetches current AQI/weather, engineers features, writes to the Feature Store |
| `backfill_pipeline.py` | One-time historical data seed (~90 days) from Open-Meteo |
| `training_pipeline.py` | Trains and evaluates Ridge/Random Forest models for 3 forecast horizons, saves best of each to the Model Registry |
| `app.py` | Streamlit dashboard: live conditions, 3-day forecast, SHAP explanations, hazard alerts |
| `eda.py` | Exploratory data analysis; generates the charts in `eda_output/` |
| `.github/workflows/` | GitHub Actions automation for the feature and training pipelines |
| `eda_output/` | Generated EDA charts (AQI trends, distributions, correlations) |
| `aqi_eda_data.csv` | Cached snapshot of feature data, so EDA can be re-run without Hopsworks access |

## Tech stack

Python 3.11, scikit-learn, Hopsworks (Feature Store + Model Registry), Streamlit, SHAP, GitHub Actions, AQICN API, Open-Meteo API.

## Running it locally

1. Clone the repo and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate      # Windows
   pip install -r requirements.txt
   pip install streamlit shap matplotlib seaborn
   ```
2. Create a `.env` file with your own credentials:
   ```
   AQICN_TOKEN=your_aqicn_token
   HOPSWORKS_API_KEY=your_hopsworks_key
   HOPSWORKS_PROJECT=your_hopsworks_project_name
   ```
3. Run the dashboard:
   ```
   python -m streamlit run app.py
   ```

## Known limitations

- AQI is reported from a single monitoring station, not a city-wide average, so it may differ from other sources.
- Day-3 forecast accuracy is meaningfully weaker than Day-1, since only current-moment weather is available as input (a production system would use an actual multi-day weather forecast as a feature).
- Only Ridge Regression and Random Forest were evaluated; a deep learning model was not implemented within the project's time/compute budget.

Full details, honest metrics, and the debugging story (a dead monitoring station, dependency conflicts, a train/serve skew bug, and a free-tier compute budget cap) are documented in the [final report](./Pearls_AQI_Predictor_Final_Report.docx).

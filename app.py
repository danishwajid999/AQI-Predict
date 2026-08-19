"""
Pearls AQI Predictor - Dashboard
===================================
Loads the 3 day-ahead models from the Hopsworks Model Registry, pulls the
most recent feature data from the Feature Store, and shows a 3-day AQI
forecast for Islamabad.

Run with: streamlit run app.py
"""

import os

import joblib
import pandas as pd
import shap
import streamlit as st
import hopsworks
from dotenv import load_dotenv

load_dotenv()


def get_secret(name):
    """Reads a credential from Streamlit Cloud's secrets manager if available,
    otherwise falls back to a local .env file (for local development)."""
    if name in st.secrets:
        return st.secrets[name]
    return os.getenv(name)


HOPSWORKS_API_KEY = get_secret("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = get_secret("HOPSWORKS_PROJECT")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

FEATURE_COLUMNS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind",
    "hour", "day", "month", "day_of_week", "aqi_change_rate",
    "aqi_lag_24h", "aqi_lag_48h", "aqi_rolling_6h", "aqi_rolling_24h",
]

DAY_MODELS = {
    "Day 1 (tomorrow)": "aqi_predictor_day_1",
    "Day 2": "aqi_predictor_day_2",
    "Day 3": "aqi_predictor_day_3",
}


def aqi_category(aqi):
    """Standard EPA AQI category + background color + a readable text color for it."""
    if aqi <= 50:
        return "Good", "#00e400", "#003d00"
    elif aqi <= 100:
        return "Moderate", "#ffff00", "#4d4d00"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "#ff7e00", "#ffffff"
    elif aqi <= 200:
        return "Unhealthy", "#ff0000", "#ffffff"
    elif aqi <= 300:
        return "Very Unhealthy", "#8f3f97", "#ffffff"
    else:
        return "Hazardous", "#7e0023", "#ffffff"


@st.cache_resource
def connect():
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    return project


@st.cache_data(ttl=3600)  # re-fetch at most once an hour, since underlying data only updates every 3 hours now
def load_latest_features(_project):
    """Pulls all feature rows and engineers the same trend features used in training."""
    fs = _project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df = df.sort_values("date").reset_index(drop=True)

    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_lag_48h"] = df["aqi"].shift(48)
    df["aqi_rolling_6h"] = df["aqi"].rolling(window=6).mean()
    df["aqi_rolling_24h"] = df["aqi"].rolling(window=24).mean()

    # Live rows (AQICN) only measure PM2.5 -- o3/no2/so2/co are always
    # missing. Fill them instead of dropping the row, so the dashboard
    # doesn't silently fall back to an old backfilled row when picking
    # the "latest" data point.
    for col in ["o3", "no2", "so2", "co", "pm10"]:
        df[col] = df[col].fillna(0)

    return df


@st.cache_resource
def load_models(_project):
    """Downloads and loads all 3 day-ahead models from the Model Registry."""
    mr = _project.get_model_registry()
    models = {}
    for label, registry_name in DAY_MODELS.items():
        model_meta = mr.get_best_model(registry_name, "rmse", "min")
        model_dir = model_meta.download()
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        models[label] = model
    return models


def compute_shap_explanation(model, X_row, background_df):
    """
    Breaks down a single prediction into how much each feature pushed it
    up or down. Works for both tree models (Random Forest) and linear
    models (Ridge) -- shap.Explainer automatically picks the right
    algorithm for each.
    """
    try:
        explainer = shap.Explainer(model, background_df)
        shap_values = explainer(X_row)
    except Exception:
        # Fallback for model types the fast path doesn't support directly
        explainer = shap.Explainer(model.predict, background_df)
        shap_values = explainer(X_row)

    values = shap_values.values[0]
    result = pd.DataFrame({"feature": FEATURE_COLUMNS, "impact": values})
    result["abs_impact"] = result["impact"].abs()
    return result.sort_values("abs_impact", ascending=False).head(6)


def main():
    st.set_page_config(page_title="Pearls AQI Predictor - Islamabad", page_icon="🌫️", layout="wide")
    st.title("🌫️ Pearls AQI Predictor")
    st.caption("3-day Air Quality Index forecast for Islamabad")

    with st.spinner("Connecting to Hopsworks..."):
        project = connect()

    with st.spinner("Loading latest data and models..."):
        df = load_latest_features(project)
        models = load_models(project)

    latest = df.dropna(subset=FEATURE_COLUMNS).iloc[-1]
    X_latest = pd.DataFrame([latest[FEATURE_COLUMNS]])

    current_aqi = latest["aqi"]
    current_category, current_color, current_text_color = aqi_category(current_aqi)

    # --- Current conditions ---
    st.subheader("Current Conditions")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current AQI", f"{current_aqi:.0f}", current_category)
    col2.metric("Temperature", f"{latest['temperature']:.0f} C")
    col3.metric("Humidity", f"{latest['humidity']:.0f}%")

    st.markdown(
        f"<div style='background-color:{current_color}; padding:10px; "
        f"border-radius:8px; text-align:center; font-weight:bold; color:{current_text_color};'>"
        f"{current_category}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # --- 3-day forecast ---
    st.subheader("3-Day Forecast")
    cols = st.columns(3)
    forecast_values = []

    for i, (label, model) in enumerate(models.items()):
        prediction = model.predict(X_latest)[0]
        forecast_values.append(prediction)
        category, color, text_color = aqi_category(prediction)

        with cols[i]:
            st.markdown(f"**{label}**")
            st.markdown(
                f"<div style='background-color:{color}; padding:20px; "
                f"border-radius:8px; text-align:center; color:{text_color};'>"
                f"<span style='font-size:32px; font-weight:bold;'>{prediction:.0f}</span><br>"
                f"{category}</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # --- SHAP explanations: why did the model predict this? ---
    st.subheader("Why these predictions? (SHAP feature importance)")
    background_sample = df[FEATURE_COLUMNS].dropna().sample(
        n=min(50, len(df.dropna(subset=FEATURE_COLUMNS))), random_state=42
    )

    for label, model in models.items():
        with st.expander(f"{label} -- what drove this prediction?"):
            with st.spinner("Computing SHAP values..."):
                explanation = compute_shap_explanation(model, X_latest, background_sample)
            explanation_display = explanation.set_index("feature")["impact"]
            st.bar_chart(explanation_display)
            st.caption(
                "Positive bars pushed the predicted AQI up; negative bars pulled it down. "
                "Only the top 6 most influential features are shown."
            )

    st.divider()

    # --- Hazard alert ---
    max_forecast = max(forecast_values)
    if max_forecast > 150:
        st.error(
            f"⚠️ HAZARD ALERT: AQI is forecast to reach {max_forecast:.0f} "
            f"({aqi_category(max_forecast)[0]}) in the next 3 days. "
            f"Consider limiting outdoor activity."
        )
    elif max_forecast > 100:
        st.warning(
            f"AQI is forecast to reach {max_forecast:.0f} "
            f"({aqi_category(max_forecast)[0]}) -- sensitive groups should take precautions."
        )
    else:
        st.success("No hazardous AQI levels forecast in the next 3 days.")

    # --- Recent trend chart ---
    st.subheader("Recent AQI Trend")
    recent = df.dropna(subset=["aqi"]).tail(72)
    st.line_chart(recent.set_index("date")["aqi"])

    st.caption(f"Last data update: {latest['date']}")


if __name__ == "__main__":
    main()
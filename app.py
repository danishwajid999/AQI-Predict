"""
Pearls AQI Predictor - Dashboard
==================================
Loads the 3 day-ahead models from the Hopsworks Model Registry, pulls the
most recent feature data from the Feature Store, and shows a 3-day AQI
forecast for Islamabad.

Run with: streamlit run app.py
"""

import os

import joblib
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
import hopsworks
from dotenv import load_dotenv


load_dotenv()


def get_secret(name):
    """Reads a credential from Streamlit Cloud's secrets manager if available,
    otherwise falls back to a local .env file (for local development)."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass

    return os.getenv(name)


HOPSWORKS_API_KEY = get_secret("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = get_secret("HOPSWORKS_PROJECT")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

FEATURE_COLUMNS = [
    "aqi",
    "pm25",
    "pm10",
    "o3",
    "no2",
    "so2",
    "co",
    "temperature",
    "humidity",
    "pressure",
    "wind",
    "hour",
    "day",
    "month",
    "day_of_week",
    "aqi_change_rate",
    "aqi_lag_24h",
    "aqi_lag_48h",
    "aqi_rolling_6h",
    "aqi_rolling_24h",
]


DAY_MODELS = {
    "Day 1 (tomorrow)": "aqi_predictor_day_1",
    "Day 2": "aqi_predictor_day_2",
    "Day 3": "aqi_predictor_day_3",
}


def aqi_category(aqi):
    """Standard EPA AQI category + background color + readable text color."""

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


@st.cache_data(ttl=3600)
def load_latest_features(_project):
    """Pulls all feature rows and engineers the same trend features used in training."""

    fs = _project.get_feature_store()

    fg = fs.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION
    )

    df = fg.read()

    df = df.sort_values("date").reset_index(drop=True)

    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_lag_48h"] = df["aqi"].shift(48)
    df["aqi_rolling_6h"] = df["aqi"].rolling(window=6).mean()
    df["aqi_rolling_24h"] = df["aqi"].rolling(window=24).mean()

    return df


@st.cache_resource
def load_models(_project):
    """Downloads and loads all 3 day-ahead models."""

    mr = _project.get_model_registry()

    models = {}
    model_info = {}

    for label, registry_name in DAY_MODELS.items():

        model_meta = mr.get_best_model(
            registry_name,
            "rmse",
            "min"
        )

        model_dir = model_meta.download()

        model = joblib.load(
            os.path.join(model_dir, "model.pkl")
        )

        models[label] = model

        model_info[label] = {
            "description": model_meta.description,
            "metrics": model_meta.training_metrics,
        }

    return models, model_info


def compute_shap_explanation(model, X_row, background_df):
    """
    Breaks down a single prediction into how much each feature pushed it
    up or down.
    """

    try:
        explainer = shap.Explainer(
            model,
            background_df
        )

        shap_values = explainer(X_row)

    except Exception:

        explainer = shap.Explainer(
            model.predict,
            background_df
        )

        shap_values = explainer(X_row)

    values = shap_values.values[0]

    result = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "impact": values
        }
    )

    result["abs_impact"] = result["impact"].abs()

    return result.sort_values(
        "abs_impact",
        ascending=False
    ).head(6)


def render_aqi_gauge(aqi_value):
    """Builds an unclipped, proportional gauge meter supporting dynamic light/dark themes."""

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=aqi_value,
            domain={"x": [0, 1], "y": [0, 1]},
            number={
                "suffix": " AQI",
                "font": {
                    "size": 22
                }
            },
            gauge={
                "axis": {
                    "range": [0, 500],
                    "tickmode": "array",
                    "tickvals": [0, 100, 200, 300, 400, 500],
                    "ticktext": ["0", "100", "200", "300", "400", "500"],
                    "tickwidth": 1,
                    "tickfont": {
                        "size": 10
                    }
                },
                "bar": {
                    "color": "black",
                    "thickness": 0.22
                },
                "steps": [
                    {"range": [0, 50], "color": "#00e400"},
                    {"range": [50, 100], "color": "#ffff00"},
                    {"range": [100, 150], "color": "#ff7e00"},
                    {"range": [150, 200], "color": "#ff0000"},
                    {"range": [200, 300], "color": "#8f3f97"},
                    {"range": [300, 500], "color": "#7e0023"},
                ],
            },
        )
    )

    fig.update_layout(
        height=145,
        margin=dict(l=15, r=15, t=25, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        autosize=True
    )

    return fig


def main():

    st.set_page_config(
        page_title="Pearls AQI Predictor - Islamabad",
        page_icon="🌫️",
        layout="wide"
    )

    st.title("🌫️ Pearls AQI Predictor")

    st.caption(
        "3-day Air Quality Index forecast for Islamabad"
    )

    with st.spinner("Connecting to Hopsworks..."):
        project = connect()

    with st.spinner("Loading latest data and models..."):
        df = load_latest_features(project)
        models, model_info = load_models(project)

    essential_cols = [
        "aqi",
        "pm25",
        "temperature",
        "humidity",
        "pressure",
        "wind",
        "hour",
        "day",
        "month",
        "day_of_week",
        "aqi_change_rate",
        "aqi_lag_24h",
        "aqi_lag_48h",
        "aqi_rolling_6h",
        "aqi_rolling_24h",
    ]

    latest = df.dropna(
        subset=essential_cols
    ).iloc[-1]

    model_input = latest[FEATURE_COLUMNS].copy()

    model_input[
        ["o3", "no2", "so2", "co", "pm10"]
    ] = model_input[
        ["o3", "no2", "so2", "co", "pm10"]
    ].fillna(0)

    X_latest = pd.DataFrame(
        [model_input]
    )

    current_aqi = latest["aqi"]

    current_category, current_color, current_text_color = aqi_category(
        current_aqi
    )

    # ==========================================================
    # CURRENT CONDITIONS
    # ==========================================================

    st.subheader("Current Conditions")
    st.markdown(
        """
        <style>
        div[data-testid="stMetric"] {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        div[data-testid="stMetric"] > div {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    col_aqi, col_temp, col_hum = st.columns([2, 1, 1], vertical_alignment="center")

    with col_aqi:
        with st.container(border=True):
            sub_metric, sub_gauge = st.columns([1, 1.2], vertical_alignment="center")

            with sub_metric:
                st.metric(
                    "Current AQI",
                    f"{current_aqi:.0f}",
                    current_category
                )

            with sub_gauge:
                st.plotly_chart(
                    render_aqi_gauge(current_aqi),
                    use_container_width=True,
                    theme="streamlit",
                    config={"displayModeBar": False}
                )

    with col_temp:
        with st.container(border=True):
            st.metric(
                "Temperature",
                f"{latest['temperature']:.0f} °C"
            )

    with col_hum:
        with st.container(border=True):
            st.metric(
                "Humidity",
                f"{latest['humidity']:.0f}%"
            )

    st.markdown(
        f"""
        <div style='
            background-color:{current_color};
            padding:10px;
            border-radius:8px;
            text-align:center;
            font-weight:bold;
            color:{current_text_color};
            margin-top: 10px;
        '>
            {current_category}
        </div>
        """,
        unsafe_allow_html=True
    )

    # ==========================================================
    # POLLUTANT READINGS
    # ==========================================================

    st.subheader("Pollutant Levels")

    def fmt_pollutant(value):
        return (
            f"{value:.0f}"
            if pd.notna(value)
            else "N/A (not measured by this station)"
        )

    pcol1, pcol2, pcol3 = st.columns(3)

    with pcol1:
        with st.container(border=True):
            st.metric("PM2.5", fmt_pollutant(latest["pm25"]))
        with st.container(border=True):
            st.metric("PM10", fmt_pollutant(latest["pm10"]))

    with pcol2:
        with st.container(border=True):
            st.metric("O3 (Ozone)", fmt_pollutant(latest["o3"]))
        with st.container(border=True):
            st.metric("NO2", fmt_pollutant(latest["no2"]))

    with pcol3:
        with st.container(border=True):
            st.metric("SO2", fmt_pollutant(latest["so2"]))
        with st.container(border=True):
            st.metric("CO", fmt_pollutant(latest["co"]))

    st.caption(
        "This station primarily measures PM2.5. PM10/O3/NO2/SO2/CO show 'N/A' "
        "for live readings the station doesn't record, and are treated as 0 "
        "internally when fed into the models."
    )

    st.divider()

    # ==========================================================
    # 3-DAY FORECAST
    # ==========================================================

    st.subheader("3-Day Forecast")

    cols = st.columns(3)

    forecast_values = []

    for i, (label, model) in enumerate(models.items()):

        prediction = model.predict(
            X_latest
        )[0]

        forecast_values.append(prediction)

        category, color, text_color = aqi_category(prediction)

        with cols[i]:
            st.markdown(f"**{label}**")
            st.markdown(
                f"""
                <div style='
                    background-color:{color};
                    padding:20px;
                    border-radius:8px;
                    text-align:center;
                    color:{text_color};
                '>
                    <span style='
                        font-size:32px;
                        font-weight:bold;
                    '>
                        {prediction:.0f}
                    </span>
                    <br>
                    {category}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    # ==========================================================
    # SHAP EXPLANATIONS
    # ==========================================================

    st.subheader("Why these predictions? (SHAP feature importance)")

    background_sample = df[
        FEATURE_COLUMNS
    ].dropna().sample(
        n=min(
            50,
            len(df.dropna(subset=FEATURE_COLUMNS))
        ),
        random_state=42
    )

    for label, model in models.items():

        with st.expander(f"{label} -- what drove this prediction?"):

            with st.spinner("Computing SHAP values..."):
                explanation = compute_shap_explanation(
                    model,
                    X_latest,
                    background_sample
                )

            explanation_display = (
                explanation
                .set_index("feature")["impact"]
                .sort_values()
            )

            st.bar_chart(
                explanation_display,
                horizontal=True
            )

            st.caption(
                "Positive bars pushed the predicted AQI up; negative bars pulled it down. "
                "Only the top 6 most influential features are shown."
            )

    st.divider()

    # ==========================================================
    # HAZARD ALERT
    # ==========================================================

    max_forecast = max(forecast_values)

    if max_forecast > 150:
        st.error(
            f"⚠️ HAZARD ALERT: AQI is forecast to reach "
            f"{max_forecast:.0f} "
            f"({aqi_category(max_forecast)[0]}) "
            f"in the next 3 days. "
            f"Consider limiting outdoor activity."
        )
    elif max_forecast > 100:
        st.warning(
            f"AQI is forecast to reach "
            f"{max_forecast:.0f} "
            f"({aqi_category(max_forecast)[0]}) "
            f"-- sensitive groups should take precautions."
        )
    else:
        st.success("No hazardous AQI levels forecast in the next 3 days.")

    # ==========================================================
    # RECENT AQI TREND
    # ==========================================================

    st.subheader("Recent AQI Trend")

    recent = df.dropna(subset=["aqi"]).tail(72)

    st.line_chart(
        recent.set_index("date")["aqi"]
    )

    st.divider()

    # ==========================================================
    # MODEL DETAILS
    # ==========================================================

    st.subheader("Model Details")

    st.caption(
        "Two model types were trained and compared for each forecast day: "
        "Ridge Regression and Random Forest. The better-performing one "
        "(by RMSE on held-out test data) was selected and deployed here."
    )

    model_rows = []

    for label in DAY_MODELS:
        info = model_info[label]
        metrics = info["metrics"] or {}

        model_rows.append(
            {
                "Forecast Day": label,
                "Algorithm Used":
                    info["description"].split("using ")[-1]
                    if info["description"]
                    else "N/A",
                "MAE": f"{metrics.get('mae', float('nan')):.2f}",
                "RMSE": f"{metrics.get('rmse', float('nan')):.2f}",
                "R²": f"{metrics.get('r2', float('nan')):.3f}",
            }
        )

    st.dataframe(
        pd.DataFrame(model_rows),
        use_container_width=True,
        hide_index=True
    )

    st.caption(f"Last data update: {latest['date']}")


if __name__ == "__main__":
    main()
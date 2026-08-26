"""
Training Pipeline 
  1. Pulls all features out of the Hopsworks Feature Store
  2. Builds the actual forecasting target: AQI 72 hours (3 days) into the future
  3. Splits data by TIME (not randomly) -- train on older data, test on newer
  4. Trains Ridge Regression and Random Forest
  5. Evaluates both with MAE, RMSE, R^2
  6. Saves whichever model performed better to the Model Registry
"""

import os
import shutil

import joblib
import numpy as np
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

FORECAST_HORIZON_HOURS = 72  

FEATURE_COLUMNS = [
    "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "pressure", "wind",
    "hour", "day", "month", "day_of_week", "aqi_change_rate",
    "aqi_lag_24h", "aqi_lag_48h", "aqi_rolling_6h", "aqi_rolling_24h",
]
TARGET_COLUMN = "aqi_target"


def load_data(fs):
    """Step 1: Pull everything from the Feature Store."""
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} rows from the Feature Store.")
    return df


def add_trend_features(df):
    """
    Gives the model actual trend information instead of just a single
    snapshot in time: what was AQI 24h and 48h ago, and what's the
    recent short/medium-term average been.
    """
    df = df.copy()
    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_lag_48h"] = df["aqi"].shift(48)
    df["aqi_rolling_6h"] = df["aqi"].rolling(window=6).mean()
    df["aqi_rolling_24h"] = df["aqi"].rolling(window=24).mean()

    for col in ["o3", "no2", "so2", "co", "pm10"]:
        df[col] = df[col].fillna(0)

    return df


def build_target(df, horizon):
    """Step 2: Create the N-hours-ahead forecasting target."""
    df = df.copy()
    df[TARGET_COLUMN] = df["aqi"].shift(-horizon)
    df = df.dropna(subset=[TARGET_COLUMN] + FEATURE_COLUMNS)
    return df


def time_based_split(df, test_fraction=0.2):
    """Step 3: Split by time -- train on the older 80%, test on the newest 20%."""
    split_index = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    return train_df, test_df


def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{model_name} results:")
    print(f"  MAE:  {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2:   {r2:.3f}")
    return {"model_name": model_name, "mae": mae, "rmse": rmse, "r2": r2}


def train_and_save_for_horizon(df, horizon, day_label, project):
    print(f"\n{'='*50}")
    print(f"Training models for {day_label} ({horizon}h ahead)")
    print(f"{'='*50}")

    target_df = build_target(df, horizon)
    print(f"{len(target_df)} rows available for this horizon.")
    train_df, test_df = time_based_split(target_df)

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df[TARGET_COLUMN]

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    ridge_results = evaluate(y_test, ridge.predict(X_test), f"{day_label} - Ridge")

    rf = RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=5,
        max_features="sqrt", random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_results = evaluate(y_test, rf.predict(X_test), f"{day_label} - Random Forest")

    if rf_results["rmse"] < ridge_results["rmse"]:
        best_model, best_name, best_results = rf, "random_forest", rf_results
    else:
        best_model, best_name, best_results = ridge, "ridge_regression", ridge_results

    print(f"Best for {day_label}: {best_name} (RMSE {best_results['rmse']:.2f})")

    model_dir = f"aqi_model_{day_label.lower().replace(' ', '_')}"
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    joblib.dump(best_model, os.path.join(model_dir, "model.pkl"))

    mr = project.get_model_registry()
    registry_name = f"aqi_predictor_{day_label.lower().replace(' ', '_')}"
    model = mr.python.create_model(
        name=registry_name,
        metrics={
            "mae": best_results["mae"],
            "rmse": best_results["rmse"],
            "r2": best_results["r2"],
        },
        description=f"AQI forecast for {day_label} ({horizon}h ahead) using {best_name}",
    )
    model.save(model_dir)
    print(f"Saved to registry as '{registry_name}'")


def main():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    df = load_data(fs)
    df = add_trend_features(df)

    horizons = [(24, "Day 1"), (48, "Day 2"), (72, "Day 3")]
    for horizon, day_label in horizons:
        train_and_save_for_horizon(df, horizon, day_label, project)

    print("\nSUCCESS: All 3 day-ahead models trained and saved to the Model Registry.")


if __name__ == "__main__":
    main()
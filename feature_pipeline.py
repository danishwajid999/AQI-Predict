"""
Feature Pipeline for Pearls AQI Predictor
==========================================
This script does 3 things every time it runs:
  1. Fetches the current AQI + weather reading for Islamabad from AQICN
  2. Engineers features from that raw reading
  3. Writes the feature row into a Hopsworks Feature Group

Run this manually for now. Later, GitHub Actions will run it automatically every hour.
"""

import os
from datetime import datetime, timezone

import pandas as pd
import requests
import hopsworks
from dotenv import load_dotenv

load_dotenv()

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")
CITY = "islamabad"  # used only as a label for our own records
# Islamabad coordinates. AQICN's geo endpoint automatically finds the
# nearest ACTIVE reporting station -- far more reliable than guessing
# a station ID from a webpage.
LATITUDE = 33.6844
LONGITUDE = 73.0479

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1  # clean restart -- old v1/v2/v3 deleted from Hopsworks


def fetch_raw_data():
    """Step 1: Get the current reading from AQICN."""
    url = f"https://api.waqi.info/feed/geo:{LATITUDE};{LONGITUDE}/?token={AQICN_TOKEN}"
    response = requests.get(url)
    data = response.json()

    if data.get("status") != "ok":
        raise RuntimeError(f"AQICN API error: {data}")

    d = data["data"]

    # Sanity check: if the station has no actual AQI value, something's
    # still wrong (e.g. bad coordinates) -- fail loudly instead of
    # silently inserting a row full of Nones.
    if d.get("aqi") is None:
        raise RuntimeError(f"AQICN returned no usable data: {data}")
    iaqi = d.get("iaqi", {})  # individual pollutant/weather readings

    # Helper to safely pull a value if it exists, else None
    def get_val(key):
        return iaqi.get(key, {}).get("v")

    return {
        "aqi": d.get("aqi"),
        "pm25": get_val("pm25"),
        "pm10": get_val("pm10"),
        "o3": get_val("o3"),
        "no2": get_val("no2"),
        "so2": get_val("so2"),
        "co": get_val("co"),
        "temperature": get_val("t"),
        "humidity": get_val("h"),
        "pressure": get_val("p"),
        "wind": get_val("w"),
        "observation_time": d.get("time", {}).get("iso"),
    }


def engineer_features(raw, previous_aqi=None):
    """Step 2: Turn the raw reading into model-ready features."""
    now = datetime.now(timezone.utc)

    aqi_change_rate = 0.0
    if previous_aqi is not None and raw["aqi"] is not None:
        aqi_change_rate = raw["aqi"] - previous_aqi

    return {
        "city": CITY,
        "date": now,  # used as the event-time column in the feature group
        "aqi": raw["aqi"],
        "pm25": raw["pm25"],
        "pm10": raw["pm10"],
        "o3": raw["o3"],
        "no2": raw["no2"],
        "so2": raw["so2"],
        "co": raw["co"],
        "temperature": raw["temperature"],
        "humidity": raw["humidity"],
        "pressure": raw["pressure"],
        "wind": raw["wind"],
        "hour": now.hour,
        "day": now.day,
        "month": now.month,
        "day_of_week": now.weekday(),  # 0 = Monday, 6 = Sunday
        "aqi_change_rate": aqi_change_rate,
    }


def get_previous_aqi(fg):
    """Look up the most recent stored AQI value, so we can compute change rate."""
    try:
        df = fg.read()
        if df.empty:
            return None
        df = df.sort_values("date", ascending=False)
        return float(df.iloc[0]["aqi"])
    except Exception:
        # Feature group might be empty or brand new
        return None


def main():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    # get_or_create_feature_group makes it the first time, and just connects
    # to it on every future run
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI + weather features for Islamabad",
        primary_key=["city", "date"],
        event_time="date",
        time_travel_format="HUDI",
    )

    print("Fetching previous AQI value (for change-rate calculation)...")
    previous_aqi = get_previous_aqi(fg)
    print(f"Previous AQI: {previous_aqi}")

    print("Fetching current data from AQICN...")
    raw = fetch_raw_data()
    print(f"Raw data: {raw}")

    features = engineer_features(raw, previous_aqi)
    print(f"Engineered features: {features}")

    df = pd.DataFrame([features])

    # Some pollutants (e.g. pm10, o3) may come back as None if the station
    # doesn't measure them. Force these columns to be proper floats (with
    # NaN for missing) so Hopsworks can infer a numeric type instead of
    # rejecting an ambiguous all-None column.
    numeric_cols = [
        "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "temperature", "humidity", "pressure", "wind",
        "hour", "day", "month", "day_of_week", "aqi_change_rate",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    print("Writing to Feature Store...")
    fg.insert(df)

    print("SUCCESS: Feature row inserted.")


if __name__ == "__main__":
    main()
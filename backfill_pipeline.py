"""
Backfill Pipeline 
AQICN (our live data source) doesn't provide historical data on the free
tier. This script uses Open-Meteo instead, a free API with a genuine
historical archive, no API key required, to pull the past 90 days of
hourly air quality + weather data for Islamabad, engineer the same
features as our live pipeline, and bulk-insert them into the Feature
Store in one go.

"""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
import hopsworks
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

CITY = "islamabad"
LATITUDE = 33.6844
LONGITUDE = 73.0479
DAYS_BACK = 90
END_BUFFER_DAYS = 3 
FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1


def calculate_aqi_from_pm25(pm25):
    """
    Converts a raw PM2.5 concentration (µg/m³) into the US EPA Air Quality
    Index, using the official EPA breakpoint table. This keeps our
    historical 'aqi' values consistent with what AQICN reports live
    (AQICN's index is also PM2.5-based for this station).
    """
    if pm25 is None or pd.isna(pm25):
        return None

    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    for c_lo, c_hi, aqi_lo, aqi_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round(
                (aqi_hi - aqi_lo) / (c_hi - c_lo) * (pm25 - c_lo) + aqi_lo
            )
    return 500  

def fetch_air_quality_history(start_date, end_date):
    """Pulls hourly pollutant history from Open-Meteo's Air Quality API."""
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Karachi",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["hourly"]
    return pd.DataFrame(data)


def fetch_weather_history(start_date, end_date):
    """Pulls hourly weather history from Open-Meteo's Weather Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m",
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Karachi",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()["hourly"]
    return pd.DataFrame(data)


def build_feature_dataframe(air_df, weather_df):
    """Merges both sources and engineers the same features as the live pipeline."""
    df = pd.merge(air_df, weather_df, on="time", how="inner")

    df["time"] = pd.to_datetime(df["time"])
    df["aqi"] = df["pm2_5"].apply(calculate_aqi_from_pm25)

    df = df.rename(
        columns={
            "time": "date",
            "pm2_5": "pm25",
            "carbon_monoxide": "co",
            "nitrogen_dioxide": "no2",
            "sulphur_dioxide": "so2",
            "ozone": "o3",
            "temperature_2m": "temperature",
            "relative_humidity_2m": "humidity",
            "surface_pressure": "pressure",
            "wind_speed_10m": "wind",
        }
    )

    df["city"] = CITY
    df["hour"] = df["date"].dt.hour
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    df = df.sort_values("date").reset_index(drop=True)
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0.0)

   
    df = df.dropna(subset=["aqi"])

    numeric_cols = [
        "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
        "temperature", "humidity", "pressure", "wind",
        "hour", "day", "month", "day_of_week", "aqi_change_rate",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # Hopsworks needs timezone-aware timestamps to match the live pipeline
    df["date"] = df["date"].dt.tz_localize("Asia/Karachi").dt.tz_convert("UTC")

    return df[
        ["city", "date", "aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
         "temperature", "humidity", "pressure", "wind",
         "hour", "day", "month", "day_of_week", "aqi_change_rate"]
    ]


def main():
    end_date = datetime.now().date() - timedelta(days=END_BUFFER_DAYS)
    start_date = end_date - timedelta(days=DAYS_BACK)
    print(f"Backfilling from {start_date} to {end_date}...")

    print("Fetching historical air quality data...")
    air_df = fetch_air_quality_history(str(start_date), str(end_date))

    print("Fetching historical weather data...")
    weather_df = fetch_weather_history(str(start_date), str(end_date))

    print("Merging and engineering features...")
    feature_df = build_feature_dataframe(air_df, weather_df)
    print(f"Built {len(feature_df)} rows of historical features.")
    print(feature_df.head())

    print("\nConnecting to Hopsworks...")
    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        description="Hourly AQI + weather features for Islamabad",
        primary_key=["city", "date"],
        event_time="date",
        time_travel_format="HUDI",
    )

    print(f"Inserting {len(feature_df)} rows into the Feature Store...")
    fg.insert(feature_df)

    print("SUCCESS: Backfill complete.")


if __name__ == "__main__":
    main()
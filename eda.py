"""
EDA (Exploratory Data Analysis) for Pearls AQI Predictor
============================================================
Fetches your feature data from Hopsworks ONCE, saves it locally as a
CSV, then does all analysis on that local file -- no repeated Hopsworks
calls, so this doesn't eat into your compute budget beyond one read.

If you run this again later, it will reuse the saved CSV instead of
re-fetching, unless you delete aqi_eda_data.csv first.

Run with: python eda.py
Output: a set of PNG charts in the 'eda_output' folder, plus printed
summary stats you can paste directly into your report.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

LOCAL_CSV = "aqi_eda_data.csv"
OUTPUT_DIR = "eda_output"


def get_data():
    """Reuses the local CSV if it exists, otherwise fetches from Hopsworks once."""
    if os.path.exists(LOCAL_CSV):
        print(f"Using cached local data: {LOCAL_CSV}")
        df = pd.read_csv(LOCAL_CSV, parse_dates=["date"])
        return df

    print("No local cache found -- fetching from Hopsworks (one-time read)...")
    import hopsworks

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    df = fg.read()
    df = df.sort_values("date").reset_index(drop=True)

    df.to_csv(LOCAL_CSV, index=False)
    print(f"Saved to {LOCAL_CSV} for future reuse.")
    return df


def summary_stats(df):
    print("\n" + "=" * 50)
    print("SUMMARY STATISTICS")
    print("=" * 50)
    print(f"\nTotal rows: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nAQI stats:\n{df['aqi'].describe()}")
    print(f"\nMissing values per column:\n{df.isnull().sum()}")


def plot_aqi_over_time(df, out_dir):
    plt.figure(figsize=(14, 5))
    plt.plot(df["date"], df["aqi"], linewidth=0.8)
    plt.title("AQI Over Time - Islamabad")
    plt.xlabel("Date")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "aqi_over_time.png"), dpi=120)
    plt.close()


def plot_aqi_distribution(df, out_dir):
    plt.figure(figsize=(8, 5))
    sns.histplot(df["aqi"].dropna(), bins=40, kde=True)
    plt.title("Distribution of AQI Values")
    plt.xlabel("AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "aqi_distribution.png"), dpi=120)
    plt.close()


def plot_aqi_by_hour(df, out_dir):
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df, x="hour", y="aqi")
    plt.title("AQI by Hour of Day")
    plt.xlabel("Hour")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "aqi_by_hour.png"), dpi=120)
    plt.close()


def plot_aqi_by_day_of_week(df, out_dir):
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df_plot = df.copy()
    df_plot["day_name"] = df_plot["day_of_week"].apply(lambda x: day_names[int(x)])
    plt.figure(figsize=(9, 5))
    sns.boxplot(data=df_plot, x="day_name", y="aqi", order=day_names)
    plt.title("AQI by Day of Week")
    plt.xlabel("Day")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "aqi_by_day_of_week.png"), dpi=120)
    plt.close()


def plot_correlation_heatmap(df, out_dir):
    cols = ["aqi", "pm25", "pm10", "o3", "no2", "so2", "co",
             "temperature", "humidity", "pressure", "wind"]
    corr = df[cols].corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "correlation_heatmap.png"), dpi=120)
    plt.close()


def plot_temperature_vs_aqi(df, out_dir):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="temperature", y="aqi", alpha=0.3, s=15)
    plt.title("Temperature vs AQI")
    plt.xlabel("Temperature (C)")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "temperature_vs_aqi.png"), dpi=120)
    plt.close()


def plot_humidity_vs_aqi(df, out_dir):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="humidity", y="aqi", alpha=0.3, s=15)
    plt.title("Humidity vs AQI")
    plt.xlabel("Humidity (%)")
    plt.ylabel("AQI")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "humidity_vs_aqi.png"), dpi=120)
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = get_data()
    summary_stats(df)

    print("\nGenerating charts...")
    plot_aqi_over_time(df, OUTPUT_DIR)
    plot_aqi_distribution(df, OUTPUT_DIR)
    plot_aqi_by_hour(df, OUTPUT_DIR)
    plot_aqi_by_day_of_week(df, OUTPUT_DIR)
    plot_correlation_heatmap(df, OUTPUT_DIR)
    plot_temperature_vs_aqi(df, OUTPUT_DIR)
    plot_humidity_vs_aqi(df, OUTPUT_DIR)

    print(f"\nSUCCESS: All charts saved to the '{OUTPUT_DIR}' folder.")
    print("Open them and pick the most interesting ones for your report.")


if __name__ == "__main__":
    main()
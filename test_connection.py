"""
Quick sanity check: confirms your .env is set up correctly and that
you can reach both AQICN and Hopsworks before writing the full pipeline.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()  # reads variables from your .env file

AQICN_TOKEN = os.getenv("AQICN_TOKEN")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT")

print("=== Step 1: Checking .env values loaded ===")
print("AQICN_TOKEN loaded:", bool(AQICN_TOKEN))
print("HOPSWORKS_API_KEY loaded:", bool(HOPSWORKS_API_KEY))
print("HOPSWORKS_PROJECT:", HOPSWORKS_PROJECT)

print("\n=== Step 2: Testing AQICN API ===")
url = f"https://api.waqi.info/feed/islamabad/?token={AQICN_TOKEN}"
response = requests.get(url)
data = response.json()

if data.get("status") == "ok":
    aqi = data["data"]["aqi"]
    city = data["data"]["city"]["name"]
    print(f"SUCCESS: Current AQI in {city} is {aqi}")
else:
    print("FAILED to fetch AQICN data. Response was:")
    print(data)

print("\n=== Step 3: Testing Hopsworks connection ===")
try:
    import hopsworks

    project = hopsworks.login(
        project=HOPSWORKS_PROJECT,
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        api_key_value=HOPSWORKS_API_KEY,
    )
    fs = project.get_feature_store()
    print(f"SUCCESS: Connected to Hopsworks project '{project.name}'")
except Exception as e:
    print("FAILED to connect to Hopsworks. Error was:")
    print(e)
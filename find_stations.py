"""
Finds AQICN monitoring stations near Islamabad and shows which ones
are actively reporting recent data vs stale/dead stations.
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()
AQICN_TOKEN = os.getenv("AQICN_TOKEN")

# Search by keyword
url = f"https://api.waqi.info/search/?keyword=islamabad&token={AQICN_TOKEN}"
response = requests.get(url)
data = response.json()

if data.get("status") != "ok":
    print("Search failed:", data)
else:
    print(f"Found {len(data['data'])} stations:\n")
    for station in data["data"]:
        name = station["station"]["name"]
        aqi = station.get("aqi")
        time_str = station["station"].get("time")
        uid = station.get("uid")
        print(f"UID: {uid} | AQI: {aqi} | Last updated: {time_str} | {name}")
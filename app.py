import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import holidays
import plotly.express as px
from datetime import date, timedelta
from zoneinfo import ZoneInfo

st.set_page_config(
    page_title="ComEd Electricity Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# Fixed coordinates for Chicago / ComEd service area
CHICAGO_LAT = 41.8781
CHICAGO_LON = -87.6298

# Load saved model pipeline
@st.cache_resource
def load_model():
    return joblib.load('energy_model.pkl')

artifact = load_model()
model = artifact['pipeline']
FEATURES = artifact['features']
RAIN_CAP = artifact['rain_cap']

# Fetch hourly weather forecast from Open-Meteo (free, no API key required)
@st.cache_data(ttl=3600)
def fetch_weather(start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": CHICAGO_LAT,
        "longitude": CHICAGO_LON,
        "hourly": "temperature_2m,apparent_temperature,rain",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": "America/Chicago"
    }
    res = requests.get(url, params=params, timeout=15)
    res.raise_for_status()
    data = res.json()["hourly"]
    return pd.DataFrame({
        "Datetime": pd.to_datetime(data["time"]),
        "temperature_c": data["temperature_2m"],
        "apparent_temp_c": data["apparent_temperature"],
        "rain_mm": data["rain"]
    })

# Engineer calendar, cyclical, holiday, and DST features
def build_features(weather_df):
    df = weather_df.copy()
    tz = ZoneInfo("America/Chicago")
    us_il_holidays = holidays.US(state='IL')

    # Clip rain to match training preprocessing
    df['rain_mm'] = df['rain_mm'].clip(upper=RAIN_CAP)

    # Extract time components
    hour = df['Datetime'].dt.hour
    month = df['Datetime'].dt.month
    dow = df['Datetime'].dt.dayofweek

    # Cyclical encodings

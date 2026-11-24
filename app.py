import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import holidays
import plotly.express as px
from datetime import date, timedelta
from zoneinfo import ZoneInfo

st.set_page_config(page_title="Electricity Consumption Predictor", page_icon="⚡", layout="wide")

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
def fetch_weather(lat, lon, start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
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
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['month_sin'] = np.sin(2 * np.pi * month / 12)
    df['month_cos'] = np.cos(2 * np.pi * month / 12)
    df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['dow_cos'] = np.cos(2 * np.pi * dow / 7)

    # Calendar flags
    df['is_weekend'] = (dow >= 5).astype(int)
    df['is_holiday'] = df['Datetime'].dt.date.apply(lambda d: 1 if d in us_il_holidays else 0)
    df['is_dst'] = df['Datetime'].apply(lambda dt: 1 if dt.replace(tzinfo=tz).dst().total_seconds() != 0 else 0)

    return df

# --- UI LAYOUT ---
st.title("⚡ Electricity Consumption Predictor")
st.markdown("Forecast hourly electricity demand (MW) by automatically fetching live weather, holiday, and calendar variables.")

with st.sidebar:
    st.header("Forecast Parameters")
    today = date.today()
    max_date = today + timedelta(days=15)  # Open-Meteo 16-day forecast window

    start_date = st.date_input("Start Date", value=today, min_value=today, max_value=max_date)
    end_date = st.date_input("End Date", value=min(today + timedelta(days=2), max_date), min_value=start_date, max_value=max_date)

    st.subheader("Location (Default: Chicago / ComEd)")
    lat = st.number_input("Latitude", value=41.8781, format="%.4f")
    lon = st.number_input("Longitude", value=-87.6298, format="%.4f")

    predict_btn = st.button("Predict Consumption", type="primary", use_container_width=True)

if predict_btn:
    with st.spinner("Fetching weather, holidays, and running model..."):
        try:
            weather_df = fetch_weather(lat, lon, start_date, end_date)
            feat_df = build_features(weather_df)
            feat_df['Predicted_MW'] = model.predict(feat_df[FEATURES])

            # KPI Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Peak Demand", f"{feat_df['Predicted_MW'].max():,.0f} MW")
            col2.metric("Average Demand", f"{feat_df['Predicted_MW'].mean():,.0f} MW")
            col3.metric("Minimum Demand", f"{feat_df['Predicted_MW'].min():,.0f} MW")
            col4.metric("Total Energy", f"{feat_df['Predicted_MW'].sum():,.0f} MWh")

            # Interactive Line Chart
            fig = px.line(
                feat_df,
                x='Datetime',
                y='Predicted_MW',
                title=f"Hourly Electricity Demand Forecast ({start_date} to {end_date})",
                labels={'Predicted_MW': 'Electricity Consumption (MW)', 'Datetime': 'Date & Time'}
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Data Table & Download
            with st.expander("View Hourly Forecast Data & Fetched Features"):
                display_cols = ['Datetime', 'Predicted_MW', 'temperature_c', 'apparent_temp_c', 'rain_mm', 'is_weekend', 'is_holiday', 'is_dst']
                st.dataframe(feat_df[display_cols], use_container_width=True)
                csv = feat_df[display_cols].to_csv(index=False).encode('utf-8')
                st.download_button("Download Forecast CSV", csv, "energy_forecast.csv", "text/csv")

        except Exception as e:
            st.error(f"Error generating forecast: {e}")
else:
    st.info("Select your desired date interval in the sidebar and click **Predict Consumption**.")

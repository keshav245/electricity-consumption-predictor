import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import holidays
import plotly.express as px
from datetime import date, timedelta
from zoneinfo import ZoneInfo

# Page Configuration
st.set_page_config(
    page_title="ComEd Electricity Consumption Predictor",
    page_icon="⚡",
    layout="wide"
)

# Fixed coordinates for Chicago / ComEd service area
CHICAGO_LAT = 41.8781
CHICAGO_LON = -87.6298
TIMEZONE = "America/Chicago"


# Load Saved Model Pipeline
@st.cache_resource
def load_model():
    return joblib.load("energy_model.pkl")


try:
    artifact = load_model()
    model = artifact["pipeline"]
    FEATURES = artifact["features"]
    RAIN_CAP = artifact["rain_cap"]
except FileNotFoundError:
    st.error("`energy_model.pkl` not found. Run `python train_model.py` first to generate the model.")
    st.stop()


# Fetch Live Hourly Weather Forecast (Open-Meteo API — Free, No API Key)
@st.cache_data(ttl=3600)
def fetch_weather(start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": CHICAGO_LAT,
        "longitude": CHICAGO_LON,
        "hourly": "temperature_2m,apparent_temperature,rain",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": TIMEZONE
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


# Engineer Calendar, Cyclical, Holiday, and DST Features
def build_features(weather_df):
    df = weather_df.copy()
    tz = ZoneInfo(TIMEZONE)
    us_il_holidays = holidays.US(state="IL")

    # Apply the same rain cap used during training
    df["rain_mm"] = df["rain_mm"].clip(upper=RAIN_CAP)

    # Extract time components
    hour = df["Datetime"].dt.hour
    month = df["Datetime"].dt.month
    dow = df["Datetime"].dt.dayofweek

    # Cyclical encodings
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Calendar flags
    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_holiday"] = df["Datetime"].dt.date.apply(
        lambda d: 1 if d in us_il_holidays else 0
    )
    df["is_dst"] = df["Datetime"].apply(
        lambda dt: 1 if dt.replace(tzinfo=tz).dst().total_seconds() != 0 else 0
    )

    return df


# --- UI LAYOUT ---
st.title("⚡ ComEd Electricity Consumption Predictor")
st.markdown(
    "Forecast hourly electricity demand (MW) for **Northern Illinois / Chicago (ComEd Grid)** "
    "using live weather forecasts, Illinois holiday calendars, and temporal features."
)

with st.sidebar:
    st.header("Forecast Interval")
    today = date.today()
    max_date = today + timedelta(days=15)  # Open-Meteo 16-day forecast window

    start_date = st.date_input(
        "Start Date",
        value=today,
        min_value=today,
        max_value=max_date
    )
    end_date = st.date_input(
        "End Date",
        value=min(today + timedelta(days=2), max_date),
        min_value=start_date,
        max_value=max_date
    )

    predict_btn = st.button("Predict Consumption", type="primary", use_container_width=True)
    st.caption("Region: Chicago / Northern Illinois (ComEd)")

if predict_btn:
    with st.spinner("Fetching weather, holidays, and generating forecast..."):
        try:
            weather_df = fetch_weather(start_date, end_date)
            feat_df = build_features(weather_df)
            feat_df["Predicted_MW"] = model.predict(feat_df[FEATURES])

            # KPI Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Peak Demand", f"{feat_df['Predicted_MW'].max():,.0f} MW")
            col2.metric("Average Demand", f"{feat_df['Predicted_MW'].mean():,.0f} MW")
            col3.metric("Minimum Demand", f"{feat_df['Predicted_MW'].min():,.0f} MW")
            col4.metric("Total Energy", f"{feat_df['Predicted_MW'].sum():,.0f} MWh")

            # Interactive Forecast Line Chart
            fig = px.line(
                feat_df,
                x="Datetime",
                y="Predicted_MW",
                title=f"ComEd Hourly Electricity Demand Forecast ({start_date} to {end_date})",
                labels={
                    "Predicted_MW": "Electricity Consumption (MW)",
                    "Datetime": "Date & Time (CST/CDT)"
                }
            )
            fig.update_layout(hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # Hourly Data Table & CSV Export
            with st.expander("View Hourly Forecast Data & Fetched Features"):
                display_cols = [
                    "Datetime", "Predicted_MW", "temperature_c",
                    "apparent_temp_c", "rain_mm", "is_weekend", "is_holiday", "is_dst"
                ]
                st.dataframe(feat_df[display_cols], use_container_width=True)

                csv = feat_df[display_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Forecast CSV",
                    data=csv,
                    file_name=f"comed_forecast_{start_date}_to_{end_date}.csv",
                    mime="text/csv"
                )

        except requests.exceptions.RequestException as e:
            st.error(f"Weather API error: {e}")
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
else:
    st.info("Select your desired date interval in the sidebar and click **Predict Consumption**.")

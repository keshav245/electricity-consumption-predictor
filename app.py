import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import holidays
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ------------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------------
st.set_page_config(
    page_title="ComEd Electricity Demand Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Fixed coordinates for Chicago / ComEd service area
CHICAGO_LAT = 41.8781
CHICAGO_LON = -87.6298
TIMEZONE = "America/Chicago"

# Model accuracy measured on the held-out test set
MODEL_MAE = 615.52   # Mean Absolute Error (MW)
MODEL_RMSE = 952.93  # Root Mean Squared Error (MW)

# ------------------------------------------------------------------
# Custom styling
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px;}

    .hero {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 45%, #2c5364 100%);
        padding: 2rem 2.2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
    }
    .hero h1 {color: #ffffff; margin: 0; font-size: 2.1rem; font-weight: 800;}
    .hero p  {color: #cfe3ee; margin: 0.5rem 0 0 0; font-size: 1.02rem;}
    .hero .badge {
        display: inline-block; margin-top: 0.9rem; margin-right: 0.4rem;
        padding: 0.25rem 0.75rem; border-radius: 999px;
        background: rgba(255,255,255,0.12); color: #ffd966;
        font-size: 0.8rem; font-weight: 600;
    }

    .kpi-card {
        border-radius: 16px; padding: 1.1rem 1.3rem;
        background: rgba(128,128,128,0.08);
        border: 1px solid rgba(128,128,128,0.2);
        border-left: 5px solid var(--accent);
        height: 100%;
    }
    .kpi-label {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; opacity: 0.7; font-weight: 600;}
    .kpi-value {font-size: 1.9rem; font-weight: 800; margin: 0.15rem 0;}
    .kpi-range {font-size: 0.82rem; opacity: 0.75;}

    .acc-card {
        border-radius: 16px; padding: 1.2rem 1.4rem;
        background: linear-gradient(135deg, rgba(255,193,7,0.10), rgba(255,87,34,0.08));
        border: 1px solid rgba(255,160,0,0.35);
    }
    .acc-card h4 {margin: 0 0 0.4rem 0;}
    .acc-big {font-size: 1.8rem; font-weight: 800;}
    .acc-sub {font-size: 0.85rem; opacity: 0.75;}

    .section-title {font-size: 1.25rem; font-weight: 700; margin: 1.6rem 0 0.6rem 0;}
    div[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,0.2);}
    </style>
    """,
    unsafe_allow_html=True,
)


def kpi_card(label, value, range_text, accent):
    return f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-range">{range_text}</div>
    </div>
    """


# ------------------------------------------------------------------
# Load saved model pipeline
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Live hourly weather forecast (Open-Meteo — free, no API key)
# ------------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_weather(start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": CHICAGO_LAT,
        "longitude": CHICAGO_LON,
        "hourly": "temperature_2m,apparent_temperature,rain",
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "timezone": TIMEZONE,
    }
    res = requests.get(url, params=params, timeout=15)
    res.raise_for_status()
    data = res.json()["hourly"]

    df = pd.DataFrame({
        "Datetime": pd.to_datetime(data["time"]),
        "temperature_c": data["temperature_2m"],
        "apparent_temp_c": data["apparent_temperature"],
        "rain_mm": data["rain"],
    })
    return df.dropna().reset_index(drop=True)


# ------------------------------------------------------------------
# Feature engineering (must match training)
# ------------------------------------------------------------------
def build_features(weather_df):
    df = weather_df.copy()
    tz = ZoneInfo(TIMEZONE)
    us_il_holidays = holidays.US(state="IL")

    df["rain_mm"] = df["rain_mm"].clip(upper=RAIN_CAP)

    hour = df["Datetime"].dt.hour
    month = df["Datetime"].dt.month
    dow = df["Datetime"].dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    df["is_weekend"] = (dow >= 5).astype(int)
    df["is_holiday"] = df["Datetime"].dt.date.apply(
        lambda d: 1 if d in us_il_holidays else 0
    )
    df["is_dst"] = df["Datetime"].apply(
        lambda dt: 1 if dt.replace(tzinfo=tz).dst().total_seconds() != 0 else 0
    )
    return df


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>⚡ ComEd Electricity Demand Predictor</h1>
        <p>Hourly demand forecast for Northern Illinois / Chicago, powered by live weather,
        Illinois holidays and time-of-day patterns.</p>
        <span class="badge">🌦 Live weather</span>
        <span class="badge">📅 IL holidays</span>
        <span class="badge">🤖 Machine learning</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Forecast Settings")
    today = datetime.now(ZoneInfo(TIMEZONE)).date()  # Chicago "today", not server time
    max_date = today + timedelta(days=15)  # Open-Meteo 16-day window

    start_date = st.date_input("Start date", value=today, min_value=today, max_value=max_date)
    end_date = st.date_input(
        "End date",
        value=min(today + timedelta(days=2), max_date),
        min_value=start_date,
        max_value=max_date,
    )

    band_choice = st.radio(
        "Uncertainty band on chart",
        ["Typical error (± MAE)", "Larger error (± RMSE)", "Hide"],
        index=0,
        help="Shows how far the real demand may be from the prediction.",
    )

    predict_btn = st.button("🔮 Predict Consumption", type="primary", use_container_width=True)

    st.divider()
    st.subheader("📏 Model Accuracy")
    st.metric("MAE", f"{MODEL_MAE:,.2f} MW")
    st.metric("RMSE", f"{MODEL_RMSE:,.2f} MW")
    st.caption("Measured on held-out test data. Region: Chicago / Northern Illinois (ComEd).")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if predict_btn:
    with st.spinner("Fetching weather, holidays, and generating forecast..."):
        try:
            weather_df = fetch_weather(start_date, end_date)
            if weather_df.empty:
                st.warning("No weather data returned for the selected dates.")
                st.stop()

            feat_df = build_features(weather_df)
            feat_df["Predicted_MW"] = model.predict(feat_df[FEATURES])
            feat_df["Date"] = feat_df["Datetime"].dt.date
            feat_df["Hour"] = feat_df["Datetime"].dt.hour

            # Error bands
            err = MODEL_MAE if band_choice.startswith("Typical") else MODEL_RMSE
            feat_df["Lower_MW"] = (feat_df["Predicted_MW"] - err).clip(lower=0)
            feat_df["Upper_MW"] = feat_df["Predicted_MW"] + err

            peak = feat_df["Predicted_MW"].max()
            avg = feat_df["Predicted_MW"].mean()
            low = feat_df["Predicted_MW"].min()
            total = feat_df["Predicted_MW"].sum()
            peak_time = feat_df.loc[feat_df["Predicted_MW"].idxmax(), "Datetime"]

            # --------------------------- KPI cards ---------------------------
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(
                kpi_card("Peak demand", f"{peak:,.0f} MW",
                         f"Likely {max(peak - MODEL_MAE, 0):,.0f} – {peak + MODEL_MAE:,.0f} MW<br>"
                         f"at {peak_time:%a %d %b, %H:%M}", "#ef4444"),
                unsafe_allow_html=True)
            c2.markdown(
                kpi_card("Average demand", f"{avg:,.0f} MW",
                         f"Likely {max(avg - MODEL_MAE, 0):,.0f} – {avg + MODEL_MAE:,.0f} MW", "#3b82f6"),
                unsafe_allow_html=True)
            c3.markdown(
                kpi_card("Minimum demand", f"{low:,.0f} MW",
                         f"Likely {max(low - MODEL_MAE, 0):,.0f} – {low + MODEL_MAE:,.0f} MW", "#10b981"),
                unsafe_allow_html=True)
            c4.markdown(
                kpi_card("Total energy", f"{total:,.0f} MWh",
                         f"Over {len(feat_df)} hours", "#f59e0b"),
                unsafe_allow_html=True)

            # --------------------------- Accuracy card ---------------------------
            mae_pct = MODEL_MAE / avg * 100
            rmse_pct = MODEL_RMSE / avg * 100
            st.markdown("<div class='section-title'>🎯 How accurate is this forecast?</div>",
                        unsafe_allow_html=True)
            a1, a2, a3 = st.columns([1, 1, 2])
            a1.markdown(
                f"""<div class="acc-card"><h4>MAE</h4>
                <div class="acc-big">± {MODEL_MAE:,.0f} MW</div>
                <div class="acc-sub">≈ {mae_pct:.1f}% of this forecast's average demand</div></div>""",
                unsafe_allow_html=True)
            a2.markdown(
                f"""<div class="acc-card"><h4>RMSE</h4>
                <div class="acc-big">± {MODEL_RMSE:,.0f} MW</div>
                <div class="acc-sub">≈ {rmse_pct:.1f}% of this forecast's average demand</div></div>""",
                unsafe_allow_html=True)
            with a3:
                st.info(
                    f"**What this means:** on average, a predicted hourly value differs from the "
                    f"real demand by about **{MODEL_MAE:,.0f} MW** (MAE). RMSE "
                    f"(**{MODEL_RMSE:,.0f} MW**) is larger because it penalises big misses more, "
                    f"so occasional errors can exceed the MAE — especially during extreme "
                    f"weather or unusual days. The shaded band on the chart shows this range."
                )

            # --------------------------- Tabs ---------------------------
            tab1, tab2, tab3, tab4 = st.tabs(
                ["📈 Forecast", "🗓 Daily view", "🌡 Weather vs demand", "📋 Data"]
            )

            with tab1:
                fig = go.Figure()
                if band_choice != "Hide":
                    label = "± MAE range" if band_choice.startswith("Typical") else "± RMSE range"
                    fig.add_trace(go.Scatter(
                        x=pd.concat([feat_df["Datetime"], feat_df["Datetime"][::-1]]),
                        y=pd.concat([feat_df["Upper_MW"], feat_df["Lower_MW"][::-1]]),
                        fill="toself", fillcolor="rgba(59,130,246,0.18)",
                        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", name=label,
                    ))
                fig.add_trace(go.Scatter(
                    x=feat_df["Datetime"], y=feat_df["Predicted_MW"],
                    mode="lines", name="Predicted demand",
                    line=dict(color="#3b82f6", width=3),
                    hovertemplate="%{y:,.0f} MW<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=[peak_time], y=[peak], mode="markers+text", name="Peak",
                    marker=dict(color="#ef4444", size=11),
                    text=[f"Peak {peak:,.0f} MW"], textposition="top center",
                ))
                fig.update_layout(
                    title=f"Hourly demand forecast ({start_date} to {end_date})",
                    xaxis_title="Date & time (Chicago local)",
                    yaxis_title="Electricity demand (MW)",
                    hovermode="x unified", height=480,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=1.1, x=0),
                    margin=dict(l=10, r=10, t=70, b=10),
                )
                fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
                fig.update_yaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)")
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                left, right = st.columns(2)
                daily = feat_df.groupby("Date").agg(
                    Peak_MW=("Predicted_MW", "max"),
                    Avg_MW=("Predicted_MW", "mean"),
                    Min_MW=("Predicted_MW", "min"),
                ).reset_index()

                bar = px.bar(
                    daily, x="Date", y="Peak_MW", text_auto=",.0f",
                    title="Daily peak demand (MW)",
                    color="Peak_MW", color_continuous_scale="YlOrRd",
                )
                bar.update_layout(
                    coloraxis_showscale=False, height=400,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                )
                left.plotly_chart(bar, use_container_width=True)

                pivot = feat_df.pivot_table(index="Date", columns="Hour",
                                            values="Predicted_MW", aggfunc="mean")
                heat = px.imshow(
                    pivot, aspect="auto", color_continuous_scale="Turbo",
                    labels=dict(x="Hour of day", y="Date", color="MW"),
                    title="Demand heatmap (hour × day)",
                )
                heat.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)")
                right.plotly_chart(heat, use_container_width=True)

                st.dataframe(
                    daily.rename(columns={"Peak_MW": "Peak (MW)", "Avg_MW": "Average (MW)",
                                          "Min_MW": "Minimum (MW)"}).round(0),
                    use_container_width=True, hide_index=True,
                )

            with tab3:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=feat_df["Datetime"], y=feat_df["Predicted_MW"],
                    name="Predicted demand (MW)", line=dict(color="#3b82f6", width=2.5)))
                fig2.add_trace(go.Scatter(
                    x=feat_df["Datetime"], y=feat_df["temperature_c"],
                    name="Temperature (°C)", yaxis="y2", line=dict(color="#f97316", width=2)))
                fig2.update_layout(
                    title="Predicted demand vs temperature", height=440,
                    yaxis=dict(title="Demand (MW)"),
                    yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
                    hovermode="x unified",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    legend=dict(orientation="h", y=1.1, x=0),
                )
                st.plotly_chart(fig2, use_container_width=True)

                sc = px.scatter(
                    feat_df, x="temperature_c", y="Predicted_MW",
                    color="Hour", color_continuous_scale="Viridis",
                    labels={"temperature_c": "Temperature (°C)", "Predicted_MW": "Predicted demand (MW)"},
                    title="How temperature drives demand",
                )
                sc.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(sc, use_container_width=True)

            with tab4:
                display_cols = [
                    "Datetime", "Predicted_MW", "Lower_MW", "Upper_MW",
                    "temperature_c", "apparent_temp_c", "rain_mm",
                    "is_weekend", "is_holiday", "is_dst",
                ]
                st.dataframe(feat_df[display_cols].round(1), use_container_width=True, hide_index=True)
                csv = feat_df[display_cols].to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download forecast CSV", data=csv,
                    file_name=f"comed_forecast_{start_date}_to_{end_date}.csv",
                    mime="text/csv",
                )

            st.caption(
                "Note: Lower/Upper columns are Predicted ± the chosen error (MAE or RMSE). "
                "They are an approximate guide based on past test error, not a guaranteed interval."
            )

        except requests.exceptions.RequestException as e:
            st.error(f"Weather API error: {e}")
        except Exception as e:
            st.error(f"Error generating forecast: {e}")
else:
    st.info("👈 Pick a date range in the sidebar and click **Predict Consumption**.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model MAE", f"{MODEL_MAE:,.2f} MW", help="Average absolute difference between predicted and actual demand")
    m2.metric("Model RMSE", f"{MODEL_RMSE:,.2f} MW", help="Like MAE but penalises large errors more")
    m3.metric("Forecast window", "Up to 16 days", help="Limited by the Open-Meteo weather forecast")

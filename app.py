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

# ----------------------------------------------------------------------------
# Page configuration
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="ComEd Electricity Consumption Predictor",
    page_icon="⚡",
    layout="wide",
)

CHICAGO_LAT = 41.8781
CHICAGO_LON = -87.6298
TIMEZONE = "America/Chicago"

# Z-scores for the confidence levels offered in the sidebar
Z_SCORES = {"80%": 1.282, "90%": 1.645, "95%": 1.960}

# ----------------------------------------------------------------------------
# Custom styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1300px;}

    .hero {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
        border-radius: 18px;
        padding: 28px 34px;
        margin-bottom: 22px;
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(30, 58, 138, 0.35);
    }
    .hero h1 {margin: 0; font-size: 2.1rem; font-weight: 800; color: #ffffff;}
    .hero p {margin: 8px 0 0 0; font-size: 1.02rem; color: #cbd5e1;}
    .hero .pill {
        display: inline-block; margin-top: 14px; margin-right: 8px;
        padding: 4px 12px; border-radius: 999px; font-size: 0.8rem;
        background: rgba(255,255,255,0.14); color: #e2e8f0;
    }

    .kpi {
        background: linear-gradient(160deg, rgba(37,99,235,0.10), rgba(37,99,235,0.02));
        border: 1px solid rgba(100,116,139,0.28);
        border-radius: 16px;
        padding: 18px 20px;
        height: 100%;
    }
    .kpi .label {font-size: 0.78rem; letter-spacing: 0.06em; text-transform: uppercase; opacity: 0.7;}
    .kpi .value {font-size: 1.9rem; font-weight: 800; margin: 4px 0 2px 0;}
    .kpi .unit {font-size: 0.95rem; font-weight: 600; opacity: 0.7; margin-left: 4px;}
    .kpi .range {
        font-size: 0.82rem; opacity: 0.85;
        background: rgba(37,99,235,0.12); border-radius: 8px;
        padding: 3px 8px; display: inline-block; margin-top: 4px;
    }
    .kpi .sub {font-size: 0.8rem; opacity: 0.65; margin-top: 6px;}

    .note {
        border-left: 4px solid #2563eb;
        background: rgba(37,99,235,0.08);
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.92rem;
        margin: 6px 0 18px 0;
    }
    div[data-testid="stSidebar"] {border-right: 1px solid rgba(100,116,139,0.25);}
    </style>
    """,
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------
# Load saved model
# ----------------------------------------------------------------------------
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


def get_model_error_pct():
    """
    Look for a saved error metric (MAPE) inside the model artifact.
    Returns (value_in_percent, source_text). Falls back to a typical value
    for hourly load forecasting if the training script did not save one.
    """
    candidates = [artifact.get("mape")]
    metrics = artifact.get("metrics")
    if isinstance(metrics, dict):
        candidates += [metrics.get("mape"), metrics.get("MAPE")]
    for val in candidates:
        if val is not None:
            try:
                val = float(val)
                if val <= 1:  # stored as a fraction
                    val *= 100
                return val, "from your trained model"
            except (TypeError, ValueError):
                pass
    return 4.0, "default estimate (model has no saved error metric)"


# ----------------------------------------------------------------------------
# Weather + feature engineering
# ----------------------------------------------------------------------------
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

    return pd.DataFrame(
        {
            "Datetime": pd.to_datetime(data["time"]),
            "temperature_c": data["temperature_2m"],
            "apparent_temp_c": data["apparent_temperature"],
            "rain_mm": data["rain"],
        }
    )


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
    df["is_holiday"] = df["Datetime"].dt.date.apply(lambda d: 1 if d in us_il_holidays else 0)
    df["is_dst"] = df["Datetime"].apply(
        lambda dt: 1 if dt.replace(tzinfo=tz).dst().total_seconds() != 0 else 0
    )
    return df


def add_uncertainty(df, today, base_mape, per_day_growth, z):
    """
    Adds lower / upper bounds around each prediction.

    - Model error: MAPE converted to a 1-sigma relative error (sigma ~ 1.25 x MAPE).
    - Forecast error: weather forecasts get less reliable the further ahead we look,
      so extra uncertainty grows with each forecast day.
    - Both are combined in quadrature, then scaled by the confidence level (z).
    """
    out = df.copy()
    days_ahead = (out["Datetime"].dt.normalize() - pd.Timestamp(today)).dt.days.clip(lower=0)
    model_sigma = base_mape * 1.25
    weather_sigma = per_day_growth * days_ahead
    sigma_pct = np.sqrt(model_sigma**2 + weather_sigma**2)

    out["error_pct"] = z * sigma_pct
    out["Lower_MW"] = out["Predicted_MW"] * (1 - out["error_pct"] / 100)
    out["Upper_MW"] = out["Predicted_MW"] * (1 + out["error_pct"] / 100)
    out["error_mw"] = out["Predicted_MW"] * out["error_pct"] / 100
    return out


# ----------------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------------
def kpi_card(label, value, unit, low, high, sub=""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="label">{label}</div>
            <div class="value">{value}<span class="unit">{unit}</span></div>
            <div class="range">Likely range: {low} – {high} {unit}</div>
            <div class="sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>⚡ ComEd Electricity Consumption Predictor</h1>
        <p>Hourly electricity demand forecast for Northern Illinois / Chicago,
        powered by live weather, Illinois holidays and calendar patterns.</p>
        <span class="pill">🌦 Live weather forecast</span>
        <span class="pill">📅 Holiday aware</span>
        <span class="pill">📊 Prediction uncertainty range</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Forecast Settings")
    today = datetime.now(ZoneInfo(TIMEZONE)).date()  # Chicago "today", not server time
    max_date = today + timedelta(days=15)

    start_date = st.date_input("Start date", value=today, min_value=today, max_value=max_date)
    end_date = st.date_input(
        "End date",
        value=min(today + timedelta(days=2), max_date),
        min_value=start_date,
        max_value=max_date,
    )

    st.markdown("---")
    st.subheader("🎯 Prediction Accuracy")
    conf_label = st.select_slider(
        "Confidence level",
        options=list(Z_SCORES.keys()),
        value="90%",
        help="Higher confidence = a wider range. A 90% range means the real demand "
        "should fall inside the band about 9 times out of 10.",
    )

    model_mape, mape_source = get_model_error_pct()
    with st.expander("Advanced error settings"):
        base_mape = st.number_input(
            "Model error – MAPE (%)",
            min_value=0.5,
            max_value=25.0,
            value=float(round(model_mape, 2)),
            step=0.5,
            help=f"Average % error of the model on test data ({mape_source}).",
        )
        per_day_growth = st.number_input(
            "Extra error per forecast day (%)",
            min_value=0.0,
            max_value=3.0,
            value=0.3,
            step=0.1,
            help="Weather forecasts get less accurate further ahead, so the "
            "range widens each day.",
        )

    predict_btn = st.button("🔮 Predict Consumption", type="primary", use_container_width=True)
    st.caption("Region: Chicago / Northern Illinois (ComEd)")

# ----------------------------------------------------------------------------
# Main content
# ----------------------------------------------------------------------------
if not predict_btn:
    st.info("👈 Choose your date range in the sidebar and click **Predict Consumption**.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("#### 🌡️ Weather\nTemperature, feels-like temperature and rainfall drive heating and cooling demand.")
    c2.markdown("#### 📅 Calendar\nWeekends, Illinois holidays, season and daylight saving time change usage patterns.")
    c3.markdown("#### 🎯 Uncertainty\nEvery forecast comes with a likely range, so you know how far off it could be.")
    st.stop()

with st.spinner("Fetching weather, holidays, and generating forecast..."):
    try:
        weather_df = fetch_weather(start_date, end_date)
        feat_df = build_features(weather_df)
        feat_df["Predicted_MW"] = model.predict(feat_df[FEATURES])
        z = Z_SCORES[conf_label]
        feat_df = add_uncertainty(feat_df, today, base_mape, per_day_growth, z)
    except requests.exceptions.RequestException as e:
        st.error(f"Weather API error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error generating forecast: {e}")
        st.stop()

pred = feat_df["Predicted_MW"]
peak_idx = pred.idxmax()
min_idx = pred.idxmin()
avg_err_pct = feat_df["error_pct"].mean()

# --- KPI cards ---------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
with k1:
    kpi_card(
        "Peak Demand",
        f"{pred.max():,.0f}",
        "MW",
        f"{feat_df.loc[peak_idx, 'Lower_MW']:,.0f}",
        f"{feat_df.loc[peak_idx, 'Upper_MW']:,.0f}",
        f"At {feat_df.loc[peak_idx, 'Datetime']:%a %d %b, %I %p}",
    )
with k2:
    kpi_card(
        "Average Demand",
        f"{pred.mean():,.0f}",
        "MW",
        f"{feat_df['Lower_MW'].mean():,.0f}",
        f"{feat_df['Upper_MW'].mean():,.0f}",
        f"Over {len(feat_df)} hours",
    )
with k3:
    kpi_card(
        "Minimum Demand",
        f"{pred.min():,.0f}",
        "MW",
        f"{feat_df.loc[min_idx, 'Lower_MW']:,.0f}",
        f"{feat_df.loc[min_idx, 'Upper_MW']:,.0f}",
        f"At {feat_df.loc[min_idx, 'Datetime']:%a %d %b, %I %p}",
    )
with k4:
    kpi_card(
        "Total Energy",
        f"{pred.sum():,.0f}",
        "MWh",
        f"{feat_df['Lower_MW'].sum():,.0f}",
        f"{feat_df['Upper_MW'].sum():,.0f}",
        f"{start_date:%d %b} → {end_date:%d %b}",
    )

st.markdown(
    f"""
    <div class="note">
    🎯 <b>How far can the prediction be off?</b> At <b>{conf_label}</b> confidence, the real demand is
    expected to stay within about <b>±{avg_err_pct:.1f}%</b> of the predicted value on average
    (roughly <b>±{feat_df['error_mw'].mean():,.0f} MW</b> per hour). The range is narrower for
    today and tomorrow and gets wider for later days because weather forecasts become less reliable.
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Tabs --------------------------------------------------------------------
tab_forecast, tab_daily, tab_weather, tab_data = st.tabs(
    ["📈 Forecast", "🗓️ Daily Summary", "🌦 Weather & Patterns", "📋 Data & Export"]
)

# Forecast chart with confidence band
with tab_forecast:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=feat_df["Datetime"], y=feat_df["Upper_MW"],
            mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=feat_df["Datetime"], y=feat_df["Lower_MW"],
            mode="lines", line=dict(width=0), fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.18)", name=f"{conf_label} range",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=feat_df["Datetime"], y=feat_df["Predicted_MW"],
            mode="lines", name="Predicted demand",
            line=dict(color="#2563eb", width=3),
            customdata=np.stack([feat_df["Lower_MW"], feat_df["Upper_MW"]], axis=-1),
            hovertemplate=(
                "<b>%{y:,.0f} MW</b><br>"
                "Range: %{customdata[0]:,.0f} – %{customdata[1]:,.0f} MW<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[feat_df.loc[peak_idx, "Datetime"]], y=[pred.max()],
            mode="markers+text", name="Peak",
            marker=dict(color="#ef4444", size=11, symbol="diamond"),
            text=[f"Peak {pred.max():,.0f} MW"], textposition="top center",
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        title=f"Hourly Demand Forecast ({start_date:%d %b %Y} to {end_date:%d %b %Y})",
        xaxis_title=f"Date & time ({TIMEZONE})",
        yaxis_title="Electricity demand (MW)",
        hovermode="x unified",
        height=520,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

# Daily summary
with tab_daily:
    daily = (
        feat_df.assign(Date=feat_df["Datetime"].dt.date)
        .groupby("Date")
        .agg(
            Peak_MW=("Predicted_MW", "max"),
            Avg_MW=("Predicted_MW", "mean"),
            Min_MW=("Predicted_MW", "min"),
            Energy_MWh=("Predicted_MW", "sum"),
            Energy_Low=("Lower_MW", "sum"),
            Energy_High=("Upper_MW", "sum"),
            Error_pct=("error_pct", "mean"),
        )
        .reset_index()
    )

    figd = go.Figure()
    figd.add_trace(
        go.Bar(
            x=daily["Date"].astype(str), y=daily["Energy_MWh"], name="Predicted energy",
            marker_color="#2563eb",
            error_y=dict(
                type="data", symmetric=False,
                array=daily["Energy_High"] - daily["Energy_MWh"],
                arrayminus=daily["Energy_MWh"] - daily["Energy_Low"],
                color="#64748b",
            ),
            hovertemplate="%{x}<br><b>%{y:,.0f} MWh</b><extra></extra>",
        )
    )
    figd.update_layout(
        title="Daily Total Energy (error bars show the likely range)",
        yaxis_title="Energy (MWh)", height=420,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    st.plotly_chart(figd, use_container_width=True)

    daily_view = daily.rename(
        columns={
            "Peak_MW": "Peak (MW)", "Avg_MW": "Average (MW)", "Min_MW": "Minimum (MW)",
            "Energy_MWh": "Energy (MWh)", "Energy_Low": "Energy low (MWh)",
            "Energy_High": "Energy high (MWh)", "Error_pct": "± Error (%)",
        }
    )
    st.dataframe(
        daily_view.style.format(
            {
                "Peak (MW)": "{:,.0f}", "Average (MW)": "{:,.0f}", "Minimum (MW)": "{:,.0f}",
                "Energy (MWh)": "{:,.0f}", "Energy low (MWh)": "{:,.0f}",
                "Energy high (MWh)": "{:,.0f}", "± Error (%)": "{:.1f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

# Weather & patterns
with tab_weather:
    left, right = st.columns(2)

    with left:
        figt = go.Figure()
        figt.add_trace(go.Scatter(x=feat_df["Datetime"], y=feat_df["temperature_c"],
                                  name="Temperature", line=dict(color="#f97316", width=2)))
        figt.add_trace(go.Scatter(x=feat_df["Datetime"], y=feat_df["apparent_temp_c"],
                                  name="Feels like", line=dict(color="#a855f7", width=2, dash="dot")))
        figt.update_layout(title="Temperature (°C)", height=360, hovermode="x unified",
                           margin=dict(l=10, r=10, t=50, b=10),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
        st.plotly_chart(figt, use_container_width=True)

    with right:
        figr = px.bar(feat_df, x="Datetime", y="rain_mm", title="Rainfall (mm)")
        figr.update_traces(marker_color="#0ea5e9")
        figr.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(figr, use_container_width=True)

    # Heatmap: day vs hour
    heat = feat_df.assign(Date=feat_df["Datetime"].dt.strftime("%a %d %b"),
                          Hour=feat_df["Datetime"].dt.hour)
    pivot = heat.pivot_table(index="Date", columns="Hour", values="Predicted_MW", sort=False)
    figh = px.imshow(
        pivot, aspect="auto", color_continuous_scale="YlOrRd",
        labels=dict(x="Hour of day", y="Day", color="MW"),
        title="Demand Heatmap (when is demand highest?)",
    )
    figh.update_layout(height=max(280, 60 * len(pivot) + 120), margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(figh, use_container_width=True)

# Data & export
with tab_data:
    display_cols = [
        "Datetime", "Predicted_MW", "Lower_MW", "Upper_MW", "error_pct",
        "temperature_c", "apparent_temp_c", "rain_mm",
        "is_weekend", "is_holiday", "is_dst",
    ]
    export_df = feat_df[display_cols].rename(columns={"error_pct": "error_pct_plus_minus"})
    st.dataframe(
        export_df.style.format(
            {"Predicted_MW": "{:,.0f}", "Lower_MW": "{:,.0f}", "Upper_MW": "{:,.0f}",
             "error_pct_plus_minus": "{:.1f}"}
        ),
        use_container_width=True,
        hide_index=True,
        height=420,
    )
    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"comed_forecast_{start_date}_to_{end_date}.csv",
        mime="text/csv",
    )

st.caption(
    f"Uncertainty range uses a model error of {base_mape:.1f}% MAPE ({mape_source}) "
    f"plus {per_day_growth:.1f}% extra error per forecast day. It is an estimate, not a guarantee."
)

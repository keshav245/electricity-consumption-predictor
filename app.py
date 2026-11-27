import random
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
# Page configuration (starts collapsed; toggle open/close anytime)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="ComEd Electricity Demand Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",  # Change to "expanded" if you want it open on first load
)

# Fixed coordinates for Chicago / ComEd service area
CHICAGO_LAT = 41.8781
CHICAGO_LON = -87.6298
TIMEZONE = "America/Chicago"

# Model accuracy measured on the held-out test set
MODEL_MAE = 615.52   # Mean Absolute Error (MW)
MODEL_RMSE = 952.93  # Root Mean Squared Error (MW)

GREEN = "#15803d"
GREEN_DARK = "#0f2e1f"
GREEN_LIGHT = "#a7d7b8"

# ------------------------------------------------------------------
# Soft muted sage-gray + green theme with animated background
# ------------------------------------------------------------------
st.markdown(
    f"""
    <style>
    /* ---------- Base: softer, low-glare sage-gray / green ---------- */
    .stApp {{
        background: linear-gradient(160deg, #dde7e0 0%, #d2dfd6 48%, #c5d6cb 100%);
        color: #0f2419;
    }}
    [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{
        background: transparent;
    }}
    h1, h2, h3, h4, h5, label, p, span, div {{color: inherit;}}
    .block-container {{
        padding-top: 2rem; padding-bottom: 3rem; max-width: 1300px;
        position: relative; z-index: 2; animation: fadeUp .7s ease both;
    }}

    /* ---------- Collapsible Sidebar & Always-Visible Toggle Buttons ---------- */
    [data-testid="stSidebar"] {{
        background: rgba(210, 223, 214, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(21, 128, 61, 0.28);
        z-index: 100;
    }}
    [data-testid="collapsedControl"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarCollapseButton"] {{
        z-index: 999 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }}
    [data-testid="collapsedControl"] button,
    [data-testid="stExpandSidebarButton"] button,
    [data-testid="stSidebarCollapseButton"] button {{
        background-color: #15803d !important;
        color: #ffffff !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        box-shadow: 0 3px 10px rgba(15, 36, 25, 0.22) !important;
    }}
    [data-testid="collapsedControl"] svg,
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stSidebarCollapseButton"] svg {{
        fill: #ffffff !important;
        color: #ffffff !important;
    }}

    /* ---------- Animated background layer (toned down) ---------- */
    .bg-layer {{position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none;}}
    .bubble {{
        position: absolute; bottom: -120px; border-radius: 50%;
        background: radial-gradient(circle at 30% 30%, rgba(235,242,237,0.65), rgba(22,163,74,0.22));
        border: 1px solid rgba(21,128,61,0.20);
        animation: floatUp linear infinite;
    }}
    .bolt {{
        position: absolute; bottom: -60px; font-size: 26px; opacity: .16;
        animation: floatUp linear infinite, sway 5s ease-in-out infinite;
    }}
    .glow {{
        position: absolute; width: 520px; height: 520px; border-radius: 50%;
        filter: blur(95px); opacity: .30;
        background: radial-gradient(circle, rgba(34,197,94,.42), rgba(210,223,214,0) 70%);
        animation: drift 18s ease-in-out infinite alternate;
    }}
    .glow.g1 {{top: -120px; left: -100px;}}
    .glow.g2 {{bottom: -160px; right: -120px; animation-delay: -9s;}}
    @keyframes floatUp {{
        0%   {{transform: translateY(0) scale(.8); opacity: 0;}}
        10%  {{opacity: .8;}}
        100% {{transform: translateY(-115vh) scale(1.15); opacity: 0;}}
    }}
    @keyframes sway {{0%,100% {{margin-left: -14px;}} 50% {{margin-left: 14px;}}}}
    @keyframes drift {{
        from {{transform: translate(0,0) scale(1);}}
        to   {{transform: translate(90px,70px) scale(1.2);}}
    }}
    @keyframes fadeUp {{from {{opacity: 0; transform: translateY(14px);}} to {{opacity: 1; transform: none;}}}}
    @keyframes shine {{0% {{background-position: 0% 50%;}} 100% {{background-position: 200% 50%;}}}}
    @keyframes pulse {{0%,100% {{transform: scale(1);}} 50% {{transform: scale(1.12);}}}}

    /* ---------- Hero (richer, less glaring green) ---------- */
    .hero {{
        background: linear-gradient(120deg, #14532d, #15803d, #16a34a, #14532d);
        background-size: 300% 300%;
        animation: shine 9s linear infinite;
        padding: 1.9rem 2.2rem; border-radius: 20px; margin-bottom: 1.5rem;
        box-shadow: 0 12px 30px rgba(20, 83, 45, 0.28);
    }}
    .hero h1 {{color: #f3f7f4 !important; margin: 0; font-size: 2.15rem; font-weight: 800;}}
    .hero h1 .bolt-icon {{display: inline-block; animation: pulse 1.6s ease-in-out infinite;}}
    .hero p {{color: #dcfce7 !important; margin: .5rem 0 0 0; font-size: 1.02rem;}}
    .hero .badge {{
        display: inline-block; margin: .9rem .4rem 0 0; padding: .25rem .8rem; border-radius: 999px;
        background: rgba(255,255,255,0.18); color: #f3f7f4 !important; font-size: .8rem; font-weight: 600;
        transition: transform .2s ease, background .2s ease;
    }}
    .hero .badge:hover {{transform: translateY(-3px); background: rgba(255,255,255,0.30);}}

    /* ---------- Cards (soft off-white sage, easy on eyes) ---------- */
    .kpi-card {{
        border-radius: 16px; padding: 1.1rem 1.3rem; height: 100%;
        background: rgba(235, 242, 237, 0.92);
        border: 1px solid rgba(21, 128, 61, 0.22);
        border-left: 6px solid var(--accent);
        box-shadow: 0 4px 14px rgba(15, 36, 25, 0.08);
        transition: transform .25s ease, box-shadow .25s ease;
        animation: fadeUp .6s ease both;
    }}
    .kpi-card:hover {{transform: translateY(-6px); box-shadow: 0 14px 28px rgba(21,128,61,0.20);}}
    .kpi-label {{font-size: .78rem; text-transform: uppercase; letter-spacing: .07em; color: #3d5a49; font-weight: 700;}}
    .kpi-value {{font-size: 1.9rem; font-weight: 800; margin: .15rem 0; color: {GREEN_DARK};}}
    .kpi-range {{font-size: .82rem; color: #3d5a49;}}

    .acc-card {{
        border-radius: 16px; padding: 1.2rem 1.4rem;
        background: linear-gradient(135deg, #d7e4db, #c9dcd0);
        border: 1px solid rgba(21, 128, 61, 0.32);
        transition: transform .25s ease, box-shadow .25s ease;
    }}
    .acc-card:hover {{transform: translateY(-4px); box-shadow: 0 12px 24px rgba(21,128,61,0.18);}}
    .acc-card h4 {{margin: 0 0 .3rem 0; color: {GREEN_DARK};}}
    .acc-big {{font-size: 1.8rem; font-weight: 800; color: {GREEN};}}
    .acc-sub {{font-size: .85rem; color: #3d5a49;}}

    .section-title {{font-size: 1.25rem; font-weight: 700; margin: 1.6rem 0 .6rem 0; color: {GREEN_DARK};}}

    /* ---------- Widgets ---------- */
    .stButton > button[kind="primary"] {{
        background: linear-gradient(120deg, #14532d, #16a34a); border: none; color: #f3f7f4;
        font-weight: 700; border-radius: 12px; transition: transform .2s ease, box-shadow .2s ease;
    }}
    .stButton > button[kind="primary"]:hover {{
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 8px 20px rgba(20, 83, 45, .35);
    }}
    .stTabs [data-baseweb="tab"] {{font-weight: 600;}}
    .stTabs [aria-selected="true"] {{color: {GREEN} !important;}}
    .stTabs [data-baseweb="tab-highlight"] {{background-color: {GREEN};}}
    [data-testid="stMetric"] {{
        background: rgba(235, 242, 237, 0.90);
        border: 1px solid rgba(21, 128, 61, 0.22);
        border-radius: 14px; padding: .6rem .9rem; transition: transform .2s ease;
    }}
    [data-testid="stMetric"]:hover {{transform: translateY(-3px);}}
    </style>
    """,
    unsafe_allow_html=True,
)


def background_layer():
    rnd = random.Random(7)
    items = ['<div class="glow g1"></div><div class="glow g2"></div>']
    for _ in range(16):
        size = rnd.randint(18, 70)
        items.append(
            f'<div class="bubble" style="left:{rnd.randint(0, 98)}%;width:{size}px;height:{size}px;'
            f'animation-duration:{rnd.randint(14, 30)}s;animation-delay:-{rnd.randint(0, 25)}s"></div>'
        )
    for _ in range(7):
        items.append(
            f'<div class="bolt" style="left:{rnd.randint(2, 96)}%;'
            f'animation-duration:{rnd.randint(16, 28)}s,{rnd.randint(4, 7)}s;'
            f'animation-delay:-{rnd.randint(0, 20)}s,0s">⚡</div>'
        )
    return f'<div class="bg-layer">{"".join(items)}</div>'


st.markdown(background_layer(), unsafe_allow_html=True)


def kpi_card(label, value, range_text, accent):
    return f"""
    <div class="kpi-card" style="--accent:{accent}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-range">{range_text}</div>
    </div>
    """


def style_fig(fig, height=450):
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(235, 242, 237, 0.82)",
        font=dict(color="#0f2419"),
        margin=dict(l=10, r=10, t=70, b=10),
    )
    fig.update_xaxes(gridcolor="rgba(21,128,61,0.14)")
    fig.update_yaxes(gridcolor="rgba(21,128,61,0.14)")
    return fig


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
    df["is_holiday"] = df["Datetime"].dt.date.apply(lambda d: 1 if d in us_il_holidays else 0)
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
        <h1><span class="bolt-icon">⚡</span> ComEd Electricity Demand Predictor</h1>
        <p>Hourly demand forecast for Northern Illinois / Chicago, powered by live weather,
        Illinois holidays and time-of-day patterns.</p>
        <span class="badge">🌦 Live weather</span>
        <span class="badge">📅 IL holidays</span>
        <span class="badge">🤖 Machine learning</span>
        <span class="badge">🎛 What-if simulator</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Sidebar (Collapsible)
# ------------------------------------------------------------------
if "has_run" not in st.session_state:
    st.session_state.has_run = False

with st.sidebar:
    st.header("⚙️ Forecast Settings")
    today = datetime.now(ZoneInfo(TIMEZONE)).date()  # Chicago "today"
    max_date = today + timedelta(days=15)  # Open-Meteo 16-day window

    quick = st.radio(
        "Quick range",
        ["Today & tomorrow", "Next 3 days", "Next 7 days", "Custom"],
        index=1,
    )
    if quick == "Custom":
        start_date = st.date_input("Start date", value=today, min_value=today, max_value=max_date)
        end_date = st.date_input(
            "End date", value=min(today + timedelta(days=2), max_date),
            min_value=start_date, max_value=max_date,
        )
    else:
        days = {"Today & tomorrow": 1, "Next 3 days": 2, "Next 7 days": 6}[quick]
        start_date = today
        end_date = min(today + timedelta(days=days), max_date)
        st.caption(f"📅 {start_date} → {end_date}")

    st.markdown("---")
    st.subheader("🎛 What-if simulator")
    temp_shift = st.slider(
        "Temperature change (°C)", -10.0, 10.0, 0.0, 0.5,
        help="Imagine it is hotter or colder than forecast and see how demand responds.",
    )

    band_choice = st.radio(
        "Uncertainty band on chart",
        ["Typical error (± MAE)", "Larger error (± RMSE)", "Hide"],
        index=0,
        help="Shows how far the real demand may be from the prediction.",
    )

    if st.button("🔮 Predict Consumption", type="primary", use_container_width=True):
        st.session_state.has_run = True

    st.divider()
    st.subheader("📏 Model Accuracy")
    st.metric("MAE", f"{MODEL_MAE:,.2f} MW")
    st.metric("RMSE", f"{MODEL_RMSE:,.2f} MW")
    st.caption("Measured on held-out test data. Region: Chicago / Northern Illinois (ComEd).")

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if st.session_state.has_run:
    with st.spinner("Fetching weather, holidays, and generating forecast..."):
        try:
            weather_df = fetch_weather(start_date, end_date)
            if weather_df.empty:
                st.warning("No weather data returned for the selected dates.")
                st.stop()

            base_df = build_features(weather_df)
            base_df["Predicted_MW"] = model.predict(base_df[FEATURES])

            # What-if: shift temperatures and re-predict
            feat_df = base_df.copy()
            if temp_shift != 0:
                shifted = weather_df.copy()
                shifted["temperature_c"] += temp_shift
                shifted["apparent_temp_c"] += temp_shift
                feat_df = build_features(shifted)
                feat_df["Predicted_MW"] = model.predict(feat_df[FEATURES])

            feat_df["Date"] = feat_df["Datetime"].dt.date
            feat_df["Hour"] = feat_df["Datetime"].dt.hour

            err = MODEL_MAE if band_choice.startswith("Typical") else MODEL_RMSE
            feat_df["Lower_MW"] = (feat_df["Predicted_MW"] - err).clip(lower=0)
            feat_df["Upper_MW"] = feat_df["Predicted_MW"] + err

            peak = feat_df["Predicted_MW"].max()
            avg = feat_df["Predicted_MW"].mean()
            low = feat_df["Predicted_MW"].min()
            total = feat_df["Predicted_MW"].sum()
            peak_time = feat_df.loc[feat_df["Predicted_MW"].idxmax(), "Datetime"]

            if temp_shift != 0:
                diff = avg - base_df["Predicted_MW"].mean()
                st.success(
                    f"🎛 What-if active: temperature {temp_shift:+.1f} °C changes average demand by "
                    f"**{diff:+,.0f} MW** compared with the normal forecast."
                )

            # --------------------------- KPI cards ---------------------------
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(
                kpi_card("Peak demand", f"{peak:,.0f} MW",
                         f"Likely {max(peak - MODEL_MAE, 0):,.0f} – {peak + MODEL_MAE:,.0f} MW<br>"
                         f"at {peak_time:%a %d %b, %H:%M}", "#ea580c"),
                unsafe_allow_html=True)
            c2.markdown(
                kpi_card("Average demand", f"{avg:,.0f} MW",
                         f"Likely {max(avg - MODEL_MAE, 0):,.0f} – {avg + MODEL_MAE:,.0f} MW", GREEN),
                unsafe_allow_html=True)
            c3.markdown(
                kpi_card("Minimum demand", f"{low:,.0f} MW",
                         f"Likely {max(low - MODEL_MAE, 0):,.0f} – {low + MODEL_MAE:,.0f} MW", "#0284c7"),
                unsafe_allow_html=True)
            c4.markdown(
                kpi_card("Total energy", f"{total:,.0f} MWh",
                         f"Over {len(feat_df)} hours", "#ca8a04"),
                unsafe_allow_html=True)

            # --------------------------- Accuracy ---------------------------
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
                    f"so occasional errors can exceed the MAE, especially in extreme weather or on "
                    f"unusual days. The shaded band on the chart shows this range."
                )

            # --------------------------- Tabs ---------------------------
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📈 Forecast", "🔍 Hour inspector", "🗓 Daily view", "🌡 Weather vs demand", "📋 Data"]
            )

            with tab1:
                fig = go.Figure()
                if band_choice != "Hide":
                    label = "± MAE range" if band_choice.startswith("Typical") else "± RMSE range"
                    fig.add_trace(go.Scatter(
                        x=pd.concat([feat_df["Datetime"], feat_df["Datetime"][::-1]]),
                        y=pd.concat([feat_df["Upper_MW"], feat_df["Lower_MW"][::-1]]),
                        fill="toself", fillcolor="rgba(21,128,61,0.18)",
                        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", name=label,
                    ))
                if temp_shift != 0:
                    fig.add_trace(go.Scatter(
                        x=base_df["Datetime"], y=base_df["Predicted_MW"], mode="lines",
                        name="Normal forecast", line=dict(color="#64748b", width=2, dash="dot"),
                        hovertemplate="%{y:,.0f} MW<extra></extra>",
                    ))
                fig.add_trace(go.Scatter(
                    x=feat_df["Datetime"], y=feat_df["Predicted_MW"], mode="lines",
                    name="Predicted demand", line=dict(color=GREEN, width=3.5),
                    hovertemplate="%{y:,.0f} MW<extra></extra>",
                ))
                fig.add_trace(go.Scatter(
                    x=[peak_time], y=[peak], mode="markers+text", name="Peak",
                    marker=dict(color="#ea580c", size=12, line=dict(color="white", width=2)),
                    text=[f"Peak {peak:,.0f} MW"], textposition="top center",
                ))
                fig.update_layout(
                    title=f"Hourly demand forecast ({start_date} to {end_date})",
                    xaxis_title="Date & time (Chicago local)", yaxis_title="Electricity demand (MW)",
                    hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0),
                )
                fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
                st.plotly_chart(style_fig(fig, 520), use_container_width=True)
                st.caption("💡 Drag the slider under the chart to zoom into any time window.")

            with tab2:
                options = list(feat_df.index)
                default_idx = int(feat_df["Predicted_MW"].idxmax())
                pick = st.select_slider(
                    "Choose an hour to inspect",
                    options=options, value=default_idx,
                    format_func=lambda i: f"{feat_df.loc[i, 'Datetime']:%a %d %b, %H:%M}",
                )
                row = feat_df.loc[pick]
                g_max = float(feat_df["Upper_MW"].max() * 1.15)
                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=float(row["Predicted_MW"]),
                    number=dict(suffix=" MW", valueformat=",.0f"),
                    title=dict(text=f"{row['Datetime']:%A, %d %b %H:%M}"),
                    gauge=dict(
                        axis=dict(range=[0, g_max]),
                        bar=dict(color=GREEN),
                        steps=[dict(range=[row["Lower_MW"], row["Upper_MW"]], color=GREEN_LIGHT)],
                        threshold=dict(line=dict(color="#ea580c", width=4), value=float(avg)),
                    ),
                ))
                gl, gr = st.columns([1.2, 1])
                gl.plotly_chart(style_fig(gauge, 360), use_container_width=True)
                with gr:
                    st.markdown(f"**Expected range:** {row['Lower_MW']:,.0f} – {row['Upper_MW']:,.0f} MW")
                    st.markdown(f"**Temperature:** {row['temperature_c']:.1f} °C "
                                f"(feels like {row['apparent_temp_c']:.1f} °C)")
                    st.markdown(f"**Rain:** {row['rain_mm']:.1f} mm")
                    st.markdown(f"**Weekend:** {'Yes' if row['is_weekend'] else 'No'}  \n"
                                f"**Holiday:** {'Yes' if row['is_holiday'] else 'No'}")
                    st.caption("Light-green zone = likely range · orange line = period average.")

            with tab3:
                left, right = st.columns(2)
                daily = feat_df.groupby("Date").agg(
                    Peak_MW=("Predicted_MW", "max"),
                    Avg_MW=("Predicted_MW", "mean"),
                    Min_MW=("Predicted_MW", "min"),
                ).reset_index()

                bar = px.bar(daily, x="Date", y="Peak_MW", text_auto=",.0f",
                             title="Daily peak demand (MW)",
                             color="Peak_MW", color_continuous_scale="Greens")
                bar.update_layout(coloraxis_showscale=False)
                left.plotly_chart(style_fig(bar, 400), use_container_width=True)

                pivot = feat_df.pivot_table(index="Date", columns="Hour",
                                            values="Predicted_MW", aggfunc="mean")
                heat = px.imshow(pivot, aspect="auto", color_continuous_scale="Greens",
                                 labels=dict(x="Hour of day", y="Date", color="MW"),
                                 title="Demand heatmap (hour × day)")
                right.plotly_chart(style_fig(heat, 400), use_container_width=True)

                st.dataframe(
                    daily.rename(columns={"Peak_MW": "Peak (MW)", "Avg_MW": "Average (MW)",
                                          "Min_MW": "Minimum (MW)"}).round(0),
                    use_container_width=True, hide_index=True,
                )

            with tab4:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=feat_df["Datetime"], y=feat_df["Predicted_MW"],
                    name="Predicted demand (MW)", line=dict(color=GREEN, width=3)))
                fig2.add_trace(go.Scatter(
                    x=feat_df["Datetime"], y=feat_df["temperature_c"],
                    name="Temperature (°C)", yaxis="y2", line=dict(color="#ea580c", width=2)))
                fig2.update_layout(
                    title="Predicted demand vs temperature",
                    yaxis=dict(title="Demand (MW)"),
                    yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
                    hovermode="x unified", legend=dict(orientation="h", y=1.12, x=0),
                )
                st.plotly_chart(style_fig(fig2, 440), use_container_width=True)

                sc = px.scatter(
                    feat_df, x="temperature_c", y="Predicted_MW", color="Hour",
                    color_continuous_scale="Greens",
                    labels={"temperature_c": "Temperature (°C)", "Predicted_MW": "Predicted demand (MW)"},
                    title="How temperature drives demand",
                )
                st.plotly_chart(style_fig(sc, 400), use_container_width=True)

            with tab5:
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
    st.info("👈 Click the green **>** button in the top-left corner to open **Forecast Settings** and click **Predict Consumption**.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model MAE", f"{MODEL_MAE:,.2f} MW", help="Average absolute difference between predicted and actual demand")
    m2.metric("Model RMSE", f"{MODEL_RMSE:,.2f} MW", help="Like MAE but penalises large errors more")
    m3.metric("Forecast window", "Up to 16 days", help="Limited by the Open-Meteo weather forecast")

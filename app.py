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
    page_title="ComEd | Electricity Demand Forecast",
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
# Premium sage + emerald dashboard theme
# ------------------------------------------------------------------
UI_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ink: #10291d;
    --muted: #607568;
    --green: #168348;
    --green-dark: #0d432b;
    --green-soft: #e6f2e9;
    --surface: rgba(250, 253, 250, .82);
    --line: rgba(28, 105, 65, .15);
    --shadow: 0 14px 40px rgba(20, 65, 43, .09);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', system-ui, sans-serif;
}
.stApp {
    color: var(--ink);
    background:
        radial-gradient(ellipse at 8% 4%, rgba(90, 170, 119, .19), transparent 31%),
        radial-gradient(ellipse at 94% 78%, rgba(93, 157, 119, .16), transparent 30%),
        linear-gradient(145deg, #edf4ef 0%, #e2eee6 52%, #d9e8de 100%);
    background-attachment: fixed;
}
[data-testid="stAppViewContainer"], [data-testid="stHeader"] { background: transparent; }
[data-testid="stHeader"] { height: 2.8rem; }
.block-container {
    max-width: 1480px;
    padding: 1.3rem 2.1rem 3.5rem;
    position: relative;
    z-index: 2;
    animation: pageEnter .65s cubic-bezier(.2,.75,.25,1) both;
}

/* Ambient background: intentionally subtle so charts stay readable. */
.bg-layer { position: fixed; inset: 0; z-index: 0; overflow: hidden; pointer-events: none; }
.ambient-orb {
    position: absolute; width: 390px; height: 390px; border-radius: 50%;
    filter: blur(75px); opacity: .17;
    background: radial-gradient(circle, #4caf73 0%, rgba(76,175,115,0) 70%);
    animation: orbDrift 24s ease-in-out infinite alternate;
}
.ambient-orb.orb-a { top: -175px; left: 8%; }
.ambient-orb.orb-b { right: -165px; bottom: -130px; animation-delay: -11s; }
.ambient-grid {
    position: absolute; inset: 0; opacity: .12;
    background-image: linear-gradient(rgba(29, 100, 62, .10) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(29, 100, 62, .10) 1px, transparent 1px);
    background-size: 56px 56px;
    mask-image: linear-gradient(to bottom, black, transparent 78%);
}

/* Restored floating bubbles and energy particles from the original design. */
.bubble-field { position: absolute; inset: 0; overflow: hidden; }
.energy-bubble {
    position: absolute; bottom: -125px; display: block; border-radius: 50%;
    background: radial-gradient(circle at 30% 28%,
        rgba(255,255,255,.62), rgba(130,205,155,.25) 48%, rgba(22,131,72,.10) 100%);
    border: 1px solid rgba(31,125,72,.20);
    box-shadow: inset 0 0 14px rgba(255,255,255,.18), 0 5px 20px rgba(31,105,65,.06);
    opacity: 0; will-change: transform, opacity;
    animation: bubbleRise var(--duration) linear var(--delay) infinite;
}
.energy-bubble::after {
    content: ''; position: absolute; width: 19%; height: 19%; top: 19%; left: 23%;
    border-radius: 50%; background: rgba(255,255,255,.52); filter: blur(1px);
}
.energy-bolt {
    position: absolute; bottom: -70px; color: rgba(21,128,61,.20);
    font-size: var(--size); line-height: 1; opacity: 0;
    text-shadow: 0 3px 12px rgba(21,128,61,.12);
    animation: boltRise var(--duration) linear var(--delay) infinite,
               boltSway 5.5s ease-in-out var(--delay) infinite;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: rgba(232, 242, 235, .94);
    border-right: 1px solid rgba(27, 105, 64, .18);
    backdrop-filter: blur(18px);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.15rem; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #123c28; font-family: 'Manrope', sans-serif; }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label { color: #294838; }
[data-testid="collapsedControl"], [data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] {
    z-index: 999 !important; opacity: 1 !important; visibility: visible !important;
}
[data-testid="collapsedControl"] button,
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapseButton"] button {
    background: linear-gradient(135deg, #0d5835, #1b9b59) !important;
    color: white !important; border: 1px solid rgba(255,255,255,.4) !important;
    border-radius: 12px !important; box-shadow: 0 6px 18px rgba(15, 81, 47, .24) !important;
}
[data-testid="collapsedControl"] svg,
[data-testid="stExpandSidebarButton"] svg,
[data-testid="stSidebarCollapseButton"] svg { color: white !important; fill: white !important; }

/* Hero */
.hero {
    position: relative; overflow: hidden; isolation: isolate;
    padding: 2.15rem 2.35rem 2rem; margin: .25rem 0 1.55rem;
    border-radius: 27px; color: #f5fff7;
    background: linear-gradient(118deg, #0b462b 0%, #11663a 46%, #16884b 100%);
    box-shadow: 0 22px 54px rgba(13, 78, 44, .22), inset 0 1px 0 rgba(255,255,255,.16);
    animation: heroEnter .75s cubic-bezier(.2,.75,.25,1) both;
}
.hero::before {
    content: ''; position: absolute; z-index: -1; width: 570px; height: 570px;
    top: -390px; right: -110px; border-radius: 50%;
    border: 1px solid rgba(255,255,255,.13);
    box-shadow: 0 0 0 38px rgba(255,255,255,.035), 0 0 0 82px rgba(255,255,255,.025);
}
.hero::after {
    content: ''; position: absolute; z-index: -1; width: 360px; height: 360px;
    right: 14%; bottom: -315px; border-radius: 50%;
    background: rgba(158, 231, 179, .13); filter: blur(5px);
}
.hero-topline { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 1.1rem; }
.hero-eyebrow { color: #bcebd0; text-transform: uppercase; letter-spacing: .16em; font-size: .70rem; font-weight: 800; }
.model-status {
    display: inline-flex; align-items: center; gap: .48rem; padding: .38rem .72rem;
    border: 1px solid rgba(223,255,232,.22); border-radius: 999px;
    background: rgba(255,255,255,.10); color: #e9fff0; font-size: .73rem; font-weight: 700;
}
.status-dot { width: 7px; height: 7px; border-radius: 50%; background: #8ff0ae; box-shadow: 0 0 0 4px rgba(143,240,174,.13); animation: statusPulse 2s infinite; }
.hero-layout { display: flex; align-items: center; justify-content: space-between; gap: 2rem; }
.hero h1 { color: #f5fff7 !important; font: 800 clamp(1.8rem, 3.2vw, 2.65rem)/1.12 'Manrope', sans-serif; letter-spacing: -.045em; margin: 0; }
.hero h1 .bolt { display: inline-block; margin-right: .3rem; filter: drop-shadow(0 4px 10px rgba(255,180,65,.3)); animation: boltFloat 2.6s ease-in-out infinite; }
.hero p { color: #d8f4e1 !important; max-width: 780px; font-size: 1rem; line-height: 1.65; margin: .85rem 0 0; }
.hero-badges { display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.25rem; }
.hero-badge { display: inline-flex; align-items: center; gap: .42rem; padding: .46rem .78rem; border-radius: 999px; background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.12); color: #f0fff4; font-size: .76rem; font-weight: 650; transition: transform .22s ease, background .22s ease, border-color .22s ease; }
.hero-badge:hover { transform: translateY(-3px); background: rgba(255,255,255,.2); border-color: rgba(255,255,255,.28); }
.hero-side {
    flex: 0 0 215px; min-width: 180px; padding: 1.15rem 1.25rem;
    border: 1px solid rgba(255,255,255,.17); border-radius: 19px;
    background: rgba(4, 50, 29, .22); backdrop-filter: blur(12px);
}
.hero-side-label { color: #c8ecd4; font-size: .69rem; font-weight: 700; text-transform: uppercase; letter-spacing: .11em; }
.hero-side-value { color: white; font: 800 1.55rem 'Manrope', sans-serif; margin: .25rem 0 .15rem; }
.hero-side-note { color: #d2f0dc; font-size: .75rem; line-height: 1.45; }

/* Section headings and status/insight panels */
.section-heading { display: flex; align-items: end; justify-content: space-between; gap: 1rem; margin: 1.45rem 0 .8rem; }
.section-heading h2 { color: #123c28; font: 800 1.35rem 'Manrope', sans-serif; letter-spacing: -.025em; margin: .1rem 0 0; }
.section-eyebrow { color: #568068; text-transform: uppercase; letter-spacing: .14em; font-size: .66rem; font-weight: 800; }
.section-meta { color: #537161; font-size: .78rem; font-weight: 600; }
.insight-banner {
    display: flex; align-items: center; gap: .8rem; padding: .9rem 1.1rem;
    border: 1px solid rgba(27,119,70,.18); border-radius: 15px;
    background: rgba(247, 252, 248, .84); color: #1b5637;
    box-shadow: 0 7px 20px rgba(20,65,43,.045); margin: .25rem 0 1rem;
    animation: fadeUp .5s ease both;
}
.insight-icon { display: grid; place-items: center; flex: 0 0 34px; height: 34px; border-radius: 11px; background: #dff2e5; font-size: 1rem; }
.insight-copy { font-size: .88rem; line-height: 1.55; }

/* KPI cards */
.kpi-card {
    position: relative; overflow: hidden; min-height: 142px; height: 100%;
    padding: 1.15rem 1.25rem 1.1rem; border-radius: 19px;
    background: linear-gradient(145deg, rgba(252,255,252,.95), rgba(237,247,240,.88));
    border: 1px solid rgba(23,117,66,.17); border-top: 3px solid var(--accent);
    box-shadow: 0 7px 23px rgba(15,59,36,.065);
    transition: transform .24s ease, box-shadow .24s ease, border-color .24s ease;
    animation: cardEnter .65s cubic-bezier(.2,.75,.25,1) both;
}
.kpi-card::after { content: ''; position: absolute; width: 110px; height: 110px; right: -48px; top: -55px; border-radius: 50%; background: var(--accent); opacity: .07; transition: transform .3s ease; }
.kpi-card:hover { transform: translateY(-6px); box-shadow: 0 17px 35px rgba(15, 75, 42, .13); border-color: rgba(23,117,66,.28); }
.kpi-card:hover::after { transform: scale(1.45); }
.kpi-label { color: #577264; font-size: .70rem; font-weight: 800; text-transform: uppercase; letter-spacing: .13em; }
.kpi-value { color: #103b27; font: 800 clamp(1.45rem, 2vw, 1.95rem)/1.25 'Manrope', sans-serif; letter-spacing: -.04em; margin: .42rem 0 .38rem; }
.kpi-range { color: #617a6b; font-size: .78rem; line-height: 1.55; }

/* Accuracy cards */
.acc-card {
    height: 100%; padding: 1.15rem 1.25rem; border-radius: 18px;
    background: linear-gradient(145deg, rgba(248,253,249,.94), rgba(228,242,232,.88));
    border: 1px solid rgba(29,120,70,.17); box-shadow: 0 8px 22px rgba(15,59,36,.05);
    transition: transform .22s ease, box-shadow .22s ease;
}
.acc-card:hover { transform: translateY(-4px); box-shadow: 0 14px 28px rgba(21,128,61,.12); }
.acc-card h4 { color: #294f3a; font: 750 1rem 'Manrope', sans-serif; margin: 0 0 .45rem; }
.acc-big { color: #168348; font: 800 1.65rem 'Manrope', sans-serif; letter-spacing: -.03em; }
.acc-sub { color: #617a6b; font-size: .78rem; line-height: 1.5; margin-top: .2rem; }
.info-card { height: 100%; padding: 1.05rem 1.15rem; border-radius: 17px; background: rgba(241,247,255,.92); border: 1px solid rgba(66,119,184,.16); color: #244c7c; font-size: .83rem; line-height: 1.7; }

/* Streamlit controls */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; font-weight: 750; min-height: 2.7rem;
    transition: transform .2s ease, box-shadow .2s ease, filter .2s ease;
}
.stButton > button[kind="primary"] {
    color: #f5fff7; border: 0;
    background: linear-gradient(120deg, #0c5734, #179653);
    box-shadow: 0 8px 18px rgba(18,106,59,.20);
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button:hover {
    transform: translateY(-2px); filter: brightness(1.04);
    box-shadow: 0 12px 23px rgba(18,106,59,.22);
}
.stDownloadButton > button { border: 1px solid rgba(20,115,64,.25); background: #edf8f0; color: #145b36; }
[data-testid="stMetric"] { background: rgba(249,253,250,.84); border: 1px solid var(--line); border-radius: 15px; padding: .75rem .9rem; box-shadow: 0 5px 16px rgba(15,59,36,.04); transition: transform .2s ease; }
[data-testid="stMetric"]:hover { transform: translateY(-3px); }
[data-testid="stMetricLabel"] { color: #5c7868; }
[data-testid="stMetricValue"] { color: #123c28; font-family: 'Manrope', sans-serif; font-weight: 800; }
.stTabs [data-baseweb="tab-list"] { gap: .38rem; border-bottom: 1px solid rgba(23,117,66,.15); }
.stTabs [data-baseweb="tab"] { padding: .7rem .9rem; color: #607768; font-weight: 700; border-radius: 10px 10px 0 0; }
.stTabs [aria-selected="true"] { color: #0d6b3b !important; background: rgba(224,241,229,.8); }
.stTabs [data-baseweb="tab-highlight"] { background: #168348; height: 3px; }
[data-testid="stDataFrame"] { border: 1px solid rgba(23,117,66,.15); border-radius: 14px; overflow: hidden; }
[data-baseweb="select"] > div, [data-baseweb="input"] > div { border-radius: 10px; border-color: rgba(28,105,65,.23); }
[data-testid="stSlider"] [data-baseweb="slider"] { padding-top: .25rem; }
hr { border-color: rgba(28,105,65,.16) !important; }

/* Welcome state */
.welcome-panel {
    position: relative; overflow: hidden; padding: 1.8rem; border-radius: 23px;
    background: rgba(249,253,250,.76); border: 1px solid rgba(26,114,67,.16);
    box-shadow: var(--shadow); margin: .2rem 0 1.1rem;
}
.welcome-panel h2 { color: #143e29; font: 800 1.65rem 'Manrope', sans-serif; margin: .2rem 0 .55rem; }
.welcome-panel p { color: #5b7465; line-height: 1.7; max-width: 790px; margin: 0; }
.welcome-kicker { color: #168348; font-size: .68rem; letter-spacing: .16em; font-weight: 800; }
.feature-card { height: 100%; padding: 1.15rem; border-radius: 17px; background: rgba(249,253,250,.76); border: 1px solid rgba(26,114,67,.14); transition: transform .22s ease, box-shadow .22s ease; }
.feature-card:hover { transform: translateY(-4px); box-shadow: 0 13px 26px rgba(16,76,43,.09); }
.feature-icon { font-size: 1.35rem; margin-bottom: .55rem; }
.feature-title { color: #17432d; font: 750 .92rem 'Manrope', sans-serif; margin-bottom: .28rem; }
.feature-copy { color: #657d6e; font-size: .78rem; line-height: 1.55; }

@keyframes pageEnter { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
@keyframes heroEnter { from { opacity: 0; transform: translateY(12px) scale(.992); } to { opacity: 1; transform: translateY(0) scale(1); } }
@keyframes cardEnter { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeUp { from { opacity: 0; transform: translateY(7px); } to { opacity: 1; transform: translateY(0); } }
@keyframes orbDrift { from { transform: translate3d(0,0,0) scale(1); } to { transform: translate3d(75px,55px,0) scale(1.18); } }
@keyframes bubbleRise {
    0% { transform: translate3d(0,0,0) scale(.72); opacity: 0; }
    9% { opacity: .48; }
    78% { opacity: .30; }
    100% { transform: translate3d(var(--drift),-115vh,0) scale(1.12); opacity: 0; }
}
@keyframes boltRise {
    0% { transform: translate3d(0,0,0) rotate(-8deg); opacity: 0; }
    12% { opacity: .34; }
    82% { opacity: .23; }
    100% { transform: translate3d(24px,-112vh,0) rotate(12deg); opacity: 0; }
}
@keyframes boltSway { 0%,100% { margin-left: -9px; } 50% { margin-left: 9px; } }
@keyframes boltFloat { 0%,100% { transform: translateY(0) rotate(-4deg); } 50% { transform: translateY(-5px) rotate(4deg); } }
@keyframes statusPulse { 0%,100% { box-shadow: 0 0 0 4px rgba(143,240,174,.13); } 50% { box-shadow: 0 0 0 7px rgba(143,240,174,.04); } }

@media (max-width: 900px) {
    .block-container { padding: 1rem 1rem 2.5rem; }
    .hero { padding: 1.55rem; border-radius: 21px; }
    .hero-layout { align-items: flex-start; }
    .hero-side { display: none; }
}
@media (max-width: 600px) {
    .hero-topline { align-items: flex-start; flex-direction: column; }
    .hero h1 { font-size: 1.75rem; }
    .hero p { font-size: .9rem; }
    .section-heading { align-items: flex-start; flex-direction: column; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; scroll-behavior: auto !important; transition-duration: .01ms !important; }
}
</style>
"""
st.markdown(UI_CSS, unsafe_allow_html=True)


def background_layer():
    # Fixed seed keeps particle positions stable across Streamlit reruns.
    rnd = random.Random(23)
    particles = []

    for _ in range(18):
        size = rnd.randint(18, 68)
        left = rnd.randint(1, 98)
        duration = rnd.randint(17, 32)
        delay = -rnd.randint(0, 28)
        drift = rnd.randint(-42, 42)
        particles.append(
            f'<span class="energy-bubble" style="left:{left}%;width:{size}px;height:{size}px;'
            f'--duration:{duration}s;--delay:{delay}s;--drift:{drift}px"></span>'
        )

    for _ in range(8):
        left = rnd.randint(3, 97)
        duration = rnd.randint(19, 31)
        delay = -rnd.randint(0, 24)
        size = rnd.randint(19, 29)
        particles.append(
            f'<span class="energy-bolt" style="left:{left}%;font-size:{size}px;'
            f'--duration:{duration}s;--delay:{delay}s">⚡</span>'
        )

    return f"""
    <div class="bg-layer" aria-hidden="true">
        <div class="ambient-grid"></div>
        <div class="ambient-orb orb-a"></div>
        <div class="ambient-orb orb-b"></div>
        <div class="bubble-field">{"".join(particles)}</div>
    </div>
    """



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
        plot_bgcolor="rgba(249,253,250,0.72)",
        font=dict(family="DM Sans, sans-serif", color="#173b29", size=12),
        margin=dict(l=14, r=14, t=64, b=14),
        title=dict(font=dict(family="Manrope, sans-serif", size=17, color="#173b29"), x=0.02),
        hoverlabel=dict(bgcolor="#103d27", font=dict(color="white", family="DM Sans, sans-serif")),
        legend=dict(font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(
        gridcolor="rgba(21,128,61,0.10)",
        zerolinecolor="rgba(21,128,61,0.12)",
        linecolor="rgba(21,128,61,0.18)",
        tickfont=dict(color="#607568"),
    )
    fig.update_yaxes(
        gridcolor="rgba(21,128,61,0.10)",
        zerolinecolor="rgba(21,128,61,0.12)",
        linecolor="rgba(21,128,61,0.18)",
        tickfont=dict(color="#607568"),
    )
    return fig


PLOTLY_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


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
        <div class="hero-topline">
            <span class="hero-eyebrow">Grid intelligence · energy analytics</span>
            <span class="model-status"><span class="status-dot"></span> MODEL READY</span>
        </div>
        <div class="hero-layout">
            <div>
                <h1><span class="bolt">⚡</span> ComEd Electricity<br>Demand Predictor</h1>
                <p>Explore hourly electricity demand across Northern Illinois. Live weather, Illinois holiday signals and time-of-day patterns come together to power your forecast.</p>
                <div class="hero-badges">
                    <span class="hero-badge">🌦 Live weather</span>
                    <span class="hero-badge">📅 Illinois holidays</span>
                    <span class="hero-badge">🤖 Machine learning</span>
                    <span class="hero-badge">🎛 What-if simulator</span>
                </div>
            </div>
            <div class="hero-side">
                <div class="hero-side-label">Forecast region</div>
                <div class="hero-side-value">Chicago, IL</div>
                <div class="hero-side-note">ComEd service area<br>Hourly demand · MW</div>
            </div>
        </div>
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
    st.markdown("## ⚙️ Forecast controls")
    st.caption("Choose your time window and simulate changing weather.")
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
    st.markdown("---")
    st.subheader("📏 Model accuracy")
    st.metric("MAE", f"{MODEL_MAE:,.2f} MW")
    st.metric("RMSE", f"{MODEL_RMSE:,.2f} MW")
    st.caption("Measured on held-out test data. Region: Chicago / Northern Illinois (ComEd).")

# ------------------------------------------------------------------
# Welcome state (shown until the first forecast is generated)
# ------------------------------------------------------------------
if not st.session_state.has_run:
    st.markdown(
        """
        <div class="welcome-panel">
            <div class="welcome-kicker">YOUR ENERGY FORECAST WORKSPACE</div>
            <h2>See what electricity demand could look like next.</h2>
            <p>Set a forecast window, experiment with a temperature shift, and generate an interactive hourly outlook. Use the Forecast controls in the sidebar, or start with the button below.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown('<div class="feature-card"><div class="feature-icon">📈</div><div class="feature-title">Explore demand patterns</div><div class="feature-copy">Inspect hourly peaks, daily averages and demand changes across your selected period.</div></div>', unsafe_allow_html=True)
    with w2:
        st.markdown('<div class="feature-card"><div class="feature-icon">🌡️</div><div class="feature-title">Run what-if scenarios</div><div class="feature-copy">Shift the forecast temperature and compare the simulated outlook with normal conditions.</div></div>', unsafe_allow_html=True)
    with w3:
        st.markdown('<div class="feature-card"><div class="feature-icon">🧭</div><div class="feature-title">Understand uncertainty</div><div class="feature-copy">Review model error metrics and an indicative range around every hourly prediction.</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height: .6rem'></div>", unsafe_allow_html=True)
    start_col, _ = st.columns([1, 2.6])
    with start_col:
        if st.button("⚡ Generate my forecast", type="primary", use_container_width=True, key="welcome_predict"):
            st.session_state.has_run = True

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
            st.markdown("<div class='section-heading'><div><div class='section-eyebrow'>Forecast snapshot</div><h2>Demand at a glance</h2></div><div class='section-meta'>Chicago local time · MW / MWh</div></div>", unsafe_allow_html=True)
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
            st.markdown("<div class='section-heading'><div><div class='section-eyebrow'>Model performance</div><h2>How accurate is this forecast?</h2></div><div class='section-meta'>Held-out test set metrics</div></div>", unsafe_allow_html=True)
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
                st.markdown(
                    f"<div class='info-card'><b>How to read these metrics</b><br>On average, an hourly prediction differs from real demand by about <b>{MODEL_MAE:,.0f} MW</b> (MAE). RMSE (<b>{MODEL_RMSE:,.0f} MW</b>) penalises larger misses more heavily, so occasional errors can be higher, especially during unusual weather or on exceptional days. The chart band is an approximate guide based on past test error.</div>",
                    unsafe_allow_html=True,
                )

            # --------------------------- Tabs ---------------------------
            tab1, tab2, tab3, tab4, tab5 = st.tabs(
                ["📈 Forecast", "🔍 Hour inspector", "🗓 Daily view", "🌡 Weather vs demand", "📋 Data"]
            )

            with tab1:
                chart_controls = st.columns([1.2, 1, 1.5])
                with chart_controls[0]:
                    chart_window = st.selectbox(
                        "Time window",
                        ["All selected hours", "Next 24 hours", "Next 48 hours"],
                        index=0,
                        key="chart_window",
                    )
                with chart_controls[1]:
                    show_peak = st.checkbox("Highlight peak", value=True, key="show_peak")
                with chart_controls[2]:
                    st.caption("Tip: hover for hourly values · scroll to zoom · drag to pan")

                chart_df = feat_df.copy()
                if chart_window == "Next 24 hours":
                    chart_df = chart_df[chart_df["Datetime"] < chart_df["Datetime"].min() + timedelta(hours=24)]
                elif chart_window == "Next 48 hours":
                    chart_df = chart_df[chart_df["Datetime"] < chart_df["Datetime"].min() + timedelta(hours=48)]

                base_chart_df = base_df.loc[chart_df.index]
                chart_peak = chart_df.loc[chart_df["Predicted_MW"].idxmax()]

                fig = go.Figure()
                if band_choice != "Hide":
                    label = "± MAE range" if band_choice.startswith("Typical") else "± RMSE range"
                    fig.add_trace(go.Scatter(
                        x=pd.concat([chart_df["Datetime"], chart_df["Datetime"][::-1]]),
                        y=pd.concat([chart_df["Upper_MW"], chart_df["Lower_MW"][::-1]]),
                        fill="toself", fillcolor="rgba(22,131,72,0.13)",
                        line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip", name=label,
                    ))
                if temp_shift != 0:
                    fig.add_trace(go.Scatter(
                        x=base_chart_df["Datetime"], y=base_chart_df["Predicted_MW"], mode="lines",
                        name="Normal forecast", line=dict(color="#82958a", width=2, dash="dot"),
                        hovertemplate="%{y:,.0f} MW<extra></extra>",
                    ))
                fig.add_trace(go.Scatter(
                    x=chart_df["Datetime"], y=chart_df["Predicted_MW"], mode="lines",
                    name="Predicted demand", line=dict(color=GREEN, width=3.4, shape="spline", smoothing=0.55),
                    fill="tozeroy", fillcolor="rgba(22,131,72,0.045)",
                    hovertemplate="<b>%{x|%a, %d %b · %H:%M}</b><br>%{y:,.0f} MW<extra></extra>",
                ))
                if show_peak:
                    fig.add_trace(go.Scatter(
                        x=[chart_peak["Datetime"]], y=[chart_peak["Predicted_MW"]], mode="markers+text", name="Peak in view",
                        marker=dict(color="#ed8736", size=12, line=dict(color="white", width=2)),
                        text=[f"Peak {chart_peak['Predicted_MW']:,.0f} MW"], textposition="top center",
                        hovertemplate="Peak: %{y:,.0f} MW<extra></extra>",
                    ))
                fig.update_layout(
                    title=f"Hourly demand forecast · {start_date} to {end_date}",
                    xaxis_title="Date & time (Chicago local)", yaxis_title="Electricity demand (MW)",
                    hovermode="x unified", legend=dict(orientation="h", y=1.13, x=0),
                )
                fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.055))
                st.plotly_chart(style_fig(fig, 500), use_container_width=True, config=PLOTLY_CONFIG)
                st.caption("The shaded area represents the selected historical error band (MAE or RMSE), not a guaranteed prediction interval.")

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
                left.plotly_chart(style_fig(bar, 400), use_container_width=True, config=PLOTLY_CONFIG)

                pivot = feat_df.pivot_table(index="Date", columns="Hour",
                                            values="Predicted_MW", aggfunc="mean")
                heat = px.imshow(pivot, aspect="auto", color_continuous_scale="Greens",
                                 labels=dict(x="Hour of day", y="Date", color="MW"),
                                 title="Demand heatmap (hour × day)")
                right.plotly_chart(style_fig(heat, 400), use_container_width=True, config=PLOTLY_CONFIG)

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
                st.plotly_chart(style_fig(fig2, 440), use_container_width=True, config=PLOTLY_CONFIG)

                sc = px.scatter(
                    feat_df, x="temperature_c", y="Predicted_MW", color="Hour",
                    color_continuous_scale="Greens",
                    labels={"temperature_c": "Temperature (°C)", "Predicted_MW": "Predicted demand (MW)"},
                    title="How temperature drives demand",
                )
                st.plotly_chart(style_fig(sc, 400), use_container_width=True, config=PLOTLY_CONFIG)

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
    st.caption("Configure a range in the sidebar to begin. The model accuracy figures below are measured on held-out test data.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model MAE", f"{MODEL_MAE:,.2f} MW", help="Average absolute difference between predicted and actual demand")
    m2.metric("Model RMSE", f"{MODEL_RMSE:,.2f} MW", help="Like MAE but penalises large errors more")
    m3.metric("Forecast window", "Up to 16 days", help="Limited by the Open-Meteo weather forecast")

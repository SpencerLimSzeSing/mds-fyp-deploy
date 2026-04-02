import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import streamlit.components.v1 as components
from streamlit_folium import st_folium
import folium
import math
import pandas as pd
import plotly.express as px
from sklearn.model_selection import KFold
import base64
import datetime

st.set_page_config(layout="wide")


def get_base64_image(file_path):
    """Convert image file to Base64."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Model loading with caching
@st.cache_resource
def load_knn_model():
    return joblib.load("models/best_knn_model.joblib")


@st.cache_resource
def load_rf_model():
    return joblib.load("models/best_rf_model.joblib")


@st.cache_resource
def load_xgb_model():
    return joblib.load("models/best_xgb_model.joblib")


@st.cache_resource
def load_meta_ann_model():
    return tf.keras.models.load_model(
        "models/Tuned_meta_ann_model.keras", compile=False
    )


# Pre-load models (will only load once due to caching)
knn_model = load_knn_model()
rf_model = load_rf_model()
xgb_model = load_xgb_model()
meta_ann = load_meta_ann_model()
base_models = [knn_model, rf_model, xgb_model]
# Category mapping
category_mapping = {
    0: "Rainfall_Category_No Rain",
    1: "Rainfall_Category_Moderate Rain",
    2: "Rainfall_Category_Heavy Rain",
    3: "Rainfall_Category_Very Heavy Rain",
}


# Function to add a background image globally
def add_bg_from_local(image_file):
    """
    Adds a background image to a Streamlit app.

    Parameters:
    - image_file: Path to the local image file
    """
    with open(image_file, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()

    # Inject CSS to set the background image globally
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Call the function once, globally
add_bg_from_local("assets/pexels.jpg")


# Load dataset
def load_data():
    file_path = "tab3/datasetdeploy.csv"
    return pd.read_csv(file_path)


data = load_data()


def styled_section(header, color):
    st.markdown(
        f"""
        <div style="background-color: {color}; padding: 2px 10px; border-radius: 5px; margin-bottom: 10px;">
            <h4 style="margin: 0; color: black;">{header}</h4>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Replace your current initialization loop with this
try:
    for key in [
        "min_temp_touched",
        "max_temp_touched",
        "temp_9am_touched",
        "temp_3pm_touched",
        "pressure_9am_touched",
        "pressure_3pm_touched",
        "humidity_9am_touched",
        "humidity_3pm_touched",
        "evaporation_touched",
        "wind_gust_speed_touched",
        "wind_speed_9am_touched",
        "wind_speed_3pm_touched",
        "sunshine_touched",
    ]:
        if key not in st.session_state:
            st.session_state[key] = False
except Exception:
    pass


# ── Helper: callback factory ─────────────────────────────────────────
def make_touch_callback(flag_key):
    def callback():
        st.session_state[flag_key] = True

    return callback


# Tabs for Introduction and Prediction
tab1, tab2, tab3 = st.tabs(["ℹ️ Introduction", "🌧️ Rainfall Prediction", "📊 Dashboard"])

# ── CSS ───────────
st.markdown(
    """
    <style>
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center !important;
        gap: 8px !important;
    }
    /* Force full-width dark tab bar */
    .stTabs {
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center !important;
        gap: 8px !important;
        background-color: transparent !important;
        border-bottom: 1px solid rgba(255,255,255,0.1) !important;
        padding-bottom: 4px !important;
        width: 100% !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.8rem !important;
        padding: 10px 24px !important;
        background-color: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #f1f5f9 !important;
        background-color: transparent !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #3b82f6 !important;
        height: 3px !important;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.8rem !important;
        padding: 10px 24px !important;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        background-color: #3b82f6 !important;
    }
           
    .main { background-color: #0e1117; }
    .project-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: white;
        padding-top: 50px;
    }
    .sub-title { color: #9ea4ad; font-size: 1.2rem; margin-bottom: 40px; }
    .card {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 15px !important;
        padding: 25px !important;
        height: 100% !important;
    }
    .card-text {
        color: #cbd5e0 !important;
        font-size: 1.2rem !important;
        line-height: 1.7 !important;
    }
    .card-title { color: white; font-weight: bold; font-size: 1.3rem; margin-top: 10px; }
    .dash-panel {
        background: #0a0e1a;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 20px 18px 24px;
    }
    .panel-header { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
    .panel-icon {
        background: rgba(59,130,246,0.18);
        border-radius: 9px;
        width: 36px; height: 36px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem;
    }
    .panel-title { font-size: 0.75rem; font-weight: 800; letter-spacing: 2.5px; text-transform: uppercase; color: #f1f5f9; }
    /* Force solid fill on bordered containers only */
    div[data-testid="stVerticalBlockBorderWrapper"],
    div[data-testid="stVerticalBlockBorderWrapper"] *,
    div[data-testid="stVerticalBlockBorderWrapper"] > div,
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div,
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div > div,
    div[data-testid="stVerticalBlockBorderWrapper"] > div > div > div > div {
        background: #111827 !important;
        background-color: #111827 !important;
        backdrop-filter: none !important;
        -webkit-backdrop-filter: none !important;
    }
    /* Keep sliders and interactive elements readable */
    div[data-testid="stVerticalBlockBorderWrapper"] [data-baseweb="slider"] * {
        background: transparent !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] [data-baseweb="slider"] [role="slider"] {
        background: #3b82f6 !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] [data-baseweb="slider"] [data-testid="stSliderTrackFill"] {
        background: #3b82f6 !important;
    }
        div[data-testid="stSlider"] > label { display: none !important; }
        div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
            background: #3b82f6 !important;
            border-color: #3b82f6 !important;
        }
    div[data-testid="stDateInput"] label {
        font-size: 0.6rem !important; font-weight: 700 !important;
        letter-spacing: 1.8px !important; text-transform: uppercase !important;
        color: #64748b !important;
    }
    div[data-testid="stDateInput"] input {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important; color: #e2e8f0 !important;
    }
    div[data-testid="stSelectbox"] label {
        font-size: 0.6rem !important; font-weight: 700 !important;
        letter-spacing: 1.8px !important; text-transform: uppercase !important;
        color: #64748b !important;
    }
    div[data-testid="stSelectbox"] > div > div {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important; color: #e2e8f0 !important;
    }
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; padding: 12px 28px !important;
        font-size: 0.88rem !important; font-weight: 700 !important;
        letter-spacing: 1px !important; text-transform: uppercase !important;
        width: 100% !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
        box-shadow: 0 8px 25px rgba(37,99,235,0.4) !important;
    }
    div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        background-color: #374151 !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 8px !important;
        color: #e2e8f0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Tab 1: Introduction
with tab1:
    # PROJECT OVERVIEW label
    st.markdown(
        '<p style="color: #3182ce; font-weight: 600; letter-spacing: 2px; font-size: 0.85rem; text-transform: uppercase; margin-bottom: -20px;">PROJECT OVERVIEW</p>',
        unsafe_allow_html=True,
    )
    # Main title
    st.markdown(
        '<h1 class="project-header" style="font-size: 2.8rem; font-weight: 700; letter-spacing: -0.5px;">Advanced Meteorological Forecasting System</h1>',
        unsafe_allow_html=True,
    )

    # Subtitle
    st.markdown(
        '<p class="sub-title" style="font-size: 1.05rem; color: #a0aec0; line-height: 1.6;">Leveraging stacking ensemble learning and deep neural networks to provide high-precision rainfall classification across Australia\'s diverse climate zones.</p>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

    st.markdown(
        f"""
                <div class="card">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                        <div style="color: #3182ce; font-size: 1.8rem;">☰</div>
                        <div class="card-title" style="margin: 0; font-size: 1.8rem; font-weight: 700; letter-spacing: 0.5px;">Project Background</div>
                    </div>
                    <p class="card-text" style="font-size: 1.2rem; color: #cbd5e0; line-height: 1.7;">
                        Accurate daily rainfall prediction is critical for effective water resource management,
                        disaster preparedness, and short-term decision-making. Traditional machine
                        learning models, while useful, but often struggle with the complexity in meteorological
                        data.
                        </p>
                        <p class="card-text" style="font-size: 1.2rem; color: #cbd5e0; line-height: 1.7;">
                        The performance of the models were evaluated and has shown that the Meta-ANN significantly outperformed the
                        standalone ANN and base models, achieving higher accuracy and balanced performance
                        across rainfall categories. Adjustments such as class weighting and hyperparameter
                        tuning further enhanced the model's ability to address class imbalances. These findings
                        highlight the effectiveness of ensemble methods in capturing the complex relationships
                        between meteorological factors, leading to a reliable approach to daily rainfall prediction.
                    </p>
                </div>
            """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='margin-top: 16px;'></div>", unsafe_allow_html=True)

    col1, gap, col2 = st.columns([1, 0.05, 1])
    with col1:
        st.markdown(
            f"""
                <div class="card">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 10px;">
                        <div style="color: #00d1b2; font-size: 1.8rem;">⚙️</div>
                        <div class="card-title" style="margin: 0; font-size: 1.8rem; font-weight: 700; letter-spacing: 0.5px;">The Methodology</div>
                    </div>
                    <p class="card-text" style="font-size: 1.2rem; color: #cbd5e0; line-height: 1.7;">
                        The system integrates base learners: <strong>K-Nearest Neighbors (KNN)</strong>, <strong>Random Forest (RF)</strong>, and <strong>Extreme Gradient Boosting (XGB)</strong> as base learners, 
                        with a Meta Learner:<strong>Meta-ANN</strong> learner optimizing the final classification. This multi-layered 
                        approach captures complex atmospheric non-linearities.
                    </p>
                </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        # --- Data Source ---
        with open("tab3/map.png", "rb") as f:
            map_b64 = base64.b64encode(f.read()).decode()

        st.markdown(
            f"""
            <div class="card">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">
                    <div style="color: #e67e22; font-size: 1.8rem;">⚡</div>
                    <div style="margin: 0; font-size: 1.8rem; font-weight: 700; letter-spacing: 0.5px; color: white;">Data Source</div>
            </div>
            <p style="font-size: 1.2rem; color: #cbd5e0; line-height: 1.7; margin-bottom: 16px;">
                The dataset is sourced from the <strong>Bureau of Meteorology, Australia</strong>, 
                comprising <strong>10 years</strong> of daily weather observations with a comprehensive 
                range of meteorological variables. Observations were gathered from weather stations 
                distributed across Australia, as shown in the map below.
            </p>
            <div style="display: flex; justify-content: center;">
                <img src="data:image/png;base64,{map_b64}" 
                    alt="Map of Weather Stations in Australia" 
                    style="width: 80%; border-radius: 8px; object-fit: contain;" />
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Tab 2: Rainfall Prediction
with tab2:
    # ── Page heading ────────────────────────────────────────────────────
    st.markdown(
        """
    <div style="margin-bottom:20px;">
        <h1 style="font-size:2.4rem;font-weight:800;color:#f1f5f9;margin:0 0 4px;">
            Predictive Modeling
        </h1>
        <p style="color:#3b82f6;font-size:0.7rem;font-weight:700;
                  letter-spacing:3px;text-transform:uppercase;margin:0 0 4px;">
            Configure meteorological variables to generate a precipitation forecast.
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ── Main layout: 4 panels + result column ───────────────────────────
    col_thermal, g1, col_atmos, g2, col_wind, g3, col_temp, g4 = st.columns(
        [1.1, 0.05, 1.2, 0.05, 1.2, 0.05, 1.2, 0.05]
    )

    # ───────────────────────────────────────

    with col_thermal:
        # Header inside the container
        st.markdown(
            """
        <div style="background:#161b22;border:1px solid #30363d;
                    border-radius:14px;padding:16px 20px 12px 20px;margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="background:rgba(59,130,246,0.18);border-radius:9px;
                            width:36px;height:36px;display:flex;align-items:center;
                            justify-content:center;font-size:1rem;">🌡️</div>
                <span style="font-size:1.5rem;font-weight:800;letter-spacing:2.5px;
                             text-transform:uppercase;color:#f1f5f9;">THERMAL PROFILE</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        # ─────────────────────────
        # MIN TEMP
        min_temp_val = st.slider(
            "min_temp",
            -10.0,
            50.0,
            -10.0,
            0.1,  # ← default changed to -10.0
            label_visibility="collapsed",
            on_change=make_touch_callback("min_temp_touched"),
            key="min_temp_slider",
        )
        min_temp = min_temp_val if st.session_state["min_temp_touched"] else None
        display_min = (
            f"{min_temp_val}°C" if st.session_state["min_temp_touched"] else "—"
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-top:-32px;margin-bottom:8px;pointer-events:none;">'
            f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:1.8px;'
            f'text-transform:uppercase;color:#cbd5e0;">MIN TEMP</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:#3b82f6;">{display_min}</span></div>',
            unsafe_allow_html=True,
        )

        # MAX TEMP
        max_temp_val = st.slider(
            "max_temp",
            -10.0,
            50.0,
            -10.0,
            0.1,  # ← default changed to -10.0
            label_visibility="collapsed",
            on_change=make_touch_callback("max_temp_touched"),
            key="max_temp_slider",
        )
        max_temp = max_temp_val if st.session_state["max_temp_touched"] else None
        display_max = (
            f"{max_temp_val}°C" if st.session_state["max_temp_touched"] else "—"
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-top:-32px;margin-bottom:8px;pointer-events:none;">'
            f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:1.8px;'
            f'text-transform:uppercase;color:#cbd5e0;">MAX TEMP</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:#3b82f6;">{display_max}</span></div>',
            unsafe_allow_html=True,
        )

        # TEMP 9AM
        temp_9am_val = st.slider(
            "temp_9am",
            -10.0,
            50.0,
            -10.0,
            0.1,  # ← default changed to -10.0
            label_visibility="collapsed",
            on_change=make_touch_callback("temp_9am_touched"),
            key="temp_9am_slider",
        )
        temp_9am = temp_9am_val if st.session_state["temp_9am_touched"] else None
        display_temp_9am = (
            f"{temp_9am_val}°C" if st.session_state["temp_9am_touched"] else "—"
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-top:-32px;margin-bottom:8px;pointer-events:none;">'
            f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:1.8px;'
            f'text-transform:uppercase;color:#cbd5e0;">TEMP (9AM)</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:#3b82f6;">{display_temp_9am}</span></div>',
            unsafe_allow_html=True,
        )

        # TEMP 3PM
        temp_3pm_val = st.slider(
            "temp_3pm",
            -10.0,
            50.0,
            -10.0,
            0.1,  # ← default changed to -10.0
            label_visibility="collapsed",
            on_change=make_touch_callback("temp_3pm_touched"),
            key="temp_3pm_slider",
        )
        temp_3pm = temp_3pm_val if st.session_state["temp_3pm_touched"] else None
        display_temp_3pm = (
            f"{temp_3pm_val}°C" if st.session_state["temp_3pm_touched"] else "—"
        )

        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-top:-32px;margin-bottom:8px;pointer-events:none;">'
            f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:1.8px;'
            f'text-transform:uppercase;color:#cbd5e0;">TEMP (3PM)</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:#3b82f6;">{display_temp_3pm}</span></div>',
            unsafe_allow_html=True,
        )

    with col_atmos:
        st.markdown(
            """
        <div style="background:#161b22;border:1px solid #30363d;
                    border-radius:14px;padding:16px 20px 12px 20px;margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="background:rgba(59,130,246,0.18);border-radius:9px;
                            width:36px;height:36px;display:flex;align-items:center;
                            justify-content:center;font-size:1rem;">🌀</div>
                <span style="font-size:1.5rem;font-weight:800;letter-spacing:2.5px;
                             text-transform:uppercase;color:#f1f5f9;">ATMOSPHERIC</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        # ─────────────────────────
        pressure_9am_val = st.slider(
            "pressure_9am",
            900.0,
            1100.0,
            900.0,
            0.1,
            label_visibility="collapsed",
            on_change=make_touch_callback("pressure_9am_touched"),
            key="pressure_9am_slider",
        )
        pressure_9am = (
            pressure_9am_val if st.session_state["pressure_9am_touched"] else None
        )
        display_pressure_9am = (
            f"{pressure_9am_val}hPa"
            if st.session_state["pressure_9am_touched"]
            else "—"
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-top:-32px;margin-bottom:8px;pointer-events:none;">'
            f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:1.8px;'
            f'text-transform:uppercase;color:#cbd5e0;">PRESSURE (9AM)</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:#3b82f6;">{display_pressure_9am}</span></div>',
            unsafe_allow_html=True,
        )

        # ─────────────────────────
        pressure_3pm_val = st.slider(
            "pressure_3pm",
            900.0,
            1100.0,
            900.0,
            0.1,
            label_visibility="collapsed",
            on_change=make_touch_callback("pressure_3pm_touched"),
            key="pressure_3pm_slider",
        )
        pressure_3pm = (
            pressure_3pm_val if st.session_state["pressure_3pm_touched"] else None
        )
        display_pressure_3pm = (
            f"{pressure_3pm_val}hPa"
            if st.session_state["pressure_3pm_touched"]
            else "—"
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-top:-32px;margin-bottom:8px;pointer-events:none;">'
            f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:1.8px;'
            f'text-transform:uppercase;color:#cbd5e0;">PRESSURE (3PM)</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:#3b82f6;">{display_pressure_3pm}</span></div>',
            unsafe_allow_html=True,
        )

        # ─────────────────────────
        humidity_9am_val = st.slider(
            "humidity_9am",
            0.0,
            100.0,
            0.0,
            1.0,
            label_visibility="collapsed",
            on_change=make_touch_callback("humidity_9am_touched"),
            key="humidity_9am_slider",
        )
        humidity_9am = (
            humidity_9am_val if st.session_state["humidity_9am_touched"] else None
        )
        display_humidity_9am = (
            f"{humidity_9am_val}%" if st.session_state["humidity_9am_touched"] else "—"
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-top:-32px;margin-bottom:8px;pointer-events:none;">'
            f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:1.8px;'
            f'text-transform:uppercase;color:#cbd5e0;">HUMIDITY (9AM)</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:#3b82f6;">{display_humidity_9am}</span></div>',
            unsafe_allow_html=True,
        )

        # ─────────────────────────
        humidity_3pm_val = st.slider(
            "humidity_3pm",
            0.0,
            100.0,
            0.0,
            1.0,
            label_visibility="collapsed",
            on_change=make_touch_callback("humidity_3pm_touched"),
            key="humidity_3pm_slider",
        )
        humidity_3pm = (
            humidity_3pm_val if st.session_state["humidity_3pm_touched"] else None
        )
        display_humidity_3pm = (
            f"{humidity_3pm_val}%" if st.session_state["humidity_3pm_touched"] else "—"
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-top:-32px;margin-bottom:8px;pointer-events:none;">'
            f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:1.8px;'
            f'text-transform:uppercase;color:#cbd5e0;">HUMIDITY (3PM)</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:#3b82f6;">{display_humidity_3pm}</span></div>',
            unsafe_allow_html=True,
        )
        # ─────────────────────────
        evaporation_val = st.slider(
            "evaporation",
            0.0,
            150.0,
            0.0,
            0.1,
            label_visibility="collapsed",
            on_change=make_touch_callback("evaporation_touched"),
            key="evaporation_slider",
        )
        evaporation = (
            evaporation_val if st.session_state["evaporation_touched"] else None
        )
        display_evaporation = (
            f"{evaporation_val}mm" if st.session_state["evaporation_touched"] else "—"
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-top:-32px;margin-bottom:8px;pointer-events:none;">'
            f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:1.8px;'
            f'text-transform:uppercase;color:#cbd5e0;">EVAPORATION</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:#3b82f6;">{display_evaporation}</span></div>',
            unsafe_allow_html=True,
        )

    with col_wind:
        st.markdown(
            """
            <div style="background:#161b22;border:1px solid #30363d;
                        border-radius:14px;padding:16px 20px 12px 20px;margin-bottom:8px;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <div style="background:rgba(59,130,246,0.18);border-radius:9px;
                                width:36px;height:36px;display:flex;align-items:center;
                                justify-content:center;font-size:1rem;">💨</div>
                    <span style="font-size:1.5rem;font-weight:800;letter-spacing:2.5px;
                                text-transform:uppercase;color:#f1f5f9;">WIND & OTHERS</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # ─────────────────────────
        wind_gust_speed_val = st.slider(
            "wind_gust_speed",
            0.0,
            200.0,
            0.0,
            1.0,
            label_visibility="collapsed",
            on_change=make_touch_callback("wind_gust_speed_touched"),
            key="wind_gust_speed_slider",
        )
        wind_gust_speed = (
            wind_gust_speed_val if st.session_state["wind_gust_speed_touched"] else None
        )
        display_wind_gust_speed = (
            f"{wind_gust_speed_val}km/h"
            if st.session_state["wind_gust_speed_touched"]
            else "—"
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-top:-32px;margin-bottom:8px;pointer-events:none;">'
            f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:1.8px;'
            f'text-transform:uppercase;color:#cbd5e0;">WIND GUST</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:#3b82f6;">{display_wind_gust_speed}</span></div>',
            unsafe_allow_html=True,
        )

        # ─────────────────────────
        wind_speed_9am_val = st.slider(
            "wind_9am",
            0.0,
            150.0,
            0.0,
            1.0,
            label_visibility="collapsed",
            on_change=make_touch_callback("wind_speed_9am_touched"),
            key="wind_speed_9am_slider",
        )
        wind_speed_9am = (
            wind_speed_9am_val if st.session_state["wind_speed_9am_touched"] else None
        )
        display_wind_speed_9am = (
            f"{wind_speed_9am_val}km/h"
            if st.session_state["wind_speed_9am_touched"]
            else "—"
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-top:-32px;margin-bottom:8px;pointer-events:none;">'
            f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:1.8px;'
            f'text-transform:uppercase;color:#cbd5e0;">WIND (9AM)</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:#3b82f6;">{display_wind_speed_9am}</span></div>',
            unsafe_allow_html=True,
        )

        # ─────────────────────────
        wind_speed_3pm_val = st.slider(
            "wind_3pm",
            0.0,
            150.0,
            0.0,
            1.0,
            label_visibility="collapsed",
            on_change=make_touch_callback("wind_speed_3pm_touched"),
            key="wind_speed_3pm_slider",
        )
        wind_speed_3pm = (
            wind_speed_3pm_val if st.session_state["wind_speed_3pm_touched"] else None
        )
        display_wind_speed_3pm = (
            f"{wind_speed_3pm_val}km/h"
            if st.session_state["wind_speed_3pm_touched"]
            else "—"
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-top:-32px;margin-bottom:8px;pointer-events:none;">'
            f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:1.8px;'
            f'text-transform:uppercase;color:#cbd5e0;">WIND (3PM)</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:#3b82f6;">{display_wind_speed_3pm}</span></div>',
            unsafe_allow_html=True,
        )

        # ─────────────────────────
        sunshine_val = st.slider(
            "sunshine",
            0.0,
            15.0,
            0.0,
            0.1,
            label_visibility="collapsed",
            on_change=make_touch_callback("sunshine_touched"),
            key="sunshine_slider",
        )
        sunshine = sunshine_val if st.session_state["sunshine_touched"] else None
        display_sunshine = (
            f"{sunshine_val}h" if st.session_state["sunshine_touched"] else "—"
        )
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-top:-32px;margin-bottom:8px;pointer-events:none;">'
            f'<span style="font-size:0.6rem;font-weight:700;letter-spacing:1.8px;'
            f'text-transform:uppercase;color:#cbd5e0;">SUNSHINE</span>'
            f'<span style="font-size:0.75rem;font-weight:700;color:#3b82f6;">{display_sunshine}</span></div>',
            unsafe_allow_html=True,
        )

    location_target_encoded = {
        "Adelaide": 1.5663539307667422,
        "Albany": 2.2638594164456234,
        "Albury": 1.9141149119893721,
        "AliceSprings": 0.8828496042216359,
        "BadgerysCreek": 2.1931010928961747,
        "Ballarat": 1.7400264200792603,
        "Bendigo": 1.6193803559657218,
        "Brisbane": 3.144890857323632,
        "Cairns": 5.742034805890228,
        "Canberra": 1.7417203042715037,
        "Cobar": 1.1273092369477913,
        "CoffsHarbour": 5.061496782932611,
        "Dartmoor": 2.1465669612508496,
        "Darwin": 5.092452239273411,
        "GoldCoast": 3.7693959731543623,
        "Hobart": 1.601819322459222,
        "Katherine": 3.2010897435897436,
        "Launceston": 2.011988110964333,
        "Melbourne": 1.8700616016427105,
        "MelbourneAirport": 1.4519774011299436,
        "Mildura": 0.9450615231127371,
        "Moree": 1.6302032235459005,
        "MountGambier": 2.087561860772022,
        "MountGinini": 3.292260061919505,
        "Newcastle": 3.183891708967851,
        "Nhil": 0.9348629700446144,
        "NorahHead": 3.387299419597132,
        "NorfolkIsland": 3.127665317139001,
        "Nuriootpa": 1.3903429903429902,
        "PearceRAAF": 1.66908037653874,
        "Penrith": 2.1753036437246966,
        "Perth": 1.906295020357031,
        "PerthAirport": 1.761648388168827,
        "Portland": 2.5303738317757007,
        "Richmond": 2.1384615384615384,
        "Sale": 1.5101666666666667,
        "SalmonGums": 1.0343824027072759,
        "Sydney": 3.324543002697033,
        "SydneyAirport": 3.009916805324459,
        "Townsville": 3.485591823277283,
        "Tuggeranong": 2.1640426951300866,
        "Uluru": 0.7843626806833114,
        "WaggaWagga": 1.7099462365591398,
        "Walpole": 2.9068463994324225,
        "Watsonia": 1.860820273424475,
        "Williamtown": 3.591108499804152,
        "Witchcliffe": 2.8956639566395665,
        "Wollongong": 3.594902749832327,
        "Woomera": 0.49040454697425606,
    }

    with col_temp:
        st.markdown(
            """
        <div style="background:#161b22;border:1px solid #30363d;
                    border-radius:14px;padding:16px 20px 12px 20px;margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="background:rgba(59,130,246,0.18);border-radius:9px;
                            width:36px;height:36px;display:flex;align-items:center;
                            justify-content:center;font-size:1rem;">📅</div>
                <span style="font-size:1.5rem;font-weight:800;letter-spacing:2.5px;
                             text-transform:uppercase;color:#f1f5f9;">TEMPORAL</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        col_d, col_m = st.columns(2)
        with col_d:
            day = st.number_input(
                "DAY",
                min_value=1,
                max_value=31,
                value=datetime.date.today().day,
                step=1,
            )
        with col_m:
            month = st.number_input(
                "MONTH",
                min_value=1,
                max_value=12,
                value=datetime.date.today().month,
                step=1,
            )

        # ── Location + Predict moved here ────────────────────────────
        st.markdown("<div style='margin-top:12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background:#161b22;border:1px solid #30363d;
                        border-radius:14px;padding:16px 20px 12px 20px;margin-bottom:8px;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <div style="background:rgba(59,130,246,0.18);border-radius:9px;
                                width:36px;height:36px;display:flex;align-items:center;
                                justify-content:center;font-size:1rem;">📍</div>
                    <span style="font-size:1rem;font-weight:800;letter-spacing:2px;
                                text-transform:uppercase;color:#f1f5f9;">OBSERVATION LOCATION</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_location = st.selectbox(
            "Select Location:",
            list(location_target_encoded.keys()),
            label_visibility="collapsed",
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("⚡  Generate Forecast")

    location_encoded = location_target_encoded[selected_location]
    # ── col_result: empty or use for live result preview ────────────────
    st.markdown(
        """
        <div style="margin-bottom:20px;">
            <h1 style="font-size:2.0rem;font-weight:800;color:#f1f5f9;margin:0 0 4px;">
                Forecast Output
            </h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not predict_clicked:
        st.markdown(
            """
            <div style="background:rgba(14,18,30,0.4);border:1px solid rgba(255,255,255,0.05);
                        border-radius:16px;padding:24px;text-align:center;color:#3b82f6;
                        font-size:1.1rem;letter-spacing:3px;margin-top:40px;">
                Awaiting Input:<br> Configure variables and click generate to see the forecast.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Derived features ─────────────────────────────────────────────────

    if all(
        v is not None
        for v in [
            min_temp,
            max_temp,
            temp_9am,
            temp_3pm,
            pressure_9am,
            pressure_3pm,
            humidity_9am,
            humidity_3pm,
            evaporation,
            wind_gust_speed,
            wind_speed_9am,
            wind_speed_3pm,
            sunshine,
        ]
    ):
        dew_point_estimate = min_temp * (humidity_9am / 100)
        avg_humidity = (humidity_9am + humidity_3pm) / 2
        avg_temperature = (min_temp + max_temp) / 2
        temp_range = max_temp - min_temp
        temp_sunshine_interaction = max_temp * sunshine
        pressure_difference = pressure_3pm - pressure_9am
        temp_difference = temp_3pm - temp_9am
    else:
        dew_point_estimate = avg_humidity = avg_temperature = 0
        temp_range = temp_sunshine_interaction = pressure_difference = (
            temp_difference
        ) = 0

    feature_names = [
        "MinTemp",
        "MaxTemp",
        "Evaporation",
        "WindGustSpeed",
        "WindSpeed9am",
        "WindSpeed3pm",
        "Humidity9am",
        "Humidity3pm",
        "Pressure9am",
        "Pressure3pm",
        "Temp9am",
        "Temp3pm",
        "Month",
        "Day",
        "Location_Target_Encoded",
        "DewPointEstimate",
        "AvgHumidity",
        "AvgTemperature",
        "TempRange",
        "TempSunshineInteraction",
        "PressureDifference",
        "TempDifference",
    ]
    input_features = np.array(
        [
            [
                min_temp,
                max_temp,
                evaporation,
                wind_gust_speed,
                wind_speed_9am,
                wind_speed_3pm,
                humidity_9am,
                humidity_3pm,
                pressure_9am,
                pressure_3pm,
                temp_9am,
                temp_3pm,
                month,
                day,
                location_encoded,
                dew_point_estimate,
                avg_humidity,
                avg_temperature,
                temp_range,
                temp_sunshine_interaction,
                pressure_difference,
                temp_difference,
            ]
        ]
    )
    input_features_df = pd.DataFrame(input_features, columns=feature_names)

    display_mapping = {
        "Rainfall_Category_No Rain": "No Rain",
        "Rainfall_Category_Moderate Rain": "Moderate Rain",
        "Rainfall_Category_Heavy Rain": "Heavy Rain",
        "Rainfall_Category_Very Heavy Rain": "Very Heavy Rain",
    }
    range_mapping = {
        "Rainfall_Category_No Rain": "0 mm to 5 mm",
        "Rainfall_Category_Moderate Rain": "5 mm to 20 mm",
        "Rainfall_Category_Heavy Rain": "20 mm to 50 mm",
        "Rainfall_Category_Very Heavy Rain": "50 mm and higher",
    }

    if predict_clicked:
        meta_features = np.zeros((1, len(base_models)))
        for i, model in enumerate(base_models):
            meta_features[0, i] = model.predict(input_features_df)[0]

        meta_pred = meta_ann.predict(meta_features)
        meta_pred_class = np.argmax(meta_pred, axis=1)[0]
        meta_pred_label = category_mapping[meta_pred_class]
        confidence = int(np.max(meta_pred) * 100)

        icon_mapping = {
            "Rainfall_Category_No Rain": get_base64_image("assets/rain1.png"),
            "Rainfall_Category_Moderate Rain": get_base64_image("assets/rain2.png"),
            "Rainfall_Category_Heavy Rain": get_base64_image("assets/rain3.png"),
            "Rainfall_Category_Very Heavy Rain": get_base64_image("assets/rain4.png"),
        }
        icon_file = icon_mapping[meta_pred_label]

        category_info = {
            "Rainfall_Category_No Rain": [
                "☀️ Dry weather, no precipitation.",
                "✅ Ideal for outdoor activities.",
            ],
            "Rainfall_Category_Light Rain": [
                "🌦 A gentle drizzle.",
                "🟡 Minimal disruption to daily activities.",
            ],
            "Rainfall_Category_Moderate Rain": [
                "🌧 Consistent moderate rainfall with overcast skies.",
                "⚠️ Minor disruptions to outdoor activities. Umbrellas recommended.",
            ],
            "Rainfall_Category_Heavy Rain": [
                "⛈ Intense rainfall with strong winds.",
                "🔴 Stay indoors. Potential flooding in low-lying areas.",
            ],
            "Rainfall_Category_Very Heavy Rain": [
                "🌊 Unusually heavy downpour.",
                "🚨 High risk of flash flooding. Stay alert.",
            ],
        }
        bullets = category_info.get(meta_pred_label, ["No description available."])
        bullets_html = "".join(
            [
                f'<div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:8px;"><span style="font-size:0.8rem;color:rgba(255,255,255,0.7);">{b}</span></div>'
                for b in bullets
            ]
        )

        col_res1, col_res2 = st.columns([1, 1])

        with col_res1:
            st.markdown(
                f"""
            <div style="background:linear-gradient(135deg,#2563eb,#1d4ed8);
                        border-radius:16px;padding:24px;color:white;margin-top:16px;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:20px;">
                    <img src="data:image/png;base64,{icon_file}" width="52"
                         style="background:rgba(255,255,255,0.15);border-radius:10px;padding:6px;"/>
                    <div style="text-align:right;">
                        <div style="font-size:0.6rem;font-weight:700;letter-spacing:2px;opacity:0.7;">CONFIDENCE</div>
                        <div style="font-size:2rem;font-weight:800;line-height:1;">{confidence}%</div>
                    </div>
                </div>
                <div style="font-size:1rem;font-weight:700;letter-spacing:2px;opacity:0.7;margin-bottom:4px;">TODAY PRECIPITATION FORECAST</div>
                <div style="font-size:1.7rem;font-weight:800;margin-bottom:2px;">{display_mapping[meta_pred_label]}</div>
                <div style="font-size:1rem;opacity:0.7;margin-bottom:18px;">{range_mapping[meta_pred_label]}</div>
                <div style="border-top:1px solid rgba(255,255,255,0.2);padding-top:14px;">
                    {bullets_html}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col_res2:
            st.markdown(
                """
                <div style="background:rgba(14,18,30,0.92);border:1px solid rgba(255,255,255,0.07);
                            border-radius:16px;padding:24px;margin-top:16px;">
                    <div style="font-size:1.2rem;font-weight:700;letter-spacing:2px;
                                text-transform:uppercase;color:#64748b;margin-bottom:12px;">
                        SAFETY PROTOCOL
                    </div>
                    <p style="color:#94a3b8;font-size:1rem;line-height:1.7;margin-bottom:12px;">
                        Forecasts are based on historical patterns. For emergency situations,
                        always prioritize official warnings from the Bureau of Meteorology.
                    </p>
                    <ul style="color:#94a3b8;font-size:1rem;line-height:1.8;padding-left:16px;margin-bottom:16px;">
                        <li>Rainfall patterns vary across Australia's diverse climate zones.</li>
                        <li>Heavy rain can be dangerous — always follow evacuation orders.</li>
                    </ul>
                    <a href="https://bom.gov.au/" target="_blank"
                    style="color:#3b82f6;font-size:1rem;font-weight:600;text-decoration:none;">
                        View Official Alerts →
                    </a>
                </div>
                """,
                unsafe_allow_html=True,
            )

# Tab 3:
with tab3:
    st.title("Historical Analytics Dashboard:")
    st.write("Deep dive into regional meteorological trends.")
    # Filters Section
    st.subheader("")
    col1, col2 = st.columns(2)

    # Add date range slider
    with col1:
        st.markdown(
            """
        <div style="background:#161b22;border:1px solid #30363d;
                    border-radius:14px;padding:16px 20px 12px 20px;margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="background:rgba(59,130,246,0.18);border-radius:9px;
                            width:36px;height:36px;display:flex;align-items:center;
                            justify-content:center;font-size:1rem;">🗓️</div>
                <span style="font-size:1.5rem;font-weight:800;letter-spacing:2.5px;
                             text-transform:uppercase;color:#f1f5f9;">Filter by Date Range</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        # styled_section("Filter by Date Range", "#f0f8ff")
        data["Date"] = pd.to_datetime(data[["Year", "Month", "Day"]])
        min_date = data["Date"].min().date()  # Extract the earliest date
        max_date = data["Date"].max().date()  # Extract the latest date
        date_range = st.slider(
            "Select Date Range:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
        )
        filtered_data = data[
            (data["Date"] >= pd.to_datetime(date_range[0]))
            & (data["Date"] <= pd.to_datetime(date_range[1]))
        ]

    # Ensure Location_Target_Encoded column exists
    from sklearn.preprocessing import LabelEncoder

    if "Location_Target_Encoded" not in data.columns:
        le = LabelEncoder()
        data["Location_Target_Encoded"] = le.fit_transform(data["Location"])

    # Create a mapping of location names to their encoded values
    location_target_encoded = dict(
        zip(data["Location"], data["Location_Target_Encoded"])
    )

    # Location Dropdown in the second column
    with col2:
        st.markdown(
            """
        <div style="background:#161b22;border:1px solid #30363d;
                    border-radius:14px;padding:16px 20px 12px 20px;margin-bottom:8px;">
            <div style="display:flex;align-items:center;gap:10px;">
                <div style="background:rgba(59,130,246,0.18);border-radius:9px;
                            width:36px;height:36px;display:flex;align-items:center;
                            justify-content:center;font-size:1rem;">📍</div>
                <span style="font-size:1.5rem;font-weight:800;letter-spacing:2.5px;
                             text-transform:uppercase;color:#f1f5f9;">Filter by Location</span>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # List of all locations
        all_locations = ["All"] + [
            "Adelaide",
            "Albany",
            "Albury",
            "AliceSprings",
            "BadgerysCreek",
            "Ballarat",
            "Bendigo",
            "Brisbane",
            "Cairns",
            "Canberra",
            "Cobar",
            "CoffsHarbour",
            "Dartmoor",
            "Darwin",
            "GoldCoast",
            "Hobart",
            "Katherine",
            "Launceston",
            "Melbourne",
            "MelbourneAirport",
            "Mildura",
            "Moree",
            "MountGambier",
            "MountGinini",
            "Newcastle",
            "Nhil",
            "NorahHead",
            "NorfolkIsland",
            "Nuriootpa",
            "PearceRAAF",
            "Penrith",
            "Perth",
            "PerthAirport",
            "Portland",
            "Richmond",
            "Sale",
            "SalmonGums",
            "Sydney",
            "SydneyAirport",
            "Townsville",
            "Tuggeranong",
            "Uluru",
            "WaggaWagga",
            "Walpole",
            "Watsonia",
            "Williamtown",
            "Witchcliffe",
            "Wollongong",
            "Woomera",
        ]

        # Dropdown to select location
        selected_location = st.selectbox(
            "Select Location:", options=all_locations, key="location_selectbox"
        )

        # Filter data by selected location
        if selected_location == "All":
            st.markdown("### Showing data for all locations.")
        else:
            st.markdown(f"### Showing data for: **{selected_location}**")
            filtered_data = filtered_data[
                filtered_data["Location"] == selected_location
            ]

        # Scorecard Section
    st.markdown(
        """
    <div style="background:#161b22;border:1px solid #30363d;
                border-radius:14px;padding:16px 20px 12px 20px;margin-bottom:8px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <div style="background:rgba(59,130,246,0.18);border-radius:9px;
                        width:36px;height:36px;display:flex;align-items:center;
                        justify-content:center;font-size:1rem;">📈</div>
            <span style="font-size:1.5rem;font-weight:800;letter-spacing:2.5px;
                            text-transform:uppercase;color:#f1f5f9;">Metrics</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    if not filtered_data.empty:
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Min Temperature (°C)", filtered_data["MinTemp"].min())
        col2.metric("Max Temperature (°C)", filtered_data["MaxTemp"].max())
        col3.metric(
            "Average Temperature (°C)",
            f"{(filtered_data['MinTemp'].mean() + filtered_data['MaxTemp'].mean()) / 2:.1f}",
        )
        col4.metric("Total Rainfall (mm)", f"{filtered_data['Rainfall'].sum():.1f}")
        col5.metric("Max Rainfall (mm)", filtered_data["Rainfall"].max())
        col6.metric(
            "Raining Days",
            len(
                filtered_data[
                    filtered_data["Rainfall_Category"].isin(
                        ["Moderate Rain", "Heavy Rain", "Very Heavy Rain"]
                    )
                ]
            ),
        )
    else:
        st.write("No data available for the selected filters.")

    # Line Graphs Section

    if not filtered_data.empty:
        # Arrange graphs in a grid
        col1, col2 = st.columns(2)

        # Rainfall Graph
        with col1:
            fig1 = px.line(
                filtered_data, x="Date", y="Rainfall", title="Rainfall Over Time"
            )
            st.plotly_chart(fig1, use_container_width=True)

        # MinTemp & MaxTemp Graph
        with col2:
            fig2 = px.line(
                filtered_data,
                x="Date",
                y=["MinTemp", "MaxTemp"],
                title="Min & Max Temperatures Over Time",
            )
            st.plotly_chart(fig2, use_container_width=True)

        # WindGustSpeed Graph
        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.line(
                filtered_data,
                x="Date",
                y="WindGustSpeed",
                title="Wind Gust Speed Over Time",
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Humidity Graph
        with col2:
            fig4 = px.line(
                filtered_data,
                x="Date",
                y=["Humidity9am", "Humidity3pm"],
                title="Humidity at 9am & 3pm Over Time",
            )
            st.plotly_chart(fig4, use_container_width=True)

        # Pressure Graph
        col1, col2 = st.columns(2)
        with col1:
            fig5 = px.line(
                filtered_data,
                x="Date",
                y=["Pressure9am", "Pressure3pm"],
                title="Pressure at 9am & 3pm Over Time",
            )
            st.plotly_chart(fig5, use_container_width=True)

        # Filtered Data Table
        with col2:
            st.markdown(
                """
            <div style="background:#161b22;border:1px solid #30363d;
                        border-radius:14px;padding:16px 20px 12px 20px;margin-bottom:8px;">
                <div style="display:flex;align-items:center;gap:10px;">
                    <div style="background:rgba(59,130,246,0.18);border-radius:9px;
                                width:36px;height:36px;display:flex;align-items:center;
                                justify-content:center;font-size:1rem;">☰</div>
                    <span style="font-size:1.5rem;font-weight:800;letter-spacing:2.5px;
                                    text-transform:uppercase;color:#f1f5f9;">Filtered Data Table</span>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.dataframe(filtered_data)
    else:
        st.write("No data available to display graphs.")

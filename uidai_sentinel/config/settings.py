"""
Aadhaar Sentinel - Configuration Settings
==========================================
Centralized configuration for UI, thresholds, and system parameters.
"""
import os
from pathlib import Path

# ============================================
# PATH CONFIGURATION
# ============================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Data file paths
ENROLLMENT_CSV = os.getenv("ENROLLMENT_CSV", str(DATA_DIR / "Enrollment.csv"))
DEMOGRAPHIC_CSV = os.getenv("DEMOGRAPHIC_CSV", str(DATA_DIR / "Demographic.csv"))

# Model paths
LSTM_MODEL_PATH = MODEL_DIR / "sentinel_lstm.keras"
SCALER_PATH = MODEL_DIR / "sentinel_scaler.pkl"

# ============================================
# UI CONFIGURATION
# ============================================
APP_CONFIG = {
    "page_title": "Aadhaar Sentinel | UIDAI Operations Intelligence",
    "page_icon": "🛡️",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Color Palette - Government Dashboard Theme
COLORS = {
    "primary": "#1B4F72",          # Deep Navy Blue
    "secondary": "#2E86AB",        # Steel Blue  
    "accent": "#F39C12",           # Saffron Orange
    "success": "#27AE60",          # Green
    "warning": "#F1C40F",          # Yellow
    "danger": "#E74C3C",           # Red
    "background": "#0E1117",       # Dark Background
    "card_bg": "#1E2530",          # Card Background
    "text": "#ECF0F1",             # Light Text
    "muted": "#7F8C8D"             # Muted Text
}

# Plotly Theme Colors
PLOTLY_COLORS = {
    "normal": "#2E86AB",           # Blue for normal data
    "anomaly": "#E74C3C",          # Red for anomalies
    "forecast": "#F39C12",         # Orange for predictions
    "actual": "#27AE60",           # Green for actual values
    "confidence": "rgba(243, 156, 18, 0.2)"  # Forecast confidence band
}

# ============================================
# ANALYTICS THRESHOLDS
# ============================================
THRESHOLDS = {
    "migration_index_high": 2.0,       # Flag as Migration Hub if > 2.0
    "enrollment_low": 100,             # Deploy Mobile Camp if < threshold
    "update_enrollment_ratio": 2.0,    # Convert to Update Center if ratio > 2
    "anomaly_contamination": 0.05,     # 5% expected anomalies for Isolation Forest
    "youth_ratio_low": 0.20            # Low youth enrollment threshold
}

# Strategic Action Labels
ACTIONS = {
    "update_center": "🔴 Convert to Update Center",
    "mobile_camp": "🟡 Deploy Mobile Camp",
    "school_camp": "🟠 Deploy School-based Camp",
    "migration_hub": "🔵 Migration Hub Detected",
    "stable": "🟢 Operations Normal"
}

# ============================================
# LSTM MODEL CONFIGURATION
# ============================================
LSTM_CONFIG = {
    "sequence_length": 30,          # Days of history for prediction
    "forecast_days": 30,            # Days to forecast ahead
    "lstm_units": 64,
    "dense_units": 32,
    "dropout_rate": 0.2,
    "epochs": 50,
    "batch_size": 32,
    "validation_split": 0.2,
    "learning_rate": 0.001
}

# ============================================
# MAP CONFIGURATION
# ============================================
# Approximate coordinates for Indian State Capitals
STATE_COORDINATES = {
    "Andhra Pradesh": (15.9129, 79.7400),
    "Arunachal Pradesh": (27.0844, 93.6053),
    "Assam": (26.2006, 92.9376),
    "Bihar": (25.0961, 85.3131),
    "Chhattisgarh": (21.2787, 81.8661),
    "Goa": (15.2993, 74.1240),
    "Gujarat": (22.2587, 71.1924),
    "Haryana": (29.0588, 76.0856),
    "Himachal Pradesh": (31.1048, 77.1734),
    "Jharkhand": (23.6102, 85.2799),
    "Karnataka": (15.3173, 75.7139),
    "Kerala": (10.8505, 76.2711),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Maharashtra": (19.7515, 75.7139),
    "Manipur": (24.6637, 93.9063),
    "Meghalaya": (25.4670, 91.3662),
    "Mizoram": (23.1645, 92.9376),
    "Nagaland": (26.1584, 94.5624),
    "Odisha": (20.9517, 85.0985),
    "Punjab": (31.1471, 75.3412),
    "Rajasthan": (27.0238, 74.2179),
    "Sikkim": (27.5330, 88.5122),
    "Tamil Nadu": (11.1271, 78.6569),
    "Telangana": (18.1124, 79.0193),
    "Tripura": (23.9408, 91.9882),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Uttarakhand": (30.0668, 79.0193),
    "West Bengal": (22.9868, 87.8550),
    "Delhi": (28.7041, 77.1025),
    "Jammu and Kashmir": (33.7782, 76.5762),
    "Ladakh": (34.1526, 77.5771)
}

# Map defaults
MAP_CONFIG = {
    "center": [22.5, 78.9],        # India center
    "zoom": 5,
    "tiles": "CartoDB dark_matter"
}

# ============================================
# DATE FORMATS
# ============================================
DATE_FORMATS = [
    "%Y-%m-%d",
    "%d-%m-%Y", 
    "%d/%m/%Y",
    "%Y/%m/%d",
    "%d-%b-%Y"
]

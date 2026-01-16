"""
Configuration settings for UIDAI Aadhaar Intelligence System.
All paths and hyperparameters are centralized here for easy management.
"""
import os
from pathlib import Path

# ============================================
# PATH CONFIGURATION
# ============================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Data file paths (can be overridden via environment variables)
ENROLLMENT_CSV = os.getenv("ENROLLMENT_CSV_PATH", str(DATA_DIR / "Enrollment.csv"))
DEMOGRAPHIC_CSV = os.getenv("DEMOGRAPHIC_CSV_PATH", str(DATA_DIR / "Demographic.csv"))

# Model file path
MODEL_PATH = os.getenv("MODEL_PATH", str(MODEL_DIR / "uidai_lstm_v1.keras"))
SCALER_PATH = os.getenv("SCALER_PATH", str(MODEL_DIR / "scaler.pkl"))

# ============================================
# LSTM MODEL HYPERPARAMETERS
# ============================================
LSTM_CONFIG = {
    "sequence_length": 12,       # Months of history to use for prediction
    "forecast_horizon": 6,       # Months to forecast ahead
    "lstm_units": 64,            # Number of LSTM units
    "dense_units": 32,           # Dense layer units
    "dropout_rate": 0.2,         # Dropout for regularization
    "epochs": 100,               # Training epochs
    "batch_size": 32,            # Batch size
    "validation_split": 0.2,     # Validation data percentage
    "early_stopping_patience": 10,  # Early stopping patience
    "learning_rate": 0.001       # Adam optimizer learning rate
}

# ============================================
# DATA VALIDATION SCHEMA
# ============================================
ENROLLMENT_REQUIRED_COLUMNS = [
    "date", "state", "district", "age_0_5", "age_5_17", "age_18_greater"
]

DEMOGRAPHIC_REQUIRED_COLUMNS = [
    "date", "state", "district", "demo_age_5_17", "demo_age_17_"
]

# Alternative column names mapping
COLUMN_ALIASES = {
    "dates": "date",
    "Date": "date",
    "State": "state",
    "District": "district"
}

# ============================================
# INTERVENTION THRESHOLDS
# ============================================
INTERVENTION_CONFIG = {
    "youth_ratio_threshold": 0.20,       # Below this triggers school camps
    "update_enrollment_ratio": 2.0,      # Above this triggers update-only center
}

# ============================================
# MAP CONFIGURATION
# ============================================
MAP_CONFIG = {
    "default_location": [22.5, 78.9],    # India center coordinates
    "default_zoom": 5,
    "tile_style": "CartoDB positron"
}

# ============================================
# UI CONFIGURATION
# ============================================
APP_CONFIG = {
    "page_title": "UIDAI Aadhaar Intelligence System",
    "page_icon": "🆔",
    "layout": "wide"
}

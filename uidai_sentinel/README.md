# 🛡️ Aadhaar Sentinel

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.18-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Folium](https://img.shields.io/badge/Folium-0.15-77B829?style=for-the-badge&logo=folium&logoColor=white)

### **UIDAI Operations Intelligence Dashboard**
*From Descriptive Analytics to Prescriptive Intelligence*

[Features](#-features) • [Quick Start](#-quick-start) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure)

---

**🏆 Built for UIDAI Data Hackathon 2026**

</div>

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [Quick Start](#-quick-start)
4. [Installation](#-installation)
5. [How to Run](#-how-to-run)
6. [Project Structure](#-project-structure)
7. [Data Format](#-data-format)
8. [Analytics Logic](#-analytics-logic)
9. [AI/ML Models](#-aiml-models)
10. [Configuration](#-configuration)
11. [API Reference](#-api-reference)
12. [Troubleshooting](#-troubleshooting)
13. [License](#-license)

---

## 🎯 Overview

**Aadhaar Sentinel** is a production-grade Streamlit dashboard that transforms raw Aadhaar enrollment and demographic data into actionable, prescriptive insights. The application helps UIDAI administrators:

- 📍 **Identify intervention hotspots** with interactive geospatial mapping
- 🔍 **Detect data anomalies** using AI-powered Isolation Forest algorithm
- 🔮 **Forecast enrollment trends** with LSTM deep learning models
- 📊 **Make data-driven decisions** through strategic action recommendations

---

## ✨ Features

### 🗺️ Tab 1: Strategic Intervention Map

| Feature | Description |
|---------|-------------|
| **Interactive India Map** | Folium-based map with clustered district markers |
| **Color-Coded Markers** | 🔴 Critical, 🟡 Warning, 🟠 Moderate, 🟢 Normal |
| **Click Popups** | Detailed district metrics on marker click |
| **Intervention Legend** | Clear visual guide for action types |
| **Action Report** | Downloadable CSV of districts needing intervention |

### 🔍 Tab 2: Data Integrity Monitor

| Feature | Description |
|---------|-------------|
| **Isolation Forest AI** | Detects statistical outliers (5% contamination) |
| **Scatter Visualization** | Blue dots = Normal, Red dots = Anomalies |
| **Distribution Histogram** | Enrollment frequency analysis |
| **Anomaly Table** | Detailed breakdown of flagged dates |
| **Real-time Metrics** | Days analyzed, anomaly count, rate percentage |

### 🔮 Tab 3: Future Forecast

| Feature | Description |
|---------|-------------|
| **LSTM Neural Network** | Deep learning time-series forecasting |
| **Exponential Smoothing** | Fallback method when LSTM unavailable |
| **Confidence Intervals** | ±15% prediction bounds |
| **Adjustable Horizon** | 7 to 60 days forecast slider |
| **Summary Cards** | Average, Total, Change %, Peak predictions |

### 📊 Dashboard Metrics (Top Row)

| Metric | Description |
|--------|-------------|
| **Total Enrollments** | Sum of all age groups with weekly delta |
| **Total Updates** | Demographic updates with weekly delta |
| **Migration Index** | Updates/Enrollments ratio (threshold: 2.0) |
| **Youth Ratio** | Percentage of enrollments ages 0-17 |

---

## 🚀 Quick Start

### One-Command Setup

```bash
# Clone and run
cd D:\UIDAI_HACK\uidai_sentinel
pip install -r requirements.txt
streamlit run app.py
```

### Open in Browser

```
http://localhost:8501
```

**That's it!** The dashboard loads with sample data automatically.

---

## 💻 Installation

### Prerequisites

| Requirement | Version |
|-------------|---------|
| Python | 3.9+ |
| pip | Latest |
| RAM | 4GB minimum |
| OS | Windows/Linux/macOS |

### Step 1: Navigate to Project

```bash
cd D:\UIDAI_HACK\uidai_sentinel
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "import streamlit; import tensorflow; import sklearn; import folium; print('✅ All dependencies installed!')"
```

---

## 🏃 How to Run

### Basic Run

```bash
streamlit run app.py
```

### Custom Port

```bash
streamlit run app.py --server.port 8502
```

### Headless Mode (Servers)

```bash
streamlit run app.py --server.headless true
```

### Network Accessible

```bash
streamlit run app.py --server.address 0.0.0.0
```

### Full Command with All Options

```bash
streamlit run app.py --server.port 8502 --server.address 0.0.0.0 --server.headless true
```

---

## 📁 Project Structure

```
uidai_sentinel/
│
├── 📄 app.py                    # Main Streamlit dashboard (800+ lines)
│                                # - Page configuration
│                                # - Custom CSS styling
│                                # - Sidebar filters
│                                # - Metric cards
│                                # - Three tabs implementation
│
├── 📄 requirements.txt          # Python dependencies
│
├── 📄 README.md                 # This documentation
│
├── 📁 config/
│   ├── __init__.py
│   └── settings.py              # Configuration settings
│                                # - COLORS: UI color palette
│                                # - THRESHOLDS: Analytics thresholds
│                                # - LSTM_CONFIG: Model hyperparameters
│                                # - STATE_COORDINATES: India state lat/longs
│                                # - MAP_CONFIG: Folium defaults
│
├── 📁 utils/
│   ├── __init__.py
│   │
│   ├── data_loader.py           # Data loading & preprocessing
│   │                            # - standardize_columns()
│   │                            # - parse_dates()
│   │                            # - load_enrollment_data()
│   │                            # - load_demographic_data()
│   │                            # - get_merged_data()
│   │                            # - get_filter_options()
│   │
│   ├── analytics.py             # Business logic & metrics
│   │                            # - calculate_migration_index()
│   │                            # - calculate_youth_ratio()
│   │                            # - get_strategic_action()
│   │                            # - calculate_metrics()
│   │                            # - get_district_analysis()
│   │                            # - get_anomaly_summary()
│   │
│   ├── ai_engine.py             # Machine learning models
│   │                            # - AnomalyDetector (Isolation Forest)
│   │                            # - EnrollmentForecaster (LSTM)
│   │                            # - quick_forecast() (Exponential Smoothing)
│   │                            # - detect_anomalies_simple() (Z-score fallback)
│   │
│   └── maps.py                  # Geospatial visualization
│                                # - get_state_coordinates()
│                                # - get_district_coordinates()
│                                # - get_marker_color()
│                                # - create_popup_html()
│                                # - create_intervention_map()
│                                # - add_legend()
│
├── 📁 data/
│   ├── Enrollment.csv           # Aadhaar enrollment data (73,050 records)
│   └── Demographic.csv          # Demographic update data
│
└── 📁 models/
    ├── sentinel_lstm.keras      # Trained LSTM model (when saved)
    └── sentinel_scaler.pkl      # Fitted MinMaxScaler
```

---

## 📊 Data Format

### Enrollment.csv

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | datetime | Record date (YYYY-MM-DD) | 2024-01-15 |
| `state` | string | Indian state name | Maharashtra |
| `district` | string | District name | Mumbai |
| `age_0_5` | integer | Enrollments age 0-5 years | 150 |
| `age_5_17` | integer | Enrollments age 5-17 years | 280 |
| `age_18_greater` | integer | Enrollments age 18+ years | 420 |

**Sample Data:**
```csv
date,state,district,age_0_5,age_5_17,age_18_greater
2024-01-01,Maharashtra,Mumbai,145,267,398
2024-01-01,Maharashtra,Pune,132,245,367
2024-01-02,Maharashtra,Mumbai,158,289,412
```

### Demographic.csv

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | datetime | Record date | 2024-01-15 |
| `state` | string | State name | Maharashtra |
| `district` | string | District name | Mumbai |
| `demo_age_5_17` | integer | Updates age 5-17 | 95 |
| `demo_age_17_` | integer | Updates age 17+ | 180 |

**Sample Data:**
```csv
date,state,district,demo_age_5_17,demo_age_17_
2024-01-01,Maharashtra,Mumbai,89,165
2024-01-01,Maharashtra,Pune,78,142
```

---

## 🧠 Analytics Logic

### Migration Index Formula

```python
Migration_Index = Total_Updates / (Total_Enrollment + 1)
```

| Value | Interpretation |
|-------|----------------|
| > 2.0 | 🔵 Migration Hub - High demographic churn |
| 1.0 - 2.0 | 🟡 Moderate activity |
| < 1.0 | 🟢 Normal operations |

### Youth Ratio Formula

```python
Youth_Ratio = (age_0_5 + age_5_17) / Total_Enrollment
```

| Value | Interpretation |
|-------|----------------|
| < 20% | 🟠 Low youth enrollment - Deploy school camps |
| 20% - 50% | 🟢 Balanced enrollment |
| > 50% | 🔵 Youth-heavy area |

### Strategic Action Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│                  STRATEGIC ACTION LOGIC                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  IF Updates > 2 × Enrollments                               │
│      └─→ 🔴 Convert to Update Center                        │
│                                                             │
│  ELSE IF Enrollments < 100/day                              │
│      └─→ 🟡 Deploy Mobile Camp                              │
│                                                             │
│  ELSE IF Youth_Ratio < 20%                                  │
│      └─→ 🟠 Deploy School-based Camp                        │
│                                                             │
│  ELSE IF Migration_Index > 2.0                              │
│      └─→ 🔵 Migration Hub Detected                          │
│                                                             │
│  ELSE                                                       │
│      └─→ 🟢 Operations Normal                               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 AI/ML Models

### 1. Anomaly Detection - Isolation Forest

```python
from sklearn.ensemble import IsolationForest

model = IsolationForest(
    contamination=0.05,      # Expect 5% anomalies
    n_estimators=100,        # Number of trees
    random_state=42,
    n_jobs=-1                # Use all CPU cores
)
```

**How It Works:**
1. Randomly selects features and split values
2. Builds isolation trees
3. Anomalies = fewer splits needed to isolate
4. Returns: -1 (anomaly), 1 (normal)

### 2. Forecasting - LSTM Neural Network

```
┌─────────────────────────────────────────────────────────────┐
│                  LSTM MODEL ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input Layer (30 timesteps, 1 feature)                      │
│           ↓                                                 │
│  Bidirectional LSTM (64 units)                              │
│           ↓                                                 │
│  Dropout (0.2)                                              │
│           ↓                                                 │
│  LSTM (32 units)                                            │
│           ↓                                                 │
│  Dropout (0.2)                                              │
│           ↓                                                 │
│  Dense (32 units, ReLU)                                     │
│           ↓                                                 │
│  Dropout (0.1)                                              │
│           ↓                                                 │
│  Output Layer (1 unit)                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Training Parameters:**
| Parameter | Value |
|-----------|-------|
| Sequence Length | 30 days |
| Epochs | 50 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Validation Split | 20% |
| Early Stopping | 10 epochs patience |

---

## ⚙️ Configuration

### Edit `config/settings.py`

#### Color Palette
```python
COLORS = {
    "primary": "#1B4F72",      # Deep Navy Blue
    "secondary": "#2E86AB",    # Steel Blue  
    "accent": "#F39C12",       # Saffron Orange
    "success": "#27AE60",      # Green
    "warning": "#F1C40F",      # Yellow
    "danger": "#E74C3C",       # Red
}
```

#### Analytics Thresholds
```python
THRESHOLDS = {
    "migration_index_high": 2.0,       # Flag as Migration Hub
    "enrollment_low": 100,             # Deploy Mobile Camp
    "update_enrollment_ratio": 2.0,    # Convert to Update Center
    "anomaly_contamination": 0.05,     # 5% expected anomalies
    "youth_ratio_low": 0.20            # 20% threshold
}
```

#### LSTM Configuration
```python
LSTM_CONFIG = {
    "sequence_length": 30,
    "forecast_days": 30,
    "lstm_units": 64,
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001
}
```

---

## 📚 API Reference

### Data Loader Functions

```python
from utils.data_loader import load_enrollment_data, get_merged_data

# Load enrollment data
df = load_enrollment_data("data/Enrollment.csv")

# Merge enrollment and demographic data
merged = get_merged_data(enrollment_df, demographic_df)
```

### Analytics Functions

```python
from utils.analytics import calculate_metrics, get_strategic_action

# Calculate dashboard metrics
metrics = calculate_metrics(df, state="Maharashtra")

# Get strategic action for a row
action = get_strategic_action(row)
```

### AI Engine Classes

```python
from utils.ai_engine import AnomalyDetector, EnrollmentForecaster

# Anomaly detection
detector = AnomalyDetector(contamination=0.05)
flags, scores = detector.fit_predict(df, ['Total_Enrollment'])

# Forecasting
forecaster = EnrollmentForecaster()
forecaster.fit(time_series)
forecast, lower, upper = forecaster.forecast(steps=30)
```

### Map Functions

```python
from utils.maps import create_intervention_map

# Create intervention map
map_obj = create_intervention_map(district_df, show_only_action_needed=True)
```

---

## 🔧 Troubleshooting

### Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Module not found** | `pip install -r requirements.txt` |
| **TensorFlow GPU errors** | `pip install tensorflow-cpu` |
| **Streamlit not found** | `python -m streamlit run app.py` |
| **Port already in use** | `streamlit run app.py --server.port 8502` |
| **Data not loading** | Check CSV files in `data/` folder |
| **Map not displaying** | `pip install folium streamlit-folium --upgrade` |
| **Memory error** | Reduce data size or use chunked loading |

### Debug Mode

```bash
streamlit run app.py --logger.level debug
```

---

## 📦 Dependencies

```
streamlit==1.29.0
pandas==2.1.4
numpy==1.26.3
scikit-learn==1.3.2
tensorflow==2.15.0
plotly==5.18.0
folium==0.15.1
streamlit-folium==0.15.1
python-dotenv==1.0.0
```

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">

### 🏆 Built with ❤️ for UIDAI Data Hackathon 2026

**Aadhaar Sentinel** | Operations Intelligence Dashboard

*Transforming Data into Actionable Insights*

---

[⬆ Back to Top](#-aadhaar-sentinel)

</div>

# 🆔 UIDAI Aadhaar Intelligence System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**AI-Powered Enrollment Forecasting & Strategic Intervention Dashboard**

[Features](#-features) • [Installation](#-installation) • [How to Run](#-how-to-run) • [Project Structure](#-project-structure) • [Screenshots](#-screenshots) • [API Reference](#-api-reference)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [How to Run](#-how-to-run)
- [Project Structure](#-project-structure)
- [Data Format](#-data-format)
- [Model Architecture](#-model-architecture)
- [Screenshots](#-screenshots)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

The **UIDAI Aadhaar Intelligence System** is a production-ready web application developed for the UIDAI Data Hackathon 2026. It transforms raw Aadhaar enrollment and demographic data into actionable insights through:

- **Deep Learning Forecasting**: LSTM neural networks for accurate enrollment predictions
- **Strategic Intervention Engine**: AI-driven recommendations for resource allocation
- **Interactive Visualizations**: Real-time dashboards with Plotly and Folium maps
- **Scalable Architecture**: Modular design with caching for high performance

---

## ✨ Features

### 📈 Enrollment Forecast Tab
- LSTM-based time series forecasting with confidence intervals
- Historical trend analysis with interactive charts
- Age distribution breakdown (0-5, 5-17, 18+ years)
- Customizable forecast horizon (1-12 months)

### 🎯 Intervention Strategy Tab
- District-level intervention recommendations
- Interactive geospatial map with Folium
- Priority-based action items (Critical, High, Low)
- Downloadable intervention reports (CSV)

### 📊 Raw Data Analysis Tab
- Data exploration with searchable tables
- State-wise enrollment comparison charts
- Data quality metrics and validation
- Export capabilities

### 🔧 Additional Features
- **Sidebar Filters**: Filter by State and District
- **Custom Data Upload**: Upload your own CSV files
- **Responsive Design**: Dark theme with modern UI
- **Performance Optimized**: Cached model loading and data processing

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Plotly, Folium |
| **Backend** | Python 3.9+ |
| **ML/DL** | TensorFlow/Keras, Scikit-learn |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly Express, Folium Maps |
| **Deployment** | Streamlit Cloud, Heroku, Docker |

---

## 💻 Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git (optional, for cloning)
- 4GB RAM minimum (for model training)

### Step-by-Step Installation

#### 1. Clone or Download the Repository

```bash
# Option A: Clone with Git
git clone https://github.com/your-username/uidai-aadhaar-system.git
cd uidai-aadhaar-system

# Option B: Or navigate to existing directory
cd D:\UIDAI_HACK
```

#### 2. Create a Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Verify Installation

```bash
python -c "import streamlit; import tensorflow; import plotly; print('All dependencies installed successfully!')"
```

---

## 🚀 How to Run

### Quick Start (3 Steps)

```bash
# Step 1: Generate sample data (if you don't have real data)
python -c "from utils.data_generator import save_sample_data; save_sample_data()"

# Step 2: Train the LSTM model
python train_model.py --generate-sample

# Step 3: Launch the dashboard
python -m streamlit run app.py
```

The application will open in your browser at **http://localhost:8501**

---

### Detailed Run Instructions

#### Option 1: Using Sample Data (Quick Demo)

```bash
# 1. Generate synthetic enrollment and demographic data
python -c "from utils.data_generator import save_sample_data; save_sample_data()"

# Output:
# [*] Generating Enrollment data...
# [OK] Saved: data\Enrollment.csv (73,050 records)
# [*] Generating Demographic data...
# [OK] Saved: data\Demographic.csv (73,050 records)

# 2. Train the LSTM forecasting model
python train_model.py --epochs 50

# 3. Start the Streamlit dashboard
python -m streamlit run app.py
```

#### Option 2: Using Your Own Data

1. **Prepare your CSV files** following the [Data Format](#-data-format) section

2. **Place files in the data folder**:
   ```
   data/
   ├── Enrollment.csv
   └── Demographic.csv
   ```

3. **Train the model on your data**:
   ```bash
   python train_model.py --data data/Enrollment.csv --epochs 100
   ```

4. **Launch the dashboard**:
   ```bash
   python -m streamlit run app.py
   ```

#### Option 3: Upload Data via Dashboard

1. Start the app without data:
   ```bash
   python -m streamlit run app.py
   ```

2. Click **"Upload Custom Data"** in the sidebar

3. Upload your `Enrollment.csv` and `Demographic.csv` files

---

### Command Line Options

#### Training Script (`train_model.py`)

```bash
python train_model.py [OPTIONS]

Options:
  --data, -d PATH       Path to Enrollment CSV file (default: data/Enrollment.csv)
  --output, -o PATH     Path to save trained model (default: models/uidai_lstm_v1.keras)
  --epochs, -e INT      Number of training epochs (default: 100)
  --generate-sample     Generate sample data if CSV not found
```

**Examples:**

```bash
# Train with default settings
python train_model.py

# Train with custom epochs
python train_model.py --epochs 150

# Train with custom data path
python train_model.py --data my_data/custom_enrollment.csv

# Generate sample data and train
python train_model.py --generate-sample --epochs 50
```

#### Streamlit App (`app.py`)

```bash
python -m streamlit run app.py [OPTIONS]

Options:
  --server.port PORT           Port to run the app (default: 8501)
  --server.address ADDRESS     Address to bind (default: localhost)
  --server.headless BOOL       Run in headless mode (default: false)
```

**Examples:**

```bash
# Run on default port
python -m streamlit run app.py

# Run on custom port
python -m streamlit run app.py --server.port 8080

# Run accessible from network
python -m streamlit run app.py --server.address 0.0.0.0

# Run in headless mode (for servers)
python -m streamlit run app.py --server.headless true
```

---

## 📁 Project Structure

```
UIDAI_HACK/
│
├── 📄 app.py                    # Main Streamlit dashboard application
├── 📄 train_model.py            # LSTM model training script
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # This documentation file
├── 📄 Procfile                  # Heroku deployment configuration
├── 📄 setup.sh                  # Automated setup script
├── 📄 runtime.txt               # Python version for cloud deployment
│
├── 📁 config/
│   ├── __init__.py
│   └── settings.py              # Centralized configuration & hyperparameters
│
├── 📁 models/
│   ├── uidai_lstm_v1.keras      # Trained LSTM model
│   └── scaler.pkl               # Fitted MinMaxScaler for preprocessing
│
├── 📁 data/
│   ├── Enrollment.csv           # Aadhaar enrollment data
│   └── Demographic.csv          # Demographic update data
│
├── 📁 utils/
│   ├── __init__.py
│   └── data_generator.py        # Sample data generation utility
│
└── 📁 .streamlit/
    └── config.toml              # Streamlit theme configuration
```

---

## 📊 Data Format

### Enrollment.csv

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | datetime | Record date (YYYY-MM-DD) | 2024-01-15 |
| `state` | string | State name | Maharashtra |
| `district` | string | District name | Mumbai |
| `age_0_5` | integer | Enrollments for age 0-5 years | 150 |
| `age_5_17` | integer | Enrollments for age 5-17 years | 280 |
| `age_18_greater` | integer | Enrollments for age 18+ years | 420 |

**Sample:**
```csv
date,state,district,age_0_5,age_5_17,age_18_greater
2024-01-01,Maharashtra,Mumbai,145,267,398
2024-01-01,Maharashtra,Pune,132,245,367
2024-01-02,Maharashtra,Mumbai,158,289,412
```

### Demographic.csv

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | datetime | Record date (YYYY-MM-DD) | 2024-01-15 |
| `state` | string | State name | Maharashtra |
| `district` | string | District name | Mumbai |
| `demo_age_5_17` | integer | Updates for age 5-17 years | 95 |
| `demo_age_17_` | integer | Updates for age 17+ years | 180 |

**Sample:**
```csv
date,state,district,demo_age_5_17,demo_age_17_
2024-01-01,Maharashtra,Mumbai,89,165
2024-01-01,Maharashtra,Pune,78,142
2024-01-02,Maharashtra,Mumbai,95,178
```

---

## 🧠 Model Architecture

### LSTM Network Structure

```
┌─────────────────────────────────────────────────────────┐
│              Input Layer (12 timesteps, 1 feature)      │
├─────────────────────────────────────────────────────────┤
│         Bidirectional LSTM (64 units, return_seq=True)  │
│                      + Dropout (0.2)                    │
├─────────────────────────────────────────────────────────┤
│                    LSTM (32 units)                      │
│                      + Dropout (0.2)                    │
├─────────────────────────────────────────────────────────┤
│                Dense (32 units, ReLU)                   │
│                      + Dropout (0.1)                    │
├─────────────────────────────────────────────────────────┤
│                 Output Layer (1 unit)                   │
└─────────────────────────────────────────────────────────┘
```

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sequence Length | 12 months | Historical window for prediction |
| Forecast Horizon | 6 months | Default prediction period |
| Epochs | 100 | Maximum training iterations |
| Batch Size | 32 | Samples per gradient update |
| Optimizer | Adam | Adaptive learning rate optimizer |
| Learning Rate | 0.001 | Initial learning rate |
| Loss Function | MSE | Mean Squared Error |
| Early Stopping | 10 epochs | Patience for validation loss |

### Intervention Logic

| Condition | Action | Rationale |
|-----------|--------|-----------|
| Youth Ratio < 20% | Deploy School-based Camps | Low child enrollment needs outreach |
| Updates > 2× Enrollments | Convert to Update-only Center | High update demand indicates mature coverage |
| Otherwise | Stable | No immediate intervention required |

---

## 📸 Screenshots

### Enrollment Forecast Dashboard
- Interactive time-series chart with LSTM predictions
- Confidence intervals for uncertainty quantification
- Age distribution pie chart

### Intervention Strategy Map
- Geospatial visualization of intervention hotspots
- Color-coded markers by action type
- Clickable popups with district details

### Raw Data Analysis
- Searchable data tables
- State-wise comparison bar charts
- Data quality metrics

---

## ☁️ Deployment

### Streamlit Cloud (Recommended)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set `app.py` as the main file
5. Deploy!

### Heroku

```bash
# Login to Heroku
heroku login

# Create a new app
heroku create uidai-aadhaar-system

# Deploy
git push heroku main

# Open the app
heroku open
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Build and run
docker build -t uidai-app .
docker run -p 8501:8501 uidai-app
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# Data paths (optional - defaults to data/ folder)
export ENROLLMENT_CSV_PATH=/path/to/Enrollment.csv
export DEMOGRAPHIC_CSV_PATH=/path/to/Demographic.csv

# Model paths (optional - defaults to models/ folder)
export MODEL_PATH=/path/to/model.keras
export SCALER_PATH=/path/to/scaler.pkl
```

### Hyperparameters (`config/settings.py`)

```python
LSTM_CONFIG = {
    "sequence_length": 12,        # Adjust historical window
    "forecast_horizon": 6,        # Adjust prediction period
    "lstm_units": 64,             # Model complexity
    "epochs": 100,                # Training iterations
    "batch_size": 32,             # Batch size
    "learning_rate": 0.001        # Learning rate
}
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. TensorFlow Import Error
```bash
# Solution: Reinstall TensorFlow
pip uninstall tensorflow
pip install tensorflow==2.15.0
```

#### 2. Model Not Found Error
```bash
# Solution: Train the model first
python train_model.py --generate-sample
```

#### 3. Streamlit Command Not Found
```bash
# Solution: Use Python module syntax
python -m streamlit run app.py
```

#### 4. Memory Error During Training
```python
# Solution: Reduce batch size in config/settings.py
LSTM_CONFIG = {
    "batch_size": 16,  # Reduced from 32
}
```

#### 5. Port Already in Use
```bash
# Solution: Use a different port
python -m streamlit run app.py --server.port 8502
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to functions
- Include type hints where possible
- Write unit tests for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **UIDAI** - For organizing the Data Hackathon 2026
- **Streamlit** - For the amazing dashboard framework
- **TensorFlow Team** - For deep learning tools
- **Plotly & Folium** - For visualization libraries

---

<div align="center">

**Built with ❤️ for UIDAI Data Hackathon 2026**

[⬆ Back to Top](#-uidai-aadhaar-intelligence-system)

</div>

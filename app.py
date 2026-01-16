"""
UIDAI Aadhaar Intelligence System - Streamlit Dashboard
========================================================
Production-ready web application for enrollment forecasting and intervention strategy.

Run with: streamlit run app.py
"""
import os
import sys
import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import (
    APP_CONFIG, MODEL_PATH, SCALER_PATH, ENROLLMENT_CSV, DEMOGRAPHIC_CSV,
    LSTM_CONFIG, INTERVENTION_CONFIG, MAP_CONFIG,
    ENROLLMENT_REQUIRED_COLUMNS, DEMOGRAPHIC_REQUIRED_COLUMNS, COLUMN_ALIASES
)
from utils.data_generator import STATE_COORDINATES, STATES_DISTRICTS

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title=APP_CONFIG['page_title'],
    page_icon=APP_CONFIG['page_icon'],
    layout=APP_CONFIG['layout'],
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    
    :root {
        --primary: #FF6B35;
        --secondary: #004E89;
        --accent: #00A896;
        --dark: #1A1A2E;
        --light: #F7F9FC;
    }
    
    .main > div {
        padding-top: 2rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background: linear-gradient(135deg, #1A1A2E 0%, #16213E 100%);
        padding: 1rem 2rem;
        border-radius: 16px;
        margin-bottom: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        color: #FFFFFF;
        background: transparent;
        border-radius: 8px;
        padding: 0 20px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #16213E 0%, #1A1A2E 100%);
        border: 1px solid rgba(255,107,53,0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF6B35 0%, #00A896 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    
    .metric-label {
        font-family: 'Space Grotesk', sans-serif;
        color: #9CA3AF;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    h1 {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 50%, #00A896 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        letter-spacing: -1px;
    }
    
    h2, h3 {
        font-family: 'Space Grotesk', sans-serif;
        color: #FFFFFF;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1A1A2E 0%, #16213E 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(255,107,53,0.4);
    }
    
    .alert-box {
        background: linear-gradient(135deg, rgba(255,107,53,0.1) 0%, rgba(247,147,30,0.1) 100%);
        border-left: 4px solid #FF6B35;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, rgba(0,168,150,0.1) 0%, rgba(0,168,150,0.2) 100%);
        border-left: 4px solid #00A896;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F0F1A 0%, #1A1A2E 100%);
    }
    
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label {
        color: #F7F9FC;
        font-family: 'Space Grotesk', sans-serif;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# DATA LOADING & CACHING
# ============================================
@st.cache_resource
def load_model():
    """Load the trained LSTM model (cached)."""
    try:
        import tensorflow as tf
        
        # Try loading .keras format first, then .h5
        model_paths = [MODEL_PATH, MODEL_PATH.replace('.keras', '.h5'), MODEL_PATH.replace('.h5', '.keras')]
        
        for path in model_paths:
            if os.path.exists(path):
                try:
                    model = tf.keras.models.load_model(path, compile=False)
                    # Recompile with explicit loss and metrics
                    model.compile(
                        optimizer='adam',
                        loss='mean_squared_error',
                        metrics=['mean_absolute_error']
                    )
                    return model
                except Exception:
                    continue
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def load_scaler():
    """Load the fitted scaler (cached)."""
    try:
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names using aliases."""
    df = df.copy()
    for old_name, new_name in COLUMN_ALIASES.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    return df


def validate_dataframe(df: pd.DataFrame, required_columns: list, data_name: str) -> tuple:
    """
    Validate DataFrame has required columns.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    df = standardize_columns(df)
    missing = set(required_columns) - set(df.columns)
    if missing:
        return False, f"{data_name} is missing columns: {', '.join(missing)}"
    return True, None


@st.cache_data
def load_enrollment_data(file_path: str = None) -> pd.DataFrame:
    """Load and preprocess enrollment data (cached)."""
    try:
        path = file_path or ENROLLMENT_CSV
        if not os.path.exists(path):
            return None
            
        df = pd.read_csv(path)
        df = standardize_columns(df)
        
        # Validate
        is_valid, error = validate_dataframe(df, ENROLLMENT_REQUIRED_COLUMNS, "Enrollment data")
        if not is_valid:
            st.error(error)
            return None
        
        # Process
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['Total_Enrollment'] = df[['age_0_5', 'age_5_17', 'age_18_greater']].sum(axis=1)
        
        return df
    except Exception as e:
        st.error(f"Error loading enrollment data: {e}")
        return None


@st.cache_data
def load_demographic_data(file_path: str = None) -> pd.DataFrame:
    """Load and preprocess demographic data (cached)."""
    try:
        path = file_path or DEMOGRAPHIC_CSV
        if not os.path.exists(path):
            return None
            
        df = pd.read_csv(path)
        df = standardize_columns(df)
        
        # Validate
        is_valid, error = validate_dataframe(df, DEMOGRAPHIC_REQUIRED_COLUMNS, "Demographic data")
        if not is_valid:
            st.error(error)
            return None
        
        # Process
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        df['Total_Updates'] = df[['demo_age_5_17', 'demo_age_17_']].sum(axis=1)
        
        return df
    except Exception as e:
        st.error(f"Error loading demographic data: {e}")
        return None


def process_uploaded_file(uploaded_file, required_columns: list, data_name: str) -> pd.DataFrame:
    """Process an uploaded CSV file with validation."""
    try:
        df = pd.read_csv(uploaded_file)
        df = standardize_columns(df)
        
        is_valid, error = validate_dataframe(df, required_columns, data_name)
        if not is_valid:
            st.error(f"❌ {error}")
            st.info(f"Required columns: {', '.join(required_columns)}")
            return None
        
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        return None


# ============================================
# FORECASTING
# ============================================
def generate_lstm_forecast(df_enrollment: pd.DataFrame, model, scaler, steps: int = 6) -> pd.DataFrame:
    """Generate LSTM-based forecast."""
    try:
        # Aggregate to monthly
        monthly = df_enrollment.set_index('date').resample('M')['Total_Enrollment'].sum()
        
        # Prepare last sequence
        seq_len = LSTM_CONFIG['sequence_length']
        if len(monthly) < seq_len:
            return None
        
        last_values = monthly.values[-seq_len:].reshape(-1, 1)
        scaled_seq = scaler.transform(last_values)
        
        # Forecast
        forecasts = []
        current_seq = scaled_seq.flatten()
        
        for _ in range(steps):
            pred = model.predict(current_seq.reshape(1, -1, 1), verbose=0)
            forecasts.append(pred[0, 0])
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = pred[0, 0]
        
        # Inverse transform
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = scaler.inverse_transform(forecasts).flatten()
        
        # Create forecast dataframe
        last_date = monthly.index[-1]
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq='M')
        
        return pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecasts,
            'lower_bound': forecasts * 0.85,
            'upper_bound': forecasts * 1.15
        })
    except Exception as e:
        st.error(f"Forecast error: {e}")
        return None


def generate_simple_forecast(df_enrollment: pd.DataFrame, steps: int = 6) -> pd.DataFrame:
    """Generate simple moving average forecast when LSTM not available."""
    monthly = df_enrollment.set_index('date').resample('M')['Total_Enrollment'].sum()
    
    # Simple exponential smoothing forecast
    last_value = monthly.values[-1]
    trend = (monthly.values[-1] - monthly.values[-12]) / 12 if len(monthly) >= 12 else 0
    
    forecasts = []
    for i in range(steps):
        forecast = last_value + trend * (i + 1)
        # Add seasonal factor
        month_idx = (monthly.index[-1].month + i) % 12
        seasonal = 1.0 + 0.1 * np.sin(2 * np.pi * month_idx / 12)
        forecasts.append(forecast * seasonal)
    
    last_date = monthly.index[-1]
    forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=steps, freq='M')
    
    return pd.DataFrame({
        'date': forecast_dates,
        'forecast': forecasts,
        'lower_bound': [f * 0.9 for f in forecasts],
        'upper_bound': [f * 1.1 for f in forecasts]
    })


# ============================================
# INTERVENTION ENGINE
# ============================================
def compute_interventions(df_enrollment: pd.DataFrame, df_demographic: pd.DataFrame) -> pd.DataFrame:
    """Compute district-level intervention recommendations."""
    # Aggregate enrollment by district
    district = df_enrollment.groupby(['state', 'district']).agg({
        'Total_Enrollment': 'sum',
        'age_0_5': 'sum',
        'age_5_17': 'sum'
    }).reset_index()
    
    # Calculate youth ratio
    district['Youth_Ratio'] = (district['age_0_5'] + district['age_5_17']) / district['Total_Enrollment']
    
    # Aggregate demographic updates
    if df_demographic is not None:
        updates = df_demographic.groupby(['state', 'district'])['Total_Updates'].sum().reset_index()
        district = district.merge(updates, on=['state', 'district'], how='left')
        district['Total_Updates'] = district['Total_Updates'].fillna(0)
    else:
        district['Total_Updates'] = 0
    
    # Determine intervention
    def get_action(row):
        if row['Youth_Ratio'] < INTERVENTION_CONFIG['youth_ratio_threshold']:
            return 'Deploy School-based Camps'
        if row['Total_Updates'] > row['Total_Enrollment'] * INTERVENTION_CONFIG['update_enrollment_ratio']:
            return 'Convert to Update-only Center'
        return 'Stable'
    
    def get_priority(row):
        if row['Action'] == 'Stable':
            return 'Low'
        if row['Youth_Ratio'] < 0.15 or row['Total_Updates'] > row['Total_Enrollment'] * 3:
            return 'Critical'
        return 'High'
    
    district['Action'] = district.apply(get_action, axis=1)
    district['Priority'] = district.apply(get_priority, axis=1)
    
    # Add coordinates
    district['lat'] = district['state'].map(lambda x: STATE_COORDINATES.get(x, (22.5, 78.9))[0])
    district['lon'] = district['state'].map(lambda x: STATE_COORDINATES.get(x, (22.5, 78.9))[1])
    
    # Add slight offset for districts in same state
    for state in district['state'].unique():
        mask = district['state'] == state
        n = mask.sum()
        if n > 1:
            offsets = np.linspace(-0.5, 0.5, n)
            district.loc[mask, 'lat'] += offsets
            district.loc[mask, 'lon'] += offsets[::-1]
    
    return district


# ============================================
# VISUALIZATION COMPONENTS
# ============================================
def create_forecast_chart(monthly: pd.Series, forecast_df: pd.DataFrame, title: str) -> go.Figure:
    """Create interactive forecast chart."""
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=monthly.index,
        y=monthly.values,
        name='Historical',
        mode='lines+markers',
        line=dict(color='#00A896', width=2),
        marker=dict(size=4)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df['date'],
        y=forecast_df['forecast'],
        name='Forecast',
        mode='lines+markers',
        line=dict(color='#FF6B35', width=3, dash='dash'),
        marker=dict(size=8, symbol='diamond')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
        y=pd.concat([forecast_df['upper_bound'], forecast_df['lower_bound'][::-1]]),
        fill='toself',
        fillcolor='rgba(255,107,53,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='white')),
        xaxis_title='Date',
        yaxis_title='Enrollments',
        template='plotly_dark',
        paper_bgcolor='rgba(26,26,46,0.8)',
        plot_bgcolor='rgba(22,33,62,0.8)',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='x unified'
    )
    
    return fig


def create_trend_chart(df: pd.DataFrame, y_col: str, title: str, state_filter: str = None) -> go.Figure:
    """Create monthly trend chart."""
    if state_filter and state_filter != "All States":
        df = df[df['state'] == state_filter]
    
    monthly = df.set_index('date').resample('M')[y_col].sum().reset_index()
    
    fig = px.area(
        monthly, x='date', y=y_col,
        title=title,
        color_discrete_sequence=['#00A896']
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(26,26,46,0.8)',
        plot_bgcolor='rgba(22,33,62,0.8)',
        xaxis_title='Date',
        yaxis_title='Count'
    )
    
    return fig


def create_age_distribution_chart(df: pd.DataFrame, state_filter: str = None) -> go.Figure:
    """Create age distribution pie chart."""
    if state_filter and state_filter != "All States":
        df = df[df['state'] == state_filter]
    
    age_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
    values = [df[col].sum() for col in age_cols]
    labels = ['0-5 Years', '5-17 Years', '18+ Years']
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=['#FF6B35', '#00A896', '#004E89']),
        textinfo='percent+label'
    )])
    
    fig.update_layout(
        title=dict(text='Age Distribution', font=dict(size=18, color='white')),
        template='plotly_dark',
        paper_bgcolor='rgba(26,26,46,0.8)',
        showlegend=True
    )
    
    return fig


def create_state_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """Create state-wise comparison bar chart."""
    state_data = df.groupby('state')['Total_Enrollment'].sum().sort_values(ascending=True).tail(10)
    
    fig = go.Figure(go.Bar(
        x=state_data.values,
        y=state_data.index,
        orientation='h',
        marker=dict(
            color=state_data.values,
            colorscale=[[0, '#004E89'], [0.5, '#00A896'], [1, '#FF6B35']],
            showscale=True,
            colorbar=dict(title='Enrollments')
        )
    ))
    
    fig.update_layout(
        title=dict(text='Top 10 States by Enrollment', font=dict(size=18, color='white')),
        template='plotly_dark',
        paper_bgcolor='rgba(26,26,46,0.8)',
        plot_bgcolor='rgba(22,33,62,0.8)',
        xaxis_title='Total Enrollments',
        yaxis_title=''
    )
    
    return fig


def create_intervention_map(district_df: pd.DataFrame) -> folium.Map:
    """Create intervention map with markers."""
    m = folium.Map(
        location=MAP_CONFIG['default_location'],
        zoom_start=MAP_CONFIG['default_zoom'],
        tiles='CartoDB dark_matter'
    )
    
    # Color mapping
    color_map = {
        'Deploy School-based Camps': 'orange',
        'Convert to Update-only Center': 'red',
        'Stable': 'green'
    }
    
    icon_map = {
        'Deploy School-based Camps': 'graduation-cap',
        'Convert to Update-only Center': 'exchange-alt',
        'Stable': 'check'
    }
    
    # Add markers
    for _, row in district_df.iterrows():
        if row['Action'] != 'Stable':
            popup_html = f"""
            <div style="font-family: Arial; min-width: 200px;">
                <h4 style="color: #FF6B35; margin: 0;">{row['district']}</h4>
                <p style="margin: 5px 0;"><b>State:</b> {row['state']}</p>
                <p style="margin: 5px 0;"><b>Action:</b> {row['Action']}</p>
                <p style="margin: 5px 0;"><b>Priority:</b> {row['Priority']}</p>
                <p style="margin: 5px 0;"><b>Youth Ratio:</b> {row['Youth_Ratio']:.1%}</p>
                <p style="margin: 5px 0;"><b>Total Enrollment:</b> {row['Total_Enrollment']:,.0f}</p>
            </div>
            """
            
            folium.Marker(
                location=[row['lat'], row['lon']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(
                    color=color_map.get(row['Action'], 'gray'),
                    icon=icon_map.get(row['Action'], 'info'),
                    prefix='fa'
                )
            ).add_to(m)
    
    # Add legend
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background: rgba(26,26,46,0.9); padding: 15px; border-radius: 8px;
                border: 1px solid #FF6B35; font-family: Arial;">
        <h4 style="color: #FF6B35; margin: 0 0 10px 0;">Intervention Legend</h4>
        <p style="margin: 5px 0;"><span style="color: orange;">●</span> School-based Camps</p>
        <p style="margin: 5px 0;"><span style="color: red;">●</span> Update-only Center</p>
        <p style="margin: 5px 0;"><span style="color: green;">●</span> Stable</p>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    
    return m


# ============================================
# SIDEBAR
# ============================================
def render_sidebar(df_enrollment: pd.DataFrame, df_demographic: pd.DataFrame):
    """Render sidebar with filters and file upload."""
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #FF6B35; margin: 0;">🆔 UIDAI</h2>
        <p style="color: #9CA3AF; font-size: 0.8rem;">Aadhaar Intelligence System</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Data upload section
    st.sidebar.markdown("### 📂 Data Source")
    
    use_uploaded = st.sidebar.checkbox("Upload Custom Data", value=False)
    
    if use_uploaded:
        st.sidebar.markdown("**Enrollment Data**")
        uploaded_enrollment = st.sidebar.file_uploader(
            "Upload Enrollment CSV",
            type=['csv'],
            key='enrollment_upload'
        )
        
        st.sidebar.markdown("**Demographic Data**")
        uploaded_demographic = st.sidebar.file_uploader(
            "Upload Demographic CSV",
            type=['csv'],
            key='demographic_upload'
        )
        
        if uploaded_enrollment:
            df_enrollment = process_uploaded_file(
                uploaded_enrollment, 
                ENROLLMENT_REQUIRED_COLUMNS, 
                "Enrollment"
            )
            if df_enrollment is not None:
                df_enrollment['Total_Enrollment'] = df_enrollment[['age_0_5', 'age_5_17', 'age_18_greater']].sum(axis=1)
        
        if uploaded_demographic:
            df_demographic = process_uploaded_file(
                uploaded_demographic,
                DEMOGRAPHIC_REQUIRED_COLUMNS,
                "Demographic"
            )
            if df_demographic is not None:
                df_demographic['Total_Updates'] = df_demographic[['demo_age_5_17', 'demo_age_17_']].sum(axis=1)
    
    st.sidebar.markdown("---")
    
    # Filters section
    st.sidebar.markdown("### 🎯 Filters")
    
    # SMART STATE LIST - Combine both files
    states_e = set(df_enrollment['state'].dropna().unique()) if df_enrollment is not None else set()
    states_d = set(df_demographic['state'].dropna().unique()) if df_demographic is not None else set()
    all_states = sorted(list(states_e.union(states_d)))
    states = ["All States"] + [str(s) for s in all_states]
    
    selected_state = st.sidebar.selectbox("Select State", states)
    
    # SMART DISTRICT LIST - Combine both files
    if selected_state != "All States":
        # Get districts for this state from both files
        dist_e = set(df_enrollment[df_enrollment['state'] == selected_state]['district'].dropna().unique()) if df_enrollment is not None else set()
        dist_d = set(df_demographic[df_demographic['state'] == selected_state]['district'].dropna().unique()) if df_demographic is not None else set()
        available_districts = sorted(list(dist_e.union(dist_d)))
    else:
        # Get ALL districts from both files
        dist_e = set(df_enrollment['district'].dropna().unique()) if df_enrollment is not None else set()
        dist_d = set(df_demographic['district'].dropna().unique()) if df_demographic is not None else set()
        available_districts = sorted(list(dist_e.union(dist_d)))
    
    districts = ["All Districts"] + available_districts
    selected_district = st.sidebar.selectbox("Select District", districts)
    
    st.sidebar.markdown("---")
    
    # Forecast settings
    st.sidebar.markdown("### 📈 Forecast Settings")
    forecast_months = st.sidebar.slider("Forecast Horizon (Months)", 1, 12, 6)
    
    st.sidebar.markdown("---")
    
    # Info section
    st.sidebar.markdown("""
    <div class="success-box" style="font-size: 0.85rem;">
        <strong>💡 Quick Tips</strong><br>
        • Use filters to analyze specific regions<br>
        • Upload custom data for personalized analysis<br>
        • Adjust forecast horizon in settings
    </div>
    """, unsafe_allow_html=True)
    
    return df_enrollment, df_demographic, selected_state, selected_district, forecast_months


# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main application entry point."""
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">
            🆔 UIDAI Aadhaar Intelligence System
        </h1>
        <p style="color: #9CA3AF; font-size: 1.1rem;">
            AI-Powered Enrollment Forecasting & Strategic Intervention Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    df_enrollment = load_enrollment_data()
    df_demographic = load_demographic_data()
    model = load_model()
    scaler = load_scaler()
    
    # Sidebar
    df_enrollment, df_demographic, selected_state, selected_district, forecast_months = render_sidebar(
        df_enrollment, df_demographic
    )
    
    # Check data availability
    if df_enrollment is None:
        st.warning("⚠️ No enrollment data found. Please upload data or generate sample data.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Generate Sample Data", use_container_width=True):
                with st.spinner("Generating sample data..."):
                    from utils.data_generator import save_sample_data
                    save_sample_data()
                    st.success("✅ Sample data generated! Please refresh the page.")
                    st.rerun()
        
        st.markdown("""
        <div class="alert-box">
            <strong>📋 Expected Data Format</strong><br><br>
            <strong>Enrollment.csv columns:</strong><br>
            <code>date, state, district, age_0_5, age_5_17, age_18_greater</code><br><br>
            <strong>Demographic.csv columns:</strong><br>
            <code>date, state, district, demo_age_5_17, demo_age_17_</code>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Apply filters
    df_enrollment_filtered = df_enrollment.copy()
    df_demographic_filtered = df_demographic.copy() if df_demographic is not None else None
    
    if selected_state != "All States":
        df_enrollment_filtered = df_enrollment_filtered[df_enrollment_filtered['state'] == selected_state]
        if df_demographic_filtered is not None:
            df_demographic_filtered = df_demographic_filtered[df_demographic_filtered['state'] == selected_state]
    
    if selected_district != "All Districts":
        df_enrollment_filtered = df_enrollment_filtered[df_enrollment_filtered['district'] == selected_district]
        if df_demographic_filtered is not None:
            df_demographic_filtered = df_demographic_filtered[df_demographic_filtered['district'] == selected_district]
    
    # Key Metrics
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{df_enrollment_filtered['Total_Enrollment'].sum():,.0f}</p>
            <p class="metric-label">Total Enrollments</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        updates = df_demographic_filtered['Total_Updates'].sum() if df_demographic_filtered is not None else 0
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{updates:,.0f}</p>
            <p class="metric-label">Total Updates</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        unique_districts = df_enrollment_filtered['district'].nunique()
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{unique_districts:,}</p>
            <p class="metric-label">Districts Covered</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        model_status = "🟢 Active" if model is not None else "🟡 Fallback"
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="font-size: 1.5rem;">{model_status}</p>
            <p class="metric-label">LSTM Model</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Enrollment Forecast", "🎯 Intervention Strategy", "📊 Raw Data Analysis"])
    
    # ============================================
    # TAB 1: ENROLLMENT FORECAST
    # ============================================
    with tab1:
        st.markdown("### 🔮 Enrollment Forecasting")
        
        # Generate forecast
        if model is not None and scaler is not None:
            forecast_df = generate_lstm_forecast(df_enrollment_filtered, model, scaler, forecast_months)
            forecast_method = "LSTM Deep Learning"
        else:
            forecast_df = generate_simple_forecast(df_enrollment_filtered, forecast_months)
            forecast_method = "Exponential Smoothing (LSTM not available)"
        
        if forecast_df is not None:
            st.markdown(f"""
            <div class="success-box">
                <strong>🤖 Forecast Method:</strong> {forecast_method}<br>
                <strong>📅 Horizon:</strong> {forecast_months} months ahead
            </div>
            """, unsafe_allow_html=True)
            
            # Forecast chart
            monthly = df_enrollment_filtered.set_index('date').resample('M')['Total_Enrollment'].sum()
            fig = create_forecast_chart(
                monthly, 
                forecast_df, 
                f"Aadhaar Enrollment Forecast - {selected_state}"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast table
            st.markdown("#### 📋 Forecast Details")
            forecast_display = forecast_df.copy()
            forecast_display['date'] = forecast_display['date'].dt.strftime('%B %Y')
            forecast_display['forecast'] = forecast_display['forecast'].apply(lambda x: f"{x:,.0f}")
            forecast_display['lower_bound'] = forecast_display['lower_bound'].apply(lambda x: f"{x:,.0f}")
            forecast_display['upper_bound'] = forecast_display['upper_bound'].apply(lambda x: f"{x:,.0f}")
            forecast_display.columns = ['Month', 'Predicted Enrollment', 'Lower Bound', 'Upper Bound']
            st.dataframe(forecast_display, use_container_width=True, hide_index=True)
        
        # Trend analysis
        col1, col2 = st.columns(2)
        with col1:
            fig_trend = create_trend_chart(
                df_enrollment_filtered, 
                'Total_Enrollment', 
                'Monthly Enrollment Trend'
            )
            st.plotly_chart(fig_trend, use_container_width=True)
        
        with col2:
            fig_age = create_age_distribution_chart(df_enrollment_filtered)
            st.plotly_chart(fig_age, use_container_width=True)
    
    # ============================================
    # TAB 2: INTERVENTION STRATEGY
    # ============================================
    with tab2:
        st.markdown("### 🎯 Strategic Intervention Engine")
        
        st.markdown("""
        <div class="alert-box">
            <strong>🔍 Intervention Logic</strong><br>
            • <strong>School-based Camps:</strong> Youth ratio < 20% → Need to boost child enrollments<br>
            • <strong>Update-only Center:</strong> Updates > 2x Enrollments → Focus on demographic updates
        </div>
        """, unsafe_allow_html=True)
        
        # Compute interventions
        district_df = compute_interventions(df_enrollment, df_demographic)
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        
        intervention_counts = district_df['Action'].value_counts()
        
        with col1:
            school_camps = intervention_counts.get('Deploy School-based Camps', 0)
            st.metric("🏫 School Camps Needed", school_camps)
        
        with col2:
            update_centers = intervention_counts.get('Convert to Update-only Center', 0)
            st.metric("🔄 Update Centers Needed", update_centers)
        
        with col3:
            stable = intervention_counts.get('Stable', 0)
            st.metric("✅ Stable Districts", stable)
        
        # Map
        st.markdown("#### 🗺️ Intervention Map")
        intervention_map = create_intervention_map(district_df)
        st_folium(intervention_map, width=None, height=500, use_container_width=True)
        
        # Intervention table
        st.markdown("#### 📋 Intervention Recommendations")
        
        action_filter = st.selectbox(
            "Filter by Action",
            ["All", "Deploy School-based Camps", "Convert to Update-only Center", "Stable"]
        )
        
        display_df = district_df[['state', 'district', 'Total_Enrollment', 'Youth_Ratio', 'Total_Updates', 'Action', 'Priority']]
        
        if action_filter != "All":
            display_df = display_df[display_df['Action'] == action_filter]
        
        display_df = display_df.sort_values('Priority', ascending=False)
        display_df['Youth_Ratio'] = display_df['Youth_Ratio'].apply(lambda x: f"{x:.1%}")
        display_df['Total_Enrollment'] = display_df['Total_Enrollment'].apply(lambda x: f"{x:,.0f}")
        display_df['Total_Updates'] = display_df['Total_Updates'].apply(lambda x: f"{x:,.0f}")
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = district_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Intervention Report",
            data=csv,
            file_name=f"intervention_report_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    # ============================================
    # TAB 3: RAW DATA ANALYSIS
    # ============================================
    with tab3:
        st.markdown("### 📊 Raw Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Enrollment Data")
            if df_enrollment_filtered is not None:
                st.dataframe(
                    df_enrollment_filtered.head(100),
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(f"Showing 100 of {len(df_enrollment_filtered):,} records")
        
        with col2:
            st.markdown("#### Demographic Data")
            if df_demographic_filtered is not None:
                st.dataframe(
                    df_demographic_filtered.head(100),
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(f"Showing 100 of {len(df_demographic_filtered):,} records")
            else:
                st.info("No demographic data available")
        
        # State comparison
        st.markdown("#### 🏆 State-wise Comparison")
        fig_states = create_state_comparison_chart(df_enrollment)
        st.plotly_chart(fig_states, use_container_width=True)
        
        # Data quality summary
        st.markdown("#### 📋 Data Quality Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Enrollment Data**")
            quality_enrollment = {
                'Total Records': f"{len(df_enrollment):,}",
                'Date Range': f"{df_enrollment['date'].min().strftime('%Y-%m-%d')} to {df_enrollment['date'].max().strftime('%Y-%m-%d')}",
                'Unique States': df_enrollment['state'].nunique(),
                'Unique Districts': df_enrollment['district'].nunique(),
                'Missing Values': df_enrollment.isnull().sum().sum()
            }
            st.json(quality_enrollment)
        
        with col2:
            st.markdown("**Demographic Data**")
            if df_demographic is not None:
                quality_demographic = {
                    'Total Records': f"{len(df_demographic):,}",
                    'Date Range': f"{df_demographic['date'].min().strftime('%Y-%m-%d')} to {df_demographic['date'].max().strftime('%Y-%m-%d')}",
                    'Unique States': df_demographic['state'].nunique(),
                    'Unique Districts': df_demographic['district'].nunique(),
                    'Missing Values': df_demographic.isnull().sum().sum()
                }
                st.json(quality_demographic)
            else:
                st.info("No demographic data available")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6B7280; font-size: 0.85rem; padding: 1rem 0;">
        <p>🆔 UIDAI Aadhaar Intelligence System | Built with ❤️ using Streamlit & TensorFlow</p>
        <p>© 2026 UIDAI Data Hackathon | All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

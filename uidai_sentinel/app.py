"""
Aadhaar Sentinel - Main Dashboard Application
==============================================
Production-grade Streamlit application for UIDAI Operations Intelligence.

Run: streamlit run app.py
"""
import os
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import APP_CONFIG, COLORS, PLOTLY_COLORS, THRESHOLDS
from utils.data_loader import (
    load_enrollment_data, load_demographic_data, 
    get_merged_data, get_filter_options, get_time_series
)
from utils.analytics import (
    calculate_metrics, get_district_analysis, 
    get_trend_data, get_state_summary, get_anomaly_summary
)
from utils.ai_engine import (
    AnomalyDetector, EnrollmentForecaster, 
    quick_forecast, detect_anomalies_simple
)
from utils.maps import create_intervention_map, create_state_markers

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title=APP_CONFIG['page_title'],
    page_icon=APP_CONFIG['page_icon'],
    layout=APP_CONFIG['layout'],
    initial_sidebar_state=APP_CONFIG['initial_sidebar_state']
)

# ============================================
# CUSTOM CSS - Government Dashboard Theme
# ============================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {{
        --primary: {COLORS['primary']};
        --secondary: {COLORS['secondary']};
        --accent: {COLORS['accent']};
        --success: {COLORS['success']};
        --warning: {COLORS['warning']};
        --danger: {COLORS['danger']};
        --bg: {COLORS['background']};
        --card: {COLORS['card_bg']};
        --text: {COLORS['text']};
        --muted: {COLORS['muted']};
    }}
    
    .main > div {{
        padding: 1rem 2rem;
    }}
    
    /* Header Styling */
    .main-header {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }}
    
    .main-header h1 {{
        color: white;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 2rem;
        margin: 0;
        letter-spacing: -0.5px;
    }}
    
    .main-header p {{
        color: rgba(255,255,255,0.85);
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
    }}
    
    /* Metric Cards */
    .metric-card {{
        background: linear-gradient(145deg, var(--card) 0%, #252d3a 100%);
        border: 1px solid rgba(46, 134, 171, 0.3);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }}
    
    .metric-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.3);
    }}
    
    .metric-value {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 2rem;
        font-weight: 700;
        color: var(--accent);
        margin: 0;
        line-height: 1.2;
    }}
    
    .metric-label {{
        font-family: 'Inter', sans-serif;
        color: var(--muted);
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }}
    
    .metric-delta {{
        font-size: 0.85rem;
        margin-top: 0.3rem;
    }}
    
    .delta-positive {{
        color: var(--success);
    }}
    
    .delta-negative {{
        color: var(--danger);
    }}
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: var(--card);
        padding: 0.75rem 1rem;
        border-radius: 12px;
        border: 1px solid rgba(46, 134, 171, 0.2);
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        color: var(--text);
    }}
    
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white !important;
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0a0f14 0%, var(--bg) 100%);
    }}
    
    [data-testid="stSidebar"] .stSelectbox label {{
        color: var(--text);
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }}
    
    /* Section Headers */
    .section-header {{
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1.25rem;
        color: var(--text);
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--secondary);
    }}
    
    /* Alert Boxes */
    .alert-box {{
        background: rgba(231, 76, 60, 0.1);
        border-left: 4px solid var(--danger);
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }}
    
    .info-box {{
        background: rgba(46, 134, 171, 0.1);
        border-left: 4px solid var(--secondary);
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }}
    
    /* Data Tables */
    .dataframe {{
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        color: var(--muted);
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(46, 134, 171, 0.2);
        margin-top: 2rem;
    }}
</style>
""", unsafe_allow_html=True)


# ============================================
# DATA LOADING (CACHED)
# ============================================
@st.cache_data(ttl=3600)
def load_data():
    """Load and merge all data sources."""
    enrollment_df = load_enrollment_data()
    demographic_df = load_demographic_data()
    
    if enrollment_df is None:
        return None, None, None
    
    merged_df = get_merged_data(enrollment_df, demographic_df)
    return enrollment_df, demographic_df, merged_df


def get_cached_district_analysis(df):
    """Get district analysis results."""
    return get_district_analysis(df.copy())


def run_anomaly_detection(df, features):
    """Run anomaly detection on the provided dataframe."""
    try:
        detector = AnomalyDetector()
        flags, scores = detector.fit_predict(df, features)
        return flags, scores
    except Exception as e:
        # Fallback to simple detection
        return detect_anomalies_simple(df, features[0] if features else 'Total_Enrollment')


# ============================================
# SIDEBAR
# ============================================
def render_sidebar(df):
    """Render sidebar with filters and info."""
    
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h2 style="color: #F39C12; margin: 0; font-size: 1.5rem;">🛡️ Aadhaar Sentinel</h2>
        <p style="color: #7F8C8D; font-size: 0.8rem; margin-top: 0.5rem;">
            Operations Intelligence Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # Filters
    st.sidebar.markdown("### 🎯 Filters")
    
    states, districts_by_state = get_filter_options(df)
    
    selected_state = st.sidebar.selectbox(
        "Select State",
        options=states,
        index=0
    )
    
    available_districts = districts_by_state.get(selected_state, ["All Districts"])
    selected_district = st.sidebar.selectbox(
        "Select District",
        options=available_districts,
        index=0
    )
    
    st.sidebar.markdown("---")
    
    # Date Range
    st.sidebar.markdown("### 📅 Date Range")
    
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    
    date_range = st.sidebar.date_input(
        "Select Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    st.sidebar.markdown("---")
    
    # Quick Stats
    st.sidebar.markdown("### 📊 Quick Stats")
    st.sidebar.markdown(f"""
    <div style="background: rgba(46,134,171,0.1); padding: 1rem; border-radius: 8px; font-size: 0.85rem;">
        <p style="margin: 5px 0;"><strong>Total Records:</strong> {len(df):,}</p>
        <p style="margin: 5px 0;"><strong>States:</strong> {df['state'].nunique()}</p>
        <p style="margin: 5px 0;"><strong>Districts:</strong> {df['district'].nunique()}</p>
        <p style="margin: 5px 0;"><strong>Date Range:</strong><br>{min_date} to {max_date}</p>
    </div>
    """, unsafe_allow_html=True)
    
    return selected_state, selected_district, date_range


# ============================================
# METRIC CARDS
# ============================================
def render_metrics(metrics):
    """Render top metric cards."""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        delta_class = "delta-positive" if metrics['enrollment_delta'] >= 0 else "delta-negative"
        delta_sign = "+" if metrics['enrollment_delta'] >= 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{metrics['total_enrollment']:,}</p>
            <p class="metric-label">Total Enrollments</p>
            <p class="metric-delta {delta_class}">{delta_sign}{metrics['enrollment_delta']}% vs last week</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        delta_class = "delta-positive" if metrics['updates_delta'] >= 0 else "delta-negative"
        delta_sign = "+" if metrics['updates_delta'] >= 0 else ""
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{metrics['total_updates']:,}</p>
            <p class="metric-label">Total Updates</p>
            <p class="metric-delta {delta_class}">{delta_sign}{metrics['updates_delta']}% vs last week</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        mi_color = COLORS['danger'] if metrics['migration_index'] > 2 else COLORS['success']
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color: {mi_color};">{metrics['migration_index']:.2f}</p>
            <p class="metric-label">Migration Index</p>
            <p class="metric-delta" style="color: {COLORS['muted']};">Threshold: 2.0</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        yr_color = COLORS['warning'] if metrics['youth_ratio'] < 20 else COLORS['success']
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value" style="color: {yr_color};">{metrics['youth_ratio']}%</p>
            <p class="metric-label">Youth Ratio</p>
            <p class="metric-delta" style="color: {COLORS['muted']};">Ages 0-17</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================
# TAB 1: STRATEGIC MAP
# ============================================
def render_strategic_map_tab(df, district_df):
    """Render the Strategic Intervention Map tab."""
    
    st.markdown('<p class="section-header">🗺️ Strategic Intervention Map</p>', unsafe_allow_html=True)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    action_counts = district_df['Strategic_Action'].value_counts()
    
    with col1:
        update_centers = action_counts.get('🔴 Convert to Update Center', 0)
        st.metric("🔴 Update Centers Needed", update_centers)
    
    with col2:
        mobile_camps = action_counts.get('🟡 Deploy Mobile Camp', 0)
        st.metric("🟡 Mobile Camps Needed", mobile_camps)
    
    with col3:
        school_camps = action_counts.get('🟠 Deploy School-based Camp', 0)
        st.metric("🟠 School Camps Needed", school_camps)
    
    with col4:
        stable = action_counts.get('🟢 Operations Normal', 0)
        st.metric("🟢 Stable Operations", stable)
    
    st.markdown("---")
    
    # Map options
    col1, col2 = st.columns([3, 1])
    with col2:
        show_only_action = st.checkbox("Show only districts needing action", value=False)
    
    # Create and display map
    try:
        intervention_map = create_intervention_map(district_df, show_only_action)
        st_folium(intervention_map, width=None, height=500, use_container_width=True)
    except Exception as e:
        st.error(f"Error creating map: {e}")
        st.info("Displaying data table instead.")
    
    # Action items table
    st.markdown('<p class="section-header">📋 Action Items</p>', unsafe_allow_html=True)
    
    action_df = district_df[district_df['Action_Needed'] == True][[
        'state', 'district', 'Total_Enrollment', 'Total_Updates', 
        'Migration_Index', 'Youth_Ratio', 'Strategic_Action'
    ]].copy()
    
    if len(action_df) > 0:
        action_df['Youth_Ratio'] = (action_df['Youth_Ratio'] * 100).round(1).astype(str) + '%'
        action_df['Migration_Index'] = action_df['Migration_Index'].round(2)
        action_df.columns = ['State', 'District', 'Enrollments', 'Updates', 'Migration Idx', 'Youth %', 'Recommended Action']
        
        st.dataframe(action_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = action_df.to_csv(index=False)
        st.download_button(
            "📥 Download Action Report",
            csv,
            f"intervention_report_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
    else:
        st.success("✅ All districts are operating normally. No immediate actions required.")


# ============================================
# TAB 2: INTEGRITY MONITOR (ANOMALY DETECTION)
# ============================================
def render_integrity_tab(df):
    """Render the Data Integrity Monitor tab with anomaly detection."""
    
    st.markdown('<p class="section-header">🔍 Data Integrity Monitor</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>🤖 AI-Powered Anomaly Detection</strong><br>
        Using Isolation Forest algorithm to identify statistical outliers in enrollment data.
        Red points indicate potential data quality issues or unusual operational patterns.
    </div>
    """, unsafe_allow_html=True)
    
    # Aggregate daily data for anomaly detection
    daily_df = df.groupby('date').agg({
        'Total_Enrollment': 'sum',
        'Total_Updates': 'sum',
        'state': 'nunique',
        'district': 'nunique'
    }).reset_index()
    
    daily_df.columns = ['date', 'Total_Enrollment', 'Total_Updates', 'Active_States', 'Active_Districts']
    
    # Run anomaly detection - ensure fresh detection each time
    with st.spinner("Running anomaly detection..."):
        try:
            # Create a fresh copy for detection
            detection_df = daily_df.copy()
            anomaly_flags, anomaly_scores = run_anomaly_detection(
                detection_df, 
                ['Total_Enrollment']
            )
            
            # Validate lengths match
            if len(anomaly_flags) != len(daily_df):
                raise ValueError(f"Length mismatch: flags={len(anomaly_flags)}, df={len(daily_df)}")
                
        except Exception as e:
            st.warning(f"Using fallback anomaly detection: {e}")
            anomaly_flags, anomaly_scores = detect_anomalies_simple(daily_df, 'Total_Enrollment')
    
    # Safely assign results
    daily_df = daily_df.copy()  # Ensure we're not modifying cached data
    daily_df['Is_Anomaly'] = list(anomaly_flags)
    daily_df['Anomaly_Score'] = list(anomaly_scores)
    daily_df['Point_Color'] = daily_df['Is_Anomaly'].map({True: 'Anomaly', False: 'Normal'})
    
    # Summary
    anomaly_summary = get_anomaly_summary(anomaly_flags, daily_df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Days Analyzed", anomaly_summary['total_records'])
    with col2:
        st.metric("🔴 Anomalies Detected", anomaly_summary['anomaly_count'])
    with col3:
        st.metric("Anomaly Rate", f"{anomaly_summary['anomaly_rate']}%")
    
    st.markdown("---")
    
    # Scatter plot
    fig = px.scatter(
        daily_df,
        x='date',
        y='Total_Enrollment',
        color='Point_Color',
        color_discrete_map={
            'Normal': PLOTLY_COLORS['normal'],
            'Anomaly': PLOTLY_COLORS['anomaly']
        },
        size=daily_df['Is_Anomaly'].map({True: 15, False: 8}),
        hover_data=['Total_Updates', 'Active_States', 'Active_Districts'],
        title='Enrollment Pattern Analysis - Anomaly Detection'
    )
    
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(30, 37, 48, 0.8)',
        plot_bgcolor='rgba(14, 17, 23, 0.8)',
        xaxis_title='Date',
        yaxis_title='Daily Enrollments',
        legend_title='Status',
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly details table
    if anomaly_summary['anomaly_count'] > 0:
        st.markdown('<p class="section-header">⚠️ Anomalous Days</p>', unsafe_allow_html=True)
        
        anomaly_days = daily_df[daily_df['Is_Anomaly'] == True][[
            'date', 'Total_Enrollment', 'Total_Updates', 'Anomaly_Score'
        ]].copy()
        
        anomaly_days['date'] = anomaly_days['date'].dt.strftime('%Y-%m-%d')
        anomaly_days['Anomaly_Score'] = anomaly_days['Anomaly_Score'].round(3)
        anomaly_days.columns = ['Date', 'Enrollments', 'Updates', 'Anomaly Score']
        
        st.dataframe(anomaly_days, use_container_width=True, hide_index=True)
    
    # Distribution chart
    st.markdown('<p class="section-header">📊 Enrollment Distribution</p>', unsafe_allow_html=True)
    
    fig_hist = px.histogram(
        daily_df,
        x='Total_Enrollment',
        color='Point_Color',
        color_discrete_map={
            'Normal': PLOTLY_COLORS['normal'],
            'Anomaly': PLOTLY_COLORS['anomaly']
        },
        nbins=30,
        title='Distribution of Daily Enrollments'
    )
    
    fig_hist.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(30, 37, 48, 0.8)',
        plot_bgcolor='rgba(14, 17, 23, 0.8)',
        xaxis_title='Daily Enrollments',
        yaxis_title='Frequency',
        bargap=0.1
    )
    
    st.plotly_chart(fig_hist, use_container_width=True)


# ============================================
# TAB 3: FORECAST
# ============================================
def render_forecast_tab(df, state, district):
    """Render the Future Forecast tab."""
    
    st.markdown('<p class="section-header">🔮 Enrollment Forecast</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>🧠 AI-Powered Forecasting</strong><br>
        Using LSTM neural network to predict future enrollment trends based on historical patterns.
    </div>
    """, unsafe_allow_html=True)
    
    # Forecast settings
    col1, col2 = st.columns([2, 1])
    with col2:
        forecast_days = st.slider("Forecast Horizon (Days)", 7, 60, 30)
    
    # Get time series data
    ts = get_time_series(df, 'Total_Enrollment', 'D', state, district)
    
    if len(ts) < 30:
        st.warning("⚠️ Insufficient data for reliable forecasting. Need at least 30 days of data.")
        return
    
    # Generate forecast
    with st.spinner("Generating forecast..."):
        try:
            # Try LSTM forecaster
            forecaster = EnrollmentForecaster()
            if forecaster.load():
                # Use pre-trained model
                forecaster.last_sequence = forecaster.scaler.transform(
                    ts.values[-30:].reshape(-1, 1)
                )
                forecast, lower, upper = forecaster.forecast(forecast_days)
                method = "LSTM Neural Network"
            else:
                # Train new model (simplified for speed)
                forecast, lower, upper = quick_forecast(ts, forecast_days)
                method = "Exponential Smoothing"
        except Exception as e:
            # Fallback to simple forecast
            forecast, lower, upper = quick_forecast(ts, forecast_days)
            method = "Exponential Smoothing (Fallback)"
    
    st.success(f"✅ Forecast generated using: **{method}**")
    
    # Create forecast dates
    last_date = ts.index[-1]
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=forecast_days, freq='D')
    
    # Create figure
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=ts.index,
        y=ts.values,
        name='Historical (Actual)',
        mode='lines',
        line=dict(color=PLOTLY_COLORS['actual'], width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast,
        name='Forecast (Predicted)',
        mode='lines',
        line=dict(color=PLOTLY_COLORS['forecast'], width=2, dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill='toself',
        fillcolor=PLOTLY_COLORS['confidence'],
        line=dict(color='rgba(0,0,0,0)'),
        name='Confidence Interval (±15%)'
    ))
    
    fig.update_layout(
        title=f'Enrollment Forecast - Next {forecast_days} Days',
        template='plotly_dark',
        paper_bgcolor='rgba(30, 37, 48, 0.8)',
        plot_bgcolor='rgba(14, 17, 23, 0.8)',
        xaxis_title='Date',
        yaxis_title='Daily Enrollments',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast summary
    st.markdown('<p class="section-header">📈 Forecast Summary</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Avg. Daily Forecast", f"{np.mean(forecast):,.0f}")
    with col2:
        st.metric("Total Forecast Period", f"{np.sum(forecast):,.0f}")
    with col3:
        current_avg = ts[-30:].mean()
        change = ((np.mean(forecast) - current_avg) / current_avg) * 100
        st.metric("Expected Change", f"{change:+.1f}%")
    with col4:
        st.metric("Peak Day Forecast", f"{np.max(forecast):,.0f}")
    
    # Forecast table
    forecast_df = pd.DataFrame({
        'Date': forecast_dates.strftime('%Y-%m-%d'),
        'Predicted': forecast.astype(int),
        'Lower Bound': lower.astype(int),
        'Upper Bound': upper.astype(int)
    })
    
    with st.expander("📋 View Detailed Forecast"):
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        csv = forecast_df.to_csv(index=False)
        st.download_button(
            "📥 Download Forecast",
            csv,
            f"enrollment_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )


# ============================================
# MAIN APPLICATION
# ============================================
def main():
    """Main application entry point."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Aadhaar Sentinel</h1>
        <p>UIDAI Operations Intelligence Dashboard | Prescriptive Analytics for Strategic Decision Making</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    enrollment_df, demographic_df, merged_df = load_data()
    
    # Check data availability
    if merged_df is None:
        st.error("⚠️ No data available. Please ensure data files are in the `data/` folder.")
        
        st.markdown("""
        <div class="alert-box">
            <strong>📁 Expected Files:</strong><br>
            <code>data/Enrollment.csv</code> - [date, state, district, age_0_5, age_5_17, age_18_greater]<br>
            <code>data/Demographic.csv</code> - [date, state, district, demo_age_5_17, demo_age_17_]
        </div>
        """, unsafe_allow_html=True)
        
        # Option to generate sample data
        if st.button("🔄 Generate Sample Data"):
            with st.spinner("Generating sample data..."):
                try:
                    # Import and run data generator
                    sys.path.insert(0, str(Path(__file__).parent.parent))
                    from utils.data_generator import save_sample_data
                    save_sample_data(str(Path(__file__).parent / "data"))
                    st.success("✅ Sample data generated! Please refresh the page.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generating data: {e}")
        return
    
    # Sidebar
    selected_state, selected_district, date_range = render_sidebar(merged_df)
    
    # Filter data by date range
    if len(date_range) == 2:
        merged_df = merged_df[
            (merged_df['date'].dt.date >= date_range[0]) &
            (merged_df['date'].dt.date <= date_range[1])
        ]
    
    # Calculate metrics
    metrics = calculate_metrics(merged_df, selected_state, selected_district)
    
    # Display metrics
    render_metrics(metrics)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Get district analysis
    district_df = get_cached_district_analysis(merged_df)
    
    # Filter district analysis if state selected
    if selected_state != "All States":
        district_df = district_df[district_df['state'] == selected_state]
    
    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "🗺️ Strategic Map",
        "🔍 Integrity Monitor",
        "🔮 Future Forecast"
    ])
    
    with tab1:
        render_strategic_map_tab(merged_df, district_df)
    
    with tab2:
        render_integrity_tab(merged_df)
    
    with tab3:
        render_forecast_tab(merged_df, selected_state, selected_district)
    
    # Footer
    st.markdown(f"""
    <div class="footer">
        <p>🛡️ Aadhaar Sentinel | UIDAI Operations Intelligence Dashboard</p>
        <p>Built for UIDAI Data Hackathon 2026 | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

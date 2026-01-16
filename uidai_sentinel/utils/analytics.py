"""
Aadhaar Sentinel - Analytics Module
====================================
Business logic for Migration Index, Youth Ratio, and Strategic Actions.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import THRESHOLDS, ACTIONS


def calculate_migration_index(total_updates: float, total_enrollment: float) -> float:
    """
    Calculate Migration Index = Total Updates / (Total Enrollment + 1)
    
    A high migration index (> 2.0) indicates the area is a migration hub
    where people are updating their Aadhaar more than new enrollments.
    
    Args:
        total_updates: Total demographic updates
        total_enrollment: Total new enrollments
        
    Returns:
        Migration index value
    """
    return total_updates / (total_enrollment + 1)


def calculate_youth_ratio(youth_enrollment: float, total_enrollment: float) -> float:
    """
    Calculate Youth Ratio = (Age 0-5 + Age 5-17) / Total Enrollment
    
    Args:
        youth_enrollment: Enrollments for ages 0-17
        total_enrollment: Total enrollments
        
    Returns:
        Youth ratio value
    """
    if total_enrollment == 0:
        return 0.0
    return youth_enrollment / total_enrollment


def get_strategic_action(row: pd.Series) -> str:
    """
    Determine strategic action based on enrollment and update patterns.
    
    Decision Logic:
    1. If Updates > 2 * Enrollments -> Convert to Update Center
    2. If Enrollments < Threshold -> Deploy Mobile Camp
    3. If Youth Ratio < 20% -> Deploy School Camp
    4. If Migration Index > 2.0 -> Migration Hub
    5. Otherwise -> Operations Normal
    
    Args:
        row: DataFrame row with Total_Enrollment, Total_Updates, Youth_Ratio
        
    Returns:
        Strategic action string with emoji indicator
    """
    try:
        enrollment = row.get('Total_Enrollment', 0)
        updates = row.get('Total_Updates', 0)
        youth_ratio = row.get('Youth_Ratio', 0.5)
        
        # Priority 1: High update to enrollment ratio
        if updates > THRESHOLDS['update_enrollment_ratio'] * enrollment:
            return ACTIONS['update_center']
        
        # Priority 2: Very low enrollment
        if enrollment < THRESHOLDS['enrollment_low']:
            return ACTIONS['mobile_camp']
        
        # Priority 3: Low youth enrollment
        if youth_ratio < THRESHOLDS['youth_ratio_low']:
            return ACTIONS['school_camp']
        
        # Priority 4: Migration hub detection
        migration_index = calculate_migration_index(updates, enrollment)
        if migration_index > THRESHOLDS['migration_index_high']:
            return ACTIONS['migration_hub']
        
        # Default: Normal operations
        return ACTIONS['stable']
        
    except Exception:
        return ACTIONS['stable']


def get_action_priority(action: str) -> int:
    """
    Get numerical priority for sorting actions.
    
    Args:
        action: Action string
        
    Returns:
        Priority number (lower = higher priority)
    """
    priority_map = {
        ACTIONS['update_center']: 1,
        ACTIONS['mobile_camp']: 2,
        ACTIONS['school_camp']: 3,
        ACTIONS['migration_hub']: 4,
        ACTIONS['stable']: 5
    }
    return priority_map.get(action, 5)


def calculate_metrics(
    df: pd.DataFrame,
    state: Optional[str] = None,
    district: Optional[str] = None
) -> Dict:
    """
    Calculate key performance metrics for the dashboard.
    
    Args:
        df: Merged data DataFrame
        state: Filter by state
        district: Filter by district
        
    Returns:
        Dictionary with calculated metrics
    """
    filtered_df = df.copy()
    
    # Apply filters
    if state and state != "All States":
        filtered_df = filtered_df[filtered_df['state'] == state]
    if district and district != "All Districts":
        filtered_df = filtered_df[filtered_df['district'] == district]
    
    # Calculate totals
    total_enrollment = filtered_df['Total_Enrollment'].sum()
    total_updates = filtered_df.get('Total_Updates', pd.Series([0])).sum()
    
    # Calculate Migration Index
    migration_index = calculate_migration_index(total_updates, total_enrollment)
    
    # Calculate Youth metrics
    youth_enrollment = filtered_df.get('Youth_Enrollment', pd.Series([0])).sum()
    youth_ratio = calculate_youth_ratio(youth_enrollment, total_enrollment)
    
    # Calculate daily averages
    date_range = (filtered_df['date'].max() - filtered_df['date'].min()).days + 1
    daily_avg_enrollment = total_enrollment / max(date_range, 1)
    daily_avg_updates = total_updates / max(date_range, 1)
    
    # Count unique locations
    unique_states = filtered_df['state'].nunique()
    unique_districts = filtered_df['district'].nunique()
    
    # Calculate deltas (compare last 7 days vs previous 7 days)
    try:
        recent = filtered_df[filtered_df['date'] >= filtered_df['date'].max() - pd.Timedelta(days=7)]
        previous = filtered_df[
            (filtered_df['date'] < filtered_df['date'].max() - pd.Timedelta(days=7)) &
            (filtered_df['date'] >= filtered_df['date'].max() - pd.Timedelta(days=14))
        ]
        
        enrollment_delta = (
            (recent['Total_Enrollment'].sum() - previous['Total_Enrollment'].sum()) /
            max(previous['Total_Enrollment'].sum(), 1) * 100
        )
        updates_delta = (
            (recent['Total_Updates'].sum() - previous['Total_Updates'].sum()) /
            max(previous['Total_Updates'].sum(), 1) * 100
        )
    except Exception:
        enrollment_delta = 0
        updates_delta = 0
    
    return {
        'total_enrollment': int(total_enrollment),
        'total_updates': int(total_updates),
        'migration_index': round(migration_index, 3),
        'youth_ratio': round(youth_ratio * 100, 1),
        'daily_avg_enrollment': int(daily_avg_enrollment),
        'daily_avg_updates': int(daily_avg_updates),
        'unique_states': unique_states,
        'unique_districts': unique_districts,
        'enrollment_delta': round(enrollment_delta, 1),
        'updates_delta': round(updates_delta, 1),
        'date_range': f"{filtered_df['date'].min().strftime('%Y-%m-%d')} to {filtered_df['date'].max().strftime('%Y-%m-%d')}"
    }


def get_district_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data by district and calculate strategic actions.
    
    Args:
        df: Merged data DataFrame
        
    Returns:
        DataFrame with district-level analysis
    """
    # Aggregate by state and district
    district_agg = df.groupby(['state', 'district']).agg({
        'Total_Enrollment': 'sum',
        'Total_Updates': 'sum',
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum'
    }).reset_index()
    
    # Calculate derived metrics
    district_agg['Youth_Enrollment'] = district_agg['age_0_5'] + district_agg['age_5_17']
    district_agg['Youth_Ratio'] = district_agg['Youth_Enrollment'] / (district_agg['Total_Enrollment'] + 1)
    district_agg['Migration_Index'] = district_agg['Total_Updates'] / (district_agg['Total_Enrollment'] + 1)
    
    # Determine strategic action for each district
    district_agg['Strategic_Action'] = district_agg.apply(get_strategic_action, axis=1)
    district_agg['Action_Priority'] = district_agg['Strategic_Action'].apply(get_action_priority)
    
    # Flag action needed
    district_agg['Action_Needed'] = district_agg['Strategic_Action'] != ACTIONS['stable']
    
    # Sort by priority
    district_agg = district_agg.sort_values('Action_Priority')
    
    return district_agg


def get_trend_data(
    df: pd.DataFrame,
    freq: str = 'D',
    state: Optional[str] = None,
    district: Optional[str] = None
) -> pd.DataFrame:
    """
    Get time-series trend data for charts.
    
    Args:
        df: Source DataFrame
        freq: Resampling frequency
        state: Filter by state
        district: Filter by district
        
    Returns:
        DataFrame with date-indexed trend data
    """
    filtered_df = df.copy()
    
    if state and state != "All States":
        filtered_df = filtered_df[filtered_df['state'] == state]
    if district and district != "All Districts":
        filtered_df = filtered_df[filtered_df['district'] == district]
    
    # Resample to frequency
    trend = filtered_df.set_index('date').resample(freq).agg({
        'Total_Enrollment': 'sum',
        'Total_Updates': 'sum',
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_greater': 'sum'
    }).reset_index()
    
    trend['Migration_Index'] = trend['Total_Updates'] / (trend['Total_Enrollment'] + 1)
    
    return trend


def get_state_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get state-level summary statistics.
    
    Args:
        df: Source DataFrame
        
    Returns:
        DataFrame with state summaries
    """
    state_agg = df.groupby('state').agg({
        'Total_Enrollment': 'sum',
        'Total_Updates': 'sum',
        'district': 'nunique'
    }).reset_index()
    
    state_agg.columns = ['State', 'Total_Enrollment', 'Total_Updates', 'Districts']
    state_agg['Migration_Index'] = state_agg['Total_Updates'] / (state_agg['Total_Enrollment'] + 1)
    state_agg = state_agg.sort_values('Total_Enrollment', ascending=False)
    
    return state_agg


def get_anomaly_summary(anomaly_flags: pd.Series, df: pd.DataFrame) -> Dict:
    """
    Summarize anomaly detection results.
    
    Args:
        anomaly_flags: Boolean series indicating anomalies
        df: Source DataFrame
        
    Returns:
        Dictionary with anomaly statistics
    """
    total_records = len(anomaly_flags)
    anomaly_count = anomaly_flags.sum()
    anomaly_rate = (anomaly_count / total_records * 100) if total_records > 0 else 0
    
    # Get anomalous records details
    anomalous_df = df[anomaly_flags]
    
    # Handle missing columns gracefully
    affected_states = 0
    affected_districts = 0
    
    if len(anomalous_df) > 0:
        if 'state' in anomalous_df.columns:
            affected_states = anomalous_df['state'].nunique()
        if 'district' in anomalous_df.columns:
            affected_districts = anomalous_df['district'].nunique()
    
    return {
        'total_records': total_records,
        'anomaly_count': int(anomaly_count),
        'anomaly_rate': round(anomaly_rate, 2),
        'affected_states': affected_states,
        'affected_districts': affected_districts
    }

"""
Aadhaar Sentinel - Data Loader Module
=====================================
Functions to load, clean, and preprocess enrollment and demographic data.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import ENROLLMENT_CSV, DEMOGRAPHIC_CSV, DATE_FORMATS


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names by stripping whitespace and converting to lowercase.
    Also handles common column name variations.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with standardized column names
    """
    # Strip whitespace and convert to lowercase
    df.columns = df.columns.str.strip().str.lower()
    
    # Common column name mappings
    column_mappings = {
        'dates': 'date',
        'state_name': 'state',
        'district_name': 'district',
        'enrolment_0_5': 'age_0_5',
        'enrolment_5_17': 'age_5_17',
        'enrolment_18_plus': 'age_18_greater',
        'enrolment_18+': 'age_18_greater',
        'demo_5_17': 'demo_age_5_17',
        'demo_17+': 'demo_age_17_',
        'demo_17_plus': 'demo_age_17_'
    }
    
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns and new_name not in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    return df


def parse_dates(df: pd.DataFrame, date_column: str = 'date') -> pd.DataFrame:
    """
    Parse date column with multiple format support.
    
    Args:
        df: Input DataFrame
        date_column: Name of the date column
        
    Returns:
        DataFrame with parsed date column
    """
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")
    
    # Try multiple date formats
    for fmt in DATE_FORMATS:
        try:
            df[date_column] = pd.to_datetime(df[date_column], format=fmt, errors='coerce')
            if df[date_column].notna().sum() > 0:
                break
        except Exception:
            continue
    
    # Fallback to pandas auto-detection
    if df[date_column].isna().all():
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    return df


def load_enrollment_data(file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load and preprocess enrollment data.
    
    Args:
        file_path: Path to enrollment CSV file
        
    Returns:
        Preprocessed DataFrame or None if loading fails
    """
    path = file_path or ENROLLMENT_CSV
    
    try:
        # Check if file exists
        if not os.path.exists(path):
            print(f"[WARN] Enrollment file not found: {path}")
            return None
        
        # Load data
        df = pd.read_csv(path)
        
        # Standardize columns
        df = standardize_columns(df)
        
        # Validate required columns
        required_cols = ['date', 'state', 'district', 'age_0_5', 'age_5_17', 'age_18_greater']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"[WARN] Missing columns in enrollment data: {missing_cols}")
            # Try to continue with available columns
        
        # Parse dates
        df = parse_dates(df)
        df = df.dropna(subset=['date'])
        
        # Convert numeric columns
        numeric_cols = ['age_0_5', 'age_5_17', 'age_18_greater']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Calculate Total Enrollment
        available_age_cols = [col for col in numeric_cols if col in df.columns]
        df['Total_Enrollment'] = df[available_age_cols].sum(axis=1)
        
        # Calculate Youth Ratio
        if 'age_0_5' in df.columns and 'age_5_17' in df.columns:
            df['Youth_Enrollment'] = df['age_0_5'] + df['age_5_17']
            df['Youth_Ratio'] = df['Youth_Enrollment'] / (df['Total_Enrollment'] + 1)
        
        # Clean string columns
        if 'state' in df.columns:
            df['state'] = df['state'].astype(str).str.strip().str.title()
        if 'district' in df.columns:
            df['district'] = df['district'].astype(str).str.strip().str.title()
        
        print(f"[OK] Loaded enrollment data: {len(df):,} records")
        return df
        
    except Exception as e:
        print(f"[ERROR] Failed to load enrollment data: {e}")
        return None


def load_demographic_data(file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load and preprocess demographic update data.
    
    Args:
        file_path: Path to demographic CSV file
        
    Returns:
        Preprocessed DataFrame or None if loading fails
    """
    path = file_path or DEMOGRAPHIC_CSV
    
    try:
        # Check if file exists
        if not os.path.exists(path):
            print(f"[WARN] Demographic file not found: {path}")
            return None
        
        # Load data
        df = pd.read_csv(path)
        
        # Standardize columns
        df = standardize_columns(df)
        
        # Parse dates
        df = parse_dates(df)
        df = df.dropna(subset=['date'])
        
        # Convert numeric columns
        numeric_cols = ['demo_age_5_17', 'demo_age_17_']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        
        # Calculate Total Updates
        available_demo_cols = [col for col in numeric_cols if col in df.columns]
        df['Total_Updates'] = df[available_demo_cols].sum(axis=1)
        
        # Clean string columns
        if 'state' in df.columns:
            df['state'] = df['state'].astype(str).str.strip().str.title()
        if 'district' in df.columns:
            df['district'] = df['district'].astype(str).str.strip().str.title()
        
        print(f"[OK] Loaded demographic data: {len(df):,} records")
        return df
        
    except Exception as e:
        print(f"[ERROR] Failed to load demographic data: {e}")
        return None


def get_merged_data(
    enrollment_df: Optional[pd.DataFrame] = None,
    demographic_df: Optional[pd.DataFrame] = None
) -> Optional[pd.DataFrame]:
    """
    Merge enrollment and demographic data on date, state, and district.
    
    Args:
        enrollment_df: Enrollment DataFrame
        demographic_df: Demographic DataFrame
        
    Returns:
        Merged DataFrame or None if merging fails
    """
    try:
        # Load data if not provided
        if enrollment_df is None:
            enrollment_df = load_enrollment_data()
        if demographic_df is None:
            demographic_df = load_demographic_data()
        
        if enrollment_df is None:
            return None
        
        if demographic_df is None:
            # Return enrollment data with zero updates
            enrollment_df['Total_Updates'] = 0
            return enrollment_df
        
        # Merge on date, state, district
        merged = pd.merge(
            enrollment_df,
            demographic_df[['date', 'state', 'district', 'Total_Updates']],
            on=['date', 'state', 'district'],
            how='left'
        )
        
        # Fill missing updates with 0
        merged['Total_Updates'] = merged['Total_Updates'].fillna(0).astype(int)
        
        print(f"[OK] Merged data: {len(merged):,} records")
        return merged
        
    except Exception as e:
        print(f"[ERROR] Failed to merge data: {e}")
        return None


def get_time_series(
    df: pd.DataFrame,
    value_column: str = 'Total_Enrollment',
    freq: str = 'D',
    state: Optional[str] = None,
    district: Optional[str] = None
) -> pd.Series:
    """
    Extract time series data for forecasting.
    
    Args:
        df: Source DataFrame
        value_column: Column to aggregate
        freq: Resampling frequency ('D' for daily, 'W' for weekly, 'M' for monthly)
        state: Filter by state
        district: Filter by district
        
    Returns:
        Time series as pandas Series
    """
    filtered_df = df.copy()
    
    if state and state != "All States":
        filtered_df = filtered_df[filtered_df['state'] == state]
    
    if district and district != "All Districts":
        filtered_df = filtered_df[filtered_df['district'] == district]
    
    # Resample to specified frequency
    ts = filtered_df.set_index('date')[value_column].resample(freq).sum()
    ts = ts.fillna(0)
    
    return ts


def get_filter_options(df: pd.DataFrame) -> Tuple[list, dict]:
    """
    Get available filter options from data.
    
    Args:
        df: Source DataFrame
        
    Returns:
        Tuple of (states list, districts dict by state)
    """
    states = ["All States"] + sorted(df['state'].unique().tolist())
    
    districts_by_state = {"All States": ["All Districts"]}
    for state in df['state'].unique():
        districts_by_state[state] = ["All Districts"] + sorted(
            df[df['state'] == state]['district'].unique().tolist()
        )
    
    return states, districts_by_state

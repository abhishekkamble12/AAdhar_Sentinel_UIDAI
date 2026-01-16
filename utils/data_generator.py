"""
Sample Data Generator for UIDAI Aadhaar System.
Generates realistic synthetic data for testing and demonstration.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Indian states and sample districts
STATES_DISTRICTS = {
    "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Thane"],
    "Uttar Pradesh": ["Lucknow", "Kanpur", "Varanasi", "Agra", "Prayagraj"],
    "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Salem", "Tiruchirappalli"],
    "Karnataka": ["Bengaluru", "Mysuru", "Hubli", "Mangaluru", "Belgaum"],
    "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Bhavnagar"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Ajmer"],
    "West Bengal": ["Kolkata", "Howrah", "Durgapur", "Asansol", "Siliguri"],
    "Madhya Pradesh": ["Bhopal", "Indore", "Gwalior", "Jabalpur", "Ujjain"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur", "Bhagalpur", "Darbhanga"],
    "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Nellore", "Kurnool"]
}

# Approximate coordinates for states (for map visualization)
STATE_COORDINATES = {
    "Maharashtra": (19.7515, 75.7139),
    "Uttar Pradesh": (26.8467, 80.9462),
    "Tamil Nadu": (11.1271, 78.6569),
    "Karnataka": (15.3173, 75.7139),
    "Gujarat": (22.2587, 71.1924),
    "Rajasthan": (27.0238, 74.2179),
    "West Bengal": (22.9868, 87.8550),
    "Madhya Pradesh": (22.9734, 78.6569),
    "Bihar": (25.0961, 85.3131),
    "Andhra Pradesh": (15.9129, 79.7400)
}


def generate_enrollment_data(
    start_date: str = "2022-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic Aadhaar enrollment data with seasonal patterns.
    
    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with enrollment data
    """
    np.random.seed(seed)
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    records = []
    
    for date in date_range:
        for state, districts in STATES_DISTRICTS.items():
            for district in districts:
                # Base enrollment with seasonal pattern
                base = 500 + np.random.randint(0, 300)
                
                # Seasonal factor (higher in Q1 and Q4)
                month = date.month
                seasonal = 1.0 + 0.3 * np.sin(2 * np.pi * month / 12)
                
                # Trend factor (slight increase over time)
                days_from_start = (date - pd.Timestamp(start_date)).days
                trend = 1.0 + 0.0001 * days_from_start
                
                # Random noise
                noise = np.random.uniform(0.8, 1.2)
                
                total = int(base * seasonal * trend * noise)
                
                # Age distribution (realistic proportions)
                age_0_5 = int(total * np.random.uniform(0.15, 0.25))
                age_5_17 = int(total * np.random.uniform(0.20, 0.35))
                age_18_greater = total - age_0_5 - age_5_17
                
                records.append({
                    "date": date,
                    "state": state,
                    "district": district,
                    "age_0_5": max(0, age_0_5),
                    "age_5_17": max(0, age_5_17),
                    "age_18_greater": max(0, age_18_greater)
                })
    
    return pd.DataFrame(records)


def generate_demographic_data(
    start_date: str = "2022-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic demographic update data.
    
    Args:
        start_date: Start date for data generation
        end_date: End date for data generation
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with demographic update data
    """
    np.random.seed(seed + 100)  # Different seed for variety
    
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    records = []
    
    for date in date_range:
        for state, districts in STATES_DISTRICTS.items():
            for district in districts:
                # Base updates
                base = 200 + np.random.randint(0, 200)
                
                # Seasonal pattern
                month = date.month
                seasonal = 1.0 + 0.2 * np.cos(2 * np.pi * month / 12)
                
                # Random variation
                noise = np.random.uniform(0.7, 1.3)
                
                total = int(base * seasonal * noise)
                
                # Age distribution for updates
                demo_5_17 = int(total * np.random.uniform(0.3, 0.5))
                demo_17_plus = total - demo_5_17
                
                records.append({
                    "date": date,
                    "state": state,
                    "district": district,
                    "demo_age_5_17": max(0, demo_5_17),
                    "demo_age_17_": max(0, demo_17_plus)
                })
    
    return pd.DataFrame(records)


def save_sample_data(output_dir: str = "data"):
    """
    Generate and save sample data files.
    
    Args:
        output_dir: Directory to save CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("[*] Generating Enrollment data...")
    enrollment_df = generate_enrollment_data()
    enrollment_path = os.path.join(output_dir, "Enrollment.csv")
    enrollment_df.to_csv(enrollment_path, index=False)
    print(f"[OK] Saved: {enrollment_path} ({len(enrollment_df):,} records)")
    
    print("[*] Generating Demographic data...")
    demographic_df = generate_demographic_data()
    demographic_path = os.path.join(output_dir, "Demographic.csv")
    demographic_df.to_csv(demographic_path, index=False)
    print(f"[OK] Saved: {demographic_path} ({len(demographic_df):,} records)")
    
    return enrollment_df, demographic_df


if __name__ == "__main__":
    save_sample_data()

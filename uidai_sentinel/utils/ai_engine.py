"""
Aadhaar Sentinel - AI Engine Module
====================================
Machine Learning models for Anomaly Detection and Time-Series Forecasting.
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict
import pickle
import os
import sys
from pathlib import Path

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.settings import THRESHOLDS, LSTM_CONFIG, LSTM_MODEL_PATH, SCALER_PATH


class AnomalyDetector:
    """
    Anomaly Detection using Isolation Forest algorithm.
    
    Identifies statistical outliers in enrollment data that may indicate
    data quality issues or unusual operational patterns.
    """
    
    def __init__(self, contamination: float = None):
        """
        Initialize the Anomaly Detector.
        
        Args:
            contamination: Expected proportion of anomalies (0.0 to 0.5)
        """
        self.contamination = contamination or THRESHOLDS['anomaly_contamination']
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
        self.is_fitted = False
        self.scaler = MinMaxScaler()
    
    def fit(self, df: pd.DataFrame, features: list = None) -> 'AnomalyDetector':
        """
        Fit the anomaly detection model.
        
        Args:
            df: DataFrame with features
            features: List of feature columns to use
            
        Returns:
            Self for method chaining
        """
        if features is None:
            features = ['Total_Enrollment']
        
        # Prepare features
        X = df[features].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled)
        self.is_fitted = True
        self.features = features
        
        return self
    
    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies in the data.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Tuple of (anomaly_flags, anomaly_scores)
            - anomaly_flags: Boolean array (True = anomaly)
            - anomaly_scores: Continuous scores (more negative = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = df[self.features].values
        X_scaled = self.scaler.transform(X)
        
        # Get predictions (-1 for anomaly, 1 for normal)
        predictions = self.model.predict(X_scaled)
        anomaly_flags = predictions == -1
        
        # Get anomaly scores
        scores = self.model.decision_function(X_scaled)
        
        return anomaly_flags, scores
    
    def fit_predict(self, df: pd.DataFrame, features: list = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and predict in one step.
        
        Args:
            df: DataFrame with features
            features: List of feature columns
            
        Returns:
            Tuple of (anomaly_flags, anomaly_scores)
        """
        self.fit(df, features)
        return self.predict(df)


class EnrollmentForecaster:
    """
    LSTM-based Time Series Forecasting for enrollment predictions.
    
    Uses a deep learning approach to capture temporal patterns
    and generate future enrollment forecasts.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize the forecaster.
        
        Args:
            config: LSTM configuration dictionary
        """
        self.config = config or LSTM_CONFIG
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
        self.history = None
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Args:
            data: Scaled time series data
            
        Returns:
            Tuple of (X, y) sequences
        """
        seq_length = self.config['sequence_length']
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        
        return np.array(X), np.array(y)
    
    def _build_model(self) -> None:
        """Build the LSTM model architecture."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
        from tensorflow.keras.optimizers import Adam
        
        self.model = Sequential([
            # Bidirectional LSTM layer
            Bidirectional(
                LSTM(
                    units=self.config['lstm_units'],
                    return_sequences=True,
                    input_shape=(self.config['sequence_length'], 1)
                )
            ),
            Dropout(self.config['dropout_rate']),
            
            # Second LSTM layer
            LSTM(units=self.config['lstm_units'] // 2),
            Dropout(self.config['dropout_rate']),
            
            # Dense layers
            Dense(self.config['dense_units'], activation='relu'),
            Dropout(self.config['dropout_rate'] / 2),
            
            # Output
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    def fit(self, time_series: pd.Series, verbose: int = 0) -> Dict:
        """
        Train the LSTM model on time series data.
        
        Args:
            time_series: Pandas Series with datetime index
            verbose: Verbosity level (0, 1, or 2)
            
        Returns:
            Training history dictionary
        """
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
        
        # Prepare data
        values = time_series.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(values)
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        if len(X) < 10:
            raise ValueError(f"Insufficient data for training. Need at least {self.config['sequence_length'] + 10} data points.")
        
        # Reshape for LSTM [samples, time steps, features]
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model
        self._build_model()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=verbose
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X, y,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_split=self.config['validation_split'],
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        self.last_sequence = scaled_data[-self.config['sequence_length']:]
        
        return {
            'loss': self.history.history['loss'],
            'val_loss': self.history.history['val_loss'],
            'best_loss': min(self.history.history['val_loss'])
        }
    
    def forecast(self, steps: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate future forecasts.
        
        Args:
            steps: Number of steps to forecast
            
        Returns:
            Tuple of (forecast, lower_bound, upper_bound)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        steps = steps or self.config['forecast_days']
        forecasts = []
        current_seq = self.last_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            X = current_seq.reshape(1, self.config['sequence_length'], 1)
            
            # Predict
            pred = self.model.predict(X, verbose=0)
            forecasts.append(pred[0, 0])
            
            # Update sequence
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = pred[0, 0]
        
        # Inverse transform
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = self.scaler.inverse_transform(forecasts).flatten()
        
        # Calculate confidence bounds (simple ±15% for demo)
        lower = forecasts * 0.85
        upper = forecasts * 1.15
        
        return forecasts, lower, upper
    
    def save(self, model_path: str = None, scaler_path: str = None) -> None:
        """
        Save the trained model and scaler.
        
        Args:
            model_path: Path to save model
            scaler_path: Path to save scaler
        """
        model_path = model_path or str(LSTM_MODEL_PATH)
        scaler_path = scaler_path or str(SCALER_PATH)
        
        # Create directories
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model
        self.model.save(model_path)
        
        # Save scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load(self, model_path: str = None, scaler_path: str = None) -> bool:
        """
        Load a pre-trained model and scaler.
        
        Args:
            model_path: Path to model file
            scaler_path: Path to scaler file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        import tensorflow as tf
        
        model_path = model_path or str(LSTM_MODEL_PATH)
        scaler_path = scaler_path or str(SCALER_PATH)
        
        try:
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = tf.keras.models.load_model(model_path, compile=False)
                self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
                
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                
                self.is_fitted = True
                return True
        except Exception as e:
            print(f"[WARN] Could not load model: {e}")
        
        return False


def quick_forecast(
    time_series: pd.Series,
    forecast_days: int = 30
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Quick forecasting using simple exponential smoothing when LSTM is not available.
    
    Args:
        time_series: Time series data
        forecast_days: Number of days to forecast
        
    Returns:
        Tuple of (forecast, lower_bound, upper_bound)
    """
    values = time_series.values
    
    if len(values) < 7:
        # Not enough data, return flat forecast
        last_val = values[-1] if len(values) > 0 else 0
        forecast = np.full(forecast_days, last_val)
        return forecast, forecast * 0.85, forecast * 1.15
    
    # Simple exponential smoothing
    alpha = 0.3
    level = values[-1]
    
    # Calculate trend from last 7 days
    if len(values) >= 7:
        trend = (values[-1] - values[-7]) / 7
    else:
        trend = 0
    
    forecasts = []
    for i in range(forecast_days):
        # Add seasonal component (weekly pattern)
        day_of_week = i % 7
        seasonal = 1.0 + 0.1 * np.sin(2 * np.pi * day_of_week / 7)
        
        forecast = (level + trend * (i + 1)) * seasonal
        forecasts.append(max(0, forecast))
    
    forecasts = np.array(forecasts)
    lower = forecasts * 0.85
    upper = forecasts * 1.15
    
    return forecasts, lower, upper


def detect_anomalies_simple(
    df: pd.DataFrame,
    column: str = 'Total_Enrollment',
    threshold: float = 2.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple anomaly detection using z-scores when Isolation Forest is slow.
    
    Args:
        df: DataFrame with data
        column: Column to check for anomalies
        threshold: Z-score threshold for anomaly
        
    Returns:
        Tuple of (anomaly_flags, z_scores)
    """
    values = df[column].values
    mean = np.mean(values)
    std = np.std(values)
    
    if std == 0:
        return np.zeros(len(values), dtype=bool), np.zeros(len(values))
    
    z_scores = np.abs((values - mean) / std)
    anomaly_flags = z_scores > threshold
    
    return anomaly_flags, z_scores

"""
UIDAI Aadhaar LSTM Model Training Script
=========================================
Trains an LSTM model for enrollment forecasting and saves it for production use.

Usage:
    python train_model.py
    python train_model.py --data path/to/Enrollment.csv --epochs 150
"""
import os
import sys
import argparse
import warnings
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config.settings import (
    LSTM_CONFIG, MODEL_PATH, SCALER_PATH, 
    ENROLLMENT_CSV, ENROLLMENT_REQUIRED_COLUMNS, COLUMN_ALIASES
)


class UIDAIForecaster:
    """
    LSTM-based forecasting model for Aadhaar enrollment prediction.
    """
    
    def __init__(self, config: dict = None):
        """Initialize the forecaster with configuration."""
        self.config = config or LSTM_CONFIG
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names using aliases."""
        df = df.copy()
        for old_name, new_name in COLUMN_ALIASES.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
        return df
    
    def _validate_data(self, df: pd.DataFrame, required_columns: list) -> bool:
        """Validate that DataFrame contains required columns."""
        df = self._standardize_columns(df)
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """
        Prepare enrollment data for LSTM training.
        
        Args:
            df: Raw enrollment DataFrame
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        df = self._standardize_columns(df)
        self._validate_data(df, ENROLLMENT_REQUIRED_COLUMNS)
        
        # Parse dates and aggregate to monthly
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
        
        # Calculate total enrollment
        df['Total_Enrollment'] = df[['age_0_5', 'age_5_17', 'age_18_greater']].sum(axis=1)
        
        # Aggregate to monthly national totals
        monthly = df.set_index('date').resample('M')['Total_Enrollment'].sum()
        
        if len(monthly) < self.config['sequence_length'] + 2:
            raise ValueError(
                f"Insufficient data: need at least {self.config['sequence_length'] + 2} months, "
                f"got {len(monthly)}"
            )
        
        # Scale the data
        values = monthly.values.reshape(-1, 1)
        scaled = self.scaler.fit_transform(values)
        
        # Create sequences
        X, y = [], []
        seq_len = self.config['sequence_length']
        
        for i in range(len(scaled) - seq_len):
            X.append(scaled[i:i + seq_len])
            y.append(scaled[i + seq_len])
        
        X, y = np.array(X), np.array(y)
        
        # Train/validation split
        split_idx = int(len(X) * (1 - self.config['validation_split']))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        print(f"[DATA] Prepared: {len(X_train)} training, {len(X_val)} validation sequences")
        
        return X_train, y_train, X_val, y_val, monthly
    
    def build_model(self) -> Sequential:
        """
        Build the LSTM model architecture.
        
        Returns:
            Compiled Keras Sequential model
        """
        model = Sequential([
            # First Bidirectional LSTM layer
            Bidirectional(
                LSTM(
                    units=self.config['lstm_units'],
                    return_sequences=True,
                    input_shape=(self.config['sequence_length'], 1)
                )
            ),
            Dropout(self.config['dropout_rate']),
            
            # Second LSTM layer
            LSTM(units=self.config['lstm_units'] // 2, return_sequences=False),
            Dropout(self.config['dropout_rate']),
            
            # Dense layers
            Dense(units=self.config['dense_units'], activation='relu'),
            Dropout(self.config['dropout_rate'] / 2),
            
            # Output layer
            Dense(units=1)
        ])
        
        optimizer = Adam(learning_rate=self.config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mean_absolute_error'])
        
        return model
    
    def train(self, df: pd.DataFrame, model_path: str = None, scaler_path: str = None) -> dict:
        """
        Train the LSTM model on enrollment data.
        
        Args:
            df: Enrollment DataFrame
            model_path: Path to save trained model
            scaler_path: Path to save fitted scaler
            
        Returns:
            Training history dictionary
        """
        model_path = model_path or MODEL_PATH
        scaler_path = scaler_path or SCALER_PATH
        
        # Prepare data
        X_train, y_train, X_val, y_val, monthly = self.prepare_data(df)
        
        # Build model
        self.model = self.build_model()
        print("\n[MODEL] Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train
        print(f"\n[TRAIN] Training for up to {self.config['epochs']} epochs...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save scaler
        os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"[SAVE] Scaler saved to: {scaler_path}")
        
        # Final metrics
        final_loss = min(self.history.history['val_loss'])
        final_mae = min(self.history.history.get('val_mean_absolute_error', self.history.history.get('val_mae', [0])))
        
        print(f"\n[OK] Training Complete!")
        print(f"   Best Validation Loss (MSE): {final_loss:.6f}")
        print(f"   Best Validation MAE: {final_mae:.6f}")
        print(f"   Model saved to: {model_path}")
        
        return {
            'history': self.history.history,
            'best_val_loss': final_loss,
            'best_val_mae': final_mae,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'data_range': (monthly.index.min(), monthly.index.max()),
            'total_samples': len(monthly)
        }
    
    def forecast(self, last_sequence: np.ndarray, steps: int = 6) -> np.ndarray:
        """
        Generate future forecasts.
        
        Args:
            last_sequence: Last sequence of scaled values
            steps: Number of steps to forecast
            
        Returns:
            Array of forecasted values (unscaled)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        forecasts = []
        current_seq = last_sequence.copy()
        
        for _ in range(steps):
            # Predict next value
            pred = self.model.predict(current_seq.reshape(1, -1, 1), verbose=0)
            forecasts.append(pred[0, 0])
            
            # Update sequence
            current_seq = np.roll(current_seq, -1)
            current_seq[-1] = pred[0, 0]
        
        # Inverse transform
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = self.scaler.inverse_transform(forecasts)
        
        return forecasts.flatten()


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description='Train UIDAI LSTM Enrollment Forecasting Model'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=ENROLLMENT_CSV,
        help='Path to Enrollment CSV file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=MODEL_PATH,
        help='Path to save trained model'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=LSTM_CONFIG['epochs'],
        help='Number of training epochs'
    )
    parser.add_argument(
        '--generate-sample',
        action='store_true',
        help='Generate sample data if CSV not found'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("UIDAI Aadhaar LSTM Model Training")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data file: {args.data}")
    print(f"Output model: {args.output}")
    print()
    
    # Check if data exists
    if not os.path.exists(args.data):
        if args.generate_sample:
            print("[WARN] Data file not found. Generating sample data...")
            from utils.data_generator import save_sample_data
            save_sample_data()
        else:
            print(f"[ERROR] Data file not found: {args.data}")
            print("   Run with --generate-sample to create sample data")
            sys.exit(1)
    
    # Load data
    try:
        print(f"[LOAD] Loading data from {args.data}...")
        df = pd.read_csv(args.data)
        print(f"   Loaded {len(df):,} records")
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        sys.exit(1)
    
    # Update config with CLI args
    config = LSTM_CONFIG.copy()
    config['epochs'] = args.epochs
    
    # Train model
    forecaster = UIDAIForecaster(config)
    
    try:
        results = forecaster.train(df, model_path=args.output)
        
        print("\n" + "=" * 60)
        print("Training Summary")
        print("=" * 60)
        print(f"   Data range: {results['data_range'][0]} to {results['data_range'][1]}")
        print(f"   Total samples: {results['total_samples']} months")
        print(f"   Best validation MSE: {results['best_val_loss']:.6f}")
        print(f"   Best validation MAE: {results['best_val_mae']:.6f}")
        print(f"\n[SUCCESS] Model ready for deployment!")
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

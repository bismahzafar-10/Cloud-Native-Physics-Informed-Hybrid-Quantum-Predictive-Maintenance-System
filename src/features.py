import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def compute_physics_features(df):
    """
    Applies numerical finite-difference calculus operators to isolate gradients
    and rolling structural variance from turbofan sensor streams.
    """
    df = df.copy()
    # Sensor columns typically indicating thermal and vibrational variations
    target_sensors = ['sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15']
    
    # Sort columns to ensure sequential consistency across operational periods
    df = df.sort_values(by=['engine_id', 'cycle'])
    
    for sensor in target_sensors:
        # 1. Finite-Difference Numerical Derivative: dS/dt ≈ (S_t - S_{t-1}) / delta_t
        df[f'{sensor}_grad'] = df.groupby('engine_id')[sensor].diff().fillna(0.0)
        
        # 2. Rolling Window Variance (Statistical Volatility Measurement)
        df[f'{sensor}_roll_std'] = df.groupby('engine_id')[sensor].transform(
            lambda x: x.rolling(window=5, min_periods=1).std()
        ).fillna(0.0)
        
    return df

def scale_features(train_df, test_df, feature_cols):
    """Normalizes the engineering matrix to a stable boundary constraint [0, 1]."""
    scaler = MinMaxScaler()
    train_scaled = train_df.copy()
    test_scaled = test_df.copy()
    
    train_scaled[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test_df[feature_cols])
    
    return train_scaled, test_scaled

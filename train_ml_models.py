"""
Train ML Models with Collected Data
"""

import pandas as pd
import numpy as np
import logging
from ml_models import MLTradingEnsemble, PricePredictionLSTM, SignalConfirmationRF, PatternRecognitionNN
from data_collector import DataCollector
import os

logging.basicConfig(level=logging.INFO)
train_logger = logging.getLogger(__name__)


def train_models():
    """Train all ML models with collected data"""
    train_logger.info("Starting ML model training...")
    
    # Load training data
    data_file = 'ml_data/training_data_BTC_USDC.csv'
    if not os.path.exists(data_file):
        train_logger.error(f"Training data not found: {data_file}")
        return False
    
    df = pd.read_csv(data_file)
    train_logger.info(f"Loaded training data: {len(df)} candles")
    
    # Clean the data - keep only numeric columns needed for training
    # yfinance data has extra metadata rows that need to be removed
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    
    # Keep only rows where all numeric columns are actually numeric
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN values in numeric columns
    df = df.dropna(subset=numeric_cols)
    
    # Handle infinity values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=numeric_cols)
    
    # Keep only the columns we need
    df = df[numeric_cols]
    
    # Remove extreme outliers (values beyond 3 standard deviations)
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:  # Avoid division by zero
            df = df[(df[col] >= mean - 3*std) & (df[col] <= mean + 3*std)]
    
    train_logger.info(f"Cleaned training data: {len(df)} candles")
    
    if len(df) < 50:
        train_logger.error("Not enough data for training")
        return False
    
    # Train LSTM model
    train_logger.info("Training LSTM model for price prediction...")
    try:
        lstm_model = PricePredictionLSTM(sequence_length=30, features=5)
        lstm_model.train(df, target_column='close', epochs=20, batch_size=16)
        lstm_model.save()
        train_logger.info("LSTM model trained and saved")
    except Exception as e:
        train_logger.error(f"LSTM training failed: {e}")
    
    # Train Random Forest model with hyperparameter tuning
    train_logger.info("Training Random Forest model for signal confirmation...")
    try:
        from sklearn.model_selection import GridSearchCV
        
        rf_model = SignalConfirmationRF(n_estimators=50, max_depth=8)
        
        # Hyperparameter grid
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [8, 12, 16, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # Prepare features
        X, y, feature_names = rf_model.prepare_features(df)
        
        # Scale features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=rf_model.model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        
        # Update model with best parameters
        rf_model.model = grid_search.best_estimator_
        rf_model.scaler = scaler
        
        train_logger.info(f"Best parameters: {grid_search.best_params_}")
        train_logger.info(f"Best cross-validation accuracy: {grid_search.best_score_:.4f}")
        
        rf_model.save()
        train_logger.info("Random Forest model trained and saved")
    except Exception as e:
        train_logger.error(f"Random Forest training failed: {e}")
    
    # Pattern recognition requires labeled data
    pattern_file = 'ml_data/pattern_data_DOGE_USDC.json'
    if os.path.exists(pattern_file):
        train_logger.info("Pattern data found - would require manual labeling for training")
        train_logger.info("Pattern recognition model skipped (requires labeled data)")
    else:
        train_logger.warning("Pattern data not found")
    
    train_logger.info("ML model training completed!")
    return True


if __name__ == "__main__":
    success = train_models()
    if success:
        print("ML models trained successfully!")
    else:
        print("ML model training failed!")

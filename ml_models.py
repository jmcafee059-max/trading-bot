"""
Machine Learning Models for Trading Bot
Includes LSTM, Random Forest, Neural Networks, and Reinforcement Learning
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neural_network import MLPRegressor, MLPClassifier
import joblib
import os
import logging

ml_logger = logging.getLogger(__name__)

# Try to import TensorFlow, use fallback if not available
try:
    import tensorflow as tf
    from tensorflow import keras
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    ml_logger.warning("TensorFlow not available, using scikit-learn fallback")

class PricePredictionLSTM:
    """LSTM model for price prediction (with scikit-learn fallback)"""
    
    def __init__(self, sequence_length=60, features=10):
        self.sequence_length = sequence_length
        self.features = features
        self.model = None
        self.scaler = MinMaxScaler()
        self.use_fallback = not TENSORFLOW_AVAILABLE
        
    def build_model(self):
        """Build LSTM architecture or fallback to MLP"""
        if self.use_fallback:
            # Use scikit-learn MLPRegressor as fallback
            self.model = MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            )
            ml_logger.info("Using scikit-learn MLPRegressor as fallback")
            return self.model
        else:
            # Use TensorFlow LSTM
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            model = Sequential([
                LSTM(128, return_sequences=True, input_shape=(self.sequence_length, self.features)),
                Dropout(0.2),
                LSTM(64, return_sequences=True),
                Dropout(0.2),
                LSTM(32, return_sequences=False),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='linear')
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            self.model = model
            return model
    
    def prepare_data(self, df, target_column='close'):
        """Prepare data for LSTM training"""
        # Feature engineering
        df = df.copy()
        
        # Technical indicators
        df['returns'] = df[target_column].pct_change()
        df['volatility'] = df[target_column].rolling(window=20).std()
        df['ma_short'] = df[target_column].rolling(window=10).mean()
        df['ma_long'] = df[target_column].rolling(window=50).mean()
        df['momentum'] = df[target_column] - df[target_column].shift(4)
        
        # Drop NaN
        df = df.dropna()
        
        # Scale features
        feature_cols = [col for col in df.columns if col != target_column]
        scaled_data = self.scaler.fit_transform(df[feature_cols])
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i+self.sequence_length])
            y.append(df[target_column].iloc[i+self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def train(self, df, target_column='close', epochs=50, batch_size=32):
        """Train LSTM model (or fallback MLP)"""
        X, y = self.prepare_data(df, target_column)
        
        if self.model is None:
            self.build_model()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        if self.use_fallback:
            # Train scikit-learn MLPRegressor
            # Flatten sequences for MLPRegressor
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            
            self.model.fit(X_train_flat, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_flat)
            mae = np.mean(np.abs(y_pred - y_test))
            
            ml_logger.info(f"MLPRegressor trained. MAE: {mae:.4f}")
        else:
            # Train TensorFlow LSTM
            from tensorflow.keras.callbacks import EarlyStopping
            
            # Callbacks
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # Train
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=1
            )
            
            ml_logger.info(f"LSTM Model trained. Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        return self.model
    
    def predict(self, df):
        """Predict next price"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X, _ = self.prepare_data(df)
        if len(X) == 0:
            return None
        
        # Get last sequence
        last_sequence = X[-1:]
        
        if self.use_fallback:
            # Flatten for MLPRegressor
            last_sequence_flat = last_sequence.reshape(1, -1)
            prediction = self.model.predict(last_sequence_flat)
            return prediction[0]
        else:
            # Use TensorFlow LSTM
            last_sequence_reshaped = last_sequence.reshape(1, self.sequence_length, self.features)
            prediction = self.model.predict(last_sequence_reshaped, verbose=0)
            return prediction[0][0]
    
    def save(self, path='lstm_model'):
        """Save model and scaler"""
        if self.model:
            if self.use_fallback:
                joblib.dump(self.model, f'{path}.pkl')
            else:
                self.model.save(f'{path}.h5')
            joblib.dump(self.scaler, 'lstm_scaler.pkl')
            ml_logger.info(f"LSTM model saved to {path}")
    
    def load(self, path='lstm_model'):
        """Load model and scaler"""
        try:
            if self.use_fallback:
                self.model = joblib.load(f'{path}.pkl')
            else:
                from tensorflow.keras.models import load_model
                self.model = load_model(f'{path}.h5')
            self.scaler = joblib.load('lstm_scaler.pkl')
            ml_logger.info(f"LSTM model loaded from {path}")
        except Exception as e:
            ml_logger.warning(f"LSTM model file not found: {path}")


class SignalConfirmationRF:
    """Random Forest for signal confirmation"""
    
    def __init__(self, n_estimators=100, max_depth=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        
    def prepare_features(self, df):
        """Prepare features for Random Forest with enhanced timing indicators"""
        df = df.copy()
        
        # Basic technical indicators
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'] = self.calculate_macd(df['close'])
        df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
        df['ema_short'] = df['close'].ewm(span=12).mean()
        df['ema_long'] = df['close'].ewm(span=26).mean()
        df['volume_change'] = df['volume'].pct_change()
        df['price_change'] = df['close'].pct_change()
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # Additional indicators
        df['atr'] = self.calculate_atr(df)
        df['williams_r'] = self.calculate_williams_r(df)
        df['stochastic_k'], df['stochastic_d'] = self.calculate_stochastic(df)
        df['momentum'] = df['close'].diff(14)
        df['roc'] = df['close'].pct_change(14) * 100  # Rate of change
        
        # Market regime features
        df['trend'] = df['close'] > df['ema_long']
        df['volatility_regime'] = df['volatility'] > df['volatility'].rolling(50).mean()
        
        # Price position relative to Bollinger Bands
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # TIMING-SPECIFIC FEATURES
        # Price momentum acceleration
        df['momentum_accel'] = df['momentum'].diff()
        
        # RSI momentum
        df['rsi_momentum'] = df['rsi'].diff()
        
        # MACD histogram momentum
        df['macd_hist'] = df['macd'] - df['macd'].ewm(span=9).mean()
        df['macd_hist_momentum'] = df['macd_hist'].diff()
        
        # Price vs EMAs relationship
        df['price_above_ema_short'] = df['close'] > df['ema_short']
        df['price_above_ema_long'] = df['close'] > df['ema_long']
        
        # EMA crossover signals
        df['ema_crossover'] = (df['ema_short'] > df['ema_long']).astype(int)
        df['ema_crossover_signal'] = df['ema_crossover'].diff()
        
        # Volatility breakout detection
        df['volatility_breakout'] = df['volatility'] > df['volatility'].rolling(20).mean() * 1.5
        
        # Target: 1 if price goes up next period, 0 otherwise
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Drop NaN
        df = df.dropna()
        
        # Handle infinity values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        feature_cols = [col for col in df.columns if col not in ['target', 'timestamp']]
        X = df[feature_cols].values
        y = df['target'].values
        
        return X, y, feature_cols
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        return macd
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def calculate_williams_r(self, df, period=14):
        """Calculate Williams %R"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        return williams_r
    
    def calculate_stochastic(self, df, k_period=14, d_period=3):
        """Calculate Stochastic Oscillator"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def train(self, df):
        """Train Random Forest model"""
        X, y, feature_names = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        ml_logger.info(f"Random Forest trained. Accuracy: {accuracy:.4f}")
        ml_logger.info(f"Feature importances: {dict(zip(feature_names, self.model.feature_importances_))}")
        
        return accuracy
    
    def predict_signal(self, df):
        """Predict buy/sell signal"""
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model not trained. Call train() first.")
        
        X, _, _ = self.prepare_features(df)
        if len(X) == 0:
            return None
        
        # Get latest features
        latest_features = X[-1:].reshape(1, -1)
        latest_scaled = self.scaler.transform(latest_features)
        
        prediction = self.model.predict(latest_scaled)[0]
        probability = self.model.predict_proba(latest_scaled)[0]
        
        return {
            'signal': prediction,  # 1 = buy, 0 = sell
            'confidence': probability[prediction],
            'buy_probability': probability[1]
        }
    
    def save(self, path='rf_model.pkl'):
        """Save model and scaler"""
        joblib.dump(self.model, path)
        joblib.dump(self.scaler, 'rf_scaler.pkl')
        ml_logger.info(f"Random Forest model saved to {path}")
    
    def load(self, path='rf_model.pkl'):
        """Load model and scaler"""
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.scaler = joblib.load('rf_scaler.pkl')
            ml_logger.info(f"Random Forest model loaded from {path}")
        else:
            ml_logger.warning(f"Random Forest model file not found: {path}")


class PatternRecognitionNN:
    """Neural Network for pattern recognition"""
    
    def __init__(self, sequence_length=30, num_patterns=10):
        self.sequence_length = sequence_length
        self.num_patterns = num_patterns
        self.model = None
        self.scaler = MinMaxScaler()
        
    def build_model(self):
        """Build CNN-LSTM hybrid for pattern recognition"""
        model = Sequential([
            # Convolutional layers for pattern extraction
            Conv1D(64, 3, activation='relu', input_shape=(self.sequence_length, 1)),
            MaxPooling1D(2),
            Conv1D(32, 3, activation='relu'),
            MaxPooling1D(2),
            
            # LSTM for temporal dependencies
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            
            # Dense layers for classification
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(self.num_patterns, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_pattern_data(self, price_sequences, pattern_labels):
        """Prepare data for pattern recognition"""
        # Normalize sequences
        normalized_sequences = []
        for seq in price_sequences:
            seq_scaled = self.scaler.fit_transform(seq.reshape(-1, 1))
            normalized_sequences.append(seq_scaled.flatten())
        
        X = np.array(normalized_sequences)
        
        # One-hot encode labels
        y = tf.keras.utils.to_categorical(pattern_labels, num_classes=self.num_patterns)
        
        return X, y
    
    def train(self, price_sequences, pattern_labels, epochs=50, batch_size=32):
        """Train pattern recognition model"""
        X, y = self.prepare_pattern_data(price_sequences, pattern_labels)
        
        if self.model is None:
            self.build_model()
        
        # Reshape for CNN
        X = X.reshape(X.shape[0], self.sequence_length, 1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        ml_logger.info(f"Pattern Recognition Model trained. Final accuracy: {history.history['accuracy'][-1]:.4f}")
        return history
    
    def recognize_pattern(self, price_sequence):
        """Recognize pattern in price sequence"""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Normalize and reshape
        seq_scaled = self.scaler.fit_transform(price_sequence.reshape(-1, 1))
        X = seq_scaled.flatten().reshape(1, self.sequence_length, 1)
        
        # Predict
        prediction = self.model.predict(X, verbose=0)
        pattern_class = np.argmax(prediction)
        confidence = prediction[0][pattern_class]
        
        return {
            'pattern': pattern_class,
            'confidence': confidence,
            'probabilities': prediction[0]
        }
    
    def save(self, path='pattern_model.h5'):
        """Save model and scaler"""
        if self.model:
            self.model.save(path)
            joblib.dump(self.scaler, 'pattern_scaler.pkl')
            ml_logger.info(f"Pattern Recognition model saved to {path}")
    
    def load(self, path='pattern_model.h5'):
        """Load model and scaler"""
        if os.path.exists(path):
            self.model = keras.models.load_model(path)
            self.scaler = joblib.load('pattern_scaler.pkl')
            ml_logger.info(f"Pattern Recognition model loaded from {path}")
        else:
            ml_logger.warning(f"Pattern Recognition model file not found: {path}")


class MLTradingEnsemble:
    """Ensemble of ML models for trading decisions"""
    
    def __init__(self):
        self.lstm_model = PricePredictionLSTM()
        self.rf_model = SignalConfirmationRF()
        self.pattern_model = PatternRecognitionNN()
        self.models_trained = False
        
    def train_all(self, historical_data):
        """Train all ML models"""
        ml_logger.info("Training ML ensemble...")
        
        # Train LSTM
        try:
            self.lstm_model.train(historical_data)
            ml_logger.info("LSTM training completed")
        except Exception as e:
            ml_logger.error(f"LSTM training failed: {e}")
        
        # Train Random Forest
        try:
            self.rf_model.train(historical_data)
            ml_logger.info("Random Forest training completed")
        except Exception as e:
            ml_logger.error(f"Random Forest training failed: {e}")
        
        # Pattern recognition requires labeled data
        # This would need manual labeling or automatic pattern detection
        ml_logger.info("Pattern recognition requires labeled pattern data")
        
        self.models_trained = True
        
    def get_trading_signal(self, current_data):
        """Get ensemble trading signal"""
        if not self.models_trained:
            return None
        
        signals = {}
        
        # LSTM prediction
        try:
            lstm_pred = self.lstm_model.predict(current_data)
            if lstm_pred is not None:
                signals['lstm'] = lstm_pred
        except Exception as e:
            ml_logger.error(f"LSTM prediction failed: {e}")
        
        # Random Forest signal
        try:
            rf_signal = self.rf_model.predict_signal(current_data)
            if rf_signal is not None:
                signals['random_forest'] = rf_signal
        except Exception as e:
            ml_logger.error(f"Random Forest prediction failed: {e}")
        
        return signals
    
    def save_all(self):
        """Save all models"""
        self.lstm_model.save()
        self.rf_model.save()
        self.pattern_model.save()
        ml_logger.info("All ML models saved")
    
    def load_all(self):
        """Load all models"""
        self.lstm_model.load()
        self.rf_model.load()
        self.pattern_model.load()
        self.models_trained = True
        ml_logger.info("All ML models loaded")


if __name__ == "__main__":
    # Test ML models
    ml_logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data for testing
    dates = pd.date_range(start='2023-01-01', periods=3000, freq='H')
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(3000) * 0.01) + 100
    volumes = np.random.randint(1000, 10000, 3000)
    
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': volumes,
        'open': prices + np.random.randn(3000) * 0.01,
        'high': prices + np.random.rand(3000) * 0.02,
        'low': prices - np.random.rand(3000) * 0.02
    })
    
    # Test ensemble
    ensemble = MLTradingEnsemble()
    ensemble.train_all(sample_df)
    ensemble.get_trading_signal(sample_df)
    ensemble.save_all()

"""
Data Collection and Training Infrastructure for ML Models
"""

import ccxt
import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

load_dotenv()

data_logger = logging.getLogger(__name__)


class DataCollector:
    """Collect historical and real-time trading data for ML training"""
    
    def __init__(self, exchange_id='coinbase', symbol='DOGE-USDC'):
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.exchange = self._init_exchange()
        self.data_dir = 'ml_data'
        os.makedirs(self.data_dir, exist_ok=True)
        
    def _init_exchange(self):
        """Initialize exchange connection"""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange_config = {
                'enableRateLimit': True,
                'apiKey': os.getenv('API_KEY'),
                'secret': os.getenv('SECRET_KEY'),
            }
            return exchange_class(exchange_config)
        except Exception as e:
            data_logger.error(f"Failed to initialize exchange: {e}")
            return None
    
    def fetch_ohlcv_data(self, timeframe='1h', limit=1000):
        """Fetch OHLCV data from exchange"""
        if not self.exchange:
            return None
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe, limit=limit)
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            data_logger.info(f"Fetched {len(df)} candles for {self.symbol}")
            return df
            
        except Exception as e:
            data_logger.error(f"Failed to fetch OHLCV data: {e}")
            return None
    
    def fetch_historical_data(self, days=30, timeframe='1h'):
        """Fetch historical data for training"""
        all_data = []
        
        # Calculate how many fetches needed
        candles_per_fetch = 1000
        candles_needed = days * 24  # Assuming 1h timeframe
        fetches_needed = (candles_needed // candles_per_fetch) + 1
        
        since = None
        
        for i in range(fetches_needed):
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    self.symbol, 
                    timeframe, 
                    limit=candles_per_fetch,
                    since=since
                )
                
                if not ohlcv:
                    break
                    
                all_data.extend(ohlcv)
                
                # Update since for next fetch
                since = ohlcv[-1][0] + 1
                
                data_logger.info(f"Fetched batch {i+1}/{fetches_needed}")
                
            except Exception as e:
                data_logger.error(f"Error in batch {i+1}: {e}")
                break
        
        if all_data:
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='last')]
            
            data_logger.info(f"Total historical data collected: {len(df)} candles")
            return df
        
        return None
    
    def save_data(self, df, filename=None):
        """Save data to file"""
        if filename is None:
            filename = f"{self.symbol.replace('-', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath)
        data_logger.info(f"Data saved to {filepath}")
        return filepath
    
    def load_data(self, filename):
        """Load data from file"""
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
            data_logger.info(f"Data loaded from {filepath}")
            return df
        return None
    
    def collect_training_data(self, days=30):
        """Collect and save training data"""
        data_logger.info(f"Collecting {days} days of training data for {self.symbol}")
        
        df = self.fetch_historical_data(days=days)
        if df is not None:
            filepath = self.save_data(df)
            return filepath
        
        return None
    
    def get_latest_data(self, limit=100):
        """Get latest data for real-time predictions"""
        df = self.fetch_ohlcv_data(limit=limit)
        return df


class PatternLabeler:
    """Label trading patterns for supervised learning"""
    
    def __init__(self):
        self.patterns = {
            'bullish_engulfing': 0,
            'bearish_engulfing': 1,
            'bullish_pin_bar': 2,
            'bearish_pin_bar': 3,
            'double_bottom': 4,
            'double_top': 5,
            'head_shoulders': 6,
            'ascending_triangle': 7,
            'descending_triangle': 8,
            'no_pattern': 9
        }
    
    def detect_and_label_patterns(self, df):
        """Detect patterns and create labels"""
        sequences = []
        labels = []
        
        window_size = 30
        
        for i in range(len(df) - window_size):
            window = df.iloc[i:i+window_size]
            sequence = window['close'].values
            
            # Detect patterns
            pattern = self._detect_pattern(window)
            label = self.patterns.get(pattern, 9)
            
            sequences.append(sequence)
            labels.append(label)
        
        return sequences, labels
    
    def _detect_pattern(self, window):
        """Detect pattern in price window"""
        # Simplified pattern detection
        # In production, use more sophisticated pattern recognition
        
        prices = window['close'].values
        
        # Check for engulfing
        if len(prices) >= 2:
            if prices[-2] < prices[-3] and prices[-1] > prices[-2]:
                if prices[-1] - prices[-2] > abs(prices[-2] - prices[-3]):
                    return 'bullish_engulfing'
            elif prices[-2] > prices[-3] and prices[-1] < prices[-2]:
                if abs(prices[-1] - prices[-2]) > abs(prices[-2] - prices[-3]):
                    return 'bearish_engulfing'
        
        # Check for double bottom
        if len(prices) >= 10:
            first_half = prices[:5]
            second_half = prices[-5:]
            if min(first_half) == min(first_half) and min(second_half) == min(second_half):
                if abs(min(first_half) - min(second_half)) < np.std(prices) * 0.5:
                    return 'double_bottom'
        
        return 'no_pattern'


class TrainingPipeline:
    """Pipeline for training ML models"""
    
    def __init__(self, symbol='DOGE-USDC'):
        self.symbol = symbol
        self.collector = DataCollector(symbol=symbol)
        self.labeler = PatternLabeler()
        
    def run_training_pipeline(self, days=30):
        """Run complete training pipeline"""
        data_logger.info("Starting ML training pipeline")
        
        # Step 1: Collect data
        data_logger.info("Step 1: Collecting historical data...")
        df = self.collector.collect_training_data(days=days)
        
        if df is None:
            data_logger.error("Failed to collect data")
            return False
        
        # Step 2: Prepare data for different models
        data_logger.info("Step 2: Preparing data for ML models...")
        
        # For LSTM and Random Forest
        training_data_path = self.collector.save_data(df, f'training_data_{self.symbol.replace("-", "_")}.csv')
        
        # For Pattern Recognition
        sequences, labels = self.labeler.detect_and_label_patterns(df)
        pattern_data_path = os.path.join(self.collector.data_dir, f'pattern_data_{self.symbol.replace("-", "_")}.json')
        
        with open(pattern_data_path, 'w') as f:
            json.dump({'sequences': [seq.tolist() for seq in sequences], 'labels': labels}, f)
        
        data_logger.info(f"Pattern data saved to {pattern_data_path}")
        
        # Step 3: Train models (this would be called from main strategy)
        data_logger.info("Step 3: Ready for model training")
        data_logger.info(f"Training data: {len(df)} candles")
        data_logger.info(f"Pattern data: {len(sequences)} sequences")
        
        return {
            'training_data': training_data_path,
            'pattern_data': pattern_data_path,
            'data_points': len(df),
            'pattern_sequences': len(sequences)
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test data collection
    pipeline = TrainingPipeline(symbol='DOGE-USDC')
    results = pipeline.run_training_pipeline(days=7)  # Collect 7 days for testing
    
    if results:
        print("Training pipeline completed successfully!")
        print(f"Data points: {results['data_points']}")
        print(f"Pattern sequences: {results['pattern_sequences']}")

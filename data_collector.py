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
import yfinance as yf

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
            # Try without API keys first for public data
            exchange_config = {
                'enableRateLimit': True,
            }
            
            # Only add API keys if they're available
            api_key = os.getenv('API_KEY')
            secret_key = os.getenv('SECRET_KEY')
            
            if api_key and api_key != 'your_api_key_here':
                exchange_config['apiKey'] = api_key
            if secret_key and secret_key != 'your_secret_key_here':
                exchange_config['secret'] = secret_key
            
            return exchange_class(exchange_config)
        except Exception as e:
            data_logger.error(f"Failed to initialize exchange: {e}")
            return None
    
    def fetch_ohlcv_data(self, timeframe='1h', limit=1000):
        """Fetch OHLCV data from exchange"""
        # Use yfinance directly for reliable data access
        return self._fetch_yfinance_data(timeframe, limit)
    
    def _fetch_yfinance_data(self, timeframe='1h', limit=1000):
        """Fetch data using yfinance as fallback"""
        try:
            # Convert symbol for yfinance
            yf_symbol = self._convert_to_yfinance_symbol()
            
            # Calculate date range
            end_date = datetime.now()
            if timeframe == '1h':
                start_date = end_date - timedelta(days=limit // 24)
            elif timeframe == '1d':
                start_date = end_date - timedelta(days=limit)
            else:
                start_date = end_date - timedelta(days=30)
            
            # Fetch data
            data = yf.download(yf_symbol, start=start_date, end=end_date, interval='1h')
            
            if data.empty:
                data_logger.warning(f"No data found for {yf_symbol}")
                return None
            
            # Rename columns to match expected format
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only needed columns
            data = data[['open', 'high', 'low', 'close', 'volume']]
            
            data_logger.info(f"Fetched {len(data)} candles from yfinance for {yf_symbol}")
            return data
            
        except Exception as e:
            data_logger.error(f"Failed to fetch yfinance data: {e}")
            return None
    
    def _convert_to_yfinance_symbol(self):
        """Convert trading symbol to yfinance format"""
        # Handle different symbol formats
        symbol = self.symbol.replace('-', '')  # Remove dash
        
        # Common crypto conversions
        conversions = {
            'DOGEUSDC': 'DOGE-USD',
            'BTCUSDC': 'BTC-USD',
            'ETHUSDC': 'ETH-USD',
            'SOLUSDC': 'SOL-USD',
        }
        
        return conversions.get(symbol, symbol)
    
    def fetch_historical_data(self, days=30, timeframe='1h'):
        """Fetch historical data for training using yfinance"""
        try:
            # Convert symbol for yfinance
            yf_symbol = self._convert_to_yfinance_symbol()
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch data
            data = yf.download(yf_symbol, start=start_date, end=end_date, interval='1h')
            
            if data.empty:
                data_logger.warning(f"No data found for {yf_symbol}")
                return None
            
            # Rename columns to match expected format
            data = data.rename(columns={
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Keep only needed columns
            data = data[['open', 'high', 'low', 'close', 'volume']]
            
            # Reset index to make timestamp a column
            data = data.reset_index()
            
            data_logger.info(f"Fetched {len(data)} candles from yfinance for {yf_symbol}")
            return data
            
        except Exception as e:
            data_logger.error(f"Failed to fetch historical data: {e}")
            return None
    
    def save_data(self, df, filename=None):
        """Save data to file"""
        if filename is None:
            filename = f"{self.symbol.replace('-', '_')}_{datetime.now().strftime('%Y%m%d')}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        df.to_csv(filepath, index=False)
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
        """Collect and return training data"""
        data_logger.info(f"Collecting {days} days of training data for {self.symbol}")
        
        df = self.fetch_historical_data(days=days)
        if df is not None:
            # Save the data as well
            self.save_data(df)
            return df
        
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
        
        # Save training data
        training_filename = f'training_data_{self.symbol.replace("-", "_")}.csv'
        training_data_path = self.collector.save_data(df, training_filename)
        
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
    results = pipeline.run_training_pipeline(days=730)  # Collect 2 years (max available for hourly data)
    
    if results:
        print("Training pipeline completed successfully!")
        print(f"Data points: {results['data_points']}")
        print(f"Pattern sequences: {results['pattern_sequences']}")

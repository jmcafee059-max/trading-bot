import logging
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CoinScanner:
    """Scans multiple USDC pairs and selects the best trading opportunity based on edge"""
    
    def __init__(self):
        # Top liquid USDC pairs on Coinbase
        self.usdc_pairs = [
            'BTC-USDC',
            'ETH-USDC', 
            'SOL-USDC',
            'DOGE-USDC',
            'ADA-USDC',
            'XRP-USDC',
            'LINK-USDC',
            'MATIC-USDC',
            'DOT-USDC',
            'AVAX-USDC',
            'UNI-USDC',
            'ATOM-USDC',
            'LTC-USDC',
            'BCH-USDC',
            'PEPE-USDC',
            'SHIB-USDC',
            'ARB-USDC',
            'OP-USDC',
            'INJ-USDC',
            'NEAR-USDC'
        ]
        
    def get_pair_data(self, symbol: str, period: str = '5d') -> pd.DataFrame:
        """Fetch recent data for a trading pair"""
        try:
            # Convert Coinbase format to yfinance format
            yf_symbol = symbol.replace('-USDC', '-USD')
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval='1h')
            
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return None
                
            return df
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average True Range for volatility"""
        if len(df) < period:
            return 0.0
            
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().iloc[-1]
        
        return atr
    
    def calculate_edge(self, symbol: str) -> Dict:
        """Calculate trading edge for a pair"""
        df = self.get_pair_data(symbol)
        
        if df is None or len(df) < 20:
            return {
                'symbol': symbol,
                'edge': 0.0,
                'volatility': 0.0,
                'volume': 0.0,
                'trend': 'NEUTRAL',
                'score': 0.0
            }
        
        # Calculate metrics
        current_price = df['Close'].iloc[-1]
        atr = self.calculate_atr(df)
        avg_volume = df['Volume'].tail(20).mean()
        
        # Calculate trend
        price_change = ((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20]) * 100
        if price_change > 2:
            trend = 'BULLISH'
        elif price_change < -2:
            trend = 'BEARISH'
        else:
            trend = 'NEUTRAL'
        
        # Calculate edge score (simplified version)
        # Higher volatility + good volume + bullish trend = higher edge
        volatility_score = min(atr / current_price * 100, 10)  # Normalize to 0-10
        volume_score = min(avg_volume / 1000000, 10)  # Normalize to 0-10
        trend_score = 5 if trend == 'BULLISH' else 2 if trend == 'NEUTRAL' else 0
        
        edge_score = (volatility_score * 0.4) + (volume_score * 0.3) + (trend_score * 0.3)
        
        return {
            'symbol': symbol,
            'edge': edge_score,
            'volatility': atr / current_price * 100 if current_price > 0 else 0,
            'volume': avg_volume,
            'trend': trend,
            'score': edge_score,
            'price': current_price if current_price > 0 else 0
        }
    
    def scan_pairs(self, top_n: int = 5) -> List[Dict]:
        """Scan all pairs and return top opportunities"""
        logger.info(f"Scanning {len(self.usdc_pairs)} USDC pairs...")
        
        results = []
        for symbol in self.usdc_pairs:
            try:
                edge_data = self.calculate_edge(symbol)
                results.append(edge_data)
                logger.info(f"{symbol}: Edge={edge_data['edge']:.2f}, Trend={edge_data['trend']}")
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
        
        # Sort by edge score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_n]
    
    def get_best_pair(self) -> Dict:
        """Get the single best trading pair"""
        top_pairs = self.scan_pairs(top_n=1)
        
        if not top_pairs:
            logger.warning("No valid pairs found")
            return None
            
        best_pair = top_pairs[0]
        logger.info(f"Best pair: {best_pair['symbol']} with edge {best_pair['edge']:.2f}")
        
        return best_pair

if __name__ == "__main__":
    scanner = CoinScanner()
    best_pair = scanner.get_best_pair()
    
    if best_pair:
        print(f"\n=== BEST TRADING OPPORTUNITY ===")
        print(f"Pair: {best_pair['symbol']}")
        print(f"Edge Score: {best_pair['edge']:.2f}")
        print(f"Volatility: {best_pair['volatility']:.2f}%")
        print(f"Volume: {best_pair['volume']:0f}")
        print(f"Trend: {best_pair['trend']}")
        print(f"Price: ${best_pair['price']:.4f}")

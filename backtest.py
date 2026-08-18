"""
Backtesting Module for Trading Strategy Validation
Tests trading strategies on historical data before live trading
Includes TradeKit integration for enhanced backtesting capabilities
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ccxt
import logging
from typing import Dict, List, Tuple, Optional
import json
import os
from dotenv import load_dotenv

# TradeKit imports for enhanced backtesting
try:
    from backtesting import Backtest, Strategy
    from backtesting.lib import crossover
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    logging.warning("backtesting library not available, TradeKit backtesting disabled")

from tradekit_adapter import TradeKitAdapter

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Backtester:
    """Backtesting engine for trading strategies with TradeKit integration"""
    
    def __init__(self, exchange_id='coinbase', symbol='SOL-USDC', timeframe='1h', use_tradekit=False):
        """
        Initialize backtester
        
        Args:
            exchange_id: Exchange to use for historical data
            symbol: Trading pair to backtest
            timeframe: Timeframe for candles (1h, 4h, 1d, etc.)
            use_tradekit: Enable TradeKit enhanced backtesting
        """
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.timeframe = timeframe
        self.use_tradekit = use_tradekit
        
        # Initialize TradeKit adapter if enabled
        self.tradekit_adapter = None
        if use_tradekit:
            config = {
                'USE_TRADEKIT': True,
                'TRADEKIT_MIN_SCORE': 80,
                'TRADEKIT_LIQUIDITY_FILTER': True,
                'TRADEKIT_ORDERBOOK_ANALYSIS': True,
                'TRADEKIT_VOLATILITY_ANALYSIS': True,
                'TRADEKIT_DEBUG': False
            }
            self.tradekit_adapter = TradeKitAdapter(config)
            logger.info("TradeKit adapter initialized for backtesting")
        
        self.exchange = self._init_exchange()
        
    def _init_exchange(self):
        """Initialize exchange connection with credentials"""
        try:
            api_key = os.getenv('API_KEY')
            secret_key = os.getenv('SECRET_KEY')
            
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class({
                'enableRateLimit': True,
                'apiKey': api_key,
                'secret': secret_key,
            })
            return exchange
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            return None
    
    def fetch_historical_data(self, days=30, limit=1000, use_synthetic=False) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from exchange or generate synthetic data
        
        Args:
            days: Number of days of historical data to fetch
            limit: Maximum number of candles to fetch
            use_synthetic: If True, generate synthetic data instead of fetching from exchange
            
        Returns:
            DataFrame with OHLCV data
        """
        if use_synthetic:
            return self._generate_synthetic_data(days, limit)

        if not self.exchange:
            logger.error("Exchange not initialized")
            return pd.DataFrame()

        try:
            # Coinbase (like most exchanges) caps fetch_ohlcv at a few hundred
            # candles per call regardless of the requested limit, so paginate
            # forward from `since` until we hit `limit` candles or catch up
            # to now.
            since = self.exchange.milliseconds() - (days * 24 * 60 * 60 * 1000)
            all_candles = []
            seen_timestamps = set()

            while len(all_candles) < limit:
                batch = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, since, limit)
                if not batch:
                    break

                new_batch = [c for c in batch if c[0] not in seen_timestamps]
                if not new_batch:
                    break

                all_candles.extend(new_batch)
                seen_timestamps.update(c[0] for c in new_batch)

                last_ts = batch[-1][0]
                if last_ts <= since:
                    break
                since = last_ts + 1

                if last_ts >= self.exchange.milliseconds() - 60_000:
                    break

            ohlcv = all_candles[-limit:]

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            logger.info(f"Fetched {len(df)} candles for {self.symbol} ({self.timeframe} timeframe)")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            logger.info("Falling back to synthetic data generation")
            return self._generate_synthetic_data(days, limit)
    
    def _generate_synthetic_data(self, days=30, limit=1000) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for testing
        
        Args:
            days: Number of days of data to generate
            limit: Maximum number of candles
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        logger.info("Generating synthetic data for backtesting")
        
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        timestamps = pd.date_range(start=start_time, end=end_time, freq='1h')
        
        # Generate price data with realistic movements
        np.random.seed(42)
        base_price = 140.0  # SOL base price
        price_changes = np.random.normal(0, 0.002, len(timestamps))  # 0.2% volatility
        
        prices = [base_price]
        for change in price_changes[1:]:
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        prices = np.array(prices)
        
        # Generate OHLC from prices
        opens = prices
        closes = np.roll(prices, -1)
        closes[-1] = opens[-1]
        
        # Generate high/low with some intraday volatility
        high_low_range = 0.01  # 1% intraday range
        highs = opens * (1 + np.abs(np.random.normal(0, high_low_range, len(timestamps))))
        lows = opens * (1 - np.abs(np.random.normal(0, high_low_range, len(timestamps))))
        
        # Ensure high >= open/close and low <= open/close
        highs = np.maximum(highs, np.maximum(opens, closes))
        lows = np.minimum(lows, np.minimum(opens, closes))
        
        # Generate volume
        volumes = np.random.lognormal(15, 0.5, len(timestamps))  # Realistic volume distribution
        
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes
        }, index=timestamps)
        
        logger.info(f"Generated {len(df)} synthetic candles")
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for backtesting with TradeKit enhancement
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added indicators
        """
        # Basic indicators
        df['rsi'] = self._calculate_rsi(df['close'], period=7)
        df['ema_short'] = df['close'].ewm(span=9, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=21, adjust=False).mean()
        df['sma'] = df['close'].rolling(window=50).mean()
        df['macd'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)
        df['atr'] = self._calculate_atr(df)
        
        # TradeKit enhanced indicators
        if self.use_tradekit and self.tradekit_adapter:
            try:
                # Convert DataFrame to OHLCV format for TradeKit
                ohlcv = df.reset_index().values.tolist()
                ohlcv = [[int(t.timestamp() * 1000), o, h, l, c, v] for t, o, h, l, c, v in ohlcv]
                
                enhanced_indicators = self.tradekit_adapter.calculate_enhanced_indicators(ohlcv)
                
                if enhanced_indicators:
                    # Add TradeKit indicators to DataFrame
                    for key, value in enhanced_indicators.items():
                        if value is not None and not pd.isna(value):
                            df[f'tradekit_{key}'] = value
                    
                    logger.info(f"Added {len(enhanced_indicators)} TradeKit indicators to backtest data")
            except Exception as e:
                logger.warning(f"TradeKit indicator calculation failed: {e}")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr
    
    def run_backtest(self, df: pd.DataFrame, initial_capital: float = 1000.0,
                    capital_percentage: float = 0.8, take_profit_pct: float = 0.3,
                    stop_loss_pct: float = 0.2, setup_score_threshold: float = 25.0,
                    use_tradekit_cost_filter: bool = True) -> Dict:
        """
        Run backtest simulation with TradeKit integration
        
        Args:
            df: DataFrame with OHLCV and indicator data
            initial_capital: Starting capital for simulation
            capital_percentage: Percentage of capital to use per trade
            take_profit_pct: Take profit percentage
            stop_loss_pct: Stop loss percentage
            setup_score_threshold: Minimum setup score to enter trade
            use_tradekit_cost_filter: Use TradeKit cost filter for trade validation
            
        Returns:
            Dictionary with backtest results
        """
        # Calculate volume rolling mean for comparison
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        capital = initial_capital
        position = None  # 'long' or 'short'
        entry_price = None
        position_size = 0.0
        trade_amount = 0.0  # Track trade amount for capital management
        trades = []
        
        for i in range(len(df)):
            if i < 50:  # Skip first 50 candles for indicator warmup
                continue
                
            current_row = df.iloc[i]
            current_price = current_row['close']
            
            # Calculate setup score
            setup_score = self._calculate_setup_score(current_row, df['volume_sma'].iloc[i])
            
            # Entry logic
            if position is None and setup_score >= setup_score_threshold:
                # Determine long or short based on indicators
                if current_row['ema_short'] > current_row['ema_long']:
                    position = 'long'
                    entry_price = current_price
                    trade_amount = capital * capital_percentage
                    position_size = trade_amount / current_price
                    
                    # TradeKit cost filter for long trades
                    if use_tradekit_cost_filter and self.use_tradekit and self.tradekit_adapter:
                        expected_exit_price = entry_price * (1 + take_profit_pct / 100)
                        costs = self.tradekit_adapter.calculate_trading_costs(entry_price, expected_exit_price, position_size)
                        if costs and not costs.get('profitable', True):
                            logger.info(f"TradeKit cost filter rejected long trade at {entry_price}")
                            position = None
                            entry_price = None
                            position_size = 0.0
                            trade_amount = 0.0
                            continue
                    
                    capital -= trade_amount  # Deduct capital used for position
                else:
                    position = 'short'
                    entry_price = current_price
                    trade_amount = capital * capital_percentage
                    position_size = trade_amount / current_price
                    
                    # TradeKit cost filter for short trades
                    if use_tradekit_cost_filter and self.use_tradekit and self.tradekit_adapter:
                        expected_exit_price = entry_price * (1 - take_profit_pct / 100)
                        costs = self.tradekit_adapter.calculate_trading_costs(entry_price, expected_exit_price, position_size)
                        if costs and not costs.get('profitable', True):
                            logger.info(f"TradeKit cost filter rejected short trade at {entry_price}")
                            position = None
                            entry_price = None
                            position_size = 0.0
                            trade_amount = 0.0
                            continue
                    
                    capital -= trade_amount  # Deduct capital used for position (margin)
                
                trades.append({
                    'type': position,
                    'entry_time': current_row.name,
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'trade_amount': trade_amount,
                    'capital_at_entry': capital,
                    'setup_score': setup_score,
                    'tradekit_filtered': use_tradekit_cost_filter and self.use_tradekit
                })
            
            # Exit logic
            elif position is not None:
                if position == 'long':
                    profit_pct = ((current_price - entry_price) / entry_price) * 100
                    profit_amount = position_size * (current_price - entry_price)
                else:  # short
                    profit_pct = ((entry_price - current_price) / entry_price) * 100
                    profit_amount = position_size * (entry_price - current_price)
                
                # Check take profit
                if profit_pct >= take_profit_pct:
                    exit_price = current_price
                    # Return capital + profit
                    capital += trade_amount + profit_amount
                    
                    trades[-1].update({
                        'exit_time': current_row.name,
                        'exit_price': exit_price,
                        'profit_pct': profit_pct,
                        'profit_amount': profit_amount,
                        'exit_reason': 'take_profit'
                    })
                    position = None
                    entry_price = None
                    position_size = 0.0
                
                # Check stop loss
                elif profit_pct <= -stop_loss_pct:
                    exit_price = current_price
                    # Return remaining capital after loss
                    capital += trade_amount + profit_amount
                    
                    trades[-1].update({
                        'exit_time': current_row.name,
                        'exit_price': exit_price,
                        'profit_pct': profit_pct,
                        'profit_amount': profit_amount,
                        'exit_reason': 'stop_loss'
                    })
                    position = None
                    entry_price = None
                    position_size = 0.0
        
        # Calculate final statistics
        # Filter only completed trades (with exit data)
        completed_trades = [t for t in trades if 'exit_price' in t]
        total_trades = len(completed_trades)
        
        if total_trades == 0:
            logger.warning("No completed trades in backtest")
            return {
                'initial_capital': initial_capital,
                'final_capital': capital,
                'total_return_pct': 0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_profit_per_trade': 0,
                'max_profit': 0,
                'max_loss': 0,
                'trades': trades
            }
        
        winning_trades = [t for t in completed_trades if t['profit_amount'] > 0]
        losing_trades = [t for t in completed_trades if t['profit_amount'] <= 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        total_profit = sum(t['profit_amount'] for t in completed_trades)
        avg_profit = np.mean([t['profit_amount'] for t in completed_trades]) if completed_trades else 0
        max_profit = max([t['profit_amount'] for t in completed_trades]) if completed_trades else 0
        max_loss = min([t['profit_amount'] for t in completed_trades]) if completed_trades else 0
        
        final_capital = capital + (position_size * entry_price if position and entry_price else 0)
        total_return = ((final_capital - initial_capital) / initial_capital) * 100
        
        results = {
            'initial_capital': initial_capital,
            'final_capital': final_capital,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit_per_trade': avg_profit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'trades': trades,
            'use_tradekit': self.use_tradekit,
            'tradekit_cost_filter': use_tradekit_cost_filter
        }
        
        return results
    
    def run_comparative_backtest(self, df: pd.DataFrame, initial_capital: float = 1000.0,
                                 capital_percentage: float = 0.8, take_profit_pct: float = 0.3,
                                 stop_loss_pct: float = 0.2, setup_score_threshold: float = 25.0) -> Dict:
        """
        Run comparative backtest with and without TradeKit to measure performance impact
        
        Args:
            df: DataFrame with OHLCV and indicator data
            initial_capital: Starting capital for simulation
            capital_percentage: Percentage of capital to use per trade
            take_profit_pct: Take profit percentage
            stop_loss_pct: Stop loss percentage
            setup_score_threshold: Minimum setup score to enter trade
            
        Returns:
            Dictionary with comparative results
        """
        logger.info("Running comparative backtest: With TradeKit vs Without TradeKit")
        
        # Run backtest without TradeKit
        backtester_without = Backtester(self.exchange_id, self.symbol, self.timeframe, use_tradekit=False)
        df_without = df.copy()
        df_without = backtester_without.calculate_indicators(df_without)
        results_without = backtester_without.run_backtest(
            df_without, initial_capital, capital_percentage, take_profit_pct,
            stop_loss_pct, setup_score_threshold, use_tradekit_cost_filter=False
        )
        
        # Run backtest with TradeKit
        backtester_with = Backtester(self.exchange_id, self.symbol, self.timeframe, use_tradekit=True)
        df_with = df.copy()
        df_with = backtester_with.calculate_indicators(df_with)
        results_with = backtester_with.run_backtest(
            df_with, initial_capital, capital_percentage, take_profit_pct,
            stop_loss_pct, setup_score_threshold, use_tradekit_cost_filter=True
        )
        
        # Calculate performance difference
        return_diff = results_with['total_return_pct'] - results_without['total_return_pct']
        profit_diff = results_with['total_profit'] - results_without['total_profit']
        win_rate_diff = results_with['win_rate'] - results_without['win_rate']
        trades_diff = results_with['total_trades'] - results_without['total_trades']
        
        comparative_results = {
            'without_tradekit': results_without,
            'with_tradekit': results_with,
            'performance_difference': {
                'return_pct_diff': return_diff,
                'profit_diff': profit_diff,
                'win_rate_diff': win_rate_diff,
                'trades_diff': trades_diff
            },
            'tradekit_improvement': return_diff > 0 and win_rate_diff >= 0
        }
        
        logger.info(f"Comparative backtest complete:")
        logger.info(f"  Without TradeKit: {results_without['total_return_pct']:.2f}% return, {results_without['total_trades']} trades")
        logger.info(f"  With TradeKit: {results_with['total_return_pct']:.2f}% return, {results_with['total_trades']} trades")
        logger.info(f"  Difference: {return_diff:+.2f}% return, {trades_diff:+d} trades")
        
        return comparative_results
    
    def _calculate_setup_score(self, row: pd.Series, volume_sma: float = None) -> float:
        """
        Calculate setup score for a single candle
        
        Args:
            row: DataFrame row with indicator values
            volume_sma: Volume SMA for comparison
            
        Returns:
            Setup score (0-100)
        """
        score = 0
        
        # EMA trend (20 points)
        if row['ema_short'] > row['ema_long']:
            score += 20
        
        # RSI (15 points)
        if 45 <= row['rsi'] <= 65:
            score += 15
        elif row['rsi'] < 30:
            score += 10
        else:
            score += 5
        
        # MACD (20 points)
        if row['macd'] > row['macd_signal']:
            score += 20
        
        # Volume (15 points) - use provided SMA
        if volume_sma and not pd.isna(volume_sma):
            if row['volume'] > volume_sma:
                score += 15
            else:
                score += 5
        else:
            score += 10  # Default score if no SMA available
        
        # Bollinger Bands (10 points)
        if row['close'] < row['bb_lower']:
            score += 10
        elif row['close'] > row['bb_upper']:
            score += 5
        else:
            score += 3
        
        # ATR volatility (20 points)
        if row['atr'] > 0:
            atr_pct = (row['atr'] / row['close']) * 100
            if 0.01 <= atr_pct <= 0.05:
                score += 20
            elif atr_pct > 0.05:
                score += 10
            else:
                score += 5
        
        return min(score, 100)
    
    def generate_report(self, results: Dict, output_file: str = 'backtest_report.json'):
        """
        Generate backtest report
        
        Args:
            results: Backtest results dictionary
            output_file: File to save report to
        """
        report = {
            'backtest_date': datetime.now().isoformat(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'summary': {
                'initial_capital': results['initial_capital'],
                'final_capital': results['final_capital'],
                'total_return_pct': results['total_return_pct'],
                'total_trades': results['total_trades'],
                'win_rate': results['win_rate'],
                'total_profit': results['total_profit']
            },
            'trades': results['trades']
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Backtest report saved to {output_file}")
        
        # Print summary
        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)
        print(f"Symbol: {self.symbol} ({self.timeframe})")
        print(f"Initial Capital: ${results['initial_capital']:.2f}")
        print(f"Final Capital: ${results['final_capital']:.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']*100:.1f}%")
        print(f"Winning Trades: {results['winning_trades']}")
        print(f"Losing Trades: {results['losing_trades']}")
        print(f"Total Profit: ${results['total_profit']:.2f}")
        print(f"Avg Profit/Trade: ${results['avg_profit_per_trade']:.2f}")
        print(f"Max Profit: ${results['max_profit']:.2f}")
        print(f"Max Loss: ${results['max_loss']:.2f}")
        print("="*50 + "\n")


def main():
    """Main function to run backtest"""
    # Initialize backtester
    backtester = Backtester(exchange_id='coinbase', symbol='SOL-USDC', timeframe='1h')
    
    # Fetch historical data (use synthetic data as fallback)
    print("Fetching historical data...")
    df = backtester.fetch_historical_data(days=30, limit=1000, use_synthetic=True)
    
    if df.empty:
        print("Failed to generate historical data")
        return
    
    # Calculate indicators
    print("Calculating indicators...")
    df = backtester.calculate_indicators(df)
    
    # Run backtest
    print("Running backtest...")
    results = backtester.run_backtest(
        df=df,
        initial_capital=1000.0,
        capital_percentage=0.8,
        take_profit_pct=0.3,
        stop_loss_pct=0.2,
        setup_score_threshold=25.0
    )
    
    # Generate report
    backtester.generate_report(results)


if __name__ == '__main__':
    main()

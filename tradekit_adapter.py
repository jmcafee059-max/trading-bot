"""
TradeKit Adapter Module

This module provides a clean interface to TradeKit components for enhanced
technical analysis, order book analysis, and backtesting capabilities.

The adapter is designed to be completely disableable via configuration and
does not interfere with the existing Coinbase execution layer.

SAFETY FEATURES:
- TradeKit can be completely disabled via USE_TRADEKIT=false in .env
- All methods have try-catch blocks that fall back gracefully on errors
- TradeKit never directly places orders - it only provides analysis
- Coinbase remains the authoritative source for all order and position data
- All TradeKit analysis is optional and the bot functions normally when disabled
- Configuration flags allow granular control of TradeKit features
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

# TradeKit imports (with fallback if not installed)
try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    logging.warning("pandas-ta not available, TradeKit enhanced indicators will be disabled")

try:
    import tulipy as ti
    TULIPY_AVAILABLE = True
except ImportError:
    TULIPY_AVAILABLE = False
    logging.warning("tulipy not available, TradeKit fast indicators will be disabled")

try:
    from backtesting import Backtest, Strategy
    BACKTESTING_AVAILABLE = True
except ImportError:
    BACKTESTING_AVAILABLE = False
    logging.warning("backtesting library not available, TradeKit backtesting will be disabled")

# Setup logging
logger = logging.getLogger(__name__)


class TradeKitAdapter:
    """
    Main adapter class for TradeKit functionality.
    
    This class provides enhanced technical analysis, order book analysis,
    and backtesting capabilities while maintaining a clean interface.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TradeKit adapter with configuration.
        
        Args:
            config: Dictionary containing TradeKit configuration
        """
        self.enabled = config.get('USE_TRADEKIT', False)
        self.min_score = config.get('TRADEKIT_MIN_SCORE', 80)
        self.liquidity_filter = config.get('TRADEKIT_LIQUIDITY_FILTER', True)
        self.orderbook_analysis = config.get('TRADEKIT_ORDERBOOK_ANALYSIS', True)
        self.volatility_analysis = config.get('TRADEKIT_VOLATILITY_ANALYSIS', True)
        self.debug = config.get('TRADEKIT_DEBUG', False)
        
        # Setup score weights (configurable)
        self.score_weights = config.get('TRADEKIT_SCORE_WEIGHTS', {
            'trend': 15,
            'momentum': 15,
            'volume': 15,
            'liquidity': 10,
            'volatility': 10,
            'support_resistance': 10,
            'relative_strength': 5,
            'btc_confirmation': 5,
            'ml_confirmation': 10,
            'trade_economics': 5
        })
        
        logger.info(f"TradeKit adapter initialized (enabled={self.enabled})")
        
        if self.enabled and not PANDAS_TA_AVAILABLE:
            logger.warning("TradeKit enabled but pandas-ta not available - using fallback indicators")
    
    def calculate_enhanced_indicators(self, ohlcv: List[List[float]]) -> Dict[str, Any]:
        """
        Calculate enhanced technical indicators using TradeKit components.
        
        Args:
            ohlcv: OHLCV data [[timestamp, open, high, low, close, volume], ...]
            
        Returns:
            Dictionary containing enhanced indicators
        """
        if not self.enabled or not ohlcv:
            return {}
        
        try:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            indicators = {}
            
            if PANDAS_TA_AVAILABLE:
                # Use pandas-ta for enhanced indicators
                indicators.update(self._calculate_pandas_ta_indicators(df))
            else:
                # Fallback to basic calculations
                indicators.update(self._calculate_basic_indicators(df))
            
            if TULIPY_AVAILABLE:
                # Use tulipy for fast indicators
                indicators.update(self._calculate_tulipy_indicators(df))
            
            if self.debug:
                logger.debug(f"TradeKit enhanced indicators: {list(indicators.keys())}")
            
            return indicators
            
        except Exception as e:
            logger.error(f"Error calculating enhanced indicators: {e}")
            return {}
    
    def _calculate_pandas_ta_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate indicators using pandas-ta library."""
        indicators = {}
        
        try:
            # Trend indicators
            indicators['ema_9'] = ta.ema(df['close'], length=9).iloc[-1] if len(df) >= 9 else None
            indicators['ema_21'] = ta.ema(df['close'], length=21).iloc[-1] if len(df) >= 21 else None
            indicators['sma_50'] = ta.sma(df['close'], length=50).iloc[-1] if len(df) >= 50 else None
            
            # Momentum indicators
            indicators['rsi_14'] = ta.rsi(df['close'], length=14).iloc[-1] if len(df) >= 14 else None
            indicators['macd'] = ta.macd(df['close']).iloc[-1] if len(df) >= 26 else None
            indicators['stoch_k'] = ta.stoch(df['high'], df['low'], df['close']).iloc[-1] if len(df) >= 14 else None
            
            # Volatility indicators
            indicators['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14).iloc[-1] if len(df) >= 14 else None
            indicators['bb_upper'] = ta.bbands(df['close'], length=20)['BBU_20_2.0'].iloc[-1] if len(df) >= 20 else None
            indicators['bb_lower'] = ta.bbands(df['close'], length=20)['BBL_20_2.0'].iloc[-1] if len(df) >= 20 else None
            
            # Volume indicators
            indicators['obv'] = ta.obv(df['close'], df['volume']).iloc[-1]
            indicators['ad_s'] = ta.ad(df['high'], df['low'], df['close'], df['volume']).iloc[-1]
            
        except Exception as e:
            logger.error(f"Error in pandas-ta calculations: {e}")
        
        return indicators
    
    def _calculate_tulipy_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate fast indicators using tulipy library."""
        indicators = {}
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Tulipy provides very fast C-based indicators
            if len(close) >= 14:
                indicators['tulip_rsi'] = ti.rsi(close, 14)[-1]
            if len(close) >= 20:
                indicators['tulip_sma'] = ti.sma(close, 20)[-1]
            if len(close) >= 12:
                indicators['tulip_ema'] = ti.ema(close, 12)[-1]
            if len(high) >= 14 and len(low) >= 14:
                indicators['tulip_atr'] = ti.atr(high, low, close, 14)[-1]
                
        except Exception as e:
            logger.error(f"Error in tulipy calculations: {e}")
        
        return indicators
    
    def _calculate_basic_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate basic indicators as fallback."""
        indicators = {}
        
        try:
            close = df['close'].values
            
            # Simple moving averages
            if len(close) >= 9:
                indicators['ema_9'] = close[-9:].mean()
            if len(close) >= 21:
                indicators['ema_21'] = close[-21:].mean()
            if len(close) >= 50:
                indicators['sma_50'] = close[-50:].mean()
            
            # RSI approximation
            if len(close) >= 14:
                deltas = np.diff(close)
                gains = np.where(deltas > 0, deltas, 0)
                losses = np.where(deltas < 0, -deltas, 0)
                avg_gain = np.mean(gains[-14:])
                avg_loss = np.mean(losses[-14:])
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    indicators['rsi_14'] = 100 - (100 / (1 + rs))
                else:
                    indicators['rsi_14'] = 100
            
            # ATR approximation
            if len(df) >= 14:
                high_low = df['high'] - df['low']
                high_close = np.abs(df['high'] - df['close'].shift())
                low_close = np.abs(df['low'] - df['close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                indicators['atr_14'] = tr[-14:].mean()
            
        except Exception as e:
            logger.error(f"Error in basic indicator calculations: {e}")
        
        return indicators
    
    def analyze_order_book(self, order_book: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze order book for liquidity and spread quality.
        
        Args:
            order_book: Order book data from exchange
            
        Returns:
            Dictionary containing order book analysis results
        """
        if not self.enabled or not self.orderbook_analysis or not order_book:
            return {}
        
        try:
            analysis = {}
            
            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])
            
            if not bids or not asks:
                return analysis
            
            # Calculate spread
            best_bid = bids[0][0] if bids else 0
            best_ask = asks[0][0] if asks else 0
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100 if best_bid > 0 else 0
            
            analysis['spread'] = spread
            analysis['spread_pct'] = spread_pct
            analysis['spread_quality'] = self._assess_spread_quality(spread_pct)
            
            # Calculate order book depth
            bid_depth = sum(bid[1] for bid in bids[:10])  # Top 10 bids
            ask_depth = sum(ask[1] for ask in asks[:10])  # Top 10 asks
            total_depth = bid_depth + ask_depth
            
            analysis['bid_depth'] = bid_depth
            analysis['ask_depth'] = ask_depth
            analysis['total_depth'] = total_depth
            analysis['depth_quality'] = self._assess_liquidity_quality(total_depth)
            
            # Calculate bid/ask imbalance
            imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            analysis['bid_ask_imbalance'] = imbalance
            analysis['imbalance_signal'] = 'bullish' if imbalance > 0.1 else 'bearish' if imbalance < -0.1 else 'neutral'
            
            # Calculate market impact estimation
            analysis['market_impact'] = self._estimate_market_impact(bids, asks)
            
            if self.debug:
                logger.debug(f"Order book analysis: {analysis}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing order book: {e}")
            return {}
    
    def _assess_spread_quality(self, spread_pct: float) -> str:
        """Assess the quality of the spread."""
        if spread_pct < 0.05:
            return 'excellent'
        elif spread_pct < 0.1:
            return 'good'
        elif spread_pct < 0.2:
            return 'fair'
        else:
            return 'poor'
    
    def _assess_liquidity_quality(self, depth: float) -> str:
        """Assess the quality of liquidity."""
        if depth > 1000000:  # $1M+
            return 'excellent'
        elif depth > 500000:  # $500K+
            return 'good'
        elif depth > 100000:  # $100K+
            return 'fair'
        else:
            return 'poor'
    
    def _estimate_market_impact(self, bids: List, asks: List, trade_size: float = 10000) -> float:
        """Estimate market impact for a given trade size."""
        try:
            # Simple estimation: how much price would move for trade_size
            remaining = trade_size
            impact = 0.0
            price = asks[0][0] if asks else 0
            
            for ask in asks:
                if remaining <= 0:
                    break
                ask_price, ask_size = ask
                if ask_size >= remaining:
                    impact += (ask_price - price) / price
                    remaining = 0
                else:
                    impact += (ask_price - price) / price * (ask_size / trade_size)
                    remaining -= ask_size
            
            return abs(impact)
            
        except Exception as e:
            logger.error(f"Error estimating market impact: {e}")
            return 0.0
    
    def analyze_volatility(self, ohlcv: List[List[float]], current_price: float) -> Dict[str, Any]:
        """
        Analyze volatility suitability for trading.
        
        Args:
            ohlcv: OHLCV data
            current_price: Current price
            
        Returns:
            Dictionary containing volatility analysis
        """
        if not self.enabled or not self.volatility_analysis or not ohlcv:
            return {}
        
        try:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            analysis = {}
            
            # Calculate volatility metrics
            returns = df['close'].pct_change().drop()
            
            if len(returns) > 0:
                analysis['volatility_std'] = returns.std()
                analysis['volatility_mean'] = returns.mean()
                analysis['volatility_current'] = returns.iloc[-1] if len(returns) > 0 else 0
                
                # Annualized volatility (assuming hourly data)
                analysis['volatility_annualized'] = returns.std() * np.sqrt(24 * 365)
                
                # Volatility suitability assessment
                vol_pct = analysis['volatility_std'] * 100
                if 0.5 <= vol_pct <= 2.0:
                    analysis['volatility_suitability'] = 'optimal'
                elif 0.3 <= vol_pct < 0.5 or 2.0 < vol_pct <= 3.0:
                    analysis['volatility_suitability'] = 'acceptable'
                else:
                    analysis['volatility_suitability'] = 'poor'
            
            # ATR-based volatility
            if len(df) >= 14:
                atr = self._calculate_atr(df)
                if atr and current_price > 0:
                    atr_pct = (atr / current_price) * 100
                    analysis['atr_pct'] = atr_pct
                    analysis['atr_suitability'] = 'optimal' if 0.3 <= atr_pct <= 1.5 else 'acceptable' if 0.2 <= atr_pct <= 2.0 else 'poor'
            
            if self.debug:
                logger.debug(f"Volatility analysis: {analysis}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            return {}
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> Optional[float]:
        """Calculate Average True Range."""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            return tr.rolling(window=period).mean().iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating ATR: {e}")
            return None
    
    def calculate_enhanced_setup_score(self, 
                                      trend_score: float,
                                      momentum_score: float,
                                      volume_score: float,
                                      liquidity_score: float,
                                      volatility_score: float,
                                      support_resistance_score: float,
                                      relative_strength_score: float,
                                      btc_confirmation_score: float,
                                      ml_confirmation_score: float,
                                      trade_economics_score: float) -> Dict[str, Any]:
        """
        Calculate enhanced setup score using TradeKit weighted components.
        
        Args:
            Individual component scores (0-100)
            
        Returns:
            Dictionary containing total score and component breakdown
        """
        if not self.enabled:
            return {'total_score': 0, 'enabled': False}
        
        try:
            # Calculate weighted score
            total_score = (
                trend_score * self.score_weights['trend'] +
                momentum_score * self.score_weights['momentum'] +
                volume_score * self.score_weights['volume'] +
                liquidity_score * self.score_weights['liquidity'] +
                volatility_score * self.score_weights['volatility'] +
                support_resistance_score * self.score_weights['support_resistance'] +
                relative_strength_score * self.score_weights['relative_strength'] +
                btc_confirmation_score * self.score_weights['btc_confirmation'] +
                ml_confirmation_score * self.score_weights['ml_confirmation'] +
                trade_economics_score * self.score_weights['trade_economics']
            ) / 100  # Normalize to 0-100
            
            result = {
                'total_score': total_score,
                'enabled': True,
                'meets_threshold': total_score >= self.min_score,
                'components': {
                    'trend': trend_score,
                    'momentum': momentum_score,
                    'volume': volume_score,
                    'liquidity': liquidity_score,
                    'volatility': volatility_score,
                    'support_resistance': support_resistance_score,
                    'relative_strength': relative_strength_score,
                    'btc_confirmation': btc_confirmation_score,
                    'ml_confirmation': ml_confirmation_score,
                    'trade_economics': trade_economics_score
                },
                'weights': self.score_weights,
                'threshold': self.min_score
            }
            
            if self.debug:
                logger.debug(f"Enhanced setup score: {total_score:.2f} (threshold: {self.min_score})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating enhanced setup score: {e}")
            return {'total_score': 0, 'enabled': True, 'error': str(e)}
    
    def calculate_trading_costs(self, 
                               entry_price: float,
                               exit_price: float,
                               position_size: float,
                               fee_rate: float = 0.005,  # 0.5% typical Coinbase fee
                               spread_estimate: float = 0.001) -> Dict[str, Any]:
        """
        Calculate comprehensive trading costs including fees, spread, and slippage.
        
        Args:
            entry_price: Entry price
            exit_price: Expected exit price
            position_size: Position size in base currency
            fee_rate: Trading fee rate (default 0.5%)
            spread_estimate: Estimated spread as percentage
            
        Returns:
            Dictionary containing cost analysis
        """
        if not self.enabled:
            return {}
        
        try:
            # Calculate gross profit
            gross_profit = (exit_price - entry_price) * position_size
            gross_profit_pct = ((exit_price - entry_price) / entry_price) * 100
            
            # Calculate fees (round-trip)
            entry_fee = entry_price * position_size * fee_rate
            exit_fee = exit_price * position_size * fee_rate
            total_fees = entry_fee + exit_fee
            total_fees_pct = (total_fees / (entry_price * position_size)) * 100
            
            # Calculate spread cost
            spread_cost = entry_price * position_size * spread_estimate
            spread_cost_pct = spread_estimate * 100
            
            # Estimate slippage (0.1% typical)
            slippage_rate = 0.001
            slippage_cost = entry_price * position_size * slippage_rate
            slippage_cost_pct = slippage_rate * 100
            
            # Calculate net profit
            total_costs = total_fees + spread_cost + slippage_cost
            net_profit = gross_profit - total_costs
            net_profit_pct = gross_profit_pct - total_fees_pct - spread_cost_pct - slippage_cost_pct
            
            result = {
                'gross_profit': gross_profit,
                'gross_profit_pct': gross_profit_pct,
                'total_fees': total_fees,
                'total_fees_pct': total_fees_pct,
                'spread_cost': spread_cost,
                'spread_cost_pct': spread_cost_pct,
                'slippage_cost': slippage_cost,
                'slippage_cost_pct': slippage_cost_pct,
                'total_costs': total_costs,
                'total_costs_pct': total_fees_pct + spread_cost_pct + slippage_cost_pct,
                'net_profit': net_profit,
                'net_profit_pct': net_profit_pct,
                'profitable': net_profit > 0
            }
            
            if self.debug:
                logger.debug(f"Trading costs analysis: net_profit_pct={net_profit_pct:.3f}%")
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating trading costs: {e}")
            return {}
    
    def is_available(self) -> bool:
        """Check if TradeKit components are available."""
        return self.enabled and (PANDAS_TA_AVAILABLE or TULIPY_AVAILABLE)
    
    def get_status(self) -> Dict[str, Any]:
        """Get TradeKit adapter status."""
        return {
            'enabled': self.enabled,
            'pandas_ta_available': PANDAS_TA_AVAILABLE,
            'tulipy_available': TULIPY_AVAILABLE,
            'backtesting_available': BACKTESTING_AVAILABLE,
            'min_score': self.min_score,
            'liquidity_filter': self.liquidity_filter,
            'orderbook_analysis': self.orderbook_analysis,
            'volatility_analysis': self.volatility_analysis,
            'score_weights': self.score_weights
        }


# Convenience function for creating adapter
def create_tradekit_adapter(config: Dict[str, Any]) -> TradeKitAdapter:
    """
    Factory function to create TradeKit adapter.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        TradeKitAdapter instance
    """
    return TradeKitAdapter(config)

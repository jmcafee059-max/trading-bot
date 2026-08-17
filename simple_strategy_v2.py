import logging
import logging.handlers
import ccxt
import pandas as pd
import numpy as np
import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from ml_models import MLTradingEnsemble, PricePredictionLSTM, SignalConfirmationRF, PatternRecognitionNN
from openai_market_analyzer import OpenAIMarketAnalyzer
from coin_scanner import CoinScanner
from enum import Enum

# State machine states
class TradingState(Enum):
    IDLE_SCANNING = "IDLE_SCANNING"
    ENTRY_SIGNAL = "ENTRY_SIGNAL"
    OPEN_POSITION = "OPEN_POSITION"
    MONITOR_POSITION = "MONITOR_POSITION"
    POSITION_CLOSED = "POSITION_CLOSED"
    RESET_STATE = "RESET_STATE"
    COOLDOWN = "COOLDOWN"

# Set up bot logger for centralized logging
bot_logger = logging.getLogger('bot')
bot_logger.setLevel(logging.INFO)

# Clear any existing handlers to prevent duplicates
bot_logger.handlers.clear()

# Add file handler to bot_logger (no rotation to avoid file access issues)
log_handler = logging.FileHandler('bot_logs.log')
log_handler.setLevel(logging.INFO)
log_handler.set_name('bot_file_handler')
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler.setFormatter(log_formatter)
bot_logger.addHandler(log_handler)

class SimpleRSIStrategy:
    def __init__(self, exchange, config):
        self.exchange = exchange
        self.config = config
        self.symbol = config.get('symbol', 'BTC/USDC')
        
        # Coin scanner integration
        self.enable_coin_scanner = config.get('enable_coin_scanner', False)
        self.coin_scanner = None
        self.scan_interval = config.get('scan_interval_minutes', 5)
        self.min_edge_score = config.get('min_edge_score', 2.0)
        self.last_scan_time = 0
        
        if self.enable_coin_scanner and self.symbol == 'AUTO':
            self.coin_scanner = CoinScanner()
            bot_logger.info("Coin scanner enabled - will automatically select best trading pair")
        
        # Strategy parameters from config
        self.rsi_period = config.get('rsi_period', 7)
        self.rsi_overbought = config.get('rsi_overbought', 65)
        self.rsi_oversold = config.get('rsi_oversold', 35)
        self.take_profit_pct = config.get('take_profit_percent', 2.0)
        self.stop_loss_pct = config.get('stop_loss_percent', 0.5)
        
        # Complex strategy parameters
        self.ema_short = config.get('ema_short', 9)
        self.ema_long = config.get('ema_long', 21)
        self.sma_period = config.get('sma_period', 50)
        self.volume_threshold = config.get('volume_threshold', 1.5)
        self.momentum_period = config.get('momentum_period', 14)
        
        # Ultra-aggressive profit parameters
        self.max_position_size = config.get('max_position_size', 0.5)  # Use 50% of capital max for safety
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.15)  # Lowered from 0.3 for more frequent trades
        self.profit_multiplier = config.get('profit_multiplier', 1.0)  # Conservative multiplier
        self.aggressive_mode = config.get('aggressive_mode', True)  # Enable aggressive trading
        
        # ATR-based trading parameters
        self.use_atr_tp_sl = config.get('use_atr_tp_sl', True)
        self.atr_tp_multiplier_low = config.get('atr_tp_multiplier_low', 0.8)
        self.atr_tp_multiplier_normal = config.get('atr_tp_multiplier_normal', 1.5)
        self.atr_tp_multiplier_high = config.get('atr_tp_multiplier_high', 2.5)
        self.atr_sl_multiplier = config.get('atr_sl_multiplier', 1.2)
        self.atr_period = config.get('atr_period', 14)
        
        # Trading cost model parameters
        self.use_trading_cost_model = config.get('use_trading_cost_model', True)
        self.maker_fee_percent = config.get('maker_fee_percent', 0.4)
        self.taker_fee_percent = config.get('taker_fee_percent', 0.6)
        self.estimated_spread_percent = config.get('estimated_spread_percent', 0.02)
        self.estimated_slippage_percent = config.get('estimated_slippage_percent', 0.05)
        self.min_expected_net_profit = config.get('min_expected_net_profit', 0.2)
        
        # Trailing stop and breakeven parameters
        self.use_trailing_stop = config.get('use_trailing_stop', True)
        self.trailing_stop_activation_pct = config.get('trailing_stop_activation_pct', 0.15)  # Activate trailing stop after 0.15% profit
        self.trailing_stop_distance_pct = config.get('trailing_stop_distance_pct', 0.10)  # Trail 0.10% behind highest price
        self.use_breakeven = config.get('use_breakeven', True)
        self.breakeven_activation_pct = config.get('breakeven_activation_pct', 0.20)  # Move to breakeven after 0.20% profit
        self.breakeven_offset_pct = config.get('breakeven_offset_pct', 0.02)  # Small buffer above entry
        
        # Adaptive confidence threshold parameters
        self.use_adaptive_confidence = config.get('use_adaptive_confidence', True)
        self.normal_confidence_threshold = config.get('normal_confidence_threshold', 0.65)
        self.strong_trend_confidence_threshold = config.get('strong_trend_confidence_threshold', 0.60)
        self.choppy_market_confidence_threshold = config.get('choppy_market_confidence_threshold', 0.75)
        self.extreme_volatility_mode = config.get('extreme_volatility_mode', False)
        
        # Risk-based position sizing parameters
        self.use_risk_based_sizing = config.get('use_risk_based_sizing', True)
        self.risk_per_trade_percent = config.get('risk_per_trade_percent', 0.5)
        self.max_position_risk_percent = config.get('max_position_risk_percent', 2.0)
        
        # BTC market weather indicator parameters
        self.use_btc_weather = config.get('use_btc_weather', True)
        self.btc_weather_weight = config.get('btc_weather_weight', 0.3)
        
        # Relative strength parameters
        self.use_relative_strength = config.get('use_relative_strength', True)
        self.relative_strength_weight = config.get('relative_strength_weight', 0.2)
        
        # 100-point setup score system parameters
        self.use_setup_score = config.get('use_setup_score', True)
        self.min_setup_score = config.get('min_setup_score', 70)
        
        # Partial profit taking parameters
        self.use_partial_profit_taking = config.get('use_partial_profit_taking', True)
        self.first_tp_percent = config.get('first_tp_percent', 0.5)
        self.second_tp_percent = config.get('second_tp_percent', 1.0)
        self.third_tp_percent = config.get('third_tp_percent', 2.0)
        self.partial_tps_taken = []  # Track which partial TPs have been taken
        
        # Don't trade engine parameters
        self.use_dont_trade_engine = config.get('use_dont_trade_engine', True)
        self.max_consecutive_losses = config.get('max_consecutive_losses', 3)
        self.cooling_off_period = config.get('cooling_off_period_minutes', 30)
        self.min_liquidity_threshold = config.get('min_liquidity_threshold', 1000000)
        self.last_loss_time = 0
        
        # Performance tracking for adaptive optimization
        self.recent_trades = []
        self.win_rate = 0.5
        self.avg_profit = 0
        self.avg_loss = 0
        self.sharpe_ratio = 0
        
        # Machine Learning Integration
        self.ml_ensemble = MLTradingEnsemble()
        self.ml_enabled = config.get('ml_enabled', True)
        self.use_ml_signals = config.get('use_ml_signals', True)
        self.ml_only = config.get('ml_only', False)
        
        # OpenAI Integration
        self.openai_analyzer = OpenAIMarketAnalyzer()
        self.openai_enabled = config.get('openai_enabled', False)
        self.last_ai_signal = None
        self.last_ai_confidence = None
        bot_logger.info(f"OpenAI enabled: {self.openai_enabled}")
        bot_logger.info(f"OpenAI analyzer enabled: {self.openai_analyzer.enabled}")
        
        # Try to load pre-trained ML models
        if self.ml_enabled:
            try:
                self.ml_ensemble.load_all()
                bot_logger.info("ML models loaded successfully")
            except Exception as e:
                bot_logger.warning(f"Could not load ML models: {e}. Will train when data available.")
        
        # Capital management
        self.starting_capital = config.get('starting_capital', 18)
        self.current_capital = self.starting_capital
        self.initial_capital = self.starting_capital
        self.capital_percentage = config.get('capital_percentage', 90)
        
        # Volatility multiplier from config (reduced to avoid insufficient funds)
        self.VOLATILITY_MULTIPLIER = config.get('volatility_multiplier', 2)
        
        # Trading state
        symbol = config.get('symbol', 'BTC/USDC')
        if '/' in symbol:
            self.currency_symbol = symbol.split('/')[1]
        elif '-' in symbol:
            self.currency_symbol = symbol.split('-')[1]
        else:
            self.currency_symbol = 'USD'  # fallback
        self.price_history = []
        self.volume_history = []  # Track volume for ETH strategy
        # Support simultaneous long and short positions
        self.long_position = None  # True if long position open
        self.short_position = None  # True if short position open
        self.trade_count = 0
        self.profit_loss = 0.0
        self.last_buy_price = None
        self.last_short_price = None  # Track short entry price
        self.long_position_size = 0.0  # Separate size for long
        self.short_position_size = 0.0  # Separate size for short
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.highest_price_since_buy = None
        self.lowest_price_since_short = None  # Track lowest price for short trailing stops
        self.trade_history = []
        self.best_trade_profit = 0.0
        self.worst_trade_loss = 0.0
        
        # Trailing stop and breakeven tracking
        self.trailing_stop_price = None  # Current trailing stop level for long
        self.breakeven_triggered = False  # Whether breakeven has been triggered
        self.short_trailing_stop_price = None  # Current trailing stop level for short
        self.short_breakeven_triggered = False
        
        # Short trading configuration
        self.enable_short_trading = config.get('enable_short_trading', False)
        bot_logger.info(f"Short trading enabled: {self.enable_short_trading}")
        
        # ETH-specific strategy parameters
        symbol_normalized = symbol.replace('/', '-').upper()
        self.is_eth_strategy = (symbol_normalized == 'ETH-USDC')
        self.is_sol_strategy = (symbol_normalized == 'SOL-USDC')
        bot_logger.info(f"Strategy Detection: Symbol='{symbol}', Normalized='{symbol_normalized}', Is_ETH={self.is_eth_strategy}, Is_SOL={self.is_sol_strategy}")
        self.eth_resistance_levels = [1900, 1922, 1950]  # Key resistance levels for ETH
        self.eth_support_levels = [1862, 1883, 1850]  # Key support levels for ETH
        self.eth_rsi_preferred_zone = (30, 70)  # Preferred RSI zone for ETH longs (widened from 45-65)
        self.eth_rsi_overbought = 70  # Overbought threshold for ETH
        self.eth_min_setup_score = 70  # Minimum score for ETH trades (lowered from 80)
        
        # SOL-specific strategy parameters (momentum scalping)
        self.sol_tp_min = 0.25  # Minimum take profit for SOL
        self.sol_tp_max = 0.40  # Maximum take profit for SOL
        self.sol_sl_min = 0.20  # Minimum stop loss for SOL
        self.sol_sl_max = 0.25  # Maximum stop loss for SOL
        self.sol_rsi_preferred_zone = (45, 65)  # Preferred RSI zone for SOL
        self.sol_rsi_overbought = 70  # Overbought threshold for SOL
        self.sol_rsi_oversold = 30  # Oversold threshold for SOL
        self.sol_min_setup_score = 25  # Minimum score for SOL trades (25% for frequent opportunities)
        self.sol_min_liquidity = 1000000  # Minimum liquidity for SOL (1M USDC)
        self.sol_max_spread_pct = 0.05  # Maximum spread percentage for SOL
        self.sol_resistance_levels = [145, 150, 155]  # Key resistance levels for SOL
        self.sol_support_levels = [135, 130, 125]  # Key support levels for SOL
        
        # SOL short-specific parameters
        self.sol_short_tp_min = 0.25  # Minimum take profit for SOL shorts
        self.sol_short_tp_max = 0.40  # Maximum take profit for SOL shorts
        self.sol_short_sl_min = 0.20  # Minimum stop loss for SOL shorts
        self.sol_short_sl_max = 0.25  # Maximum stop loss for SOL shorts
        self.sol_short_rsi_preferred_zone = (30, 55)  # Preferred RSI zone for SOL shorts (oversold to neutral)
        self.sol_short_min_setup_score = 25  # Minimum score for SOL short trades (25% for frequent opportunities)
        
        # State machine
        self.trading_state = TradingState.IDLE_SCANNING
        self.last_state_transition_time = time.time()
        self.cooldown_end_time = 0
        self.last_entry_signal_time = 0
        
        # Load saved state
        self.load_capital_state()
        
        bot_logger.info(f"Strategy initialized with {self.VOLATILITY_MULTIPLIER}x volatility multiplier")
        bot_logger.info(f"Initial state: {self.trading_state.value}")
    
    def transition_state(self, new_state, reason=""):
        """Transition to a new state with logging"""
        old_state = self.trading_state
        self.trading_state = new_state
        self.last_state_transition_time = time.time()
        bot_logger.info(f"STATE TRANSITION: {old_state.value} → {new_state.value} | Reason: {reason} | Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def load_capital_state(self):
        """Load capital state from file for persistence"""
        state_file = 'capital_state.json'
        try:
            # If starting_capital is 'auto', fetch actual balance from exchange
            if self.starting_capital == 'auto':
                try:
                    balance = self.exchange.fetch_balance()
                    # Extract quote currency from symbol (e.g., USDC from UNI-USDC)
                    quote_currency = self.symbol.split('/')[1] if '/' in self.symbol else self.symbol.split('-')[1]
                    actual_balance = balance.get(quote_currency, {}).get('free', 0)
                    if actual_balance > 0:
                        self.current_capital = actual_balance
                        bot_logger.info(f"✓ Coinbase state verified: {quote_currency} balance = {self.currency_symbol}{self.current_capital:.2f}")
                        
                        # Verify against saved state for consistency (Railway restart detection)
                        if os.path.exists(state_file):
                            with open(state_file, 'r') as f:
                                saved_state = json.load(f)
                                saved_capital = saved_state.get('current_capital', 0)
                                if abs(actual_balance - saved_capital) > 1.0:  # More than $1 difference
                                    bot_logger.warning(f"⚠ Coinbase balance ({actual_balance:.2f}) differs from saved state ({saved_capital:.2f}) - possible Railway restart or external trade")
                    else:
                        # Fallback to saved state if balance is 0
                        if os.path.exists(state_file):
                            with open(state_file, 'r') as f:
                                state = json.load(f)
                                self.current_capital = state.get('current_capital', 18)
                                bot_logger.warning(f"Zero balance detected, using saved state: {self.currency_symbol}{self.current_capital:.2f}")
                        else:
                            self.current_capital = 18  # Default fallback
                except Exception as e:
                    bot_logger.error(f"Error fetching balance from Coinbase: {e}")
                    # Fallback to saved state
                    if os.path.exists(state_file):
                        with open(state_file, 'r') as f:
                            state = json.load(f)
                            self.current_capital = state.get('current_capital', 18)
                            bot_logger.warning(f"Coinbase API unavailable, using saved state: {self.currency_symbol}{self.current_capital:.2f}")
                    else:
                        self.current_capital = 18  # Default fallback
            else:
                # Use saved state if starting_capital is not auto
                if os.path.exists(state_file):
                    with open(state_file, 'r') as f:
                        state = json.load(f)
                        self.current_capital = state.get('current_capital', self.starting_capital)
                        self.trade_count = state.get('trade_count', 0)
                        self.profit_loss = state.get('profit_loss', 0.0)
                        self.consecutive_wins = state.get('consecutive_wins', 0)
                        self.consecutive_losses = state.get('consecutive_losses', 0)
                        bot_logger.info(f"Loaded capital state: {self.currency_symbol}{self.current_capital:.2f}, Trades: {self.trade_count}")
                else:
                    self.current_capital = self.starting_capital
        except Exception as e:
            bot_logger.warning(f"Could not load capital state: {e}")
            self.current_capital = self.starting_capital if self.starting_capital != 'auto' else 18
    
    def save_capital_state(self):
        """Save capital state to file"""
        state_file = 'capital_state.json'
        try:
            state = {
                'current_capital': self.current_capital,
                'trade_count': self.trade_count,
                'profit_loss': self.profit_loss,
                'consecutive_wins': self.consecutive_wins,
                'consecutive_losses': self.consecutive_losses
            }
            with open(state_file, 'w') as f:
                json.dump(state, f)
        except Exception as e:
            bot_logger.warning(f"Could not save capital state: {e}")
    
    def calculate_setup_score(self, rsi, trend, ema_short, ema_long, macd_bullish, price_near_lower, 
                             btc_weather, relative_strength, ml_buy_score, market_regime):
        """Calculate 100-point setup score for trade quality assessment"""
        if not self.use_setup_score:
            return 100  # Disable scoring if not enabled
        
        score = 0
        max_score = 100
        
        # RSI Score (0-15 points)
        if rsi < 30:
            score += 15  # Strongly oversold
        elif rsi < 40:
            score += 10  # Moderately oversold
        elif rsi < 50:
            score += 5   # Slightly oversold
        elif rsi > 70:
            score -= 10  # Overbought - penalty
        
        # Trend Score (0-15 points)
        if trend == "BULLISH":
            score += 15
        elif trend == "NEUTRAL":
            score += 5
        elif trend == "BEARISH":
            score -= 10
        
        # EMA Crossover Score (0-10 points)
        if ema_short > ema_long:
            score += 10
        else:
            score -= 5
        
        # MACD Score (0-10 points)
        if macd_bullish:
            score += 10
        else:
            score -= 5
        
        # Price Position Score (0-10 points)
        if price_near_lower:
            score += 10  # Near support
        else:
            score += 5   # Neutral position
        
        # BTC Market Weather Score (0-15 points)
        if btc_weather and btc_weather['signal'] == 'STRONG_BULLISH':
            score += 15
        elif btc_weather and btc_weather['signal'] == 'BULLISH':
            score += 10
        elif btc_weather and btc_weather['signal'] == 'NEUTRAL':
            score += 5
        elif btc_weather and btc_weather['signal'] == 'BEARISH':
            score -= 5
        elif btc_weather and btc_weather['signal'] == 'STRONG_BEARISH':
            score -= 15
        
        # Relative Strength Score (0-10 points)
        if relative_strength and relative_strength['signal'] == 'STRONG_OUTPERFORMING':
            score += 10
        elif relative_strength and relative_strength['signal'] == 'OUTPERFORMING':
            score += 7
        elif relative_strength and relative_strength['signal'] == 'NEUTRAL':
            score += 3
        elif relative_strength and relative_strength['signal'] == 'UNDERPERFORMING':
            score -= 3
        elif relative_strength and relative_strength['signal'] == 'STRONG_UNDERPERFORMING':
            score -= 7
        
        # ML Signal Score (0-15 points)
        if ml_buy_score >= 5:
            score += 15  # Strong ML signal
        elif ml_buy_score >= 3:
            score += 10  # Moderate ML signal
        elif ml_buy_score >= 2:
            score += 5   # Weak ML signal
        elif ml_buy_score < 0:
            score -= 10  # Bearish ML signal
        
        # Market Regime Score (0-10 points)
        if 'TRENDING_UP' in market_regime:
            score += 10
        elif 'TRENDING' in market_regime:
            score += 5
        elif 'RANGING' in market_regime:
            score += 0
        elif 'SIDEWAYS' in market_regime:
            score -= 5
        
        # Ensure score is within bounds
        score = max(0, min(100, score))
        
        bot_logger.info(f"Setup Score: {score}/100 (RSI={rsi:.1f}, Trend={trend}, BTC={btc_weather['signal'] if btc_weather else 'N/A'}, RS={relative_strength['signal'] if relative_strength else 'N/A'}, ML={ml_buy_score})")
        
        return score
    
    def get_relative_strength(self, current_price):
        """Calculate relative strength of current pair vs BTC"""
        if not self.use_relative_strength:
            return {'strength': 0, 'signal': 'NEUTRAL'}
        
        try:
            import yfinance as yf
            
            # Get current pair data
            current_symbol = self.symbol.replace('/', '-')
            if '-USDC' in current_symbol:
                current_symbol = current_symbol.replace('-USDC', '-USD')
            
            current_ticker = yf.Ticker(current_symbol)
            current_data = current_ticker.history(period='2d', interval='1h')
            
            # Get BTC data
            btc_ticker = yf.Ticker('BTC-USD')
            btc_data = btc_ticker.history(period='2d', interval='1h')
            
            if current_data.empty or btc_data.empty or len(current_data) < 5 or len(btc_data) < 5:
                return {'strength': 0, 'signal': 'NEUTRAL'}
            
            # Calculate returns for current pair
            current_prices = current_data['Close'].tolist()
            current_return_5h = ((current_prices[-1] - current_prices[-5]) / current_prices[-5]) * 100 if len(current_prices) >= 5 else 0
            
            # Calculate returns for BTC
            btc_prices = btc_data['Close'].tolist()
            btc_return_5h = ((btc_prices[-1] - btc_prices[-5]) / btc_prices[-5]) * 100 if len(btc_prices) >= 5 else 0
            
            # Calculate relative strength
            relative_strength = current_return_5h - btc_return_5h
            
            # Determine signal
            if relative_strength > 1.0:
                rs_signal = 'STRONG_OUTPERFORMING'
            elif relative_strength > 0.5:
                rs_signal = 'OUTPERFORMING'
            elif relative_strength < -1.0:
                rs_signal = 'STRONG_UNDERPERFORMING'
            elif relative_strength < -0.5:
                rs_signal = 'UNDERPERFORMING'
            else:
                rs_signal = 'NEUTRAL'
            
            rs_data = {
                'strength': relative_strength,
                'signal': rs_signal,
                'current_return': current_return_5h,
                'btc_return': btc_return_5h
            }
            
            bot_logger.info(f"Relative Strength: {rs_signal} ({relative_strength:+.2f}% vs BTC)")
            
            return rs_data
            
        except Exception as e:
            bot_logger.warning(f"Failed to calculate relative strength: {e}")
            return {'strength': 0, 'signal': 'NEUTRAL'}
    
    def get_btc_market_weather(self):
        """Get BTC market conditions as a weather indicator for overall crypto market"""
        if not self.use_btc_weather:
            return {'trend': 'NEUTRAL', 'momentum': 0, 'volatility': 0, 'signal': 'NEUTRAL'}
        
        try:
            import yfinance as yf
            btc_ticker = yf.Ticker('BTC-USD')
            btc_data = btc_ticker.history(period='2d', interval='1h')
            
            if btc_data.empty:
                return {'trend': 'NEUTRAL', 'momentum': 0, 'volatility': 0, 'signal': 'NEUTRAL'}
            
            btc_prices = btc_data['Close'].tolist()
            
            # Calculate BTC trend
            if len(btc_prices) >= 24:
                btc_change_24h = ((btc_prices[-1] - btc_prices[-24]) / btc_prices[-24]) * 100
                btc_change_5h = ((btc_prices[-1] - btc_prices[-5]) / btc_prices[-5]) * 100 if len(btc_prices) >= 5 else 0
                
                # Determine BTC trend
                if btc_change_24h > 2:
                    btc_trend = 'BULLISH'
                elif btc_change_24h < -2:
                    btc_trend = 'BEARISH'
                else:
                    btc_trend = 'NEUTRAL'
                
                # Calculate BTC volatility
                btc_volatility = (btc_data['Close'].tail(24).std() / btc_data['Close'].tail(24).mean()) * 100
                
                # Determine BTC signal
                if btc_trend == 'BULLISH' and btc_change_5h > 0.5:
                    btc_signal = 'STRONG_BULLISH'
                elif btc_trend == 'BULLISH':
                    btc_signal = 'BULLISH'
                elif btc_trend == 'BEARISH' and btc_change_5h < -0.5:
                    btc_signal = 'STRONG_BEARISH'
                elif btc_trend == 'BEARISH':
                    btc_signal = 'BEARISH'
                else:
                    btc_signal = 'NEUTRAL'
                
                weather = {
                    'trend': btc_trend,
                    'momentum': btc_change_5h,
                    'volatility': btc_volatility,
                    'signal': btc_signal,
                    'change_24h': btc_change_24h
                }
                
                bot_logger.info(f"BTC Weather: {btc_signal} (24h: {btc_change_24h:.2f}%, 5h: {btc_change_5h:.2f}%, Vol: {btc_volatility:.2f}%)")
                
                return weather
            else:
                return {'trend': 'NEUTRAL', 'momentum': 0, 'volatility': 0, 'signal': 'NEUTRAL'}
                
        except Exception as e:
            bot_logger.warning(f"Failed to get BTC market weather: {e}")
            return {'trend': 'NEUTRAL', 'momentum': 0, 'volatility': 0, 'signal': 'NEUTRAL'}
    
    def calculate_risk_based_position_size(self, current_price, stop_distance_pct):
        """Calculate position size based on risk per trade rather than fixed capital percentage"""
        if not self.use_risk_based_sizing:
            return None  # Fall back to traditional sizing
        
        # Calculate risk amount (0.5% of account by default)
        risk_amount = self.current_capital * (self.risk_per_trade_percent / 100)
        
        # Calculate position size based on stop distance
        if stop_distance_pct > 0:
            position_size = risk_amount / (current_price * (stop_distance_pct / 100))
        else:
            # Default to fixed percentage if no stop distance
            position_size = (self.current_capital * (self.capital_percentage / 100)) / current_price
        
        # Calculate actual position risk
        position_value = position_size * current_price
        position_risk_pct = (position_value / self.current_capital) * 100
        
        # Cap at maximum position risk
        if position_risk_pct > self.max_position_risk_percent:
            position_size = (self.current_capital * (self.max_position_risk_percent / 100)) / current_price
            bot_logger.warning(f"Position size capped at {self.max_position_risk_percent}% due to risk limits")
        
        bot_logger.info(f"Risk-based sizing: Risk=${risk_amount:.2f} ({self.risk_per_trade_percent}%), Position=${position_value:.2f} ({position_risk_pct:.2f}%)")
        
        return position_size
    
    def calculate_position_size(self, current_price):
        """Calculate position size with dynamic sizing based on confidence and volatility"""
        # Try risk-based sizing first if enabled
        if self.use_atr_tp_sl:
            # Calculate stop distance for risk-based sizing
            _, stop_distance_pct = self.calculate_atr_tp_sl(current_price, "NORMAL")
            risk_based_size = self.calculate_risk_based_position_size(current_price, stop_distance_pct)
            
            if risk_based_size is not None:
                return max(risk_based_size, 0.00001)
        
        # Fall back to traditional sizing
        # Calculate base trade amount
        base_trade_amount = self.current_capital * (self.capital_percentage / 100)
        
        # Calculate volatility-adjusted position size
        if len(self.price_history) >= 14:
            atr = self.calculate_atr(self.price_history)
            volatility_ratio = atr / current_price if atr > 0 else 0.01
            # Higher volatility = smaller position size for risk management
            volatility_adjustment = max(0.5, min(2.0, 1 / (volatility_ratio * 10)))
            trade_amount = base_trade_amount * volatility_adjustment
        else:
            trade_amount = base_trade_amount
        
        # Apply aggressive mode multiplier if enabled
        if self.aggressive_mode:
            trade_amount *= 1.5
        
        # Cap at maximum position size
        max_allowed = self.current_capital * self.max_position_size
        trade_amount = min(trade_amount, max_allowed)
        
        # Calculate position size in base currency
        position_size = trade_amount / current_price
        
        # Minimum position size
        min_position = 0.00001
        
        bot_logger.info(f"=== POSITION CALCULATION ===")
        bot_logger.info(f"Current Capital: ${self.current_capital:.2f}")
        bot_logger.info(f"Capital Percentage: {self.capital_percentage}%")
        bot_logger.info(f"Base Trade Amount: ${base_trade_amount:.2f}")
        bot_logger.info(f"Adjusted Trade Amount: ${trade_amount:.2f}")
        bot_logger.info(f"Current Price: ${current_price:.2f}")
        bot_logger.info(f"Position Size: {position_size:.6f}")
        bot_logger.info(f"Expected Profit at {self.take_profit_pct}%: ${trade_amount * (self.take_profit_pct / 100):.2f}")
        bot_logger.info(f"=========================")
        
        return max(position_size, min_position)
    
    def calculate_rsi(self, prices):
        """Calculate RSI using pandas"""
        try:
            if len(prices) < 2:
                return 50
            df = pd.DataFrame({'price': prices})
            delta = df['price'].diff()
            
            period = max(2, min(self.rsi_period, len(prices) - 1))
            
            gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
            
            if loss.iloc[-1] == 0:
                if gain.iloc[-1] == 0:
                    return 50
                return 90
            
            rs = gain / loss
            rsiValues = 100 - (100 / (1 + rs))
            
            rsi_value = rsiValues.iloc[-1]
            if pd.isna(rsi_value):
                return 50
            if rsi_value > 95:
                return 95
            if rsi_value < 5:
                return 5
            return rsi_value
        except Exception as e:
            bot_logger.warning(f"RSI calculation failed: {e}")
            return 50
    
    def calculate_sma(self, prices, period):
        """Calculate SMA using pandas"""
        try:
            if len(prices) < period:
                return prices[-1] if prices else None
            df = pd.DataFrame({'price': prices})
            sma_values = df['price'].rolling(window=period).mean()
            if len(sma_values) == 0:
                return prices[-1] if prices else None
            return sma_values.iloc[-1]
        except Exception as e:
            bot_logger.warning(f"SMA calculation failed: {e}")
            return prices[-1] if prices else None
    
    def calculate_ema(self, prices, period):
        """Calculate EMA using pandas"""
        try:
            if len(prices) < period:
                return prices[-1] if prices else None
            df = pd.DataFrame({'price': prices})
            ema_values = df['price'].ewm(span=period, adjust=False).mean()
            if len(ema_values) == 0:
                return prices[-1] if prices else None
            return ema_values.iloc[-1]
        except Exception as e:
            bot_logger.warning(f"EMA calculation failed: {e}")
            return prices[-1] if prices else None
    
    def calculate_macd(self, prices):
        """Calculate MACD using pandas"""
        try:
            if len(prices) < 26:
                return None, None, None
            df = pd.DataFrame({'price': prices})
            
            # Calculate EMAs
            ema_fast = df['price'].ewm(span=12, adjust=False).mean()
            ema_slow = df['price'].ewm(span=26, adjust=False).mean()
            
            # MACD line
            macd_line = ema_fast - ema_slow
            
            # Signal line
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            # Histogram
            histogram = macd_line - signal_line
            
            if len(macd_line) == 0 or len(signal_line) == 0 or len(histogram) == 0:
                return None, None, None
                
            return macd_line.iloc[-1], signal_line.iloc[-1], histogram.iloc[-1]
        except Exception as e:
            bot_logger.warning(f"MACD calculation failed: {e}")
            return None, None, None
    
    def calculate_momentum(self, prices, period):
        """Calculate momentum indicator"""
        try:
            if len(prices) < period + 1:
                return 0
            return prices[-1] - prices[-period - 1]
        except Exception as e:
            bot_logger.warning(f"Momentum calculation failed: {e}")
            return 0
    
    def calculate_atr(self, prices, period=14):
        """Calculate Average True Range for volatility-based trading"""
        try:
            if len(prices) < period + 1:
                return 0.0
                
            df = pd.DataFrame({'price': prices})
            
            # Calculate True Range
            high = df['price']
            low = df['price']
            close = df['price'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.0
        except Exception as e:
            bot_logger.warning(f"ATR calculation failed: {e}")
            return 0.0
    
    def calculate_trading_costs(self, trade_amount, is_maker=True):
        """Calculate total trading costs including fees, spread, and slippage"""
        if not self.use_trading_cost_model:
            return 0.0
            
        fee_percent = self.maker_fee_percent if is_maker else self.taker_fee_percent
        fee_amount = trade_amount * (fee_percent / 100)
        spread_amount = trade_amount * (self.estimated_spread_percent / 100)
        slippage_amount = trade_amount * (self.estimated_slippage_percent / 100)
        
        total_cost = fee_amount + spread_amount + slippage_amount
        cost_percent = (total_cost / trade_amount) * 100 if trade_amount > 0 else 0
        
        bot_logger.info(f"Trading costs: Fee={fee_amount:.2f} ({fee_percent}%), Spread={spread_amount:.2f}, Slippage={slippage_amount:.2f}, Total={total_cost:.2f} ({cost_percent:.2f}%)")
        
        return total_cost
    
    def calculate_expected_net_profit(self, entry_price, target_price, trade_amount, is_maker=True):
        """Calculate expected net profit after all trading costs"""
        gross_profit = (target_price - entry_price) / entry_price * trade_amount
        entry_cost = self.calculate_trading_costs(trade_amount, is_maker)
        exit_cost = self.calculate_trading_costs(trade_amount, False)  # Assume taker for exit
        
        net_profit = gross_profit - entry_cost - exit_cost
        net_profit_percent = (net_profit / trade_amount) * 100 if trade_amount > 0 else 0
        
        bot_logger.info(f"Expected net profit: Gross=${gross_profit:.2f}, Entry Cost=${entry_cost:.2f}, Exit Cost=${exit_cost:.2f}, Net=${net_profit:.2f} ({net_profit_percent:.2f}%)")
        
        return net_profit, net_profit_percent
    
    def get_adaptive_confidence_threshold(self, market_regime, trend_strength):
        """Calculate adaptive confidence threshold based on market conditions"""
        if not self.use_adaptive_confidence:
            return self.min_confidence_threshold
        
        # Extreme volatility - no trading
        if self.extreme_volatility_mode:
            bot_logger.warning("Extreme volatility mode - no trading allowed")
            return 1.0  # Impossible threshold
        
        # Strong trend - lower threshold (more trades)
        if 'TRENDING' in market_regime and trend_strength > 0.7:
            threshold = self.strong_trend_confidence_threshold
            bot_logger.info(f"Strong trend detected - using lower threshold: {threshold}")
        
        # Choppy/ranging market - higher threshold (fewer trades)
        elif 'RANGING' in market_regime or 'SIDEWAYS' in market_regime:
            threshold = self.choppy_market_confidence_threshold
            bot_logger.info(f"Choppy market detected - using higher threshold: {threshold}")
        
        # Normal conditions
        else:
            threshold = self.normal_confidence_threshold
            bot_logger.info(f"Normal market conditions - using standard threshold: {threshold}")
        
        return threshold
    
    def is_trade_profitable(self, entry_price, target_price, trade_amount):
        """Check if trade is profitable after accounting for costs"""
        if not self.use_trading_cost_model:
            return True
            
        _, net_profit_percent = self.calculate_expected_net_profit(entry_price, target_price, trade_amount)
        
        is_profitable = net_profit_percent >= self.min_expected_net_profit
        
        if not is_profitable:
            bot_logger.warning(f"Trade not profitable after costs: Net profit {net_profit_percent:.2f}% < Minimum {self.min_expected_net_profit}%")
        
        return is_profitable
    
    def calculate_atr_tp_sl(self, current_price, volatility_regime):
        """Calculate ATR-based take profit and stop loss based on market volatility"""
        if not self.use_atr_tp_sl or len(self.price_history) < self.atr_period:
            # Fall back to fixed percentages
            return self.take_profit_pct, self.stop_loss_pct
            
        atr = self.calculate_atr(self.price_history, self.atr_period)
        if atr == 0:
            return self.take_profit_pct, self.stop_loss_pct
        
        # Determine volatility regime
        atr_pct = (atr / current_price) * 100
        
        if 'LOW_VOL' in volatility_regime:
            tp_multiplier = self.atr_tp_multiplier_low
        elif 'HIGH_VOL' in volatility_regime:
            tp_multiplier = self.atr_tp_multiplier_high
        else:
            tp_multiplier = self.atr_tp_multiplier_normal
        
        # Calculate TP and SL as percentages
        atr_tp_pct = (atr * tp_multiplier / current_price) * 100
        atr_sl_pct = (atr * self.atr_sl_multiplier / current_price) * 100
        
        bot_logger.info(f"ATR-based TP/SL: TP={atr_tp_pct:.2f}%, SL={atr_sl_pct:.2f}% (ATR={atr:.4f}, regime={volatility_regime})")
        
        return atr_tp_pct, atr_sl_pct
    
    def calculate_kelly_position_size(self, win_rate, avg_win, avg_loss):
        """Calculate optimal position size using Kelly Criterion"""
        if avg_loss == 0:
            return 0.5  # Conservative default
        
        kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        # Kelly can be aggressive, so use half-Kelly for safety
        half_kelly = max(0.1, min(0.5, kelly_fraction * 0.5))
        return half_kelly
    
    def detect_market_regime(self, prices):
        """Detect market regime (trending, ranging, volatile)"""
        if len(prices) < 20:
            return "NEUTRAL"
        
        try:
            df = pd.DataFrame({'price': prices})
            sma_short = df['price'].rolling(window=10).mean().iloc[-1]
            sma_long = df['price'].rolling(window=20).mean().iloc[-1]
            std = df['price'].rolling(window=20).std().iloc[-1]
            current_price = prices[-1]
            
            # Trend detection
            if sma_short > sma_long and current_price > sma_short:
                regime = "TRENDING_UP"
            elif sma_short < sma_long and current_price < sma_short:
                regime = "TRENDING_DOWN"
            else:
                regime = "RANGING"
            
            # Volatility detection
            volatility = std / current_price if current_price > 0 else 0
            if volatility > 0.05:
                regime += "_HIGH_VOL"
            elif volatility < 0.01:
                regime += "_LOW_VOL"
            
            return regime
        except Exception as e:
            bot_logger.warning(f"Market regime detection failed: {e}")
            return "NEUTRAL"
    
    def detect_engulfing_pattern(self, prices):
        """Detect engulfing candle patterns - highly reliable for gold trading"""
        try:
            if len(prices) < 2:
                return None
            
            # Get last two candles
            current_price = prices[-1]
            previous_price = prices[-2]
            
            # Need OHLC data for proper candle analysis
            # Since we only have closing prices, we'll use a simplified approach
            # based on price direction and magnitude
            
            # Calculate price changes
            current_change = current_price - prices[-2] if len(prices) >= 2 else 0
            previous_change = prices[-2] - prices[-3] if len(prices) >= 3 else 0
            
            # Bullish engulfing: previous was down, current is up with larger magnitude
            if previous_change < 0 and current_change > 0:
                if abs(current_change) > abs(previous_change):
                    return "BULLISH_ENGULFING"
            
            # Bearish engulfing: previous was up, current is down with larger magnitude
            elif previous_change > 0 and current_change < 0:
                if abs(current_change) > abs(previous_change):
                    return "BEARISH_ENGULFING"
            
            return None
        except Exception as e:
            bot_logger.warning(f"Engulfing pattern detection failed: {e}")
            return None
    
    def detect_pin_bar(self, prices):
        """Detect pin bar (rejection candle) patterns - clean signals for gold"""
        try:
            if len(prices) < 3:
                return None
            
            # Simplified pin bar detection using price changes
            # In real implementation, would need OHLC data for wick analysis
            current = prices[-1]
            previous = prices[-2]
            before_previous = prices[-3]
            
            # Calculate recent volatility
            recent_range = max(prices[-10:]) - min(prices[-10:]) if len(prices) >= 10 else 0
            
            if recent_range == 0:
                return None
            
            # Bullish pin bar: price dropped then recovered (lower wick)
            if before_previous > previous and current > previous:
                rejection_ratio = (before_previous - previous) / recent_range
                if rejection_ratio > 0.3:  # Significant rejection
                    return "BULLISH_PIN_BAR"
            
            # Bearish pin bar: price rose then rejected (upper wick)
            elif before_previous < previous and current < previous:
                rejection_ratio = (previous - before_previous) / recent_range
                if rejection_ratio > 0.3:  # Significant rejection
                    return "BEARISH_PIN_BAR"
            
            return None
        except Exception as e:
            bot_logger.warning(f"Pin bar detection failed: {e}")
            return None
    
    def calculate_atr(self, prices, period=14):
        """Calculate Average True Range for volatility"""
        try:
            if len(prices) < period + 1:
                return 0
            df = pd.DataFrame({'price': prices})
            high = df['price']
            low = df['price'].shift(1)
            tr = pd.concat([high - low, (high - df['price'].shift(1)).abs(), (low - df['price'].shift(1)).abs()], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean()
            return atr.iloc[-1] if len(atr) > 0 else 0
        except Exception as e:
            bot_logger.warning(f"ATR calculation failed: {e}")
            return 0
    
    def calculate_volume_sma(self, volumes, period):
        """Calculate volume SMA"""
        try:
            if len(volumes) < period:
                return volumes[-1] if volumes else 0
            df = pd.DataFrame({'volume': volumes})
            volume_sma = df['volume'].rolling(window=period).mean()
            return volume_sma.iloc[-1] if len(volume_sma) > 0 else 0
        except Exception as e:
            bot_logger.warning(f"Volume SMA calculation failed: {e}")
            return 0
    
    def should_block_trade(self, current_price):
        """Don't trade engine: check if trading should be blocked based on various filters"""
        if not self.use_dont_trade_engine:
            return False, "Don't trade engine disabled"
        
        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return True, f"Too many consecutive losses ({self.consecutive_losses})"
        
        # Check cooling off period after losses
        if self.last_loss_time > 0:
            time_since_loss = time.time() - self.last_loss_time
            if time_since_loss < self.cooling_off_period * 60:
                remaining_time = self.cooling_off_period * 60 - time_since_loss
                return True, f"Cooling off period ({remaining_time/60:.1f} minutes remaining)"
        
        # Check minimum liquidity (placeholder - would need real volume data)
        # For now, assume liquidity is sufficient
        
        # Check if capital is too low
        if self.current_capital < 10:  # Minimum capital threshold
            return True, "Capital too low for trading"
        
        # Check if win rate is too low (recent performance)
        if len(self.recent_trades) >= 10:
            recent_wins = sum(1 for trade in self.recent_trades[-10:] if trade['profit'] > 0)
            recent_win_rate = recent_wins / 10
            if recent_win_rate < 0.2:  # Less than 20% win rate in last 10 trades
                return True, f"Recent win rate too low ({recent_win_rate*100:.0f}%)"
        
        return False, "All filters passed"
    
    def handle_partial_profit_taking(self, current_price):
        """Handle partial profit taking at predefined levels"""
        if not self.use_partial_profit_taking or self.current_position != 'long':
            return False
        
        if not self.last_buy_price:
            return False
        
        profit_pct = ((current_price - self.last_buy_price) / self.last_buy_price) * 100
        
        # First TP: 25% of position
        if 'first' not in self.partial_tps_taken and profit_pct >= self.first_tp_percent:
            partial_size = self.position_size * 0.25
            self.position_size -= partial_size
            self.partial_tps_taken.append('first')
            
            profit_amount = (current_price - self.last_buy_price) * partial_size
            self.current_capital += profit_amount
            
            bot_logger.info(f"First partial TP taken: Sold {partial_size:.6f} at {current_price:.4f} (Profit: ${profit_amount:.2f}, {profit_pct:.2f}%)")
            return True
        
        # Second TP: 25% of position
        elif 'second' not in self.partial_tps_taken and profit_pct >= self.second_tp_percent:
            partial_size = self.position_size * 0.25
            self.position_size -= partial_size
            self.partial_tps_taken.append('second')
            
            profit_amount = (current_price - self.last_buy_price) * partial_size
            self.current_capital += profit_amount
            
            bot_logger.info(f"Second partial TP taken: Sold {partial_size:.6f} at {current_price:.4f} (Profit: ${profit_amount:.2f}, {profit_pct:.2f}%)")
            return True
        
        # Third TP: 50% with trailing stop
        elif 'third' not in self.partial_tps_taken and profit_pct >= self.third_tp_percent:
            partial_size = self.position_size  # Remaining 50%
            self.position_size -= partial_size
            self.partial_tps_taken.append('third')
            
            profit_amount = (current_price - self.last_buy_price) * partial_size
            self.current_capital += profit_amount
            
            bot_logger.info(f"Third partial TP taken: Sold {partial_size:.6f} at {current_price:.4f} (Profit: ${profit_amount:.2f}, {profit_pct:.2f}%)")
            
            # Reset position after final partial TP
            self.current_position = None
            self.last_buy_price = None
            self.highest_price_since_buy = None
            self.partial_tps_taken = []
            
            return True
        
        return False
    
    def detect_trend(self, prices):
        """Detect overall trend using multiple indicators"""
        try:
            if len(prices) < self.sma_period:
                return "NEUTRAL"
            
            sma = self.calculate_sma(prices, self.sma_period)
            ema_short = self.calculate_ema(prices, self.ema_short)
            ema_long = self.calculate_ema(prices, self.ema_long)
            current_price = prices[-1]
            
            # Trend conditions
            bullish = current_price > sma and ema_short > ema_long
            bearish = current_price < sma and ema_short < ema_long
            
            if bullish:
                return "BULLISH"
            elif bearish:
                return "BEARISH"
            else:
                return "NEUTRAL"
        except Exception as e:
            bot_logger.warning(f"Trend detection failed: {e}")
            return "NEUTRAL"
    
    def calculate_bollinger_bands(self, prices):
        """Calculate Bollinger Bands using pandas"""
        try:
            if len(prices) < 20:
                return None, None, None
            df = pd.DataFrame({'price': prices})
            
            # Middle band (SMA)
            middle = df['price'].rolling(window=20).mean()
            
            # Standard deviation
            std = df['price'].rolling(window=20).std()
            
            # Upper and lower bands
            upper = middle + (std * 2)
            lower = middle - (std * 2)
            
            if len(middle) == 0 or len(upper) == 0 or len(lower) == 0:
                return None, None, None
                
            return upper.iloc[-1], middle.iloc[-1], lower.iloc[-1]
        except Exception as e:
            bot_logger.warning(f"Bollinger Bands calculation failed: {e}")
            return None, None, None
    
    def detect_rsi_divergence(self, prices, rsi_values):
        """Detect bullish RSI divergence (price lower low, RSI higher low)"""
        try:
            if len(prices) < 10 or len(rsi_values) < 10:
                return False
            
            # Get recent price lows
            recent_prices = prices[-10:]
            recent_rsi = rsi_values[-10:]
            
            # Find local lows
            price_lows = []
            rsi_lows = []
            
            for i in range(2, len(recent_prices) - 2):
                if (recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i-2] and
                    recent_prices[i] < recent_prices[i+1] and recent_prices[i] < recent_prices[i+2]):
                    price_lows.append((i, recent_prices[i]))
                    rsi_lows.append(recent_rsi[i])
            
            # Check for bullish divergence (price lower low, RSI higher low)
            if len(price_lows) >= 2:
                last_price_low = price_lows[-1][1]
                prev_price_low = price_lows[-2][1]
                last_rsi_low = rsi_lows[-1]
                prev_rsi_low = rsi_lows[-2]
                
                if last_price_low < prev_price_low and last_rsi_low > prev_rsi_low:
                    return True
            
            return False
        except Exception as e:
            bot_logger.warning(f"RSI divergence detection failed: {e}")
            return False
    
    def detect_support_resistance(self, prices):
        """Detect support and resistance levels"""
        try:
            if len(prices) < 20:
                return None, None
            
            recent_prices = prices[-20:]
            
            # Calculate support (recent lows) and resistance (recent highs)
            lows = []
            highs = []
            
            for i in range(2, len(recent_prices) - 2):
                if (recent_prices[i] < recent_prices[i-1] and recent_prices[i] < recent_prices[i-2] and
                    recent_prices[i] < recent_prices[i+1] and recent_prices[i] < recent_prices[i+2]):
                    lows.append(recent_prices[i])
                
                if (recent_prices[i] > recent_prices[i-1] and recent_prices[i] > recent_prices[i-2] and
                    recent_prices[i] > recent_prices[i+1] and recent_prices[i] > recent_prices[i+2]):
                    highs.append(recent_prices[i])
            
            support = sum(lows) / len(lows) if lows else None
            resistance = sum(highs) / len(highs) if highs else None
            
            return support, resistance
        except Exception as e:
            bot_logger.warning(f"Support/Resistance detection failed: {e}")
            return None, None
    
    def detect_eth_pullback(self, current_price, ema_short, ema_long, rsi):
        """Detect ETH pullback pattern for entry"""
        if not self.is_eth_strategy:
            return False, "Not ETH strategy"
        
        try:
            # Pullback conditions: price pulls back toward EMA20/EMA50
            price_near_ema_short = abs(current_price - ema_short) / ema_short < 0.01  # Within 1% of EMA20
            price_near_ema_long = abs(current_price - ema_long) / ema_long < 0.015  # Within 1.5% of EMA50
            
            # RSI in preferred zone (45-65)
            rsi_in_zone = self.eth_rsi_preferred_zone[0] <= rsi <= self.eth_rsi_preferred_zone[1]
            
            # EMA trend confirmation
            ema_trend_up = ema_short > ema_long
            
            # Price not excessively extended above EMA20
            price_not_extended = current_price < ema_short * 1.02  # Less than 2% above EMA20
            
            if price_near_ema_short and rsi_in_zone and ema_trend_up and price_not_extended:
                return True, "ETH pullback to EMA20 with RSI in preferred zone"
            elif price_near_ema_long and rsi_in_zone and ema_trend_up:
                return True, "ETH pullback to EMA50 with RSI in preferred zone"
            
            return False, "Pullback conditions not met"
        except Exception as e:
            bot_logger.warning(f"ETH pullback detection failed: {e}")
            return False, "Detection error"
    
    def detect_eth_breakout_retest(self, current_price, prices):
        """Detect ETH breakout and retest pattern"""
        if not self.is_eth_strategy:
            return False, "Not ETH strategy"
        
        try:
            if len(prices) < 10:
                return False, "Insufficient price history"
            
            recent_prices = prices[-10:]
            
            # Detect recent resistance breakout
            resistance, _ = self.detect_support_resistance(prices)
            if resistance is None:
                return False, "No resistance detected"
            
            # Check if price recently broke above resistance
            broke_out = any(p > resistance for p in recent_prices[-5:])
            
            # Check if price is retesting resistance (now support)
            near_retest = abs(current_price - resistance) / resistance < 0.01  # Within 1%
            
            # Check if support is holding (price bouncing off)
            if broke_out and near_retest and current_price >= resistance * 0.99:
                return True, f"ETH breakout retest at ${resistance:.2f} holding as support"
            
            return False, "Breakout retest pattern not detected"
        except Exception as e:
            bot_logger.warning(f"ETH breakout retest detection failed: {e}")
            return False, "Detection error"
    
    def check_eth_resistance_avoidance(self, current_price):
        """Check if ETH is near resistance to avoid chasing"""
        if not self.is_eth_strategy:
            return False, "Not ETH strategy"
        
        try:
            for level in self.eth_resistance_levels:
                distance = abs(current_price - level) / level
                if distance < 0.005:  # Within 0.5% of resistance (relaxed from 1%)
                    return True, f"ETH within 0.5% of resistance at ${level:.2f} - avoid chasing"
            
            return False, "Not near resistance"
        except Exception as e:
            bot_logger.warning(f"ETH resistance avoidance check failed: {e}")
            return False, "Check error"
    
    def check_volume_confirmation(self, current_volume=None):
        """Check volume confirmation for ETH strategy"""
        if not self.is_eth_strategy or current_volume is None:
            return True, "Volume check not applicable"
        
        try:
            if len(self.volume_history) < 20:
                return True, "Insufficient volume history - using default"
            
            avg_volume = sum(self.volume_history[-20:]) / 20
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            # Volume should be 1.2x average for breakout
            if volume_ratio >= 1.2:
                return True, f"Volume confirmed: {volume_ratio:.2f}x average"
            else:
                return False, f"Volume insufficient: {volume_ratio:.2f}x average (need 1.2x)"
        except Exception as e:
            bot_logger.warning(f"Volume confirmation check failed: {e}")
            return True, "Volume check error - proceeding"
    
    def calculate_eth_setup_score(self, rsi, ema_short, ema_long, macd_bullish, 
                                  price_near_lower, current_price, volume_confirmed,
                                  pullback_detected, breakout_retest_detected, near_resistance):
        """Calculate ETH-specific 100-point setup score"""
        if not self.is_eth_strategy:
            return 0
        
        score = 0
        
        # EMA trend (15 points)
        if ema_short > ema_long:
            score += 15
            if ema_short > ema_long * 1.01:  # Strong trend
                score += 3
        
        # MACD (15 points)
        if macd_bullish:
            score += 15
        
        # RSI (10 points)
        if self.eth_rsi_preferred_zone[0] <= rsi <= self.eth_rsi_preferred_zone[1]:
            score += 10  # Preferred zone
        elif rsi < self.eth_rsi_preferred_zone[0]:
            score += 5  # Oversold - acceptable
        elif rsi < self.eth_rsi_overbought:
            score += 3  # Elevated but not overbought
        else:
            score -= 10  # Overbought - penalty
        
        # Volume (15 points)
        if volume_confirmed:
            score += 15
        else:
            score += 5  # Volume not confirmed but not critical
        
        # Pullback/Pattern (15 points)
        if pullback_detected:
            score += 15
        elif breakout_retest_detected:
            score += 15
        elif price_near_lower:
            score += 10  # Near support
        else:
            score += 5  # No clear pattern
        
        # Price extension (10 points)
        price_extension = (current_price - ema_short) / ema_short
        if 0 <= price_extension <= 0.02:  # 0-2% above EMA20 - ideal
            score += 10
        elif -0.01 <= price_extension <= 0:  # At or slightly below EMA20 - good
            score += 8
        elif 0.02 < price_extension <= 0.05:  # 2-5% above - acceptable
            score += 5
        else:
            score -= 5  # Too extended
        
        # Resistance avoidance (15 points)
        if not near_resistance:
            score += 15
        else:
            score -= 20  # Heavy penalty for being near resistance
        
        # Momentum (10 points)
        if len(self.price_history) >= 3:
            momentum = self.price_history[-1] - self.price_history[-3]
            if momentum > 0:
                score += 10
            else:
                score += 3
        
        # BTC confirmation (5 points) - placeholder
        score += 5
        
        # ML probability (5 points) - placeholder
        score += 5
        
        return min(score, 100)  # Cap at 100
    
    def detect_sol_momentum_setup(self, current_price, ema_short, ema_long, rsi, macd_bullish):
        """Detect SOL momentum scalping setup"""
        if not self.is_sol_strategy:
            return False, "Not SOL strategy"
        
        try:
            reasons = []
            
            # EMA trend (1H direction proxy)
            if ema_short > ema_long:
                reasons.append("EMA bullish")
            
            # RSI in preferred zone
            if self.sol_rsi_preferred_zone[0] <= rsi <= self.sol_rsi_preferred_zone[1]:
                reasons.append("RSI in preferred zone")
            
            # MACD bullish
            if macd_bullish:
                reasons.append("MACD bullish")
            
            # Price not overbought
            if rsi < self.sol_rsi_overbought:
                reasons.append("Not overbought")
            
            if len(reasons) >= 3:
                return True, ", ".join(reasons)
            else:
                return False, f"Insufficient momentum signals: {', '.join(reasons) if reasons else 'none'}"
        except Exception as e:
            bot_logger.warning(f"SOL momentum detection failed: {e}")
            return False, "Detection error"
    
    def check_sol_liquidity_and_spread(self, current_price):
        """Check SOL liquidity and spread for trading"""
        if not self.is_sol_strategy:
            return True, "Not SOL strategy"
        
        try:
            # Fetch ticker for liquidity and spread data
            ticker = self.exchange.fetch_ticker(self.symbol)
            
            # Check liquidity (24h volume)
            volume_24h = ticker.get('quoteVolume', 0)
            if volume_24h is None or volume_24h == 0:
                # Allow trading if volume data unavailable (fallback to other checks)
                bot_logger.warning("Volume data unavailable - proceeding with caution")
                return True, "Volume data unavailable - proceeding"
            if volume_24h < self.sol_min_liquidity:
                return False, f"Insufficient liquidity: ${volume_24h:,.0f} < ${self.sol_min_liquidity:,.0f}"
            
            # Check spread
            bid = ticker.get('bid', 0)
            ask = ticker.get('ask', 0)
            if bid is None or ask is None or bid == 0 or ask == 0:
                return False, "Bid/ask data unavailable"
            spread_pct = ((ask - bid) / bid) * 100
            if spread_pct > self.sol_max_spread_pct:
                return False, f"Spread too wide: {spread_pct:.3f}% > {self.sol_max_spread_pct:.3f}%"
            
            return True, f"Liquidity OK: ${volume_24h:,.0f}, Spread OK"
        except Exception as e:
            bot_logger.warning(f"SOL liquidity/spread check failed: {e}")
            return False, "Check error"
    
    def calculate_sol_setup_score(self, rsi, ema_short, ema_long, macd_bullish, price_near_lower, current_price, volume_confirmed, momentum_detected, near_resistance, liquidity_ok):
        """Calculate SOL-specific setup score (100-point system)"""
        if not self.is_sol_strategy:
            return 100
        
        score = 0
        
        # EMA trend (20 points)
        if ema_short > ema_long:
            score += 20
        
        # MACD (20 points)
        if macd_bullish:
            score += 20
        
        # RSI (15 points)
        if self.sol_rsi_preferred_zone[0] <= rsi <= self.sol_rsi_preferred_zone[1]:
            score += 15
        elif rsi < self.sol_rsi_preferred_zone[0]:
            score += 5  # Oversold but could be bounce
        else:
            score += 0  # Too overbought
        
        # Volume (15 points)
        if volume_confirmed:
            score += 15
        else:
            score += 5
        
        # Momentum setup (15 points)
        if momentum_detected:
            score += 15
        else:
            score += 0
        
        # Price extension (10 points)
        if price_near_lower:
            score += 10
        else:
            score += 5
        
        # Resistance avoidance (10 points)
        if not near_resistance:
            score += 10
        else:
            score -= 15  # Penalty for being near resistance
        
        # Liquidity and spread (5 points)
        if liquidity_ok:
            score += 5
        else:
            score -= 10  # Heavy penalty for poor liquidity
        
        return min(score, 100)  # Cap at 100
    
    def detect_sol_short_setup(self, current_price, ema_short, ema_long, rsi, macd_bearish):
        """Detect SOL short momentum setup"""
        if not self.is_sol_strategy or not self.enable_short_trading:
            return False, "Not SOL strategy or short trading disabled"
        
        try:
            reasons = []
            
            # EMA trend (bearish for short)
            if ema_short < ema_long:
                reasons.append("EMA bearish")
            
            # RSI in preferred short zone (oversold to neutral)
            if self.sol_short_rsi_preferred_zone[0] <= rsi <= self.sol_short_rsi_preferred_zone[1]:
                reasons.append("RSI in short preferred zone")
            elif rsi > self.sol_rsi_overbought:
                reasons.append("RSI overbought - good for short")
            
            # MACD bearish
            if macd_bearish:
                reasons.append("MACD bearish")
            
            # Price not oversold
            if rsi > self.sol_rsi_oversold:
                reasons.append("Not oversold")
            
            if len(reasons) >= 3:
                return True, ", ".join(reasons)
            else:
                return False, f"Insufficient short signals: {', '.join(reasons) if reasons else 'none'}"
        except Exception as e:
            bot_logger.warning(f"SOL short detection failed: {e}")
            return False, "Detection error"
    
    def calculate_sol_short_setup_score(self, rsi, ema_short, ema_long, macd_bearish, price_near_upper, current_price, volume_confirmed, short_momentum_detected, near_support, liquidity_ok):
        """Calculate SOL-specific short setup score (100-point system)"""
        if not self.is_sol_strategy:
            return 100
        
        score = 0
        
        # EMA trend (20 points) - bearish for short
        if ema_short < ema_long:
            score += 20
        
        # MACD (20 points) - bearish for short
        if macd_bearish:
            score += 20
        
        # RSI (15 points) - prefer overbought or neutral for shorts
        if rsi > self.sol_rsi_overbought:
            score += 15  # Overbought is great for shorts
        elif self.sol_short_rsi_preferred_zone[0] <= rsi <= self.sol_short_rsi_preferred_zone[1]:
            score += 10  # Neutral zone
        elif rsi < self.sol_rsi_oversold:
            score += 0  # Too oversold
        else:
            score += 5
        
        # Volume (15 points)
        if volume_confirmed:
            score += 15
        else:
            score += 5
        
        # Short momentum setup (15 points)
        if short_momentum_detected:
            score += 15
        else:
            score += 0
        
        # Price extension (10 points) - prefer near resistance for shorts
        if price_near_upper:
            score += 10
        else:
            score += 5
        
        # Support avoidance (10 points) - avoid being too close to support
        if not near_support:
            score += 10
        else:
            score -= 15  # Penalty for being near support
        
        # Liquidity and spread (5 points)
        if liquidity_ok:
            score += 5
        else:
            score -= 10  # Heavy penalty for poor liquidity
        
        return min(score, 100)  # Cap at 100
    
    def detect_breakout(self, current_price, support, resistance):
        """Detect price breakout from support/resistance"""
        try:
            if support is None or resistance is None:
                return False, False
            
            # Bullish breakout (price breaks resistance)
            bullish_breakout = current_price > resistance * 1.005
            
            # Bearish breakdown (price breaks support)
            bearish_breakdown = current_price < support * 0.995
            
            return bullish_breakout, bearish_breakdown
        except Exception as e:
            bot_logger.warning(f"Breakout detection failed: {e}")
            return False, False
    
    def place_buy_order(self, current_price):
        """Place buy order"""
        position_size = self.calculate_position_size(current_price)
        trade_value = position_size * current_price
        
        # Check minimum order size (Coinbase minimum is typically $1-2)
        MIN_ORDER_SIZE = 2.0  # $2 minimum order size
        if trade_value < MIN_ORDER_SIZE:
            bot_logger.warning(f"Order size ${trade_value:.2f} below minimum ${MIN_ORDER_SIZE:.2f} - insufficient capital for real trading")
            bot_logger.warning(f"Current capital: ${self.current_capital:.2f} - deposit more funds or use paper trading")
            return 0
        
        order_successful = False
        order_id = None
        
        # Execute real buy order on exchange
        try:
            # Coinbase requires specific configuration for market orders
            # Set createMarketBuyOrderRequiresPrice to False and pass cost in amount
            self.exchange.options['createMarketBuyOrderRequiresPrice'] = False
            
            order = self.exchange.create_market_buy_order(
                self.symbol,
                trade_value  # pass cost directly as amount
            )
            order_id = order.get('id', 'N/A')
            bot_logger.info(f"[REAL BUY ORDER PLACED] Order ID: {order_id}")
            
            # Verify order was actually filled
            if order_id and order_id != 'N/A':
                time.sleep(2)  # Wait for order to settle
                try:
                    # Check order status
                    order_status = self.exchange.fetch_order(order_id, self.symbol)
                    if order_status.get('status') == 'closed':
                        # Check if we actually received the base currency (SAND, ETH, etc.)
                        balance = self.exchange.fetch_balance()
                        # Extract base currency from symbol (e.g., SAND from SAND-USDC)
                        base_currency = self.symbol.split('/')[0] if '/' in self.symbol else self.symbol.split('-')[0]
                        base_balance = balance.get(base_currency, {}).get('free', 0)
                        if base_balance > 0:
                            order_successful = True
                            bot_logger.info(f"Order verified - received {base_balance:.6f} {base_currency}")
                        else:
                            bot_logger.warning(f"Order closed but no {base_currency} received - balance: {base_balance:.6f}")
                    else:
                        bot_logger.warning(f"Order not closed yet - status: {order_status.get('status')}")
                except Exception as e:
                    bot_logger.error(f"Error verifying order: {e}")
                    # Assume order failed if verification fails
                    order_successful = False
        except Exception as e:
            bot_logger.error(f"Failed to place real buy order: {e}")
            # Try alternative method with price
            try:
                order = self.exchange.create_order(
                    self.symbol,
                    'market',
                    'buy',
                    position_size,  # actual position size
                    current_price  # current price
                )
                order_id = order.get('id', 'N/A')
                bot_logger.info(f"[REAL BUY ORDER PLACED (alt method)] Order ID: {order_id}")
                
                # Verify alternative order
                if order_id and order_id != 'N/A':
                    time.sleep(2)
                    try:
                        balance = self.exchange.fetch_balance()
                        # Extract base currency from symbol (e.g., SAND from SAND-USDC)
                        base_currency = self.symbol.split('/')[0] if '/' in self.symbol else self.symbol.split('-')[0]
                        base_balance = balance.get(base_currency, {}).get('free', 0)
                        if base_balance > 0:
                            order_successful = True
                            bot_logger.info(f"Alternative order verified - received {base_balance:.6f} {base_currency}")
                    except Exception as e2:
                        bot_logger.error(f"Error verifying alternative order: {e2}")
            except Exception as e2:
                bot_logger.error(f"Alternative method also failed: {e2}")
                # Fall back to paper trading if order fails
                bot_logger.warning("Falling back to paper trading for this order")
        
        # Only set position if order was actually successful
        if order_successful:
            self.long_position = True
            self.last_buy_price = current_price
            self.long_position_size = position_size
            self.highest_price_since_buy = current_price
            self.partial_tps_taken = []  # Reset partial TPs for new position
        else:
            bot_logger.warning("Buy order not verified - not setting position (paper trading)")
            # Still update position for paper trading
            self.long_position = True
            self.last_buy_price = current_price
            self.long_position_size = position_size
            self.highest_price_since_buy = current_price
            self.partial_tps_taken = []  # Reset partial TPs for new position
        
        bot_logger.info(f"[BUY #{self.trade_count + 1}] {self.currency_symbol}{current_price:.2f} | Size: {position_size:.6f} | Value: ${trade_value:.2f} | Volatility: {self.VOLATILITY_MULTIPLIER}x | Real: {order_successful}")
        
        return position_size
    
    def place_short_order(self, current_price):
        """Place short order (sell borrowed asset)"""
        if not self.enable_short_trading:
            bot_logger.warning("Short trading is disabled - cannot place short order")
            return 0
        
        position_size = self.calculate_position_size(current_price)
        trade_value = position_size * current_price
        
        # Check minimum order size
        MIN_ORDER_SIZE = 2.0  # $2 minimum order size
        if trade_value < MIN_ORDER_SIZE:
            bot_logger.warning(f"Order size ${trade_value:.2f} below minimum ${MIN_ORDER_SIZE:.2f}")
            return 0
        
        order_successful = False
        order_id = None
        
        # Execute real short order (sell borrowed asset)
        try:
            # For shorting, we sell the base currency
            self.exchange.options['createMarketBuyOrderRequiresPrice'] = False
            
            # Check if we have enough balance to short
            balance = self.exchange.fetch_balance()
            base_currency = self.symbol.split('/')[0] if '/' in self.symbol else self.symbol.split('-')[0]
            base_balance = balance.get(base_currency, {}).get('free', 0)
            
            if base_balance < position_size:
                bot_logger.warning(f"Insufficient {base_currency} balance for short. Have: {base_balance:.6f}, Need: {position_size:.6f}")
                # For paper trading, proceed anyway
                order_successful = False
            else:
                order = self.exchange.create_market_sell_order(
                    self.symbol,
                    position_size
                )
                order_id = order.get('id', 'N/A')
                bot_logger.info(f"[REAL SHORT ORDER PLACED] Order ID: {order_id}")
                
                # Verify order
                if order_id and order_id != 'N/A':
                    time.sleep(2)
                    try:
                        order_status = self.exchange.fetch_order(order_id, self.symbol)
                        if order_status.get('status') == 'closed':
                            order_successful = True
                            bot_logger.info(f"Short order verified - sold {position_size:.6f} {base_currency}")
                    except Exception as e:
                        bot_logger.error(f"Error verifying short order: {e}")
                        order_successful = False
        except Exception as e:
            bot_logger.error(f"Failed to place real short order: {e}")
            order_successful = False
        
        # Only set short position if order was successful
        if order_successful:
            self.short_position = True
            self.last_short_price = current_price
            self.short_position_size = position_size
            self.lowest_price_since_short = current_price
            self.partial_tps_taken = []
            
            # Initialize trailing stop and breakeven for short position
            self.short_trailing_stop_price = None
            self.short_breakeven_triggered = False
            
            bot_logger.info(f"[SHORT #{self.trade_count + 1}] {self.currency_symbol}{current_price:.2f} | Size: {position_size:.6f} | Value: ${trade_value:.2f} | Real: {order_successful}")
        else:
            bot_logger.warning(f"Short order failed - not setting position")
            return 0
        
        return position_size
    
    def place_sell_order(self, current_price, reason):
        """Place sell order (close long position)"""
        if not self.long_position or self.last_buy_price is None:
            return
        
        # Extract base currency from symbol (e.g., SAND from SAND-USDC)
        base_currency = self.symbol.split('/')[0] if '/' in self.symbol else self.symbol.split('-')[0]
        
        # Check actual base currency balance before attempting to sell
        try:
            balance = self.exchange.fetch_balance()
            base_balance = balance.get(base_currency, {}).get('free', 0)
            bot_logger.info(f"Actual {base_currency} balance: {base_balance:.6f}, Attempting to sell: {self.long_position_size:.6f}")
            
            if base_balance < self.long_position_size:
                bot_logger.warning(f"Insufficient {base_currency} balance. Have: {base_balance:.6f}, Need: {self.long_position_size:.6f}")
                # Adjust sell amount to actual available balance
                if base_balance > 0:
                    self.long_position_size = base_balance
                    bot_logger.info(f"Adjusted sell amount to available balance: {self.long_position_size:.6f}")
                else:
                    bot_logger.error(f"No {base_currency} available to sell, skipping real order")
                    # Fall back to paper trading
                    self._execute_paper_sell(current_price, reason)
                    return
        except Exception as e:
            bot_logger.error(f"Error checking {base_currency} balance: {e}")
        
        # Execute real sell order on exchange
        try:
            order = self.exchange.create_market_sell_order(self.symbol, self.long_position_size)
            bot_logger.info(f"[REAL SELL ORDER PLACED] Order ID: {order.get('id', 'N/A')}")
        except Exception as e:
            bot_logger.error(f"Failed to place real sell order: {e}")
            # Fall back to paper trading if order fails
            bot_logger.warning("Falling back to paper trading for this order")
            self._execute_paper_sell(current_price, reason)
            return
        
        # Calculate profit/loss
        profit_pct = ((current_price - self.last_buy_price) / self.last_buy_price) * 100
        profit_amount = (current_price - self.last_buy_price) * self.long_position_size
        
        # Update statistics
        self.trade_count += 1
        self.profit_loss += profit_amount
        self.current_capital += profit_amount
        
        if profit_amount > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            if profit_amount > self.best_trade_profit:
                self.best_trade_profit = profit_amount
            bot_logger.info(f"[WIN #{self.trade_count}] Profit: ${profit_amount:.2f} ({profit_pct:+.2f}%) | Consecutive Wins: {self.consecutive_wins} | Capital: ${self.current_capital:.2f} | Best Trade: ${self.best_trade_profit:.2f}")
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.last_loss_time = time.time()  # Record loss time for cooling off period
            if profit_amount < self.worst_trade_loss:
                self.worst_trade_loss = profit_amount
            bot_logger.info(f"[LOSS #{self.trade_count}] Loss: ${profit_amount:.2f} ({profit_pct:+.2f}%) | Consecutive Losses: {self.consecutive_losses} | Capital: ${self.current_capital:.2f} | Worst Trade: ${self.worst_trade_loss:.2f}")
        
        # Record trade
        trade_record = {
            'trade_number': self.trade_count,
            'type': 'long',
            'buy_price': self.last_buy_price,
            'sell_price': current_price,
            'position_size': self.long_position_size,
            'profit': profit_amount,
            'profit_pct': profit_pct,
            'reason': reason
        }
        self.trade_history.append(trade_record)
        
        bot_logger.info(f"[SELL #{self.trade_count}] {self.currency_symbol}{current_price:.2f} | Profit: ${profit_amount:.2f} ({profit_pct:+.2f}%) | Reason: {reason} | Capital: ${self.current_capital:.2f}")
        
        # Reset long position only (keep short position if open)
        self.long_position = None
        self.last_buy_price = None
        self.long_position_size = 0.0
        self.highest_price_since_buy = None
        
        # Reset trailing stop and breakeven for long position
        self.trailing_stop_price = None
        self.breakeven_triggered = False
        
        # Save state
        self.save_capital_state()
        
        return profit_amount
    
    def place_cover_order(self, current_price, reason):
        """Place cover order (close short position by buying back)"""
        if not self.short_position or self.last_short_price is None:
            return
        
        # Extract base currency from symbol
        base_currency = self.symbol.split('/')[0] if '/' in self.symbol else self.symbol.split('-')[0]
        
        # Check if we have enough USDC to cover
        try:
            balance = self.exchange.fetch_balance()
            usdc_balance = balance.get('USDC', {}).get('free', 0)
            trade_value = self.short_position_size * current_price
            
            bot_logger.info(f"USDC balance: ${usdc_balance:.2f}, Needed to cover: ${trade_value:.2f}")
            
            if usdc_balance < trade_value:
                bot_logger.warning(f"Insufficient USDC to cover short. Have: ${usdc_balance:.2f}, Need: ${trade_value:.2f}")
                # For paper trading, proceed anyway
                order_successful = False
            else:
                # Execute real cover order (buy back the asset)
                self.exchange.options['createMarketBuyOrderRequiresPrice'] = False
                order = self.exchange.create_market_buy_order(
                    self.symbol,
                    trade_value  # pass cost directly
                )
                order_id = order.get('id', 'N/A')
                bot_logger.info(f"[REAL COVER ORDER PLACED] Order ID: {order_id}")
                
                # Verify order
                if order_id and order_id != 'N/A':
                    time.sleep(2)
                    try:
                        order_status = self.exchange.fetch_order(order_id, self.symbol)
                        if order_status.get('status') == 'closed':
                            order_successful = True
                            bot_logger.info(f"Cover order verified - bought back {self.short_position_size:.6f} {base_currency}")
                    except Exception as e:
                        bot_logger.error(f"Error verifying cover order: {e}")
                        order_successful = False
        except Exception as e:
            bot_logger.error(f"Failed to place real cover order: {e}")
            order_successful = False
        
        # Calculate profit/loss for short (profit when price goes down)
        profit_pct = ((self.last_short_price - current_price) / self.last_short_price) * 100
        profit_amount = (self.last_short_price - current_price) * self.short_position_size
        
        # Update statistics
        self.trade_count += 1
        self.profit_loss += profit_amount
        self.current_capital += profit_amount
        
        if profit_amount > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            if profit_amount > self.best_trade_profit:
                self.best_trade_profit = profit_amount
            bot_logger.info(f"[SHORT WIN #{self.trade_count}] Profit: ${profit_amount:.2f} ({profit_pct:+.2f}%) | Consecutive Wins: {self.consecutive_wins} | Capital: ${self.current_capital:.2f}")
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.last_loss_time = time.time()
            if profit_amount < self.worst_trade_loss:
                self.worst_trade_loss = profit_amount
            bot_logger.info(f"[SHORT LOSS #{self.trade_count}] Loss: ${profit_amount:.2f} ({profit_pct:+.2f}%) | Consecutive Losses: {self.consecutive_losses} | Capital: ${self.current_capital:.2f}")
        
        # Record trade
        trade_record = {
            'trade_number': self.trade_count,
            'type': 'short',
            'entry_price': self.last_short_price,
            'exit_price': current_price,
            'position_size': self.short_position_size,
            'profit': profit_amount,
            'profit_pct': profit_pct,
            'reason': reason
        }
        self.trade_history.append(trade_record)
        
        bot_logger.info(f"[COVER #{self.trade_count}] {self.currency_symbol}{current_price:.2f} | Profit: ${profit_amount:.2f} ({profit_pct:+.2f}%) | Reason: {reason} | Capital: ${self.current_capital:.2f}")
        
        # Reset short position only (keep long position if open)
        self.short_position = None
        self.last_short_price = None
        self.short_position_size = 0.0
        self.lowest_price_since_short = None
        
        # Reset trailing stop and breakeven for short position
        self.short_trailing_stop_price = None
        self.short_breakeven_triggered = False
        
        # Save state
        self.save_capital_state()
        
        return profit_amount
    
    def _execute_paper_sell(self, current_price, reason):
        """Execute paper trading sell when real order fails"""
        # Calculate profit/loss
        profit_pct = ((current_price - self.last_buy_price) / self.last_buy_price) * 100
        profit_amount = (current_price - self.last_buy_price) * self.long_position_size
        
        # Update statistics
        self.trade_count += 1
        self.profit_loss += profit_amount
        self.current_capital += profit_amount
        
        if profit_amount > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            if profit_amount > self.best_trade_profit:
                self.best_trade_profit = profit_amount
            bot_logger.info(f"[WIN #{self.trade_count}] Profit: ${profit_amount:.2f} ({profit_pct:+.2f}%) | Consecutive Wins: {self.consecutive_wins} | Capital: ${self.current_capital:.2f} | Best Trade: ${self.best_trade_profit:.2f} | Paper Trading")
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.last_loss_time = time.time()  # Record loss time for cooling off period
            if profit_amount < self.worst_trade_loss:
                self.worst_trade_loss = profit_amount
            bot_logger.info(f"[LOSS #{self.trade_count}] Loss: ${profit_amount:.2f} ({profit_pct:+.2f}%) | Consecutive Losses: {self.consecutive_losses} | Capital: ${self.current_capital:.2f} | Worst Trade: ${self.worst_trade_loss:.2f} | Paper Trading")
        
        # Record trade
        trade_record = {
            'trade_number': self.trade_count,
            'type': 'long',
            'buy_price': self.last_buy_price,
            'sell_price': current_price,
            'position_size': self.long_position_size,
            'profit': profit_amount,
            'profit_pct': profit_pct,
            'reason': f"{reason} (Paper Trading)"
        }
        self.trade_history.append(trade_record)
        
        bot_logger.info(f"[SELL #{self.trade_count}] {self.currency_symbol}{current_price:.2f} | Profit: ${profit_amount:.2f} ({profit_pct:+.2f}%) | Reason: {reason} (Paper Trading) | Capital: ${self.current_capital:.2f}")
        
        # Reset long position only (keep short position if open)
        self.long_position = None
        self.last_buy_price = None
        self.long_position_size = 0.0
        self.highest_price_since_buy = None
        
        # Save state
        self.save_capital_state()
    
    def scan_for_best_pair(self):
        """Scan for the best trading pair using coin scanner"""
        if not self.coin_scanner:
            return None
            
        current_time = time.time()
        if current_time - self.last_scan_time < self.scan_interval * 60:
            return None  # Not time to scan yet
            
        try:
            best_pair = self.coin_scanner.get_best_pair()
            self.last_scan_time = current_time
            
            if best_pair:
                # Validate the selected pair
                if best_pair['edge'] < self.min_edge_score:
                    bot_logger.warning(f"❌ Coin scanner rejected: edge score {best_pair['edge']:.2f} below minimum {self.min_edge_score}")
                    return None
                
                if not best_pair.get('symbol') or '-' not in best_pair['symbol']:
                    bot_logger.warning(f"❌ Coin scanner rejected: invalid symbol format '{best_pair.get('symbol')}'")
                    return None
                
                # Convert to exchange format
                new_symbol = best_pair['symbol'].replace('-', '/')
                
                if new_symbol != self.symbol:
                    bot_logger.info(f"✓ Coin scanner validated: switching from {self.symbol} to {new_symbol} (edge: {best_pair['edge']:.2f}, volume: {best_pair.get('volume', 'N/A')})")
                    self.symbol = new_symbol
                    self.currency_symbol = best_pair['symbol'].split('-')[0]
                    
                    # Reset price history for new pair
                    self.price_history = []
                    bot_logger.info(f"✓ Price history reset for new pair {new_symbol}")
                else:
                    bot_logger.info(f"✓ Coin scanner validated: keeping current pair {self.symbol} (edge: {best_pair['edge']:.2f})")
                
                return best_pair
            else:
                bot_logger.warning("Coin scanner returned no valid pairs")
        except Exception as e:
            bot_logger.error(f"❌ Coin scanner error: {e}")
            
        return None
    
    def handle_trade_event(self, current_price):
        """Main trading logic with state machine for continuous trading"""
        # Format position status for logging
        position_status = []
        if self.long_position:
            position_status.append(f"LONG(${self.last_buy_price:.2f})")
        if self.short_position:
            position_status.append(f"SHORT(${self.last_short_price:.2f})")
        position_str = ", ".join(position_status) if position_status else "None"
        
        bot_logger.info(f"=== handle_trade_event called === Price: ${current_price:.2f}, Position: {position_str}, State: {self.trading_state.value}")
        
        # Scan for best pair if enabled
        if self.enable_coin_scanner and self.symbol == 'AUTO':
            self.scan_for_best_pair()
        
        # Get BTC market weather for overall market context
        btc_weather = self.get_btc_market_weather()
        
        # Get relative strength vs BTC
        relative_strength = self.get_relative_strength(current_price)
        
        # Add price to history
        self.price_history.append(current_price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)
        
        # Add volume to history for ETH strategy
        if self.is_eth_strategy or self.is_sol_strategy:
            # Fetch ticker with volume data
            try:
                ticker = self.exchange.fetch_ticker(self.symbol)
                volume = ticker.get('baseVolume', 1)  # Default to 1 if volume not available
                self.volume_history.append(volume)
                if len(self.volume_history) > 100:
                    self.volume_history.pop(0)
            except Exception as e:
                bot_logger.warning(f"Failed to fetch volume data: {e}")
                # Add placeholder volume
                self.volume_history.append(1)
                if len(self.volume_history) > 100:
                    self.volume_history.pop(0)
        
        bot_logger.info(f"#{len(self.price_history)} | Price: ${current_price:.2f} | Capital: ${self.current_capital:.2f} | Position: {position_str} | Trades: {self.trade_count} | State: {self.trading_state.value}")
        
        # Need at least some data for indicators
        if len(self.price_history) < 5:
            if not self.long_position and not self.short_position and self.trade_count == 0:
                # Force first trade
                bot_logger.info("Forcing first trade (insufficient data for indicators)")
                self.transition_state(TradingState.ENTRY_SIGNAL, "Force first trade")
                self.place_buy_order(current_price)
                if self.long_position:
                    self.transition_state(TradingState.MONITOR_POSITION, "Position opened")
            return
        
        # Calculate indicators using complex strategy
        rsi = self.calculate_rsi(self.price_history)
        ema_short = self.calculate_ema(self.price_history, self.ema_short)
        ema_long = self.calculate_ema(self.price_history, self.ema_long)
        sma = self.calculate_sma(self.price_history, self.sma_period)
        macd_line, signal_line, histogram = self.calculate_macd(self.price_history)
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(self.price_history)
        momentum = self.calculate_momentum(self.price_history, self.momentum_period)
        atr = self.calculate_atr(self.price_history)
        trend = self.detect_trend(self.price_history)
        market_regime = self.detect_market_regime(self.price_history)
        
        # Calculate RSI history for divergence detection
        rsi_history = [self.calculate_rsi(self.price_history[:i+1]) for i in range(len(self.price_history))]
        
        # Detect trading patterns
        rsi_divergence = self.detect_rsi_divergence(self.price_history, rsi_history)
        support, resistance = self.detect_support_resistance(self.price_history)
        bullish_breakout, bearish_breakdown = self.detect_breakout(current_price, support, resistance)
        engulfing_pattern = self.detect_engulfing_pattern(self.price_history)
        pin_bar = self.detect_pin_bar(self.price_history)
        
        if rsi is None or ema_short is None or ema_long is None:
            return
        
        # MACD signal
        macd_bullish = macd_line is not None and signal_line is not None and macd_line > signal_line
        macd_bearish = macd_line is not None and signal_line is not None and macd_line < signal_line
        
        # Bollinger Bands signal
        price_near_lower = bb_lower is not None and current_price <= bb_lower * 1.02
        price_near_upper = bb_upper is not None and current_price >= bb_upper * 0.98
        
        # Complex strategy signals
        ema_crossover_bullish = ema_short > ema_long and self.price_history[-2] and ema_short > ema_long
        price_above_sma = sma is not None and current_price > sma
        momentum_positive = momentum > 0
        atr_signal = atr > 0 and (current_price - self.last_buy_price) / self.last_buy_price < (atr / current_price) if self.last_buy_price else True
        
        bot_logger.info(f"RSI: {rsi:.1f} | Trend: {trend} | MACD: {'BULL' if macd_bullish else 'BEAR'} | BB: {'LOWER' if price_near_lower else 'UPPER' if price_near_upper else 'MID'} | EMA: {'BULL' if ema_short > ema_long else 'BEAR'} | SMA: {'ABOVE' if price_above_sma else 'BELOW'} | Mom: {'POS' if momentum_positive else 'NEG'} | ATR: {atr:.4f} | Regime: {market_regime} | Engulfing: {engulfing_pattern or 'NONE'} | PinBar: {pin_bar or 'NONE'} | Vol: {self.VOLATILITY_MULTIPLIER}x")
        
        # STATE MACHINE LOGIC
        if self.trading_state == TradingState.COOLDOWN:
            # Check if cooldown period is over
            if time.time() >= self.cooldown_end_time:
                self.transition_state(TradingState.IDLE_SCANNING, "Cooldown period ended")
            else:
                cooldown_remaining = int(self.cooldown_end_time - time.time())
                bot_logger.info(f"In cooldown - {cooldown_remaining}s remaining")
                return
        
        if self.trading_state == TradingState.IDLE_SCANNING:
            # Check if we should enter cooldown
            if self.use_dont_trade_engine and self.consecutive_losses >= self.max_consecutive_losses:
                self.cooldown_end_time = time.time() + (self.cooling_off_period * 60)
                self.transition_state(TradingState.COOLDOWN, f"Max consecutive losses ({self.consecutive_losses}) reached")
                return
            
            # Transition to ENTRY_SIGNAL if no position
            if not self.long_position and not self.short_position:
                self.transition_state(TradingState.ENTRY_SIGNAL, "No position - scanning for entry")
        
        if self.trading_state == TradingState.ENTRY_SIGNAL:
            # Check for entry conditions - allow entry if the specific position type is not open
            # Allow long entry if no long position (even if short is open)
            # Allow short entry if no short position (even if long is open)
            
            # Prevent stale signals - require minimum time between entry signals
            current_time = time.time()
            if self.last_entry_signal_time > 0 and (current_time - self.last_entry_signal_time) < 30:
                bot_logger.info(f"Stale signal prevention - last entry was {current_time - self.last_entry_signal_time:.1f}s ago (min 30s)")
                return
            
            # Evaluate buy signals (existing logic for longs)
            should_buy = self.evaluate_buy_signals(rsi, trend, ema_short, ema_long, price_above_sma, macd_bullish, 
                                                   momentum_positive, price_near_lower, rsi_divergence, 
                                                   bullish_breakout, engulfing_pattern, pin_bar, market_regime,
                                                   btc_weather, relative_strength, current_price)
            
            # Evaluate short signals (if enabled)
            should_short = False
            if self.enable_short_trading:
                # Reuse evaluate_buy_signals but with inverted logic for shorts
                # For now, we'll use the should_short flag set in evaluate_buy_signals
                pass
            
            if should_buy and not self.long_position:
                self.last_entry_signal_time = time.time()
                self.transition_state(TradingState.OPEN_POSITION, "Entry signal confirmed")
                self.place_buy_order(current_price)
                if self.long_position:
                    self.transition_state(TradingState.MONITOR_POSITION, "Long position opened successfully")
                else:
                    self.transition_state(TradingState.IDLE_SCANNING, "Position open failed - return to scanning")
            elif should_short and not self.short_position:
                self.last_entry_signal_time = time.time()
                self.transition_state(TradingState.OPEN_POSITION, "Short entry signal confirmed")
                self.place_short_order(current_price)
                if self.short_position:
                    self.transition_state(TradingState.MONITOR_POSITION, "Short position opened successfully")
                else:
                    self.transition_state(TradingState.IDLE_SCANNING, "Short position open failed - return to scanning")
            else:
                bot_logger.info("No valid entry signal - continue scanning")
        
        elif self.trading_state == TradingState.MONITOR_POSITION:
            if not self.long_position and not self.short_position:
                # Both positions were closed externally, transition to reset
                self.transition_state(TradingState.POSITION_CLOSED, "All positions closed externally")
                return
            
            # Update highest price for trailing stop (long positions)
            if self.long_position:
                if self.highest_price_since_buy is None or current_price > self.highest_price_since_buy:
                    self.highest_price_since_buy = current_price
                
                # Update trailing stop for long position
                if self.use_trailing_stop and self.highest_price_since_buy:
                    long_profit_pct = ((current_price - self.last_buy_price) / self.last_buy_price) * 100
                    if long_profit_pct >= self.trailing_stop_activation_pct:
                        # Activate trailing stop
                        new_trailing_stop = self.highest_price_since_buy * (1 - self.trailing_stop_distance_pct / 100)
                        if self.trailing_stop_price is None or new_trailing_stop > self.trailing_stop_price:
                            self.trailing_stop_price = new_trailing_stop
                            bot_logger.info(f"Trailing stop updated to ${self.trailing_stop_price:.4f} (highest: ${self.highest_price_since_buy:.4f})")
                
                # Update breakeven for long position
                if self.use_breakeven and not self.breakeven_triggered:
                    long_profit_pct = ((current_price - self.last_buy_price) / self.last_buy_price) * 100
                    if long_profit_pct >= self.breakeven_activation_pct:
                        self.breakeven_triggered = True
                        breakeven_price = self.last_buy_price * (1 + self.breakeven_offset_pct / 100)
                        bot_logger.info(f"Breakeven triggered at ${breakeven_price:.4f}")
            
            # Update lowest price for trailing stop (short positions)
            if self.short_position:
                if self.lowest_price_since_short is None or current_price < self.lowest_price_since_short:
                    self.lowest_price_since_short = current_price
                
                # Update trailing stop for short position
                if self.use_trailing_stop and self.lowest_price_since_short:
                    short_profit_pct = ((self.last_short_price - current_price) / self.last_short_price) * 100
                    if short_profit_pct >= self.trailing_stop_activation_pct:
                        # Activate trailing stop (trails upward for shorts)
                        new_trailing_stop = self.lowest_price_since_short * (1 + self.trailing_stop_distance_pct / 100)
                        if self.short_trailing_stop_price is None or new_trailing_stop < self.short_trailing_stop_price:
                            self.short_trailing_stop_price = new_trailing_stop
                            bot_logger.info(f"Short trailing stop updated to ${self.short_trailing_stop_price:.4f} (lowest: ${self.lowest_price_since_short:.4f})")
                
                # Update breakeven for short position
                if self.use_breakeven and not self.short_breakeven_triggered:
                    short_profit_pct = ((self.last_short_price - current_price) / self.last_short_price) * 100
                    if short_profit_pct >= self.breakeven_activation_pct:
                        self.short_breakeven_triggered = True
                        breakeven_price = self.last_short_price * (1 - self.breakeven_offset_pct / 100)
                        bot_logger.info(f"Short breakeven triggered at ${breakeven_price:.4f}")
            
            # Calculate profit percentage for both positions
            long_profit_pct = 0
            short_profit_pct = 0
            
            if self.long_position:
                long_profit_pct = ((current_price - self.last_buy_price) / self.last_buy_price) * 100
            
            if self.short_position:
                short_profit_pct = ((self.last_short_price - current_price) / self.last_short_price) * 100
            
            # Use the position with the larger profit/loss for decision making
            profit_pct = long_profit_pct if abs(long_profit_pct) > abs(short_profit_pct) else short_profit_pct
            
            # Debug logging for sell logic
            bot_logger.info(f"Position Check: Profit%={profit_pct:.2f}%, TP={self.take_profit_pct}%, SL={self.stop_loss_pct}%, RSI={rsi:.1f}, MACD={'BULL' if macd_bullish else 'BEAR'}, BB={'LOWER' if price_near_lower else 'UPPER' if price_near_upper else 'MID'}")
            
            # Handle partial profit taking before main sell logic
            partial_tp_taken = self.handle_partial_profit_taking(current_price)
            if partial_tp_taken:
                return  # Exit after partial TP, position may be closed
            
            # Evaluate sell signals for each position separately
            should_sell_long = False
            should_sell_short = False
            sell_reason_long = ""
            sell_reason_short = ""
            
            if self.long_position:
                should_sell_long, sell_reason_long = self.evaluate_sell_signals(long_profit_pct, current_price, rsi, macd_bearish, 
                                                             ema_short, market_regime, dynamic_tp=None, dynamic_sl=None)
            
            if self.short_position:
                should_sell_short, sell_reason_short = self.evaluate_sell_signals(short_profit_pct, current_price, rsi, macd_bearish, 
                                                             ema_short, market_regime, dynamic_tp=None, dynamic_sl=None)
            
            # Close positions based on their individual signals
            if should_sell_long:
                self.transition_state(TradingState.POSITION_CLOSED, f"Long exit signal: {sell_reason_long}")
                self.place_sell_order(current_price, sell_reason_long)
            
            if should_sell_short:
                self.transition_state(TradingState.POSITION_CLOSED, f"Short exit signal: {sell_reason_short}")
                self.place_cover_order(current_price, sell_reason_short)
            
            # If both positions closed, transition to reset
            if not self.long_position and not self.short_position:
                self.transition_state(TradingState.RESET_STATE, "All positions closed successfully")
        
        elif self.trading_state == TradingState.POSITION_CLOSED:
            # Ensure at least one position is still open or transition to reset
            if not self.long_position and not self.short_position:
                self.transition_state(TradingState.RESET_STATE, "All positions confirmed closed")
            else:
                # One position still open, continue monitoring
                self.transition_state(TradingState.MONITOR_POSITION, "One position still open, continue monitoring")
        
        elif self.trading_state == TradingState.RESET_STATE:
            # Reset all trade-specific variables (only reset what's not already reset)
            self.long_position = None
            self.short_position = None
            self.last_buy_price = None
            self.last_short_price = None
            self.long_position_size = 0.0
            self.short_position_size = 0.0
            self.highest_price_since_buy = None
            self.lowest_price_since_short = None
            self.partial_tps_taken = []
            
            bot_logger.info(f"Trade reset complete | Capital: ${self.current_capital:.2f} | Trades: {self.trade_count}")
            
            # Save state
            self.save_capital_state()
            
            # Return to scanning for next trade
            self.transition_state(TradingState.IDLE_SCANNING, "Ready for next trade")
    
    def evaluate_buy_signals(self, rsi, trend, ema_short, ema_long, price_above_sma, macd_bullish, 
                           momentum_positive, price_near_lower, rsi_divergence, bullish_breakout, 
                           engulfing_pattern, pin_bar, market_regime, btc_weather, relative_strength, current_price):
        """Evaluate buy signals - extracted from main logic for state machine"""
        should_buy = False
        bullish_signals = 0
        total_signals = 0
        
        # RSI signals
        total_signals += 1
        if rsi < self.rsi_oversold:
            bullish_signals += 1
        elif rsi < 40:
            bullish_signals += 0.5
        
        # Trend signals
        total_signals += 1
        if trend == "BULLISH":
            bullish_signals += 1
        
        # EMA crossover
        total_signals += 1
        if ema_short > ema_long:
            bullish_signals += 1
        
        # SMA position
        total_signals += 1
        if price_above_sma:
            bullish_signals += 1
        
        # MACD
        total_signals += 1
        if macd_bullish:
            bullish_signals += 1
        
        # Momentum
        total_signals += 1
        if momentum_positive:
            bullish_signals += 1
        
        # Bollinger Bands
        total_signals += 1
        if price_near_lower:
            bullish_signals += 1
        
        # RSI divergence
        total_signals += 1
        if rsi_divergence:
            bullish_signals += 1
        
        # Bullish breakout
        total_signals += 1
        if bullish_breakout:
            bullish_signals += 1
        
        # Engulfing pattern - highly reliable for gold
        total_signals += 2  # Give it more weight
        if engulfing_pattern == "BULLISH_ENGULFING":
            bullish_signals += 2  # Strong signal
        elif engulfing_pattern == "BEARISH_ENGULFING":
            bullish_signals -= 1  # Negative signal
        
        # Pin bar - clean signals for gold
        total_signals += 1
        if pin_bar == "BULLISH_PIN_BAR":
            bullish_signals += 1
        elif pin_bar == "BEARISH_PIN_BAR":
            bullish_signals -= 0.5
        
        # Market regime bonus/penalty
        total_signals += 1
        if "TRENDING_UP" in market_regime:
            bullish_signals += 1
        elif "TRENDING_DOWN" in market_regime:
            bullish_signals -= 0.5
        
        # Volatility adjustment
        if "HIGH_VOL" in market_regime:
            # Be more cautious in high volatility
            total_signals += 0.5
        elif "LOW_VOL" in market_regime:
            # Can be more aggressive in low volatility
            total_signals -= 0.5
        
        # Calculate bullish percentage
        bullish_pct = (bullish_signals / total_signals) * 100 if total_signals > 0 else 0
        
        # Adaptive threshold based on recent performance
        if len(self.recent_trades) >= 5:
            recent_wins = sum(1 for t in self.recent_trades[-5:] if t['profit'] > 0)
            if recent_wins >= 4:  # Hot streak - be more aggressive
                self.min_confidence_threshold = 0.10
            elif recent_wins <= 1:  # Cold streak - be more conservative
                self.min_confidence_threshold = 0.25
            else:
                self.min_confidence_threshold = 0.15
        else:
            # Default threshold for consistent buying - extremely permissive
            self.min_confidence_threshold = 0.10
        
        # ML-Primary Trading Decision
        if self.ml_enabled and self.use_ml_signals and len(self.price_history) >= 5:
            try:
                # Create DataFrame for ML prediction
                ml_data = pd.DataFrame({
                    'close': self.price_history,
                    'volume': [1] * len(self.price_history),  # Placeholder volume
                    'open': self.price_history,  # Use close as placeholder
                    'high': self.price_history,  # Use close as placeholder
                    'low': self.price_history   # Use close as placeholder
                })
                
                ml_signals = None
                try:
                    ml_signals = self.ml_ensemble.get_trading_signal(ml_data)
                    bot_logger.info(f"ML signals generated successfully with {len(self.price_history)} price points")
                except Exception as e:
                    bot_logger.warning(f"ML signal generation failed with {len(self.price_history)} price points: {e} - falling back to traditional indicators")
                    ml_signals = None
                
                if ml_signals:
                    bot_logger.info(f"ML Signals: {ml_signals}")
                    
                    # ML-Primary Decision Logic
                    ml_buy_score = 0
                    ml_sell_score = 0
                    
                    # LSTM prediction (price direction)
                    if 'lstm' in ml_signals:
                        lstm_pred = ml_signals['lstm']
                        price_change_pct = ((lstm_pred - current_price) / current_price) * 100
                        
                        if price_change_pct > 0.5:  # Strong bullish prediction
                            ml_buy_score += 3
                        elif price_change_pct > 0.2:  # Moderate bullish
                            ml_buy_score += 2
                        elif price_change_pct < -0.5:  # Strong bearish
                            ml_sell_score += 3
                        elif price_change_pct < -0.2:  # Moderate bearish
                            ml_sell_score += 2
                    
                    # Random Forest signal
                    if 'random_forest' in ml_signals:
                        rf_signal = ml_signals['random_forest']
                        if rf_signal['signal'] == 1:
                            if rf_signal['confidence'] > 0.7:
                                ml_buy_score += 4  # Strong buy signal
                            elif rf_signal['confidence'] > 0.5:
                                ml_buy_score += 2  # Moderate buy signal
                        elif rf_signal['signal'] == 0:
                            if rf_signal['confidence'] > 0.7:
                                ml_sell_score += 4  # Strong sell signal
                            elif rf_signal['confidence'] > 0.5:
                                ml_sell_score += 2  # Moderate sell signal
                    
                    # ML-Primary Decision with OpenAI Integration
                    if self.openai_enabled:
                        # Get OpenAI market analysis
                        price_df = pd.DataFrame({'close': self.price_history})
                        openai_signal = self.openai_analyzer.analyze_market_conditions(price_df, current_price)
                        
                        # Cross-reference OpenAI and ML signals
                        combined_signal = self.openai_analyzer.cross_reference_signals(
                            openai_signal, ml_buy_score, ml_sell_score
                        )
                        
                        bot_logger.info(f"Combined Signal: {combined_signal['recommendation']} (confidence: {combined_signal['confidence']})")
                        bot_logger.info(f"Reasoning: {combined_signal['reasoning']}")
                        
                        # Track last AI signal for display
                        self.last_ai_signal = combined_signal['recommendation']
                        self.last_ai_confidence = combined_signal['confidence']
                        
                        # Use combined signal for decision
                        if 'STRONG BUY' in combined_signal['recommendation']:
                            should_buy = True
                            bullish_pct = 100
                            bot_logger.info("COMBINED STRONG BUY signal")
                        elif 'BUY' in combined_signal['recommendation']:
                            should_buy = True
                            bullish_pct = max(bullish_pct, 80)
                            bot_logger.info("COMBINED BUY signal")
                        elif 'STRONG SELL' in combined_signal['recommendation']:
                            should_buy = False
                            bullish_pct = 0
                            bot_logger.info("COMBINED STRONG SELL signal")
                        elif 'SELL' in combined_signal['recommendation']:
                            should_buy = False
                            bullish_pct = min(bullish_pct, 20)
                            bot_logger.info("COMBINED SELL signal")
                        else:
                            # Fallback to ML-only logic
                            if self.ml_only:
                                if ml_buy_score >= 3:
                                    should_buy = True
                                    bullish_pct = 100
                                    bot_logger.info(f"ML-ONLY BUY signal - score: {ml_buy_score}")
                                elif ml_sell_score >= 3:
                                    should_buy = False
                                    bullish_pct = 0
                                    bot_logger.info(f"ML-ONLY SELL signal - score: {ml_sell_score}")
                                else:
                                    should_buy = False
                                    bot_logger.info(f"ML-ONLY uncertain - no trade")
                            else:
                                if ml_buy_score >= 5:
                                    should_buy = True
                                    bullish_pct = 100
                                    bot_logger.info(f"ML STRONG BUY signal - score: {ml_buy_score}")
                                elif ml_buy_score >= 2:
                                    should_buy = True
                                    bullish_pct = max(bullish_pct, 70)
                                    bot_logger.info(f"ML BUY signal - score: {ml_buy_score}")
                                elif ml_sell_score >= 3:
                                    should_buy = False
                                    bullish_pct = 0
                                    bot_logger.info(f"ML SELL signal - score: {ml_sell_score}")
                                else:
                                    adaptive_threshold = self.get_adaptive_confidence_threshold(market_regime, bullish_pct / 100)
                                    should_buy = bullish_pct >= (adaptive_threshold * 100)
                                    bot_logger.info(f"ML uncertain - using traditional indicators")
                    elif self.ml_only:
                        # ML-Only Mode: Ignore traditional indicators completely
                        if ml_buy_score >= 2:  # Lowered from 3 for more frequent trades
                            should_buy = True
                            bullish_pct = 100  # ML-only mode
                            bot_logger.info(f"ML-ONLY BUY signal - score: {ml_buy_score}")
                        elif ml_sell_score >= 2:  # Lowered from 3 for more frequent trades
                            should_buy = False
                            bullish_pct = 0  # ML-only mode
                            bot_logger.info(f"ML-ONLY SELL signal - score: {ml_sell_score}")
                        else:
                            should_buy = False  # ML uncertain - don't trade
                            bot_logger.info(f"ML-ONLY uncertain - no trade")
                    else:
                        # ML-Enhanced Mode: Use ML as primary, traditional as fallback
                        if ml_buy_score >= 5:
                            should_buy = True
                            bullish_pct = 100  # Override traditional indicators
                            bot_logger.info(f"ML STRONG BUY signal - score: {ml_buy_score}")
                        elif ml_buy_score >= 2:
                            should_buy = True
                            bullish_pct = max(bullish_pct, 70)  # Boost confidence
                            bot_logger.info(f"ML BUY signal - score: {ml_buy_score}")
                        elif ml_sell_score >= 3:
                            should_buy = False
                            bullish_pct = 0  # Override traditional indicators
                            bot_logger.info(f"ML SELL signal - score: {ml_sell_score}")
                        else:
                            adaptive_threshold = self.get_adaptive_confidence_threshold(market_regime, bullish_pct / 100)
                            should_buy = bullish_pct >= (adaptive_threshold * 100)
                            bot_logger.info(f"ML uncertain - using traditional indicators")
                    
                    bot_logger.info(f"Final bullish percentage: {bullish_pct:.1f}%")
                    
            except Exception as e:
                bot_logger.warning(f"ML signal integration failed: {e}")
                if self.ml_only:
                    should_buy = False  # ML-only mode: don't trade if ML fails
                else:
                    # Fallback to traditional indicators
                    adaptive_threshold = self.get_adaptive_confidence_threshold(market_regime, bullish_pct / 100)
                    should_buy = bullish_pct >= (adaptive_threshold * 100)
        else:
            # IMPROVED BUY TIMING with enhanced traditional indicators
            # Momentum confirmation for better entry timing
            momentum_surge = momentum > 0 and momentum > self.price_history[-2] - self.price_history[-3] if len(self.price_history) >= 3 else False
            
            # Volume spike confirmation (if we had real volume data)
            volume_confirmation = True  # Placeholder since we don't have real volume
            
            # Price action confirmation - recent upward movement
            price_action_bullish = current_price > self.price_history[-5] if len(self.price_history) >= 5 else True
            
            # Optimal entry conditions
            optimal_entry = (
                rsi < 40 and  # Not overbought
                ema_short > ema_long and  # Uptrend
                macd_bullish and  # MACD bullish
                momentum_positive and  # Positive momentum
                price_near_lower  # Near support
            )
            
            # Calculate adaptive threshold BEFORE using it for decision
            adaptive_threshold = self.get_adaptive_confidence_threshold(market_regime, bullish_pct / 100)
            
            # Enhanced buy conditions with timing confirmation
            if optimal_entry and momentum_surge and volume_confirmation:
                should_buy = True
                bullish_pct = 100
                bot_logger.info("OPTIMAL ENTRY: All timing conditions met")
            elif bullish_pct >= (adaptive_threshold * 100):
                # Standard buy with timing confirmation
                if momentum_surge and price_action_bullish:
                    should_buy = True
                    bot_logger.info("BUY with momentum confirmation")
                else:
                    should_buy = bullish_pct >= (adaptive_threshold * 100)
            else:
                should_buy = bullish_pct >= (adaptive_threshold * 100)
        
        # Force buy if no trades yet and we have some bullish signals
        if self.trade_count == 0 and bullish_pct > 0:
            should_buy = True
            bot_logger.info("Force first trade - no previous trades")
        
        # ETH-SPECIFIC STRATEGY
        if self.is_eth_strategy:
            bot_logger.info("🔷 ETH-SPECIFIC STRATEGY ACTIVE")
            
            # Detect ETH patterns
            pullback_detected, pullback_reason = self.detect_eth_pullback(current_price, ema_short, ema_long, rsi)
            breakout_retest_detected, breakout_reason = self.detect_eth_breakout_retest(current_price, self.price_history)
            near_resistance, resistance_reason = self.check_eth_resistance_avoidance(current_price)
            
            # Get current volume for confirmation
            current_volume = self.volume_history[-1] if self.volume_history else None
            volume_confirmed, volume_reason = self.check_volume_confirmation(current_volume)
            
            # Log ETH pattern detection
            if pullback_detected:
                bot_logger.info(f"✅ ETH PULLBACK DETECTED: {pullback_reason}")
            if breakout_retest_detected:
                bot_logger.info(f"✅ ETH BREAKOUT RETEST DETECTED: {breakout_reason}")
            if near_resistance:
                bot_logger.warning(f"⚠️ ETH RESISTANCE AVOIDANCE: {resistance_reason}")
            bot_logger.info(f"📊 ETH Volume: {volume_reason}")
            
            # Apply ETH resistance avoidance (hard block)
            if near_resistance:
                should_buy = False
                bot_logger.warning(f"❌ ETH TRADE BLOCKED: {resistance_reason}")
            
            # Calculate ETH-specific setup score
            eth_setup_score = self.calculate_eth_setup_score(
                rsi, ema_short, ema_long, macd_bullish, price_near_lower,
                current_price, volume_confirmed, pullback_detected,
                breakout_retest_detected, near_resistance
            )
            
            bot_logger.info(f"🔷 ETH SETUP SCORE: {eth_setup_score}/100 (min: {self.eth_min_setup_score})")
            
            # Apply ETH-specific scoring
            if eth_setup_score >= 90:
                bot_logger.info("✅ ETH A+ SETUP (90-100) - Trade approved")
                should_buy = True
            elif eth_setup_score >= 85:
                bot_logger.info("✅ ETH STRONG SETUP (85-89) - Trade approved")
                should_buy = True
            elif eth_setup_score >= 80:
                bot_logger.info("⚠️ ETH BORDERLINE SETUP (80-84) - Small position only")
                # Could reduce position size here
                should_buy = should_buy  # Keep existing decision
            else:
                bot_logger.warning(f"❌ ETH SETUP TOO LOW ({eth_setup_score} < 80) - Trade rejected")
                should_buy = False
        
        # SOL-SPECIFIC STRATEGY (Momentum Scalping)
        if self.is_sol_strategy:
            bot_logger.info("🟣 SOL-SPECIFIC STRATEGY ACTIVE (Momentum Scalping)")
            
            # Check liquidity and spread
            liquidity_ok, liquidity_reason = self.check_sol_liquidity_and_spread(current_price)
            bot_logger.info(f"💧 SOL Liquidity/Spread: {liquidity_reason}")
            
            # Detect SOL momentum setup (for longs)
            momentum_detected, momentum_reason = self.detect_sol_momentum_setup(current_price, ema_short, ema_long, rsi, macd_bullish)
            if momentum_detected:
                bot_logger.info(f"✅ SOL MOMENTUM DETECTED: {momentum_reason}")
            
            # Check resistance avoidance
            near_resistance, resistance_reason = self.check_eth_resistance_avoidance(current_price)  # Reuse ETH method
            if near_resistance:
                bot_logger.warning(f"⚠️ SOL RESISTANCE AVOIDANCE: {resistance_reason}")
            
            # Get current volume for confirmation
            current_volume = self.volume_history[-1] if self.volume_history else None
            volume_confirmed, volume_reason = self.check_volume_confirmation(current_volume)
            bot_logger.info(f"📊 SOL Volume: {volume_reason}")
            
            # Apply liquidity check (hard block)
            if not liquidity_ok:
                should_buy = False
                bot_logger.warning(f"❌ SOL TRADE BLOCKED: {liquidity_reason}")
            
            # Calculate SOL-specific setup score (for longs)
            sol_setup_score = self.calculate_sol_setup_score(
                rsi, ema_short, ema_long, macd_bullish, price_near_lower,
                current_price, volume_confirmed, momentum_detected, near_resistance, liquidity_ok
            )
            
            bot_logger.info(f"🟣 SOL LONG SETUP SCORE: {sol_setup_score}/100 (min: {self.sol_min_setup_score})")
            
            # Apply SOL-specific scoring (for longs)
            if sol_setup_score >= 90:
                bot_logger.info("✅ SOL A+ LONG SETUP (90-100) - Trade approved")
                should_buy = True
            elif sol_setup_score >= 85:
                bot_logger.info("✅ SOL STRONG LONG SETUP (85-89) - Trade approved")
                should_buy = True
            elif sol_setup_score >= 80:
                bot_logger.info("⚠️ SOL BORDERLINE LONG SETUP (80-84) - Small position only")
                should_buy = should_buy  # Keep existing decision
            elif sol_setup_score >= self.sol_min_setup_score:
                bot_logger.info(f"✅ SOL LONG SETUP MEETS THRESHOLD ({sol_setup_score} >= {self.sol_min_setup_score}) - Trade approved")
                should_buy = True
            else:
                bot_logger.warning(f"❌ SOL LONG SETUP TOO LOW ({sol_setup_score} < {self.sol_min_setup_score}) - Trade rejected")
                should_buy = False
        
        # SOL SHORT STRATEGY (if enabled)
        if self.is_sol_strategy and self.enable_short_trading:
            bot_logger.info("🟣 SOL SHORT STRATEGY ACTIVE")
            
            # Detect SOL short setup
            short_momentum_detected, short_momentum_reason = self.detect_sol_short_setup(current_price, ema_short, ema_long, rsi, not macd_bullish)
            if short_momentum_detected:
                bot_logger.info(f"✅ SOL SHORT MOMENTUM DETECTED: {short_momentum_reason}")
            
            # Check support avoidance (avoid being too close to support when shorting)
            near_support, support_reason = self.check_eth_resistance_avoidance(current_price)  # Reuse method for support
            if near_support:
                bot_logger.warning(f"⚠️ SOL SUPPORT AVOIDANCE: {support_reason}")
            
            # Calculate SOL short setup score
            sol_short_setup_score = self.calculate_sol_short_setup_score(
                rsi, ema_short, ema_long, not macd_bullish, price_near_upper,
                current_price, volume_confirmed, short_momentum_detected, near_support, liquidity_ok
            )
            
            bot_logger.info(f"🟣 SOL SHORT SETUP SCORE: {sol_short_setup_score}/100 (min: {self.sol_short_min_setup_score})")
            
            # Apply SOL short scoring
            if sol_short_setup_score >= 90:
                bot_logger.info("✅ SOL A+ SHORT SETUP (90-100) - Short approved")
                should_short = True
            elif sol_short_setup_score >= 85:
                bot_logger.info("✅ SOL STRONG SHORT SETUP (85-89) - Short approved")
                should_short = True
            elif sol_short_setup_score >= 80:
                bot_logger.info("⚠️ SOL BORDERLINE SHORT SETUP (80-84) - Small position only")
                should_short = True
            elif sol_short_setup_score >= self.sol_short_min_setup_score:
                bot_logger.info(f"✅ SOL SHORT SETUP MEETS THRESHOLD ({sol_short_setup_score} >= {self.sol_short_min_setup_score}) - Short approved")
                should_short = True
            else:
                bot_logger.warning(f"❌ SOL SHORT SETUP TOO LOW ({sol_short_setup_score} < {self.sol_short_min_setup_score}) - Short rejected")
                should_short = False
        else:
            should_short = False
        
        # Apply BTC market weather filter
        if self.use_btc_weather and btc_weather:
            if btc_weather['signal'] == 'STRONG_BEARISH':
                # Don't trade if BTC is strongly bearish
                should_buy = False
                bot_logger.warning("BTC strongly bearish - trade blocked")
            elif btc_weather['signal'] == 'BEARISH':
                # Reduce position size or confidence if BTC is bearish
                bullish_pct *= (1 - self.btc_weather_weight)
                bot_logger.info(f"BTC bearish - reduced bullish confidence to {bullish_pct:.1f}%")
            elif btc_weather['signal'] == 'STRONG_BULLISH':
                # Boost confidence if BTC is strongly bullish
                bullish_pct *= (1 + self.btc_weather_weight)
                bot_logger.info(f"BTC strongly bullish - boosted confidence to {bullish_pct:.1f}%")
        
        # Apply relative strength filter
        if self.use_relative_strength and relative_strength:
            if relative_strength['signal'] == 'STRONG_OUTPERFORMING':
                # Boost confidence if strongly outperforming BTC
                bullish_pct *= (1 + self.relative_strength_weight)
                bot_logger.info(f"Strongly outperforming BTC - boosted confidence to {bullish_pct:.1f}%")
            elif relative_strength['signal'] == 'OUTPERFORMING':
                # Slightly boost confidence if outperforming BTC
                bullish_pct *= (1 + self.relative_strength_weight * 0.5)
                bot_logger.info(f"Outperforming BTC - boosted confidence to {bullish_pct:.1f}%")
            elif relative_strength['signal'] == 'STRONG_UNDERPERFORMING':
                # Reduce confidence if strongly underperforming BTC
                bullish_pct *= (1 - self.relative_strength_weight)
                bot_logger.info(f"Strongly underperforming BTC - reduced confidence to {bullish_pct:.1f}%")
            elif relative_strength['signal'] == 'UNDERPERFORMING':
                # Slightly reduce confidence if underperforming BTC
                bullish_pct *= (1 - self.relative_strength_weight * 0.5)
                bot_logger.info(f"Underperforming BTC - reduced confidence to {bullish_pct:.1f}%")
        
        # Calculate 100-point setup score
        setup_score = self.calculate_setup_score(
            rsi, trend, ema_short, ema_long, macd_bullish, price_near_lower,
            btc_weather, relative_strength, 0, market_regime  # ml_buy_score not available here
        )
        
        # Apply setup score filter
        if self.use_setup_score and setup_score < self.min_setup_score:
            should_buy = False
            bot_logger.warning(f"❌ SETUP SCORE BLOCKED: {setup_score} < {self.min_setup_score} - trade rejected")
        else:
            bot_logger.info(f"✓ Setup score passed: {setup_score}/100 (min: {self.min_setup_score})")
        
        # Apply don't trade engine filters
        should_block, block_reason = self.should_block_trade(current_price)
        if should_block:
            should_buy = False
            bot_logger.warning(f"❌ DON'T TRADE ENGINE BLOCKED: {block_reason} - trade rejected")
        else:
            bot_logger.info(f"✓ Don't trade engine passed")
        
        # Calculate adaptive threshold for logging (already used for decision above)
        if 'adaptive_threshold' not in locals():
            adaptive_threshold = self.get_adaptive_confidence_threshold(market_regime, bullish_pct / 100)
        
        # Final decision logging with detailed reasoning
        if should_buy:
            bot_logger.info(f"✅ BUY ACCEPTED: Bullish={bullish_pct:.1f}% >= Threshold={adaptive_threshold*100:.1f}% | Setup Score={setup_score}/100")
        else:
            bot_logger.info(f"❌ BUY REJECTED: Bullish={bullish_pct:.1f}% < Threshold={adaptive_threshold*100:.1f}% | Setup Score={setup_score}/100")
        
        if should_buy:
            # Calculate trade amount for profitability check
            base_trade_amount = self.current_capital * (self.capital_percentage / 100)
            
            # Calculate target price based on ATR or fixed TP
            if self.use_atr_tp_sl:
                dynamic_tp, _ = self.calculate_atr_tp_sl(current_price, market_regime)
            else:
                dynamic_tp = self.take_profit_pct
                
            target_price = current_price * (1 + dynamic_tp / 100)
            
            # Check if trade is profitable after costs
            if not self.is_trade_profitable(current_price, target_price, base_trade_amount):
                bot_logger.warning("Trade rejected: Not profitable after trading costs")
                should_buy = False
        
        return should_buy
    
    def evaluate_sell_signals(self, profit_pct, current_price, rsi, macd_bearish, ema_short, market_regime, dynamic_tp=None, dynamic_sl=None):
        """Evaluate sell signals - extracted from main logic for state machine"""
        should_sell = False
        reason = ""
        
        # Check trailing stop for long position
        if self.long_position and self.use_trailing_stop and self.trailing_stop_price:
            if current_price <= self.trailing_stop_price:
                should_sell = True
                reason = f"Trailing stop hit at ${current_price:.4f}"
                bot_logger.info(f"TRAILING STOP TRIGGERED: {reason}")
                return should_sell, reason
        
        # Check breakeven for long position
        if self.long_position and self.use_breakeven and self.breakeven_triggered:
            breakeven_price = self.last_buy_price * (1 + self.breakeven_offset_pct / 100)
            if current_price <= breakeven_price:
                should_sell = True
                reason = f"Breakeven hit at ${current_price:.4f}"
                bot_logger.info(f"BREAKEVEN TRIGGERED: {reason}")
                return should_sell, reason
        
        # Check trailing stop for short position
        if self.short_position and self.use_trailing_stop and self.short_trailing_stop_price:
            if current_price >= self.short_trailing_stop_price:
                should_sell = True
                reason = f"Short trailing stop hit at ${current_price:.4f}"
                bot_logger.info(f"SHORT TRAILING STOP TRIGGERED: {reason}")
                return should_sell, reason
        
        # Check breakeven for short position
        if self.short_position and self.use_breakeven and self.short_breakeven_triggered:
            breakeven_price = self.last_short_price * (1 - self.breakeven_offset_pct / 100)
            if current_price >= breakeven_price:
                should_sell = True
                reason = f"Short breakeven hit at ${current_price:.4f}"
                bot_logger.info(f"SHORT BREAKEVEN TRIGGERED: {reason}")
                return should_sell, reason
        
        # Use ATR-based TP/SL if enabled, otherwise use dynamic fixed percentages
        if self.use_atr_tp_sl:
            dynamic_tp, dynamic_sl = self.calculate_atr_tp_sl(current_price, market_regime)
        else:
            # Dynamic take profit based on market conditions (legacy method)
            if self.is_sol_strategy:
                if self.short_position:
                    # SOL-specific short parameters
                    dynamic_tp = (self.sol_short_tp_min + self.sol_short_tp_max) / 2  # Average 0.325%
                    dynamic_sl = (self.sol_short_sl_min + self.sol_short_sl_max) / 2  # Average 0.225%
                else:
                    # SOL-specific long parameters
                    dynamic_tp = (self.sol_tp_min + self.sol_tp_max) / 2  # Average 0.325%
                    dynamic_sl = (self.sol_tp_min + self.sol_tp_max) / 2  # Average 0.325%
            else:
                # Generic parameters
                dynamic_tp = self.take_profit_pct
                if "HIGH_VOL" in market_regime:
                    dynamic_tp *= 1.5  # Higher targets in volatile markets
                elif "LOW_VOL" in market_regime:
                    dynamic_tp *= 0.8  # Lower targets in calm markets
                dynamic_sl = self.stop_loss_pct
        
        # Take profit with dynamic adjustment
        if profit_pct >= dynamic_tp - 0.01:  # Larger tolerance for floating point precision
            should_sell = True
            reason = f"Take Profit ({dynamic_tp:.2f}%)"
            bot_logger.info(f"Take profit triggered at {profit_pct:.2f}% (target: {dynamic_tp:.2f}%)")
        
        # Improved stop loss with ATR-based adjustment
        elif profit_pct <= -dynamic_sl:
            should_sell = True
            reason = "Stop Loss"
            bot_logger.info(f"Stop loss triggered at {profit_pct:.2f}%")
        
        # Enhanced trailing stop with dynamic adjustment
        elif self.highest_price_since_buy > self.last_buy_price:
            trailing_stop_pct = ((self.highest_price_since_buy - current_price) / self.highest_price_since_buy) * 100
            dynamic_trailing = 0.5 if "HIGH_VOL" in market_regime else 1.0
            if trailing_stop_pct >= dynamic_trailing and profit_pct > 1.0:
                should_sell = True
                reason = f"Trailing Stop ({dynamic_trailing:.1f}%)"
                bot_logger.info(f"Trailing stop triggered at {trailing_stop_pct:.2f}% with profit {profit_pct:.2f}%")
        
        # RSI-based exit for overbought conditions
        elif rsi > 75 and profit_pct > 0.5:
            should_sell = True
            reason = "RSI Overbought"
            bot_logger.info(f"RSI overbought exit at {rsi:.1f} with profit {profit_pct:.2f}%")
        
        # MACD bearish crossover for timing
        elif macd_bearish and profit_pct > 0.3:
            should_sell = True
            reason = "MACD Bearish Crossover"
            bot_logger.info(f"MACD bearish crossover with profit {profit_pct:.2f}%")
        
        # Price below EMA short as exit signal
        elif current_price < ema_short and profit_pct > 0.5:
            should_sell = True
            reason = "Price Below EMA Short"
            bot_logger.info(f"Price below EMA short with profit {profit_pct:.2f}%")
        
        # Force sell after extended hold if not profitable
        elif len(self.price_history) > 300 and self.current_position == 'long' and profit_pct < 0.2:
            should_sell = True
            reason = "Max Hold Time (Low Profit)"
            bot_logger.info(f"Forcing sell due to max hold time at low profit {profit_pct:.2f}%")
        
        return should_sell, reason

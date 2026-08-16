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
        self.min_confidence_threshold = config.get('min_confidence_threshold', 0.3)  # Lower threshold for more trades
        self.profit_multiplier = config.get('profit_multiplier', 1.0)  # Conservative multiplier
        self.aggressive_mode = config.get('aggressive_mode', True)  # Enable aggressive trading
        
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
        self.current_position = None
        self.trade_count = 0
        self.profit_loss = 0.0
        self.last_buy_price = None
        self.position_size = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.highest_price_since_buy = None
        self.trade_history = []
        self.best_trade_profit = 0.0
        self.worst_trade_loss = 0.0
        
        # Load saved state
        self.load_capital_state()
        
        bot_logger.info(f"Strategy initialized with {self.VOLATILITY_MULTIPLIER}x volatility multiplier")
    
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
                        bot_logger.info(f"Auto-detected {quote_currency} balance: {self.currency_symbol}{self.current_capital:.2f}")
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
                    bot_logger.error(f"Error fetching balance: {e}")
                    # Fallback to saved state
                    if os.path.exists(state_file):
                        with open(state_file, 'r') as f:
                            state = json.load(f)
                            self.current_capital = state.get('current_capital', 18)
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
    
    def calculate_position_size(self, current_price):
        """Calculate position size with dynamic sizing based on confidence and volatility"""
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
        
        order_successful = False
        order_id = None
        
        # Execute real buy order on exchange
        try:
            # Coinbase requires cost parameter for market buy orders
            # Use create_market_buy_order with cost parameter
            order = self.exchange.create_market_buy_order(
                self.symbol,
                trade_value  # pass cost directly
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
            self.current_position = 'long'
            self.last_buy_price = current_price
            self.position_size = position_size
            self.highest_price_since_buy = current_price
        else:
            bot_logger.warning("Buy order not verified - not setting position (paper trading)")
            # Still update position for paper trading
            self.current_position = 'long'
            self.last_buy_price = current_price
            self.position_size = position_size
            self.highest_price_since_buy = current_price
        
        bot_logger.info(f"[BUY #{self.trade_count + 1}] {self.currency_symbol}{current_price:.2f} | Size: {position_size:.6f} | Value: ${trade_value:.2f} | Volatility: {self.VOLATILITY_MULTIPLIER}x | Real: {order_successful}")
        
        return position_size
    
    def place_sell_order(self, current_price, reason):
        """Place sell order"""
        if self.current_position != 'long' or self.last_buy_price is None:
            return
        
        # Check actual base currency balance before attempting to sell
        try:
            balance = self.exchange.fetch_balance()
            # Extract base currency from symbol (e.g., SAND from SAND-USDC)
            base_currency = self.symbol.split('/')[0] if '/' in self.symbol else self.symbol.split('-')[0]
            base_balance = balance.get(base_currency, {}).get('free', 0)
            bot_logger.info(f"Actual {base_currency} balance: {base_balance:.6f}, Attempting to sell: {self.position_size:.6f}")
            
            if base_balance < self.position_size:
                bot_logger.warning(f"Insufficient {base_currency} balance. Have: {base_balance:.6f}, Need: {self.position_size:.6f}")
                # Adjust sell amount to actual available balance
                if base_balance > 0:
                    self.position_size = base_balance
                    bot_logger.info(f"Adjusted sell amount to available balance: {self.position_size:.6f}")
                else:
                    bot_logger.error(f"No {base_currency} available to sell, skipping real order")
                    # Fall back to paper trading
                    self._execute_paper_sell(current_price, reason)
                    return
        except Exception as e:
            bot_logger.error(f"Error checking {base_currency} balance: {e}")
        
        # Execute real sell order on exchange
        try:
            order = self.exchange.create_market_sell_order(self.symbol, self.position_size)
            bot_logger.info(f"[REAL SELL ORDER PLACED] Order ID: {order.get('id', 'N/A')}")
        except Exception as e:
            bot_logger.error(f"Failed to place real sell order: {e}")
            # Fall back to paper trading if order fails
            bot_logger.warning("Falling back to paper trading for this order")
            self._execute_paper_sell(current_price, reason)
            return
        
        # Calculate profit/loss
        profit_pct = ((current_price - self.last_buy_price) / self.last_buy_price) * 100
        profit_amount = (current_price - self.last_buy_price) * self.position_size
        
        # Update statistics
        self.trade_count += 1
        self.profit_loss += profit_amount
        self.current_capital += profit_amount
        
        if profit_amount > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            if profit_amount > self.best_trade_profit:
                self.best_trade_profit = profit_amount
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            if profit_amount < self.worst_trade_loss:
                self.worst_trade_loss = profit_amount
        
        # Record trade
        trade_record = {
            'trade_number': self.trade_count,
            'buy_price': self.last_buy_price,
            'sell_price': current_price,
            'position_size': self.position_size,
            'profit': profit_amount,
            'profit_pct': profit_pct,
            'reason': reason
        }
        self.trade_history.append(trade_record)
        
        bot_logger.info(f"[SELL #{self.trade_count}] {self.currency_symbol}{current_price:.2f} | Profit: ${profit_amount:.2f} ({profit_pct:+.2f}%) | Reason: {reason} | Capital: ${self.current_capital:.2f}")
        
        # Reset position
        self.current_position = None
        self.last_buy_price = None
        self.position_size = 0.0
        self.highest_price_since_buy = None
        
        # Save state
        self.save_capital_state()
        
        return profit_amount
    
    def _execute_paper_sell(self, current_price, reason):
        """Execute paper trading sell when real order fails"""
        # Calculate profit/loss
        profit_pct = ((current_price - self.last_buy_price) / self.last_buy_price) * 100
        profit_amount = (current_price - self.last_buy_price) * self.position_size
        
        # Update statistics
        self.trade_count += 1
        self.profit_loss += profit_amount
        self.current_capital += profit_amount
        
        if profit_amount > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            if profit_amount > self.best_trade_profit:
                self.best_trade_profit = profit_amount
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            if profit_amount < self.worst_trade_loss:
                self.worst_trade_loss = profit_amount
        
        # Record trade
        trade_record = {
            'trade_number': self.trade_count,
            'buy_price': self.last_buy_price,
            'sell_price': current_price,
            'position_size': self.position_size,
            'profit': profit_amount,
            'profit_pct': profit_pct,
            'reason': f"{reason} (Paper Trading)"
        }
        self.trade_history.append(trade_record)
        
        bot_logger.info(f"[SELL #{self.trade_count}] {self.currency_symbol}{current_price:.2f} | Profit: ${profit_amount:.2f} ({profit_pct:+.2f}%) | Reason: {reason} (Paper Trading) | Capital: ${self.current_capital:.2f}")
        
        # Reset position
        self.current_position = None
        self.last_buy_price = None
        self.position_size = 0.0
        self.highest_price_since_buy = None
        
        # Save state
        self.save_capital_state()
    
    def handle_trade_event(self, current_price):
        """Main trading logic"""
        bot_logger.info(f"=== handle_trade_event called === Price: ${current_price:.2f}, Position: {self.current_position}")
        
        # Add price to history
        self.price_history.append(current_price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)
        
        bot_logger.info(f"#{len(self.price_history)} | Price: ${current_price:.2f} | Capital: ${self.current_capital:.2f} | Position: {self.current_position or 'None'} | Trades: {self.trade_count}")
        
        # Need at least some data for indicators
        if len(self.price_history) < 5:
            if self.current_position is None and self.trade_count == 0:
                # Force first trade
                self.place_buy_order(current_price)
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
        
        # Trading logic
        if self.current_position is None:
            # Complex buy signals - require multiple confirmations
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
            if self.ml_enabled and self.use_ml_signals and len(self.price_history) >= 100:
                try:
                    # Create DataFrame for ML prediction
                    ml_data = pd.DataFrame({
                        'close': self.price_history,
                        'volume': [1] * len(self.price_history),  # Placeholder volume
                        'open': self.price_history,  # Use close as placeholder
                        'high': self.price_history,  # Use close as placeholder
                        'low': self.price_history   # Use close as placeholder
                    })
                    
                    ml_signals = self.ml_ensemble.get_trading_signal(ml_data)
                    
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
                        
                        # ML-Primary Decision
                        if self.ml_only:
                            # ML-Only Mode: Ignore traditional indicators completely
                            if ml_buy_score >= 3:
                                should_buy = True
                                bullish_pct = 100  # ML-only mode
                                bot_logger.info(f"ML-ONLY BUY signal - score: {ml_buy_score}")
                            elif ml_sell_score >= 3:
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
                                # Fallback to traditional indicators if ML is uncertain
                                should_buy = bullish_pct >= (self.min_confidence_threshold * 100)
                                bot_logger.info(f"ML uncertain - using traditional indicators")
                            
                        bot_logger.info(f"Final bullish percentage: {bullish_pct:.1f}%")
                        
                except Exception as e:
                    bot_logger.warning(f"ML signal integration failed: {e}")
                    if self.ml_only:
                        should_buy = False  # ML-only mode: don't trade if ML fails
                    else:
                        # Fallback to traditional indicators
                        should_buy = bullish_pct >= (self.min_confidence_threshold * 100)
            else:
                # Use traditional indicators if ML not available
                should_buy = bullish_pct >= (self.min_confidence_threshold * 100)
            
            # Force buy if no trades yet and we have some bullish signals
            if self.trade_count == 0 and bullish_pct > 0:
                should_buy = True
            
            bot_logger.info(f"Buy Check: Bullish={bullish_pct:.1f}% ({bullish_signals:.1f}/{total_signals}) | Threshold={self.min_confidence_threshold*100:.1f}% | ShouldBuy={should_buy}")
            
            if should_buy:
                self.place_buy_order(current_price)
        
        elif self.current_position == 'long':
            # Update highest price for trailing stop
            if current_price > self.highest_price_since_buy:
                self.highest_price_since_buy = current_price
            
            profit_pct = ((current_price - self.last_buy_price) / self.last_buy_price) * 100
            
            # Debug logging for sell logic
            bot_logger.info(f"Position Check: Profit%={profit_pct:.2f}%, TP={self.take_profit_pct}%, SL={self.stop_loss_pct}%, RSI={rsi:.1f}, MACD={'BULL' if macd_bullish else 'BEAR'}, BB={'LOWER' if price_near_lower else 'UPPER' if price_near_upper else 'MID'}")
            
            # Sell signals - SIMPLIFIED for profitability
            should_sell = False
            reason = ""
            
            # Take profit - only sell when we hit target
            if profit_pct >= self.take_profit_pct:
                should_sell = True
                reason = "Take Profit"
                bot_logger.info(f"Take profit triggered at {profit_pct:.2f}%")
            
            # Stop loss - only sell if we're losing significantly
            elif profit_pct <= -self.stop_loss_pct:
                should_sell = True
                reason = "Stop Loss"
                bot_logger.info(f"Stop loss triggered at {profit_pct:.2f}%")
            
            # Trailing stop loss - lock in profits when price drops 1% from peak
            elif self.highest_price_since_buy > self.last_buy_price:
                trailing_stop_pct = ((self.highest_price_since_buy - current_price) / self.highest_price_since_buy) * 100
                if trailing_stop_pct >= 1.0 and profit_pct > 2.0:
                    should_sell = True
                    reason = "Trailing Stop"
                    bot_logger.info(f"Trailing stop triggered at {trailing_stop_pct:.2f}% with profit {profit_pct:.2f}%")
            
            # Only force sell after 200 ticks if we're losing significantly
            elif len(self.price_history) > 200 and self.current_position == 'long' and profit_pct < -2.0:
                should_sell = True
                reason = "Max Hold Time (Loss)"
                bot_logger.info(f"Forcing sell due to max hold time at loss {profit_pct:.2f}%")
            
            if should_sell:
                self.place_sell_order(current_price, reason)

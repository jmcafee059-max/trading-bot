import logging
import logging.handlers
import pandas as pd
import numpy as np
import json
import os

# Set up bot logger for centralized logging
bot_logger = logging.getLogger('bot')
bot_logger.setLevel(logging.INFO)

# Clear any existing handlers to prevent duplicates
bot_logger.handlers.clear()

# Add file handler to bot_logger
log_handler = logging.handlers.RotatingFileHandler('bot_logs.log', maxBytes=10000, backupCount=5)
log_handler.setLevel(logging.INFO)
log_handler.set_name('bot_file_handler')
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_handler.setFormatter(log_formatter)
bot_logger.addHandler(log_handler)

class SimpleRSIStrategy:
    """
    Aggressive RSI-based trading strategy for small capital
    - Optimized for £18 starting capital
    - Multiple trading signals for maximum profit opportunities
    - Tight risk management to minimize losses
    """
    
    def __init__(self, exchange, config):
        self.exchange = exchange
        self.config = config
        self.rsi_period = config.get('rsi_period', 7)
        self.rsi_overbought = config.get('rsi_overbought', 65)
        self.rsi_oversold = config.get('rsi_oversold', 35)
        self.take_profit_pct = config.get('take_profit_percent', 2.0)
        self.stop_loss_pct = config.get('stop_loss_percent', 0.5)
        
        # Dynamic position sizing based on capital
        self.starting_capital = config.get('starting_capital', 18)
        self.actual_balance = config.get('actual_balance', self.starting_capital)  # Use actual exchange balance if available
        self.initial_capital = self.actual_balance  # Keep track of initial capital
        self.capital_percentage = config.get('capital_percentage', 80)
        
        self.currency_symbol = config.get('symbol', 'BTC/GBP').split('/')[1]  # Get currency symbol
        
        # Initialize default values
        self.price_history = []
        self.current_position = None
        self.trade_count = 0
        self.profit_loss = 0.0
        self.last_buy_price = None
        self.position_size = 0.0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.current_capital = self.actual_balance
        
        # Advanced features
        self.highest_price_since_buy = None  # For trailing stop loss
        self.volatility_multiplier = 5.0  # Fixed 5x volatility multiplier (engrained)
        self.ma_short_period = 9
        self.ma_long_period = 21
        self.trade_history = []  # Track all trades for analytics
        self.best_trade_profit = 0.0
        self.worst_trade_loss = 0.0
        
        # Load saved capital state if exists (will override defaults)
        self.load_capital_state()
    
    def load_capital_state(self):
        """Load capital state from file for persistence between sessions"""
        state_file = 'capital_state.json'
        try:
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.current_capital = state.get('current_capital', self.actual_balance)
                    self.trade_count = state.get('trade_count', 0)
                    self.profit_loss = state.get('profit_loss', 0.0)
                    self.consecutive_wins = state.get('consecutive_wins', 0)
                    self.consecutive_losses = state.get('consecutive_losses', 0)
                    bot_logger.info(f"Loaded capital state: {self.currency_symbol}{self.current_capital:.2f}, Trades: {self.trade_count}")
            else:
                self.current_capital = self.actual_balance
        except Exception as e:
            bot_logger.warning(f"Could not load capital state: {e}")
            self.current_capital = self.actual_balance
    
    def save_capital_state(self):
        """Save capital state to file for persistence between sessions"""
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
        
    def calculate_rsi(self, prices, period=7):
        """Calculate RSI indicator"""
        if len(prices) < 2:
            return None
            
        df = pd.DataFrame({'price': prices})
        delta = df['price'].diff()
        
        # Use available data length if period is larger than data length
        actual_period = min(period, len(prices) - 1)
        if actual_period < 1:
            actual_period = 1
        
        gain = (delta.where(delta > 0, 0)).rolling(window=actual_period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=actual_period, min_periods=1).mean()
        
        # Avoid division by zero
        if loss.iloc[-1] == 0:
            return 100 if gain.iloc[-1] > 0 else 50
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1]
    
    def calculate_position_size(self, current_price):
        """Calculate dynamic position size based on available capital"""
        # Try to fetch actual balance from exchange
        try:
            if hasattr(self.exchange, 'apiKey') and self.exchange.apiKey:
                balance = self.exchange.fetch_balance()
                symbol_quote = self.config.get('symbol', 'BTC/GBP').split('/')[1]
                
                # Handle Coinbase balance structure
                if 'free' in balance and symbol_quote in balance['free']:
                    actual_available = float(balance['free'][symbol_quote])
                    if actual_available > 0:
                        self.current_capital = actual_available
                        logging.info(f"Updated capital from exchange: {self.current_capital:.2f} {symbol_quote}")
        except Exception as e:
            logging.debug(f"Could not update balance: {e}")
        
        # Use 90% of available capital per trade with 5x volatility multiplier
        trade_amount = self.current_capital * (self.capital_percentage / 100) * self.volatility_multiplier
        
        # Debug logging to verify calculation
        bot_logger.info(f"Position Calculation: Capital=${self.current_capital:.2f}, Cap%={self.capital_percentage}, Vol={self.volatility_multiplier}, TradeAmount=${trade_amount:.2f}")
        
        # Calculate position size in BTC
        position_size = trade_amount / current_price
        
        # Minimum position size to avoid dust trades
        min_position = 0.00001  # Coinbase minimum
        
        return max(position_size, min_position)
    
    def calculate_volatility(self, prices):
        """Calculate price volatility for dynamic position sizing"""
        # Fixed 5x volatility multiplier as requested
        return 5.0
    
    def calculate_ema(self, prices, period=9):
        if len(prices) < 2:
            return None
            
        df = pd.DataFrame({'price': prices})
        # Use available data length if period is larger than data length
        actual_period = min(period, len(prices))
        if actual_period < 1:
            actual_period = 1
        ema = df['price'].ewm(span=actual_period, adjust=False).mean()
        return ema.iloc[-1]
    
    def on_event(self, event):
        """Handle trading events"""
        if hasattr(event, 'price'):
            self.handle_trade_event(event)
        elif hasattr(event, 'bids'):
            self.handle_book_event(event)
        elif hasattr(event, 'status'):
            self.handle_order_event(event)
    
    def handle_trade_event(self, event):
        """Handle trade events - update price history and check signals"""
        current_price = event.price if hasattr(event, 'price') else event
        self.price_history.append(current_price)
        
        # Keep last 50 prices for faster signal response
        if len(self.price_history) > 50:
            self.price_history = self.price_history[-50:]
        
        currency_symbol = self.currency_symbol
        
        # Log every iteration to show bot is working
        bot_logger.info(f"#{len(self.price_history)} | Price: {currency_symbol}{current_price:.2f} | Capital: {currency_symbol}{self.current_capital:.2f} | Position: {self.current_position or 'None'} | Trades: {self.trade_count}")
        
        # Calculate indicators immediately (force RSI calculation with minimal data)
        if len(self.price_history) >= 2:
            rsi = self.calculate_rsi(self.price_history, self.rsi_period)
            ema_short = self.calculate_ema(self.price_history, 5)
            ema_long = self.calculate_ema(self.price_history, 15)
            ma_short = self.calculate_ema(self.price_history, self.ma_short_period)
            ma_long = self.calculate_ema(self.price_history, self.ma_long_period)
            
            # Volatility multiplier is hardcoded to 5.0 in __init__ - don't override it
            # self.volatility_multiplier = self.calculate_volatility(self.price_history)
            
            if rsi is not None and ema_short is not None and ema_long is not None:
                trend = "BULLISH" if ema_short > ema_long else "BEARISH"
                ma_trend = "BULLISH" if ma_short and ma_long and ma_short > ma_long else "BEARISH"
                
                # Log RSI analysis
                bot_logger.info(f"RSI: {rsi:.1f} | Trend: {trend} | MA Trend: {ma_trend} | Volatility Multiplier: {self.volatility_multiplier:.2f} | Overbought: {self.rsi_overbought} | Oversold: {self.rsi_oversold}")
                
                # AGGRESSIVE ENTRY SIGNALS (multiple conditions)
                should_buy = False
                should_sell = False
                
                if self.current_position is None:
                    # FORCE FIRST TRADE FOR DEMONSTRATION
                    if self.trade_count == 0:
                        should_buy = True
                        bot_logger.info("FORCING FIRST TRADE FOR DEMONSTRATION")
                    # Strong buy signals with MA confirmation
                    elif (rsi < self.rsi_oversold or  # Oversold
                        (rsi < 45 and trend == "BULLISH" and ma_trend == "BULLISH") or  # Strong bullish momentum
                        (rsi < 50 and self.consecutive_losses < 2 and ma_trend == "BULLISH") or  # Recovery with trend
                        (rsi < 55 and trend == "BULLISH" and self.consecutive_wins >= 2)):  # Momentum trading
                        should_buy = True
                
                elif self.current_position == 'long' and self.last_buy_price:
                    profit_pct = ((current_price - self.last_buy_price) / self.last_buy_price) * 100
                    
                    # Update highest price for trailing stop loss
                    if self.highest_price_since_buy is None or current_price > self.highest_price_since_buy:
                        self.highest_price_since_buy = current_price
                    
                    # Calculate trailing stop loss (0.5% below highest price)
                    trailing_stop_pct = ((current_price - self.highest_price_since_buy) / self.highest_price_since_buy) * 100
                    
                    # Exit signals with trailing stop loss
                    if (profit_pct >= self.take_profit_pct or  # Take profit
                        rsi > self.rsi_overbought or  # Overbought
                        profit_pct <= -self.stop_loss_pct or  # Fixed stop loss
                        trailing_stop_pct <= -0.5 or  # Trailing stop loss
                        (profit_pct >= 0.3 and trend == "BEARISH" and ma_trend == "BEARISH") or  # Trend reversal
                        (profit_pct >= 0.8 and rsi > 55)):  # Quick profit taking
                        should_sell = True
                        if trailing_stop_pct <= -0.5:
                            bot_logger.info(f"TRAILING STOP LOSS TRIGGERED: {trailing_stop_pct:.2f}%")
                
                # Execute trades with dynamic position sizing
                if should_buy:
                    self.position_size = self.calculate_position_size(current_price)
                    self.place_buy_order(current_price)
                elif should_sell:
                    self.place_sell_order(current_price)
    
    def handle_book_event(self, event):
        """Handle order book events"""
        pass
    
    def handle_order_event(self, event):
        """Handle order events"""
        bot_logger.info(f"Order event: {event}")
        if hasattr(event, 'status') and event.status == 'filled':
            if hasattr(event, 'side') and event.side == 'buy':
                self.current_position = 'long'
                bot_logger.info(f"Position opened at {event.price}")
            elif hasattr(event, 'side') and event.side == 'sell':
                self.current_position = None
                bot_logger.info(f"Position closed at {event.price}")
    
    def place_buy_order(self, price):
        """Place a buy order with dynamic position sizing"""
        try:
            symbol = self.config.get('symbol', 'BTC/USD')
            
            # Check if exchange has API keys (paper trading mode)
            if not hasattr(self.exchange, 'apiKey') or not self.exchange.apiKey or not self.exchange.apiKey:
                self.trade_count += 1
                self.last_buy_price = price
                self.current_position = 'long'
                self.highest_price_since_buy = price  # Reset trailing stop
                trade_value = self.position_size * price
                bot_logger.info(f"[PAPER BUY #{self.trade_count}] {self.currency_symbol}{price:.2f} | Size: {self.position_size:.6f} BTC | Value: {self.currency_symbol}{trade_value:.2f} | Volatility Mult: {self.volatility_multiplier:.2f}")
                return
            
            # Calculate the cost in quote currency for market buy
            cost = self.position_size * price
            
            # Coinbase requires price for market buy orders or use cost parameter
            order = self.exchange.create_order(
                symbol, 
                'market', 
                'buy', 
                self.position_size, 
                price,  # Provide price for Coinbase
                {'cost': cost}  # Additional cost parameter
            )
            self.trade_count += 1
            self.last_buy_price = price
            self.current_position = 'long'
            trade_value = self.position_size * price
            bot_logger.info(f"[BUY #{self.trade_count}] {self.currency_symbol}{price:.2f} | Size: {self.position_size:.6f} BTC | Value: {self.currency_symbol}{trade_value:.2f}")
        except Exception as e:
            bot_logger.error(f"Error placing buy order: {e}")
    
    def place_sell_order(self, price):
        """Place a sell order with capital tracking"""
        try:
            symbol = self.config.get('symbol', 'BTC/USD')
            
            # Calculate profit/loss
            trade_profit = 0.0
            profit_pct = 0.0
            
            if self.last_buy_price:
                trade_profit = (price - self.last_buy_price) * self.position_size
                profit_pct = ((price - self.last_buy_price) / self.last_buy_price) * 100
                self.profit_loss += trade_profit
                
                # Update capital and track performance
                if profit_pct > 0:
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                    self.current_capital += trade_profit
                    # Update best trade
                    if trade_profit > self.best_trade_profit:
                        self.best_trade_profit = trade_profit
                else:
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                    self.current_capital += trade_profit  # Can be negative
                    # Update worst trade
                    if trade_profit < self.worst_trade_loss:
                        self.worst_trade_loss = trade_profit
                
                # Record trade history
                self.trade_history.append({
                    'entry_price': self.last_buy_price,
                    'exit_price': price,
                    'profit': trade_profit,
                    'profit_pct': profit_pct,
                    'position_size': self.position_size
                })
                
                # Save capital state after each trade
                self.save_capital_state()
            
            # Check if exchange has API keys (paper trading mode)
            if not hasattr(self.exchange, 'apiKey') or not self.exchange.apiKey:
                self.current_position = None
                self.highest_price_since_buy = None  # Reset trailing stop
                
                # Enhanced trade logging with analytics
                avg_profit = sum(t['profit'] for t in self.trade_history) / len(self.trade_history) if self.trade_history else 0
                win_rate = len([t for t in self.trade_history if t['profit'] > 0]) / len(self.trade_history) if self.trade_history else 0
                
                bot_logger.info(f"[SELL #{self.trade_count}] {self.currency_symbol}{price:.2f} | P/L: {self.currency_symbol}{trade_profit:.2f} ({profit_pct:+.2f}%) | Total: {self.currency_symbol}{self.current_capital:.2f} | Streak: {self.consecutive_wins}W/{self.consecutive_losses}L | Win Rate: {win_rate:.1%} | Best: {self.currency_symbol}{self.best_trade_profit:.2f} | Worst: {self.currency_symbol}{self.worst_trade_loss:.2f}")
                self.last_buy_price = None
                return
            
            order = self.exchange.create_market_sell_order(
                symbol, 
                self.position_size
            )
            self.current_position = None
            bot_logger.info(f"[SELL #{self.trade_count}] {self.currency_symbol}{price:.2f} | P/L: {self.currency_symbol}{trade_profit:.2f} ({profit_pct:+.2f}%) | Total: {self.currency_symbol}{self.current_capital:.2f} | Streak: {self.consecutive_wins}W/{self.consecutive_losses}L")
            self.last_buy_price = None
        except Exception as e:
            bot_logger.error(f"Error placing sell order: {e}")
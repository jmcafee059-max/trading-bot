import logging
import os
import time
from dotenv import load_dotenv
import ccxt
from simple_strategy import SimpleRSIStrategy

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradingBot:
    def __init__(self):
        self.config = {
            'exchange_id': os.getenv('EXCHANGE_ID', 'binance'),
            'api_key': os.getenv('API_KEY'),
            'secret_key': os.getenv('SECRET_KEY'),
            'symbol': os.getenv('SYMBOL', 'BTC/USD'),
            'timeframe': os.getenv('TIMEFRAME', '15m'),
            'starting_capital': float(os.getenv('STARTING_CAPITAL', '18')),
            'capital_percentage': float(os.getenv('CAPITAL_PERCENTAGE', '80')),
            'risk_percentage': float(os.getenv('RISK_PERCENTAGE', '2.0')),
            'rsi_period': int(os.getenv('RSI_PERIOD', '7')),
            'rsi_overbought': int(os.getenv('RSI_OVERBOUGHT', '65')),
            'rsi_oversold': int(os.getenv('RSI_OVERSOLD', '35')),
            'take_profit_percent': float(os.getenv('TAKE_PROFIT_PERCENT', '2.0')),
            'stop_loss_percent': float(os.getenv('STOP_LOSS_PERCENT', '0.5')),
        }
        
        self.exchange = None
        self.strategy = None
        
    def initialize_exchange(self):
        """Initialize the exchange connection"""
        try:
            # Create CCXT exchange instance
            exchange_class = getattr(ccxt, self.config['exchange_id'])
            
            # Only add API keys if they are provided
            exchange_config = {
                'enableRateLimit': True,
            }
            
            if self.config['api_key'] and self.config['api_key'] != 'your_api_key_here':
                exchange_config['apiKey'] = self.config['api_key']
                exchange_config['secret'] = self.config['secret_key']
                exchange_config['createMarketBuyOrderRequiresPrice'] = False  # Fix for Coinbase
                logger.info("Using authenticated exchange connection")
            else:
                logger.info("Using unauthenticated exchange connection (paper trading mode)")
            
            self.exchange = exchange_class(exchange_config)
            
            # Test connection
            markets = self.exchange.load_markets()
            logger.info(f"Connected to {self.config['exchange_id']}")
            logger.info(f"Available markets: {len(markets)}")
            
            # Check if symbol exists
            symbol = self.config['symbol']
            if symbol not in markets:
                logger.warning(f"Symbol {symbol} not found. Available similar symbols:")
                similar = [s for s in markets.keys() if 'BTC' in s and 'USD' in s][:5]
                for s in similar:
                    logger.warning(f"  - {s}")
            
            # Fetch and display available balance
            if hasattr(self.exchange, 'apiKey') and self.exchange.apiKey:
                try:
                    balance = self.exchange.fetch_balance()
                    symbol_quote = symbol.split('/')[1]  # Get quote currency (GBP)
                    
                    logger.info(f"Balance structure keys: {list(balance.keys())}")
                    
                    # Coinbase returns balance in a specific structure
                    if 'total' in balance:
                        logger.info("Available balances:")
                        for currency, amount in balance['total'].items():
                            if amount > 0:
                                free_amount = balance['free'].get(currency, 0)
                                logger.info(f"  {currency}: {free_amount} (available) / {amount} (total)")
                        
                        if symbol_quote in balance['free']:
                            available_balance = balance['free'][symbol_quote]
                            logger.info(f"Available {symbol_quote} balance: {available_balance}")
                            
                            # Update strategy with actual balance
                            self.config['actual_balance'] = float(available_balance)
                            self.config['starting_capital'] = float(available_balance)
                        else:
                            logger.warning(f"No {symbol_quote} balance found")
                    else:
                        logger.warning(f"Unexpected balance structure")
                except Exception as e:
                    logger.warning(f"Could not fetch balance: {e}")
                    logger.info("Using configured starting capital instead")
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize exchange: {e}")
            return False
    
    def initialize_strategy(self):
        """Initialize the trading strategy"""
        try:
            self.strategy = SimpleRSIStrategy(self.exchange, self.config)
            logger.info("Strategy initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize strategy: {e}")
            return False
    
    def run(self):
        """Run the trading bot"""
        logger.info("Starting trading bot...")
        
        # Initialize components
        if not self.initialize_exchange():
            logger.error("Failed to initialize exchange. Exiting.")
            return
        
        if not self.initialize_strategy():
            logger.error("Failed to initialize strategy. Exiting.")
            return
        
        # Check if API keys are provided
        if not self.config['api_key'] or self.config['api_key'] == 'your_api_key_here':
            logger.info("Running in paper trading mode (no API keys provided)")
            logger.info("Set API_KEY and SECRET_KEY in .env file for live trading")
        
        try:
            # Get current ticker data
            symbol = self.config['symbol']
            logger.info(f"Starting trading loop for {symbol}")
            
            # Continuous trading loop
            iteration_count = 0
            while True:
                try:
                    iteration_count += 1
                    
                    # Get current price
                    ticker = self.exchange.fetch_ticker(symbol)
                    current_price = ticker['last']
                    
                    # Directly call strategy with price (bypass event system for paper trading)
                    self.strategy.handle_trade_event(current_price)
                    
                    # Log performance summary every 5 iterations
                    if iteration_count % 5 == 0:
                        currency_symbol = self.strategy.currency_symbol
                        logger.info(f"=== PERFORMANCE SUMMARY ===")
                        logger.info(f"Total Trades: {self.strategy.trade_count}")
                        logger.info(f"Current Capital: {currency_symbol}{self.strategy.current_capital:.2f}")
                        logger.info(f"Total P/L: {currency_symbol}{self.strategy.profit_loss:.2f}")
                        logger.info(f"Win Rate: {self.strategy.consecutive_wins}W/{self.strategy.consecutive_losses}L")
                        logger.info(f"========================")
                    
                    # Shorter wait time for more opportunities (30 seconds)
                    time.sleep(30)
                    
                except KeyboardInterrupt:
                    logger.info("Stopping trading bot...")
                    currency_symbol = self.strategy.currency_symbol
                    logger.info(f"=== FINAL RESULTS ===")
                    logger.info(f"Total Trades: {self.strategy.trade_count}")
                    logger.info(f"Final Capital: {currency_symbol}{self.strategy.current_capital:.2f}")
                    logger.info(f"Total P/L: {currency_symbol}{self.strategy.profit_loss:.2f}")
                    logger.info(f"Return: {((self.strategy.current_capital - self.strategy.initial_capital) / self.strategy.initial_capital) * 100:.2f}%")
                    break
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                    time.sleep(15)  # Shorter wait before retrying
            
            # In a real implementation, you would set up websockets
            # to continuously receive trade events
            # For this example, we'll just fetch data periodically
            
        except KeyboardInterrupt:
            logger.info("Stopping trading bot...")
        except Exception as e:
            logger.error(f"Error in trading bot: {e}")
        finally:
            if self.exchange:
                logger.info("Exchange connection closed")

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
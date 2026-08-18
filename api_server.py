from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import threading
import os
import json
from dotenv import load_dotenv
import ccxt
from simple_strategy_v2 import SimpleRSIStrategy
from tradekit_signals import signal_store
import time
import logging
from logging.handlers import RotatingFileHandler

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler('bot_logs.log', maxBytes=1024*1024, backupCount=5),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger(__name__)

# Global bot state
bot_thread = None
bot_running = False
strategy_instance = None
bot_config = {
    'exchange_id': os.getenv('EXCHANGE_ID', 'binance'),
    'api_key': os.getenv('API_KEY'),
    'secret_key': os.getenv('SECRET_KEY'),
    'symbol': os.getenv('SYMBOL', 'BTC/USDC'),
    # These were previously never passed through, so ENABLE_COIN_SCANNER had
    # no effect - the strategy's enable_coin_scanner always fell back to its
    # own default (False) regardless of the env var.
    'enable_coin_scanner': os.getenv('ENABLE_COIN_SCANNER', 'false').lower() == 'true',
    'scan_interval_minutes': int(os.getenv('SCAN_INTERVAL_MINUTES', '5')),
    'min_edge_score': float(os.getenv('MIN_EDGE_SCORE', '2.0')),
    'max_pairs_to_scan': int(os.getenv('MAX_PAIRS_TO_SCAN', '20')),
    'timeframe': os.getenv('TIMEFRAME', '15m'),
    'starting_capital': float(os.getenv('STARTING_CAPITAL', '18')) if os.getenv('STARTING_CAPITAL', '18') != 'auto' else 18,
    'capital_percentage': float(os.getenv('CAPITAL_PERCENTAGE', '90')),
    'risk_percentage': float(os.getenv('RISK_PERCENTAGE', '2.0')),
    'rsi_period': int(os.getenv('RSI_PERIOD', '3')),
    'rsi_overbought': int(os.getenv('RSI_OVERBOUGHT', '65')),
    'rsi_oversold': int(os.getenv('RSI_OVERSOLD', '35')),
    'take_profit_percent': float(os.getenv('TAKE_PROFIT_PERCENT', '5.0')),
    'stop_loss_percent': float(os.getenv('STOP_LOSS_PERCENT', '0.1')),
    'volatility_multiplier': float(os.getenv('VOLATILITY_MULTIPLIER', '2')),
    'ema_short': int(os.getenv('EMA_SHORT', '9')),
    'ema_long': int(os.getenv('EMA_LONG', '21')),
    'sma_period': int(os.getenv('SMA_PERIOD', '50')),
    'volume_threshold': float(os.getenv('VOLUME_THRESHOLD', '1.5')),
    'momentum_period': int(os.getenv('MOMENTUM_PERIOD', '14')),
    'max_position_size': float(os.getenv('MAX_POSITION_SIZE', '0.5')),
    'min_confidence_threshold': float(os.getenv('MIN_CONFIDENCE_THRESHOLD', '0.5')),
    'profit_multiplier': float(os.getenv('PROFIT_MULTIPLIER', '1.0')),
    'aggressive_mode': os.getenv('AGGRESSIVE_MODE', 'false').lower() == 'true',
    'ml_enabled': os.getenv('ML_ENABLED', 'true').lower() == 'true',
    'use_ml_signals': os.getenv('USE_ML_SIGNALS', 'true').lower() == 'true',
    'ml_only': os.getenv('ML_ONLY', 'false').lower() == 'true',
    'openai_enabled': os.getenv('OPENAI_ENABLED', 'false').lower() == 'true',
    'use_atr_tp_sl': os.getenv('USE_ATR_TP_SL', 'false').lower() == 'true',
    'use_risk_based_sizing': os.getenv('USE_RISK_BASED_SIZING', 'false').lower() == 'true',
    'risk_per_trade_percent': float(os.getenv('RISK_PER_TRADE_PERCENT', '5.0')),
    'max_position_risk_percent': float(os.getenv('MAX_POSITION_RISK_PERCENT', '10.0')),
    'use_dont_trade_engine': os.getenv('USE_DONT_TRADE_ENGINE', 'false').lower() == 'true',
    'max_consecutive_losses': int(os.getenv('MAX_CONSECUTIVE_LOSSES', '3')),
    'cooling_off_period_minutes': int(os.getenv('COOLING_OFF_PERIOD_MINUTES', '30')),
    'use_partial_profit_taking': os.getenv('USE_PARTIAL_PROFIT_TAKING', 'false').lower() == 'true',
    'first_tp_percent': float(os.getenv('FIRST_TP_PERCENT', '0.5')),
    'second_tp_percent': float(os.getenv('SECOND_TP_PERCENT', '1.0')),
    'third_tp_percent': float(os.getenv('THIRD_TP_PERCENT', '2.0')),
    'use_setup_score': os.getenv('USE_SETUP_SCORE', 'false').lower() == 'true',
    'min_setup_score': int(os.getenv('MIN_SETUP_SCORE', '50')),
    'use_relative_strength': os.getenv('USE_RELATIVE_STRENGTH', 'false').lower() == 'true',
    'relative_strength_weight': float(os.getenv('RELATIVE_STRENGTH_WEIGHT', '0.2')),
    'use_btc_weather': os.getenv('USE_BTC_WEATHER', 'false').lower() == 'true',
    'btc_weather_weight': float(os.getenv('BTC_WEATHER_WEIGHT', '0.3')),
    'use_adaptive_confidence': os.getenv('USE_ADAPTIVE_CONFIDENCE', 'false').lower() == 'true',
    'normal_confidence_threshold': float(os.getenv('NORMAL_CONFIDENCE_THRESHOLD', '0.65')),
    'strong_trend_confidence_threshold': float(os.getenv('STRONG_TREND_CONFIDENCE_THRESHOLD', '0.60')),
    'choppy_market_confidence_threshold': float(os.getenv('CHOPPY_MARKET_CONFIDENCE_THRESHOLD', '0.75')),
    'extreme_volatility_mode': os.getenv('EXTREME_VOLATILITY_MODE', 'false').lower() == 'true',
    'use_trading_cost_model': os.getenv('USE_TRADING_COST_MODEL', 'false').lower() == 'true',
    'maker_fee_percent': float(os.getenv('MAKER_FEE_PERCENT', '0.4')),
    'taker_fee_percent': float(os.getenv('TAKER_FEE_PERCENT', '0.6')),
    'estimated_spread_percent': float(os.getenv('ESTIMATED_SPREAD_PERCENT', '0.02')),
    'estimated_slippage_percent': float(os.getenv('ESTIMATED_SLIPPAGE_PERCENT', '0.05')),
    'min_expected_net_profit': float(os.getenv('MIN_EXPECTED_NET_PROFIT', '0.2')),
    'atr_tp_multiplier_low': float(os.getenv('ATR_TP_MULTIPLIER_LOW', '0.8')),
    'atr_tp_multiplier_normal': float(os.getenv('ATR_TP_MULTIPLIER_NORMAL', '1.5')),
    'atr_tp_multiplier_high': float(os.getenv('ATR_TP_MULTIPLIER_HIGH', '2.5')),
    'atr_sl_multiplier': float(os.getenv('ATR_SL_MULTIPLIER', '1.2')),
    'atr_period': int(os.getenv('ATR_PERIOD', '14')),
    # Paper trading: simulate fills against real live prices without ever
    # placing a real order. Starting balance is fixed rather than fetched
    # from the real Coinbase account.
    'paper_trading': os.getenv('PAPER_TRADING', 'false').lower() == 'true',
    'paper_trading_balance': float(os.getenv('PAPER_TRADING_BALANCE', '100')),
    # These were previously never passed through, so USE_TRAILING_STOP /
    # USE_BREAKEVEN env vars had no effect - the strategy always fell back to
    # its own hardcoded default (True) regardless of what was configured.
    'use_trailing_stop': os.getenv('USE_TRAILING_STOP', 'true').lower() == 'true',
    'trailing_stop_activation_pct': float(os.getenv('TRAILING_STOP_ACTIVATION_PCT', '0.15')),
    'trailing_stop_distance_pct': float(os.getenv('TRAILING_STOP_DISTANCE_PCT', '0.10')),
    'use_breakeven': os.getenv('USE_BREAKEVEN', 'true').lower() == 'true',
    'breakeven_activation_pct': float(os.getenv('BREAKEVEN_ACTIVATION_PCT', '0.20')),
    'breakeven_offset_pct': float(os.getenv('BREAKEVEN_OFFSET_PCT', '0.02')),
    'enable_short_trading': os.getenv('ENABLE_SHORT_TRADING', 'false').lower() == 'true',
    'max_runtime_hours': int(os.getenv('MAX_RUNTIME_HOURS', '24')),
    'hedged_mode': os.getenv('HEDGED_MODE', 'false').lower() == 'true',
    'max_signal_age_seconds': int(os.getenv('MAX_SIGNAL_AGE_SECONDS', '30')),
    'use_synchronized_execution': os.getenv('USE_SYNCHRONIZED_EXECUTION', 'false').lower() == 'true',
    'use_tradekit': os.getenv('USE_TRADEKIT', 'false').lower() == 'true',
    'tradekit_min_score': int(os.getenv('TRADEKIT_MIN_SCORE', '80')),
    'tradekit_liquidity_filter': os.getenv('TRADEKIT_LIQUIDITY_FILTER', 'true').lower() == 'true',
    'tradekit_orderbook_analysis': os.getenv('TRADEKIT_ORDERBOOK_ANALYSIS', 'true').lower() == 'true',
    'tradekit_volatility_analysis': os.getenv('TRADEKIT_VOLATILITY_ANALYSIS', 'true').lower() == 'true',
    'tradekit_backtesting': os.getenv('TRADEKIT_BACKTESTING', 'true').lower() == 'true',
    'tradekit_debug': os.getenv('TRADEKIT_DEBUG', 'false').lower() == 'true',
    # Real TradeKit (trader.dev) live signal webhook - separate from the
    # local indicator adapter above. Off by default; does not change any
    # existing trading behavior unless explicitly enabled.
    'tradekit_live_signals': os.getenv('TRADEKIT_LIVE_SIGNALS', 'false').lower() == 'true',
    'tradekit_max_signal_age_seconds': int(os.getenv('TRADEKIT_MAX_SIGNAL_AGE_SECONDS', '900')),
    'tradekit_signal_bonus': float(os.getenv('TRADEKIT_SIGNAL_BONUS', '8')),
}

TRADEKIT_WEBHOOK_SECRET = os.getenv('TRADEKIT_WEBHOOK_SECRET', '')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(bot_config)

@app.route('/webhook/tradekit', methods=['POST'])
def tradekit_webhook():
    """
    Receives alert payloads from a real TradeKit (trader.dev) Webhook alert
    channel. This never places or modifies orders itself - it only records
    the latest signal so the strategy can optionally use it as one more
    input, gated behind TRADEKIT_LIVE_SIGNALS (off by default).
    """
    if not TRADEKIT_WEBHOOK_SECRET:
        return jsonify({'error': 'webhook not configured (TRADEKIT_WEBHOOK_SECRET unset)'}), 503

    provided_secret = request.headers.get('X-Tradekit-Secret', '')
    if provided_secret != TRADEKIT_WEBHOOK_SECRET:
        return jsonify({'error': 'unauthorized'}), 401

    payload = request.get_json(silent=True) or {}
    entry = signal_store.record_signal(payload)
    logger.info(f"TradeKit webhook signal received: {entry['asset']} {entry['direction']} ({entry['signal_type']})")
    return jsonify({'status': 'ok', 'recorded': entry}), 200

@app.route('/webhook/tradekit/status', methods=['GET'])
def tradekit_webhook_status():
    return jsonify(signal_store.all_signals())

@app.route('/api/config', methods=['POST'])
def update_config():
    data = request.json
    for key, value in data.items():
        if key in bot_config:
            bot_config[key] = value
    
    # Update .env file
    env_updates = {
        'EXCHANGE_ID': bot_config['exchange_id'],
        'SYMBOL': bot_config['symbol'],
        'TIMEFRAME': bot_config['timeframe'],
        'STARTING_CAPITAL': str(bot_config['starting_capital']),
        'CAPITAL_PERCENTAGE': str(bot_config['capital_percentage']),
        'RISK_PERCENTAGE': str(bot_config['risk_percentage']),
        'RSI_PERIOD': str(bot_config['rsi_period']),
        'RSI_OVERBOUGHT': str(bot_config['rsi_overbought']),
        'RSI_OVERSOLD': str(bot_config['rsi_oversold']),
        'TAKE_PROFIT_PERCENT': str(bot_config['take_profit_percent']),
        'STOP_LOSS_PERCENT': str(bot_config['stop_loss_percent']),
        'VOLATILITY_MULTIPLIER': str(bot_config['volatility_multiplier']),
        'EMA_SHORT': str(bot_config['ema_short']),
        'EMA_LONG': str(bot_config['ema_long']),
        'SMA_PERIOD': str(bot_config['sma_period']),
        'VOLUME_THRESHOLD': str(bot_config['volume_threshold']),
        'MOMENTUM_PERIOD': str(bot_config['momentum_period']),
        'MAX_POSITION_SIZE': str(bot_config['max_position_size']),
        'MIN_CONFIDENCE_THRESHOLD': str(bot_config['min_confidence_threshold']),
        'PROFIT_MULTIPLIER': str(bot_config['profit_multiplier']),
        'AGGRESSIVE_MODE': str(bot_config['aggressive_mode']).lower(),
    }
    
    with open('.env', 'w') as f:
        f.write("# Exchange Configuration\n")
        f.write(f"EXCHANGE_ID={env_updates['EXCHANGE_ID']}\n")
        f.write(f"API_KEY={os.getenv('API_KEY', 'your_api_key_here')}\n")
        f.write(f"SECRET_KEY={os.getenv('SECRET_KEY', 'your_secret_key_here')}\n\n")
        f.write("# Trading Configuration\n")
        f.write(f"SYMBOL={env_updates['SYMBOL']}\n")
        f.write(f"TIMEFRAME={env_updates['TIMEFRAME']}\n")
        f.write(f"STARTING_CAPITAL={env_updates['STARTING_CAPITAL']}\n")
        f.write(f"CAPITAL_PERCENTAGE={env_updates['CAPITAL_PERCENTAGE']}\n")
        f.write(f"RISK_PERCENTAGE={env_updates['RISK_PERCENTAGE']}\n\n")
        f.write("# Strategy Configuration\n")
        f.write(f"RSI_PERIOD={env_updates['RSI_PERIOD']}\n")
        f.write(f"RSI_OVERBOUGHT={env_updates['RSI_OVERBOUGHT']}\n")
        f.write(f"RSI_OVERSOLD={env_updates['RSI_OVERSOLD']}\n")
        f.write(f"TAKE_PROFIT_PERCENT={env_updates['TAKE_PROFIT_PERCENT']}\n")
        f.write(f"STOP_LOSS_PERCENT={env_updates['STOP_LOSS_PERCENT']}\n")
        f.write(f"VOLATILITY_MULTIPLIER={env_updates['VOLATILITY_MULTIPLIER']}\n")
        f.write(f"EMA_SHORT={env_updates['EMA_SHORT']}\n")
        f.write(f"EMA_LONG={env_updates['EMA_LONG']}\n")
        f.write(f"SMA_PERIOD={env_updates['SMA_PERIOD']}\n")
        f.write(f"VOLUME_THRESHOLD={env_updates['VOLUME_THRESHOLD']}\n")
        f.write(f"MOMENTUM_PERIOD={env_updates['MOMENTUM_PERIOD']}\n")
        f.write(f"MAX_POSITION_SIZE={env_updates['MAX_POSITION_SIZE']}\n")
        f.write(f"MIN_CONFIDENCE_THRESHOLD={env_updates['MIN_CONFIDENCE_THRESHOLD']}\n")
        f.write(f"PROFIT_MULTIPLIER={env_updates['PROFIT_MULTIPLIER']}\n")
        f.write(f"AGGRESSIVE_MODE={env_updates['AGGRESSIVE_MODE']}\n")
    
    return jsonify({'success': True, 'config': bot_config})

def get_usdc_balance():
    """Fetch available USDC balance from Coinbase account"""
    try:
        exchange_class = getattr(ccxt, bot_config['exchange_id'])
        exchange_config = {
            'enableRateLimit': True,
            'createMarketBuyOrderRequiresPrice': False,
            'apiKey': bot_config['api_key'],
            'secret': bot_config['secret_key'],
        }
        exchange = exchange_class(exchange_config)
        
        # Fetch balance
        balance = exchange.fetch_balance()
        
        # Get USDC balance
        usdc_balance = balance.get('USDC', {}).get('free', 0)
        
        if usdc_balance > 0:
            logging.info(f"Found USDC balance: ${usdc_balance:.2f}")
            return float(usdc_balance)
        else:
            logging.warning("No USDC balance found, using default starting capital")
            return float(bot_config['starting_capital'])
            
    except Exception as e:
        logging.error(f"Error fetching USDC balance: {e}")
        logging.info("Using default starting capital")
        return float(bot_config['starting_capital'])

@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    global bot_thread, bot_running, strategy_instance
    
    if bot_running:
        return jsonify({'success': False, 'message': 'Bot is already running'})
    
    try:
        if bot_config.get('paper_trading'):
            actual_capital = bot_config['paper_trading_balance']
            bot_config['starting_capital'] = actual_capital
            logging.info(f"Paper trading enabled - using simulated balance: ${actual_capital:.2f}")
        else:
            # Fetch actual USDC balance from account
            actual_capital = get_usdc_balance()
            bot_config['starting_capital'] = actual_capital
            logging.info(f"Using actual USDC balance: ${actual_capital:.2f}")
        
        # Initialize exchange
        exchange_class = getattr(ccxt, bot_config['exchange_id'])
        exchange_config = {
            'enableRateLimit': True,
            'createMarketBuyOrderRequiresPrice': False  # Allow Coinbase to accept cost directly
        }
        
        # Only add API keys if they're provided and not placeholder
        if bot_config.get('api_key') and bot_config['api_key'] != 'your_api_key_here':
            exchange_config['apiKey'] = bot_config['api_key']
            exchange_config['secret'] = bot_config['secret_key']
        else:
            # For paper trading without API keys, use sandbox mode if available
            if hasattr(exchange_class, 'sandbox'):
                exchange_config['sandbox'] = True
        
        exchange = exchange_class(exchange_config)
        exchange.load_markets()
        
        # Initialize strategy with actual capital
        strategy_instance = SimpleRSIStrategy(exchange, bot_config)
        
        bot_running = True
        
        def run_bot():
            global bot_running
            iteration_count = 0
            start_time = time.time()
            max_runtime = bot_config.get('max_runtime_hours', 24) * 3600  # Convert to seconds
            
            while bot_running:
                # Check if max runtime exceeded
                elapsed_time = time.time() - start_time
                if elapsed_time >= max_runtime:
                    logging.info(f"Max runtime of {bot_config.get('max_runtime_hours', 24)} hours reached. Stopping bot.")
                    bot_running = False
                    break
                
                try:
                    iteration_count += 1
                    ticker = exchange.fetch_ticker(bot_config['symbol'])
                    current_price = ticker['last']
                    logging.info(f"Bot loop #{iteration_count}: Fetched price ${current_price:.2f} for {bot_config['symbol']} (Runtime: {elapsed_time/3600:.1f}h/{max_runtime/3600:.0f}h)")
                    strategy_instance.handle_trade_event(current_price)
                    
                    if iteration_count % 5 == 0:
                        save_bot_state()
                    
                    time.sleep(30)
                    logging.info(f"Bot loop #{iteration_count}: Waiting 30 seconds before next iteration...")
                except Exception as e:
                    logging.error(f"Error in bot loop #{iteration_count}: {e}")
                    time.sleep(15)
        
        bot_thread = threading.Thread(target=run_bot, daemon=True)
        bot_thread.start()
        
        return jsonify({'success': True, 'message': 'Bot started successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    global bot_running
    bot_running = False
    return jsonify({'success': True, 'message': 'Bot stopped'})

@app.route('/api/bot/status', methods=['GET'])
def bot_status():
    global bot_running, strategy_instance
    
    if not strategy_instance:
        return jsonify({
            'running': False,
            'capital': bot_config['starting_capital'],
            'trades': 0,
            'position': None,
            'profit_loss': 0,
            'win_rate': 0,
        })
    
    win_rate = 0
    if strategy_instance.trade_history:
        win_rate = len([t for t in strategy_instance.trade_history if t['profit'] > 0]) / len(strategy_instance.trade_history)
    
    # Calculate current position profit percentage for both positions
    long_profit_pct = 0.0
    short_profit_pct = 0.0
    
    if strategy_instance.long_position and strategy_instance.last_buy_price and strategy_instance.price_history:
        current_price = strategy_instance.price_history[-1]
        long_profit_pct = ((current_price - strategy_instance.last_buy_price) / strategy_instance.last_buy_price) * 100
    
    if strategy_instance.short_position and strategy_instance.last_short_price and strategy_instance.price_history:
        current_price = strategy_instance.price_history[-1]
        short_profit_pct = ((strategy_instance.last_short_price - current_price) / strategy_instance.last_short_price) * 100
    
    return jsonify({
        'running': bot_running,
        'capital': strategy_instance.current_capital,
        'trades': strategy_instance.trade_count,
        'long_position': strategy_instance.long_position,
        'short_position': strategy_instance.short_position,
        'long_entry_price': strategy_instance.last_buy_price,
        'short_entry_price': strategy_instance.last_short_price,
        'long_position_size': strategy_instance.long_position_size,
        'short_position_size': strategy_instance.short_position_size,
        'profit_loss': strategy_instance.profit_loss,
        'win_rate': win_rate,
        'consecutive_wins': strategy_instance.consecutive_wins,
        'consecutive_losses': strategy_instance.consecutive_losses,
        'best_trade': strategy_instance.best_trade_profit,
        'worst_trade': strategy_instance.worst_trade_loss,
        'volatility_multiplier': strategy_instance.VOLATILITY_MULTIPLIER,
        'long_profit_pct': long_profit_pct,
        'short_profit_pct': short_profit_pct,
        'ml_enabled': strategy_instance.ml_enabled,
        'ml_only': strategy_instance.ml_only,
        'openai_enabled': strategy_instance.openai_enabled,
        'last_ai_signal': getattr(strategy_instance, 'last_ai_signal', None),
        'last_ai_confidence': getattr(strategy_instance, 'last_ai_confidence', None),
        'enable_short_trading': strategy_instance.enable_short_trading,
    })

@app.route('/api/bot/trades', methods=['GET'])
def get_trades():
    global strategy_instance
    
    if not strategy_instance:
        return jsonify([])
    
    trades = []
    for trade in strategy_instance.trade_history[-20:]:  # Last 20 trades
        trades.append({
            'entry_price': trade.get('buy_price', trade.get('entry_price', 0)),
            'exit_price': trade.get('sell_price', trade.get('exit_price', 0)),
            'profit': trade.get('profit', 0),
            'profit_pct': trade.get('profit_pct', 0),
            'position_size': trade.get('position_size', 0),
        })
    
    return jsonify(trades)

@app.route('/api/bot/manual-sell', methods=['POST'])
def manual_sell():
    global strategy_instance
    
    if not strategy_instance:
        return jsonify({'error': 'Bot not initialized'}), 400
    
    if not strategy_instance.long_position:
        return jsonify({'error': 'No long position to sell'}), 400
    
    try:
        current_price = strategy_instance.price_history[-1] if strategy_instance.price_history else 0
        if current_price == 0:
            return jsonify({'error': 'No price data available'}), 400
        
        strategy_instance.place_sell_order(current_price, 'Manual sell')
        return jsonify({'success': True, 'message': 'Long position closed manually'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/manual-cover', methods=['POST'])
def manual_cover():
    global strategy_instance
    
    if not strategy_instance:
        return jsonify({'error': 'Bot not initialized'}), 400
    
    if not strategy_instance.short_position:
        return jsonify({'error': 'No short position to cover'}), 400
    
    try:
        current_price = strategy_instance.price_history[-1] if strategy_instance.price_history else 0
        if current_price == 0:
            return jsonify({'error': 'No price data available'}), 400
        
        strategy_instance.place_cover_order(current_price, 'Manual cover')
        return jsonify({'success': True, 'message': 'Short position covered manually'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/settings', methods=['GET'])
def get_settings():
    """Get current trading settings"""
    try:
        env_path = '.env'
        settings = {}
        
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line:
                        key, value = line.split('=', 1)
                        settings[key] = value
        
        return jsonify({
            'success': True,
            'symbol': settings.get('SYMBOL', 'BTC/USDT'),
            'volatility_multiplier': int(settings.get('VOLATILITY_MULTIPLIER', '20'))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/settings', methods=['POST'])
def update_settings():
    """Update trading settings (symbol and volatility multiplier)"""
    global strategy_instance
    global bot_running
    
    data = request.json
    new_symbol = data.get('symbol')
    new_volatility = data.get('volatility_multiplier')
    
    if not new_symbol or not new_volatility:
        return jsonify({'error': 'Missing required fields'}), 400
    
    try:
        # Stop bot if running
        if bot_running:
            bot_running = False
        
        # Update .env file
        env_path = '.env'
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                lines = f.readlines()
            
            with open(env_path, 'w') as f:
                symbol_updated = False
                volatility_updated = False
                for line in lines:
                    if line.startswith('SYMBOL='):
                        f.write(f'SYMBOL={new_symbol}\n')
                        symbol_updated = True
                    else:
                        f.write(line)
                
                # Add VOLATILITY_MULTIPLIER if not exists
                if not volatility_updated:
                    f.write(f'VOLATILITY_MULTIPLIER={new_volatility}\n')
        
        # Update strategy volatility multiplier if it exists
        if strategy_instance:
            strategy_instance.VOLATILITY_MULTIPLIER = float(new_volatility)
            strategy_instance.symbol = new_symbol
        
        return jsonify({'success': True, 'message': 'Settings updated'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bot/reset', methods=['POST'])
def reset_bot():
    global strategy_instance
    
    # Reset capital state
    if os.path.exists('capital_state.json'):
        os.remove('capital_state.json')
    
    return jsonify({'success': True, 'message': 'Bot state reset'})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    """Get recent bot logs"""
    try:
        if os.path.exists('bot_logs.log'):
            with open('bot_logs.log', 'r') as f:
                logs = f.readlines()
            # Return last 100 lines
            return jsonify({'logs': logs[-100:]})
        return jsonify({'logs': []})
    except Exception as e:
        return jsonify({'logs': [], 'error': str(e)})

def save_bot_state():
    # This is called periodically to ensure state is saved
    pass

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import threading
import os
import json
from dotenv import load_dotenv
import ccxt
from simple_strategy_v2 import SimpleRSIStrategy
import time
import logging
from logging.handlers import RotatingFileHandler

load_dotenv()

app = Flask(__name__)
CORS(app)

# Global bot state
bot_thread = None
bot_running = False
strategy_instance = None
bot_config = {
    'exchange_id': os.getenv('EXCHANGE_ID', 'binance'),
    'api_key': os.getenv('API_KEY'),
    'secret_key': os.getenv('SECRET_KEY'),
    'symbol': os.getenv('SYMBOL', 'BTC/USDC'),
    'timeframe': os.getenv('TIMEFRAME', '15m'),
    'starting_capital': float(os.getenv('STARTING_CAPITAL', '18')) if os.getenv('STARTING_CAPITAL', '18') != 'auto' else 18,
    'capital_percentage': float(os.getenv('CAPITAL_PERCENTAGE', '90')),
    'risk_percentage': float(os.getenv('RISK_PERCENTAGE', '2.0')),
    'rsi_period': int(os.getenv('RSI_PERIOD', '3')),
    'rsi_overbought': int(os.getenv('RSI_OVERBOUGHT', '65')),
    'rsi_oversold': int(os.getenv('RSI_OVERSOLD', '35')),
    'take_profit_percent': float(os.getenv('TAKE_PROFIT_PERCENT', '5.0')),
    'stop_loss_percent': float(os.getenv('STOP_LOSS_PERCENT', '0.1')),
}

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify(bot_config)

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
            while bot_running:
                try:
                    iteration_count += 1
                    ticker = exchange.fetch_ticker(bot_config['symbol'])
                    current_price = ticker['last']
                    strategy_instance.handle_trade_event(current_price)
                    
                    if iteration_count % 5 == 0:
                        save_bot_state()
                    
                    time.sleep(30)
                except Exception as e:
                    print(f"Error in bot loop: {e}")
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
    
    # Calculate current position profit percentage
    current_position_profit_pct = 0.0
    if strategy_instance.current_position == 'long' and strategy_instance.last_buy_price and strategy_instance.price_history:
        current_price = strategy_instance.price_history[-1]
        current_position_profit_pct = ((current_price - strategy_instance.last_buy_price) / strategy_instance.last_buy_price) * 100
    
    return jsonify({
        'running': bot_running,
        'capital': strategy_instance.current_capital,
        'trades': strategy_instance.trade_count,
        'position': strategy_instance.current_position,
        'profit_loss': strategy_instance.profit_loss,
        'win_rate': win_rate,
        'consecutive_wins': strategy_instance.consecutive_wins,
        'consecutive_losses': strategy_instance.consecutive_losses,
        'best_trade': strategy_instance.best_trade_profit,
        'worst_trade': strategy_instance.worst_trade_loss,
        'volatility_multiplier': strategy_instance.VOLATILITY_MULTIPLIER,
        'current_position_profit_pct': current_position_profit_pct,
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
    
    if strategy_instance.current_position != 'long':
        return jsonify({'error': 'No position to sell'}), 400
    
    try:
        # Get current price
        if not strategy_instance.price_history:
            return jsonify({'error': 'No price data available'}), 400
        
        current_price = strategy_instance.price_history[-1]
        strategy_instance.place_sell_order(current_price, "Manual Sell")
        
        return jsonify({'success': True, 'message': 'Manual sell executed', 'price': current_price})
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

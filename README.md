# Trading Bot with TradingKit

A simple cryptocurrency trading bot using TradingKit and CCXT libraries.

## Features

- RSI-based trading strategy
- Support for multiple exchanges (via CCXT)
- Paper trading mode (no API keys required)
- Configurable trading parameters
- Real-time trade monitoring

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables:
```bash
cp .env.example .env
```

3. Edit `.env` file with your exchange API credentials:
```
EXCHANGE_ID=binance
API_KEY=your_api_key_here
SECRET_KEY=your_secret_key_here
SYMBOL=BTC/USDT
POSITION_SIZE=0.001
```

## Usage

Run the trading bot:
```bash
python trading_bot.py
```

## Strategy

The bot uses a simple RSI (Relative Strength Index) strategy:

- **Buy Signal**: When RSI falls below 30 (oversold condition)
- **Sell Signal**: When RSI rises above 70 (overbought condition)

## Configuration Parameters

- `EXCHANGE_ID`: Exchange to use (binance, coinbase, kraken, etc.)
- `API_KEY`: Your exchange API key
- `SECRET_KEY`: Your exchange API secret
- `SYMBOL`: Trading pair (e.g., BTC/USDT)
- `TIMEFRAME`: Timeframe for analysis
- `POSITION_SIZE`: Size of each trade
- `RISK_PERCENTAGE`: Risk management parameter
- `RSI_PERIOD`: RSI calculation period (default: 14)
- `RSI_OVERBOUGHT`: RSI overbought threshold (default: 70)
- `RSI_OVERSOLD`: RSI oversold threshold (default: 30)

## Paper Trading Mode

If you don't provide API keys, the bot will run in paper trading mode, simulating trades without real money.

## Disclaimer

This bot is for educational purposes only. Trading cryptocurrencies involves significant risk. Always test thoroughly with paper trading before using real money.

## MCP Server

To add the trader-dev MCP server (requires Claude CLI):
```bash
claude mcp add --transport sse --scope user trader-dev https://mcp.trader.dev/sse
```
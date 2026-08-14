import ccxt
import pandas as pd
from datetime import datetime

def scan_profitable_pairs():
    """Scan Binance for most profitable crypto pairs based on volatility, volume, and 24h change"""
    
    print("=== Scanning Binance for Most Profitable Crypto Pairs ===\n")
    
    # Initialize Binance exchange (no API keys needed for public data)
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # Load markets
    exchange.load_markets()
    
    # Major crypto pairs to analyze
    major_pairs = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
        'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT',
        'MATIC/USDT', 'UNI/USDT', 'ATOM/USDT', 'LTC/USDT', 'ETC/USDT',
        'XLM/USDT', 'ALGO/USDT', 'VET/USDT', 'FIL/USDT', 'ICP/USDT',
        'NEAR/USDT', 'AAVE/USDT', 'APE/USDT', 'SAND/USDT', 'MANA/USDT'
    ]
    
    results = []
    
    for pair in major_pairs:
        try:
            if pair not in exchange.markets:
                continue
            
            # Fetch ticker data
            ticker = exchange.fetch_ticker(pair)
            
            # Extract key metrics
            price = ticker['last']
            volume_24h = ticker['baseVolume']  # Base currency volume
            change_24h = ticker['change']  # 24h price change
            change_pct_24h = ticker['percentage']  # 24h percentage change
            high_24h = ticker['high']
            low_24h = ticker['low']
            
            # Calculate volatility (high-low range as percentage of price)
            if price > 0:
                volatility = ((high_24h - low_24h) / price) * 100
            else:
                volatility = 0
            
            # Calculate profitability score (weighted combination of factors)
            # Higher score = more profitable potential
            # We want: high volatility, high volume, positive 24h change
            score = (volatility * 0.4) + (min(volume_24h / 1000000, 10) * 0.3) + (max(change_pct_24h, 0) * 0.3)
            
            results.append({
                'Pair': pair,
                'Price': price,
                '24h Change %': change_pct_24h,
                '24h Volume': volume_24h,
                'Volatility %': volatility,
                'Profit Score': score,
                'High 24h': high_24h,
                'Low 24h': low_24h
            })
            
            print(f"✓ {pair}: ${price:.2f} | 24h: {change_pct_24h:+.2f}% | Vol: {volatility:.2f}% | Vol: ${volume_24h:,.0f}")
            
        except Exception as e:
            print(f"✗ {pair}: Error - {e}")
            continue
    
    # Convert to DataFrame and sort by profit score
    df = pd.DataFrame(results)
    
    if len(df) == 0:
        print("\nNo data retrieved. Check internet connection.")
        return
    
    df = df.sort_values('Profit Score', ascending=False)
    
    print("\n" + "="*80)
    print("TOP 10 MOST PROFITABLE CRYPTO PAIRS")
    print("="*80)
    print(f"{'Rank':<5} {'Pair':<12} {'Price':<12} {'24h Change':<12} {'Volatility':<12} {'Volume':<15} {'Score':<8}")
    print("-"*80)
    
    for i, row in df.head(10).iterrows():
        print(f"{i+1:<5} {row['Pair']:<12} ${row['Price']:<11.2f} {row['24h Change %']:+<11.2f}% {row['Volatility %']:<11.2f}% ${row['24h Volume']:>13,.0f} {row['Profit Score']:<8.2f}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if len(df) > 0:
        top_pair = df.iloc[0]
        print(f"🏆 BEST PAIR: {top_pair['Pair']}")
        print(f"   Price: ${top_pair['Price']:.2f}")
        print(f"   24h Change: {top_pair['24h Change %']:+.2f}%")
        print(f"   Volatility: {top_pair['Volatility %']:.2f}%")
        print(f"   Profit Score: {top_pair['Profit Score']:.2f}")
        print(f"\n   This pair has the highest combination of volatility, volume, and positive momentum.")
    
    print("\n" + "="*80)
    print("TOP 5 BY VOLATILITY (Best for 1% take profit strategy)")
    print("="*80)
    
    df_vol = df.sort_values('Volatility %', ascending=False)
    for i, row in df_vol.head(5).iterrows():
        print(f"{i+1}. {row['Pair']}: {row['Volatility %']:.2f}% volatility | 24h: {row['24h Change %']:+.2f}%")

if __name__ == "__main__":
    scan_profitable_pairs()

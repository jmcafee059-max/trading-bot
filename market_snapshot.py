"""
Market Snapshot Module
Captures a single synchronized market state for both long and short trading decisions
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class MarketSnapshot:
    """
    Single synchronized market snapshot used for both long and short trading decisions.
    Ensures both directions use identical market data, indicators, and timestamps.
    """
    # Basic market data
    timestamp: datetime
    symbol: str
    current_price: float
    bid: float
    ask: float
    spread: float
    spread_pct: float
    
    # OHLCV data
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    # Technical indicators
    rsi: float
    ema_short: float
    ema_long: float
    sma: float
    macd: float
    macd_signal: float
    macd_histogram: float
    atr: float
    momentum: float
    
    # Bollinger Bands
    bb_upper: float
    bb_middle: float
    bb_lower: float
    
    # Market conditions
    btc_weather: Optional[Dict[str, Any]] = None
    relative_strength: Optional[float] = None
    market_regime: str = "NEUTRAL"
    
    # ML signals
    ml_prediction: Optional[float] = None
    ml_confidence: Optional[float] = None
    openai_signal: Optional[str] = None
    openai_confidence: Optional[float] = None
    
    # SOL-specific data (if applicable)
    is_sol: bool = False
    sol_liquidity_ok: bool = True
    sol_spread_ok: bool = True
    sol_momentum_detected: bool = False
    sol_short_momentum_detected: bool = False
    
    # Signal data (calculated from snapshot)
    long_setup_score: float = 0.0
    short_setup_score: float = 0.0
    long_confidence: float = 0.0
    short_confidence: float = 0.0
    long_expected_value: float = 0.0
    short_expected_value: float = 0.0
    
    # Entry/exit data
    long_entry_price: Optional[float] = None
    short_entry_price: Optional[float] = None
    long_tp: Optional[float] = None
    short_tp: Optional[float] = None
    long_sl: Optional[float] = None
    short_sl: Optional[float] = None
    
    # Signal metadata
    signal_timestamp: Optional[datetime] = None
    signal_age: float = 0.0
    
    def is_stale(self, max_age_seconds: int = 30) -> bool:
        """Check if the snapshot is too old to use for trading"""
        if self.signal_timestamp is None:
            return False
        age = (datetime.now() - self.signal_timestamp).total_seconds()
        return age > max_age_seconds
    
    def get_age_seconds(self) -> float:
        """Get the age of the snapshot in seconds"""
        if self.signal_timestamp is None:
            return 0.0
        return (datetime.now() - self.signal_timestamp).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary for logging/dashboard"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'current_price': self.current_price,
            'bid': self.bid,
            'ask': self.ask,
            'spread_pct': self.spread_pct,
            'rsi': self.rsi,
            'ema_short': self.ema_short,
            'ema_long': self.ema_long,
            'macd': self.macd,
            'atr': self.atr,
            'market_regime': self.market_regime,
            'long_setup_score': self.long_setup_score,
            'short_setup_score': self.short_setup_score,
            'long_confidence': self.long_confidence,
            'short_confidence': self.short_confidence,
            'signal_age': self.get_age_seconds(),
        }


@dataclass
class TradingDecision:
    """
    Synchronized trading decision for long and short directions.
    Contains the final decision after evaluating both directions.
    """
    direction: str  # 'LONG', 'SHORT', 'BOTH', 'NONE'
    long_confidence: float = 0.0
    short_confidence: float = 0.0
    long_setup_score: float = 0.0
    short_setup_score: float = 0.0
    long_expected_value: float = 0.0
    short_expected_value: float = 0.0
    long_entry_price: Optional[float] = None
    short_entry_price: Optional[float] = None
    long_tp: Optional[float] = None
    short_tp: Optional[float] = None
    long_sl: Optional[float] = None
    short_sl: Optional[float] = None
    signal_timestamp: Optional[datetime] = None
    reason: str = ""
    
    def is_valid(self) -> bool:
        """Check if the decision is valid for execution"""
        return self.direction in ['LONG', 'SHORT', 'BOTH']
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary for logging/dashboard"""
        return {
            'direction': self.direction,
            'long_confidence': self.long_confidence,
            'short_confidence': self.short_confidence,
            'long_setup_score': self.long_setup_score,
            'short_setup_score': self.short_setup_score,
            'long_expected_value': self.long_expected_value,
            'short_expected_value': self.short_expected_value,
            'reason': self.reason,
        }

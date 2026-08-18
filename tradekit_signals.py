"""
TradeKit Live Signal Store

Receives alert payloads pushed by the real TradeKit (trader.dev) webhook
channel and makes the latest signal per symbol available to the strategy.

TradeKit strategies trade Bybit USDT perpetuals (e.g. BYBIT:ETHUSDT.P) while
this bot trades Coinbase spot (e.g. ETH-USDC). Signals are matched by base
asset (ETH, SOL, ...), not by exact instrument string.

Safety: this store only ever answers "what did TradeKit last say" - it never
places or cancels an order itself. The strategy decides whether/how to use it,
and only does so when TRADEKIT_LIVE_SIGNALS is explicitly enabled.
"""

import re
import threading
import time
from typing import Any, Dict, Optional


def base_asset(symbol: str) -> str:
    """Normalize a symbol like 'BYBIT:ETHUSDT.P', 'ETHUSDT', or 'ETH-USDC' to 'ETH'."""
    s = symbol.upper()
    s = s.split(":")[-1]  # strip exchange prefix e.g. BYBIT:
    s = s.rstrip(".P")
    s = re.sub(r"(USDT|USDC|USD)$", "", s)
    s = s.strip("-_")
    return s


class TradeKitSignalStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._signals: Dict[str, Dict[str, Any]] = {}  # base_asset -> latest signal

    def record_signal(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(payload.get("symbol") or payload.get("ticker") or "")
        asset = base_asset(symbol)

        raw_side = str(payload.get("side") or payload.get("direction") or payload.get("action") or "").lower()
        if "long" in raw_side or "buy" in raw_side:
            direction = "long"
        elif "short" in raw_side or "sell" in raw_side:
            direction = "short"
        else:
            direction = "unknown"

        signal_type = str(payload.get("signalType") or payload.get("type") or "entry").lower()

        entry = {
            "asset": asset,
            "symbol": symbol,
            "direction": direction,
            "signal_type": signal_type,
            "price": payload.get("price"),
            "strategy_id": payload.get("strategyId") or payload.get("strategy_id"),
            "strategy_name": payload.get("strategyName") or payload.get("name"),
            "received_at": time.time(),
            "raw": payload,
        }

        with self._lock:
            self._signals[asset] = entry

        return entry

    def get_latest(self, symbol: str, max_age_seconds: Optional[float] = None) -> Optional[Dict[str, Any]]:
        asset = base_asset(symbol)
        with self._lock:
            entry = self._signals.get(asset)

        if entry is None:
            return None

        if max_age_seconds is not None:
            age = time.time() - entry["received_at"]
            if age > max_age_seconds:
                return None

        return entry

    def all_signals(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return dict(self._signals)


signal_store = TradeKitSignalStore()

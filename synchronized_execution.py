"""
Synchronized Execution Engine
Handles unified long/short trading decisions from a single market snapshot
"""

import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging

from market_snapshot import MarketSnapshot, TradingDecision

logger = logging.getLogger(__name__)


class ExecutionLock:
    """
    Thread-safe execution lock to prevent race conditions.
    Only one execution decision can modify account state at a time.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._locked_by: Optional[str] = None
        self._lock_time: Optional[datetime] = None
    
    def acquire(self, owner: str) -> bool:
        """Acquire the execution lock"""
        if self._lock.acquire(blocking=False):
            self._locked_by = owner
            self._lock_time = datetime.now()
            logger.info(f"Execution lock acquired by {owner}")
            return True
        else:
            logger.warning(f"Execution lock busy, held by {self._locked_by}")
            return False
    
    def release(self, owner: str):
        """Release the execution lock"""
        if self._locked_by == owner:
            self._locked_by = None
            self._lock_time = None
            self._lock.release()
            logger.info(f"Execution lock released by {owner}")
        else:
            logger.warning(f"Attempt to release lock by non-owner {owner}, held by {self._locked_by}")
    
    def is_locked(self) -> bool:
        """Check if lock is currently held"""
        return self._lock.locked()
    
    def get_lock_age_seconds(self) -> float:
        """Get how long the lock has been held"""
        if self._lock_time is None:
            return 0.0
        return (datetime.now() - self._lock_time).total_seconds()


@dataclass
class OrderStatus:
    """Track order status for reconciliation"""
    order_id: str
    direction: str  # 'LONG' or 'SHORT'
    requested_quantity: float
    requested_price: float
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    status: str = "PENDING"  # PENDING, FILLED, PARTIAL, CANCELLED, REJECTED, FAILED
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def is_complete(self) -> bool:
        """Check if order is completely filled"""
        return self.status == "FILLED" and self.filled_quantity >= self.requested_quantity
    
    def is_partial(self) -> bool:
        """Check if order is partially filled"""
        return self.status == "PARTIAL" or (self.filled_quantity > 0 and self.filled_quantity < self.requested_quantity)
    
    def is_failed(self) -> bool:
        """Check if order failed"""
        return self.status in ["CANCELLED", "REJECTED", "FAILED"]


class SynchronizedExecutionEngine:
    """
    Unified execution engine for long and short trading decisions.
    Ensures both directions use the same market snapshot and decision logic.
    """
    
    def __init__(self, strategy_instance, hedged_mode: bool = False, max_signal_age_seconds: int = 30):
        self.strategy = strategy_instance
        self.hedged_mode = hedged_mode
        self.max_signal_age_seconds = max_signal_age_seconds
        
        # Execution lock
        self.execution_lock = ExecutionLock()
        
        # Current state
        self.current_snapshot: Optional[MarketSnapshot] = None
        self.current_decision: Optional[TradingDecision] = None
        self.pending_orders: Dict[str, OrderStatus] = {}
        
        # State machine
        self.state = "IDLE"
        self.state_transitions = {
            "IDLE": ["MARKET_SNAPSHOT"],
            "MARKET_SNAPSHOT": ["CALCULATE_SIGNALS"],
            "CALCULATE_SIGNALS": ["EVALUATE_DECISION"],
            "EVALUATE_DECISION": ["RISK_CHECK", "IDLE"],
            "RISK_CHECK": ["EXECUTION_LOCK", "IDLE"],
            "EXECUTION_LOCK": ["PLACE_ORDER", "IDLE"],
            "PLACE_ORDER": ["RECONCILE", "IDLE"],
            "RECONCILE": ["MONITOR_POSITION", "IDLE"],
            "MONITOR_POSITION": ["POSITION_CLOSED", "IDLE"],
            "POSITION_CLOSED": ["RESET", "IDLE"],
            "RESET": ["IDLE"],
        }
    
    def transition_state(self, new_state: str, reason: str = ""):
        """Transition to a new state in the state machine"""
        if new_state not in self.state_transitions.get(self.state, []):
            logger.warning(f"Invalid state transition: {self.state} -> {new_state}")
            return False
        
        logger.info(f"STATE TRANSITION: {self.state} -> {new_state} | Reason: {reason}")
        self.state = new_state
        return True
    
    def capture_market_snapshot(self) -> Optional[MarketSnapshot]:
        """
        Capture a single synchronized market snapshot.
        This is the only place where market data should be fetched.
        """
        if not self.transition_state("MARKET_SNAPSHOT", "Capturing market snapshot"):
            return None
        
        try:
            # Fetch market data once
            ticker = self.strategy.exchange.fetch_ticker(self.strategy.symbol)
            current_price = ticker.get('last', 0)
            bid = ticker.get('bid', 0)
            ask = ticker.get('ask', 0)
            
            if current_price == 0:
                logger.error("Invalid current price in market snapshot")
                self.transition_state("IDLE", "Invalid price data")
                return None
            
            # Calculate spread
            spread = ask - bid if bid > 0 and ask > 0 else 0
            spread_pct = (spread / bid * 100) if bid > 0 else 0
            
            # Fetch OHLCV data
            ohlcv = self.strategy.exchange.fetch_ohlcv(self.strategy.symbol, self.strategy.timeframe, limit=100)
            if len(ohlcv) < 50:
                logger.warning("Insufficient OHLCV data for indicators")
                self.transition_state("IDLE", "Insufficient data")
                return None
            
            latest_candle = ohlcv[-1]
            open_price, high_price, low_price, close_price, volume = latest_candle[1:6]
            
            # Calculate indicators using the strategy's existing methods
            # This ensures consistency with existing logic
            indicators = self.strategy._calculate_all_indicators(ohlcv)
            
            # Create snapshot
            snapshot = MarketSnapshot(
                timestamp=datetime.now(),
                symbol=self.strategy.symbol,
                current_price=current_price,
                bid=bid,
                ask=ask,
                spread=spread,
                spread_pct=spread_pct,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume,
                rsi=indicators.get('rsi', 50),
                ema_short=indicators.get('ema_short', current_price),
                ema_long=indicators.get('ema_long', current_price),
                sma=indicators.get('sma', current_price),
                macd=indicators.get('macd', 0),
                macd_signal=indicators.get('macd_signal', 0),
                macd_histogram=indicators.get('macd_histogram', 0),
                atr=indicators.get('atr', 0),
                momentum=indicators.get('momentum', 0),
                bb_upper=indicators.get('bb_upper', current_price),
                bb_middle=indicators.get('bb_middle', current_price),
                bb_lower=indicators.get('bb_lower', current_price),
                is_sol=self.strategy.is_sol_strategy,
            )
            
            self.current_snapshot = snapshot
            logger.info(f"Market snapshot captured: {snapshot.symbol} @ ${current_price:.2f}")
            return snapshot
            
        except Exception as e:
            logger.error(f"Failed to capture market snapshot: {e}")
            self.transition_state("IDLE", "Snapshot error")
            return None
    
    def calculate_synchronized_signals(self, snapshot: MarketSnapshot) -> TradingDecision:
        """
        Calculate both long and short signals from the same snapshot.
        Returns a unified trading decision.
        """
        if not self.transition_state("CALCULATE_SIGNALS", "Calculating synchronized signals"):
            return TradingDecision(direction="NONE", reason="State transition failed")
        
        try:
            # Calculate long signal using snapshot data
            long_signal = self.strategy.evaluate_buy_signals_sync(
                snapshot.current_price,
                snapshot.rsi,
                snapshot.ema_short,
                snapshot.ema_long,
                snapshot.macd > snapshot.macd_signal,
                snapshot.close < snapshot.bb_lower,
                snapshot.close > snapshot.bb_upper,
                snapshot.volume,
                snapshot.atr,
                snapshot.momentum
            )
            
            # Calculate short signal using snapshot data
            short_signal = self.strategy.evaluate_short_signals(
                snapshot.current_price,
                snapshot.rsi,
                snapshot.ema_short,
                snapshot.ema_long,
                snapshot.macd > snapshot.macd_signal,
                snapshot.close < snapshot.bb_lower,
                snapshot.close > snapshot.bb_upper,
                snapshot.volume,
                snapshot.atr,
                snapshot.momentum
            )
            
            # Update snapshot with calculated scores
            snapshot.long_setup_score = long_signal.get('setup_score', 0)
            snapshot.short_setup_score = short_signal.get('setup_score', 0)
            snapshot.long_confidence = long_signal.get('confidence', 0)
            snapshot.short_confidence = short_signal.get('confidence', 0)
            snapshot.signal_timestamp = datetime.now()
            
            # Create decision
            decision = self._evaluate_decision(snapshot, long_signal, short_signal)
            self.current_decision = decision
            
            logger.info(f"Synchronized signals calculated: LONG={snapshot.long_setup_score}/SHORT={snapshot.short_setup_score}")
            return decision
            
        except Exception as e:
            logger.error(f"Failed to calculate synchronized signals: {e}")
            self.transition_state("IDLE", "Signal calculation error")
            return TradingDecision(direction="NONE", reason=str(e))
    
    def _evaluate_decision(self, snapshot: MarketSnapshot, long_signal: Dict, short_signal: Dict) -> TradingDecision:
        """
        Evaluate the final trading decision from both long and short signals.
        Implements the logic for choosing between long/short/both/none.
        """
        if not self.transition_state("EVALUATE_DECISION", "Evaluating trading decision"):
            return TradingDecision(direction="NONE", reason="State transition failed")
        
        long_valid = long_signal.get('should_buy', False)
        short_valid = short_signal.get('should_short', False)
        
        # Neither signal valid
        if not long_valid and not short_valid:
            return TradingDecision(
                direction="NONE",
                reason="Neither long nor short signals are valid"
            )
        
        # Only long valid
        if long_valid and not short_valid:
            return TradingDecision(
                direction="LONG",
                long_confidence=snapshot.long_confidence,
                long_setup_score=snapshot.long_setup_score,
                long_expected_value=long_signal.get('expected_value', 0),
                long_entry_price=snapshot.current_price,
                signal_timestamp=snapshot.signal_timestamp,
                reason="Only long signal valid"
            )
        
        # Only short valid
        if short_valid and not long_valid:
            return TradingDecision(
                direction="SHORT",
                short_confidence=snapshot.short_confidence,
                short_setup_score=snapshot.short_setup_score,
                short_expected_value=short_signal.get('expected_value', 0),
                short_entry_price=snapshot.current_price,
                signal_timestamp=snapshot.signal_timestamp,
                reason="Only short signal valid"
            )
        
        # Both valid - choose based on hedged mode
        if self.hedged_mode:
            # Check if combined risk is acceptable
            combined_risk = self._calculate_combined_risk(long_signal, short_signal)
            if combined_risk > self.strategy.max_risk_per_trade:
                # Risk too high, choose stronger signal
                return self._choose_stronger_signal(snapshot, long_signal, short_signal)
            
            return TradingDecision(
                direction="BOTH",
                long_confidence=snapshot.long_confidence,
                short_confidence=snapshot.short_confidence,
                long_setup_score=snapshot.long_setup_score,
                short_setup_score=snapshot.short_setup_score,
                long_expected_value=long_signal.get('expected_value', 0),
                short_expected_value=short_signal.get('expected_value', 0),
                long_entry_price=snapshot.current_price,
                short_entry_price=snapshot.current_price,
                signal_timestamp=snapshot.signal_timestamp,
                reason="Both signals valid, hedged mode enabled"
            )
        else:
            # Choose stronger signal
            return self._choose_stronger_signal(snapshot, long_signal, short_signal)
    
    def _choose_stronger_signal(self, snapshot: MarketSnapshot, long_signal: Dict, short_signal: Dict) -> TradingDecision:
        """Choose the stronger signal between long and short"""
        long_score = snapshot.long_setup_score * snapshot.long_confidence
        short_score = snapshot.short_setup_score * snapshot.short_confidence
        
        if long_score >= short_score:
            return TradingDecision(
                direction="LONG",
                long_confidence=snapshot.long_confidence,
                long_setup_score=snapshot.long_setup_score,
                long_expected_value=long_signal.get('expected_value', 0),
                long_entry_price=snapshot.current_price,
                signal_timestamp=snapshot.signal_timestamp,
                reason=f"Long signal stronger ({long_score:.2f} vs {short_score:.2f})"
            )
        else:
            return TradingDecision(
                direction="SHORT",
                short_confidence=snapshot.short_confidence,
                short_setup_score=snapshot.short_setup_score,
                short_expected_value=short_signal.get('expected_value', 0),
                short_entry_price=snapshot.current_price,
                signal_timestamp=snapshot.signal_timestamp,
                reason=f"Short signal stronger ({short_score:.2f} vs {long_score:.2f})"
            )
    
    def _calculate_combined_risk(self, long_signal: Dict, short_signal: Dict) -> float:
        """Calculate combined risk for hedged positions"""
        # Simplified risk calculation
        long_risk = long_signal.get('risk', 0.5)
        short_risk = short_signal.get('risk', 0.5)
        return (long_risk + short_risk) / 2
    
    def execute_decision(self, decision: TradingDecision) -> bool:
        """
        Execute the trading decision with proper locking and reconciliation.
        """
        if not decision.is_valid():
            logger.info(f"Invalid decision, skipping execution: {decision.direction}")
            self.transition_state("IDLE", "Invalid decision")
            return False
        
        # Check signal age
        if decision.signal_timestamp:
            signal_age = (datetime.now() - decision.signal_timestamp).total_seconds()
            if signal_age > self.max_signal_age_seconds:
                logger.warning(f"Signal too old: {signal_age:.2f}s > {self.max_signal_age_seconds}s")
                self.transition_state("IDLE", "Signal expired")
                return False
        
        # Acquire execution lock
        if not self.execution_lock.acquire("execute_decision"):
            logger.warning("Could not acquire execution lock")
            self.transition_state("IDLE", "Lock busy")
            return False
        
        try:
            if not self.transition_state("EXECUTION_LOCK", "Lock acquired"):
                return False
            
            # Risk check
            if not self.transition_state("RISK_CHECK", "Checking risk"):
                return False
            
            if not self._perform_risk_check(decision):
                logger.warning("Risk check failed")
                self.transition_state("IDLE", "Risk check failed")
                return False
            
            # Place order
            if not self.transition_state("PLACE_ORDER", "Placing order"):
                return False
            
            order_success = self._place_order(decision)
            
            if order_success:
                # Reconcile
                if not self.transition_state("RECONCILE", "Reconciling order"):
                    return False
                
                reconciliation_success = self._reconcile_order(decision)
                
                if reconciliation_success:
                    if not self.transition_state("MONITOR_POSITION", "Monitoring position"):
                        return False
                    return True
                else:
                    logger.warning("Order reconciliation failed")
                    self.transition_state("IDLE", "Reconciliation failed")
                    return False
            else:
                logger.warning("Order placement failed")
                self.transition_state("IDLE", "Order failed")
                return False
                
        finally:
            self.execution_lock.release("execute_decision")
    
    def _perform_risk_check(self, decision: TradingDecision) -> bool:
        """Perform risk checks before execution"""
        # Check balance
        try:
            balance = self.strategy.exchange.fetch_balance()
            usdc_balance = balance.get('USDC', {}).get('free', 0)
            
            if decision.direction in ['LONG', 'BOTH']:
                trade_amount = self.strategy.current_capital * self.strategy.capital_percentage
                if usdc_balance < trade_amount:
                    logger.warning(f"Insufficient USDC balance: ${usdc_balance:.2f} < ${trade_amount:.2f}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Risk check failed: {e}")
            return False
    
    def _place_order(self, decision: TradingDecision) -> bool:
        """Place the order based on decision"""
        try:
            if decision.direction == 'LONG':
                position_size = self.strategy.place_buy_order(decision.long_entry_price)
                return position_size > 0
            elif decision.direction == 'SHORT':
                position_size = self.strategy.place_short_order(decision.short_entry_price)
                return position_size > 0
            elif decision.direction == 'BOTH':
                # Place both orders
                long_size = self.strategy.place_buy_order(decision.long_entry_price)
                short_size = self.strategy.place_short_order(decision.short_entry_price)
                return long_size > 0 and short_size > 0
            
            return False
            
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return False
    
    def _reconcile_order(self, decision: TradingDecision) -> bool:
        """Reconcile order status with exchange"""
        # For now, assume success if order placement succeeded
        # In full implementation, this would query exchange for actual status
        logger.info("Order reconciliation: Assuming success (simplified)")
        return True
    
    def reset_state(self):
        """Reset the execution engine state"""
        self.transition_state("RESET", "Resetting state")
        self.current_snapshot = None
        self.current_decision = None
        self.pending_orders.clear()
        self.transition_state("IDLE", "State reset complete")

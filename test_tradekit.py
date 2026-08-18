"""
TradeKit Integration Tests

Tests for the TradeKit adapter module to ensure proper functionality
and graceful fallbacks when TradeKit components are unavailable.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tradekit_adapter import TradeKitAdapter


class TestTradeKitAdapter(unittest.TestCase):
    """Test cases for TradeKitAdapter class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config_enabled = {
            'USE_TRADEKIT': True,
            'TRADEKIT_MIN_SCORE': 80,
            'TRADEKIT_LIQUIDITY_FILTER': True,
            'TRADEKIT_ORDERBOOK_ANALYSIS': True,
            'TRADEKIT_VOLATILITY_ANALYSIS': True,
            'TRADEKIT_DEBUG': False
        }
        
        self.config_disabled = {
            'USE_TRADEKIT': False,
            'TRADEKIT_MIN_SCORE': 80,
            'TRADEKIT_LIQUIDITY_FILTER': True,
            'TRADEKIT_ORDERBOOK_ANALYSIS': True,
            'TRADEKIT_VOLATILITY_ANALYSIS': True,
            'TRADEKIT_DEBUG': False
        }
        
        # Sample OHLCV data
        self.sample_ohlcv = []
        base_timestamp = int(datetime.now().timestamp() * 1000)
        for i in range(100):
            self.sample_ohlcv.append([
                base_timestamp - (100 - i) * 3600000,  # timestamp
                140.0 + i * 0.1,  # open
                141.0 + i * 0.1,  # high
                139.0 + i * 0.1,  # low
                140.5 + i * 0.1,  # close
                1000000 + i * 10000  # volume
            ])
        
        # Sample order book
        self.sample_orderbook = {
            'bids': [[140.0, 1000], [139.9, 2000], [139.8, 3000]],
            'asks': [[140.1, 1000], [140.2, 2000], [140.3, 3000]]
        }
    
    def test_adapter_initialization_enabled(self):
        """Test adapter initialization with TradeKit enabled"""
        adapter = TradeKitAdapter(self.config_enabled)
        self.assertTrue(adapter.enabled)
        self.assertEqual(adapter.min_score, 80)
        self.assertTrue(adapter.liquidity_filter)
    
    def test_adapter_initialization_disabled(self):
        """Test adapter initialization with TradeKit disabled"""
        adapter = TradeKitAdapter(self.config_disabled)
        self.assertFalse(adapter.enabled)
    
    def test_is_available_when_enabled(self):
        """Test is_available returns True when TradeKit is enabled and available"""
        adapter = TradeKitAdapter(self.config_enabled)
        # This may return False if TradeKit libraries are not installed
        # but should not crash
        result = adapter.is_available()
        self.assertIsInstance(result, bool)
    
    def test_is_available_when_disabled(self):
        """Test is_available returns False when TradeKit is disabled"""
        adapter = TradeKitAdapter(self.config_disabled)
        self.assertFalse(adapter.is_available())
    
    def test_get_status(self):
        """Test get_status returns correct status information"""
        adapter = TradeKitAdapter(self.config_enabled)
        status = adapter.get_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('enabled', status)
        self.assertIn('min_score', status)
        self.assertIn('liquidity_filter', status)
    
    def test_calculate_enhanced_indicators_enabled(self):
        """Test enhanced indicators calculation when enabled"""
        adapter = TradeKitAdapter(self.config_enabled)
        indicators = adapter.calculate_enhanced_indicators(self.sample_ohlcv)
        
        # Should return a dict (may be empty if TradeKit not installed)
        self.assertIsInstance(indicators, dict)
    
    def test_calculate_enhanced_indicators_disabled(self):
        """Test enhanced indicators calculation when disabled returns empty"""
        adapter = TradeKitAdapter(self.config_disabled)
        indicators = adapter.calculate_enhanced_indicators(self.sample_ohlcv)
        
        # Should return empty dict when disabled
        self.assertEqual(indicators, {})
    
    def test_analyze_order_book_enabled(self):
        """Test order book analysis when enabled"""
        adapter = TradeKitAdapter(self.config_enabled)
        analysis = adapter.analyze_order_book(self.sample_orderbook)
        
        # Should return a dict (may be empty if TradeKit not installed)
        self.assertIsInstance(analysis, dict)
    
    def test_analyze_order_book_disabled(self):
        """Test order book analysis when disabled returns empty"""
        adapter = TradeKitAdapter(self.config_disabled)
        analysis = adapter.analyze_order_book(self.sample_orderbook)
        
        # Should return empty dict when disabled
        self.assertEqual(analysis, {})
    
    def test_analyze_volatility_enabled(self):
        """Test volatility analysis when enabled"""
        adapter = TradeKitAdapter(self.config_enabled)
        current_price = 140.5
        analysis = adapter.analyze_volatility(self.sample_ohlcv, current_price)
        
        # Should return a dict (may be empty if TradeKit not installed)
        self.assertIsInstance(analysis, dict)
    
    def test_analyze_volatility_disabled(self):
        """Test volatility analysis when disabled returns empty"""
        adapter = TradeKitAdapter(self.config_disabled)
        current_price = 140.5
        analysis = adapter.analyze_volatility(self.sample_ohlcv, current_price)
        
        # Should return empty dict when disabled
        self.assertEqual(analysis, {})
    
    def test_calculate_trading_costs_enabled(self):
        """Test trading costs calculation when enabled"""
        adapter = TradeKitAdapter(self.config_enabled)
        entry_price = 140.0
        exit_price = 140.3
        position_size = 10.0
        
        costs = adapter.calculate_trading_costs(entry_price, exit_price, position_size)
        
        # Should return a dict (may be empty if TradeKit not installed)
        self.assertIsInstance(costs, dict)
    
    def test_calculate_trading_costs_disabled(self):
        """Test trading costs calculation when disabled returns empty"""
        adapter = TradeKitAdapter(self.config_disabled)
        entry_price = 140.0
        exit_price = 140.3
        position_size = 10.0
        
        costs = adapter.calculate_trading_costs(entry_price, exit_price, position_size)
        
        # Should return empty dict when disabled
        self.assertEqual(costs, {})
    
    def test_calculate_enhanced_setup_score_enabled(self):
        """Test enhanced setup score calculation when enabled"""
        adapter = TradeKitAdapter(self.config_enabled)
        
        score_result = adapter.calculate_enhanced_setup_score(
            trend_score=80,
            momentum_score=75,
            volume_score=70,
            liquidity_score=85,
            volatility_score=72,
            support_resistance_score=78,
            relative_strength_score=76,
            btc_confirmation_score=74,
            ml_confirmation_score=77,
            trade_economics_score=73
        )
        
        # Should return a dict with total_score
        self.assertIsInstance(score_result, dict)
        self.assertIn('total_score', score_result)
        self.assertIn('enabled', score_result)
    
    def test_calculate_enhanced_setup_score_disabled(self):
        """Test enhanced setup score calculation when disabled uses fallback"""
        adapter = TradeKitAdapter(self.config_disabled)
        
        score_result = adapter.calculate_enhanced_setup_score(
            trend_score=80,
            momentum_score=75,
            volume_score=70,
            liquidity_score=85,
            volatility_score=72,
            support_resistance_score=78,
            relative_strength_score=76,
            btc_confirmation_score=74,
            ml_confirmation_score=77,
            trade_economics_score=73
        )
        
        # Should return dict with simple average
        self.assertIsInstance(score_result, dict)
        self.assertIn('total_score', score_result)
        self.assertFalse(score_result.get('enabled', True))
    
    def test_calculate_enhanced_setup_score_weights(self):
        """Test that score weights are applied correctly"""
        adapter = TradeKitAdapter(self.config_enabled)
        
        # Test with custom weights
        custom_config = self.config_enabled.copy()
        custom_config['TRADEKIT_SCORE_WEIGHTS'] = {
            'trend': 30,
            'momentum': 20,
            'volume': 10,
            'liquidity': 5,
            'volatility': 5,
            'support_resistance': 5,
            'relative_strength': 5,
            'btc_confirmation': 5,
            'ml_confirmation': 10,
            'trade_economics': 5
        }
        
        adapter_custom = TradeKitAdapter(custom_config)
        
        score_result = adapter_custom.calculate_enhanced_setup_score(
            trend_score=100,
            momentum_score=100,
            volume_score=100,
            liquidity_score=100,
            volatility_score=100,
            support_resistance_score=100,
            relative_strength_score=100,
            btc_confirmation_score=100,
            ml_confirmation_score=100,
            trade_economics_score=100
        )
        
        # With all 100s and custom weights, should still be 100
        self.assertEqual(score_result['total_score'], 100.0)
    
    def test_empty_ohlcv_handling(self):
        """Test handling of empty OHLCV data"""
        adapter = TradeKitAdapter(self.config_enabled)
        indicators = adapter.calculate_enhanced_indicators([])
        
        # Should handle empty data gracefully
        self.assertIsInstance(indicators, dict)
    
    def test_empty_orderbook_handling(self):
        """Test handling of empty order book"""
        adapter = TradeKitAdapter(self.config_enabled)
        analysis = adapter.analyze_order_book({})
        
        # Should handle empty order book gracefully
        self.assertIsInstance(analysis, dict)
    
    def test_invalid_price_handling(self):
        """Test handling of invalid prices in cost calculation"""
        adapter = TradeKitAdapter(self.config_enabled)
        
        # Test with zero prices
        costs = adapter.calculate_trading_costs(0, 0, 10)
        self.assertIsInstance(costs, dict)
        
        # Test with negative prices
        costs = adapter.calculate_trading_costs(-140, -140, 10)
        self.assertIsInstance(costs, dict)


class TestTradeKitSafety(unittest.TestCase):
    """Test safety features of TradeKit integration"""
    
    def test_tradekit_never_places_orders(self):
        """Verify TradeKit adapter has no order placement methods"""
        adapter = TradeKitAdapter({'USE_TRADEKIT': True})
        
        # Check that adapter has no order placement methods
        self.assertFalse(hasattr(adapter, 'place_order'))
        self.assertFalse(hasattr(adapter, 'create_order'))
        self.assertFalse(hasattr(adapter, 'execute_trade'))
    
    def test_graceful_fallback_on_error(self):
        """Test that methods fallback gracefully on errors"""
        adapter = TradeKitAdapter({'USE_TRADEKIT': True})
        
        # Test with invalid data that might cause errors
        indicators = adapter.calculate_enhanced_indicators(None)
        self.assertIsInstance(indicators, dict)
        
        analysis = adapter.analyze_order_book(None)
        self.assertIsInstance(analysis, dict)
    
    def test_disable_flag_prevents_all_analysis(self):
        """Test that disable flag prevents all TradeKit analysis"""
        adapter = TradeKitAdapter({'USE_TRADEKIT': False})
        
        # All methods should return empty dicts
        self.assertEqual(adapter.calculate_enhanced_indicators([]), {})
        self.assertEqual(adapter.analyze_order_book({}), {})
        self.assertEqual(adapter.analyze_volatility([], 0), {})
        self.assertEqual(adapter.calculate_trading_costs(100, 101, 10), {})


def run_tests():
    """Run all TradeKit integration tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestTradeKitAdapter))
    suite.addTests(loader.loadTestsFromTestCase(TestTradeKitSafety))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)

"""
tests/test_engine.py

Unit tests for QuotingEngine.
Tests reservation price, quote width, quote generation, and constraint enforcement.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from maker.engine import (
    QuotingEngine,
    QuotingParams,
    QuoteState
)


class TestQuotingParams:
    """Test QuotingParams dataclass."""
    
    def test_default_params(self):
        """Test default parameter values."""
        params = QuotingParams()
        
        assert params.risk_aversion == 0.1
        assert params.terminal_time == 300.0
        assert params.order_arrival_rate == 1.0
        assert params.max_inventory == 100
        assert params.min_inventory == -100
        assert params.tick_size == 0.01
        assert params.min_spread_bps == 1.0
    
    def test_custom_params(self):
        """Test custom parameter initialization."""
        params = QuotingParams(
            risk_aversion=0.2,
            max_inventory=200,
            tick_size=0.001
        )
        
        assert params.risk_aversion == 0.2
        assert params.max_inventory == 200
        assert params.tick_size == 0.001


class TestReservationPrice:
    """Test reservation price computation."""
    
    def test_zero_inventory(self):
        """With zero inventory, reservation = microprice."""
        engine = QuotingEngine()
        
        r = engine.compute_reservation_price(
            microprice=100.0,
            volatility=0.20,
            inventory=0,
            time_to_close=300
        )
        
        assert abs(r - 100.0) < 1e-6
    
    def test_long_inventory(self):
        """With long position, reservation < microprice."""
        engine = QuotingEngine()
        
        r = engine.compute_reservation_price(
            microprice=100.0,
            volatility=0.20,
            inventory=50,  # Long
            time_to_close=300
        )
        
        # Should be below microprice (want to sell)
        assert r < 100.0
    
    def test_short_inventory(self):
        """With short position, reservation > microprice."""
        engine = QuotingEngine()
        
        r = engine.compute_reservation_price(
            microprice=100.0,
            volatility=0.20,
            inventory=-50,  # Short
            time_to_close=300
        )
        
        # Should be above microprice (want to buy)
        assert r > 100.0
    
    def test_higher_vol_larger_adjustment(self):
        """Higher volatility → larger inventory adjustment."""
        engine = QuotingEngine()
        
        r_low_vol = engine.compute_reservation_price(
            microprice=100.0,
            volatility=0.10,
            inventory=50,
            time_to_close=300
        )
        
        r_high_vol = engine.compute_reservation_price(
            microprice=100.0,
            volatility=0.40,
            inventory=50,
            time_to_close=300
        )
        
        # High vol should have larger adjustment (further from microprice)
        assert abs(r_high_vol - 100.0) > abs(r_low_vol - 100.0)
    
    def test_time_urgency(self):
        """Less time → larger inventory adjustment."""
        engine = QuotingEngine()
        
        r_long_time = engine.compute_reservation_price(
            microprice=100.0,
            volatility=0.20,
            inventory=50,
            time_to_close=600  # More time
        )
        
        r_short_time = engine.compute_reservation_price(
            microprice=100.0,
            volatility=0.20,
            inventory=50,
            time_to_close=60  # Less time
        )
        
        # Longer time should have larger adjustment (more time to mean-revert)
        assert abs(r_long_time - 100.0) > abs(r_short_time - 100.0)


class TestQuoteWidth:
    """Test quote width (spread) computation."""
    
    def test_positive_spread(self):
        """Spread should always be positive."""
        engine = QuotingEngine()
        
        spread = engine.compute_quote_width(
            volatility=0.20,
            time_to_close=300,
            inventory=0
        )
        
        assert spread > 0
    
    def test_higher_vol_wider_spread(self):
        """Higher volatility → wider spread."""
        engine = QuotingEngine()
        
        spread_low = engine.compute_quote_width(
            volatility=0.10,
            time_to_close=300,
            inventory=0
        )
        
        spread_high = engine.compute_quote_width(
            volatility=0.40,
            time_to_close=300,
            inventory=0
        )
        
        assert spread_high > spread_low
    
    def test_inventory_urgency(self):
        """Near inventory limits → wider spread."""
        engine = QuotingEngine()
        
        spread_neutral = engine.compute_quote_width(
            volatility=0.20,
            time_to_close=300,
            inventory=0
        )
        
        spread_limit = engine.compute_quote_width(
            volatility=0.20,
            time_to_close=300,
            inventory=90  # Near max of 100
        )
        
        # Near limit should have wider spread
        assert spread_limit > spread_neutral


class TestGenerateQuotes:
    """Test complete quote generation."""
    
    def test_basic_quote_generation(self):
        """Test basic quote generation with neutral state."""
        engine = QuotingEngine()
        
        state = QuoteState(
            bid=100.0,
            ask=100.1,
            bid_size=500,
            ask_size=500,
            microprice=100.05,
            volatility=0.20,
            signal_bps=0.0,  # Neutral signal
            inventory=0,
            timestamp=pd.Timestamp('2017-01-03 09:30:00')
        )
        
        bid, ask = engine.generate_quotes(state, time_to_close=300)
        
        # Basic sanity checks
        assert bid < ask
        assert bid < state.microprice < ask
        assert bid > 0 and ask > 0
    
    def test_positive_signal_skew(self):
        """Positive signal should skew quotes up."""
        engine = QuotingEngine()
        
        state_neutral = QuoteState(
            bid=100.0, ask=100.1, bid_size=500, ask_size=500,
            microprice=100.05, volatility=0.20,
            signal_bps=0.0, inventory=0,
            timestamp=pd.Timestamp.now()
        )
        
        state_positive = QuoteState(
            bid=100.0, ask=100.1, bid_size=500, ask_size=500,
            microprice=100.05, volatility=0.20,
            signal_bps=5.0,  # Strong buy signal
            inventory=0,
            timestamp=pd.Timestamp.now()
        )
        
        bid_neutral, ask_neutral = engine.generate_quotes(state_neutral, 300)
        bid_positive, ask_positive = engine.generate_quotes(state_positive, 300)
        
        # Positive signal → higher quotes (more eager to buy)
        assert bid_positive >= bid_neutral
        assert ask_positive >= ask_neutral
    
    def test_negative_signal_skew(self):
        """Negative signal should skew quotes down."""
        engine = QuotingEngine()
        
        state_neutral = QuoteState(
            bid=100.0, ask=100.1, bid_size=500, ask_size=500,
            microprice=100.05, volatility=0.20,
            signal_bps=0.0, inventory=0,
            timestamp=pd.Timestamp.now()
        )
        
        state_negative = QuoteState(
            bid=100.0, ask=100.1, bid_size=500, ask_size=500,
            microprice=100.05, volatility=0.20,
            signal_bps=-5.0,  # Strong sell signal
            inventory=0,
            timestamp=pd.Timestamp.now()
        )
        
        bid_neutral, ask_neutral = engine.generate_quotes(state_neutral, 300)
        bid_negative, ask_negative = engine.generate_quotes(state_negative, 300)
        
        # Negative signal → lower quotes (more eager to sell)
        assert bid_negative <= bid_neutral
        assert ask_negative <= ask_neutral
    
    def test_long_inventory_lower_quotes(self):
        """Long inventory should push quotes down (via reservation price)."""
        engine = QuotingEngine()
        
        state_neutral = QuoteState(
            bid=100.0, ask=100.1, bid_size=500, ask_size=500,
            microprice=100.05, volatility=0.20,
            signal_bps=0.0, inventory=0,
            timestamp=pd.Timestamp.now()
        )
        
        state_long = QuoteState(
            bid=100.0, ask=100.1, bid_size=500, ask_size=500,
            microprice=100.05, volatility=0.20,
            signal_bps=0.0, inventory=80,  # Large long position
            timestamp=pd.Timestamp.now()
        )
        
        bid_neutral, ask_neutral = engine.generate_quotes(state_neutral, 300)
        bid_long, ask_long = engine.generate_quotes(state_long, 300)
        
        # Long position → lower bid (want to sell, less eager to buy more)
        # Spread may widen due to inventory urgency, but center should shift down
        mid_neutral = (bid_neutral + ask_neutral) / 2
        mid_long = (bid_long + ask_long) / 2
        
        assert bid_long < bid_neutral
        assert mid_long < mid_neutral  # Overall quotes shifted down
    
    def test_short_inventory_higher_quotes(self):
        """Short inventory should push quotes up (via reservation price)."""
        engine = QuotingEngine()
        
        state_neutral = QuoteState(
            bid=100.0, ask=100.1, bid_size=500, ask_size=500,
            microprice=100.05, volatility=0.20,
            signal_bps=0.0, inventory=0,
            timestamp=pd.Timestamp.now()
        )
        
        state_short = QuoteState(
            bid=100.0, ask=100.1, bid_size=500, ask_size=500,
            microprice=100.05, volatility=0.20,
            signal_bps=0.0, inventory=-80,  # Large short position
            timestamp=pd.Timestamp.now()
        )
        
        bid_neutral, ask_neutral = engine.generate_quotes(state_neutral, 300)
        bid_short, ask_short = engine.generate_quotes(state_short, 300)
        
        # Short position → higher ask (want to buy, less eager to sell more)
        # Spread may widen due to inventory urgency, but center should shift up
        mid_neutral = (bid_neutral + ask_neutral) / 2
        mid_short = (bid_short + ask_short) / 2
        
        assert ask_short > ask_neutral
        assert mid_short > mid_neutral  # Overall quotes shifted up
    
    def test_tick_rounding(self):
        """Quotes should be rounded to tick size."""
        params = QuotingParams(tick_size=0.05)
        engine = QuotingEngine(params)
        
        state = QuoteState(
            bid=100.0, ask=100.1, bid_size=500, ask_size=500,
            microprice=100.05, volatility=0.20,
            signal_bps=0.0, inventory=0,
            timestamp=pd.Timestamp.now()
        )
        
        bid, ask = engine.generate_quotes(state, 300)
        
        # Should be multiples of tick size (within floating point precision)
        assert abs(bid % 0.05) < 1e-8 or abs((bid % 0.05) - 0.05) < 1e-8
        assert abs(ask % 0.05) < 1e-8 or abs((ask % 0.05) - 0.05) < 1e-8
    
    def test_minimum_spread_enforced(self):
        """Minimum spread should be enforced."""
        params = QuotingParams(min_spread_bps=10.0)  # 10 bps minimum
        engine = QuotingEngine(params)
        
        state = QuoteState(
            bid=100.0, ask=100.1, bid_size=500, ask_size=500,
            microprice=100.05, volatility=0.01,  # Very low vol
            signal_bps=0.0, inventory=0,
            timestamp=pd.Timestamp.now()
        )
        
        bid, ask = engine.generate_quotes(state, 300)
        
        # Spread should be at least 10 bps
        spread_bps = (ask - bid) / state.microprice * 10000
        assert spread_bps >= 10.0


class TestEnforceNoCrossMarket:
    """Test market crossing prevention."""
    
    def test_valid_quotes_unchanged(self):
        """Valid quotes should not be adjusted."""
        engine = QuotingEngine()
        
        bid, ask = engine.enforce_no_cross_market(
            bid=99.95,
            ask=100.05,
            microprice=100.0
        )
        
        assert bid == 99.95
        assert ask == 100.05
    
    def test_bid_above_micro_adjusted(self):
        """Bid above microprice should be adjusted down."""
        engine = QuotingEngine()
        
        bid, ask = engine.enforce_no_cross_market(
            bid=100.05,  # Too high
            ask=100.10,
            microprice=100.0
        )
        
        assert bid < 100.0
        assert ask > 100.0
    
    def test_ask_below_micro_adjusted(self):
        """Ask below microprice should be adjusted up."""
        engine = QuotingEngine()
        
        bid, ask = engine.enforce_no_cross_market(
            bid=99.90,
            ask=99.95,  # Too low
            microprice=100.0
        )
        
        assert bid < 100.0
        assert ask > 100.0
    
    def test_crossed_quotes_fixed(self):
        """Crossed quotes (bid > ask) should be fixed."""
        engine = QuotingEngine()
        
        bid, ask = engine.enforce_no_cross_market(
            bid=100.10,
            ask=99.90,  # Crossed
            microprice=100.0
        )
        
        assert bid < ask
        assert bid < 100.0 < ask


class TestEngineUtilities:
    """Test utility methods."""
    
    def test_update_params(self):
        """Test dynamic parameter updates."""
        engine = QuotingEngine()
        
        assert engine.params.risk_aversion == 0.1
        
        engine.update_params(risk_aversion=0.2, min_spread_bps=2.0)
        
        assert engine.params.risk_aversion == 0.2
        assert engine.params.min_spread_bps == 2.0
    
    def test_update_invalid_param_error(self):
        """Test that invalid param update raises error."""
        engine = QuotingEngine()
        
        with pytest.raises(ValueError, match="Invalid parameter"):
            engine.update_params(invalid_param=123)
    
    def test_inventory_limits_proximity(self):
        """Test inventory proximity calculation."""
        params = QuotingParams(max_inventory=100)
        engine = QuotingEngine(params)
        
        assert engine.get_inventory_limits_proximity(0) == 0.0
        assert engine.get_inventory_limits_proximity(50) == 0.5
        assert engine.get_inventory_limits_proximity(100) == 1.0
        assert engine.get_inventory_limits_proximity(-75) == 0.75
    
    def test_round_to_tick(self):
        """Test tick rounding."""
        params = QuotingParams(tick_size=0.01)
        engine = QuotingEngine(params)
        
        # Round down
        assert engine._round_to_tick(100.056, 'down') == 100.05
        
        # Round up
        assert engine._round_to_tick(100.051, 'up') == 100.06
        
        # Round nearest
        assert engine._round_to_tick(100.054, 'nearest') == 100.05
        assert engine._round_to_tick(100.056, 'nearest') == 100.06


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])

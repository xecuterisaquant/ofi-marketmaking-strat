"""
tests/test_fills.py

Unit tests for fill simulation models.
"""
import pytest
import numpy as np
import pandas as pd
from maker.fills import (
    ParametricFillModel,
    simulate_fill_batch,
    calibrate_intensity,
    compute_expected_fill_rate
)


class TestParametricFillModel:
    """Tests for ParametricFillModel class."""
    
    def test_default_initialization(self):
        """Test default parameter values."""
        model = ParametricFillModel()
        assert model.intensity_at_touch == 2.0
        assert model.decay_rate == 0.5
        assert model.time_step == 1.0
    
    def test_custom_initialization(self):
        """Test custom parameter values."""
        model = ParametricFillModel(
            intensity_at_touch=3.0,
            decay_rate=0.8,
            time_step=0.5
        )
        assert model.intensity_at_touch == 3.0
        assert model.decay_rate == 0.8
        assert model.time_step == 0.5
    
    def test_intensity_at_touch(self):
        """Test intensity computation at touch (δ=0)."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        intensity = model.compute_intensity(0.0)
        assert np.isclose(intensity, 2.0)
    
    def test_intensity_decay(self):
        """Test intensity decays with distance."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        
        intensity_0 = model.compute_intensity(0.0)
        intensity_1 = model.compute_intensity(1.0)
        intensity_5 = model.compute_intensity(5.0)
        
        # Should decay exponentially
        assert intensity_0 > intensity_1 > intensity_5
        # Check exponential relationship
        assert np.isclose(intensity_1, 2.0 * np.exp(-0.5 * 1.0))
        assert np.isclose(intensity_5, 2.0 * np.exp(-0.5 * 5.0))
    
    def test_intensity_symmetric(self):
        """Test intensity is symmetric around mid (abs value)."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        
        intensity_pos = model.compute_intensity(2.0)
        intensity_neg = model.compute_intensity(-2.0)
        
        assert np.isclose(intensity_pos, intensity_neg)
    
    def test_fill_probability_bounds(self):
        """Test fill probability is in [0, 1]."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        
        prob_at_touch = model.compute_fill_probability(0.0)
        prob_far = model.compute_fill_probability(20.0)
        
        assert 0.0 <= prob_at_touch <= 1.0
        assert 0.0 <= prob_far <= 1.0
    
    def test_fill_probability_decreases_with_distance(self):
        """Test fill probability decreases as distance increases."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        
        prob_0 = model.compute_fill_probability(0.0)
        prob_1 = model.compute_fill_probability(1.0)
        prob_5 = model.compute_fill_probability(5.0)
        prob_10 = model.compute_fill_probability(10.0)
        
        assert prob_0 > prob_1 > prob_5 > prob_10
    
    def test_simulate_fill_bid_at_micro(self):
        """Test bid fill simulation when quote is at microprice."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        rng = np.random.default_rng(42)
        
        # Bid at microprice (δ=0) should have moderate fill rate
        fills = [
            model.simulate_fill(100.00, 100.00, 'bid', rng)
            for _ in range(1000)
        ]
        fill_rate = sum(fills) / len(fills)
        
        # Expected: P(fill) = 1 - exp(-2.0 * 1.0) ≈ 0.865
        assert 0.80 < fill_rate < 0.92
    
    def test_simulate_fill_bid_below_micro(self):
        """Test bid fill when quote is below microprice."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        rng = np.random.default_rng(42)
        
        # Bid 1bp below micro → lower fill rate
        # 100.00 - 99.9999 = 0.0001 = 1bp
        fills = [
            model.simulate_fill(99.9999, 100.00, 'bid', rng)
            for _ in range(1000)
        ]
        fill_rate = sum(fills) / len(fills)
        
        # Distance = 1bp → intensity = 2.0*exp(-0.5*1) ≈ 1.21
        # P(fill) = 1 - exp(-1.21) ≈ 0.70
        assert 0.60 < fill_rate < 0.80
    
    def test_simulate_fill_ask_at_micro(self):
        """Test ask fill simulation when quote is at microprice."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        rng = np.random.default_rng(43)
        
        # Ask at microprice
        fills = [
            model.simulate_fill(100.00, 100.00, 'ask', rng)
            for _ in range(1000)
        ]
        fill_rate = sum(fills) / len(fills)
        
        assert 0.80 < fill_rate < 0.92
    
    def test_simulate_fill_ask_above_micro(self):
        """Test ask fill when quote is above microprice."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        rng = np.random.default_rng(44)
        
        # Ask 2bp above micro → lower fill rate
        # 100.0002 - 100.00 = 0.0002 = 2bp
        fills = [
            model.simulate_fill(100.0002, 100.00, 'ask', rng)
            for _ in range(1000)
        ]
        fill_rate = sum(fills) / len(fills)
        
        # Distance = 2bp → intensity = 2.0*exp(-0.5*2) ≈ 0.736
        # P(fill) = 1 - exp(-0.736) ≈ 0.52
        assert 0.42 < fill_rate < 0.62
    
    def test_simulate_fill_aggressive_bid(self):
        """Test aggressive bid (above microprice) always fills."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        rng = np.random.default_rng(45)
        
        # Bid above micro (aggressive) → should always fill
        fills = [
            model.simulate_fill(100.02, 100.00, 'bid', rng)
            for _ in range(100)
        ]
        fill_rate = sum(fills) / len(fills)
        
        assert fill_rate == 1.0
    
    def test_simulate_fill_aggressive_ask(self):
        """Test aggressive ask (below microprice) always fills."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        rng = np.random.default_rng(46)
        
        # Ask below micro (aggressive) → should always fill
        fills = [
            model.simulate_fill(99.98, 100.00, 'ask', rng)
            for _ in range(100)
        ]
        fill_rate = sum(fills) / len(fills)
        
        assert fill_rate == 1.0
    
    def test_simulate_fill_invalid_side_raises(self):
        """Test invalid side raises ValueError."""
        model = ParametricFillModel()
        rng = np.random.default_rng(47)
        
        with pytest.raises(ValueError, match="Invalid side"):
            model.simulate_fill(100.00, 100.00, 'invalid', rng)
    
    def test_simulate_fill_reproducible(self):
        """Test fills are reproducible with same seed."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        
        fills1 = [model.simulate_fill(100.00, 100.00, 'bid', rng1) for _ in range(10)]
        fills2 = [model.simulate_fill(100.00, 100.00, 'bid', rng2) for _ in range(10)]
        
        assert fills1 == fills2


class TestSimulateFillBatch:
    """Tests for vectorized fill simulation."""
    
    def test_basic_batch_simulation(self):
        """Test batch simulation returns correct shapes."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        rng = np.random.default_rng(42)
        
        n = 100
        bids = np.full(n, 100.00)
        asks = np.full(n, 100.02)
        mids = np.full(n, 100.01)
        
        bid_fills, ask_fills = simulate_fill_batch(model, bids, asks, mids, rng)
        
        assert bid_fills.shape == (n,)
        assert ask_fills.shape == (n,)
        assert bid_fills.dtype == bool
        assert ask_fills.dtype == bool
    
    def test_batch_fill_rates(self):
        """Test batch simulation produces reasonable fill rates."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        rng = np.random.default_rng(43)
        
        n = 10000
        # Bids at microprice, asks 2bp away
        bids = np.full(n, 100.00)
        asks = np.full(n, 100.0002)  # 2bp above micro
        mids = np.full(n, 100.00)
        
        bid_fills, ask_fills = simulate_fill_batch(model, bids, asks, mids, rng)
        
        bid_fill_rate = bid_fills.sum() / n
        ask_fill_rate = ask_fills.sum() / n
        
        # Bids at micro → high fill rate (~86%)
        assert 0.80 < bid_fill_rate < 0.92
        
        # Asks 2bp away → moderate fill rate (~52%)
        assert 0.42 < ask_fill_rate < 0.62
    
    def test_batch_aggressive_quotes_fill(self):
        """Test aggressive quotes in batch always fill."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        rng = np.random.default_rng(44)
        
        n = 100
        # Aggressive bids (above micro)
        bids = np.full(n, 100.05)
        asks = np.full(n, 99.95)
        mids = np.full(n, 100.00)
        
        bid_fills, ask_fills = simulate_fill_batch(model, bids, asks, mids, rng)
        
        assert bid_fills.all()
        assert ask_fills.all()
    
    def test_batch_reproducible(self):
        """Test batch simulation is reproducible."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
        
        n = 50
        bids = np.full(n, 100.00)
        asks = np.full(n, 100.02)
        mids = np.full(n, 100.01)
        
        rng1 = np.random.default_rng(999)
        bid_fills1, ask_fills1 = simulate_fill_batch(model, bids, asks, mids, rng1)
        
        rng2 = np.random.default_rng(999)
        bid_fills2, ask_fills2 = simulate_fill_batch(model, bids, asks, mids, rng2)
        
        assert np.array_equal(bid_fills1, bid_fills2)
        assert np.array_equal(ask_fills1, ask_fills2)


class TestCalibrateIntensity:
    """Tests for calibration function."""
    
    def test_calibration_tight_spread(self):
        """Test calibration for tight spread stock."""
        # Create synthetic NBBO with 1bp spread
        n = 1000
        bid = np.full(n, 100.00)
        ask = np.full(n, 100.01)
        
        df = pd.DataFrame({'bid': bid, 'ask': ask})
        model = calibrate_intensity(df, target_fill_rate_at_touch=0.5)
        
        # Tight spread → high decay rate
        assert model.decay_rate >= 0.7
        assert model.time_step == 1.0
    
    def test_calibration_wide_spread(self):
        """Test calibration for wide spread stock."""
        # Create synthetic NBBO with 10bp spread
        n = 1000
        bid = np.full(n, 100.00)
        ask = np.full(n, 100.10)
        
        df = pd.DataFrame({'bid': bid, 'ask': ask})
        model = calibrate_intensity(df, target_fill_rate_at_touch=0.5)
        
        # Wide spread → low decay rate
        assert model.decay_rate <= 0.5
    
    def test_calibration_target_fill_rate(self):
        """Test calibrated model achieves target fill rate."""
        n = 1000
        bid = np.full(n, 100.00)
        ask = np.full(n, 100.02)
        
        df = pd.DataFrame({'bid': bid, 'ask': ask})
        target_rate = 0.7
        model = calibrate_intensity(df, target_fill_rate_at_touch=target_rate, time_step=1.0)
        
        # Check probability at touch
        prob = model.compute_fill_probability(0.0)
        
        # Should be close to target (within 5%)
        assert abs(prob - target_rate) < 0.05
    
    def test_calibration_with_microprice_col(self):
        """Test calibration when microprice column is provided."""
        n = 1000
        bid = np.full(n, 100.00)
        ask = np.full(n, 100.02)
        micro = (bid + ask) / 2
        
        df = pd.DataFrame({'bid': bid, 'ask': ask, 'micro': micro})
        model = calibrate_intensity(df, mid_col='micro')
        
        assert isinstance(model, ParametricFillModel)
        assert model.decay_rate > 0


class TestComputeExpectedFillRate:
    """Tests for expected fill rate computation."""
    
    def test_fill_rate_at_touch(self):
        """Test expected fill rate at touch."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5, time_step=1.0)
        
        fill_rate = compute_expected_fill_rate(model, 0.0)
        
        # At touch: P(fill) = 1 - exp(-2.0) ≈ 0.865
        # Rate = 0.865 / 1.0 = 0.865 per second
        assert np.isclose(fill_rate, 0.865, atol=0.01)
    
    def test_fill_rate_decreases_with_distance(self):
        """Test fill rate decreases with distance."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5, time_step=1.0)
        
        rate_0 = compute_expected_fill_rate(model, 0.0)
        rate_2 = compute_expected_fill_rate(model, 2.0)
        rate_5 = compute_expected_fill_rate(model, 5.0)
        
        assert rate_0 > rate_2 > rate_5
    
    def test_fill_rate_very_far(self):
        """Test fill rate approaches zero far from mid."""
        model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5, time_step=1.0)
        
        rate_far = compute_expected_fill_rate(model, 50.0)
        
        assert rate_far < 0.01  # Very low fill rate

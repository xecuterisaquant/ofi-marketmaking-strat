"""
tests/test_features.py

Unit tests for feature engineering functions.
Tests compute_ofi_signal, compute_microprice, compute_ewma_volatility,
compute_imbalance, and blend_signals with synthetic data.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from maker.features import (
    compute_ofi_signal,
    compute_microprice,
    compute_ewma_volatility,
    compute_imbalance,
    blend_signals,
    compute_signal_stats
)


class TestComputeOFISignal:
    """Test OFI signal computation."""
    
    def test_basic_conversion(self):
        """Test basic OFI to bps conversion."""
        ofi = pd.Series([1.0, -1.0, 0.0, 0.5])
        beta = 0.036
        
        signal = compute_ofi_signal(ofi, beta=beta)
        
        # 1.0 * 0.036 * 100 = 3.6 bps
        expected = pd.Series([3.6, -3.6, 0.0, 1.8])
        pd.testing.assert_series_equal(signal, expected)
    
    def test_custom_beta(self):
        """Test with custom beta value."""
        ofi = pd.Series([2.0, -1.5])
        beta = 0.05  # Higher beta
        
        signal = compute_ofi_signal(ofi, beta=beta)
        
        expected = pd.Series([10.0, -7.5])
        pd.testing.assert_series_equal(signal, expected)
    
    def test_zero_beta(self):
        """Test with zero beta (no signal)."""
        ofi = pd.Series([1.0, 2.0, 3.0])
        
        signal = compute_ofi_signal(ofi, beta=0.0)
        
        expected = pd.Series([0.0, 0.0, 0.0])
        pd.testing.assert_series_equal(signal, expected)
    
    def test_preserves_index(self):
        """Test that signal preserves input index."""
        index = pd.date_range('2017-01-03 09:30:00', periods=5, freq='1s')
        ofi = pd.Series([0.5, -0.3, 0.1, 0.0, 0.8], index=index)
        
        signal = compute_ofi_signal(ofi)
        
        assert signal.index.equals(index)


class TestComputeMicroprice:
    """Test microprice computation."""
    
    def test_balanced_book(self):
        """Test microprice with equal bid/ask sizes."""
        bid = pd.Series([100.0, 100.5])
        ask = pd.Series([100.1, 100.6])
        bid_sz = pd.Series([500.0, 500.0])
        ask_sz = pd.Series([500.0, 500.0])
        
        mp = compute_microprice(bid, ask, bid_sz, ask_sz)
        
        # With equal sizes, microprice = midprice
        midprice = (bid + ask) / 2.0
        pd.testing.assert_series_equal(mp, midprice)
    
    def test_bid_heavy(self):
        """Test microprice when bid size >> ask size."""
        bid = pd.Series([100.0])
        ask = pd.Series([100.1])
        bid_sz = pd.Series([1000.0])  # Large bid
        ask_sz = pd.Series([100.0])   # Small ask
        
        mp = compute_microprice(bid, ask, bid_sz, ask_sz)
        
        # Expected: (100.1 * 1000 + 100.0 * 100) / 1100 = 100.09090...
        expected = (100.1 * 1000 + 100.0 * 100) / 1100
        assert abs(mp.iloc[0] - expected) < 1e-6
        
        # Should be closer to ask (upward pressure)
        midprice = 100.05
        assert mp.iloc[0] > midprice
    
    def test_ask_heavy(self):
        """Test microprice when ask size >> bid size."""
        bid = pd.Series([100.0])
        ask = pd.Series([100.1])
        bid_sz = pd.Series([100.0])   # Small bid
        ask_sz = pd.Series([1000.0])  # Large ask
        
        mp = compute_microprice(bid, ask, bid_sz, ask_sz)
        
        # Expected: (100.1 * 100 + 100.0 * 1000) / 1100 = 100.00909...
        expected = (100.1 * 100 + 100.0 * 1000) / 1100
        assert abs(mp.iloc[0] - expected) < 1e-6
        
        # Should be closer to bid (downward pressure)
        midprice = 100.05
        assert mp.iloc[0] < midprice
    
    def test_zero_depth(self):
        """Test fallback to midprice when depth is zero."""
        bid = pd.Series([100.0, 100.5])
        ask = pd.Series([100.1, 100.6])
        bid_sz = pd.Series([0.0, 500.0])
        ask_sz = pd.Series([0.0, 500.0])
        
        mp = compute_microprice(bid, ask, bid_sz, ask_sz)
        
        # First should be midprice (zero depth)
        assert mp.iloc[0] == 100.05
        # Second should also be midprice (equal depth)
        assert mp.iloc[1] == 100.55
    
    def test_numpy_arrays(self):
        """Test that function works with numpy arrays."""
        bid = np.array([100.0, 100.5])
        ask = np.array([100.1, 100.6])
        bid_sz = np.array([500.0, 500.0])
        ask_sz = np.array([500.0, 500.0])
        
        mp = compute_microprice(bid, ask, bid_sz, ask_sz)
        
        assert isinstance(mp, pd.Series)
        assert len(mp) == 2


class TestComputeEWMAVolatility:
    """Test EWMA volatility estimation."""
    
    def test_constant_price(self):
        """Test volatility of constant prices (should be ~zero)."""
        prices = pd.Series([100.0] * 100)
        
        vol = compute_ewma_volatility(prices, halflife_seconds=10, min_periods=5)
        
        # Should be zero or very close (after min_periods)
        assert vol.iloc[-1] < 1e-6
    
    def test_increasing_volatility(self):
        """Test that volatility increases with price variability."""
        np.random.seed(42)
        
        # Low volatility series
        prices_low = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.01))
        vol_low = compute_ewma_volatility(prices_low, halflife_seconds=20, min_periods=10)
        
        # High volatility series  
        prices_high = pd.Series(100 + np.cumsum(np.random.randn(100) * 0.1))
        vol_high = compute_ewma_volatility(prices_high, halflife_seconds=20, min_periods=10)
        
        # High vol series should have higher estimated volatility
        assert vol_high.iloc[-1] > vol_low.iloc[-1]
    
    def test_min_periods(self):
        """Test that min_periods produces NaN before threshold."""
        prices = pd.Series([100.0, 100.1, 100.05, 100.15, 100.1])
        
        vol = compute_ewma_volatility(prices, halflife_seconds=2, min_periods=3)
        
        # First 3 should be NaN (need min_periods observations)
        assert pd.isna(vol.iloc[0])
        assert pd.isna(vol.iloc[1])
        assert pd.isna(vol.iloc[2])
        # Fourth should have value
        assert not pd.isna(vol.iloc[3])
    
    def test_halflife_effect(self):
        """Test that shorter halflife reacts faster to changes."""
        # Create series with volatility spike
        prices = pd.Series([100.0] * 50 + [100.5, 99.5, 100.5, 99.5] * 5 + [100.0] * 50)
        
        vol_short = compute_ewma_volatility(prices, halflife_seconds=5, min_periods=5)
        vol_long = compute_ewma_volatility(prices, halflife_seconds=30, min_periods=5)
        
        # At spike (around idx 50-70), short halflife should be higher
        spike_idx = 60
        assert vol_short.iloc[spike_idx] > vol_long.iloc[spike_idx]


class TestComputeImbalance:
    """Test depth imbalance computation."""
    
    def test_balanced(self):
        """Test imbalance with equal bid/ask sizes."""
        bid_sz = pd.Series([500.0, 1000.0, 250.0])
        ask_sz = pd.Series([500.0, 1000.0, 250.0])
        
        imb = compute_imbalance(bid_sz, ask_sz)
        
        expected = pd.Series([0.0, 0.0, 0.0])
        pd.testing.assert_series_equal(imb, expected)
    
    def test_bid_heavy(self):
        """Test positive imbalance (bid > ask)."""
        bid_sz = pd.Series([1000.0])
        ask_sz = pd.Series([500.0])
        
        imb = compute_imbalance(bid_sz, ask_sz)
        
        # (1000 - 500) / (1000 + 500) = 500 / 1500 = 0.333...
        expected = 1.0 / 3.0
        assert abs(imb.iloc[0] - expected) < 1e-6
    
    def test_ask_heavy(self):
        """Test negative imbalance (ask > bid)."""
        bid_sz = pd.Series([300.0])
        ask_sz = pd.Series([700.0])
        
        imb = compute_imbalance(bid_sz, ask_sz)
        
        # (300 - 700) / (300 + 700) = -400 / 1000 = -0.4
        expected = -0.4
        assert abs(imb.iloc[0] - expected) < 1e-6
    
    def test_extreme_imbalance(self):
        """Test maximum imbalance (one side zero)."""
        # All bid, no ask
        bid_sz = pd.Series([1000.0])
        ask_sz = pd.Series([0.0])
        imb_max = compute_imbalance(bid_sz, ask_sz)
        assert imb_max.iloc[0] == 1.0
        
        # All ask, no bid
        bid_sz = pd.Series([0.0])
        ask_sz = pd.Series([1000.0])
        imb_min = compute_imbalance(bid_sz, ask_sz)
        assert imb_min.iloc[0] == -1.0
    
    def test_zero_depth(self):
        """Test imbalance with zero depth on both sides."""
        bid_sz = pd.Series([0.0, 500.0])
        ask_sz = pd.Series([0.0, 500.0])
        
        imb = compute_imbalance(bid_sz, ask_sz)
        
        # Zero depth should return 0 (neutral)
        assert imb.iloc[0] == 0.0
        assert imb.iloc[1] == 0.0
    
    def test_numpy_arrays(self):
        """Test with numpy arrays."""
        bid_sz = np.array([600.0, 400.0])
        ask_sz = np.array([400.0, 600.0])
        
        imb = compute_imbalance(bid_sz, ask_sz)
        
        assert isinstance(imb, pd.Series)
        assert imb.iloc[0] > 0  # First is bid-heavy
        assert imb.iloc[1] < 0  # Second is ask-heavy


class TestBlendSignals:
    """Test signal blending."""
    
    def test_basic_blend(self):
        """Test basic two-signal blend."""
        ofi_sig = pd.Series([2.0, -1.5, 0.5])
        imb = pd.Series([0.3, -0.2, 0.1])
        
        blended = blend_signals(ofi_sig, imb, alpha_ofi=0.7, alpha_imbalance=0.3)
        
        # Expected: 0.7 * ofi + 0.3 * imb
        expected = pd.Series([
            0.7 * 2.0 + 0.3 * 0.3,    # = 1.49
            0.7 * -1.5 + 0.3 * -0.2,  # = -1.11
            0.7 * 0.5 + 0.3 * 0.1     # = 0.38
        ])
        pd.testing.assert_series_equal(blended, expected)
    
    def test_equal_weights(self):
        """Test blending with equal weights."""
        ofi_sig = pd.Series([1.0, -1.0])
        imb = pd.Series([1.0, 1.0])
        
        blended = blend_signals(ofi_sig, imb, alpha_ofi=0.5, alpha_imbalance=0.5)
        
        expected = pd.Series([1.0, 0.0])  # Average
        pd.testing.assert_series_equal(blended, expected)
    
    def test_ofi_only(self):
        """Test with only OFI signal (zero imbalance weight)."""
        ofi_sig = pd.Series([2.0, -1.5])
        imb = pd.Series([0.5, -0.5])
        
        blended = blend_signals(ofi_sig, imb, alpha_ofi=1.0, alpha_imbalance=0.0)
        
        pd.testing.assert_series_equal(blended, ofi_sig)
    
    def test_additional_signals(self):
        """Test blending with additional signals."""
        ofi_sig = pd.Series([2.0, -1.0])
        imb = pd.Series([0.2, -0.2])
        momentum = pd.Series([1.0, 0.5])
        
        blended = blend_signals(
            ofi_sig, imb,
            alpha_ofi=0.5, alpha_imbalance=0.2,
            additional_signals={'momentum': momentum},
            additional_alphas={'momentum': 0.3}
        )
        
        expected = pd.Series([
            0.5 * 2.0 + 0.2 * 0.2 + 0.3 * 1.0,   # = 1.34
            0.5 * -1.0 + 0.2 * -0.2 + 0.3 * 0.5  # = -0.39
        ])
        pd.testing.assert_series_equal(blended, expected)
    
    def test_missing_alpha_error(self):
        """Test that missing alpha for additional signal raises error."""
        ofi_sig = pd.Series([1.0])
        imb = pd.Series([0.0])
        extra = pd.Series([0.5])
        
        with pytest.raises(ValueError, match="Weight.*not provided"):
            blend_signals(
                ofi_sig, imb,
                additional_signals={'extra': extra},
                additional_alphas={}  # Missing weight
            )


class TestComputeSignalStats:
    """Test signal statistics computation."""
    
    def test_basic_stats(self):
        """Test that all statistics are computed."""
        signal = pd.Series(np.random.randn(100))
        
        stats = compute_signal_stats(signal, window_seconds=20)
        
        # Check columns exist
        assert 'mean' in stats.columns
        assert 'std' in stats.columns
        assert 'min' in stats.columns
        assert 'max' in stats.columns
        assert 'p25' in stats.columns
        assert 'p50' in stats.columns
        assert 'p75' in stats.columns
    
    def test_constant_signal_stats(self):
        """Test stats for constant signal."""
        signal = pd.Series([5.0] * 100)
        
        stats = compute_signal_stats(signal, window_seconds=10)
        
        # After warm-up, all should be 5.0 for constant signal
        assert stats['mean'].iloc[-1] == 5.0
        assert stats['min'].iloc[-1] == 5.0
        assert stats['max'].iloc[-1] == 5.0
        assert stats['p50'].iloc[-1] == 5.0
        assert stats['std'].iloc[-1] == 0.0
    
    def test_preserves_index(self):
        """Test that output preserves input index."""
        index = pd.date_range('2017-01-03 09:30:00', periods=50, freq='1s')
        signal = pd.Series(np.random.randn(50), index=index)
        
        stats = compute_signal_stats(signal, window_seconds=10)
        
        assert stats.index.equals(index)


# Run tests if executed directly
if __name__ == '__main__':
    pytest.main([__file__, '-v'])

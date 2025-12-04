"""
tests/test_metrics.py

Unit tests for performance metrics.
Tests use known-answer synthetic data to verify correctness.
NO data-dependent assertions - only mathematical correctness.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from maker.metrics import (
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_max_drawdown,
    compute_fill_metrics,
    compute_fill_edge,
    compute_adverse_selection,
    compute_inventory_metrics,
    compute_signal_correlation,
    compute_all_metrics,
    PerformanceMetrics,
)
from maker.backtest import Fill


@pytest.fixture
def timestamps():
    """Create test timestamps."""
    start = pd.Timestamp('2017-01-03 09:30:00', tz='America/New_York')
    return pd.date_range(start=start, periods=100, freq='1s')


@pytest.fixture
def zero_returns():
    """Returns series of all zeros."""
    return pd.Series(np.zeros(100))


@pytest.fixture
def constant_returns():
    """Constant positive returns."""
    return pd.Series(np.ones(100) * 0.01)


@pytest.fixture
def normal_returns():
    """Normal distributed returns with seed."""
    np.random.seed(42)
    return pd.Series(np.random.randn(100) * 0.01)


class TestSharpeRatio:
    """Test Sharpe ratio computation."""
    
    def test_zero_returns(self, zero_returns):
        """Zero returns should give Sharpe = 0."""
        sharpe = compute_sharpe_ratio(zero_returns)
        assert sharpe == 0.0
    
    def test_constant_returns(self, constant_returns):
        """Constant returns should give Sharpe = 0 (no volatility)."""
        sharpe = compute_sharpe_ratio(constant_returns)
        assert sharpe == 0.0
    
    def test_positive_mean_positive_sharpe(self):
        """Positive mean with volatility should give positive Sharpe."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01])
        sharpe = compute_sharpe_ratio(returns, periods_per_year=252)
        assert sharpe > 0
    
    def test_negative_mean_negative_sharpe(self):
        """Negative mean should give negative Sharpe."""
        returns = pd.Series([-0.01, -0.02, 0.01, -0.03, -0.01])
        sharpe = compute_sharpe_ratio(returns, periods_per_year=252)
        assert sharpe < 0
    
    def test_empty_returns(self):
        """Empty returns should give Sharpe = 0."""
        returns = pd.Series([], dtype=float)
        sharpe = compute_sharpe_ratio(returns)
        assert sharpe == 0.0
    
    def test_single_return(self):
        """Single return should give Sharpe = 0 (can't compute std)."""
        returns = pd.Series([0.01])
        sharpe = compute_sharpe_ratio(returns)
        assert sharpe == 0.0
    
    def test_annualization_scaling(self):
        """Sharpe should scale with sqrt(periods)."""
        returns = pd.Series(np.random.randn(1000) * 0.01)
        
        # Daily vs hourly should differ by sqrt factor
        sharpe_daily = compute_sharpe_ratio(returns, periods_per_year=252)
        sharpe_hourly = compute_sharpe_ratio(returns, periods_per_year=252*6.5)
        
        # Hourly Sharpe should be larger (more periods to compound)
        assert abs(sharpe_hourly) > abs(sharpe_daily)


class TestSortinoRatio:
    """Test Sortino ratio computation."""
    
    def test_no_downside_returns(self):
        """All positive returns should give infinite Sortino."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.01])
        sortino = compute_sortino_ratio(returns)
        assert sortino == np.inf
    
    def test_only_downside_returns(self):
        """All negative returns should give negative Sortino."""
        returns = pd.Series([-0.01, -0.02, -0.03, -0.01])
        sortino = compute_sortino_ratio(returns)
        assert sortino < 0
    
    def test_mixed_returns(self):
        """Mixed returns should give finite Sortino."""
        returns = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01])
        sortino = compute_sortino_ratio(returns, periods_per_year=252)
        assert np.isfinite(sortino)
    
    def test_sortino_vs_sharpe(self):
        """Sortino should be >= Sharpe for same positive-mean returns."""
        # Returns with positive mean and some downside
        returns = pd.Series([0.03, -0.01, 0.02, -0.005, 0.025])
        
        sharpe = compute_sharpe_ratio(returns, periods_per_year=252)
        sortino = compute_sortino_ratio(returns, periods_per_year=252)
        
        # Sortino only penalizes downside, so should be >= Sharpe
        assert sortino >= sharpe


class TestMaxDrawdown:
    """Test maximum drawdown computation."""
    
    def test_monotonic_increase(self):
        """Monotonically increasing P&L should have zero drawdown."""
        pnl = pd.Series([0, 10, 20, 30, 40, 50])
        max_dd, max_dd_pct = compute_max_drawdown(pnl)
        assert max_dd == 0.0
        assert max_dd_pct == 0.0
    
    def test_monotonic_decrease(self):
        """Monotonically decreasing P&L should have max drawdown = total drop."""
        pnl = pd.Series([100, 90, 80, 70, 60, 50])
        max_dd, max_dd_pct = compute_max_drawdown(pnl)
        assert max_dd == 50.0  # 100 - 50
        assert max_dd_pct == 0.5  # 50%
    
    def test_single_drop(self):
        """Single drop should be captured correctly."""
        pnl = pd.Series([0, 100, 200, 150, 250])
        max_dd, max_dd_pct = compute_max_drawdown(pnl)
        assert max_dd == 50.0  # 200 - 150
        assert max_dd_pct == 0.25  # 25%
    
    def test_multiple_drops(self):
        """Should find the maximum among multiple drops."""
        pnl = pd.Series([0, 100, 80, 120, 60, 100])
        max_dd, max_dd_pct = compute_max_drawdown(pnl)
        # Max at 120, min after at 60, DD = 60
        assert max_dd == 60.0
        assert max_dd_pct == 0.5  # 50%
    
    def test_empty_pnl(self):
        """Empty P&L should give zero drawdown."""
        pnl = pd.Series([], dtype=float)
        max_dd, max_dd_pct = compute_max_drawdown(pnl)
        assert max_dd == 0.0
        assert max_dd_pct == 0.0


class TestFillMetrics:
    """Test fill-related metrics."""
    
    def test_fill_metrics_basic(self, timestamps):
        """Test basic fill metrics computation."""
        fills = [
            Fill(timestamp=timestamps[10], side='bid', price=100.0, size=100),
            Fill(timestamp=timestamps[20], side='ask', price=100.1, size=100),
            Fill(timestamp=timestamps[30], side='bid', price=99.9, size=100),
        ]
        
        duration = 100.0  # seconds
        fill_rate, total_fills, total_volume = compute_fill_metrics(fills, duration)
        
        assert total_fills == 3
        assert total_volume == 300
        assert fill_rate == 0.03  # 3 fills / 100 seconds
    
    def test_empty_fills(self):
        """Empty fills should give zero metrics."""
        fills = []
        fill_rate, total_fills, total_volume = compute_fill_metrics(fills, 100.0)
        
        assert fill_rate == 0.0
        assert total_fills == 0
        assert total_volume == 0
    
    def test_zero_duration(self):
        """Zero duration should give zero fill rate."""
        fills = [Fill(timestamp=pd.Timestamp.now(), side='bid', price=100.0, size=100)]
        fill_rate, total_fills, total_volume = compute_fill_metrics(fills, 0.0)
        
        assert fill_rate == 0.0
        assert total_fills == 1
        assert total_volume == 100


class TestFillEdge:
    """Test fill edge computation."""
    
    def test_bid_fill_positive_edge(self, timestamps):
        """Bid fill above mid should have positive edge."""
        fills = [Fill(timestamp=timestamps[10], side='bid', price=100.05, size=100)]
        mid_prices = pd.Series(100.0, index=timestamps)
        
        avg_edge, std_edge = compute_fill_edge(fills, mid_prices)
        
        # Sold at 100.05, mid was 100.00
        # Edge = (100.05 - 100.00) / 100.00 * 10000 = 5 bps
        assert avg_edge == pytest.approx(5.0, abs=0.1)
    
    def test_ask_fill_positive_edge(self, timestamps):
        """Ask fill below mid should have positive edge."""
        fills = [Fill(timestamp=timestamps[10], side='ask', price=99.95, size=100)]
        mid_prices = pd.Series(100.0, index=timestamps)
        
        avg_edge, std_edge = compute_fill_edge(fills, mid_prices)
        
        # Bought at 99.95, mid was 100.00
        # Edge = (100.00 - 99.95) / 100.00 * 10000 = 5 bps
        assert avg_edge == pytest.approx(5.0, abs=0.1)
    
    def test_bid_fill_negative_edge(self, timestamps):
        """Bid fill below mid should have negative edge (adverse)."""
        fills = [Fill(timestamp=timestamps[10], side='bid', price=99.95, size=100)]
        mid_prices = pd.Series(100.0, index=timestamps)
        
        avg_edge, std_edge = compute_fill_edge(fills, mid_prices)
        
        # Sold at 99.95, mid was 100.00 - bad fill
        assert avg_edge == pytest.approx(-5.0, abs=0.1)
    
    def test_multiple_fills(self, timestamps):
        """Multiple fills should average correctly."""
        fills = [
            Fill(timestamp=timestamps[10], side='bid', price=100.05, size=100),  # +5 bps
            Fill(timestamp=timestamps[20], side='ask', price=99.95, size=100),   # +5 bps
            Fill(timestamp=timestamps[30], side='bid', price=100.00, size=100),  # 0 bps
        ]
        mid_prices = pd.Series(100.0, index=timestamps)
        
        avg_edge, std_edge = compute_fill_edge(fills, mid_prices)
        
        assert avg_edge == pytest.approx(3.333, abs=0.1)  # (5 + 5 + 0) / 3
        assert std_edge > 0  # Should have some variance
    
    def test_empty_fills(self, timestamps):
        """Empty fills should give zero edge."""
        fills = []
        mid_prices = pd.Series(100.0, index=timestamps)
        
        avg_edge, std_edge = compute_fill_edge(fills, mid_prices)
        
        assert avg_edge == 0.0
        assert std_edge == 0.0


class TestAdverseSelection:
    """Test adverse selection computation."""
    
    def test_bid_fill_adverse_price_rise(self, timestamps):
        """Bid fill followed by price rise is adverse."""
        fills = [Fill(timestamp=timestamps[10], side='bid', price=100.0, size=100)]
        
        # Price rises after fill (adverse for seller)
        mid_prices = pd.Series(100.0, index=timestamps)
        mid_prices.iloc[11] = 100.10  # 1s later, +10 bps
        
        as_metrics = compute_adverse_selection(fills, mid_prices, horizons=[1])
        
        # Adverse = (100.10 - 100.00) / 100.00 * 10000 = 10 bps
        assert as_metrics['as_1s_bps'] == pytest.approx(10.0, abs=0.1)
    
    def test_ask_fill_adverse_price_fall(self, timestamps):
        """Ask fill followed by price fall is adverse."""
        fills = [Fill(timestamp=timestamps[10], side='ask', price=100.0, size=100)]
        
        # Price falls after fill (adverse for buyer)
        mid_prices = pd.Series(100.0, index=timestamps)
        mid_prices.iloc[11] = 99.90  # 1s later, -10 bps
        
        as_metrics = compute_adverse_selection(fills, mid_prices, horizons=[1])
        
        # Adverse = (100.00 - 99.90) / 100.00 * 10000 = 10 bps
        assert as_metrics['as_1s_bps'] == pytest.approx(10.0, abs=0.1)
    
    def test_favorable_selection(self, timestamps):
        """Favorable price moves should give negative adverse selection."""
        fills = [Fill(timestamp=timestamps[10], side='bid', price=100.0, size=100)]
        
        # Price falls after sell (favorable)
        mid_prices = pd.Series(100.0, index=timestamps)
        mid_prices.iloc[11] = 99.90
        
        as_metrics = compute_adverse_selection(fills, mid_prices, horizons=[1])
        
        assert as_metrics['as_1s_bps'] < 0  # Favorable
    
    def test_multiple_horizons(self, timestamps):
        """Should compute multiple horizons correctly."""
        fills = [Fill(timestamp=timestamps[10], side='bid', price=100.0, size=100)]
        
        mid_prices = pd.Series(100.0, index=timestamps)
        mid_prices.iloc[11] = 100.05  # 1s: +5 bps
        mid_prices.iloc[15] = 100.10  # 5s: +10 bps
        mid_prices.iloc[20] = 100.15  # 10s: +15 bps
        
        as_metrics = compute_adverse_selection(fills, mid_prices, horizons=[1, 5, 10])
        
        assert as_metrics['as_1s_bps'] == pytest.approx(5.0, abs=0.1)
        assert as_metrics['as_5s_bps'] == pytest.approx(10.0, abs=0.1)
        assert as_metrics['as_10s_bps'] == pytest.approx(15.0, abs=0.1)
    
    def test_empty_fills(self, timestamps):
        """Empty fills should give zero adverse selection."""
        fills = []
        mid_prices = pd.Series(100.0, index=timestamps)
        
        as_metrics = compute_adverse_selection(fills, mid_prices, horizons=[1, 5, 10])
        
        assert as_metrics['as_1s_bps'] == 0.0
        assert as_metrics['as_5s_bps'] == 0.0
        assert as_metrics['as_10s_bps'] == 0.0


class TestInventoryMetrics:
    """Test inventory risk metrics."""
    
    def test_zero_inventory(self):
        """Zero inventory should give all zeros."""
        inventory = pd.Series(np.zeros(100))
        
        avg_inv, std_inv, max_inv, min_inv, time_at_limit = compute_inventory_metrics(
            inventory, inventory_limit=100
        )
        
        assert avg_inv == 0.0
        assert std_inv == 0.0
        assert max_inv == 0
        assert min_inv == 0
        assert time_at_limit == 0.0
    
    def test_constant_inventory(self):
        """Constant inventory should have zero std."""
        inventory = pd.Series([50] * 100)
        
        avg_inv, std_inv, max_inv, min_inv, time_at_limit = compute_inventory_metrics(
            inventory, inventory_limit=100
        )
        
        assert avg_inv == 50.0
        assert std_inv == 0.0
        assert max_inv == 50
        assert min_inv == 50
        assert time_at_limit == 0.0
    
    def test_inventory_at_limit(self):
        """Inventory at limit should be detected."""
        inventory = pd.Series([100, 100, 50, 100, 0, -100, -100])
        
        avg_inv, std_inv, max_inv, min_inv, time_at_limit = compute_inventory_metrics(
            inventory, inventory_limit=100
        )
        
        assert max_inv == 100
        assert min_inv == -100
        # At limit: 5 out of 7 times (100, 100, 100, -100, -100)
        # Note: abs(inventory) >= limit checks absolute value
        assert time_at_limit == pytest.approx(5/7 * 100, abs=1.0)
    
    def test_inventory_statistics(self):
        """Inventory stats should be correct."""
        inventory = pd.Series([0, 10, 20, 10, 0, -10, -20, -10, 0])
        
        avg_inv, std_inv, max_inv, min_inv, time_at_limit = compute_inventory_metrics(
            inventory, inventory_limit=100
        )
        
        assert avg_inv == 0.0  # Symmetric
        assert std_inv > 0  # Has variance
        assert max_inv == 20
        assert min_inv == -20


class TestSignalCorrelation:
    """Test signal correlation computation."""
    
    def test_perfect_positive_correlation(self):
        """Perfect positive correlation should give 1.0."""
        signal = pd.Series(np.arange(100))
        returns = pd.Series(np.arange(100))
        
        corr = compute_signal_correlation(signal, returns)
        assert corr == pytest.approx(1.0, abs=0.01)
    
    def test_perfect_negative_correlation(self):
        """Perfect negative correlation should give -1.0."""
        signal = pd.Series(np.arange(100))
        returns = pd.Series(-np.arange(100))
        
        corr = compute_signal_correlation(signal, returns)
        assert corr == pytest.approx(-1.0, abs=0.01)
    
    def test_zero_correlation(self):
        """Uncorrelated series should give ~0."""
        np.random.seed(42)
        signal = pd.Series(np.random.randn(200))
        returns = pd.Series(np.random.randn(200))
        
        corr = compute_signal_correlation(signal, returns)
        assert abs(corr) < 0.2  # Should be near zero
    
    def test_insufficient_data(self):
        """Too few observations should give 0."""
        signal = pd.Series([1, 2, 3])
        returns = pd.Series([1, 2, 3])
        
        corr = compute_signal_correlation(signal, returns, min_periods=100)
        assert corr == 0.0


class TestPerformanceMetricsDataclass:
    """Test PerformanceMetrics dataclass."""
    
    def test_creation(self):
        """Should create with all required fields."""
        metrics = PerformanceMetrics(
            total_pnl=1000.0,
            total_return=10.0,
            sharpe_ratio=1.5,
            sortino_ratio=2.0,
            max_drawdown=100.0,
            max_drawdown_pct=5.0,
            total_fills=100,
            total_volume=10000,
            fill_rate=0.1,
            avg_fill_edge_bps=2.5,
            fill_edge_std_bps=1.0,
            adverse_selection_1s_bps=0.5,
            adverse_selection_5s_bps=1.0,
            adverse_selection_10s_bps=1.5,
            avg_inventory=10.0,
            inventory_std=5.0,
            max_inventory=50,
            min_inventory=-50,
            time_at_limit_pct=1.0,
        )
        
        assert metrics.total_pnl == 1000.0
        assert metrics.sharpe_ratio == 1.5
        assert metrics.ofi_signal_corr is None


class TestComputeAllMetrics:
    """Test complete metrics computation from BacktestResult."""
    
    def test_compute_all_metrics_integration(self, timestamps):
        """Integration test with mock BacktestResult."""
        from maker.backtest import BacktestResult
        
        # Create mock result
        n = len(timestamps)
        result = BacktestResult(
            symbol='TEST',
            date=pd.Timestamp('2017-01-03', tz='America/New_York'),
            timestamps=timestamps,
            inventory=pd.Series(np.zeros(n), index=timestamps),
            cash=pd.Series(np.arange(n), index=timestamps),
            pnl=pd.Series(np.arange(n), index=timestamps),
            bid=pd.Series(99.95, index=timestamps),
            ask=pd.Series(100.05, index=timestamps),
            mid=pd.Series(100.0, index=timestamps),
            ofi_features={'ofi_5s': pd.Series(np.random.randn(n), index=timestamps)},
            microprice=pd.Series(100.0, index=timestamps),
            volatility=pd.Series(0.02, index=timestamps),
            our_bid=pd.Series(99.90, index=timestamps),
            our_ask=pd.Series(100.10, index=timestamps),
            fills=[],
        )
        
        metrics = compute_all_metrics(result, inventory_limit=100)
        
        # Should compute without errors
        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.total_pnl == result.final_pnl
        assert metrics.total_fills == 0
        assert metrics.total_volume == 0

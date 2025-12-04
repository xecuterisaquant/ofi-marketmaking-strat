"""
tests/test_backtest.py

Integration tests for backtesting framework.
Tests order lifecycle, P&L reconciliation, and end-to-end execution.
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from maker.backtest import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    Order,
    Fill,
    load_nbbo_day,
    preprocess_nbbo,
)
from maker.engine import QuotingParams
from maker.fills import ParametricFillModel


@pytest.fixture
def simple_config():
    """Create a simple backtest configuration."""
    return BacktestConfig(
        quoting_params=QuotingParams(
            risk_aversion=0.1,
            terminal_time=300.0,
            tick_size=0.01,
        ),
        fill_model=ParametricFillModel(
            intensity_at_touch=2.0,
            decay_rate=0.5,
            time_step=1.0,
        ),
        ofi_windows=[5, 10, 30],
        volatility_window=60,
        initial_inventory=0,
        initial_cash=0.0,
        random_seed=42,
    )


@pytest.fixture
def synthetic_nbbo():
    """Create synthetic NBBO data for testing."""
    # Create 60 seconds of data (1 minute)
    start = pd.Timestamp('2017-01-03 09:30:00', tz='America/New_York')
    timestamps = pd.date_range(start=start, periods=60, freq='1s')
    
    # Simple constant market: bid=99.95, ask=100.05, mid=100.00
    data = pd.DataFrame({
        'bid': 99.95,
        'ask': 100.05,
        'bid_sz': 100,
        'ask_sz': 100,
    }, index=timestamps)
    
    return data


@pytest.fixture
def trending_nbbo():
    """Create trending NBBO data for testing."""
    start = pd.Timestamp('2017-01-03 09:30:00', tz='America/New_York')
    timestamps = pd.date_range(start=start, periods=120, freq='1s')
    
    # Upward trending market
    mid_prices = 100.0 + np.linspace(0, 1.0, len(timestamps))
    
    data = pd.DataFrame({
        'bid': mid_prices - 0.05,
        'ask': mid_prices + 0.05,
        'bid_sz': 100,
        'ask_sz': 100,
    }, index=timestamps)
    
    return data


class TestOrder:
    """Test Order dataclass."""
    
    def test_order_creation(self):
        """Test creating a valid order."""
        order = Order(
            side='bid',
            price=100.0,
            size=100,
            timestamp=pd.Timestamp('2017-01-03 09:30:00', tz='America/New_York')
        )
        
        assert order.side == 'bid'
        assert order.price == 100.0
        assert order.size == 100
    
    def test_invalid_side(self):
        """Test that invalid side raises error."""
        with pytest.raises(ValueError, match="Invalid side"):
            Order(
                side='invalid',
                price=100.0,
                size=100,
                timestamp=pd.Timestamp('2017-01-03 09:30:00', tz='America/New_York')
            )
    
    def test_invalid_size(self):
        """Test that non-positive size raises error."""
        with pytest.raises(ValueError, match="Size must be positive"):
            Order(
                side='bid',
                price=100.0,
                size=0,
                timestamp=pd.Timestamp('2017-01-03 09:30:00', tz='America/New_York')
            )


class TestFill:
    """Test Fill dataclass."""
    
    def test_bid_fill_cash_flow(self):
        """Test cash flow for bid fill (we sold)."""
        fill = Fill(
            timestamp=pd.Timestamp('2017-01-03 09:30:00', tz='America/New_York'),
            side='bid',
            price=100.0,
            size=10,
        )
        
        # Sold at 100.0, receive cash
        assert fill.cash_flow == 1000.0
        assert fill.inventory_change == -10
    
    def test_ask_fill_cash_flow(self):
        """Test cash flow for ask fill (we bought)."""
        fill = Fill(
            timestamp=pd.Timestamp('2017-01-03 09:30:00', tz='America/New_York'),
            side='ask',
            price=100.0,
            size=10,
        )
        
        # Bought at 100.0, pay cash
        assert fill.cash_flow == -1000.0
        assert fill.inventory_change == 10


class TestBacktestEngine:
    """Test BacktestEngine core functionality."""
    
    def test_initialization(self, simple_config):
        """Test engine initialization."""
        engine = BacktestEngine(simple_config)
        
        assert engine.inventory == 0
        assert engine.cash == 0.0
        assert engine.active_bid is None
        assert engine.active_ask is None
        assert len(engine.fills) == 0
    
    def test_reset(self, simple_config):
        """Test engine reset."""
        engine = BacktestEngine(simple_config)
        
        # Modify state
        engine.inventory = 50
        engine.cash = 1000.0
        engine.fills.append(Fill(
            timestamp=pd.Timestamp('2017-01-03 09:30:00', tz='America/New_York'),
            side='bid',
            price=100.0,
            size=10,
        ))
        
        # Reset
        engine.reset()
        
        assert engine.inventory == 0
        assert engine.cash == 0.0
        assert len(engine.fills) == 0
    
    def test_place_orders(self, simple_config):
        """Test order placement."""
        engine = BacktestEngine(simple_config)
        timestamp = pd.Timestamp('2017-01-03 09:30:00', tz='America/New_York')
        
        engine.place_orders(timestamp, bid_price=99.90, ask_price=100.10, size=100)
        
        assert engine.active_bid is not None
        assert engine.active_bid.price == 99.90
        assert engine.active_bid.size == 100
        
        assert engine.active_ask is not None
        assert engine.active_ask.price == 100.10
        assert engine.active_ask.size == 100
    
    def test_place_orders_cancels_previous(self, simple_config):
        """Test that placing new orders cancels previous ones."""
        engine = BacktestEngine(simple_config)
        timestamp = pd.Timestamp('2017-01-03 09:30:00', tz='America/New_York')
        
        # Place first orders
        engine.place_orders(timestamp, bid_price=99.90, ask_price=100.10, size=100)
        old_bid = engine.active_bid
        
        # Place new orders
        engine.place_orders(timestamp, bid_price=99.95, ask_price=100.05, size=100)
        
        # New orders should be different
        assert engine.active_bid is not old_bid
        assert engine.active_bid.price == 99.95
    
    def test_compute_pnl(self, simple_config):
        """Test P&L computation."""
        engine = BacktestEngine(simple_config)
        
        # Initial P&L should be 0
        assert engine.compute_pnl(100.0) == 0.0
        
        # Buy 10 shares at 100.0
        engine.cash = -1000.0
        engine.inventory = 10
        
        # At mid=100.0, P&L should be 0
        assert engine.compute_pnl(100.0) == 0.0
        
        # At mid=101.0, P&L should be +10 (gained $1 per share)
        assert engine.compute_pnl(101.0) == 10.0
        
        # At mid=99.0, P&L should be -10 (lost $1 per share)
        assert engine.compute_pnl(99.0) == -10.0
    
    def test_update_inventory_bid_fill(self, simple_config):
        """Test inventory update for bid fill (we sold)."""
        engine = BacktestEngine(simple_config)
        
        fill = Fill(
            timestamp=pd.Timestamp('2017-01-03 09:30:00', tz='America/New_York'),
            side='bid',
            price=100.0,
            size=10,
        )
        
        engine.update_inventory([fill])
        
        assert engine.cash == 1000.0  # Received cash
        assert engine.inventory == -10  # Sold shares
        assert len(engine.fills) == 1
    
    def test_update_inventory_ask_fill(self, simple_config):
        """Test inventory update for ask fill (we bought)."""
        engine = BacktestEngine(simple_config)
        
        fill = Fill(
            timestamp=pd.Timestamp('2017-01-03 09:30:00', tz='America/New_York'),
            side='ask',
            price=100.0,
            size=10,
        )
        
        engine.update_inventory([fill])
        
        assert engine.cash == -1000.0  # Paid cash
        assert engine.inventory == 10  # Bought shares
        assert len(engine.fills) == 1
    
    def test_check_inventory_limits(self, simple_config):
        """Test inventory limit checking."""
        engine = BacktestEngine(simple_config)
        
        # Within limits
        engine.inventory = 0
        assert engine.check_inventory_limits()
        
        engine.inventory = 50
        assert engine.check_inventory_limits()
        
        # At limits
        engine.inventory = 100
        assert engine.check_inventory_limits()
        
        engine.inventory = -100
        assert engine.check_inventory_limits()


class TestBacktestIntegration:
    """Integration tests for full backtest runs."""
    
    def test_run_single_day_basic(self, simple_config, synthetic_nbbo):
        """Test running a basic backtest."""
        engine = BacktestEngine(simple_config)
        
        result = engine.run_single_day(
            nbbo_data=synthetic_nbbo,
            symbol='TEST',
            date=pd.Timestamp('2017-01-03', tz='America/New_York'),
        )
        
        assert result.symbol == 'TEST'
        assert len(result.timestamps) > 0
        assert len(result.inventory) == len(result.timestamps)
        assert len(result.pnl) == len(result.timestamps)
    
    def test_run_single_day_quotes_generated(self, simple_config, synthetic_nbbo):
        """Test that quotes are generated."""
        engine = BacktestEngine(simple_config)
        
        result = engine.run_single_day(
            nbbo_data=synthetic_nbbo,
            symbol='TEST',
            date=pd.Timestamp('2017-01-03', tz='America/New_York'),
        )
        
        # Should have bid/ask quotes
        assert result.our_bid.notna().any()
        assert result.our_ask.notna().any()
        
        # Our bid should be <= market ask
        # Our ask should be >= market bid
        assert (result.our_bid <= result.ask).all()
        assert (result.our_ask >= result.bid).all()
    
    def test_pnl_reconciliation(self, simple_config, synthetic_nbbo):
        """Test that P&L = cash + inventory * mid."""
        engine = BacktestEngine(simple_config)
        
        result = engine.run_single_day(
            nbbo_data=synthetic_nbbo,
            symbol='TEST',
            date=pd.Timestamp('2017-01-03', tz='America/New_York'),
        )
        
        # Recompute P&L from cash and inventory
        computed_pnl = result.cash + result.inventory * result.mid
        
        # Should match reported P&L (within floating point error)
        np.testing.assert_allclose(result.pnl.values, computed_pnl.values, rtol=1e-10)
    
    def test_fills_affect_inventory(self, simple_config, synthetic_nbbo):
        """Test that fills change inventory."""
        # Use aggressive fill model to ensure fills happen
        config = BacktestConfig(
            quoting_params=simple_config.quoting_params,
            fill_model=ParametricFillModel(
                intensity_at_touch=10.0,  # Very aggressive
                decay_rate=0.1,
                time_step=1.0,
            ),
            random_seed=42,
        )
        
        engine = BacktestEngine(config)
        
        result = engine.run_single_day(
            nbbo_data=synthetic_nbbo,
            symbol='TEST',
            date=pd.Timestamp('2017-01-03', tz='America/New_York'),
        )
        
        # With aggressive fills, inventory should change
        inventory_changed = (result.inventory != 0).any()
        assert inventory_changed or len(result.fills) == 0  # Either changed or no fills
    
    def test_result_to_dataframe(self, simple_config, synthetic_nbbo):
        """Test converting result to DataFrame."""
        engine = BacktestEngine(simple_config)
        
        result = engine.run_single_day(
            nbbo_data=synthetic_nbbo,
            symbol='TEST',
            date=pd.Timestamp('2017-01-03', tz='America/New_York'),
        )
        
        df = result.to_dataframe()
        
        # Check expected columns
        assert 'inventory' in df.columns
        assert 'cash' in df.columns
        assert 'pnl' in df.columns
        assert 'our_bid' in df.columns
        assert 'our_ask' in df.columns
        assert 'ofi_5s' in df.columns  # OFI feature
        
        # Check index
        assert isinstance(df.index, pd.DatetimeIndex)
    
    def test_fills_dataframe(self, simple_config, synthetic_nbbo):
        """Test converting fills to DataFrame."""
        config = BacktestConfig(
            quoting_params=simple_config.quoting_params,
            fill_model=ParametricFillModel(
                intensity_at_touch=10.0,
                decay_rate=0.1,
                time_step=1.0,
            ),
            random_seed=42,
        )
        
        engine = BacktestEngine(config)
        
        result = engine.run_single_day(
            nbbo_data=synthetic_nbbo,
            symbol='TEST',
            date=pd.Timestamp('2017-01-03', tz='America/New_York'),
        )
        
        fills_df = result.fills_dataframe()
        
        # Should have expected columns even if empty
        expected_cols = {'timestamp', 'side', 'price', 'size', 'cash_flow', 'inventory_change'}
        assert expected_cols.issubset(set(fills_df.columns))
    
    def test_result_summary_stats(self, simple_config, synthetic_nbbo):
        """Test result summary statistics."""
        config = BacktestConfig(
            quoting_params=simple_config.quoting_params,
            fill_model=ParametricFillModel(
                intensity_at_touch=10.0,
                decay_rate=0.1,
                time_step=1.0,
            ),
            random_seed=42,
        )
        
        engine = BacktestEngine(config)
        
        result = engine.run_single_day(
            nbbo_data=synthetic_nbbo,
            symbol='TEST',
            date=pd.Timestamp('2017-01-03', tz='America/New_York'),
        )
        
        # Summary stats should match
        assert result.total_fills == len(result.fills)
        assert result.total_volume == sum(f.size for f in result.fills)
        
        if len(result.pnl) > 0:
            assert result.final_pnl == result.pnl.iloc[-1]
        
        if len(result.inventory) > 0:
            assert result.final_inventory == result.inventory.iloc[-1]


class TestDataLoading:
    """Test data loading utilities."""
    
    def test_load_nbbo_day_file_exists(self):
        """Test loading NBBO data from actual file."""
        # This test requires actual data file
        test_file = 'd:/Harshu/UIUC/Education/Semester 3/FIN 554-Algo Trading Sys Design & Testing/Assignments/ofi-marketmaking-strat/data/NBBO/2017-01-03.rda'
        
        try:
            nbbo_data, trading_day = load_nbbo_day(test_file)
            
            # Check basic properties
            assert isinstance(nbbo_data, pd.DataFrame)
            assert isinstance(nbbo_data.index, pd.DatetimeIndex)
            assert 'bid' in nbbo_data.columns
            assert 'ask' in nbbo_data.columns
            assert 'bid_sz' in nbbo_data.columns
            assert 'ask_sz' in nbbo_data.columns
            
            # Check trading day
            assert trading_day.year == 2017
            assert trading_day.month == 1
            assert trading_day.day == 3
            
        except FileNotFoundError:
            pytest.skip("Test data file not found")
    
    def test_preprocess_nbbo(self):
        """Test NBBO preprocessing."""
        # Create data with some invalid rows
        data = pd.DataFrame({
            'bid': [99.9, 100.0, 0.0, 99.8],  # Third row has 0 bid
            'ask': [100.1, 100.2, 100.1, 99.7],  # Fourth row is crossed
            'bid_sz': [100, 100, 100, 100],
            'ask_sz': [100, 100, 100, 100],
        })
        
        cleaned = preprocess_nbbo(data)
        
        # Should have only 2 valid rows
        assert len(cleaned) == 2
        assert (cleaned['bid'] > 0).all()
        assert (cleaned['ask'] > 0).all()
        assert (cleaned['ask'] >= cleaned['bid']).all()


class TestBacktestValidation:
    """Validation tests for backtest correctness."""
    
    def test_no_initial_pnl_without_fills(self, simple_config, synthetic_nbbo):
        """Test that P&L stays near 0 without fills."""
        # Use very passive fill model (unlikely to fill)
        config = BacktestConfig(
            quoting_params=simple_config.quoting_params,
            fill_model=ParametricFillModel(
                intensity_at_touch=0.001,  # Very low
                decay_rate=2.0,
                time_step=1.0,
            ),
            random_seed=42,
        )
        
        engine = BacktestEngine(config)
        
        result = engine.run_single_day(
            nbbo_data=synthetic_nbbo,
            symbol='TEST',
            date=pd.Timestamp('2017-01-03', tz='America/New_York'),
        )
        
        # With no fills, P&L should stay at 0
        if len(result.fills) == 0:
            assert (result.pnl == 0.0).all()
    
    def test_reproducibility_with_seed(self, simple_config, synthetic_nbbo):
        """Test that results are reproducible with same seed."""
        engine1 = BacktestEngine(simple_config)
        engine2 = BacktestEngine(simple_config)
        
        result1 = engine1.run_single_day(
            nbbo_data=synthetic_nbbo,
            symbol='TEST',
            date=pd.Timestamp('2017-01-03', tz='America/New_York'),
        )
        
        result2 = engine2.run_single_day(
            nbbo_data=synthetic_nbbo,
            symbol='TEST',
            date=pd.Timestamp('2017-01-03', tz='America/New_York'),
        )
        
        # Results should be identical
        np.testing.assert_array_equal(result1.pnl.values, result2.pnl.values)
        np.testing.assert_array_equal(result1.inventory.values, result2.inventory.values)
        assert result1.total_fills == result2.total_fills

"""
maker/backtest.py

Event-driven backtesting framework for market making strategies.
Integrates feature computation, quote generation, and fill simulation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple
from pathlib import Path

from maker.features import compute_ofi_signal, compute_microprice, compute_ewma_volatility
from maker.engine import QuotingEngine, QuotingParams, QuoteState
from maker.fills import ParametricFillModel
from src.ofi_utils import (
    read_rda, 
    resolve_columns, 
    build_tob_series_1s, 
    parse_trading_day_from_filename,
    compute_ofi_depth_mid,
    normalize_ofi,
)


@dataclass
class BacktestConfig:
    """Configuration for backtest execution."""
    
    # Strategy parameters
    quoting_params: QuotingParams = field(default_factory=QuotingParams)
    fill_model: ParametricFillModel = field(default_factory=ParametricFillModel)
    
    # OFI feature windows (seconds)
    ofi_windows: List[int] = field(default_factory=lambda: [5, 10, 30])
    
    # Volatility lookback (seconds)
    volatility_window: int = 60
    
    # Initial inventory
    initial_inventory: int = 0
    initial_cash: float = 0.0
    
    # Inventory limits
    max_inventory: int = 100
    min_inventory: int = -100
    
    # Random seed for reproducibility
    random_seed: Optional[int] = None


@dataclass
class Order:
    """Represents a limit order."""
    
    side: str  # 'bid' or 'ask'
    price: float
    size: int
    timestamp: pd.Timestamp
    
    def __post_init__(self):
        if self.side not in ['bid', 'ask']:
            raise ValueError(f"Invalid side: {self.side}")
        if self.size <= 0:
            raise ValueError(f"Size must be positive: {self.size}")


@dataclass
class Fill:
    """Represents an executed fill."""
    
    timestamp: pd.Timestamp
    side: str  # 'bid' (we sold) or 'ask' (we bought)
    price: float
    size: int
    
    @property
    def cash_flow(self) -> float:
        """Cash flow from this fill (positive = cash in)."""
        return self.price * self.size * (1 if self.side == 'bid' else -1)
    
    @property
    def inventory_change(self) -> int:
        """Change in inventory from this fill."""
        return -self.size if self.side == 'bid' else self.size


@dataclass
class BacktestResult:
    """
    Results from a single backtest run.
    
    Contains complete history of state, quotes, fills, and P&L.
    """
    
    # Metadata
    symbol: str
    date: pd.Timestamp
    
    # Time series (indexed by timestamp)
    timestamps: pd.DatetimeIndex
    inventory: pd.Series  # shares
    cash: pd.Series  # dollars
    pnl: pd.Series  # mark-to-market P&L
    
    # Market data
    bid: pd.Series
    ask: pd.Series
    mid: pd.Series
    
    # Features
    ofi_features: Dict[str, pd.Series]  # {f'ofi_{window}s': series}
    microprice: pd.Series
    volatility: pd.Series
    
    # Strategy outputs
    our_bid: pd.Series
    our_ask: pd.Series
    
    # Trading activity
    fills: List[Fill]
    
    # Summary statistics
    total_fills: int = 0
    total_volume: int = 0
    final_pnl: float = 0.0
    final_inventory: int = 0
    
    def __post_init__(self):
        """Compute summary statistics."""
        self.total_fills = len(self.fills)
        self.total_volume = sum(f.size for f in self.fills)
        self.final_pnl = float(self.pnl.iloc[-1]) if len(self.pnl) > 0 else 0.0
        self.final_inventory = int(self.inventory.iloc[-1]) if len(self.inventory) > 0 else 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert time series data to a single DataFrame."""
        df = pd.DataFrame({
            'inventory': self.inventory,
            'cash': self.cash,
            'pnl': self.pnl,
            'bid': self.bid,
            'ask': self.ask,
            'mid': self.mid,
            'our_bid': self.our_bid,
            'our_ask': self.our_ask,
            'microprice': self.microprice,
            'volatility': self.volatility,
        }, index=self.timestamps)
        
        # Add OFI features
        for name, series in self.ofi_features.items():
            df[name] = series
        
        return df
    
    def fills_dataframe(self) -> pd.DataFrame:
        """Convert fills to DataFrame."""
        if not self.fills:
            return pd.DataFrame(columns=['timestamp', 'side', 'price', 'size', 'cash_flow', 'inventory_change'])
        
        return pd.DataFrame([{
            'timestamp': f.timestamp,
            'side': f.side,
            'price': f.price,
            'size': f.size,
            'cash_flow': f.cash_flow,
            'inventory_change': f.inventory_change,
        } for f in self.fills])


class BacktestEngine:
    """
    Event-driven backtesting engine for market making strategies.
    
    Simulates order placement, fill simulation, and inventory management
    at 1-second intervals throughout the trading day.
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.
        
        Parameters
        ----------
        config : BacktestConfig
            Configuration parameters for the backtest.
        """
        self.config = config
        self.engine = QuotingEngine(config.quoting_params)
        
        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)
        
        # State tracking
        self.inventory: int = config.initial_inventory
        self.cash: float = config.initial_cash
        
        # Active orders (cancelled each second)
        self.active_bid: Optional[Order] = None
        self.active_ask: Optional[Order] = None
        
        # History tracking
        self.fills: List[Fill] = []
        self.state_history: List[Dict] = []
    
    def reset(self):
        """Reset engine state for new backtest."""
        self.inventory = self.config.initial_inventory
        self.cash = self.config.initial_cash
        self.active_bid = None
        self.active_ask = None
        self.fills = []
        self.state_history = []
        
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
    
    def place_orders(self, timestamp: pd.Timestamp, bid_price: float, ask_price: float, size: int = 100):
        """
        Place new bid and ask orders.
        
        Parameters
        ----------
        timestamp : pd.Timestamp
            Current time.
        bid_price : float
            Bid price to quote.
        ask_price : float
            Ask price to quote.
        size : int, default=100
            Order size in shares.
        """
        # Cancel any existing orders
        self.active_bid = None
        self.active_ask = None
        
        # Place new orders
        if not np.isnan(bid_price) and bid_price > 0:
            self.active_bid = Order(side='bid', price=bid_price, size=size, timestamp=timestamp)
        
        if not np.isnan(ask_price) and ask_price > 0:
            self.active_ask = Order(side='ask', price=ask_price, size=size, timestamp=timestamp)
    
    def simulate_fills(self, timestamp: pd.Timestamp, market_mid: float) -> List[Fill]:
        """
        Simulate fills for active orders.
        
        Parameters
        ----------
        timestamp : pd.Timestamp
            Current time.
        market_mid : float
            Current market midpoint.
        
        Returns
        -------
        fills : List[Fill]
            List of executed fills (0, 1, or 2 fills possible).
        """
        fills = []
        
        # Try to fill bid order
        if self.active_bid is not None:
            filled = self.config.fill_model.simulate_fill(
                quote_price=self.active_bid.price,
                microprice=market_mid,
                side='bid',
            )
            
            if filled:
                fill = Fill(
                    timestamp=timestamp,
                    side='bid',
                    price=self.active_bid.price,
                    size=self.active_bid.size,
                )
                fills.append(fill)
                self.active_bid = None  # Order filled, remove from book
        
        # Try to fill ask order
        if self.active_ask is not None:
            filled = self.config.fill_model.simulate_fill(
                quote_price=self.active_ask.price,
                microprice=market_mid,
                side='ask',
            )
            
            if filled:
                fill = Fill(
                    timestamp=timestamp,
                    side='ask',
                    price=self.active_ask.price,
                    size=self.active_ask.size,
                )
                fills.append(fill)
                self.active_ask = None  # Order filled, remove from book
        
        return fills
    
    def update_inventory(self, fills: List[Fill]):
        """
        Update cash and inventory based on fills.
        
        Parameters
        ----------
        fills : List[Fill]
            List of executed fills.
        """
        for fill in fills:
            self.cash += fill.cash_flow
            self.inventory += fill.inventory_change
            self.fills.append(fill)
    
    def compute_pnl(self, market_mid: float) -> float:
        """
        Compute mark-to-market P&L.
        
        P&L = cash + inventory * mid
        
        Parameters
        ----------
        market_mid : float
            Current market midpoint.
        
        Returns
        -------
        pnl : float
            Mark-to-market P&L in dollars.
        """
        return self.cash + self.inventory * market_mid
    
    def check_inventory_limits(self) -> bool:
        """
        Check if inventory is within limits.
        
        Returns
        -------
        within_limits : bool
            True if inventory is within configured limits.
        """
        return self.config.min_inventory <= self.inventory <= self.config.max_inventory
    
    def run_single_day(
        self,
        nbbo_data: pd.DataFrame,
        symbol: str,
        date: pd.Timestamp,
    ) -> BacktestResult:
        """
        Run backtest for a single trading day.
        
        Parameters
        ----------
        nbbo_data : pd.DataFrame
            NBBO data with columns: bid, ask, bid_sz, ask_sz
            Index must be DatetimeIndex at 1-second frequency.
        symbol : str
            Stock symbol.
        date : pd.Timestamp
            Trading date.
        
        Returns
        -------
        result : BacktestResult
            Complete backtest results.
        """
        self.reset()
        
        # Validate input data
        if not isinstance(nbbo_data.index, pd.DatetimeIndex):
            raise ValueError("nbbo_data must have DatetimeIndex")
        
        required_cols = ['bid', 'ask', 'bid_sz', 'ask_sz']
        missing = set(required_cols) - set(nbbo_data.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Compute features upfront (vectorized)
        bid_series = nbbo_data['bid']
        ask_series = nbbo_data['ask']
        bid_sz_series = nbbo_data['bid_sz']
        ask_sz_series = nbbo_data['ask_sz']
        
        # Compute OFI using ofi_utils workflow
        ofi_df = compute_ofi_depth_mid(nbbo_data)
        ofi_features = {}
        for window in self.config.ofi_windows:
            # Rolling sum of OFI over window
            ofi_rolling = ofi_df['ofi'].rolling(window=window, min_periods=1).sum()
            # Normalize by depth
            depth_rolling = ofi_df['depth'].rolling(window=window, min_periods=1).mean()
            ofi_normalized = ofi_rolling / depth_rolling.replace(0, np.nan)
            ofi_features[f'ofi_{window}s'] = ofi_normalized.fillna(0.0)
        
        mid_series = (bid_series + ask_series) / 2
        microprice_series = compute_microprice(bid_series, ask_series, bid_sz_series, ask_sz_series)
        volatility_series = compute_ewma_volatility(mid_series, halflife_seconds=self.config.volatility_window)
        
        # Event loop: iterate over each timestamp
        timestamps = []
        inventory_history = []
        cash_history = []
        pnl_history = []
        our_bid_history = []
        our_ask_history = []
        
        for idx, timestamp in enumerate(nbbo_data.index):
            # Get current market state
            bid = float(bid_series.iloc[idx])
            ask = float(ask_series.iloc[idx])
            mid = float(mid_series.iloc[idx])
            microprice = float(microprice_series.iloc[idx])
            volatility = float(volatility_series.iloc[idx])
            
            # Get OFI features
            ofi_values = {name: float(series.iloc[idx]) for name, series in ofi_features.items()}
            
            # Skip if any critical values are NaN
            if np.isnan(bid) or np.isnan(ask) or np.isnan(mid):
                continue
            
            # Use primary OFI window (first in list)
            ofi_normalized = ofi_values.get(f'ofi_{self.config.ofi_windows[0]}s', 0.0)
            if np.isnan(ofi_normalized):
                ofi_normalized = 0.0
            
            # Convert OFI to signal in basis points
            signal_bps = compute_ofi_signal(pd.Series([ofi_normalized]), beta=0.036).iloc[0]
            
            # Prepare quote state
            quote_state = QuoteState(
                bid=bid,
                ask=ask,
                bid_size=100,
                ask_size=100,
                microprice=microprice if not np.isnan(microprice) else mid,
                volatility=volatility if not np.isnan(volatility) else 0.02,  # Default 2% vol
                signal_bps=signal_bps,
                inventory=self.inventory,
                timestamp=timestamp,
            )
            
            # Generate quotes
            our_bid, our_ask = self.engine.generate_quotes(quote_state, time_to_close=300.0)
            
            # Place orders
            self.place_orders(timestamp, our_bid, our_ask, size=100)
            
            # Simulate fills
            fills = self.simulate_fills(timestamp, mid)
            
            # Update inventory
            self.update_inventory(fills)
            
            # Compute P&L
            pnl = self.compute_pnl(mid)
            
            # Record state
            timestamps.append(timestamp)
            inventory_history.append(self.inventory)
            cash_history.append(self.cash)
            pnl_history.append(pnl)
            our_bid_history.append(our_bid)
            our_ask_history.append(our_ask)
        
        # Build result
        result_index = pd.DatetimeIndex(timestamps)
        
        result = BacktestResult(
            symbol=symbol,
            date=date,
            timestamps=result_index,
            inventory=pd.Series(inventory_history, index=result_index),
            cash=pd.Series(cash_history, index=result_index),
            pnl=pd.Series(pnl_history, index=result_index),
            bid=bid_series.loc[result_index],
            ask=ask_series.loc[result_index],
            mid=mid_series.loc[result_index],
            ofi_features={name: series.loc[result_index] for name, series in ofi_features.items()},
            microprice=microprice_series.loc[result_index],
            volatility=volatility_series.loc[result_index],
            our_bid=pd.Series(our_bid_history, index=result_index),
            our_ask=pd.Series(our_ask_history, index=result_index),
            fills=self.fills,
        )
        
        return result


def load_nbbo_day(file_path: str) -> Tuple[pd.DataFrame, pd.Timestamp]:
    """
    Load NBBO data from .rda file and preprocess.
    
    Parameters
    ----------
    file_path : str
        Path to .rda file (e.g., 'data/NBBO/2017-01-03.rda').
    
    Returns
    -------
    nbbo_data : pd.DataFrame
        Preprocessed NBBO data with columns: bid, ask, bid_sz, ask_sz
        Indexed by timestamp at 1-second frequency during RTH.
    trading_day : pd.Timestamp
        Trading date parsed from filename.
    """
    # Read .rda file
    df = read_rda(file_path)
    
    # Resolve column names
    cmap = resolve_columns(df)
    
    # Parse trading day from filename
    trading_day = parse_trading_day_from_filename(file_path)
    
    # Build 1-second time series for RTH
    nbbo_data = build_tob_series_1s(df, cmap, trading_day, freq='1s')
    
    return nbbo_data, trading_day


def preprocess_nbbo(nbbo_data: pd.DataFrame) -> pd.DataFrame:
    """
    Additional preprocessing for NBBO data.
    
    Parameters
    ----------
    nbbo_data : pd.DataFrame
        Raw NBBO data with columns: bid, ask, bid_sz, ask_sz.
    
    Returns
    -------
    cleaned : pd.DataFrame
        Cleaned data with invalid rows removed.
    """
    # Filter out rows with non-positive prices
    cleaned = nbbo_data[
        (nbbo_data['bid'] > 0) &
        (nbbo_data['ask'] > 0) &
        (nbbo_data['bid_sz'] > 0) &
        (nbbo_data['ask_sz'] > 0)
    ].copy()
    
    # Filter out crossed markets (should already be done, but double-check)
    cleaned = cleaned[cleaned['ask'] >= cleaned['bid']]
    
    return cleaned

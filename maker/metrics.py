"""
maker/metrics.py

Performance metrics for market making strategies.
All functions are pure calculations - no parameter fitting or optimization.

Anti-Overfitting Design:
- All metrics computed from observed data only
- No curve fitting or parameter optimization
- Conservative risk adjustments
- Out-of-sample testing required before use
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class PerformanceMetrics:
    """Complete performance summary for a backtest run."""
    
    # P&L metrics
    total_pnl: float
    total_return: float  # percentage
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    
    # Trading activity
    total_fills: int
    total_volume: int
    fill_rate: float  # fills per second
    
    # Fill quality
    avg_fill_edge_bps: float  # average edge per fill
    fill_edge_std_bps: float
    
    # Adverse selection
    adverse_selection_1s_bps: float  # post-fill drift at 1s
    adverse_selection_5s_bps: float  # post-fill drift at 5s
    adverse_selection_10s_bps: float  # post-fill drift at 10s
    
    # Inventory risk
    avg_inventory: float
    inventory_std: float
    max_inventory: int
    min_inventory: int
    time_at_limit_pct: float  # % of time at inventory limits
    
    # Signal effectiveness (if applicable)
    ofi_signal_corr: Optional[float] = None  # correlation(OFI, future returns)


def compute_sharpe_ratio(
    returns: pd.Series,
    periods_per_year: int = 252 * 6.5 * 3600,
    risk_free_rate: float = 0.0
) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Parameters
    ----------
    returns : pd.Series
        Period returns (not cumulative)
    periods_per_year : int, default=252*6.5*3600
        Number of periods per year (default: seconds in trading year)
    risk_free_rate : float, default=0.0
        Annualized risk-free rate
        
    Returns
    -------
    sharpe : float
        Annualized Sharpe ratio
        
    Notes
    -----
    Conservative design:
    - Uses sample std (not population)
    - Returns 0.0 if std=0 (no risk-adjustment for constant returns)
    - No small-sample corrections applied
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = returns.mean()
    std_return = returns.std(ddof=1)  # Sample std
    
    # Check for zero or near-zero std (constant returns)
    if std_return < 1e-10 or not np.isfinite(std_return):
        return 0.0
    
    # Annualize
    sharpe = (mean_return - risk_free_rate / periods_per_year) / std_return * np.sqrt(periods_per_year)
    
    if not np.isfinite(sharpe):
        return 0.0
    
    return float(sharpe)


def compute_sortino_ratio(
    returns: pd.Series,
    periods_per_year: int = 252 * 6.5 * 3600,
    risk_free_rate: float = 0.0,
    target_return: float = 0.0
) -> float:
    """
    Compute annualized Sortino ratio (downside deviation).
    
    Parameters
    ----------
    returns : pd.Series
        Period returns
    periods_per_year : int
        Periods per year for annualization
    risk_free_rate : float, default=0.0
        Annualized risk-free rate
    target_return : float, default=0.0
        Target return threshold (typically 0)
        
    Returns
    -------
    sortino : float
        Annualized Sortino ratio
        
    Notes
    -----
    Only penalizes downside volatility (returns < target).
    """
    if len(returns) < 2:
        return 0.0
    
    mean_return = returns.mean()
    
    # Downside returns only
    downside_returns = returns[returns < target_return]
    
    if len(downside_returns) == 0:
        return np.inf if mean_return > target_return else 0.0
    
    downside_std = downside_returns.std(ddof=1)
    
    if downside_std == 0:
        return 0.0
    
    # Annualize
    sortino = (mean_return - risk_free_rate / periods_per_year) / downside_std * np.sqrt(periods_per_year)
    
    return float(sortino)


def compute_max_drawdown(pnl: pd.Series) -> Tuple[float, float]:
    """
    Compute maximum drawdown in absolute and percentage terms.
    
    Parameters
    ----------
    pnl : pd.Series
        Mark-to-market P&L time series
        
    Returns
    -------
    max_dd : float
        Maximum drawdown in dollars
    max_dd_pct : float
        Maximum drawdown as percentage of running max
        
    Notes
    -----
    Drawdown = (current - running_max)
    Always non-positive. Returns absolute value.
    """
    if len(pnl) == 0:
        return 0.0, 0.0
    
    # Running maximum
    running_max = pnl.expanding().max()
    
    # Drawdown at each point
    drawdown = pnl - running_max
    
    max_dd = abs(float(drawdown.min()))
    
    # Percentage drawdown
    # Avoid division by zero or negative running max
    valid_pct_dd = []
    for i in range(len(pnl)):
        if running_max.iloc[i] > 0:
            pct_dd = (pnl.iloc[i] - running_max.iloc[i]) / running_max.iloc[i]
            valid_pct_dd.append(pct_dd)
    
    if valid_pct_dd:
        max_dd_pct = abs(float(min(valid_pct_dd)))
    else:
        max_dd_pct = 0.0
    
    return max_dd, max_dd_pct


def compute_fill_metrics(
    fills: List,
    duration_seconds: float
) -> Tuple[float, int, int]:
    """
    Compute fill-related metrics.
    
    Parameters
    ----------
    fills : List[Fill]
        List of Fill objects from backtest
    duration_seconds : float
        Total backtest duration in seconds
        
    Returns
    -------
    fill_rate : float
        Fills per second
    total_fills : int
        Total number of fills
    total_volume : int
        Total shares traded
    """
    total_fills = len(fills)
    total_volume = sum(f.size for f in fills)
    fill_rate = total_fills / duration_seconds if duration_seconds > 0 else 0.0
    
    return fill_rate, total_fills, total_volume


def compute_fill_edge(
    fills: List,
    mid_at_fill: pd.Series
) -> Tuple[float, float]:
    """
    Compute fill edge: average profit per fill.
    
    Fill edge = (mid_price - fill_price) * direction
    - Positive edge: filled better than mid
    - Negative edge: adverse selection
    
    Parameters
    ----------
    fills : List[Fill]
        List of Fill objects
    mid_at_fill : pd.Series
        Mid price at time of each fill (indexed by fill timestamp)
        
    Returns
    -------
    avg_edge_bps : float
        Average fill edge in basis points
    std_edge_bps : float
        Standard deviation of fill edge
        
    Notes
    -----
    Conservative calculation:
    - Uses mid price at exact fill time (no look-ahead)
    - Positive edge = we got better price than mid
    """
    if len(fills) == 0:
        return 0.0, 0.0
    
    edges = []
    
    for fill in fills:
        try:
            mid = mid_at_fill.loc[fill.timestamp]
        except KeyError:
            continue  # Skip if mid not available
        
        if fill.side == 'bid':
            # We sold at fill.price, mid is reference
            # Edge = fill.price - mid (positive if we sold above mid)
            edge = fill.price - mid
        else:  # ask
            # We bought at fill.price
            # Edge = mid - fill.price (positive if we bought below mid)
            edge = mid - fill.price
        
        # Convert to basis points
        edge_bps = (edge / mid) * 10000
        edges.append(edge_bps)
    
    if len(edges) == 0:
        return 0.0, 0.0
    
    avg_edge_bps = float(np.mean(edges))
    std_edge_bps = float(np.std(edges, ddof=1)) if len(edges) > 1 else 0.0
    
    return avg_edge_bps, std_edge_bps


def compute_adverse_selection(
    fills: List,
    mid_prices: pd.Series,
    horizons: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Compute adverse selection: post-fill price drift.
    
    Adverse selection occurs when prices move against us after filling.
    Measures how much price moves in unfavorable direction after fill.
    
    Parameters
    ----------
    fills : List[Fill]
        List of Fill objects
    mid_prices : pd.Series
        Complete mid price time series (1-second)
    horizons : List[int], default=[1, 5, 10]
        Horizons in seconds to measure drift
        
    Returns
    -------
    adverse_selection : Dict[str, float]
        {f'as_{h}s_bps': value} for each horizon
        Positive = adverse (price moved against us)
        
    Notes
    -----
    For bid fills (we sold):
        AS = (mid[t+h] - mid[t]) - positive means price rose after we sold
    For ask fills (we bought):
        AS = (mid[t] - mid[t+h]) - positive means price fell after we bought
    
    Uses only data available at fill time (no look-ahead).
    """
    result = {}
    
    if len(fills) == 0:
        return {f'as_{h}s_bps': 0.0 for h in horizons}
    
    for horizon in horizons:
        adverse_moves = []
        
        for fill in fills:
            try:
                # Mid at fill time
                mid_at_fill = mid_prices.loc[fill.timestamp]
                
                # Mid at horizon later
                fill_idx = mid_prices.index.get_loc(fill.timestamp)
                future_idx = fill_idx + horizon
                
                if future_idx >= len(mid_prices):
                    continue  # Not enough future data
                
                mid_future = mid_prices.iloc[future_idx]
                
                # Compute adverse move
                if fill.side == 'bid':
                    # We sold - adverse if price rises
                    adverse_move = mid_future - mid_at_fill
                else:  # ask
                    # We bought - adverse if price falls
                    adverse_move = mid_at_fill - mid_future
                
                # Convert to bps
                adverse_move_bps = (adverse_move / mid_at_fill) * 10000
                adverse_moves.append(adverse_move_bps)
                
            except (KeyError, IndexError):
                continue
        
        if len(adverse_moves) > 0:
            avg_adverse = float(np.mean(adverse_moves))
        else:
            avg_adverse = 0.0
        
        result[f'as_{horizon}s_bps'] = avg_adverse
    
    return result


def compute_inventory_metrics(
    inventory: pd.Series,
    inventory_limit: int
) -> Tuple[float, float, int, int, float]:
    """
    Compute inventory risk metrics.
    
    Parameters
    ----------
    inventory : pd.Series
        Inventory time series (shares)
    inventory_limit : int
        Max allowed inventory (absolute value)
        
    Returns
    -------
    avg_inventory : float
        Mean inventory over period
    inventory_std : float
        Standard deviation of inventory
    max_inventory : int
        Maximum inventory reached
    min_inventory : int
        Minimum inventory reached
    time_at_limit_pct : float
        Percentage of time at inventory limits
    """
    if len(inventory) == 0:
        return 0.0, 0.0, 0, 0, 0.0
    
    avg_inventory = float(inventory.mean())
    inventory_std = float(inventory.std(ddof=1)) if len(inventory) > 1 else 0.0
    max_inventory = int(inventory.max())
    min_inventory = int(inventory.min())
    
    # Time at limits
    at_limit = (inventory.abs() >= inventory_limit).sum()
    time_at_limit_pct = float(at_limit / len(inventory) * 100)
    
    return avg_inventory, inventory_std, max_inventory, min_inventory, time_at_limit_pct


def compute_signal_correlation(
    signal: pd.Series,
    future_returns: pd.Series,
    min_periods: int = 100
) -> float:
    """
    Compute correlation between signal and future returns.
    
    Validates if signal actually predicts price direction.
    
    Parameters
    ----------
    signal : pd.Series
        Signal values (e.g., OFI)
    future_returns : pd.Series
        Forward-looking returns (already computed, not look-ahead if used correctly)
    min_periods : int, default=100
        Minimum observations required
        
    Returns
    -------
    correlation : float
        Pearson correlation coefficient
        
    Notes
    -----
    WARNING: future_returns must be computed without look-ahead bias.
    Use only for validation, not for parameter fitting.
    """
    # Align and drop NaN
    aligned = pd.DataFrame({'signal': signal, 'returns': future_returns}).dropna()
    
    if len(aligned) < min_periods:
        return 0.0
    
    corr = float(aligned['signal'].corr(aligned['returns']))
    
    return corr if not np.isnan(corr) else 0.0


def compute_all_metrics(
    result,
    inventory_limit: int = 100,
    beta_from_replication: float = 0.036
) -> PerformanceMetrics:
    """
    Compute complete performance metrics from BacktestResult.
    
    Parameters
    ----------
    result : BacktestResult
        Backtest result object
    inventory_limit : int, default=100
        Inventory limit for risk metrics
    beta_from_replication : float, default=0.036
        Beta coefficient from separate replication study (not fitted)
        
    Returns
    -------
    metrics : PerformanceMetrics
        Complete metrics summary
        
    Notes
    -----
    This function ONLY computes metrics from observed data.
    No parameter optimization or curve fitting.
    All parameters (beta, limits, etc.) are fixed inputs.
    """
    # P&L metrics
    pnl = result.pnl
    returns = pnl.diff()  # Period returns
    
    total_pnl = result.final_pnl
    total_return = (total_pnl / abs(pnl.iloc[0])) * 100 if len(pnl) > 0 and pnl.iloc[0] != 0 else 0.0
    
    sharpe = compute_sharpe_ratio(returns)
    sortino = compute_sortino_ratio(returns)
    max_dd, max_dd_pct = compute_max_drawdown(pnl)
    
    # Fill metrics
    duration_seconds = (result.timestamps[-1] - result.timestamps[0]).total_seconds() if len(result.timestamps) > 1 else 1.0
    fill_rate, total_fills, total_volume = compute_fill_metrics(result.fills, duration_seconds)
    
    # Fill edge
    avg_edge, std_edge = compute_fill_edge(result.fills, result.mid)
    
    # Adverse selection
    as_metrics = compute_adverse_selection(result.fills, result.mid, horizons=[1, 5, 10])
    
    # Inventory metrics
    avg_inv, std_inv, max_inv, min_inv, time_at_limit = compute_inventory_metrics(
        result.inventory, inventory_limit
    )
    
    # Signal correlation (validation only - uses fixed beta, not fitted)
    ofi_signal_corr = None
    if len(result.ofi_features) > 0:
        primary_ofi = list(result.ofi_features.values())[0]
        future_returns = result.mid.pct_change().shift(-1) * 10000  # Next period return in bps
        ofi_signal_corr = compute_signal_correlation(primary_ofi, future_returns)
    
    return PerformanceMetrics(
        total_pnl=total_pnl,
        total_return=total_return,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_pct=max_dd_pct,
        total_fills=total_fills,
        total_volume=total_volume,
        fill_rate=fill_rate,
        avg_fill_edge_bps=avg_edge,
        fill_edge_std_bps=std_edge,
        adverse_selection_1s_bps=as_metrics.get('as_1s_bps', 0.0),
        adverse_selection_5s_bps=as_metrics.get('as_5s_bps', 0.0),
        adverse_selection_10s_bps=as_metrics.get('as_10s_bps', 0.0),
        avg_inventory=avg_inv,
        inventory_std=std_inv,
        max_inventory=max_inv,
        min_inventory=min_inv,
        time_at_limit_pct=time_at_limit,
        ofi_signal_corr=ofi_signal_corr,
    )

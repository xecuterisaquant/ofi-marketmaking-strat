"""
maker/features.py

Feature engineering for OFI-driven market making strategy.
Implements signal computation, microprice, volatility estimation, and signal blending.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, Union


def compute_ofi_signal(
    ofi_normalized: pd.Series,
    beta: float = 0.036,
    horizon_seconds: int = 60
) -> pd.Series:
    """
    Convert normalized OFI to expected price drift signal in basis points.
    
    Based on replication results, OFI predicts future price changes.
    Mean beta across 8 symbols (Jan 2017) = 0.036, meaning 1 unit of 
    normalized OFI predicts 3.6 bps drift over the horizon.
    
    Parameters
    ----------
    ofi_normalized : pd.Series
        Normalized order flow imbalance (OFI / sqrt(cumulative_volume))
    beta : float, default=0.036
        Coefficient from OFI regression, symbol-specific or average
    horizon_seconds : int, default=60
        Forecast horizon in seconds (affects scaling)
        
    Returns
    -------
    pd.Series
        Expected price drift in basis points
        
    Notes
    -----
    The signal is scaled linearly with beta. For real-time trading:
    - Positive signal → expect price to rise → quote tighter on ask
    - Negative signal → expect price to fall → quote tighter on bid
    
    Examples
    --------
    >>> ofi = pd.Series([0.5, -0.3, 1.2, 0.0])
    >>> signal = compute_ofi_signal(ofi, beta=0.036)
    >>> signal.values
    array([ 1.8, -1.08,  4.32,  0.  ])  # in basis points
    """
    # Convert OFI to expected drift in bps
    # beta represents bps per unit normalized OFI over horizon
    signal_bps = ofi_normalized * beta * 100  # Convert to bps (basis points = 0.01%)
    
    return signal_bps


def compute_microprice(
    bid: Union[pd.Series, np.ndarray],
    ask: Union[pd.Series, np.ndarray],
    bid_size: Union[pd.Series, np.ndarray],
    ask_size: Union[pd.Series, np.ndarray]
) -> pd.Series:
    """
    Compute depth-weighted microprice.
    
    The microprice incorporates order book depth to provide a more
    informative reference price than the simple midpoint. It weights
    bid and ask prices by the opposite side's depth.
    
    Formula: (P_ask * Q_bid + P_bid * Q_ask) / (Q_bid + Q_ask)
    
    Parameters
    ----------
    bid : pd.Series or np.ndarray
        Bid prices
    ask : pd.Series or np.ndarray  
        Ask prices
    bid_size : pd.Series or np.ndarray
        Bid sizes (depth)
    ask_size : pd.Series or np.ndarray
        Ask sizes (depth)
        
    Returns
    -------
    pd.Series
        Microprice series
        
    Notes
    -----
    When bid_size >> ask_size, microprice closer to bid (expect upward pressure)
    When ask_size >> bid_size, microprice closer to ask (expect downward pressure)
    When sizes equal, microprice = midprice
    
    Handles edge cases:
    - Zero sizes: falls back to midprice
    - NaN values: propagated through calculation
    
    Examples
    --------
    >>> bid = pd.Series([100.0, 100.5])
    >>> ask = pd.Series([100.1, 100.6])
    >>> bid_sz = pd.Series([500, 1000])
    >>> ask_sz = pd.Series([500, 200])
    >>> mp = compute_microprice(bid, ask, bid_sz, ask_sz)
    >>> mp.values
    array([100.05, 100.533...])
    """
    # Convert to Series for consistent handling
    if not isinstance(bid, pd.Series):
        bid = pd.Series(bid)
    if not isinstance(ask, pd.Series):
        ask = pd.Series(ask)
    if not isinstance(bid_size, pd.Series):
        bid_size = pd.Series(bid_size)
    if not isinstance(ask_size, pd.Series):
        ask_size = pd.Series(ask_size)
    
    # Calculate total depth
    total_depth = bid_size + ask_size
    
    # Compute microprice with fallback to midprice when depth is zero
    microprice = (ask * bid_size + bid * ask_size) / total_depth
    
    # Fallback to midprice where total_depth is 0 or NaN
    midprice = (bid + ask) / 2.0
    microprice = microprice.fillna(midprice)
    microprice = microprice.where(total_depth > 0, midprice)
    
    return microprice


def compute_ewma_volatility(
    prices: pd.Series,
    halflife_seconds: float = 60.0,
    min_periods: int = 10
) -> pd.Series:
    """
    Compute exponentially weighted moving average (EWMA) volatility.
    
    Uses squared returns to estimate volatility with recent observations
    weighted more heavily. EWMA reacts faster to volatility changes than
    simple rolling windows.
    
    Parameters
    ----------
    prices : pd.Series
        Price series (e.g., midprice or microprice)
    halflife_seconds : float, default=60.0
        Half-life for exponential weighting in seconds
        Smaller = more reactive, larger = smoother
    min_periods : int, default=10
        Minimum observations required for valid estimate
        
    Returns
    -------
    pd.Series
        Annualized volatility in same units as prices
        
    Notes
    -----
    Process:
    1. Compute log returns
    2. Square returns  
    3. Apply EWMA with specified half-life
    4. Take square root to get volatility
    5. Annualize assuming 6.5 trading hours/day, 252 days/year
    
    The EWMA decay factor: alpha = 1 - exp(log(0.5) / halflife)
    
    Examples
    --------
    >>> prices = pd.Series([100, 100.1, 100.05, 100.2, 100.15])
    >>> vol = compute_ewma_volatility(prices, halflife_seconds=60)
    >>> vol.iloc[-1]  # Latest volatility estimate
    0.0123...  # Annualized
    """
    # Compute log returns
    log_returns = np.log(prices / prices.shift(1))
    
    # Square returns (variance proxy)
    squared_returns = log_returns ** 2
    
    # Compute EWMA of squared returns
    # pandas ewm uses span, convert from halflife: span = halflife / log(2) 
    # But we want halflife in terms of observations, not time
    # Assuming prices are 1-second sampled, halflife_seconds = number of observations
    ewma_var = squared_returns.ewm(halflife=halflife_seconds, min_periods=min_periods).mean()
    
    # Take square root to get volatility (standard deviation)
    ewma_vol = np.sqrt(ewma_var)
    
    # Annualize: sqrt(252 * 6.5 * 3600) for 1-second data
    # = sqrt(5,896,800) ≈ 2,428
    annualization_factor = np.sqrt(252 * 6.5 * 3600)
    annualized_vol = ewma_vol * annualization_factor
    
    return annualized_vol


def compute_imbalance(
    bid_size: Union[pd.Series, np.ndarray],
    ask_size: Union[pd.Series, np.ndarray]
) -> pd.Series:
    """
    Compute order book depth imbalance.
    
    Measures the relative pressure from bid vs ask side depth.
    Ranges from -1 (all depth on ask) to +1 (all depth on bid).
    
    Formula: (Q_bid - Q_ask) / (Q_bid + Q_ask)
    
    Parameters
    ----------
    bid_size : pd.Series or np.ndarray
        Bid depth
    ask_size : pd.Series or np.ndarray
        Ask depth
        
    Returns
    -------
    pd.Series
        Imbalance metric in [-1, 1]
        
    Notes
    -----
    Interpretation:
    - imbalance > 0: More buying pressure (bid-heavy)
    - imbalance < 0: More selling pressure (ask-heavy)  
    - imbalance ≈ 0: Balanced book
    
    Edge cases:
    - Both sizes zero: returns 0 (neutral)
    - One side zero: returns ±1 (maximum imbalance)
    
    Examples
    --------
    >>> bid_sz = pd.Series([1000, 500, 800])
    >>> ask_sz = pd.Series([500, 500, 200])
    >>> imb = compute_imbalance(bid_sz, ask_sz)
    >>> imb.values
    array([0.333..., 0.0, 0.6])
    """
    # Convert to Series
    if not isinstance(bid_size, pd.Series):
        bid_size = pd.Series(bid_size)
    if not isinstance(ask_size, pd.Series):
        ask_size = pd.Series(ask_size)
    
    # Compute imbalance
    total_depth = bid_size + ask_size
    imbalance = (bid_size - ask_size) / total_depth
    
    # Handle zero depth case (return 0 for neutral)
    imbalance = imbalance.fillna(0.0)
    imbalance = imbalance.where(total_depth > 0, 0.0)
    
    return imbalance


def blend_signals(
    ofi_signal: pd.Series,
    imbalance: pd.Series,
    alpha_ofi: float = 0.7,
    alpha_imbalance: float = 0.3,
    additional_signals: Optional[Dict[str, pd.Series]] = None,
    additional_alphas: Optional[Dict[str, float]] = None
) -> pd.Series:
    """
    Blend multiple signals into a single composite signal.
    
    Combines OFI drift signal, depth imbalance, and optional additional
    signals using weighted average. Weights should sum to 1.0 for
    interpretability but are not enforced.
    
    Parameters
    ----------
    ofi_signal : pd.Series
        OFI-based drift signal in bps
    imbalance : pd.Series
        Depth imbalance in [-1, 1]
    alpha_ofi : float, default=0.7
        Weight for OFI signal
    alpha_imbalance : float, default=0.3
        Weight for imbalance signal
    additional_signals : dict, optional
        Additional signals to blend, keyed by name
    additional_alphas : dict, optional
        Weights for additional signals, keyed by name (must match additional_signals keys)
        
    Returns
    -------
    pd.Series
        Blended composite signal
        
    Notes
    -----
    Default weights (70% OFI, 30% imbalance) reflect that OFI is the
    primary predictive signal from replication study, while imbalance
    captures short-term execution risk.
    
    The imbalance is scaled to bps for combination with OFI signal.
    Typical scaling: imbalance * 5 bps (configurable in engine).
    
    Examples
    --------
    >>> ofi_sig = pd.Series([2.0, -1.5, 0.5])
    >>> imb = pd.Series([0.3, -0.2, 0.1])
    >>> blended = blend_signals(ofi_sig, imb, alpha_ofi=0.7, alpha_imbalance=0.3)
    >>> blended.values
    array([1.49, -1.11, 0.38])
    
    With additional signals:
    >>> momentum = pd.Series([1.0, 0.5, -0.5])
    >>> blended = blend_signals(
    ...     ofi_sig, imb, 
    ...     alpha_ofi=0.5, alpha_imbalance=0.2,
    ...     additional_signals={'momentum': momentum},
    ...     additional_alphas={'momentum': 0.3}
    ... )
    """
    # Start with primary signals
    blended = alpha_ofi * ofi_signal + alpha_imbalance * imbalance
    
    # Add additional signals if provided
    if additional_signals is not None and additional_alphas is not None:
        for name, signal in additional_signals.items():
            if name in additional_alphas:
                alpha = additional_alphas[name]
                blended = blended + alpha * signal
            else:
                raise ValueError(f"Weight (alpha) not provided for signal '{name}'")
    
    return blended


def compute_signal_stats(
    signal: pd.Series,
    window_seconds: int = 300
) -> pd.DataFrame:
    """
    Compute rolling statistics of a signal for monitoring and diagnostics.
    
    Calculates mean, std, min, max, and percentiles over rolling windows.
    Useful for understanding signal behavior and setting thresholds.
    
    Parameters
    ----------
    signal : pd.Series
        Input signal series
    window_seconds : int, default=300
        Rolling window size in seconds (default 5 minutes)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: mean, std, min, p25, p50, p75, max
        
    Examples
    --------
    >>> signal = pd.Series(np.random.randn(1000))
    >>> stats = compute_signal_stats(signal, window_seconds=60)
    >>> stats[['mean', 'std']].tail()
    """
    stats = pd.DataFrame(index=signal.index)
    stats['mean'] = signal.rolling(window_seconds).mean()
    stats['std'] = signal.rolling(window_seconds).std()
    stats['min'] = signal.rolling(window_seconds).min()
    stats['p25'] = signal.rolling(window_seconds).quantile(0.25)
    stats['p50'] = signal.rolling(window_seconds).quantile(0.50)
    stats['p75'] = signal.rolling(window_seconds).quantile(0.75)
    stats['max'] = signal.rolling(window_seconds).max()
    
    return stats

"""
maker/fills.py

Fill simulation models for market making strategy.
Implements parametric fill probability model based on distance from mid.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ParametricFillModel:
    """
    Parametric fill probability model based on queue distance.
    
    Models fill intensity as: λ(δ) = A * exp(-k * δ)
    where δ = distance from mid in basis points
    
    Parameters
    ----------
    intensity_at_touch : float, default=2.0
        Fill intensity (fills/second) when quoting at touch (δ=0)
        Higher values → more aggressive fills
    decay_rate : float, default=0.5
        Exponential decay rate with distance
        Higher values → faster decay (fills concentrated near touch)
    time_step : float, default=1.0
        Time step in seconds for discrete simulation
        
    Notes
    -----
    Fill probability over time_step Δt:
    P(fill | δ) = 1 - exp(-λ(δ) * Δt)
    
    Typical calibration (for liquid stocks):
    - At touch (δ=0): ~50% fill rate per second
    - At +1 bps: ~30% fill rate
    - At +5 bps: <5% fill rate
    """
    intensity_at_touch: float = 2.0  # A parameter
    decay_rate: float = 0.5          # k parameter
    time_step: float = 1.0           # Δt in seconds
    
    def compute_intensity(self, distance_bps: float) -> float:
        """
        Compute fill intensity λ(δ) for given distance from mid.
        
        Parameters
        ----------
        distance_bps : float
            Distance from microprice in basis points
            
        Returns
        -------
        float
            Fill intensity in fills/second
        """
        return self.intensity_at_touch * np.exp(-self.decay_rate * abs(distance_bps))
    
    def compute_fill_probability(self, distance_bps: float) -> float:
        """
        Compute fill probability over one time step.
        
        Parameters
        ----------
        distance_bps : float
            Distance from microprice in basis points
            
        Returns
        -------
        float
            Probability of fill in [0, 1]
            
        Notes
        -----
        Uses Poisson process: P(fill) = 1 - exp(-λ * Δt)
        """
        intensity = self.compute_intensity(distance_bps)
        prob = 1.0 - np.exp(-intensity * self.time_step)
        return np.clip(prob, 0.0, 1.0)
    
    def simulate_fill(
        self,
        quote_price: float,
        microprice: float,
        side: str,
        rng: Optional[np.random.Generator] = None
    ) -> bool:
        """
        Simulate whether a limit order gets filled in this time step.
        
        Parameters
        ----------
        quote_price : float
            Limit order price
        microprice : float
            Current microprice (reference price)
        side : str
            'bid' or 'ask'
        rng : np.random.Generator, optional
            Random number generator for reproducibility
            
        Returns
        -------
        bool
            True if order is filled, False otherwise
            
        Notes
        -----
        - Bid orders: distance = (microprice - quote_price) * 10000
        - Ask orders: distance = (quote_price - microprice) * 10000
        - Negative distance (aggressive quote) → very high fill probability
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Compute distance in basis points
        if side.lower() == 'bid':
            distance_bps = (microprice - quote_price) * 10000
        elif side.lower() == 'ask':
            distance_bps = (quote_price - microprice) * 10000
        else:
            raise ValueError(f"Invalid side: {side}. Must be 'bid' or 'ask'")
        
        # Aggressive quotes (cross the market) almost always fill
        # Bid above microprice or ask below microprice
        if distance_bps < -5.0:  # More than 5bp through the market
            return True
        
        # Compute fill probability and simulate
        fill_prob = self.compute_fill_probability(distance_bps)
        return rng.random() < fill_prob


def simulate_fill_batch(
    model: ParametricFillModel,
    bid_prices: np.ndarray,
    ask_prices: np.ndarray,
    microprices: np.ndarray,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized fill simulation for multiple time steps.
    
    Parameters
    ----------
    model : ParametricFillModel
        Fill model with calibrated parameters
    bid_prices : np.ndarray
        Bid limit order prices (shape: n_steps)
    ask_prices : np.ndarray
        Ask limit order prices (shape: n_steps)
    microprices : np.ndarray
        Microprice series (shape: n_steps)
    rng : np.random.Generator, optional
        Random number generator
        
    Returns
    -------
    bid_fills : np.ndarray
        Boolean array indicating bid fills
    ask_fills : np.ndarray
        Boolean array indicating ask fills
        
    Examples
    --------
    >>> model = ParametricFillModel(intensity_at_touch=2.0, decay_rate=0.5)
    >>> bids = np.array([100.00, 100.01, 100.02])
    >>> asks = np.array([100.02, 100.03, 100.04])
    >>> mids = np.array([100.01, 100.02, 100.03])
    >>> rng = np.random.default_rng(42)
    >>> bid_fills, ask_fills = simulate_fill_batch(model, bids, asks, mids, rng)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(bid_prices)
    bid_fills = np.zeros(n, dtype=bool)
    ask_fills = np.zeros(n, dtype=bool)
    
    # Compute distances in basis points
    bid_distances = (microprices - bid_prices) * 10000
    ask_distances = (ask_prices - microprices) * 10000
    
    # Compute intensities
    bid_intensities = model.intensity_at_touch * np.exp(-model.decay_rate * np.abs(bid_distances))
    ask_intensities = model.intensity_at_touch * np.exp(-model.decay_rate * np.abs(ask_distances))
    
    # Compute probabilities
    bid_probs = 1.0 - np.exp(-bid_intensities * model.time_step)
    ask_probs = 1.0 - np.exp(-ask_intensities * model.time_step)
    
    # Clip to [0, 1]
    bid_probs = np.clip(bid_probs, 0.0, 1.0)
    ask_probs = np.clip(ask_probs, 0.0, 1.0)
    
    # Simulate fills
    bid_fills = rng.random(n) < bid_probs
    ask_fills = rng.random(n) < ask_probs
    
    # Aggressive quotes always fill (more than 5bp through market)
    bid_fills[bid_distances < -5.0] = True
    ask_fills[ask_distances < -5.0] = True
    
    return bid_fills, ask_fills


def calibrate_intensity(
    nbbo_df: pd.DataFrame,
    bid_col: str = 'bid',
    ask_col: str = 'ask',
    mid_col: Optional[str] = None,
    time_step: float = 1.0,
    target_fill_rate_at_touch: float = 0.5
) -> ParametricFillModel:
    """
    Calibrate fill model parameters from historical NBBO data.
    
    This is a simplified calibration based on heuristics.
    More sophisticated approach would fit to actual trade data.
    
    Parameters
    ----------
    nbbo_df : pd.DataFrame
        Historical NBBO data
    bid_col : str
        Column name for bid price
    ask_col : str
        Column name for ask price
    mid_col : str, optional
        Column name for mid/microprice. If None, computed as (bid+ask)/2
    time_step : float
        Time step in seconds
    target_fill_rate_at_touch : float
        Desired fill rate when quoting at touch (per second)
        
    Returns
    -------
    ParametricFillModel
        Calibrated model
        
    Notes
    -----
    Heuristic calibration:
    - A (intensity_at_touch) set to achieve target_fill_rate
    - k (decay_rate) estimated from spread statistics
    - Wider spreads → slower decay (k smaller)
    - Tighter spreads → faster decay (k larger)
    """
    # Compute mid if not provided
    if mid_col is None:
        mid = (nbbo_df[bid_col] + nbbo_df[ask_col]) / 2.0
    else:
        mid = nbbo_df[mid_col]
    
    # Compute spread statistics
    spread_bps = (nbbo_df[ask_col] - nbbo_df[bid_col]) / mid * 10000
    avg_spread_bps = spread_bps.mean()
    
    # Heuristic: decay_rate inversely related to spread
    # Tight spreads (1-2 bps) → high decay (k ~ 1.0)
    # Wide spreads (5-10 bps) → low decay (k ~ 0.3)
    if avg_spread_bps < 2.0:
        decay_rate = 1.0
    elif avg_spread_bps < 5.0:
        decay_rate = 0.7
    elif avg_spread_bps < 10.0:
        decay_rate = 0.5
    else:
        decay_rate = 0.3
    
    # Convert target fill rate to intensity
    # P(fill) = 1 - exp(-λ * Δt)
    # λ = -ln(1 - P) / Δt
    if target_fill_rate_at_touch >= 1.0:
        intensity_at_touch = 10.0  # Cap at very high value
    else:
        intensity_at_touch = -np.log(1.0 - target_fill_rate_at_touch) / time_step
    
    return ParametricFillModel(
        intensity_at_touch=intensity_at_touch,
        decay_rate=decay_rate,
        time_step=time_step
    )


def compute_expected_fill_rate(
    model: ParametricFillModel,
    distance_bps: float
) -> float:
    """
    Compute expected fill rate (fills per second) at given distance.
    
    Parameters
    ----------
    model : ParametricFillModel
        Fill model
    distance_bps : float
        Distance from microprice in basis points
        
    Returns
    -------
    float
        Expected fills per second
    """
    intensity = model.compute_intensity(distance_bps)
    prob_per_step = 1.0 - np.exp(-intensity * model.time_step)
    fills_per_second = prob_per_step / model.time_step
    return fills_per_second

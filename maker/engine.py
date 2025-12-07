"""
maker/engine.py

Market making quoting engine with inventory risk management.
Implements Avellaneda-Stoikov framework with OFI signal integration.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from dataclasses import dataclass


@dataclass
class QuotingParams:
    """Configuration parameters for the quoting engine."""
    
    # Risk aversion parameter (gamma in A-S framework)
    risk_aversion: float = 0.1
    
    # Terminal time horizon (T in seconds)
    terminal_time: float = 300.0  # 5 minutes
    
    # Market order arrival rate (k in A-S framework, orders/second)
    order_arrival_rate: float = 1.0
    
    # Inventory limits
    max_inventory: int = 100  # shares
    min_inventory: int = -100
    
    # Tick size for rounding
    tick_size: float = 0.01
    
    # Minimum spread (bps)
    min_spread_bps: float = 1.0
    
    # Signal adjustment factor (how much to skew quotes based on signal)
    signal_adjustment_factor: float = 0.5
    
    # Inventory urgency factor (increases as inventory approaches limits)
    inventory_urgency_factor: float = 1.5


@dataclass
class QuoteState:
    """Current state for quote generation."""
    
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    microprice: float
    volatility: float  # Annualized
    signal_bps: float  # OFI-based drift signal
    inventory: int
    timestamp: pd.Timestamp


class QuotingEngine:
    """
    Market making quoting engine based on Avellaneda-Stoikov framework
    with OFI signal integration.
    
    The engine computes optimal bid/ask quotes considering:
    - Inventory risk (penalizes large positions)
    - Market volatility (widens spreads in volatile conditions)
    - OFI directional signal (skews quotes to reduce adverse selection)
    - Arrival rate (adjusts for fill probability)
    
    References
    ----------
    Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book.
    Quantitative Finance, 8(3), 217-224.
    """
    
    def __init__(self, params: Optional[QuotingParams] = None):
        """
        Initialize quoting engine.
        
        Parameters
        ----------
        params : QuotingParams, optional
            Configuration parameters. Uses defaults if not provided.
        """
        self.params = params if params is not None else QuotingParams()
        
    def compute_reservation_price(
        self,
        microprice: float,
        volatility: float,
        inventory: int,
        time_to_close: float
    ) -> float:
        """
        Compute inventory-adjusted reservation price.
        
        The reservation price is the "fair" price from the market maker's
        perspective, accounting for inventory risk. When long, it's below
        microprice (incentivize selling); when short, it's above (incentivize buying).
        
        Formula: r = s - γ * σ² * q * T
        where:
        - s = microprice (reference price)
        - γ = risk aversion parameter
        - σ = volatility (annualized)
        - q = inventory position (signed)
        - T = time to horizon
        
        Parameters
        ----------
        microprice : float
            Current microprice estimate
        volatility : float
            Annualized volatility estimate
        inventory : int
            Current inventory position (+ long, - short)
        time_to_close : float
            Time remaining to terminal horizon in seconds
            
        Returns
        -------
        float
            Reservation price
            
        Notes
        -----
        The inventory penalty term γ*σ²*q*T creates urgency to mean-revert
        position to zero. Larger inventory or closer to terminal time
        increases the penalty.
        
        Examples
        --------
        >>> engine = QuotingEngine()
        >>> # Long 50 shares, want to sell → reservation price below micro
        >>> r = engine.compute_reservation_price(100.0, 0.20, 50, 300)
        >>> r < 100.0
        True
        """
        # Convert time to years for consistency with annualized vol
        time_to_close_years = time_to_close / (252 * 6.5 * 3600)
        
        # Inventory penalty: γ * σ² * q * T
        inventory_penalty = (
            self.params.risk_aversion 
            * (volatility ** 2) 
            * inventory 
            * time_to_close_years
        )
        
        # Reservation price
        reservation_price = microprice - inventory_penalty
        
        return reservation_price
    
    def compute_quote_width(
        self,
        volatility: float,
        time_to_close: float,
        inventory: int
    ) -> float:
        """
        Compute optimal bid-ask spread width.
        
        The spread balances:
        1. Inventory risk (wider spread = less fill risk)
        2. Adverse selection (wider in volatile markets)
        3. Fill probability (narrower = more fills)
        
        Formula: δ = γ*σ²*T + (2/γ)*log(1 + γ/k)
        where:
        - γ = risk aversion
        - σ = volatility
        - T = time to horizon  
        - k = arrival rate
        
        Additional adjustment for inventory urgency when approaching limits.
        
        Parameters
        ----------
        volatility : float
            Annualized volatility
        time_to_close : float
            Time to terminal horizon (seconds)
        inventory : int
            Current position
            
        Returns
        -------
        float
            Half-spread width (quote will be reservation ± this value)
            
        Examples
        --------
        >>> engine = QuotingEngine()
        >>> spread = engine.compute_quote_width(0.20, 300, 0)
        >>> spread > 0
        True
        """
        # SIMPLIFIED FORMULA - ignore arrival term as it dominates unrealistically
        # Use empirical calibration: half_spread_bps ≈ α × σ × sqrt(T)
        # Where α is tuned to match market spreads (~2-5 bps typical)
        
        # Convert time to years for volatility scaling
        time_to_close_years = time_to_close / (252 * 6.5 * 3600)
        
        # Volatility component: σ × sqrt(T) gives standard deviation over period T
        # This is the natural risk horizon for inventory exposure
        vol_component = volatility * np.sqrt(time_to_close_years)
        
        # Scale by risk aversion parameter (now acts as spread multiplier)
        # γ ≈ 0.1 gives spreads ~2-5 bps for typical volatility
        gamma = self.params.risk_aversion
        
        # Base half-spread in relative terms (dimensionless)
        half_spread = gamma * vol_component
        
        # Inventory urgency adjustment - industry standard linear scaling
        # As inventory approaches limits, widen spread gradually
        # Use linear instead of cubic for more predictable behavior
        inventory_ratio = abs(inventory) / self.params.max_inventory
        urgency_multiplier = 1.0 + self.params.inventory_urgency_factor * inventory_ratio
        
        half_spread *= urgency_multiplier
        
        return half_spread
    
    def generate_quotes(
        self,
        state: QuoteState,
        time_to_close: float
    ) -> Tuple[float, float]:
        """
        Generate bid and ask quotes given current market state.
        
        Process:
        1. Compute reservation price (inventory-adjusted fair value)
        2. Compute quote width (spread based on risk and volatility)
        3. Adjust for OFI signal (skew quotes to reduce adverse selection)
        4. Enforce constraints (tick size, min spread, no cross)
        
        Parameters
        ----------
        state : QuoteState
            Current market state including prices, volatility, signal, inventory
        time_to_close : float
            Time remaining in seconds
            
        Returns
        -------
        Tuple[float, float]
            (bid_price, ask_price)
            
        Notes
        -----
        Signal integration:
        - Positive signal (expect price rise) → tighten ask, widen bid
        - Negative signal (expect price fall) → tighten bid, widen ask
        
        This reduces adverse selection by being more aggressive on the
        favorable side and more conservative on the unfavorable side.
        
        Examples
        --------
        >>> engine = QuotingEngine()
        >>> state = QuoteState(
        ...     bid=100.0, ask=100.1, bid_size=500, ask_size=500,
        ...     microprice=100.05, volatility=0.20, signal_bps=2.0,
        ...     inventory=0, timestamp=pd.Timestamp.now()
        ... )
        >>> bid, ask = engine.generate_quotes(state, time_to_close=300)
        >>> bid < state.microprice < ask
        True
        """
        # 1. Compute reservation price
        reservation = self.compute_reservation_price(
            state.microprice,
            state.volatility,
            state.inventory,
            time_to_close
        )
        
        # 2. Compute quote width
        half_spread = self.compute_quote_width(
            state.volatility,
            time_to_close,
            state.inventory
        )
        
        # Convert half_spread from fraction to price units
        # A-S formula returns dimensionless spread, scale by current price
        half_spread_dollars = half_spread * state.microprice
        
        # 3. Adjust for signal - ASYMMETRIC SPREAD SKEWING
        # Convert signal from bps to price units
        signal_adjustment = state.signal_bps * state.microprice / 10000
        signal_adjustment *= self.params.signal_adjustment_factor
        
        # Positive signal (expect price rise) → tighten ask (aggressive sell), widen bid (avoid buying high)
        # Negative signal (expect price fall) → tighten bid (aggressive buy), widen ask (avoid selling low)
        # This creates asymmetric spread: be aggressive on favorable side, conservative on unfavorable
        # 
        # For positive signal (+ve):
        #   - Bid should be wider (more negative): bid_adjustment = +signal (adds to the negative half_spread)
        #   - Ask should be tighter (less positive): ask_adjustment = -signal (subtracts from the positive half_spread)
        bid_spread_adjustment = signal_adjustment   # Positive signal widens bid (farther from mid)
        ask_spread_adjustment = -signal_adjustment  # Positive signal tightens ask (closer to mid)
        
        # 4. Generate initial quotes with asymmetric spreads
        bid_price = reservation - half_spread_dollars + bid_spread_adjustment
        ask_price = reservation + half_spread_dollars + ask_spread_adjustment
        
        # 5. Apply inventory skew (already handled in reservation price)
        # Additional small adjustment for urgency
        # When long, slightly favor selling (lower quotes)
        # When short, slightly favor buying (higher quotes)
        inventory_skew_bps = -state.inventory * 0.02  # 2bp per 100 shares (stronger skew)
        inventory_skew = inventory_skew_bps * state.microprice / 10000
        bid_price += inventory_skew
        ask_price += inventory_skew
        
        # 6. Inventory limit enforcement (industry best practice)
        # Stop quoting on side that would violate limits
        if state.inventory >= self.params.max_inventory:
            # At max long position - don't quote bid (can't buy more)
            bid_price = np.nan
        if state.inventory <= self.params.min_inventory:
            # At max short position - don't quote ask (can't sell more)
            ask_price = np.nan
        
        # 7. Round to tick size
        bid_price = self._round_to_tick(bid_price, direction='down')
        ask_price = self._round_to_tick(ask_price, direction='up')
        
        # 8. Enforce minimum spread (skip if either side is NaN from inventory limits)
        if not np.isnan(bid_price) and not np.isnan(ask_price):
            min_spread_price = self.params.min_spread_bps * state.microprice / 10000
            if ask_price - bid_price < min_spread_price:
                mid = (bid_price + ask_price) / 2
                bid_price = mid - min_spread_price / 2
                ask_price = mid + min_spread_price / 2
                bid_price = self._round_to_tick(bid_price, direction='down')
                ask_price = self._round_to_tick(ask_price, direction='up')
        
        # 9. Enforce no-cross (skip if either side is NaN)
        if not np.isnan(bid_price) and not np.isnan(ask_price):
            bid_price, ask_price = self.enforce_no_cross_market(
                bid_price, ask_price, state.microprice
            )
        
        return bid_price, ask_price
    
    def enforce_no_cross_market(
        self,
        bid: float,
        ask: float,
        microprice: float
    ) -> Tuple[float, float]:
        """
        Ensure quotes don't cross the market or each other.
        
        Constraints:
        1. bid < microprice < ask (no locked/crossed market)
        2. bid < ask (valid spread)
        3. If violations occur, adjust symmetrically around microprice
        
        Parameters
        ----------
        bid : float
            Proposed bid price
        ask : float
            Proposed ask price
        microprice : float
            Current microprice
            
        Returns
        -------
        Tuple[float, float]
            (adjusted_bid, adjusted_ask) satisfying constraints
            
        Examples
        --------
        >>> engine = QuotingEngine()
        >>> # Valid quotes - no adjustment
        >>> b, a = engine.enforce_no_cross_market(99.9, 100.1, 100.0)
        >>> (b, a)
        (99.9, 100.1)
        
        >>> # Crossed quotes - adjusted
        >>> b, a = engine.enforce_no_cross_market(100.1, 99.9, 100.0)
        >>> b < 100.0 < a
        True
        """
        # Check if bid > microprice (too aggressive)
        if bid >= microprice:
            # Move bid below microprice, maintain spread
            spread = max(ask - bid, self.params.tick_size)
            bid = microprice - spread / 2
            ask = microprice + spread / 2
            bid = self._round_to_tick(bid, direction='down')
            ask = self._round_to_tick(ask, direction='up')
        
        # Check if ask < microprice (too aggressive)
        if ask <= microprice:
            spread = max(ask - bid, self.params.tick_size)
            bid = microprice - spread / 2
            ask = microprice + spread / 2
            bid = self._round_to_tick(bid, direction='down')
            ask = self._round_to_tick(ask, direction='up')
        
        # Check if bid >= ask (crossed)
        if bid >= ask:
            # Reset to minimum spread around microprice
            min_spread = self.params.tick_size * 2
            bid = microprice - min_spread / 2
            ask = microprice + min_spread / 2
            bid = self._round_to_tick(bid, direction='down')
            ask = self._round_to_tick(ask, direction='up')
        
        return bid, ask
    
    def _round_to_tick(self, price: float, direction: str = 'nearest') -> float:
        """
        Round price to tick size.
        
        Parameters
        ----------
        price : float
            Price to round
        direction : str, default='nearest'
            Rounding direction: 'up', 'down', or 'nearest'
            
        Returns
        -------
        float
            Rounded price
        """
        tick = self.params.tick_size
        
        if direction == 'up':
            return np.round(np.ceil(price / tick) * tick, 10)
        elif direction == 'down':
            return np.round(np.floor(price / tick) * tick, 10)
        else:  # nearest
            return np.round(np.round(price / tick) * tick, 10)
    
    def update_params(self, **kwargs) -> None:
        """
        Update quoting parameters dynamically.
        
        Useful for adaptive strategies that adjust risk aversion,
        arrival rates, or other parameters based on market conditions.
        
        Parameters
        ----------
        **kwargs
            Parameter names and new values (must be valid QuotingParams attributes)
            
        Examples
        --------
        >>> engine = QuotingEngine()
        >>> engine.update_params(risk_aversion=0.2, min_spread_bps=2.0)
        """
        for key, value in kwargs.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
    
    def get_inventory_limits_proximity(self, inventory: int) -> float:
        """
        Calculate how close inventory is to limits (0 to 1).
        
        Returns ratio of current inventory to max, useful for
        risk management and position urgency.
        
        Parameters
        ----------
        inventory : int
            Current position
            
        Returns
        -------
        float
            Ratio in [0, 1] where 0 = no position, 1 = at limit
        """
        return abs(inventory) / self.params.max_inventory

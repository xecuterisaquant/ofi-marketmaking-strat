# System Architecture

## Overview

This document provides a detailed technical architecture of the OFI-driven market making system.

## System Components

### 1. Data Layer (`src/ofi_utils.py`)

**Purpose**: Load and preprocess market data from RDA files.

**Key Functions**:
```python
load_nbbo_data(date: str) -> pd.DataFrame
    # Loads NBBO snapshots for a given date
    # Returns: DataFrame with columns [time, bid, ask, bid_size, ask_size]
    
load_trade_data(date: str) -> pd.DataFrame
    # Loads trade executions for a given date
    # Returns: DataFrame with columns [time, price, size]
```

**Data Flow**:
```
RDA Files → R Loader → Pandas DataFrame → Feature Engineering
```

---

### 2. Feature Engineering Layer (`maker/features.py`)

**Purpose**: Compute trading signals from raw market data.

#### 2.1 OFI Computation

**Mathematical Definition**:

$$
\text{OFI}_t = \Delta Q^{\text{bid}}_t - \Delta Q^{\text{ask}}_t
$$

**Implementation**:
```python
class OFICalculator:
    def compute_ofi(self, bid, ask, bid_size, ask_size):
        # Compute changes in liquidity
        delta_bid = self._compute_delta(bid, bid_size)
        delta_ask = self._compute_delta(ask, ask_size)
        return delta_bid - delta_ask
    
    def _compute_delta(self, price, size):
        # Tracks additions/removals at each price level
        if price_improved:
            return +size  # New liquidity added
        elif price_worsened:
            return -size  # Liquidity removed
        else:
            return size.diff()  # Size change at same price
```

**Normalization**:

$$
\text{OFI}^{\text{norm}}_t = \frac{\text{OFI}_t}{\sigma_{\text{OFI}}}
$$

Where $\sigma_{\text{OFI}}$ is the rolling standard deviation over the past 60 seconds.

#### 2.2 Volatility Estimation

**Method**: Rolling standard deviation of mid-price returns.

$$
\sigma_t = \sqrt{\frac{1}{N-1} \sum_{i=t-N}^{t} (r_i - \bar{r})^2}
$$

Where:
- $r_i = \log(m_i / m_{i-1})$ is the log-return
- $N = 60$ seconds (rolling window)
- $m_i$ is the mid-price at time $i$

**Implementation**:
```python
class VolatilityEstimator:
    def __init__(self, window_seconds=60):
        self.window = window_seconds
    
    def estimate(self, mid_prices, times):
        returns = np.log(mid_prices).diff()
        rolling_std = returns.rolling(window=self.window).std()
        return rolling_std.fillna(method='bfill')
```

#### 2.3 Microprice

**Definition**: Volume-weighted average of best bid and ask.

$$
p^{\text{micro}}_t = \frac{Q^{\text{ask}}_t \cdot p^{\text{bid}}_t + Q^{\text{bid}}_t \cdot p^{\text{ask}}_t}{Q^{\text{bid}}_t + Q^{\text{ask}}_t}
$$

**Purpose**: More accurate price estimate than simple mid-price, accounting for liquidity imbalance.

---

### 3. Quoting Engine (`maker/engine.py`)

**Purpose**: Generate optimal bid/ask quotes using the Avellaneda-Stoikov framework.

#### 3.1 Reservation Price

**Formula**:

$$
r_t = S_t - q_t \cdot \gamma \cdot \sigma^2 \cdot (T - t) + \kappa \cdot \text{OFI}^{\text{norm}}_t
$$

Where:
- $S_t$ = Current mid-price (or microprice)
- $q_t$ = Current inventory position
- $\gamma$ = Risk aversion parameter (0.1)
- $\sigma$ = Volatility estimate
- $T - t$ = Time remaining (we use constant 3600s)
- $\kappa$ = OFI sensitivity (0.001)

**Components**:
1. **Base Price** ($S_t$): Current market price
2. **Inventory Penalty** ($ \-q \gamma \sigma^2 (T-t)$): Makes reservation price less favorable when holding inventory
3. **OFI Skew** ($\kappa \cdot \text{OFI}^{\text{norm}}_t$): Shifts price based on order flow imbalance

**Implementation**:
```python
def compute_reservation_price(
    self,
    mid_price: float,
    inventory: int,
    volatility: float,
    ofi_normalized: float = 0.0,
    time_remaining: float = 3600.0
) -> float:
    # Inventory penalty
    inventory_penalty = (
        inventory * self.risk_aversion * volatility**2 * time_remaining
    )
    
    # OFI skew
    ofi_skew = self.ofi_sensitivity * ofi_normalized
    
    # Reservation price
    reservation = mid_price - inventory_penalty + ofi_skew
    
    return reservation
```

#### 3.2 Optimal Spread

**Formula**:

$$
\delta_t = \gamma \sigma^2 (T - t) + \frac{2}{\gamma} \log(1 + \gamma / \lambda) + \eta \cdot |\text{OFI}^{\text{norm}}_t|
$$

Where:
- First term: volatility-based spread
- Second term: adverse selection protection
- Third term: OFI-based spread widening ($\eta = 0.5$)

**Implementation**:
```python
def compute_optimal_spread(
    self,
    volatility: float,
    ofi_normalized: float = 0.0,
    time_remaining: float = 3600.0
) -> float:
    # Volatility component
    vol_spread = self.risk_aversion * volatility**2 * time_remaining
    
    # Adverse selection component
    adverse_selection = (2 / self.risk_aversion) * np.log(
        1 + self.risk_aversion / self.fill_intensity
    )
    
    # OFI spread widening
    ofi_spread = self.spread_multiplier * abs(ofi_normalized)
    
    # Total spread (capped)
    spread = vol_spread + adverse_selection + ofi_spread
    return min(spread, self.max_spread)
```

#### 3.3 Quote Generation

**Formulas**:

$$
p^{\text{bid}}_t = r_t - \delta_t / 2
$$

$$
p^{\text{ask}}_t = r_t + \delta_t / 2
$$

Where:
- $r_t$ = Reservation price
- $\delta_t$ = Optimal spread

**Quote Validity**:
```python
def generate_quotes(self, ...):
    reservation = self.compute_reservation_price(...)
    spread = self.compute_optimal_spread(...)
    
    bid = reservation - spread / 2
    ask = reservation + spread / 2
    
    # Ensure quotes don't cross market
    bid = min(bid, market_bid)
    ask = max(ask, market_ask)
    
    # Ensure minimum spread
    if ask - bid < self.min_spread:
        ask = bid + self.min_spread
    
    return bid, ask
```

---

### 4. Fill Simulation (`maker/fills.py`)

**Purpose**: Simulate trade executions using parametric intensity models.

#### 4.1 Fill Intensity

**Formula**:

$$
\lambda^{\text{bid}}_t = \lambda_0 \cdot \exp\left(-\alpha \cdot \frac{p^{\text{bid}}_t - p^{\text{market,bid}}_t}{p^{\text{market,mid}}_t}\right)
$$

Where:
- $\lambda_0$ = Base fill intensity (1.0)
- $\alpha$ = Price sensitivity (100.0)
- Numerator = How much worse our quote is than the market

**Interpretation**:
- Better quote (closer to mid) → Higher intensity → More fills
- Worse quote (farther from mid) → Lower intensity → Fewer fills

#### 4.2 Fill Probability

**Per-Second Probability**:

$$
P(\text{fill}) = 1 - \exp(-\lambda_t \cdot \Delta t)
$$

Where $\Delta t = 1$ second (our time granularity).

**Implementation**:
```python
def simulate_fill(
    self,
    our_bid: float,
    our_ask: float,
    market_bid: float,
    market_ask: float,
    market_mid: float
) -> Dict[str, Any]:
    # Compute fill intensities
    bid_intensity = self._compute_intensity(
        our_bid, market_bid, market_mid, side='bid'
    )
    ask_intensity = self._compute_intensity(
        our_ask, market_ask, market_mid, side='ask'
    )
    
    # Fill probabilities
    bid_prob = 1 - np.exp(-bid_intensity * self.dt)
    ask_prob = 1 - np.exp(-ask_intensity * self.dt)
    
    # Simulate fills
    bid_fill = np.random.random() < bid_prob
    ask_fill = np.random.random() < ask_prob
    
    return {
        'bid_fill': bid_fill,
        'ask_fill': ask_fill,
        'bid_intensity': bid_intensity,
        'ask_intensity': ask_intensity
    }
```

---

### 5. Backtest Engine (`maker/backtest.py`)

**Purpose**: Event-driven simulation of market making strategy.

#### 5.1 State Management

**State Variables**:
```python
class BacktestState:
    inventory: int = 0          # Current position (shares)
    cash: float = 0.0           # Cash balance ($)
    pnl: float = 0.0            # Mark-to-market PnL ($)
    fills: List[Fill] = []      # Fill history
    quotes: List[Quote] = []    # Quote history
```

**PnL Calculation**:

$$
\text{PnL}_t = \text{Cash}_t + Q_t \cdot S_t
$$

Where:
- $\text{Cash}_t$ = Cumulative cash from trades
- $Q_t$ = Current inventory
- $S_t$ = Current mid-price (mark-to-market)

#### 5.2 Event Loop

**Pseudocode**:
```python
def run_backtest(data, strategy):
    state = BacktestState()
    
    for t in range(len(data)):
        # 1. Compute features
        ofi = compute_ofi(data[:t])
        volatility = compute_volatility(data[:t])
        
        # 2. Generate quotes
        bid, ask = strategy.generate_quotes(
            mid=data.mid[t],
            inventory=state.inventory,
            volatility=volatility,
            ofi=ofi
        )
        
        # 3. Simulate fills
        fills = simulate_fills(
            our_bid=bid,
            our_ask=ask,
            market_bid=data.bid[t],
            market_ask=data.ask[t]
        )
        
        # 4. Update state
        if fills.bid_fill:
            state.inventory += 100
            state.cash -= bid * 100
        if fills.ask_fill:
            state.inventory -= 100
            state.cash += ask * 100
        
        # 5. Compute PnL
        state.pnl = state.cash + state.inventory * data.mid[t]
        
        # 6. Log state
        log_state(t, state, bid, ask)
    
    return state
```

---

## Data Flow Diagram

```
┌──────────────┐
│  RDA Files   │
│  (NBBO/Trade)│
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│    Data Loading Layer        │
│  • Load NBBO snapshots       │
│  • Load trade data           │
│  • Align timestamps          │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Feature Engineering Layer   │
│  • Compute OFI               │
│  • Estimate volatility       │
│  • Calculate microprice      │
│  • Normalize signals         │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│     Quoting Engine Layer     │
│  • Reservation price (AS)    │
│  • Optimal spread            │
│  • OFI-based adjustments     │
│  • Quote validation          │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│   Fill Simulation Layer      │
│  • Intensity calculation     │
│  • Fill probability          │
│  • Random fill generation    │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│    Backtest Engine Layer     │
│  • Event loop                │
│  • State management          │
│  • PnL tracking              │
│  • Metrics computation       │
└──────┬───────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│      Results Output          │
│  • Detailed parquet files    │
│  • Summary statistics        │
│  • Performance metrics       │
└──────────────────────────────┘
```

---

## Parameter Configuration

### Global Parameters

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| Risk Aversion | $\gamma$ | 0.1 | Avellaneda & Stoikov (2008) |
| Fill Intensity | $\lambda_0$ | 1.0 | Calibrated |
| Time Horizon | $T$ | 3600s | Standard |
| Volatility Window | - | 60s | Common practice |

### OFI-Specific Parameters

| Parameter | Symbol | Value | Source |
|-----------|--------|-------|--------|
| OFI Sensitivity | $\kappa$ | 0.001 | Calibrated |
| Spread Multiplier | $\eta$ | 0.5 | Calibrated |
| OFI Normalization | - | Rolling std (60s) | Statistical |

### Constraints

| Constraint | Value | Reason |
|------------|-------|--------|
| Min Spread | 1 bps | Realistic minimum |
| Max Spread | 50 bps | Prevent extreme widening |
| Max Inventory | ±1000 shares | Risk management |

---

## Performance Considerations

### Computational Complexity

- **OFI Computation**: O(n) per time step
- **Volatility Estimation**: O(w) where w = window size
- **Quote Generation**: O(1) per time step
- **Fill Simulation**: O(1) per time step
- **Total Backtest**: O(n) where n = number of time steps

### Memory Usage

- **NBBO Data**: ~5MB per symbol-day
- **Trade Data**: ~2MB per symbol-day
- **Feature Cache**: ~1MB per symbol-day
- **Results**: ~500KB per backtest

### Optimization Opportunities

1. **Vectorization**: Use NumPy for batch operations
2. **Caching**: Store computed features
3. **Parallelization**: Run multiple backtests in parallel
4. **Data Structures**: Use efficient containers (deque for rolling windows)

---

## Testing Strategy

### Unit Tests (141 total)

1. **Feature Tests** (27): OFI computation, volatility, microprice
2. **Engine Tests** (25): Quote generation, inventory management
3. **Fill Tests** (26): Intensity calculation, fill simulation
4. **Backtest Tests** (24): Event loop, state management
5. **Metrics Tests** (39): Performance calculation, anti-overfitting

### Integration Tests

- End-to-end backtest workflows
- Multi-day simulation
- Strategy comparison

### Validation Tests

- Parameter sensitivity analysis
- Regime testing (high/low volatility)
- Edge case handling

---

## Future Enhancements

### Short-term

1. **Multi-level OFI**: Incorporate deeper order book levels
2. **Adaptive Parameters**: Dynamically adjust γ, κ based on market conditions
3. **Transaction Costs**: Add explicit fee modeling

### Long-term

1. **Multi-asset**: Extend to portfolio market making
2. **Machine Learning**: Neural network for OFI prediction
3. **Real-time Deployment**: Live trading infrastructure
4. **Execution Optimization**: TWAP/VWAP for inventory unwinds

---

## References

See main README for academic references and citations.

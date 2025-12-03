# Full Context Prompt for OFI Market Making Strategy Project

## Project Overview

Build an **OFI-driven market making strategy** that extends a completed replication of *Cont, Kukanov & Stoikov (2014): "The Price Impact of Order Book Events"*. The replication established that Order Flow Imbalance (OFI) predicts short-term price movements with:
- Mean R² = 8.1% across 40 symbol-days (8 symbols × 5 days in January 2017)
- 100% positive beta coefficients (mean β = 0.036)
- Strong statistical significance (95%+ of regressions)

**Goal**: Leverage these OFI signals to build a practical market maker that reduces adverse selection and improves profitability.

---

## Current Progress (Phases 0-2 Complete)

### ✅ Phase 0: Repository Setup
**GitHub**: `https://github.com/xecuterisaquant/ofi-marketmaking-strat`

**Structure created**:
```
ofi-marketmaking-strat/
├── maker/                  # Market making modules
│   ├── __init__.py
│   ├── features.py        # ✅ COMPLETE (406 lines, 27 tests passing)
│   └── engine.py          # ✅ COMPLETE (465 lines, 25 tests passing)
├── src/                   # Infrastructure from replication
│   ├── __init__.py
│   └── ofi_utils.py       # ✅ Ported (245 lines) - OFI calculation, NBBO handling
├── scripts/               # Executable scripts (TO DO)
├── tests/                 # Unit tests
│   ├── test_features.py   # ✅ COMPLETE (427 lines, 27 passing)
│   └── test_engine.py     # ✅ COMPLETE (427 lines, 25 passing)
├── configs/               # Strategy configurations (TO DO)
├── data/raw/              # TAQ .rda files (2 sample files copied)
├── results/               # Backtest outputs (TO DO)
└── figures/               # Plots (TO DO)
```

**Files in place**:
- `README.md` - Full project documentation
- `requirements.txt` - Dependencies (pandas, numpy, scipy, statsmodels, etc.)
- `LICENSE` - MIT
- `.gitignore` - Standard Python + data exclusions

---

### ✅ Phase 1: Feature Engineering (`maker/features.py`)

**All functions implemented and tested (27/27 tests passing)**:

1. **`compute_ofi_signal(ofi_normalized, beta=0.036, horizon_seconds=60)`**
   - Converts normalized OFI → expected drift signal in basis points
   - Formula: `signal_bps = ofi_normalized * beta * 100`
   - Default β = 0.036 (mean from replication)
   - Returns: pd.Series of drift signals

2. **`compute_microprice(bid, ask, bid_size, ask_size)`**
   - Depth-weighted midpoint: `(P_ask * Q_bid + P_bid * Q_ask) / (Q_bid + Q_ask)`
   - More informative than simple mid when book is imbalanced
   - Handles edge cases: zero depth → falls back to midprice
   - Returns: pd.Series

3. **`compute_ewma_volatility(prices, halflife_seconds=60.0, min_periods=10)`**
   - Exponentially weighted moving average volatility from squared returns
   - Process: log returns → square → EWMA → sqrt → annualize
   - Annualization: sqrt(252 * 6.5 * 3600) ≈ 2,428 for 1-second data
   - Returns: pd.Series (annualized volatility)

4. **`compute_imbalance(bid_size, ask_size)`**
   - Depth imbalance: `(Q_bid - Q_ask) / (Q_bid + Q_ask)`
   - Range: [-1, 1] where +1 = all buying pressure, -1 = all selling
   - Returns: pd.Series

5. **`blend_signals(ofi_signal, imbalance, alpha_ofi=0.7, alpha_imbalance=0.3, additional_signals=None, additional_alphas=None)`**
   - Weighted combination of signals
   - Default: 70% OFI, 30% imbalance (OFI is primary predictive signal)
   - Extensible for momentum, spread, or other signals
   - Returns: pd.Series (composite signal)

6. **`compute_signal_stats(signal, window_seconds=300)`**
   - Rolling statistics: mean, std, min, max, percentiles
   - Used for monitoring and threshold setting
   - Returns: pd.DataFrame

**Key validations**:
- All functions preserve input index (timestamps)
- Handle both pd.Series and np.ndarray inputs
- Edge cases tested (zero depth, NaN values, constant signals)
- Mathematical correctness verified with synthetic data

---

### ✅ Phase 2: Quoting Engine (`maker/engine.py`)

**Avellaneda-Stoikov framework with OFI integration (25/25 tests passing)**:

#### **Classes**:

1. **`QuotingParams`** (dataclass)
   - `risk_aversion: float = 0.1` (γ in A-S framework)
   - `terminal_time: float = 300.0` (5 minutes)
   - `order_arrival_rate: float = 1.0` (k, orders/second)
   - `max_inventory: int = 100` shares
   - `min_inventory: int = -100`
   - `tick_size: float = 0.01`
   - `min_spread_bps: float = 1.0`
   - `signal_adjustment_factor: float = 0.5`
   - `inventory_urgency_factor: float = 1.5`

2. **`QuoteState`** (dataclass)
   - Current market state: bid, ask, bid_size, ask_size, microprice
   - volatility (annualized), signal_bps (OFI-based)
   - inventory (current position), timestamp

3. **`QuotingEngine`**
   - Main market making logic

#### **Methods**:

1. **`compute_reservation_price(microprice, volatility, inventory, time_to_close)`**
   - Inventory-adjusted fair value
   - Formula: `r = s - γ * σ² * q * T`
   - Long position → r < microprice (want to sell)
   - Short position → r > microprice (want to buy)
   - Returns: float

2. **`compute_quote_width(volatility, time_to_close, inventory)`**
   - Optimal bid-ask spread
   - Formula: `δ = γ*σ²*T + (2/γ)*log(1 + γ/k)`
   - Wider in high volatility
   - Wider near inventory limits (cubic urgency: `1 + 1.5 * (|q|/q_max)³`)
   - Returns: float (half-spread)

3. **`generate_quotes(state: QuoteState, time_to_close)`**
   - Complete quote generation pipeline:
     1. Compute reservation price
     2. Compute quote width
     3. Adjust for OFI signal (positive → shift up, negative → shift down)
     4. Apply inventory skew (1bp per 100 shares)
     5. Round to tick size
     6. Enforce minimum spread
     7. Enforce no-cross market
   - Returns: (bid_price, ask_price)

4. **`enforce_no_cross_market(bid, ask, microprice)`**
   - Ensures bid < microprice < ask
   - Prevents locked/crossed markets
   - Adjusts quotes symmetrically around microprice if violated
   - Returns: (adjusted_bid, adjusted_ask)

5. **`update_params(**kwargs)`**
   - Dynamic parameter updates (useful for adaptive strategies)

6. **`get_inventory_limits_proximity(inventory)`**
   - Returns ratio in [0, 1]: 0 = no position, 1 = at limit

**Key behaviors validated**:
- Zero inventory → symmetric quotes
- Long inventory → quotes shift down, spread widens
- Short inventory → quotes shift up, spread widens  
- Positive OFI signal → quotes shift up (expect price rise)
- Negative OFI signal → quotes shift down (expect price fall)
- High volatility → wider spreads
- Tick rounding precision (handles floating point errors)

---

## Data Available

**Source**: TAQ NBBO quote data from WRDS

**Symbols**: AAPL, AMD, AMZN, JPM, MSFT, NVDA, SPY, TSLA (8 total)

**Days**: January 2017 (20 trading days, 3-31)
- Currently have 2 sample files: `2017-01-03.rda`, `2017-01-04.rda`
- Full month available in replication repository

**Format**: R data files (.rda) with columns:
- `sym_root` / `symbol` / `sym` (symbol identifier)
- `time_m` (milliseconds since midnight, ET)
- `best_bid`, `best_ask` (NBBO prices)
- `best_bidsiz`, `best_asksiz` (NBBO sizes, in round lots)

**Preprocessing done by `src/ofi_utils.py`**:
- Timestamp normalization to America/New_York
- RTH filtering (09:30-16:00)
- 1-second resampling with forward-fill
- Crossed quote removal (ask < bid)
- OFI calculation using CK&S rules

---

## Next Phases (TO DO)

### Phase 3: Fill Simulation (`maker/fills.py`)

**Objective**: Model when limit orders get filled

**Two approaches**:

1. **Queue-reactive model** (if trades available):
   - Track position in queue: `A_t` = shares ahead of us
   - Update on market orders (consume queue), cancels (modify queue)
   - Fill when `A_t <= 0`

2. **Parametric model** (NBBO-only):
   - Fill intensity: `λ(δ) = A * exp(-k * δ)` where δ = distance from mid
   - Calibrate A, k from historical data
   - Simulate fills stochastically

**Functions needed**:
- `QueuePosition` class (track state, update on events)
- `ParametricFillModel` class (intensity calculation)
- `simulate_fill(quote, nbbo_state, model)` → bool (filled or not)

**Tests**: 
- Queue decreases on market orders
- Fill probability decreases with distance from mid
- Zero fill at large distances

---

### Phase 4: Backtest Framework (`maker/backtest.py`)

**Objective**: Event-driven simulation loop

**Components**:

1. **`BacktestEngine`** class:
   - Initialize: load data, set parameters, create engine
   - `run_day(symbol, date)`:
     - Loop over 1-second timestamps (09:30-16:00)
     - Update features (OFI, microprice, volatility, imbalance)
     - Generate quotes using `QuotingEngine`
     - Simulate fills using fill model
     - Update inventory, cash, P&L
   - Track: fills, inventory history, quote history, P&L path

2. **Order lifecycle**:
   - Place limit orders (bid/ask) each second
   - Cancel previous orders (if not filled)
   - Check for fills
   - Update position

3. **Inventory management**:
   - Track `inventory_t`, `cash_t`
   - Mark-to-market P&L: `pnl_t = cash_t + inventory_t * mid_t`
   - Enforce inventory limits (stop trading if at max)

**Functions**:
- `BacktestEngine.__init__(params, fill_model, data_loader)`
- `run_single_day(symbol, date)` → BacktestResult
- `run_multi_day(symbols, dates)` → List[BacktestResult]

---

### Phase 5: Performance Metrics (`maker/metrics.py`)

**Objective**: Evaluate strategy performance

**Metrics to compute**:

1. **P&L metrics**:
   - Total P&L, Sharpe ratio (annualized)
   - Max drawdown (% and $)
   - Sortino ratio (downside deviation)

2. **Fill metrics**:
   - Fill ratio (filled orders / total orders)
   - Average edge per fill: `(mid - fill_price) * sign(side)`
   - Fill size distribution

3. **Inventory metrics**:
   - Inventory variance
   - Time at inventory limits
   - Average absolute inventory

4. **Adverse selection**:
   - Post-fill price movement (1s, 5s, 10s horizons)
   - Cost: `(mid_t+k - fill_price) * sign(side)`

5. **Signal effectiveness**:
   - Correlation(OFI signal, subsequent returns)
   - Signal accuracy (correct direction %)

**Functions**:
- `compute_sharpe(pnl_series, periods_per_year=23400)` (6.5hrs * 3600s)
- `compute_max_drawdown(pnl_series)`
- `compute_fill_edge(fills_df)`
- `compute_adverse_selection(fills_df, price_series, horizons=[1,5,10])`
- `generate_performance_report(backtest_results)` → dict

---

### Phase 6: Baseline Strategies

**Objective**: Compare OFI-driven maker against baselines

**Strategies to implement**:

1. **Symmetric Baseline**:
   - No signal skew
   - Quotes centered on mid
   - Width = volatility-based only
   - Config: `configs/symmetric_baseline.yaml`

2. **Microprice Only**:
   - Skew based on microprice tilt only
   - No OFI integration
   - Config: `configs/microprice_only.yaml`

3. **OFI Full** (main strategy):
   - OFI signal + microprice + inventory skew
   - Full feature set
   - Config: `configs/ofi_full.yaml`

4. **OFI Ablations**:
   - OFI only (no microprice)
   - Different OFI windows (5min, 10min, 15min normalization)
   - Different signal weights

**Implementation**:
- YAML config files with parameter overrides
- `StrategyFactory.create(config)` → QuotingEngine

---

### Phase 7: Execution Scripts (`scripts/`)

**Scripts needed**:

1. **`run_maker_backtest.py`**:
   - CLI interface for running backtests
   - Args: `--symbols`, `--days`, `--config`, `--output-dir`
   - Parallel execution for multiple symbol-days
   - Save results to Parquet

2. **`generate_report.py`**:
   - Load backtest results
   - Compute all metrics
   - Generate comparison plots:
     - P&L curves (strategy comparison)
     - Inventory time series
     - Fill edge distribution
     - Signal vs returns scatter
   - Export PDF report

3. **`calibrate_fill_model.py`** (if using parametric):
   - Estimate λ(δ) parameters from historical data
   - Cross-validation
   - Save calibrated model

**Example usage**:
```bash
# Run baseline comparison
python scripts/run_maker_backtest.py \
    --data-dir data/raw \
    --config configs/ofi_full.yaml \
    --symbols AAPL AMD NVDA \
    --days 2017-01-03 2017-01-04 2017-01-05 \
    --output results/ofi_test

# Generate report
python scripts/generate_report.py \
    --results-dir results/ofi_test \
    --output reports/ofi_test_report.pdf
```

---

### Phase 8: Testing & Validation

**Test coverage needed**:

1. **Unit tests** (already started):
   - ✅ `test_features.py` (27 tests)
   - ✅ `test_engine.py` (25 tests)
   - TODO: `test_fills.py` (queue model, parametric model)
   - TODO: `test_backtest.py` (order lifecycle, P&L calculation)
   - TODO: `test_metrics.py` (Sharpe, drawdown, edge)

2. **Integration tests**:
   - End-to-end backtest on synthetic data
   - Multi-day simulation
   - Strategy comparison

3. **Validation checks**:
   - P&L reconciliation (cash + inventory = total)
   - No impossible fills (quotes must be valid)
   - Inventory limits respected
   - Quote constraints enforced

**Target**: 90%+ test coverage, all edge cases handled

---

### Phase 9: Optimization & Analysis

**Experiments to run**:

1. **Parameter sweeps**:
   - Risk aversion γ ∈ [0.05, 0.1, 0.2, 0.5]
   - Signal weights α_ofi ∈ [0.5, 0.7, 0.9]
   - OFI normalization windows [300s, 600s, 900s]

2. **Frequency analysis**:
   - Compare 1s vs 500ms vs 100ms sampling
   - Trade-off: more fills vs more adverse selection

3. **Symbol analysis**:
   - Which symbols benefit most from OFI?
   - Correlation with liquidity (spread, depth)

4. **Regime analysis**:
   - Performance in high/low volatility
   - Morning vs afternoon

**Deliverable**: Optimization report with best parameters

---

### Phase 10: Documentation & Presentation

**Deliverables**:

1. **Technical report** (PDF):
   - Strategy description
   - Methodology (A-S + OFI)
   - Backtest results with tables/figures
   - Baseline comparisons
   - Key findings and recommendations

2. **README updates**:
   - Complete installation instructions
   - Usage examples
   - Results summary

3. **Code documentation**:
   - Docstrings for all public functions
   - Inline comments for complex logic
   - Architecture diagram

4. **Presentation slides** (optional):
   - 10-15 slides for TA/professor
   - Key results, plots, conclusions

---

## Technical Specifications

### Development Environment
- **OS**: Windows (PowerShell)
- **Python**: 3.13.0
- **IDE**: VS Code with GitHub Copilot

### Dependencies (requirements.txt)
```
pandas>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
statsmodels>=0.14.0
pyarrow>=14.0.0
pyreadr>=0.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyyaml>=6.0
pytest>=7.0.0
```

### Coding Guidelines

1. **Style**:
   - Type hints for function signatures
   - Google-style docstrings
   - Max line length: 100 characters
   - f-strings for formatting

2. **Organization**:
   - One class per file for major components
   - Keep functions small (<50 lines)
   - Separate business logic from I/O

3. **Performance**:
   - Vectorize with pandas/numpy where possible
   - Avoid loops over DataFrames
   - Use Parquet for data storage

4. **Testing**:
   - pytest for all tests
   - Fixtures for common test data
   - Parametrize tests for multiple scenarios

5. **Configuration**:
   - YAML for strategy configs
   - Avoid hard-coded constants
   - Environment-independent paths

---

## Key Constraints & Considerations

### Data Handling
- ✅ Already have `src/ofi_utils.py` for OFI calculation - REUSE IT
- ✅ Timestamp normalization and RTH filtering already implemented
- ⚠️ Need trade data for queue model (check with TA)
- ⚠️ If no trades available, use parametric fill model

### Financial Logic
- **OFI normalization**: Use rolling mean of depth (600s window, 50 min periods)
- **Beta estimates**: Symbol-specific or use average β = 0.036
- **Volatility**: Annualized for comparability (sqrt(252*6.5*3600))
- **Inventory limits**: Enforce strictly (±100 shares default)
- **Spread floors**: Minimum 1 bps (configurable)

### Backtest Realism
- No look-ahead bias (only use t-1 signals for t quotes)
- Respect tick size in all quotes
- Fill model uncertainty (sensitivity analysis)
- Transaction costs (if data available, otherwise assume zero)
- No shorting costs (simplification)

### Performance Expectations
- **Sharpe > 2.0**: Excellent (high frequency context)
- **Fill ratio 30-50%**: Typical for limit orders
- **Edge 1-3 bps**: Realistic for market making
- **Max inventory < 80%**: Good risk control

---

## Success Criteria

### Phase 1-2 (COMPLETED ✅)
- [x] All feature functions implemented and tested
- [x] Quoting engine with A-S framework working
- [x] 52/52 tests passing
- [x] Code committed to GitHub

### Phase 3-4 (IN PROGRESS)
- [ ] Fill model implemented (parametric or queue-based)
- [ ] Backtest engine runs end-to-end on sample data
- [ ] Order lifecycle correct (place, fill, cancel, P&L)
- [ ] Tests cover main scenarios

### Phase 5-7 (NEXT)
- [ ] All metrics computed correctly
- [ ] Baseline strategies implemented
- [ ] CLI scripts functional
- [ ] Results saved and loadable

### Phase 8-10 (FINAL)
- [ ] 90%+ test coverage
- [ ] Parameter optimization complete
- [ ] Technical report written
- [ ] Presentation ready

### Overall
- [ ] OFI strategy outperforms symmetric baseline (Sharpe, edge)
- [ ] Adverse selection reduced vs naive maker
- [ ] Inventory well-controlled
- [ ] Code well-documented and reproducible

---

## Quick Start for New Chat

```python
# To pick up where we left off:

# 1. Verify current state
import subprocess
subprocess.run(["git", "log", "--oneline", "-5"])  # See recent commits
subprocess.run(["pytest", "tests/", "-v"])  # Run all tests (should be 52 passing)

# 2. Next immediate task: Phase 3 (Fill Simulation)
# Create: maker/fills.py with:
# - ParametricFillModel class
# - QueuePosition class (if trade data available)
# - simulate_fill() function
# - Unit tests in tests/test_fills.py

# 3. Then: Phase 4 (Backtest Framework)
# Create: maker/backtest.py with:
# - BacktestEngine class
# - run_single_day() method
# - Order lifecycle logic
# - Unit tests in tests/test_backtest.py

# 4. Reference implementations:
# - Features: maker/features.py (406 lines, 6 functions)
# - Engine: maker/engine.py (465 lines, QuotingEngine class)
# - Tests: tests/test_features.py, tests/test_engine.py
```

---

## Additional Context

### Original Replication Repository
- **Location**: `d:\Harshu\UIUC\...\ofi-replication\`
- **Key files**: 
  - `src/ofi_utils.py` (already ported)
  - `data/raw/*.rda` (20 days of TAQ data)
  - `report/Cont-OFI-HarshH-Report.pdf` (replication results)

### Extension Specification
- **Source**: `Extension.md` (read fully before starting)
- **Strategy design**: Signal layer → Quoting logic → Fill simulation → Backtest
- **Evaluation**: P&L, Sharpe, fill edge, adverse selection

### Current Repository
- **GitHub**: `https://github.com/xecuterisaquant/ofi-marketmaking-strat`
- **Branch**: `main`
- **Last commit**: "Phase 2: Quoting engine - Avellaneda-Stoikov framework with OFI integration, 25 passing tests"
- **Status**: Phases 0-2 complete, ready for Phase 3

---

## Questions to Ask User (if needed)

1. **Trade data availability**: "Do you have TAQ trades data, or should I implement parametric fill model only?"

2. **Target symbols/days**: "Which symbols and days should I prioritize for initial testing? (Currently have AAPL, AMD on 2017-01-03/04)"

3. **Fill model preference**: "Should I start with parametric (simpler, no trade data needed) or queue-reactive (more realistic, needs trades)?"

4. **Performance baseline**: "What Sharpe ratio / edge would you consider 'good' for this strategy?"

5. **Optimization scope**: "How extensive should parameter sweeps be? (Quick grid search vs comprehensive optimization)"

---

## Final Notes

- **All Phase 1-2 code is tested and working** - build on this foundation
- **Reuse `src/ofi_utils.py`** - don't reimplement OFI calculation
- **Follow existing code style** - see `maker/features.py` and `maker/engine.py` for patterns
- **Write tests as you go** - aim for same coverage as Phases 1-2 (100% function coverage)
- **Commit frequently** - one phase per commit with descriptive messages
- **Document financial logic** - explain WHY not just WHAT in docstrings

Good luck! 🚀

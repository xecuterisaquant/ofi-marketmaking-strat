# OFI-Driven Market Making Strategy

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-141%20passing-brightgreen.svg)](https://github.com/xecuterisaquant/ofi-marketmaking-strat)

Extension of Cont, Kukanov, & Stoikov (2014) OFI Replication - Implementing an OFI-driven market making strategy using the Avellaneda-Stoikov framework.

## 🎯 Project Status: 🚧 PHASES 0-5 COMPLETE

**Completed Components**:
- ✅ Features engineering with OFI signals (27 tests)
- ✅ Avellaneda-Stoikov quoting engine (25 tests)
- ✅ Parametric fill simulation model (26 tests)
- ✅ Event-driven backtest framework (24 tests)
- ✅ Performance metrics with anti-overfitting design (39 tests)
- ✅ Comprehensive unit test coverage (141/141 tests, 100% success)
- ✅ Documentation and anti-overfitting protocol

**In Progress**:
- 🚧 Strategy configurations and execution scripts (Phases 6-7)

---

## 📊 Project Overview

This project develops a **market making strategy** that integrates **Order Flow Imbalance (OFI)** signals to reduce adverse selection and improve profitability. Building on a completed OFI replication that demonstrated:

- **100% positive beta rate** (40/40 symbol-days)
- **Mean R² = 8.1%** (OFI explains 8.1% of 1-second price variance)
- **Strong statistical significance** (95% of regressions p < 0.05)

We operationalize these insights into a practical market making engine that:

1. **Computes OFI signals** from normalized order flow imbalance
2. **Generates optimal quotes** using Avellaneda-Stoikov framework with OFI adjustment
3. **Manages inventory risk** through reservation price and dynamic spread widening
4. **Simulates fills** using parametric intensity models
5. **Evaluates performance** via Sharpe ratio, fill edge, and adverse selection metrics

**Key Hypothesis**: Skewing quotes based on OFI signals reduces trades at unfavorable prices, improving market making profitability compared to symmetric baselines.

---

## 📁 Repository Structure

```
ofi-marketmaking-strat/
├── 📄 README.md                    # This file
├── 📄 PROJECT_CONTEXT.md           # Comprehensive project documentation
├── 📄 REPRODUCTION_GUIDE.md        # Step-by-step reproduction instructions
├── 📄 requirements.txt             # Python dependencies
│
├── 📂 maker/                       # Market making modules (PHASES 1-5 ✅)
│   ├── __init__.py
│   ├── features.py                 # ✅ OFI signals, volatility, microprice (406 lines)
│   ├── engine.py                   # ✅ Avellaneda-Stoikov quoting engine (465 lines)
│   ├── fills.py                    # ✅ Parametric fill simulation (473 lines)
│   ├── backtest.py                 # ✅ Event-driven backtest framework (530 lines)
│   └── metrics.py                  # ✅ Performance metrics (450 lines)
│
├── 📂 src/                         # Infrastructure from replication
│   ├── __init__.py
│   └── ofi_utils.py                # ✅ OFI calculation, NBBO handling (245 lines)
│
├── 📂 tests/                       # Unit tests (141 passing ✅)
│   ├── test_features.py            # ✅ 27 tests for feature engineering
│   ├── test_engine.py              # ✅ 25 tests for quoting engine
│   ├── test_fills.py               # ✅ 26 tests for fill simulation
│   ├── test_backtest.py            # ✅ 24 tests for backtest framework
│   └── test_metrics.py             # ✅ 39 tests for performance metrics
│
├── 📂 scripts/                     # Executable scripts (Phases 5-7)
│   ├── run_maker_backtest.py       # Main backtest runner
│   ├── run_strategy_comparison.py  # Compare OFI vs baselines
│   ├── compute_metrics.py          # Performance metrics
│   └── make_figures.py             # Generate plots
│
├── 📂 configs/                     # Strategy configurations
│   ├── ofi_full.yaml               # OFI + microprice + inventory
│   ├── microprice_only.yaml        # Microprice skew only
│   └── symmetric_baseline.yaml     # No signal skew
│
├── 📂 data/                        # TAQ NBBO data (from replication)
│   └── NBBO/
│       ├── 2017-01-03.rda
│       ├── ... (20 days total)
│       └── 2017-01-31.rda
│
├── 📂 results/                     # Backtest outputs
│   ├── strategy_comparison/
│   └── metrics/
│
├── 📂 figures/                     # Generated plots
│
├── 📂 report/                      # R Markdown report
│   ├── OFI-MarketMaker-Report.Rmd  # Main report template
│   ├── references.bib              # BibTeX citations
│   ├── CITATION_GUIDE.md           # Citation instructions
│   ├── render_report.R             # Report rendering script
│   └── arxiv.sty                   # ArXiv style file
│
└── 📂 references/                  # Research papers
```

---

## 🚀 Quick Start

### Prerequisites
```bash
Python 3.13+ (or 3.10+)
R 4.4.2+ (for report generation)
pandas 2.3.2
numpy 2.3.3
scipy 1.14.1
statsmodels 0.14.5
pytest 8.3.4
```

### Installation
```bash
git clone https://github.com/xecuterisaquant/ofi-marketmaking-strat.git
cd ofi-marketmaking-strat
pip install -r requirements.txt
```

### Run Tests
```bash
# Run all unit tests (141 tests)
pytest tests/ -v

# Expected output:
# tests/test_features.py::test_compute_ofi_signal PASSED           [  1/141]
# tests/test_features.py::test_compute_microprice PASSED           [  2/141]
# ... (139 more tests)
# ====================== 141 passed in X.XXs ======================

# Test breakdown:
# - test_features.py: 27 tests (feature engineering)
# - test_engine.py: 25 tests (quoting engine)
# - test_fills.py: 26 tests (fill simulation)
# - test_backtest.py: 24 tests (backtest framework)
# - test_metrics.py: 39 tests (performance metrics)
```

### Quick Validation
```bash
# Test feature computation on real data (when backtest implemented)
python scripts/validate_single_day.py --symbol AAPL --date 2017-01-03
```

---

## 🔬 Methodology

### Feature Engineering (`maker/features.py`)

**Six key functions** compute signals and market features:

1. **`compute_ofi_signal(ofi_normalized, beta=0.036, horizon_seconds=60)`**
   - Converts normalized OFI → expected drift in basis points
   - Formula: `signal_bps = ofi_normalized * beta * 100`
   - Uses mean β = 0.036 from replication study

2. **`compute_microprice(bid, ask, bid_size, ask_size)`**
   - Depth-weighted mid: `(ask * bid_size + bid * ask_size) / (bid_size + ask_size)`
   - More informative than simple mid when book is imbalanced

3. **`compute_ewma_volatility(prices, halflife_seconds=60.0, min_periods=10)`**
   - Exponentially weighted volatility from squared log returns
   - Annualized: `√(252 × 6.5 × 3600)` for 1-second data

4. **`compute_imbalance(bid_size, ask_size)`**
   - Depth imbalance: `(bid_size - ask_size) / (bid_size + ask_size)`
   - Range: [-1, 1]

5. **`blend_signals(ofi_signal, imbalance, alpha_ofi=0.7, alpha_imbalance=0.3)`**
   - Weighted combination: 70% OFI + 30% imbalance (default)
   - Extensible for additional signals

6. **`compute_signal_stats(signal, window_seconds=300)`**
   - Rolling statistics for monitoring and threshold setting

**All functions tested** with 27 passing unit tests covering edge cases, mathematical correctness, and index preservation.

### Fill Simulation (`maker/fills.py`)

**Parametric fill model** for limit order execution:

#### Core Components:

1. **Fill Intensity**:
   ```
   λ(δ) = A * exp(-k * δ)
   ```
   - `δ`: distance from microprice (in bps)
   - `A = 2.0`: base intensity (fills/second at δ=0)
   - `k = 0.5`: decay rate

2. **Fill Probability**:
   ```
   P(fill|δ, Δt) = 1 - exp(-λ(δ) * Δt)
   ```
   - Exponential survival model
   - Δt = 1 second (typical timestep)

3. **Calibration**: Heuristic parameters expected ~86% fill at microprice, ~33% at 10 bps away

**26 passing tests** validate intensity decay, probability bounds, calibration accuracy, and reproducibility.

### Backtest Framework (`maker/backtest.py`)

**Event-driven simulation** for market making:

#### Simulation Loop:

1. **Initialize**: Load NBBO data, set parameters, create engine
2. **Each Second** (9:30-16:00 ET):
   - Update features (OFI, microprice, volatility)
   - Generate quotes using `QuotingEngine`
   - Simulate fills using `ParametricFillModel`
   - Update inventory and cash
   - Track P&L: `pnl_t = cash_t + inventory_t * mid_t`
3. **Output**: Complete trading history with fills, inventory, quotes

**24 passing tests** validate order lifecycle, P&L calculation, inventory management, and data pipeline integration.

### Performance Metrics (`maker/metrics.py`)

**Comprehensive evaluation metrics** with strict anti-overfitting design:

#### Key Metrics:

1. **Sharpe Ratio**: Annualized risk-adjusted return
2. **Sortino Ratio**: Downside deviation only (no penalty for upside volatility)
3. **Maximum Drawdown**: Peak-to-trough decline
4. **Fill Edge**: Profitability per fill vs microprice
5. **Adverse Selection**: Post-fill price drift at 1s/5s/10s horizons
6. **Inventory Metrics**: Position risk statistics
7. **Signal Correlation**: OFI validation (not for parameter tuning)

**Anti-Overfitting Safeguards**:
- All parameters fixed from literature (β=0.036, γ=0.1, A=2.0, k=0.5)
- No parameter optimization on backtest data
- Pre-defined 4 strategy configurations
- Data split: Week 1 validation, Weeks 2-4 test
- Report ALL results (no cherry-picking)

**39 passing tests** validate metric calculations using synthetic data with known answers.

### Quoting Engine (`maker/engine.py`)

**Avellaneda-Stoikov framework** with OFI integration:

#### Core Components:

1. **Reservation Price** (inventory-adjusted fair value):
   ```
   r_t = mid_t - γ * σ² * q_t * T
   ```
   - `γ = 0.1`: risk aversion
   - `q_t`: inventory (shares)
   - `T`: time to close

2. **Quote Width** (optimal half-spread):
   ```
   δ_t = γ * σ² * T + (2/γ) * log(1 + γ/k)
   ```
   - Widens with volatility and inventory
   - Inventory urgency: cubic scaling near limits

3. **OFI Signal Adjustment**:
   - Positive OFI → shift quotes up (expect price rise)
   - Negative OFI → shift quotes down (expect price fall)
   - Moderated by `signal_adjustment_factor = 0.5`

4. **Quote Generation Pipeline**:
   - Compute reservation price
   - Compute quote width
   - Apply OFI skew
   - Apply inventory skew (1 bp per 100 shares)
   - Round to tick size (0.01)
   - Enforce minimum spread (1 bp)
   - Check for crossed market

**25 passing tests** validate zero-inventory symmetry, inventory skew, OFI signal effects, volatility widening, and tick rounding precision.

---

## 📚 Documentation

- **[PROJECT_CONTEXT.md](PROJECT_CONTEXT.md)**: Comprehensive project overview, completed phases, next steps, technical specifications
- **[REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md)**: Step-by-step instructions to reproduce all results
- **[report/CITATION_GUIDE.md](report/CITATION_GUIDE.md)**: How to cite references in R Markdown report

---

## 🎓 Academic Context

### Foundation: OFI Replication

**Original Paper**: Cont, R., Kukanov, A., & Stoikov, S. (2014). The Price Impact of Order Book Events. *Journal of Financial Econometrics*, 12(1), 47-88.

**Our Replication Results** (completed, see `ofi-replication/`):
- ✅ 100% positive beta rate (40/40 symbol-days)
- ✅ Mean R² = 8.1% (OFI explains price variance)
- ✅ All statistical tests passing

### Extension: Market Making Framework

**Theoretical Basis**: Avellaneda, M., & Stoikov, S. (2008). High-frequency trading in a limit order book. *Quantitative Finance*, 8(3), 217-224.

**Key Innovation**: Integrating OFI signals into optimal market making to reduce adverse selection.

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **Cont et al. (2014)** for the OFI framework
- **Avellaneda & Stoikov (2008)** for the market making model
- **TAQ Database (WRDS)** for providing high-frequency data
- **GitHub Copilot** for AI-assisted development

---

## 📮 Contact

- **GitHub**: https://github.com/xecuterisaquant/ofi-marketmaking-strat
- **Author**: Harsh Hari (harsh6@illinois.edu)
- **Institution**: University of Illinois, Department of Finance

---

**Last Updated**: December 3, 2025  
**Status**: 🚧 Phases 0-5 Complete - All Core Components Tested (141/141 tests passing, 100% success)  
**Python Version**: 3.13.0  
**Next Milestone**: Phase 6 (Strategy Configurations)

**Anti-Overfitting Protocol**: See `ANTI_OVERFITTING_PROTOCOL.md` for complete testing strategy and safeguards.

See `REPRODUCTION_GUIDE.md` for detailed setup instructions. Run `pytest tests/ -v` to validate installation.

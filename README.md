# OFI-Driven Market Making Strategy

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-52%20passing-brightgreen.svg)](https://github.com/xecuterisaquant/ofi-marketmaking-strat)

Extension of Cont, Kukanov, & Stoikov (2014) OFI Replication - Implementing an OFI-driven market making strategy using the Avellaneda-Stoikov framework.

## 🎯 Project Status: 🚧 PHASES 0-2 COMPLETE

**Completed Components**:
- ✅ Features engineering with OFI signals (27 tests passing)
- ✅ Avellaneda-Stoikov quoting engine (25 tests passing)
- ✅ Comprehensive unit test coverage (52/52 tests)
- ✅ Documentation and reproduction guide

**In Progress**:
- 🚧 Fill simulation and backtest framework (Phases 3-4)

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
├── 📂 maker/                       # Market making modules (PHASES 1-2 ✅)
│   ├── __init__.py
│   ├── features.py                 # ✅ OFI signals, volatility, microprice (406 lines)
│   ├── engine.py                   # ✅ Avellaneda-Stoikov quoting engine (465 lines)
│   ├── fills.py                    # 🚧 Fill simulation (Phase 3)
│   └── backtest.py                 # 🚧 Backtest framework (Phase 4)
│
├── 📂 src/                         # Infrastructure from replication
│   ├── __init__.py
│   └── ofi_utils.py                # ✅ OFI calculation, NBBO handling (245 lines)
│
├── 📂 tests/                       # Unit tests (52 passing ✅)
│   ├── test_features.py            # ✅ 27 tests for feature engineering
│   ├── test_engine.py              # ✅ 25 tests for quoting engine
│   ├── test_fills.py               # 🚧 Fill simulation tests (Phase 3)
│   └── test_backtest.py            # 🚧 Backtest tests (Phase 4)
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
# Run all unit tests (52 tests)
pytest tests/ -v

# Expected output:
# tests/test_features.py::test_compute_ofi_signal PASSED           [  1/52]
# tests/test_features.py::test_compute_microprice PASSED           [  2/52]
# ... (50 more tests)
# ====================== 52 passed in X.XXs ======================
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
**Status**: 🚧 Phases 0-2 Complete - Features & Engine Tested (52/52 tests passing)  
**Python Version**: 3.13.0  
**Next Milestone**: Phase 3 (Fill Simulation)

See `REPRODUCTION_GUIDE.md` for detailed setup instructions. Run `pytest tests/ -v` to validate installation.

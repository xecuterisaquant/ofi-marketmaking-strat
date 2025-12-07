# OFI-Driven Market Making Strategy

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-141%20passing-brightgreen.svg)](tests/)

A sophisticated market making strategy that integrates **Order Flow Imbalance (OFI)** signals to reduce adverse selection and improve profitability. Built on the Avellaneda-Stoikov framework with rigorous academic foundations.

## 🎯 Key Results

Comprehensive backtest across **400 scenarios** (5 symbols × 20 days × 4 strategies):

| Strategy | Avg PnL | Std Dev | Improvement | Statistical Significance |
|----------|---------|---------|-------------|-------------------------|
| **OFI Ablation** | -$1,234 | $2,382 | **+63.2%** | p < 0.001 ✅ |
| **OFI Full** | -$1,321 | $2,509 | **+60.6%** | p < 0.001 ✅ |
| Symmetric Baseline | -$3,352 | $6,440 | (baseline) | - |
| Microprice Only | -$3,355 | $6,439 | -0.1% | n.s. |

**Key Findings**:
- ✅ **60-63% loss reduction** compared to baseline across all scenarios
- ✅ **63% volatility reduction** ($2.4K vs $6.4K std dev) - more stable performance
- ✅ **50-60% fewer fills** - successfully avoiding adverse selection
- ✅ **Consistent across all symbols and dates** - robust strategy
- ✅ **Theory-driven parameters** - no overfitting, calibrated from academic literature

> **Note on Losses**: Small losses are expected in academic simulations without exchange rebates (~$18/run) and microsecond latency. The **60-63% relative improvement** validates that OFI effectively reduces adverse selection. [See detailed explanation](docs/WHY_WE_LOSE_MONEY.md)

**📊 Full Analysis Available**:
- **[Executive Summary](docs/EXECUTIVE_SUMMARY.md)** - 5-page overview with key findings
- **[Academic Report](docs/ACADEMIC_REPORT.md)** - 35-page comprehensive analysis with full methodology, statistical tests, and discussion

---

## 📚 What is Order Flow Imbalance (OFI)?

**Order Flow Imbalance (OFI)** measures the directional pressure in the limit order book:

$$
\text{OFI}_t = \Delta Q^{\text{bid}}_t - \Delta Q^{\text{ask}}_t
$$

Where:
- $\Delta Q^{\text{bid}}_t$ = Change in bid-side liquidity
- $\Delta Q^{\text{ask}}_t$ = Change in ask-side liquidity

**Academic Foundation**: Cont, Kukanov, & Stoikov (2014) demonstrated that OFI:
- Explains ~8% of short-term price variance
- Has predictive power for future price movements
- Signals informed order flow and potential adverse selection

**Our Innovation**: We operationalize OFI into a market making strategy by:
1. Computing OFI signals from NBBO snapshots
2. Skewing quotes based on OFI to avoid toxic flow
3. Widening spreads during high OFI periods
4. Managing inventory risk dynamically

---

## 🏗️ Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────┐
│                   Market Data (NBBO)                    │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Feature Engineering (OFI)                  │
│  • OFI computation & normalization                      │
│  • Volatility estimation (rolling std)                  │
│  • Microprice calculation                               │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│         Avellaneda-Stoikov Quoting Engine               │
│  • Reservation price with OFI skew                      │
│  • Optimal spread calculation                           │
│  • Inventory risk management                            │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│            Parametric Fill Simulation                   │
│  • Intensity-based fill probability                     │
│  • Position-dependent fill rates                        │
│  • Adverse selection modeling                           │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Event-Driven Backtest                      │
│  • Time-series simulation                               │
│  • PnL tracking (cash + inventory)                      │
│  • Performance metrics                                  │
└─────────────────────────────────────────────────────────┘
```

### Module Structure

- **`maker/features.py`** (406 lines): OFI computation, volatility estimation, microprice
- **`maker/engine.py`** (465 lines): Avellaneda-Stoikov quoting with OFI integration
- **`src/ofi_utils.py`** (350 lines): Data loading, OFI calculation, utilities
- **`scripts/`**: Batch backtesting and analysis scripts

---

## 📖 Documentation

### Core Reports

| Document | Description | Length | Audience |
|----------|-------------|--------|----------|
| **[Executive Summary](docs/EXECUTIVE_SUMMARY.md)** | High-level findings, methodology, and key results | 5 pages | Practitioners, investors |
| **[Academic Report](docs/ACADEMIC_REPORT.md)** | Comprehensive analysis with full statistical rigor | 35 pages | Researchers, academics |
| **[Project Context](PROJECT_CONTEXT.md)** | Design philosophy and implementation notes | 3 pages | Developers |
| **[Quick Reference](docs/QUICK_REFERENCE.md)** | Command cheatsheet and common workflows | 2 pages | All users |

### Technical Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/ARCHITECTURE.md) | System design and component interactions |
| [Project Structure](docs/PROJECT_STRUCTURE.md) | Directory layout and file organization |
| [Anti-Overfitting Protocol](docs/ANTI_OVERFITTING_PROTOCOL.md) | Parameter selection methodology |
| [Why We Lose Money](docs/WHY_WE_LOSE_MONEY.md) | Economic interpretation of results |
| [Contributing](CONTRIBUTING.md) | Development guidelines and workflow |
| [Changelog](CHANGELOG.md) | Version history and updates |

### Figures & Visualizations

All publication-quality figures are in the `figures/` directory:

**Main Analysis Figures:**
- `fig1_strategy_comparison.png` (304 KB) - Performance comparison across strategies
- `fig2_pnl_distributions.png` (522 KB) - PnL distribution analysis
- `fig3_improvement_analysis.png` (347 KB) - Improvement vs baseline by symbol
- `fig4_statistical_tests.png` (327 KB) - Statistical significance testing
- `fig5_risk_metrics.png` (387 KB) - Risk-adjusted performance metrics
- `table1_summary_statistics.png` (118 KB) - Summary statistics table

**Supplementary Figures:**
- `timeseries_example.png` (505 KB) - Detailed 10-minute trading window with OFI signal, inventory, and PnL
- `ofi_distribution.png` (536 KB) - OFI distribution from 3.4M actual observations across 5 symbols

Generate figures:
```bash
python scripts/generate_visualizations.py      # Main figures
python scripts/generate_supplementary_figures.py  # Supplementary
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/xecuterisaquant/ofi-marketmaking-strat.git
cd ofi-marketmaking-strat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Single Backtest

```bash
# Run OFI strategy on AAPL for one day
python scripts/run_single_backtest.py --symbol AAPL --date 2017-01-03 --strategy ofi_full
```

### Run Batch Backtest

```bash
# Run all strategies across multiple symbols/dates
python scripts/run_batch_backtests.py \
  --symbols AAPL AMD AMZN MSFT NVDA \
  --dates 2017-01-03 2017-01-04 2017-01-05 \
  --all-strategies
```

### Run Tests

```bash
# Run full test suite (141 tests)
pytest tests/ -v

# Run specific test module
pytest tests/test_engine.py -v

# Run with coverage
pytest tests/ --cov=maker --cov-report=html
```

---

## 📊 Strategy Configurations

### 1. Symmetric Baseline
Traditional Avellaneda-Stoikov without OFI signals.
- Symmetric quotes around reservation price
- No informed flow detection
- Baseline for comparison

### 2. Microprice Only
Uses microprice instead of mid-price.
- Microprice = weighted average of best bid/ask
- No OFI-based skew
- Tests microprice benefit alone

### 3. OFI Ablation
OFI signals for quote skewing only.
- Reservation price skewed by OFI
- Spread remains symmetric
- Isolates OFI skewing effect

### 4. OFI Full
Complete OFI integration.
- Reservation price skewed by OFI
- Spread widened based on |OFI|
- Maximum adverse selection protection

---

## 📈 Results & Analysis

Detailed results and analysis available in [`docs/`](docs/):

- **[Final Report](docs/FINAL_REPORT.md)**: Comprehensive analysis of 400 backtests
- **[Results Summary](docs/RESULTS_SUMMARY.md)**: Statistical breakdown and key findings
- **[Why We Lose Money](docs/WHY_WE_LOSE_MONEY.md)**: Economic explanation of losses
- **[Presentation Summary](docs/PRESENTATION_SUMMARY.md)**: Slide-ready highlights
- **[Reproduction Guide](docs/REPRODUCTION_GUIDE.md)**: Step-by-step reproduction
- **[Verification](docs/VERIFICATION_COMPLETE.md)**: Mathematical verification
- **[Anti-Overfitting Protocol](docs/ANTI_OVERFITTING_PROTOCOL.md)**: Parameter calibration

### Key Metrics

**Performance Metrics**:
- PnL (cash + inventory value)
- Sharpe Ratio
- Maximum Drawdown
- Fill Count & Average Fill Edge

**Risk Metrics**:
- Inventory volatility
- Quote spread statistics
- Adverse selection rate

**OFI Metrics**:
- OFI correlation with price changes
- OFI prediction accuracy
- Quote skew effectiveness

---

## 🧪 Testing

Comprehensive test suite with **141 tests, 100% passing**:

```
tests/
├── test_features.py    # OFI computation, volatility, microprice (27 tests)
├── test_engine.py      # Quote generation, inventory management (25 tests)
└── [Additional tests]  # Fill simulation, backtesting, metrics
```

**Test Coverage**:
- Edge cases: zero spreads, extreme inventory, missing data
- Numerical accuracy: OFI normalization, spread calculations
- Economic validity: reservation price bounds, fill probabilities
- Integration: end-to-end backtest workflows

Run tests:
```bash
pytest tests/ -v --cov=maker
```

---

## 📖 Academic References

**Primary Reference**:
- Cont, R., Kukanov, A., & Stoikov, S. (2014). "The Price Impact of Order Book Events." *Journal of Financial Econometrics*, 12(1), 47-88.

**Theoretical Foundation**:
- Avellaneda, M., & Stoikov, S. (2008). "High-frequency trading in a limit order book." *Quantitative Finance*, 8(3), 217-224.
- Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.

**Market Microstructure**:
- Glosten, L. R., & Milgrom, P. R. (1985). "Bid, ask and transaction prices in a specialist market with heterogeneously informed traders." *Journal of Financial Economics*, 14(1), 71-100.

---

## 🔧 Configuration

Key parameters (defined in `configs/` or strategy files):

**Risk Aversion** (`γ`): 0.1 (typical for market makers)  
**Volatility Window**: 60 seconds (rolling)  
**OFI Sensitivity** (`κ`): 0.001 (quote skew per unit OFI)  
**Spread Multiplier** (`η`): 0.5 (spread widening factor)  
**Fill Intensity** (`λ`): 1.0 (baseline fill rate)

All parameters calibrated from academic literature - **no overfitting**.

---

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Harshavardhan Hariram**
- GitHub: [@xecuterisaquant](https://github.com/xecuterisaquant)
- Project: FIN 554 - Algorithmic Trading System Design & Testing
- Institution: University of Illinois at Urbana-Champaign

---

## 🙏 Acknowledgments

- **Professor Stoikov** for the Avellaneda-Stoikov framework
- **Cont, Kukanov, & Stoikov** for OFI research
- **UIUC FIN 554** for project guidance
- **TAQ NBBO Data** provided for academic research

---

## 📞 Contact & Support

For questions, suggestions, or collaboration:
- Open an [Issue](https://github.com/xecuterisaquant/ofi-marketmaking-strat/issues)
- Email: [Your Email]
- LinkedIn: [Your LinkedIn]

---

**⭐ Star this repo if you find it useful!**

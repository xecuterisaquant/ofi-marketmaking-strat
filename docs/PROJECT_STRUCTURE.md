# Project Structure

## Directory Layout

```
ofi-marketmaking-strat/
│
├── 📄 Core Documentation
│   ├── README.md                          # Main project overview
│   ├── LICENSE                            # MIT License
│   ├── CONTRIBUTING.md                    # Contribution guidelines
│   ├── requirements.txt                   # Python dependencies
│   └── .gitignore                        # Git ignore patterns
│
├── 📂 maker/                              # Core Market Making Modules
│   ├── __init__.py                       # Package initialization
│   ├── features.py                       # OFI, volatility, microprice (406 lines)
│   └── engine.py                         # Avellaneda-Stoikov quoting (465 lines)
│
├── 📂 src/                                # Utilities & Data Processing
│   ├── __init__.py                       # Package initialization
│   └── ofi_utils.py                      # Data loading, OFI calculation (350 lines)
│
├── 📂 scripts/                            # Execution Scripts
│   ├── run_single_backtest.py           # Single symbol/date backtest
│   └── run_batch_backtests.py           # Batch backtest runner
│
├── 📂 tests/                              # Test Suite (141 tests, 100% pass)
│   ├── test_features.py                  # Feature engineering tests (27 tests)
│   ├── test_engine.py                    # Quoting engine tests (25 tests)
│   ├── test_fills.py                     # Fill simulation tests (26 tests)
│   ├── test_backtest.py                  # Backtest framework tests (24 tests)
│   └── test_metrics.py                   # Performance metrics tests (39 tests)
│
├── 📂 docs/                               # Documentation
│   ├── ARCHITECTURE.md                   # Technical architecture
│   ├── FINAL_REPORT.md                   # Comprehensive results (400 backtests)
│   ├── RESULTS_SUMMARY.md                # Statistical summary
│   ├── WHY_WE_LOSE_MONEY.md             # Economic explanation
│   ├── PRESENTATION_SUMMARY.md           # Slide-ready highlights
│   ├── REPRODUCTION_GUIDE.md             # Step-by-step reproduction
│   ├── VERIFICATION_COMPLETE.md          # Mathematical verification
│   └── ANTI_OVERFITTING_PROTOCOL.md     # Parameter calibration
│
├── 📂 configs/                            # Strategy Configurations
│   ├── symmetric_baseline.yaml          # Traditional Avellaneda-Stoikov
│   ├── microprice_only.yaml             # Microprice variant
│   ├── ofi_ablation.yaml                # OFI skewing only
│   └── ofi_full.yaml                    # Full OFI integration
│
├── 📂 data/                               # Market Data (gitignored)
│   ├── NBBO/                            # NBBO snapshots (1-second)
│   │   └── *.rda                        # RDA files (20 days)
│   └── Trade/                           # Trade executions
│       └── *.rda                        # RDA files (20 days)
│
├── 📂 results/                            # Backtest Results (gitignored)
│   ├── batch/                           # Batch summary CSVs
│   │   └── batch_summary_*.csv         # Aggregate statistics
│   └── detailed/                        # Detailed parquet files
│       └── {SYMBOL}_{DATE}_{STRATEGY}.parquet
│
├── 📂 figures/                            # Plots & Visualizations
│   ├── pnl_distribution.png            # PnL histogram
│   ├── strategy_comparison.png         # Box plots
│   └── ofi_correlation.png             # OFI vs price change
│
├── 📂 ofi-replication/                    # Prior Work (OFI Replication)
│   ├── README.md                        # Replication documentation
│   ├── scripts/                         # Analysis scripts
│   ├── report/                          # LaTeX report
│   └── results/                         # Replication results
│
└── 📂 references/                         # Academic Papers
    ├── cont_2014_ofi.pdf               # Cont, Kukanov, Stoikov (2014)
    ├── avellaneda_2008_mm.pdf          # Avellaneda & Stoikov (2008)
    └── glosten_1985_adversesel.pdf     # Glosten & Milgrom (1985)
```

---

## File Descriptions

### Core Modules

#### `maker/features.py`
Feature engineering module for market making signals.

**Key Classes**:
- `OFICalculator`: Computes normalized OFI from NBBO changes
- `VolatilityEstimator`: Rolling volatility estimation (60s window)
- `MicropriceCalculator`: Volume-weighted bid/ask average

**Lines**: 406 | **Tests**: 27

---

#### `maker/engine.py`
Avellaneda-Stoikov quoting engine with OFI integration.

**Key Classes**:
- `AvellanedaStoikovEngine`: Core quoting logic
  - `compute_reservation_price()`: r = mid - q·γ·σ²·(T-t) + κ·OFI
  - `compute_optimal_spread()`: δ = γ·σ²·(T-t) + adverse_selection + η·|OFI|
  - `generate_quotes()`: bid/ask quote generation

**Lines**: 465 | **Tests**: 25

---

#### `src/ofi_utils.py`
Data loading and OFI calculation utilities.

**Key Functions**:
- `load_nbbo_data()`: Load NBBO snapshots from RDA files
- `load_trade_data()`: Load trade executions
- `compute_ofi_from_nbbo()`: OFI calculation wrapper
- `normalize_ofi()`: Rolling standardization

**Lines**: 350 | **Tests**: Covered in integration tests

---

### Documentation

#### `docs/ARCHITECTURE.md`
Detailed technical architecture covering:
- System components and data flow
- Mathematical formulations
- Implementation details
- Performance considerations

#### `docs/FINAL_REPORT.md`
Comprehensive analysis of 400 backtests:
- Statistical results
- Strategy comparisons
- Hypothesis testing
- Detailed findings

#### `docs/WHY_WE_LOSE_MONEY.md`
Economic explanation of losses:
- Adverse selection mechanics
- Missing infrastructure (rebates, latency)
- Why relative improvement matters
- Mathematical breakdown

#### `docs/REPRODUCTION_GUIDE.md`
Step-by-step instructions:
- Environment setup
- Data preparation
- Running backtests
- Reproducing all 400 results

---

### Scripts

#### `scripts/run_single_backtest.py`
Single backtest execution.

**Usage**:
```bash
python scripts/run_single_backtest.py \
  --symbol AAPL \
  --date 2017-01-03 \
  --strategy ofi_full
```

**Output**: Detailed parquet file in `results/detailed/`

---

#### `scripts/run_batch_backtests.py`
Batch backtest runner for comprehensive analysis.

**Usage**:
```bash
python scripts/run_batch_backtests.py \
  --symbols AAPL AMD AMZN MSFT NVDA \
  --dates 2017-01-03 2017-01-04 ... \
  --all-strategies
```

**Output**: 
- Detailed parquet files
- Batch summary CSV
- Aggregate statistics

---

### Tests

Comprehensive test suite with 141 tests (100% passing):

| Test Module | Tests | Coverage | Description |
|-------------|-------|----------|-------------|
| `test_features.py` | 27 | OFI, volatility, microprice | Feature engineering |
| `test_engine.py` | 25 | Quoting logic | AS framework |
| `test_fills.py` | 26 | Fill simulation | Intensity models |
| `test_backtest.py` | 24 | Event loop | State management |
| `test_metrics.py` | 39 | Performance | Sharpe, drawdown |

**Run Tests**:
```bash
pytest tests/ -v --cov=maker --cov=src
```

---

## Data Organization

### Input Data Structure

```
data/
├── NBBO/
│   ├── 2017-01-03.rda    # NBBO snapshots (1-second)
│   ├── 2017-01-04.rda
│   └── ...               # 20 trading days
└── Trade/
    ├── 2017-01-03.rda    # Trade executions
    ├── 2017-01-04.rda
    └── ...
```

**Data Format (NBBO)**:
- `time`: Unix timestamp (1-second resolution)
- `bid`: Best bid price
- `ask`: Best ask price
- `bid_size`: Size at best bid
- `ask_size`: Size at best ask

**Data Size**: ~7MB per symbol-day

---

### Results Structure

```
results/
├── batch/
│   └── batch_summary_20251206_194455.csv    # Aggregate statistics
│       Columns: [symbol, date, strategy, final_pnl, sharpe, 
│                 max_dd, fill_count, avg_spread, ...]
└── detailed/
    ├── AAPL_2017-01-03_symmetric_baseline.parquet
    ├── AAPL_2017-01-03_ofi_full.parquet
    └── ...                                   # 400 detailed files
```

**Detailed Parquet Format**:
- `time`: Timestamp
- `bid`, `ask`, `mid`: Market prices
- `our_bid`, `our_ask`: Our quotes
- `inventory`: Position
- `cash`: Cash balance
- `pnl`: Mark-to-market PnL
- `ofi_normalized`: OFI signal
- `volatility`: Volatility estimate

**File Size**: ~500KB per backtest

---

## Configuration Files

### Strategy Configs

Located in `configs/`:

**`symmetric_baseline.yaml`**:
```yaml
strategy: symmetric_baseline
risk_aversion: 0.1
fill_intensity: 1.0
use_microprice: false
use_ofi_skew: false
use_ofi_spread: false
```

**`ofi_full.yaml`**:
```yaml
strategy: ofi_full
risk_aversion: 0.1
fill_intensity: 1.0
ofi_sensitivity: 0.001
spread_multiplier: 0.5
use_microprice: true
use_ofi_skew: true
use_ofi_spread: true
```

---

## Dependencies

### Production Dependencies

```
numpy>=1.24.0         # Numerical computations
pandas>=2.0.0         # Data manipulation
scipy>=1.10.0         # Statistical functions
pyarrow>=12.0.0       # Parquet I/O
pyyaml>=6.0          # Config parsing
```

### Development Dependencies

```
pytest>=7.4.0         # Testing framework
pytest-cov>=4.1.0     # Coverage reporting
black>=23.0.0         # Code formatting
mypy>=1.4.0          # Type checking
pylint>=2.17.0       # Linting
```

**Install**:
```bash
pip install -r requirements.txt
```

---

## Size Summary

| Category | Count | Total Size |
|----------|-------|------------|
| Python Modules | 5 | ~2,500 lines |
| Test Files | 5 | ~1,800 lines |
| Documentation | 8 MD files | ~50KB |
| Data Files | 40 RDA files | ~280MB |
| Results | 400 parquet | ~200MB |
| Figures | 10+ plots | ~5MB |

**Total Repository Size**: ~500MB (with data)  
**Total Repository Size**: ~5MB (without data/results)

---

## Git Ignore Patterns

Key patterns in `.gitignore`:

```gitignore
# Large data files
data/
results/detailed/*.parquet
results/backtest_summary_*.csv

# Python artifacts
__pycache__/
*.pyc
venv/

# Development files
*_OLD.md
*_NEW.md
*.tmp

# IDE
.vscode/
.idea/

# Logs
logs/*.log
```

**Tracked Files**: Source code, tests, docs, configs  
**Ignored Files**: Data, results, artifacts, logs

---

## Maintenance

### Adding New Strategy

1. Create config in `configs/new_strategy.yaml`
2. Implement logic in `maker/engine.py`
3. Add tests in `tests/test_engine.py`
4. Update documentation

### Adding New Feature

1. Implement in `maker/features.py`
2. Add tests in `tests/test_features.py`
3. Integrate into engine
4. Update docs

### Adding New Data

1. Place RDA files in `data/NBBO/` or `data/Trade/`
2. Update date ranges in scripts
3. Run backtests
4. Analyze results

---

## Contact

For questions about project structure:
- GitHub: [@xecuterisaquant](https://github.com/xecuterisaquant)
- Issues: [Project Issues](https://github.com/xecuterisaquant/ofi-marketmaking-strat/issues)

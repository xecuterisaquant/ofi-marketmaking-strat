# OFI-Driven Market Making Strategy

## Overview

This project extends the OFI (Order Flow Imbalance) replication study by implementing an **inventory-aware market maker** with OFI-based signal skewing. The strategy uses normalized OFI signals to adjust quote placement, reducing adverse selection and improving profitability.

## Background

Based on the replication of *Cont, Kukanov & Stoikov (2014): "The Price Impact of Order Book Events"*, which established that:
- Order flow imbalance (OFI) predicts short-term price movements
- Mean R² = 8.1% across 40 symbol-days
- 100% positive beta coefficients (strong evidence of OFI-price relationship)

This project leverages those findings to build a practical market-making strategy.

## Project Structure

```
ofi-marketmaking-strat/
├── maker/                  # Core market making modules
│   ├── __init__.py
│   ├── features.py        # Signal computation (OFI, microprice, volatility)
│   ├── engine.py          # Quoting logic (reservation price, bid/ask)
│   ├── fills.py           # Fill simulation models
│   ├── backtest.py        # Simulation loop
│   └── metrics.py         # Performance evaluation
├── src/                   # Ported infrastructure from replication
│   ├── __init__.py
│   └── ofi_utils.py       # OFI calculation, NBBO handling, timestamp utils
├── scripts/               # Executable scripts
│   └── run_maker_backtest.py
├── tests/                 # Unit and integration tests
│   ├── test_features.py
│   ├── test_engine.py
│   ├── test_fills.py
│   └── test_backtest.py
├── configs/               # Strategy configurations
│   ├── symmetric_baseline.yaml
│   ├── microprice_only.yaml
│   └── ofi_full.yaml
├── data/                  # Data directory
│   └── raw/              # Symlink to TAQ .rda files
├── results/              # Backtest results
└── figures/              # Generated plots
```

## Installation

```bash
# Clone the repository
git clone https://github.com/xecuterisaquant/ofi-marketmaking-strat.git
cd ofi-marketmaking-strat

# Install dependencies
pip install -r requirements.txt

# (Optional) Link to TAQ data
# On Windows PowerShell:
# New-Item -ItemType SymbolicLink -Path "data\raw" -Target "path\to\replication\data\raw"
```

## Quick Start

```bash
# Run baseline comparison (symmetric vs OFI-driven)
python scripts/run_maker_backtest.py \
    --data-dir data/raw \
    --config configs/ofi_full.yaml \
    --symbols AAPL AMD \
    --days 2017-01-03 2017-01-04 \
    --output results/baseline

# Generate performance report
python scripts/generate_report.py \
    --results-dir results/baseline \
    --output reports/baseline_report.pdf
```

## Strategy Design

### 1. Signal Layer
- **OFI Signal**: `signal_t = beta_hat * normalized_OFI_t` (in bps)
- **Microprice Tilt**: `(microprice - mid) / half_spread`
- **Imbalance**: `(bid_size - ask_size) / total_depth`
- **Blended Signal**: Weighted combination with configurable weights

### 2. Quoting Logic
- **Reservation Price**: `r_t = mid_t + k1 * signal_t - k2 * inventory_t`
- **Quote Width**: `w_t = w0 + a_sigma * volatility_t + a_q * |inventory| + a_s * |signal|`
- **Bid/Ask**: `bid = r - w/2`, `ask = r + w/2`
- **Constraints**: Tick size, minimum spread, no market crossing

### 3. Fill Simulation
Two models supported:
- **Parametric**: Fill probability based on distance from mid
- **Queue-based**: Tracks position in limit order book queue

### 4. Performance Metrics
- PnL and Sharpe ratio
- Inventory variance and max drawdown
- Fill ratio and average edge per fill
- Adverse selection cost

## Baseline Strategies

1. **Symmetric Baseline**: Mid-centered quotes, no skew
2. **Microprice Only**: Skew based on microprice tilt only
3. **OFI Full**: Full model with OFI + microprice + inventory skew

## Data

Uses TAQ NBBO quote data from January 2017:
- **Symbols**: AAPL, AMD, AMZN, JPM, MSFT, NVDA, SPY, TSLA
- **Days**: 5 trading days (Jan 3-9, 2017)
- **Frequency**: 1-second resampled top-of-book
- **Source**: WRDS TAQ database

## Results

(To be populated after backtests)

### Key Findings
- [ ] OFI signal reduces adverse selection by X%
- [ ] Average edge per fill improves by Y bps
- [ ] Sharpe ratio: OFI strategy vs baselines
- [ ] Optimal signal blending weights

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_features.py -v

# Run with coverage
pytest tests/ --cov=maker --cov-report=html
```

## Development Status

**Phase 0**: ✅ Repository setup, infrastructure porting  
**Phase 1**: 🚧 Feature engineering (OFI signal, volatility, microprice)  
**Phase 2**: ⏳ Quoting engine  
**Phase 3**: ⏳ Fill simulation  
**Phase 4**: ⏳ Backtest loop  
**Phase 5**: ⏳ Metrics & evaluation  
**Phase 6**: ⏳ Baseline strategies  
**Phase 7**: ⏳ Testing & validation  
**Phase 8**: ⏳ Batch processing & CLI  
**Phase 9**: ⏳ Ablation studies  
**Phase 10**: ⏳ Documentation & results  

## References

- Cont, R., Kukanov, A., & Stoikov, S. (2014). The price impact of order book events. *Journal of Financial Econometrics*, 12(1), 47-88.
- Original replication repository: [github.com/xecuterisaquant/replication-cont-ofi](https://github.com/xecuterisaquant/replication-cont-ofi)

## License

MIT License - See LICENSE file for details

## Author

Harsh Hari  
University of Illinois at Urbana-Champaign  
FIN 554 - Algorithmic Trading Systems Design & Testing  
November 2025

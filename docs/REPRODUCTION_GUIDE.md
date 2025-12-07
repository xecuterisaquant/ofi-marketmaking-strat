# Complete Reproduction Guide

This document provides step-by-step instructions to reproduce all results from the OFI Market Making Strategy project.

## 📦 Prerequisites

### Software Requirements
- **Python 3.13** (or 3.10+)
- **R 4.4.2** (or 4.0+) for report generation
- **Git** (for cloning repository)

### R Packages
```r
install.packages("rmarkdown")
install.packages("rticles")
install.packages("knitr")
```

### Python Packages
```bash
pip install -r requirements.txt
```

**Contents of requirements.txt**:
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

---

## 📁 Data Setup

### TAQ Data Source
This project uses **Trade and Quote (TAQ)** NBBO data from WRDS (Wharton Research Data Services).

**Access Requirements**:
- WRDS subscription with TAQ access
- Institutional login credentials

**Data Specification**:
- **Dataset**: TAQ NBBO (National Best Bid and Offer)
- **Period**: January 3-31, 2017 (20 trading days)
- **Symbols**: AAPL, AMD, AMZN, JPM, MSFT, NVDA, SPY, TSLA
- **Format**: R data files (`.rda`) containing quote updates
- **Fields**: `symbol`, `time_m`, `best_bid`, `best_ask`, `best_bidsiz`, `best_asksiz`

### Data Directory Structure

Place TAQ data files in the following structure:

```
data/
├── NBBO/
│   ├── 2017-01-03.rda
│   ├── 2017-01-04.rda
│   ├── ... (20 files total)
│   └── 2017-01-31.rda
└── Trade/  (optional, for advanced fill models)
```

**File Naming Convention**: `YYYY-MM-DD.rda` (matches ofi-replication format)

**Note**: Data files should already be present from the completed OFI replication project.

---

## 🔄 Reproduction Steps

### Step 1: Clone Repository
```bash
git clone https://github.com/xecuterisaquant/ofi-marketmaking-strat.git
cd ofi-marketmaking-strat
```

### Step 2: Install Dependencies
```bash
# Python packages
pip install -r requirements.txt

# R packages (in R console)
install.packages(c("rmarkdown", "rticles", "knitr"))
```

### Step 3: Verify Data Placement
Ensure TAQ `.rda` files are in `data/NBBO/` directory.

```powershell
# Check data files exist (PowerShell)
Get-ChildItem data\NBBO\*.rda | Measure-Object
# Should show 20 files for full reproduction
```

### Step 4: Run Unit Tests
```bash
# Run all tests (should show 52 passing)
pytest tests/ -v

# Expected output:
# tests/test_features.py::test_compute_ofi_signal PASSED
# ... (50 more tests)
# ====================== 52 passed in X.XXs ======================
```

### Step 5: Run Backtest (When Implemented)
```bash
# Quick validation (single day)
python scripts/validate_single_day.py --symbol AAPL --date 2017-01-03

# Full week backtest
python scripts/run_maker_backtest.py \
    --symbols AAPL AMD NVDA SPY \
    --days 2017-01-03 2017-01-04 2017-01-05 \
    --config configs/ofi_full.yaml \
    --output results/week1
```

### Step 6: Generate Figures
```bash
python scripts/make_figures.py \
    --results-dir results/week1 \
    --output-dir figures
```

### Step 7: Render PDF Report
```bash
cd report
Rscript render_report.R
```

---

## ✅ Expected Results

### Unit Tests
- **52 tests total** (27 features + 25 engine)
- **100% passing** rate
- **Full coverage** of public functions

### Performance Metrics (After Backtest)
- OFI Full Sharpe > Microprice Only > Symmetric
- Fill edge improvement of X.X+ bps
- Adverse selection reduction of XX%

---

## 🐛 Troubleshooting

### "Module 'maker' not found"
```bash
cd path/to/ofi-marketmaking-strat
python -c "import maker; print(maker.__file__)"
```

### "Tests failing"
```bash
pip install --upgrade -r requirements.txt
pytest tests/ -v
```

### "PDF won't render"
Install TeX distribution:
- **Windows**: [MiKTeX](https://miktex.org/download)
- **Mac**: [MacTeX](https://www.tug.org/mactex/)
- **Linux**: `sudo apt-get install texlive-xetex`

---

## 📧 Contact

**Author**: Harsh Hari  
**Email**: harsh6@illinois.edu  
**Institution**: University of Illinois, Department of Finance

**Repository**: https://github.com/xecuterisaquant/ofi-marketmaking-strat

---

**Last Updated**: December 3, 2025  
**Status**: Phases 0-2 Complete (52/52 tests passing)

# Complete Reproduction Guide

This document provides step-by-step instructions to reproduce all results from the OFI replication project.

## 📦 Prerequisites

### Software Requirements
- **Python 3.13** (or 3.10+)
- **R 4.4.2** (or 4.0+)
- **Git** (for cloning repository)

### R Packages
```r
install.packages("rmarkdown")
install.packages("rticles")
```

### Python Packages
```bash
pip install -r requirements.txt
```

**Contents of requirements.txt**:
```
pandas>=2.0.0
numpy>=1.24.0
statsmodels>=0.14.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyarrow>=12.0.0  # for parquet support
pytest>=7.3.0
```

---

## 📁 Data Setup

### TAQ Data Source
This project uses **Trade and Quote (TAQ)** data from WRDS (Wharton Research Data Services).

**Access Requirements**:
- WRDS subscription with TAQ access
- Institutional login credentials

**Data Specification**:
- **Dataset**: TAQ NBBO (National Best Bid and Offer)
- **Period**: January 3-9, 2017 (first full trading week)
- **Symbols**: AAPL, AMD, AMZN, JPM, MSFT, NVDA, SPY, TSLA
- **Format**: R data files (`.rda`) containing quote updates
- **Fields**: `Symbol`, `time_m`, `best_bid`, `best_ask`, `best_bidsiz`, `best_asksiz`

### Data Directory Structure

Place TAQ data files in the following structure:

```
data/
└── raw/
    ├── AAPL_20170103.rda
    ├── AAPL_20170104.rda
    ├── ... (5 days × 8 symbols = 40 files)
    └── TSLA_20170109.rda
```

**File Naming Convention**: `{SYMBOL}_{YYYYMMDD}.rda`

### Data Size Note
Complete TAQ data for this project is approximately **478 MB**. Due to size and licensing restrictions, data availability:
- **Included in full ZIP**: Yes (if provided separately)
- **Licensing**: WRDS subscription required for download

---

## 🔄 Reproduction Steps

### Step 1: Clone Repository
```bash
git clone https://github.com/xecuterisaquant/replication-cont-ofi.git
cd replication-cont-ofi
```

### Step 2: Install Dependencies
```bash
# Python packages
pip install -r requirements.txt

# R packages (in R console)
install.packages(c("rmarkdown", "rticles"))
```

### Step 3: Verify Data Placement
Ensure TAQ `.rda` files are in `data/raw/` directory.

```powershell
# Check data files exist (PowerShell)
Get-ChildItem data\raw\*.rda | Measure-Object
# Should show 40 files
```

### Step 4: Run Analysis Pipeline

**Option A: Full Batch Processing (Recommended)**
```bash
# Process all 40 symbol-days and generate results
python scripts/run_ofi_batch.py

# Expected output:
# - results_fixed/regressions/by_symbol_day.parquet
# - Runtime: ~15-20 minutes
```

**Option B: Single Symbol Validation (Quick Test)**
```bash
# Test with AMD (5 days)
python scripts/validate_amd_week.py

# Expected output:
# - Validation results printed to console
# - Runtime: ~2-3 minutes
```

### Step 5: Generate Figures
```bash
python scripts/make_figures.py

# Generates 6 publication-quality figures:
# - fig1_beta_distribution.png
# - fig2_rsquared_analysis.png
# - fig3_scatter_examples.png
# - fig4_summary_table.png
# - fig5_timeseries_example.png
# - fig6_before_after_comparison.png
#
# Output location: figures_presentation/
```

### Step 6: Render PDF Report

**Using R (Command Line)**:
```bash
# Navigate to report directory
cd report

# Render PDF (PowerShell)
Rscript -e "rmarkdown::render('Cont-OFI-HarshH-Report.Rmd', output_format='rticles::arxiv_article')"

# Output: Cont-OFI-HarshH-Report.pdf
```

**Using RStudio (GUI)**:
1. Open `report/Cont-OFI-HarshH-Report.Rmd` in RStudio
2. Click **Knit** button
3. Select **Knit to PDF**

**Note**: PDF rendering uses **xelatex** engine for Unicode support. If you encounter errors, ensure:
- TeX distribution is installed (MiKTeX on Windows, TeX Live on Linux/Mac)
- `xelatex` is in your system PATH

---

## ✅ Expected Results

### Regression Statistics

**Key Metrics**:
- **40 regressions** (8 symbols × 5 days)
- **100% positive betas** (all 40/40)
- **Mean β = 0.820** (price impact coefficient)
- **Mean R² = 0.081** (8.1% variance explained)
- **95% p < 0.05** (statistical significance)

### Figures

All 6 figures should match those in the PDF report:
1. **Beta Distribution**: Histogram + boxplots showing all betas > 0
2. **R² Analysis**: Distribution showing mean R² = 8.1%
3. **Scatter Examples**: OFI vs. price change for 4 symbols
4. **Summary Table**: Regression statistics by symbol
5. **Time Series**: 10-minute AMD window showing OFI-price co-movement
6. **Before/After Bug Fix**: Comparison showing 89× R² improvement

### PDF Report

**Final Report Specifications**:
- **Pages**: ~25-30 pages
- **File Size**: ~2.0 MB (includes 6 embedded figures)
- **Format**: ArXiv-style academic paper
- **Sections**: Abstract, Intro, Literature Review, Hypotheses, Data, Methods, Results, Conclusions
- **Figures**: All 6 embedded at appropriate locations
- **Citations**: BibTeX references to Cont et al. (2014) and GitHub Copilot (2025)

---

## 🧪 Validation & Testing

### Unit Tests
```bash
# Run test suite (7 tests for OFI sign conventions)
pytest tests/test_ofi_sign_conventions.py -v

# Expected output:
# ✓ test_aggressive_buy
# ✓ test_aggressive_sell
# ✓ test_passive_increase
# ✓ test_mixed_events
# ✓ test_zero_sizes
# ✓ test_ofi_price_correlation
# ✓ test_ofi_magnitude_bounds
#
# PASSED: 7/7
```

### Manual Verification

**Check 1: Data Loading**
```python
# Quick check in Python
import pandas as pd
from src.ofi_utils import load_taq_data

df = load_taq_data('data/raw/AMD_20170103.rda')
print(f"Rows: {len(df)}")  # Should be ~187,000
print(df.head())
```

**Check 2: OFI Calculation**
```python
from src.ofi_utils import compute_ofi_depth_mid

ofi, depth, mid = compute_ofi_depth_mid(df)
print(f"OFI mean: {ofi.mean():.2f}")  # Should be negative (net selling)
print(f"OFI std: {ofi.std():.2f}")    # Should be ~5,000
```

---

## 🐛 Troubleshooting

### Issue: "No module named 'rpy2'"
**Solution**: This project doesn't use rpy2. Ensure you're running the correct Python environment.

### Issue: "File not found: data/raw/..."
**Solution**: Ensure TAQ data files are downloaded from WRDS and placed in `data/raw/` with correct naming.

### Issue: "xelatex not found"
**Solution**: Install a TeX distribution:
- Windows: [MiKTeX](https://miktex.org/download)
- Mac: [MacTeX](https://www.tug.org/mactex/)
- Linux: `sudo apt-get install texlive-xetex`

### Issue: R Markdown won't knit
**Solution**: 
```r
# Install missing packages
install.packages("rmarkdown")
install.packages("rticles")
install.packages("knitr")
```

### Issue: Figures not appearing in PDF
**Solution**: 
1. Ensure `scripts/make_figures.py` has run successfully
2. Check `figures_presentation/` contains all 6 PNG files
3. Verify figure paths in `.Rmd` are correct: `../figures_presentation/fig*.png`

### Issue: Different results than report
**Possible causes**:
- Different Python/R versions (tested: Python 3.13, R 4.4.2)
- Different data files (verify TAQ data is January 2017)
- Random seed differences (should be minimal for OLS regressions)

If results differ significantly, contact harsh6@illinois.edu.

---

## 📧 Contact & Support

**Author**: Harsh Hari  
**Email**: harsh6@illinois.edu  
**Institution**: University of Illinois, Department of Finance  
**Course**: FIN 554 - Algorithmic Trading System Design & Testing

**Repository**: https://github.com/xecuterisaquant/replication-cont-ofi

---

## 📜 License & Acknowledgments

**Code License**: MIT License  
**Report License**: CC-BY 4.0

**Acknowledgments**:
- Original paper: Cont, Kukanov & Stoikov (2014)
- Data source: WRDS TAQ Database
- AI assistance: GitHub Copilot (debugging and testing)

---

**Last Updated**: October 19, 2025  
**Project Status**: ✅ Complete & Reproducible

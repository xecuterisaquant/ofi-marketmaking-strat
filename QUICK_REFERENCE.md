# 🚀 Quick Reference Guide

A one-page reference for navigating and using this repository.

---

## 📂 Where to Find Things

| What You Need | Where to Look |
|---------------|---------------|
| **Quick overview** | `README.md` (root) |
| **How to contribute** | `CONTRIBUTING.md` (root) |
| **Version history** | `CHANGELOG.md` (root) |
| **Technical details** | `docs/ARCHITECTURE.md` |
| **File organization** | `docs/PROJECT_STRUCTURE.md` |
| **Comprehensive results** | `docs/FINAL_REPORT.md` |
| **Why we lose money** | `docs/WHY_WE_LOSE_MONEY.md` |
| **How to reproduce** | `docs/REPRODUCTION_GUIDE.md` |
| **Statistical summary** | `docs/RESULTS_SUMMARY.md` |
| **Presentation slides** | `docs/PRESENTATION_SUMMARY.md` |
| **Math verification** | `docs/VERIFICATION_COMPLETE.md` |
| **Parameter calibration** | `docs/ANTI_OVERFITTING_PROTOCOL.md` |

---

## ⚡ Quick Commands

### Installation
```bash
git clone https://github.com/xecuterisaquant/ofi-marketmaking-strat.git
cd ofi-marketmaking-strat
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run Single Backtest
```bash
python scripts/run_single_backtest.py --symbol AAPL --date 2017-01-03 --strategy ofi_full
```

### Run Batch Backtest
```bash
python scripts/run_batch_backtests.py --symbols AAPL AMD --dates 2017-01-03 2017-01-04 --all-strategies
```

### Run Tests
```bash
pytest tests/ -v                    # All tests
pytest tests/test_engine.py -v     # Specific module
pytest tests/ --cov=maker          # With coverage
```

### Format Code
```bash
black maker/ src/ tests/
isort maker/ src/ tests/
```

---

## 📊 Key Results (Quick Reference)

| Strategy | Avg PnL | Improvement | Significance |
|----------|---------|-------------|--------------|
| **OFI Ablation** | -$1,234 | **+63.2%** | p < 0.001 ✅ |
| **OFI Full** | -$1,321 | **+60.6%** | p < 0.001 ✅ |
| Baseline | -$3,352 | - | - |

**Key Takeaway**: OFI reduces losses by 60-63% across all scenarios.

---

## 🔑 Key Formulas

### Reservation Price
$$r_t = S_t - q_t \cdot \gamma \cdot \sigma^2 \cdot (T - t) + \kappa \cdot \text{OFI}^{\text{norm}}_t$$

### Optimal Spread
$$\delta_t = \gamma \sigma^2 (T - t) + \frac{2}{\gamma} \log(1 + \gamma / \lambda) + \eta \cdot |\text{OFI}^{\text{norm}}_t|$$

### OFI
$$\text{OFI}_t = \Delta Q^{\text{bid}}_t - \Delta Q^{\text{ask}}_t$$

---

## 🗺️ Repository Map

```
ofi-marketmaking-strat/
├── README.md              ← Start here
├── CONTRIBUTING.md        ← Want to contribute?
├── CHANGELOG.md           ← Version history
│
├── docs/                  ← All documentation
│   ├── ARCHITECTURE.md    ← Technical deep-dive
│   ├── FINAL_REPORT.md    ← Comprehensive results
│   └── ...
│
├── maker/                 ← Core code
│   ├── features.py        ← OFI, volatility
│   └── engine.py          ← Quoting logic
│
├── scripts/               ← Run backtests
│   ├── run_single_backtest.py
│   └── run_batch_backtests.py
│
└── tests/                 ← 141 tests
    ├── test_features.py
    └── test_engine.py
```

---

## 🎯 Common Tasks

### I want to...

**...understand the project**
→ Read `README.md`

**...see the results**
→ Read `docs/FINAL_REPORT.md` or `docs/RESULTS_SUMMARY.md`

**...understand the code**
→ Read `docs/ARCHITECTURE.md`

**...reproduce the results**
→ Follow `docs/REPRODUCTION_GUIDE.md`

**...understand why we lose money**
→ Read `docs/WHY_WE_LOSE_MONEY.md`

**...contribute**
→ Read `CONTRIBUTING.md`

**...run a backtest**
→ Use `scripts/run_single_backtest.py`

**...modify a strategy**
→ Edit `maker/engine.py` and add tests

---

## 📈 Parameters Quick Reference

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Risk Aversion | γ | 0.1 | Inventory penalty |
| Fill Intensity | λ | 1.0 | Base fill rate |
| OFI Sensitivity | κ | 0.001 | Quote skew |
| Spread Multiplier | η | 0.5 | OFI spread widening |
| Volatility Window | - | 60s | Rolling window |

---

## 🧪 Test Coverage

| Module | Tests | Coverage |
|--------|-------|----------|
| Features | 27 | 100% |
| Engine | 25 | 100% |
| Fills | 26 | 100% |
| Backtest | 24 | 100% |
| Metrics | 39 | 100% |
| **Total** | **141** | **100%** |

---

## 📚 Essential Reading

1. **First Time**: `README.md` → `docs/FINAL_REPORT.md`
2. **Understanding Code**: `docs/ARCHITECTURE.md`
3. **Reproducing**: `docs/REPRODUCTION_GUIDE.md`
4. **Contributing**: `CONTRIBUTING.md`
5. **Economics**: `docs/WHY_WE_LOSE_MONEY.md`

---

## 🆘 Getting Help

- **Documentation**: Check `docs/` folder
- **Code Questions**: See `docs/ARCHITECTURE.md`
- **Issues**: Open GitHub Issue
- **Discussions**: Use GitHub Discussions

---

## ⭐ Star This Repo!

If you find this project useful:
1. Click the ⭐ at the top of the page
2. Share with colleagues/classmates
3. Cite in your own work

---

## 📞 Contact

- **GitHub**: [@xecuterisaquant](https://github.com/xecuterisaquant)
- **Issues**: [Report a bug](https://github.com/xecuterisaquant/ofi-marketmaking-strat/issues)
- **Contribute**: [Pull requests welcome](https://github.com/xecuterisaquant/ofi-marketmaking-strat/pulls)

---

**Last Updated**: December 6, 2025  
**Version**: 2.0.0  
**Status**: Production Ready ✅

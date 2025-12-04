# Anti-Overfitting Protocol

## Critical Design Principles

### 1. No Parameter Optimization on Test Data ❌

**What we DON'T do:**
- ❌ Fit beta coefficient on backtest data
- ❌ Optimize risk aversion (γ) to maximize Sharpe
- ❌ Tune fill model parameters (A, k) for best P&L
- ❌ Adjust OFI windows based on test results
- ❌ Cherry-pick best-performing configurations

**What we DO:**
- ✅ Use beta = 0.036 from **separate replication study** (Jan 2017, 40 symbol-days)
- ✅ Use γ = 0.1 from **Avellaneda-Stoikov literature**
- ✅ Use fill model A=2.0, k=0.5 from **market microstructure theory**
- ✅ Test multiple fixed configurations (defined before seeing results)
- ✅ Report ALL configurations tested (no selective reporting)

### 2. Data Split Strategy 📅

**Available Data:** January 2017 (20 trading days)

**Split:**
- **Week 1 (Jan 3-6, 4 days):** Validation/sanity check
  - Use to verify system works correctly
  - Check for implementation bugs
  - Ensure metrics are reasonable
  - **NO parameter tuning**

- **Weeks 2-4 (Jan 9-27, 16 days):** Out-of-sample test
  - Final performance evaluation
  - All parameters frozen before running
  - Results reported as-is

**Symbols:** 8 tickers (AAPL, AMD, AMZN, JPM, MSFT, NVDA, SPY, TSLA)

### 3. Feature Engineering (No Leakage) 🔍

**Rolling Windows (Correct ✅):**
```python
ofi_rolling = ofi.rolling(window=5, min_periods=1).sum()  # Only looks backward
volatility = prices.ewm(halflife=60).std()                # Only uses past data
```

**Forbidden (❌):**
```python
ofi_future = ofi.shift(-5)  # Look-ahead bias!
vol = prices.rolling(window=60, center=True).std()  # Uses future data!
```

**Our Implementation:**
- ✅ OFI: `rolling()` with no `center=True`
- ✅ Volatility: EWMA (exponential decay of **past** returns)
- ✅ Microprice: Computed from **current** tick only
- ✅ All features use `min_periods` to handle startup gracefully

### 4. Fixed Parameters from Literature 📚

| Parameter | Value | Source |
|-----------|-------|--------|
| **OFI Beta (β)** | 0.036 | Your replication study (separate data) |
| **Risk Aversion (γ)** | 0.1 | Avellaneda-Stoikov (2008) |
| **Fill Intensity (A)** | 2.0 | Market microstructure (50% at touch) |
| **Fill Decay (k)** | 0.5 | Conservative (5bp spread regime) |
| **OFI Windows** | [5, 10, 30]s | Cont et al. (2014) |
| **Vol Halflife** | 60s | Standard HFT practice |
| **Inventory Limit** | ±100 shares | Institutional standard |

**Rationale:** All values from prior research or industry standards, **NOT optimized**.

### 5. Strategy Configurations (Pre-Defined) 🎯

**We will test exactly 4 configurations** (defined NOW, before seeing results):

1. **Symmetric Baseline** (No signals)
   - No OFI signal
   - No microprice tilt
   - Pure Avellaneda-Stoikov (inventory + volatility only)
   - Purpose: Benchmark

2. **Microprice Only**
   - Tilt toward microprice (depth imbalance)
   - No OFI signal
   - Purpose: Test if depth alone helps

3. **OFI Full** (Our main strategy)
   - OFI signal integration
   - Microprice tilt
   - Inventory management
   - Purpose: Test complete system

4. **OFI Ablation** (OFI only, no microprice)
   - OFI signal only
   - No microprice tilt
   - Purpose: Isolate OFI contribution

**We report results for ALL 4** - no cherry-picking.

### 6. Evaluation Metrics (No Fitting) 📊

**Primary Metrics (Pre-Registered):**
1. Sharpe Ratio (annualized)
2. Fill Edge (bps per fill)
3. Adverse Selection (1s/5s/10s post-fill drift)
4. Maximum Drawdown
5. Inventory Variance

**Secondary Metrics:**
- Total P&L
- Sortino Ratio
- Fill Rate
- Time at inventory limits

**All metrics are pure calculations** - no curve fitting or optimization.

### 7. Results Reporting (Full Transparency) 📝

**We will report:**
- ✅ Results for ALL 4 configurations
- ✅ Results for ALL 8 symbols
- ✅ Both positive AND negative findings
- ✅ Statistical significance (if applicable)
- ✅ Failure modes and limitations

**We will NOT:**
- ❌ Report only best configuration
- ❌ Report only best symbols
- ❌ Adjust parameters after seeing poor results
- ❌ Re-run with "better" settings

### 8. Validation Checks 🔍

**Before accepting results:**
1. ✅ Verify no data leakage in feature computation
2. ✅ Check for implementation bugs (unit tests)
3. ✅ Ensure P&L reconciliation (cash + inventory × mid = total P&L)
4. ✅ Verify fills are realistic (not 100% fill rate)
5. ✅ Check inventory doesn't exceed limits
6. ✅ Confirm reproducibility (same seed → same results)

### 9. Known Limitations (Document Honestly) ⚠️

**Our backtest HAS these limitations:**
1. **No transaction costs** - Will overestimate P&L
2. **No exchange fees** - Real trading has fees
3. **No queue position** - Assumes random fill order
4. **No partial fills** - All-or-nothing assumption
5. **No market impact** - Our orders don't move prices
6. **Simplified fill model** - Real fills more complex
7. **Perfect data** - No connectivity issues, misfeeds
8. **Single contract size** - Always 100 shares

**We document these upfront** and acknowledge results are optimistic.

### 10. Walk-Forward Protocol (If We Had More Data) 🚶

**Ideal approach (requires more data):**
1. Training: Jan 2017
2. Validation: Feb 2017  
3. Testing: Mar 2017
4. Out-of-sample: Apr+ 2017

**Our approach (limited data):**
- Week 1: Sanity check only
- Weeks 2-4: Test with frozen parameters
- Acknowledge this is **not ideal** but best we can do with available data

---

## Summary: How We Avoid Overfitting

1. ✅ **Parameters from literature** (not fitted)
2. ✅ **Beta from separate study** (your replication)
3. ✅ **No look-ahead** in features
4. ✅ **Pre-defined configurations** (4 strategies)
5. ✅ **Report all results** (no cherry-picking)
6. ✅ **Pure metric calculations** (no optimization)
7. ✅ **Document limitations** (honest about assumptions)
8. ✅ **Reproducible** (fixed seeds, version control)

**Key Insight:** We're **validating** whether OFI signals help market making, **NOT optimizing** parameters to maximize backtest P&L.

This is the difference between **research** (honest validation) and **data mining** (overfitted garbage).

# Walk Forward Setup
dates = sorted(all_dates)  # 20 days
train_window = 5  # days
test_window = 1   # day

wf_results = []
for i in range(train_window, len(dates)):
    # Train period
    train_dates = dates[i-train_window:i]

    # Optimize parameters on train set (hypothetically)
    # In practice: keep fixed to avoid overfitting
    params_train = {'kappa': 0.001, 'eta': 0.5}  

    # Test on next day
    test_date = dates[i]
    test_result = run_backtest(test_date, params_train)

    wf_results.append({
        'test_date': test_date,
        'train_period': f"{train_dates[0]} to {train_dates[-1]}",
        'pnl': test_result.final_pnl,
        'sharpe': test_result.sharpe_ratio
    })

# Plot out-of-sample performance over time
plot_wf_results(wf_results)
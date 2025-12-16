gammas = [0.05, 0.1, 0.15, 0.2]
kappas = [0.0005, 0.001, 0.0015, 0.002]
etas = [0.25, 0.5, 0.75, 1.0]

results = []
for gamma in gammas:
    for kappa in kappas:
        for eta in etas:
            config = BacktestConfig(risk_aversion=gamma, ofi_kappa=kappa, spread_eta=eta)
            result = run_backtest(config)
            results.append({
                'gamma': gamma, 'kappa': kappa, 'eta': eta,
                'sharpe': result.sharpe_ratio,
                'pnl': result.final_pnl,
                'fills': result.total_fills
            })

# Visualize parameter surface
plot_parameter_surface(results)
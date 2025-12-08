"""
Generate comprehensive visualizations for OFI market making strategy analysis.

This script creates publication-quality figures for:
- Strategy performance comparisons
- PnL distributions
- OFI signal analysis
- Risk metrics
- Statistical significance tests
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def load_batch_results(results_dir: Path = Path('results/batch')) -> pd.DataFrame:
    """Load batch summary from parquet files if CSV doesn't exist."""
    # First try to load from CSV
    batch_files = list(results_dir.glob('batch_summary_*.csv'))
    
    if batch_files:
        # Get most recent file
        latest_file = max(batch_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_csv(latest_file)
        
        # Check if it has enough data
        if len(df) >= 100:  # Should have ~400 for full dataset
            print(f"Loading: {latest_file.name}")
            return df
    
    # If no CSV or insufficient data, load from parquets
    print("Loading from parquet files...")
    detailed_dir = Path('results/detailed')
    parquet_files = list(detailed_dir.glob('*.parquet'))
    
    if not parquet_files:
        raise FileNotFoundError("No parquet or batch summary files found")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Extract summary metrics from each parquet
    summaries = []
    for pq_file in parquet_files:
        # Parse filename: SYMBOL_DATE_STRATEGY.parquet
        parts = pq_file.stem.split('_')
        symbol = parts[0]
        date = parts[1]
        strategy = '_'.join(parts[2:])
        
        # Load parquet and compute metrics
        df_run = pd.read_parquet(pq_file)
        
        if len(df_run) == 0:
            continue
        
        # Calculate fills and metrics
        fills = df_run[(df_run['our_bid'].notna() | df_run['our_ask'].notna()) & 
                      (df_run['inventory'].diff() != 0)].copy()
        
        n_fills = len(fills)
        total_pnl = df_run['pnl'].iloc[-1] if len(df_run) > 0 else 0
        
        # Calculate returns for Sharpe
        returns = df_run['pnl'].diff().fillna(0)
        sharpe = (returns.mean() / returns.std() * np.sqrt(252 * 6.5 * 3600)) if returns.std() > 0 else 0
        
        # Fill edge calculation (approximation)
        if n_fills > 0:
            avg_fill_edge = abs(fills['pnl'].diff().mean()) if 'pnl' in fills.columns else 0
        else:
            avg_fill_edge = 0
            
        summary = {
            'symbol': symbol,
            'date': date,
            'strategy': strategy,
            'total_pnl': total_pnl,
            'total_fills': n_fills,
            'sharpe_ratio': sharpe,
            'max_inventory': df_run['inventory'].max() if 'inventory' in df_run.columns else 0,
            'min_inventory': df_run['inventory'].min() if 'inventory' in df_run.columns else 0,
            'avg_fill_edge_bps': avg_fill_edge,
            'fill_edge_std_bps': 0,  # Placeholder
            'max_drawdown': (df_run['pnl'] - df_run['pnl'].cummax()).min(),
            'total_return': total_pnl,
        }
        summaries.append(summary)
    
    df = pd.DataFrame(summaries)
    print(f"Loaded {len(df)} backtest results from parquets")
    
    return df


def plot_strategy_comparison(df: pd.DataFrame, save_dir: Path):
    """Create comprehensive strategy comparison visualizations."""
    
    # Figure 1: PnL Box Plot Comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Strategy Performance Comparison Across 400 Backtests', 
                 fontsize=14, fontweight='bold')
    
    # 1a. Final PnL Distribution
    ax = axes[0, 0]
    strategies = df['strategy'].unique()
    pnl_data = [df[df['strategy'] == s]['total_pnl'].values for s in strategies]
    bp = ax.boxplot(pnl_data, labels=strategies, patch_artist=True)
    
    # Color boxes
    colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Final PnL ($)', fontweight='bold')
    ax.set_title('(a) PnL Distribution', fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=15)
    
    # Add mean markers
    for i, s in enumerate(strategies):
        mean_val = df[df['strategy'] == s]['total_pnl'].mean()
        ax.plot(i+1, mean_val, 'r*', markersize=10, label='Mean' if i == 0 else '')
    
    # 1b. Sharpe Ratio Comparison
    ax = axes[0, 1]
    sharpe_data = [df[df['strategy'] == s]['sharpe_ratio'].values for s in strategies]
    bp = ax.boxplot(sharpe_data, labels=strategies, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Sharpe Ratio', fontweight='bold')
    ax.set_title('(b) Risk-Adjusted Performance', fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=15)
    
    # 1c. Fill Count Comparison
    ax = axes[1, 0]
    fill_data = [df[df['strategy'] == s]['total_fills'].values for s in strategies]
    bp = ax.boxplot(fill_data, labels=strategies, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Number of Fills', fontweight='bold')
    ax.set_title('(c) Trading Activity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=15)
    
    # 1d. Average Fill Edge
    ax = axes[1, 1]
    spread_data = [df[df['strategy'] == s]['avg_fill_edge_bps'].values for s in strategies]
    bp = ax.boxplot(spread_data, labels=strategies, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Average Fill Edge (bps)', fontweight='bold')
    ax.set_title('(d) Fill Quality', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=15)
    
    plt.tight_layout()
    save_path = save_dir / 'fig1_strategy_comparison.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Saved: {save_path.name}")
    plt.close()


def plot_pnl_distributions(df: pd.DataFrame, save_dir: Path):
    """Create detailed PnL distribution analysis."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('PnL Distribution Analysis', fontsize=14, fontweight='bold')
    
    strategies = df['strategy'].unique()
    colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']
    
    # 2a. Overlapping Histograms
    ax = axes[0, 0]
    for strategy, color in zip(strategies, colors):
        data = df[df['strategy'] == strategy]['total_pnl']
        ax.hist(data, bins=30, alpha=0.5, label=strategy, color=color, edgecolor='black')
    
    ax.set_xlabel('Final PnL ($)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('(a) PnL Histogram', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.grid(True, alpha=0.3)
    
    # 2b. Kernel Density Estimation
    ax = axes[0, 1]
    for strategy, color in zip(strategies, colors):
        data = df[df['strategy'] == strategy]['total_pnl']
        data.plot.kde(ax=ax, label=strategy, color=color, linewidth=2)
    
    ax.set_xlabel('Final PnL ($)', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')
    ax.set_title('(b) Kernel Density Estimate', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.grid(True, alpha=0.3)
    
    # 2c. Cumulative Distribution
    ax = axes[1, 0]
    for strategy, color in zip(strategies, colors):
        data = df[df['strategy'] == strategy]['total_pnl'].sort_values()
        cumulative = np.arange(1, len(data) + 1) / len(data)
        ax.plot(data, cumulative, label=strategy, color=color, linewidth=2)
    
    ax.set_xlabel('Final PnL ($)', fontweight='bold')
    ax.set_ylabel('Cumulative Probability', fontweight='bold')
    ax.set_title('(c) Cumulative Distribution Function', fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.grid(True, alpha=0.3)
    
    # 2d. Q-Q Plot (Normal comparison)
    ax = axes[1, 1]
    for strategy, color in zip(strategies[:2], colors[:2]):  # Compare top 2
        data = df[df['strategy'] == strategy]['total_pnl']
        stats.probplot(data, dist="norm", plot=ax)
        ax.get_lines()[-2].set_color(color)
        ax.get_lines()[-2].set_label(strategy)
        ax.get_lines()[-1].set_linestyle('--')
    
    ax.set_xlabel('Theoretical Quantiles', fontweight='bold')
    ax.set_ylabel('Sample Quantiles', fontweight='bold')
    ax.set_title('(d) Q-Q Plot (Normality Test)', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = save_dir / 'fig2_pnl_distributions.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Saved: {save_path.name}")
    plt.close()


def plot_improvement_analysis(df: pd.DataFrame, save_dir: Path):
    """Analyze and visualize OFI strategy improvements."""
    
    # Calculate improvements vs baseline
    baseline = df[df['strategy'] == 'symmetric_baseline']
    
    improvements = []
    for symbol in df['symbol'].unique():
        for date in df['date'].unique():
            baseline_pnl = baseline[(baseline['symbol'] == symbol) & 
                                   (baseline['date'] == date)]['total_pnl'].values
            
            if len(baseline_pnl) == 0:
                continue
            
            baseline_pnl = baseline_pnl[0]
            
            for strategy in ['microprice_only', 'ofi_ablation', 'ofi_full']:
                strat_data = df[(df['strategy'] == strategy) & 
                              (df['symbol'] == symbol) & 
                              (df['date'] == date)]
                
                if len(strat_data) == 0:
                    continue
                
                strat_pnl = strat_data['total_pnl'].values[0]
                
                # Calculate improvement (higher PnL = better, less negative = improvement)
                # For losses: baseline=-3352, strat=-1234 → improvement = (strat - baseline) / |baseline| * 100
                #           = (-1234 - (-3352)) / 3352 * 100 = 2118/3352 * 100 = +63.2%
                if baseline_pnl != 0:
                    improvement_pct = ((strat_pnl - baseline_pnl) / abs(baseline_pnl)) * 100
                else:
                    improvement_pct = 0
                
                improvements.append({
                    'symbol': symbol,
                    'date': date,
                    'strategy': strategy,
                    'baseline_pnl': baseline_pnl,
                    'strategy_pnl': strat_pnl,
                    'improvement_pct': improvement_pct,
                    'absolute_improvement': strat_pnl - baseline_pnl  # Positive = improvement
                })
    
    imp_df = pd.DataFrame(improvements)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('OFI Strategy Improvement Analysis', fontsize=14, fontweight='bold')
    
    # 3a. Improvement Distribution by Strategy
    ax = axes[0, 0]
    strategies = ['microprice_only', 'ofi_ablation', 'ofi_full']
    colors = ['#ffcc99', '#99ccff', '#99ff99']
    
    imp_data = [imp_df[imp_df['strategy'] == s]['improvement_pct'].values 
                for s in strategies]
    bp = ax.boxplot(imp_data, labels=strategies, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.set_ylabel('Improvement vs Baseline (%)', fontweight='bold')
    ax.set_title('(a) Improvement Distribution', fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=15)
    
    # Add mean values as text
    for i, s in enumerate(strategies):
        mean_imp = imp_df[imp_df['strategy'] == s]['improvement_pct'].mean()
        ax.text(i+1, ax.get_ylim()[1]*0.9, f'μ={mean_imp:.1f}%', 
               ha='center', fontsize=8, fontweight='bold')
    
    # 3b. Improvement by Symbol - IMPROVED with grouped bars and better layout
    ax = axes[0, 1]
    symbols = sorted(imp_df['symbol'].unique())
    x = np.arange(len(symbols))
    width = 0.25
    
    # Calculate mean improvement per symbol per strategy
    symbol_means = {}
    for strategy in strategies:
        means = []
        for sym in symbols:
            sym_data = imp_df[(imp_df['strategy'] == strategy) & (imp_df['symbol'] == sym)]
            if len(sym_data) > 0:
                means.append(sym_data['improvement_pct'].mean())
            else:
                means.append(0)
        symbol_means[strategy] = means
    
    # Plot grouped bars
    for i, (strategy, color) in enumerate(zip(strategies, colors)):
        offset = (i - 1) * width  # Center the middle bar
        bars = ax.bar(x + offset, symbol_means[strategy], width, 
                     label=strategy.replace('_', ' ').title(), 
                     color=color, alpha=0.85, edgecolor='black', linewidth=0.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 2:  # Only label if visible
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}%', ha='center', 
                       va='bottom' if height > 0 else 'top',
                       fontsize=7, fontweight='bold')
    
    ax.set_xlabel('Symbol', fontweight='bold')
    ax.set_ylabel('Mean Improvement (%)', fontweight='bold')
    ax.set_title('(b) Improvement by Symbol', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(symbols, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3c. Scatter: Strategy PnL vs Improvement - IMPROVED with better visualization
    ax = axes[1, 0]
    
    # Create scatter with better separation and styling
    for strategy, color in zip(strategies, colors):
        data = imp_df[imp_df['strategy'] == strategy]
        
        # Add jitter to prevent overplotting
        jitter_x = np.random.normal(0, 1, size=len(data))
        jitter_y = np.random.normal(0, 0.5, size=len(data))
        
        ax.scatter(data['strategy_pnl'] + jitter_x, 
                  data['improvement_pct'] + jitter_y, 
                  alpha=0.5, s=40, label=strategy.replace('_', ' ').title(), 
                  color=color, edgecolors='black', linewidth=0.3)
        
        # Add mean marker
        mean_pnl = data['strategy_pnl'].mean()
        mean_imp = data['improvement_pct'].mean()
        ax.scatter(mean_pnl, mean_imp, s=200, color=color, 
                  marker='*', edgecolors='black', linewidth=1.5, 
                  zorder=10, label=f'{strategy} (mean)')
    
    ax.set_xlabel('Strategy PnL ($)', fontweight='bold')
    ax.set_ylabel('Improvement (%)', fontweight='bold')
    ax.set_title('(c) Strategy Performance vs Improvement', fontweight='bold')
    ax.legend(fontsize=6, loc='best', ncol=2)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)
    
    # Add quadrant labels
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.text(xlim[0] + (xlim[1]-xlim[0])*0.05, ylim[1]*0.9, 
           'Better Performance\n& Improvement', 
           fontsize=7, alpha=0.6, style='italic')
    
    # 3d. Win Rate (% of runs with positive improvement)
    ax = axes[1, 1]
    win_rates = []
    for strategy in strategies:
        data = imp_df[imp_df['strategy'] == strategy]
        win_rate = (data['improvement_pct'] > 0).sum() / len(data) * 100
        win_rates.append(win_rate)
    
    bars = ax.bar(strategies, win_rates, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Win Rate (%)', fontweight='bold')
    ax.set_title('(d) Consistency (% Runs Better than Baseline)', fontweight='bold')
    ax.set_ylim([0, 100])
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1, 
              label='50% (Random)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=15)
    
    # Add value labels on bars
    for bar, rate in zip(bars, win_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    save_path = save_dir / 'fig3_improvement_analysis.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Saved: {save_path.name}")
    plt.close()
    
    return imp_df


def plot_statistical_tests(df: pd.DataFrame, save_dir: Path):
    """Create statistical significance visualizations."""
    
    baseline = df[df['strategy'] == 'symmetric_baseline']['total_pnl'].values
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Statistical Significance Testing', fontsize=14, fontweight='bold')
    
    strategies = ['microprice_only', 'ofi_ablation', 'ofi_full']
    colors = ['#ffcc99', '#99ccff', '#99ff99']
    
    # 4a. T-Test Results
    ax = axes[0, 0]
    t_stats = []
    p_values = []
    
    for strategy in strategies:
        strategy_pnl = df[df['strategy'] == strategy]['total_pnl'].values
        t_stat, p_val = stats.ttest_ind(baseline, strategy_pnl)
        t_stats.append(t_stat)
        p_values.append(p_val)
    
    x = np.arange(len(strategies))
    bars = ax.bar(x, np.abs(t_stats), color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('|t-statistic|', fontweight='bold')
    ax.set_title('(a) Two-Sample t-Test Results', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15)
    ax.axhline(y=1.96, color='red', linestyle='--', alpha=0.7, linewidth=1.5,
              label='95% CI (|t|=1.96)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add p-value labels
    for i, (bar, p_val) in enumerate(zip(bars, p_values)):
        height = bar.get_height()
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'p<0.001\n{sig}' if p_val < 0.001 else f'p={p_val:.3f}\n{sig}',
               ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    # 4b. Effect Size (Cohen's d)
    ax = axes[0, 1]
    effect_sizes = []
    
    for strategy in strategies:
        strategy_pnl = df[df['strategy'] == strategy]['total_pnl'].values
        mean_diff = np.mean(baseline) - np.mean(strategy_pnl)
        pooled_std = np.sqrt((np.var(baseline) + np.var(strategy_pnl)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std != 0 else 0
        effect_sizes.append(cohens_d)
    
    bars = ax.bar(x, effect_sizes, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel("Cohen's d", fontweight='bold')
    ax.set_title("(b) Effect Size (Cohen's d)", fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15)
    ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Small (0.2)')
    ax.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='Medium (0.5)')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Large (0.8)')
    ax.legend(fontsize=6, loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4c. Confidence Intervals
    ax = axes[1, 0]
    all_strategies = ['symmetric_baseline'] + strategies
    all_colors = ['#ff9999'] + colors
    
    means = []
    cis = []
    
    for strategy in all_strategies:
        data = df[df['strategy'] == strategy]['total_pnl'].values
        mean = np.mean(data)
        se = stats.sem(data)
        ci = se * stats.t.ppf((1 + 0.95) / 2, len(data) - 1)
        means.append(mean)
        cis.append(ci)
    
    y_pos = np.arange(len(all_strategies))
    ax.barh(y_pos, means, xerr=cis, color=all_colors, alpha=0.8, 
           edgecolor='black', capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_strategies, fontsize=8)
    ax.set_xlabel('Mean PnL ($) ± 95% CI', fontweight='bold')
    ax.set_title('(c) Mean PnL with Confidence Intervals', fontweight='bold')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # 4d. Wilcoxon Signed-Rank Test (paired, non-parametric)
    ax = axes[1, 1]
    w_stats = []
    w_p_values = []
    
    # Match pairs for each symbol-date
    for strategy in strategies:
        pairs_base = []
        pairs_strat = []
        
        for symbol in df['symbol'].unique():
            for date in df['date'].unique():
                base_val = df[(df['strategy'] == 'symmetric_baseline') & 
                            (df['symbol'] == symbol) & 
                            (df['date'] == date)]['total_pnl'].values
                strat_val = df[(df['strategy'] == strategy) & 
                             (df['symbol'] == symbol) & 
                             (df['date'] == date)]['total_pnl'].values
                
                if len(base_val) > 0 and len(strat_val) > 0:
                    pairs_base.append(base_val[0])
                    pairs_strat.append(strat_val[0])
        
        if len(pairs_base) > 0:
            w_stat, w_p = stats.wilcoxon(pairs_base, pairs_strat)
            w_stats.append(w_stat)
            w_p_values.append(w_p)
        else:
            w_stats.append(0)
            w_p_values.append(1)
    
    bars = ax.bar(x, w_stats, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Wilcoxon W Statistic', fontweight='bold')
    ax.set_title('(d) Wilcoxon Signed-Rank Test (Paired)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add p-value labels
    for i, (bar, p_val) in enumerate(zip(bars, w_p_values)):
        height = bar.get_height()
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'p<0.001\n{sig}' if p_val < 0.001 else f'p={p_val:.3f}\n{sig}',
               ha='center', va='bottom', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    save_path = save_dir / 'fig4_statistical_tests.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Saved: {save_path.name}")
    plt.close()


def plot_risk_metrics(df: pd.DataFrame, save_dir: Path):
    """Visualize risk-adjusted performance metrics."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Risk-Adjusted Performance Metrics', fontsize=14, fontweight='bold')
    
    strategies = df['strategy'].unique()
    colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']
    
    # 5a. Sharpe Ratio Comparison
    ax = axes[0, 0]
    sharpe_means = [df[df['strategy'] == s]['sharpe_ratio'].mean() for s in strategies]
    sharpe_stds = [df[df['strategy'] == s]['sharpe_ratio'].std() for s in strategies]
    
    x = np.arange(len(strategies))
    bars = ax.bar(x, sharpe_means, yerr=sharpe_stds, color=colors, alpha=0.8,
                  edgecolor='black', capsize=5)
    ax.set_ylabel('Sharpe Ratio', fontweight='bold')
    ax.set_title('(a) Mean Sharpe Ratio ± Std Dev', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean in zip(bars, sharpe_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{mean:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 5b. Maximum Drawdown
    ax = axes[0, 1]
    if 'max_drawdown' in df.columns:
        dd_data = [df[df['strategy'] == s]['max_drawdown'].values for s in strategies]
        bp = ax.boxplot(dd_data, labels=strategies, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Maximum Drawdown ($)', fontweight='bold')
        ax.set_title('(b) Maximum Drawdown Distribution', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=15)
    
    # 5c. Return/Risk Scatter
    ax = axes[1, 0]
    for strategy, color in zip(strategies, colors):
        data = df[df['strategy'] == strategy]
        returns = data['total_pnl'].mean()
        risk = data['total_pnl'].std()
        ax.scatter(risk, returns, s=200, alpha=0.7, color=color, 
                  edgecolor='black', linewidth=2, label=strategy)
        ax.annotate(strategy, (risk, returns), fontsize=7, ha='right')
    
    ax.set_xlabel('Risk (Std Dev of PnL, $)', fontweight='bold')
    ax.set_ylabel('Return (Mean PnL, $)', fontweight='bold')
    ax.set_title('(c) Risk-Return Profile', fontweight='bold')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc='best')
    
    # 5d. Fill Efficiency (PnL per fill)
    ax = axes[1, 1]
    efficiency = []
    for strategy in strategies:
        data = df[df['strategy'] == strategy]
        mean_pnl = data['total_pnl'].mean()
        mean_fills = data['total_fills'].mean()
        pnl_per_fill = mean_pnl / mean_fills if mean_fills > 0 else 0
        efficiency.append(pnl_per_fill)
    
    bars = ax.bar(x, efficiency, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('PnL per Fill ($)', fontweight='bold')
    ax.set_title('(d) Fill Efficiency', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, eff in zip(bars, efficiency):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{eff:.2f}', ha='center', va='bottom' if eff >= 0 else 'top', 
               fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    save_path = save_dir / 'fig5_risk_metrics.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✅ Saved: {save_path.name}")
    plt.close()


def create_summary_table(df: pd.DataFrame, save_dir: Path):
    """Create publication-quality summary statistics table."""
    
    strategies = df['strategy'].unique()
    
    # Calculate statistics
    summary_data = []
    for strategy in strategies:
        data = df[df['strategy'] == strategy]
        
        summary_data.append({
            'Strategy': strategy,
            'N': len(data),
            'Mean PnL': f"${data['total_pnl'].mean():.2f}",
            'Std Dev': f"${data['total_pnl'].std():.2f}",
            'Sharpe': f"{data['sharpe_ratio'].mean():.3f}",
            'Min PnL': f"${data['total_pnl'].min():.2f}",
            'Max PnL': f"${data['total_pnl'].max():.2f}",
            'Avg Fills': f"{data['total_fills'].mean():.1f}",
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create table figure
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values, colLabels=summary_df.columns,
                    cellLoc='center', loc='center', 
                    colColours=['#f0f0f0']*len(summary_df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows
    colors = ['#ff9999', '#ffcc99', '#99ccff', '#99ff99']
    for i in range(len(summary_df)):
        table[(i+1, 0)].set_facecolor(colors[i % len(colors)])
        table[(i+1, 0)].set_text_props(weight='bold')
    
    plt.title('Summary Statistics by Strategy', fontsize=13, fontweight='bold', pad=20)
    
    save_path = save_dir / 'table1_summary_statistics.png'
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"✅ Saved: {save_path.name}")
    plt.close()
    
    return summary_df


def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("GENERATING COMPREHENSIVE VISUALIZATIONS")
    print("="*80 + "\n")
    
    # Setup directories
    results_dir = Path('results/batch')
    save_dir = Path('figures')
    save_dir.mkdir(exist_ok=True)
    
    # Load data
    print("📊 Loading batch results...")
    df = load_batch_results(results_dir)
    print(f"   Loaded {len(df)} backtest results")
    print(f"   Strategies: {df['strategy'].unique()}")
    print(f"   Symbols: {df['symbol'].unique()}")
    print(f"   Dates: {len(df['date'].unique())} days\n")
    
    # Generate visualizations
    print("📈 Generating visualizations...\n")
    
    plot_strategy_comparison(df, save_dir)
    plot_pnl_distributions(df, save_dir)
    imp_df = plot_improvement_analysis(df, save_dir)
    plot_statistical_tests(df, save_dir)
    plot_risk_metrics(df, save_dir)
    summary_df = create_summary_table(df, save_dir)
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80 + "\n")
    print(summary_df.to_string(index=False))
    
    # Print improvement statistics
    print("\n" + "="*80)
    print("IMPROVEMENT vs BASELINE")
    print("="*80 + "\n")
    for strategy in ['microprice_only', 'ofi_ablation', 'ofi_full']:
        data = imp_df[imp_df['strategy'] == strategy]['improvement_pct']
        print(f"{strategy:20s}: {data.mean():6.2f}% ± {data.std():5.2f}% "
              f"(min: {data.min():6.2f}%, max: {data.max():6.2f}%)")
    
    print("\n" + "="*80)
    print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY!")
    print("="*80 + "\n")
    print(f"📁 Saved to: {save_dir.absolute()}")
    print("\nFigures created:")
    for fig_file in sorted(save_dir.glob('*.png')):
        print(f"   ✅ {fig_file.name}")
    print()


if __name__ == "__main__":
    main()

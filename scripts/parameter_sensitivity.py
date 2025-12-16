#!/usr/bin/env python3
"""
Parameter Sensitivity Analysis

Tests strategy robustness across parameter variations.
Generates heatmaps showing performance surface over parameter space.

Usage:
    python scripts/parameter_sensitivity.py --quick  # Fast test (3×3 grid)
    python scripts/parameter_sensitivity.py --full   # Full analysis (6×6 grid)
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_backtest import run_single_backtest


def create_sensitivity_configs(base_config_path: str, output_dir: str, param_grid: dict):
    """
    Create config files for parameter sensitivity testing.
    
    Parameters
    ----------
    base_config_path : str
        Path to base configuration file
    output_dir : str
        Directory to save generated configs
    param_grid : dict
        Parameter grid to test, e.g., {'risk_aversion': [0.05, 0.1, 0.15]}
    
    Returns
    -------
    list
        Paths to generated config files with metadata
    """
    # Load base config
    with open(base_config_path, 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    configs = []
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = [param_grid[name] for name in param_names]
    
    # Cartesian product
    import itertools
    for combo in itertools.product(*param_values):
        # Create config copy
        config = base_config.copy()
        
        # Update parameters
        param_dict = dict(zip(param_names, combo))
        for param_name, param_value in param_dict.items():
            # Navigate nested structure
            if param_name == 'risk_aversion':
                config['quoting']['risk_aversion'] = param_value
            elif param_name == 'signal_adjustment_factor':
                config['quoting']['signal_adjustment_factor'] = param_value
            elif param_name == 'base_intensity':
                config['fill_model']['base_intensity'] = param_value
            elif param_name == 'decay_rate':
                config['fill_model']['decay_rate'] = param_value
        
        # Save config
        config_name = '_'.join([f"{k}={v:.4f}" for k, v in param_dict.items()])
        config_path = output_path / f"sensitivity_{config_name}.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        configs.append({
            'path': str(config_path),
            'params': param_dict
        })
    
    return configs


def run_sensitivity_analysis(configs: list, test_symbols: list = ['AAPL'], test_dates: list = ['2017-01-03']):
    """
    Run backtests across parameter grid.
    
    Parameters
    ----------
    configs : list
        List of config dicts with 'path' and 'params'
    test_symbols : list
        Symbols to test
    test_dates : list
        Dates to test
    
    Returns
    -------
    pd.DataFrame
        Results with parameters and performance metrics
    """
    results = []
    
    total_runs = len(configs) * len(test_symbols) * len(test_dates)
    print(f"Running {total_runs} backtests ({len(configs)} configs × {len(test_symbols)} symbols × {len(test_dates)} dates)\n")
    
    with tqdm(total=total_runs) as pbar:
        for config_info in configs:
            config_path = config_info['path']
            params = config_info['params']
            
            for symbol in test_symbols:
                for date in test_dates:
                    try:
                        result = run_single_backtest(symbol, date, config_path, verbose=False)
                        
                        if result:
                            results.append({
                                **params,
                                'symbol': symbol,
                                'date': date,
                                'pnl': result['total_pnl'],
                                'sharpe': result['sharpe_ratio'],
                                'fills': result['total_fills'],
                                'volatility': result['pnl_std']
                            })
                    except Exception as e:
                        print(f"Error: {symbol} {date} - {e}")
                    
                    pbar.update(1)
    
    return pd.DataFrame(results)


def plot_sensitivity_heatmap(df: pd.DataFrame, param1: str, param2: str, metric: str = 'pnl', 
                             base_values: dict = None, output_path: str = 'figures/sensitivity.png'):
    """
    Generate heatmap of metric vs two parameters.
    
    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    param1 : str
        First parameter (x-axis)
    param2 : str
        Second parameter (y-axis)
    metric : str
        Metric to plot (default: 'pnl')
    base_values : dict
        Baseline parameter values to mark with star
    output_path : str
        Path to save figure
    """
    # Aggregate across symbols/dates
    pivot_data = df.groupby([param1, param2])[metric].mean().reset_index()
    pivot_table = pivot_data.pivot(index=param2, columns=param1, values=metric)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(pivot_table, annot=True, fmt='.0f', cmap='RdYlGn', center=0,
                cbar_kws={'label': metric.upper()})
    
    # Mark baseline if provided
    if base_values and param1 in base_values and param2 in base_values:
        # Find closest values in grid
        p1_vals = sorted(pivot_table.columns)
        p2_vals = sorted(pivot_table.index)
        
        p1_idx = min(range(len(p1_vals)), key=lambda i: abs(p1_vals[i] - base_values[param1]))
        p2_idx = min(range(len(p2_vals)), key=lambda i: abs(p2_vals[i] - base_values[param2]))
        
        plt.plot(p1_idx + 0.5, p2_idx + 0.5, 'r*', markersize=20, 
                label=f'Hand-calibrated\n({base_values[param1]:.3f}, {base_values[param2]:.3f})')
        plt.legend(loc='upper right')
    
    plt.title(f'{metric.upper()} vs {param1} and {param2}')
    plt.xlabel(param1.replace('_', ' ').title())
    plt.ylabel(param2.replace('_', ' ').title())
    plt.tight_layout()
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved heatmap: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Parameter sensitivity analysis')
    parser.add_argument('--quick', action='store_true', help='Quick test (3×3 grid, 1 symbol-date)')
    parser.add_argument('--full', action='store_true', help='Full analysis (6×6 grid, all symbols)')
    parser.add_argument('--base-config', type=str, default='configs/ofi_ablation.yaml',
                       help='Base configuration file')
    
    args = parser.parse_args()
    
    if args.quick:
        # Quick test: 3×3 grid, 1 symbol-date
        param_grid = {
            'risk_aversion': [0.05, 0.063, 0.08],
            'signal_adjustment_factor': [0.05, 0.075, 0.10]
        }
        test_symbols = ['AAPL']
        test_dates = ['2017-01-03', '2017-01-04']
        grid_name = 'quick'
    elif args.full:
        # Full analysis: 6×6 grid, multiple symbols
        param_grid = {
            'risk_aversion': [0.04, 0.05, 0.063, 0.08, 0.10, 0.12],
            'signal_adjustment_factor': [0.0, 0.05, 0.075, 0.10, 0.125, 0.15]
        }
        test_symbols = ['AAPL', 'AMD', 'AMZN']
        test_dates = ['2017-01-03', '2017-01-04', '2017-01-05']
        grid_name = 'full'
    else:
        print("❌ Must specify --quick or --full")
        return
    
    print(f"Parameter Sensitivity Analysis ({grid_name.upper()} mode)")
    print("="*80)
    print(f"Base config: {args.base_config}")
    print(f"Parameter grid:")
    for param, values in param_grid.items():
        print(f"  {param}: {values}")
    print(f"Test data: {len(test_symbols)} symbols × {len(test_dates)} dates")
    print("="*80 + "\n")
    
    # Generate configs
    print("Generating configuration files...")
    configs = create_sensitivity_configs(
        args.base_config,
        f'configs/sensitivity_{grid_name}',
        param_grid
    )
    print(f"✅ Created {len(configs)} configurations\n")
    
    # Run backtests
    print("Running backtests...")
    results_df = run_sensitivity_analysis(configs, test_symbols, test_dates)
    
    # Save results
    results_path = f'results/parameter_sensitivity_{grid_name}.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✅ Results saved: {results_path}\n")
    
    # Print summary
    print("="*80)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("="*80)
    print(f"\nMean PnL by parameter combination:")
    summary = results_df.groupby(['risk_aversion', 'signal_adjustment_factor'])['pnl'].agg(['mean', 'std', 'count'])
    print(summary)
    
    # Generate heatmaps
    print("\nGenerating visualizations...")
    
    # Baseline values (from ofi_ablation config)
    baseline = {
        'risk_aversion': 0.063,
        'signal_adjustment_factor': 0.075
    }
    
    # PnL heatmap
    plot_sensitivity_heatmap(
        results_df,
        'risk_aversion',
        'signal_adjustment_factor',
        metric='pnl',
        base_values=baseline,
        output_path=f'figures/sensitivity_{grid_name}_pnl.png'
    )
    
    # Fill count heatmap
    plot_sensitivity_heatmap(
        results_df,
        'risk_aversion',
        'signal_adjustment_factor',
        metric='fills',
        base_values=baseline,
        output_path=f'figures/sensitivity_{grid_name}_fills.png'
    )
    
    print("\n" + "="*80)
    print("✅ SENSITIVITY ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults show performance is {'ROBUST' if results_df['pnl'].std() < 1000 else 'SENSITIVE'} to parameter variations")
    print(f"Coefficient of variation: {results_df['pnl'].std() / abs(results_df['pnl'].mean()):.2f}")


if __name__ == '__main__':
    main()

"""
Main backtest execution script for OFI market making strategies.

Runs backtests across multiple strategies, symbols, and dates.
Saves results to Parquet files for analysis.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from maker.config import load_config
from maker.engine import QuotingEngine, QuotingParams, QuoteState
from maker.fills import ParametricFillModel
from maker.backtest import BacktestEngine, BacktestConfig
from maker.metrics import compute_all_metrics


def load_data_for_symbol_date(symbol: str, date: str, data_dir: str = "data"):
    """
    Load NBBO data for a specific symbol-date.
    
    Uses the fixed loading approach that filters BEFORE resampling
    to avoid mixing multiple symbols.
    """
    from src.ofi_utils import read_rda, resolve_columns, parse_trading_day_from_filename, build_tob_series_1s
    
    # Construct file path: data/NBBO/{date}.rda
    file_path = Path(data_dir) / "NBBO" / f"{date}.rda"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load raw RDA file using ofi_utils
    df_raw = read_rda(str(file_path))
    
    # Get column mapping (pass DataFrame, not list)
    cmap = resolve_columns(df_raw)
    
    # Extract trading day
    trading_day = parse_trading_day_from_filename(file_path.name)
    
    # Filter to specific symbol BEFORE resampling
    if symbol not in df_raw['sym_root'].values:
        raise ValueError(f"Symbol {symbol} not found in {file_path}")
    
    df_symbol = df_raw[df_raw['sym_root'] == symbol].copy()
    
    # Build 1-second NBBO series for this symbol only
    nbbo_df = build_tob_series_1s(df_symbol, cmap, trading_day, freq='1s')
    
    return nbbo_df


def run_single_backtest(symbol: str, date: str, config_path: str, 
                       data_dir: str = "data", verbose: bool = False):
    """
    Run backtest for single symbol-date-strategy combination.
    
    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., 'AAPL')
    date : str
        Trading date in YYYY-MM-DD format
    config_path : str
        Path to strategy config YAML file
    data_dir : str
        Directory containing NBBO data
    verbose : bool
        Print progress messages
    
    Returns
    -------
    dict
        Backtest results and metrics
    """
    if verbose:
        print(f"  Loading config: {Path(config_path).stem}")
    
    # Load configuration
    config = load_config(config_path)
    
    if verbose:
        print(f"  Loading NBBO data: {symbol} {date}")
    
    # Load NBBO data for this symbol-date
    try:
        nbbo_df = load_data_for_symbol_date(symbol, date, data_dir)
    except Exception as e:
        print(f"  ❌ Failed to load data: {e}")
        return None
    
    if nbbo_df is None or len(nbbo_df) == 0:
        print(f"  ❌ No data available for {symbol} {date}")
        return None
    
    # Rename columns to match backtest engine expectations
    nbbo_df = nbbo_df.rename(columns={
        'best_bid': 'bid',
        'best_ask': 'ask',
        'best_bidsiz': 'bid_sz',
        'best_asksiz': 'ask_sz',
    })
    
    if verbose:
        print(f"  Running backtest ({len(nbbo_df)} rows)...")
    
    # Create backtest config
    backtest_config = BacktestConfig(
        quoting_params=QuotingParams(
            risk_aversion=config.risk_aversion,
            terminal_time=config.terminal_time,
            order_arrival_rate=config.order_arrival_rate,
            max_inventory=config.max_inventory,
            min_inventory=config.min_inventory,
            tick_size=config.tick_size,
            min_spread_bps=config.min_spread_bps,
            signal_adjustment_factor=config.signal_adjustment_factor,
            inventory_urgency_factor=config.inventory_urgency_factor,
        ),
        fill_model=ParametricFillModel(
            intensity_at_touch=config.base_intensity,
            decay_rate=config.decay_rate,
            time_step=1.0,
        ),
        ofi_windows=[config.ofi_horizon_seconds],
        volatility_window=int(config.volatility_halflife_seconds),
        initial_cash=config.initial_cash,
        initial_inventory=config.initial_inventory,
        max_inventory=config.max_inventory,
        min_inventory=config.min_inventory,
        random_seed=config.random_seed,
    )
    
    # Run backtest
    engine = BacktestEngine(backtest_config)
    result = engine.run_single_day(
        nbbo_data=nbbo_df,
        symbol=symbol,
        date=pd.to_datetime(date)
    )
    
    if verbose:
        print(f"  Computing metrics...")
    
    # Compute metrics
    metrics = compute_all_metrics(
        result,
        inventory_limit=config.max_inventory,
        beta_from_replication=config.ofi_beta
    )
    
    if verbose:
        print(f"  ✅ Complete: {len(result.fills)} fills, P&L=${result.final_pnl:.2f}")
    
    return {
        'symbol': symbol,
        'date': date,
        'config_name': config.name,
        'backtest_result': result,
        'metrics': metrics,
    }


def run_batch_backtests(symbols: list, dates: list, config_paths: list,
                        data_dir: str = "data", output_dir: str = "results",
                        verbose: bool = True):
    """
    Run backtests for multiple symbol-date-strategy combinations.
    
    Parameters
    ----------
    symbols : list
        List of stock symbols
    dates : list
        List of trading dates (YYYY-MM-DD format)
    config_paths : list
        List of strategy config file paths
    data_dir : str
        Directory containing NBBO data
    output_dir : str
        Directory to save results
    verbose : bool
        Print progress messages
    
    Returns
    -------
    pd.DataFrame
        Summary of all backtest results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    total_runs = len(symbols) * len(dates) * len(config_paths)
    
    print(f"\n{'='*80}")
    print(f"OFI Market Making Backtest Execution")
    print(f"{'='*80}")
    print(f"Symbols: {len(symbols)} ({', '.join(symbols[:3])}{'...' if len(symbols) > 3 else ''})")
    print(f"Dates: {len(dates)} ({dates[0]} to {dates[-1]})")
    print(f"Strategies: {len(config_paths)}")
    print(f"Total Runs: {total_runs}")
    print(f"{'='*80}\n")
    
    with tqdm(total=total_runs, desc="Running backtests") as pbar:
        for symbol in symbols:
            for date in dates:
                for config_path in config_paths:
                    config_name = Path(config_path).stem
                    pbar.set_description(f"{symbol} {date} {config_name}")
                    
                    try:
                        result = run_single_backtest(
                            symbol, date, config_path, data_dir, verbose=False
                        )
                        
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        import traceback
                        print(f"\n[ERROR] {symbol} {date} {config_name}:")
                        print(f"  {type(e).__name__}: {str(e)}")
                        if verbose:
                            traceback.print_exc()
                    
                    pbar.update(1)
    
    if len(results) == 0:
        print("\n[ERROR] No successful backtests")
        return None
    
    print(f"\n[SUCCESS] Completed {len(results)}/{total_runs} backtests successfully")
    
    # Create summary DataFrame
    summary_data = []
    for r in results:
        metrics = r['metrics']
        summary_data.append({
            'symbol': r['symbol'],
            'date': r['date'],
            'config_name': r['config_name'],
            'total_pnl': metrics.total_pnl,
            'total_return': metrics.total_return,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'max_drawdown': metrics.max_drawdown,
            'max_drawdown_pct': metrics.max_drawdown_pct,
            'total_fills': metrics.total_fills,
            'total_volume': metrics.total_volume,
            'fill_rate': metrics.fill_rate,
            'avg_fill_edge_bps': metrics.avg_fill_edge_bps,
            'fill_edge_std_bps': metrics.fill_edge_std_bps,
            'adverse_selection_1s_bps': metrics.adverse_selection_1s_bps,
            'adverse_selection_5s_bps': metrics.adverse_selection_5s_bps,
            'adverse_selection_10s_bps': metrics.adverse_selection_10s_bps,
            'avg_inventory': metrics.avg_inventory,
            'inventory_std': metrics.inventory_std,
            'max_inventory': metrics.max_inventory,
            'min_inventory': metrics.min_inventory,
            'time_at_limit_pct': metrics.time_at_limit_pct,
            'ofi_signal_corr': metrics.ofi_signal_corr,
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary_path = output_path / f"backtest_summary_{timestamp}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[SAVED] Summary: {summary_path}")
    
    # Save detailed results (Parquet format)
    for r in results:
        detail_filename = f"{r['symbol']}_{r['date']}_{r['config_name']}.parquet"
        detail_path = output_path / "detailed" / detail_filename
        detail_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create detailed DataFrame
        result_obj = r['backtest_result']
        detail_df = pd.DataFrame({
            'timestamp': result_obj.timestamps,
            'inventory': result_obj.inventory,
            'cash': result_obj.cash,
            'pnl': result_obj.pnl,
            'bid': result_obj.bid,
            'ask': result_obj.ask,
            'our_bid': result_obj.our_bid,
            'our_ask': result_obj.our_ask,
        })
        detail_df.to_parquet(detail_path, index=False)
    
    print(f"[SAVED] {len(results)} detailed results to: {output_path / 'detailed'}")
    
    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Run OFI market making backtests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single symbol-date-strategy
  python scripts/run_backtest.py --symbol AAPL --date 2017-01-03 --config configs/ofi_full.yaml
  
  # Run multiple symbols for one date
  python scripts/run_backtest.py --symbols AAPL AMD AMZN --date 2017-01-03 --config configs/ofi_full.yaml
  
  # Run all 4 strategies for validation week (Week 1)
  python scripts/run_backtest.py --symbols AAPL AMD AMZN JPM MSFT NVDA SPY TSLA \\
      --dates 2017-01-03 2017-01-04 2017-01-05 2017-01-06 \\
      --configs configs/symmetric_baseline.yaml configs/microprice_only.yaml \\
               configs/ofi_full.yaml configs/ofi_ablation.yaml
        """
    )
    
    parser.add_argument('--symbol', type=str, help='Single symbol to backtest')
    parser.add_argument('--symbols', type=str, nargs='+', help='Multiple symbols to backtest')
    parser.add_argument('--date', type=str, help='Single date (YYYY-MM-DD)')
    parser.add_argument('--dates', type=str, nargs='+', help='Multiple dates')
    parser.add_argument('--config', type=str, help='Single config file path')
    parser.add_argument('--configs', type=str, nargs='+', help='Multiple config files')
    parser.add_argument('--data-dir', type=str, default='data', help='NBBO data directory')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Parse symbols
    if args.symbols:
        symbols = args.symbols
    elif args.symbol:
        symbols = [args.symbol]
    else:
        # Default: all 8 symbols
        symbols = ['AAPL', 'AMD', 'AMZN', 'JPM', 'MSFT', 'NVDA', 'SPY', 'TSLA']
    
    # Parse dates
    if args.dates:
        dates = args.dates
    elif args.date:
        dates = [args.date]
    else:
        # Default: Week 1 (validation)
        dates = ['2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06']
    
    # Parse configs
    if args.configs:
        config_paths = args.configs
    elif args.config:
        config_paths = [args.config]
    else:
        # Default: all 4 strategies
        config_paths = [
            'configs/symmetric_baseline.yaml',
            'configs/microprice_only.yaml',
            'configs/ofi_full.yaml',
            'configs/ofi_ablation.yaml',
        ]
    
    # Run backtests
    summary_df = run_batch_backtests(
        symbols=symbols,
        dates=dates,
        config_paths=config_paths,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        verbose=args.verbose
    )
    
    if summary_df is not None:
        # Print summary statistics
        print(f"\n{'='*80}")
        print("Summary Statistics by Strategy")
        print(f"{'='*80}")
        
        strategy_summary = summary_df.groupby('config_name').agg({
            'total_pnl': ['mean', 'std', 'min', 'max'],
            'sharpe_ratio': ['mean', 'std'],
            'total_fills': ['mean', 'sum'],
            'avg_fill_edge_bps': 'mean',
        }).round(4)
        
        print(strategy_summary)
        print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

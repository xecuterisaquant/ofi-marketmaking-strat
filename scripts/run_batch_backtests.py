#!/usr/bin/env python3
"""
Batch Backtest Execution Script

Runs all strategy configurations across multiple symbols and dates for comparative analysis.

Usage:
    python scripts/run_batch_backtests.py --symbols AAPL AMD --dates 2017-01-03 2017-01-04
    python scripts/run_batch_backtests.py --all-symbols --all-dates
    python scripts/run_batch_backtests.py --quick-test  # Run subset for validation
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import subprocess

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Create logs directory if it doesn't exist
Path('logs').mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/batch_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Available symbols (from OFI replication study)
ALL_SYMBOLS = ['AAPL', 'AMD', 'AMZN', 'FB', 'GOOGL', 'MSFT', 'NVDA']

# Available dates (January 2017 trading days)
ALL_DATES = [
    '2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06',
    '2017-01-09', '2017-01-10', '2017-01-11', '2017-01-12', '2017-01-13',
    '2017-01-17', '2017-01-18', '2017-01-19', '2017-01-20',
    '2017-01-23', '2017-01-24', '2017-01-25', '2017-01-26', '2017-01-27',
    '2017-01-30', '2017-01-31'
]

# Strategy configurations
ALL_CONFIGS = [
    'configs/symmetric_baseline.yaml',
    'configs/microprice_only.yaml',
    'configs/ofi_full.yaml',
    'configs/ofi_ablation.yaml'
]


def run_single_backtest(args):
    """
    Run a single backtest by calling run_backtest.py as subprocess.
    
    Args:
        args: Tuple of (symbol, date, config_path)
        
    Returns:
        dict: Results summary with metadata
    """
    symbol, date, config_path = args
    config_name = Path(config_path).stem
    
    try:
        start_time = time.time()
        logger.info(f"Starting: {symbol} {date} {config_name}")
        
        # Get list of existing summary files before running
        results_dir = Path('results')
        existing_summaries = set(results_dir.glob('backtest_summary_*.csv'))
        
        # Call run_backtest.py as subprocess
        cmd = [
            sys.executable,
            'scripts/run_backtest.py',
            '--symbol', symbol,
            '--date', date,
            '--config', config_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed = time.time() - start_time
        
        # Find the new summary file that was created
        current_summaries = set(results_dir.glob('backtest_summary_*.csv'))
        new_summaries = current_summaries - existing_summaries
        
        if new_summaries:
            summary_file = list(new_summaries)[0]
            summary_df = pd.read_csv(summary_file)
            summary = summary_df.iloc[0].to_dict()
        else:
            logger.warning(f"No new summary file found for {symbol} {date} {config_name}")
            summary = {}
        
        # Add metadata
        result_dict = {
            'symbol': symbol,
            'date': date,
            'strategy': config_name,
            'success': True,
            'elapsed_seconds': elapsed,
            **summary
        }
        
        logger.info(
            f"Completed: {symbol} {date} {config_name} "
            f"(PnL: ${summary.get('final_pnl', 0):.2f}, "
            f"Fills: {summary.get('total_fills', 0)}, "
            f"{elapsed:.1f}s)"
        )
        
        return result_dict
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {symbol} {date} {config_name} - {e.stderr}")
        return {
            'symbol': symbol,
            'date': date,
            'strategy': config_name,
            'success': False,
            'error': e.stderr[:200],  # First 200 chars of error
            'elapsed_seconds': time.time() - start_time
        }
    except Exception as e:
        logger.error(f"Failed: {symbol} {date} {config_name} - {str(e)}")
        return {
            'symbol': symbol,
            'date': date,
            'strategy': config_name,
            'success': False,
            'error': str(e),
            'elapsed_seconds': time.time() - start_time
        }


def main():
    parser = argparse.ArgumentParser(
        description='Run batch backtests across strategies, symbols, and dates'
    )
    
    # Symbol selection
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to test (e.g., AAPL AMD)'
    )
    parser.add_argument(
        '--all-symbols',
        action='store_true',
        help='Run all 7 symbols'
    )
    
    # Date selection
    parser.add_argument(
        '--dates',
        nargs='+',
        help='Specific dates to test (e.g., 2017-01-03 2017-01-04)'
    )
    parser.add_argument(
        '--all-dates',
        action='store_true',
        help='Run all 20 dates'
    )
    
    # Strategy selection
    parser.add_argument(
        '--strategies',
        nargs='+',
        choices=['symmetric_baseline', 'microprice_only', 'ofi_full', 'ofi_ablation'],
        help='Specific strategies to run'
    )
    parser.add_argument(
        '--all-strategies',
        action='store_true',
        help='Run all 4 strategies'
    )
    
    # Quick test mode
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Quick validation: 2 symbols × 2 dates × 2 strategies = 8 runs'
    )
    
    # Execution options
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum parallel workers (default: 4)'
    )
    parser.add_argument(
        '--sequential',
        action='store_true',
        help='Run sequentially instead of parallel (for debugging)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/batch'),
        help='Output directory for summary results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Logs directory already created at module level
    
    # Determine symbols to run
    if args.quick_test:
        symbols = ['AAPL', 'AMD']
        dates = ['2017-01-03', '2017-01-04']
        config_paths = [
            'configs/symmetric_baseline.yaml',
            'configs/ofi_full.yaml'
        ]
    else:
        # Symbols
        if args.all_symbols:
            symbols = ALL_SYMBOLS
        elif args.symbols:
            symbols = args.symbols
        else:
            symbols = ['AAPL']  # Default
        
        # Dates
        if args.all_dates:
            dates = ALL_DATES
        elif args.dates:
            dates = args.dates
        else:
            dates = ['2017-01-03']  # Default
        
        # Strategies
        if args.all_strategies:
            config_paths = ALL_CONFIGS
        elif args.strategies:
            config_paths = [f'configs/{s}.yaml' for s in args.strategies]
        else:
            config_paths = ALL_CONFIGS  # Default: run all
    
    # Generate all combinations
    tasks = [
        (symbol, date, config)
        for symbol in symbols
        for date in dates
        for config in config_paths
    ]
    
    total_tasks = len(tasks)
    logger.info(f"="*80)
    logger.info(f"BATCH BACKTEST EXECUTION")
    logger.info(f"="*80)
    logger.info(f"Symbols: {len(symbols)} - {symbols}")
    logger.info(f"Dates: {len(dates)} - {dates[0]} to {dates[-1]}")
    logger.info(f"Strategies: {len(config_paths)}")
    logger.info(f"Total backtests: {total_tasks}")
    logger.info(f"Parallel workers: {args.max_workers if not args.sequential else 1}")
    logger.info(f"="*80)
    
    # Run backtests
    results = []
    start_time = time.time()
    
    if args.sequential:
        # Sequential execution (easier debugging)
        for i, task in enumerate(tasks, 1):
            logger.info(f"Progress: {i}/{total_tasks}")
            result = run_single_backtest(task)
            results.append(result)
    else:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(run_single_backtest, task): task for task in tasks}
            
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results.append(result)
                
                # Progress update
                successes = sum(1 for r in results if r['success'])
                logger.info(f"Progress: {i}/{total_tasks} ({successes} successful)")
    
    # Total elapsed time
    total_elapsed = time.time() - start_time
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = args.output_dir / f'batch_summary_{timestamp}.csv'
    results_df.to_csv(summary_file, index=False)
    
    # Print summary statistics
    logger.info(f"\n{'='*80}")
    logger.info(f"BATCH EXECUTION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"Total backtests: {total_tasks}")
    logger.info(f"Successful: {results_df['success'].sum()}")
    logger.info(f"Failed: {(~results_df['success']).sum()}")
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")
    logger.info(f"Average time per backtest: {total_elapsed/total_tasks:.1f}s")
    logger.info(f"\nResults saved to: {summary_file}")
    
    if results_df['success'].sum() > 0:
        successful_df = results_df[results_df['success']].copy()
        
        # Check if we have the necessary columns (handle both old and new naming)
        pnl_col = 'total_pnl' if 'total_pnl' in successful_df.columns else 'final_pnl'
        
        if pnl_col in successful_df.columns:
            logger.info(f"\n{'='*80}")
            logger.info(f"PERFORMANCE SUMMARY (Successful runs)")
            logger.info(f"{'='*80}")
            
            # Overall statistics
            logger.info(f"\nOverall Metrics:")
            logger.info(f"  Mean PnL: ${successful_df[pnl_col].mean():.2f}")
            logger.info(f"  Median PnL: ${successful_df[pnl_col].median():.2f}")
            logger.info(f"  Mean Sharpe: {successful_df['sharpe_ratio'].mean():.2f}")
            logger.info(f"  Mean fills: {successful_df['total_fills'].mean():.0f}")
            
            # By strategy
            logger.info(f"\nBy Strategy:")
            strategy_stats = successful_df.groupby('strategy').agg({
                pnl_col: ['mean', 'median', 'std'],
                'sharpe_ratio': ['mean', 'median'],
                'total_fills': ['mean', 'median'],
                'symbol': 'count'
            }).round(2)
            print(strategy_stats)
            
            # Best/worst runs
            logger.info(f"\nTop 5 Runs by PnL:")
            top5 = successful_df.nlargest(5, pnl_col)[
                ['symbol', 'date', 'strategy', pnl_col, 'sharpe_ratio', 'total_fills']
            ]
            print(top5.to_string(index=False))
            
            logger.info(f"\nBottom 5 Runs by PnL:")
            bottom5 = successful_df.nsmallest(5, pnl_col)[
                ['symbol', 'date', 'strategy', pnl_col, 'sharpe_ratio', 'total_fills']
            ]
            print(bottom5.to_string(index=False))
        else:
            logger.warning("Summary metrics not available in results. Check summary file loading.")
    
    # List any failures
    if (~results_df['success']).sum() > 0:
        logger.warning(f"\n{'='*80}")
        logger.warning(f"FAILED RUNS:")
        logger.warning(f"{'='*80}")
        failed_df = results_df[~results_df['success']]
        for _, row in failed_df.iterrows():
            logger.warning(
                f"  {row['symbol']} {row['date']} {row['strategy']}: {row.get('error', 'Unknown error')}"
            )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"All results saved to: {summary_file}")
    logger.info(f"Detailed outputs in: results/detailed/")
    logger.info(f"{'='*80}\n")
    
    return 0 if results_df['success'].all() else 1


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""
Signal Validation Analysis

Tests OFI predictive power BEFORE integration into strategy.
Validates that OFI signal has genuine forecasting ability independent of backtest performance.

Usage:
    python scripts/validate_signal.py --symbol AAPL --date 2017-01-03
    python scripts/validate_signal.py --all  # Run across all data
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ofi_utils import compute_ofi_depth_mid, read_rda, resolve_columns, parse_trading_day_from_filename, build_tob_series_1s


def validate_ofi_signal(symbol: str, date: str, horizons: list = [5, 10, 30, 60]):
    """
    Validate OFI signal predictive power at multiple horizons.
    
    Parameters
    ----------
    symbol : str
        Stock symbol (e.g., 'AAPL')
    date : str
        Trading date in YYYY-MM-DD format
    horizons : list
        Forecast horizons in seconds
        
    Returns
    -------
    dict
        Validation metrics for each horizon
    """
    # Load NBBO data
    file_path = Path("data") / "NBBO" / f"{date}.rda"
    
    if not file_path.exists():
        print(f"❌ Data file not found: {file_path}")
        return None
    
    # Load and process data
    df_raw = read_rda(str(file_path))
    cmap = resolve_columns(df_raw)
    trading_day = parse_trading_day_from_filename(file_path.name)
    
    # Filter to symbol
    df_symbol = df_raw[df_raw['sym_root'] == symbol].copy()
    
    if len(df_symbol) == 0:
        print(f"❌ No data for {symbol} in {date}")
        return None
    
    # Build 1-second NBBO series
    nbbo_df = build_tob_series_1s(df_symbol, cmap, trading_day, freq='1s')
    
    # Compute OFI from the NBBO series
    # Rename columns to match what compute_ofi_depth_mid expects
    nbbo_for_ofi = nbbo_df.rename(columns={
        'best_bid': 'bid',
        'best_ask': 'ask',
        'best_bidsiz': 'bid_sz',
        'best_asksiz': 'ask_sz'
    })
    
    ofi_df = compute_ofi_depth_mid(nbbo_for_ofi)
    
    # Normalize OFI
    from src.ofi_utils import normalize_ofi
    ofi_df = normalize_ofi(ofi_df, window_secs=60, min_periods=10)
    
    # Merge OFI with NBBO (they should already be aligned since computed from same data)
    ofi_df['timestamp'] = pd.to_datetime(ofi_df.index)
    nbbo_df['timestamp'] = pd.to_datetime(nbbo_df.index)
    
    merged = pd.merge_asof(
        nbbo_df.sort_values('timestamp'),
        ofi_df[['timestamp', 'normalized_OFI']].sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )
    
    # Rename normalized_OFI to ofi_normalized for consistency
    merged = merged.rename(columns={'normalized_OFI': 'ofi_normalized'})
    
    # Compute mid-price (use existing mid from ofi_df if available, otherwise compute)
    if 'mid' not in merged.columns:
        # Check which column names we have
        if 'best_bid' in merged.columns:
            merged['mid'] = (merged['best_bid'] + merged['best_ask']) / 2
        elif 'bid' in merged.columns:
            merged['mid'] = (merged['bid'] + merged['ask']) / 2
        else:
            print(f"⚠️ Available columns: {merged.columns.tolist()}")
            return None
    
    # Drop NaN OFI values
    merged = merged.dropna(subset=['ofi_normalized'])
    
    if len(merged) < 100:
        print(f"⚠️ Insufficient data after merge: {len(merged)} rows")
        return None
    
    results = {}
    
    for horizon in horizons:
        # Compute forward returns
        merged[f'ret_{horizon}s'] = (
            merged['mid'].shift(-horizon) / merged['mid'] - 1
        ) * 10000  # in bps
        
        # Drop NaN forward returns
        valid = merged.dropna(subset=[f'ret_{horizon}s', 'ofi_normalized'])
        
        if len(valid) < 50:
            continue
        
        ofi_signal = valid['ofi_normalized'].values
        forward_return = valid[f'ret_{horizon}s'].values
        
        # Pearson correlation
        corr, pval = pearsonr(ofi_signal, forward_return)
        
        # Directional accuracy (sign agreement)
        correct = (np.sign(ofi_signal) == np.sign(forward_return)).mean()
        
        # Information Coefficient (Spearman rank correlation)
        ic, _ = spearmanr(ofi_signal, forward_return)
        
        # Mean Absolute Error (using beta = 0.036 from replication)
        beta = 0.036
        predicted_return = ofi_signal * beta
        mae = mean_absolute_error(forward_return, predicted_return)
        
        # Hit rate (% of non-zero predictions that are correct)
        non_zero_pred = ofi_signal != 0
        if non_zero_pred.sum() > 0:
            hit_rate = (
                np.sign(ofi_signal[non_zero_pred]) == np.sign(forward_return[non_zero_pred])
            ).mean()
        else:
            hit_rate = np.nan
        
        results[horizon] = {
            'correlation': corr,
            'p_value': pval,
            'directional_accuracy': correct,
            'information_coefficient': ic,
            'mae_bps': mae,
            'hit_rate': hit_rate,
            'n_obs': len(valid)
        }
    
    return results


def aggregate_results(all_results: list) -> pd.DataFrame:
    """
    Aggregate validation results across multiple symbol-dates.
    
    Parameters
    ----------
    all_results : list
        List of dicts from validate_ofi_signal
        
    Returns
    -------
    pd.DataFrame
        Aggregated metrics by horizon
    """
    records = []
    for result_dict in all_results:
        if result_dict is None:
            continue
        for horizon, metrics in result_dict.items():
            records.append({
                'horizon': horizon,
                **metrics
            })
    
    df = pd.DataFrame(records)
    
    # Compute mean and std for each horizon
    agg = df.groupby('horizon').agg({
        'correlation': ['mean', 'std'],
        'p_value': 'mean',
        'directional_accuracy': ['mean', 'std'],
        'information_coefficient': ['mean', 'std'],
        'mae_bps': ['mean', 'std'],
        'hit_rate': ['mean', 'std'],
        'n_obs': 'mean'
    })
    
    return agg


def main():
    parser = argparse.ArgumentParser(description='Validate OFI signal predictive power')
    parser.add_argument('--symbol', type=str, help='Stock symbol (e.g., AAPL)')
    parser.add_argument('--date', type=str, help='Trading date (YYYY-MM-DD)')
    parser.add_argument('--all', action='store_true', help='Run across all symbols and dates')
    parser.add_argument('--horizons', type=int, nargs='+', default=[5, 10, 30, 60],
                       help='Forecast horizons in seconds')
    
    args = parser.parse_args()
    
    if args.all:
        # Run across all available data
        symbols = ['AAPL', 'AMD', 'AMZN', 'MSFT', 'NVDA']
        dates = [
            '2017-01-03', '2017-01-04', '2017-01-05', '2017-01-06',
            '2017-01-09', '2017-01-10', '2017-01-11', '2017-01-12', '2017-01-13',
            '2017-01-17', '2017-01-18', '2017-01-19', '2017-01-20',
            '2017-01-23', '2017-01-24', '2017-01-25', '2017-01-26', '2017-01-27',
            '2017-01-30', '2017-01-31'
        ]
        
        all_results = []
        
        print(f"Running validation across {len(symbols)} symbols × {len(dates)} dates = {len(symbols)*len(dates)} combinations\n")
        
        for symbol in symbols:
            symbol_results = []
            for date in dates:
                print(f"Processing {symbol} {date}...")
                result = validate_ofi_signal(symbol, date, args.horizons)
                if result:
                    all_results.append(result)
                    symbol_results.append(result)
            
            # Print symbol-level summary
            if symbol_results:
                agg = aggregate_results(symbol_results)
                print(f"\n{symbol} Summary:")
                print(agg)
                print("\n" + "="*80 + "\n")
        
        # Print overall summary
        print("\n" + "="*80)
        print("OVERALL SUMMARY ACROSS ALL SYMBOLS AND DATES")
        print("="*80 + "\n")
        
        agg = aggregate_results(all_results)
        print(agg)
        
        # Save results
        output_file = Path('results') / 'signal_validation_summary.csv'
        agg.to_csv(output_file)
        print(f"\n✅ Results saved to: {output_file}")
        
        # Format for report
        print("\n" + "="*80)
        print("FORMATTED FOR REPORT")
        print("="*80 + "\n")
        
        print("| Horizon | Correlation | Dir. Accuracy | Info Coef | MAE (bps) | Hit Rate | p-value |")
        print("|---------|-------------|---------------|-----------|-----------|----------|---------|")
        
        for horizon in sorted(args.horizons):
            if horizon in agg.index:
                row = agg.loc[horizon]
                print(f"| {horizon}s | {row[('correlation', 'mean')]:.3f} ± {row[('correlation', 'std')]:.3f} | "
                      f"{row[('directional_accuracy', 'mean')]:.1%} | "
                      f"{row[('information_coefficient', 'mean')]:.3f} | "
                      f"{row[('mae_bps', 'mean')]:.2f} | "
                      f"{row[('hit_rate', 'mean')]:.1%} | "
                      f"{row[('p_value', 'mean')]:.4f} |")
        
    else:
        # Single symbol-date
        if not args.symbol or not args.date:
            print("❌ Must specify --symbol and --date, or use --all")
            return
        
        print(f"Validating OFI signal: {args.symbol} {args.date}\n")
        
        results = validate_ofi_signal(args.symbol, args.date, args.horizons)
        
        if results:
            print("\nValidation Results:")
            print("="*80)
            
            df = pd.DataFrame(results).T
            print(df)
            
            print("\n✅ Signal shows predictive power!" if df['p_value'].mean() < 0.05 else "\n⚠️ Signal may lack statistical significance")


if __name__ == '__main__':
    main()

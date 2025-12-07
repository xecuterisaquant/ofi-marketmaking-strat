"""
Comprehensive data analysis to understand NBBO and Trade characteristics.

This script analyzes:
1. Price levels and spreads across symbols
2. Quote update frequency
3. Book depth characteristics
4. Volatility estimation
5. Effective spreads and trade characteristics

Purpose: Calibrate strategy parameters properly for different symbols/markets.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from maker.backtest import load_nbbo_day
from src.ofi_utils import read_rda, resolve_columns, parse_trading_day_from_filename, build_tob_series_1s


def load_nbbo_for_symbol(file_path: str, symbol: str):
    """Load NBBO data for a specific symbol from .rda file."""
    # Read raw data
    df = read_rda(file_path)
    
    # Filter to specific symbol BEFORE resampling
    symbol_col = 'sym_root' if 'sym_root' in df.columns else 'symbol'
    df = df[df[symbol_col] == symbol].copy()
    
    if len(df) == 0:
        return None, None
    
    # Resolve column names
    cmap = resolve_columns(df)
    
    # Parse trading day
    trading_day = parse_trading_day_from_filename(file_path)
    
    # Build 1-second time series for RTH
    nbbo_data = build_tob_series_1s(df, cmap, trading_day, freq='1s')
    
    return nbbo_data, trading_day


def analyze_nbbo_file(file_path: str, symbol: str = None):
    """
    Analyze NBBO data from a single file.
    
    Parameters
    ----------
    file_path : str
        Path to .rda NBBO file
    symbol : str, optional
        Specific symbol to analyze (if None, analyze all)
    
    Returns
    -------
    dict
        Analysis results
    """
    print(f"\nAnalyzing: {file_path}")
    print("="*80)
    
    # Load data for specific symbol
    if symbol:
        nbbo_df, trading_day = load_nbbo_for_symbol(file_path, symbol)
    else:
        # Load all symbols together (original behavior)
        nbbo_df, trading_day = load_nbbo_day(file_path)
    
    if nbbo_df is None or len(nbbo_df) == 0:
        print("No data loaded")
        return None
    
    # Filter to specific symbol if requested
    if symbol:
        if 'symbol' in nbbo_df.columns:
            nbbo_df = nbbo_df[nbbo_df['symbol'] == symbol].copy()
        elif 'sym_root' in nbbo_df.columns:
            nbbo_df = nbbo_df[nbbo_df['sym_root'] == symbol].copy()
    
    if len(nbbo_df) == 0:
        print(f"No data for symbol {symbol}")
        return None
    
    # Basic statistics
    print(f"\n1. DATA OVERVIEW")
    print(f"-" * 80)
    print(f"Trading Day: {trading_day}")
    print(f"Total Rows: {len(nbbo_df):,}")
    print(f"Time Range: {nbbo_df.index[0]} to {nbbo_df.index[-1]}")
    print(f"Duration: {(nbbo_df.index[-1] - nbbo_df.index[0]).total_seconds() / 3600:.2f} hours")
    
    # Check for required columns
    required_cols = ['bid', 'ask', 'bid_sz', 'ask_sz']
    for col in required_cols:
        if col not in nbbo_df.columns:
            print(f"WARNING: Missing column {col}")
            return None
    
    # Price analysis
    print(f"\n2. PRICE LEVELS")
    print(f"-" * 80)
    mid = (nbbo_df['bid'] + nbbo_df['ask']) / 2
    print(f"Mid Price - Mean: ${mid.mean():.2f}, Std: ${mid.std():.2f}")
    print(f"Mid Price - Min: ${mid.min():.2f}, Max: ${mid.max():.2f}")
    print(f"Price Range: ${mid.max() - mid.min():.2f} ({(mid.max() - mid.min()) / mid.mean() * 100:.2f}%)")
    
    # Spread analysis
    print(f"\n3. BID-ASK SPREADS")
    print(f"-" * 80)
    spread_dollars = nbbo_df['ask'] - nbbo_df['bid']
    spread_bps = (spread_dollars / mid) * 10000
    
    print(f"Spread (dollars):")
    print(f"  Mean: ${spread_dollars.mean():.4f}")
    print(f"  Median: ${spread_dollars.median():.4f}")
    print(f"  Std: ${spread_dollars.std():.4f}")
    print(f"  Min: ${spread_dollars.min():.4f}, Max: ${spread_dollars.max():.4f}")
    
    print(f"\nSpread (basis points):")
    print(f"  Mean: {spread_bps.mean():.2f} bps")
    print(f"  Median: {spread_bps.median():.2f} bps")
    print(f"  Std: {spread_bps.std():.2f} bps")
    print(f"  25th pct: {spread_bps.quantile(0.25):.2f} bps")
    print(f"  75th pct: {spread_bps.quantile(0.75):.2f} bps")
    print(f"  99th pct: {spread_bps.quantile(0.99):.2f} bps")
    
    # Tick size analysis
    tick_size = 0.01  # Standard US equity tick
    print(f"\n4. TICK ANALYSIS (tick size = ${tick_size})")
    print(f"-" * 80)
    spread_in_ticks = spread_dollars / tick_size
    print(f"Spread in Ticks:")
    print(f"  Mean: {spread_in_ticks.mean():.2f} ticks")
    print(f"  Median: {spread_in_ticks.median():.2f} ticks")
    print(f"  Mode: {spread_in_ticks.mode().values[0] if len(spread_in_ticks.mode()) > 0 else 'N/A'} ticks")
    print(f"  % at 1 tick: {(spread_in_ticks == 1).sum() / len(spread_in_ticks) * 100:.1f}%")
    print(f"  % at 2 ticks: {(spread_in_ticks == 2).sum() / len(spread_in_ticks) * 100:.1f}%")
    print(f"  % at 3+ ticks: {(spread_in_ticks >= 3).sum() / len(spread_in_ticks) * 100:.1f}%")
    
    # Depth analysis
    print(f"\n5. BOOK DEPTH")
    print(f"-" * 80)
    print(f"Bid Size (shares):")
    print(f"  Mean: {nbbo_df['bid_sz'].mean():.0f}")
    print(f"  Median: {nbbo_df['bid_sz'].median():.0f}")
    print(f"  Std: {nbbo_df['bid_sz'].std():.0f}")
    
    print(f"\nAsk Size (shares):")
    print(f"  Mean: {nbbo_df['ask_sz'].mean():.0f}")
    print(f"  Median: {nbbo_df['ask_sz'].median():.0f}")
    print(f"  Std: {nbbo_df['ask_sz'].std():.0f}")
    
    # Depth imbalance
    total_depth = nbbo_df['bid_sz'] + nbbo_df['ask_sz']
    imbalance = (nbbo_df['bid_sz'] - nbbo_df['ask_sz']) / total_depth
    print(f"\nDepth Imbalance [(bid_sz - ask_sz) / (bid_sz + ask_sz)]:")
    print(f"  Mean: {imbalance.mean():.4f}")
    print(f"  Std: {imbalance.std():.4f}")
    print(f"  Min: {imbalance.min():.4f}, Max: {imbalance.max():.4f}")
    
    # Volatility analysis
    print(f"\n6. VOLATILITY")
    print(f"-" * 80)
    
    # 1-second returns
    returns_1s = mid.pct_change()
    vol_1s_annualized = returns_1s.std() * np.sqrt(252 * 6.5 * 3600)  # Annualize
    
    print(f"1-Second Returns:")
    print(f"  Mean: {returns_1s.mean() * 10000:.4f} bps")
    print(f"  Std: {returns_1s.std() * 10000:.4f} bps")
    print(f"  Annualized Vol: {vol_1s_annualized * 100:.2f}%")
    
    # 5-second returns
    mid_5s = mid.resample('5S').last()
    returns_5s = mid_5s.pct_change()
    vol_5s_annualized = returns_5s.std() * np.sqrt(252 * 6.5 * 3600 / 5)
    
    print(f"\n5-Second Returns:")
    print(f"  Mean: {returns_5s.mean() * 10000:.4f} bps")
    print(f"  Std: {returns_5s.std() * 10000:.4f} bps")
    print(f"  Annualized Vol: {vol_5s_annualized * 100:.2f}%")
    
    # 1-minute returns
    mid_1min = mid.resample('1min').last()
    returns_1min = mid_1min.pct_change()
    vol_1min_annualized = returns_1min.std() * np.sqrt(252 * 6.5 * 60)
    
    print(f"\n1-Minute Returns:")
    print(f"  Mean: {returns_1min.mean() * 10000:.4f} bps")
    print(f"  Std: {returns_1min.std() * 10000:.4f} bps")
    print(f"  Annualized Vol: {vol_1min_annualized * 100:.2f}%")
    
    # Quote update frequency
    print(f"\n7. QUOTE UPDATE FREQUENCY")
    print(f"-" * 80)
    quote_changes = (
        (nbbo_df['bid'] != nbbo_df['bid'].shift()) |
        (nbbo_df['ask'] != nbbo_df['ask'].shift())
    ).sum()
    print(f"Total Quote Updates: {quote_changes:,}")
    print(f"Update Frequency: {quote_changes / len(nbbo_df) * 100:.1f}% of seconds")
    print(f"Avg Time Between Updates: {len(nbbo_df) / quote_changes:.2f} seconds")
    
    # Return statistics for calibration
    results = {
        'symbol': symbol,
        'date': trading_day,
        'n_observations': len(nbbo_df),
        'mean_price': mid.mean(),
        'price_std': mid.std(),
        'mean_spread_bps': spread_bps.mean(),
        'median_spread_bps': spread_bps.median(),
        'mean_spread_ticks': spread_in_ticks.mean(),
        'pct_1tick_spread': (spread_in_ticks == 1).sum() / len(spread_in_ticks),
        'mean_bid_size': nbbo_df['bid_sz'].mean(),
        'mean_ask_size': nbbo_df['ask_sz'].mean(),
        'annualized_vol_1s': vol_1s_annualized,
        'annualized_vol_5s': vol_5s_annualized,
        'annualized_vol_1min': vol_1min_annualized,
        'quote_update_pct': quote_changes / len(nbbo_df),
    }
    
    return results


def compare_symbols(data_dir: str = "data", date: str = "2017-01-03"):
    """
    Compare characteristics across multiple symbols on the same date.
    
    Parameters
    ----------
    data_dir : str
        Data directory
    date : str
        Trading date
    """
    # Exclude JPM - data corruption detected (76% intraday move, 8537% vol)
    symbols = ['AAPL', 'AMD', 'AMZN', 'MSFT', 'NVDA', 'SPY', 'TSLA']
    
    file_path = Path(data_dir) / "NBBO" / f"{date}.rda"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"CROSS-SYMBOL COMPARISON - {date}")
    print(f"{'='*80}")
    
    all_results = []
    
    for symbol in symbols:
        print(f"\n{symbol}")
        print("-" * 80)
        results = analyze_nbbo_file(str(file_path), symbol=symbol)
        if results:
            all_results.append(results)
    
    if len(all_results) == 0:
        print("No results to compare")
        return
    
    # Create comparison table
    comparison_df = pd.DataFrame(all_results)
    
    print(f"\n{'='*80}")
    print(f"SUMMARY COMPARISON")
    print(f"{'='*80}\n")
    
    # Format for display
    display_df = comparison_df[['symbol', 'mean_price', 'median_spread_bps', 
                                  'mean_spread_ticks', 'pct_1tick_spread',
                                  'annualized_vol_1min']].copy()
    
    display_df['mean_price'] = display_df['mean_price'].apply(lambda x: f"${x:.2f}")
    display_df['median_spread_bps'] = display_df['median_spread_bps'].apply(lambda x: f"{x:.2f}")
    display_df['mean_spread_ticks'] = display_df['mean_spread_ticks'].apply(lambda x: f"{x:.2f}")
    display_df['pct_1tick_spread'] = display_df['pct_1tick_spread'].apply(lambda x: f"{x*100:.1f}%")
    display_df['annualized_vol_1min'] = display_df['annualized_vol_1min'].apply(lambda x: f"{x*100:.1f}%")
    
    display_df.columns = ['Symbol', 'Avg Price', 'Median Spread (bps)', 
                          'Avg Spread (ticks)', '% 1-Tick Spread', 'Ann. Vol (1min)']
    
    print(display_df.to_string(index=False))
    
    print(f"\n{'='*80}")
    print("KEY INSIGHTS FOR STRATEGY CALIBRATION:")
    print(f"{'='*80}")
    
    # Derive calibration insights
    avg_spread_bps = comparison_df['median_spread_bps'].mean()
    avg_vol = comparison_df['annualized_vol_1min'].mean()
    avg_1tick_pct = comparison_df['pct_1tick_spread'].mean()
    
    print(f"\n1. Spread Characteristics:")
    print(f"   - Average median spread: {avg_spread_bps:.2f} bps")
    print(f"   - Range: {comparison_df['median_spread_bps'].min():.2f} - {comparison_df['median_spread_bps'].max():.2f} bps")
    print(f"   - Average 1-tick spread %: {avg_1tick_pct*100:.1f}%")
    print(f"   => CALIBRATION: Min spread should be ~{avg_spread_bps:.1f} bps")
    
    print(f"\n2. Volatility:")
    print(f"   - Average annualized vol: {avg_vol*100:.1f}%")
    print(f"   - Range: {comparison_df['annualized_vol_1min'].min()*100:.1f}% - {comparison_df['annualized_vol_1min'].max()*100:.1f}%")
    print(f"   => CALIBRATION: Expected σ ~ {avg_vol:.4f} for quote width calculation")
    
    print(f"\n3. Recommended A-S Parameters (PRICE-NORMALIZED):")
    # A-S spread formula: δ = γ*σ²*T + (2/γ)*log(1 + γ/k)
    # δ is a RELATIVE spread (dimensionless), NOT in dollars
    # To get dollar spread: half_spread_dollars = δ * price
    # To get bps: half_spread_bps = δ * 10000
    
    target_half_spread_bps = avg_spread_bps / 2  # Target median half-spread
    
    # For typical parameters: T=300s, σ=avg_vol, k=1.0
    T = 300  # 5 minutes
    sigma = avg_vol
    k = 1.0
    
    # Try different gamma values
    print(f"   Target half-spread: {target_half_spread_bps:.2f} bps")
    print(f"   Using σ={sigma:.4f}, T={T}s, k={k:.1f}")
    print(f"\n   Testing different γ (risk aversion) values:")
    
    best_gamma = None
    best_error = float('inf')
    
    for gamma in [0.0001, 0.0005, 0.001, 0.005, 0.01]:
        vol_term = gamma * (sigma ** 2) * T
        arrival_term = (2 / gamma) * np.log(1 + gamma / k)
        delta_rel = vol_term + arrival_term  # Dimensionless relative spread
        half_spread_bps = delta_rel * 10000
        
        error = abs(half_spread_bps - target_half_spread_bps)
        if error < best_error:
            best_error = error
            best_gamma = gamma
        
        print(f"   γ={gamma:.4f}: δ={delta_rel:.6f} → half-spread = {half_spread_bps:.2f} bps")
        print(f"             (vol term: {vol_term*10000:.2f} bps, arrival term: {arrival_term*10000:.2f} bps)")
    
    print(f"\n   => RECOMMENDATION: Use γ ≈ {best_gamma:.4f} for target spread ~{target_half_spread_bps:.1f} bps")
    print(f"   NOTE: This is for price-NORMALIZED spreads. Formula returns δ (dimensionless).")
    print(f"         Convert to dollars: half_spread_$ = δ * mid_price")
    print(f"         Convert to bps: half_spread_bps = δ * 10000")
    
    return comparison_df


def main():
    """Main analysis routine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze NBBO data characteristics")
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--date', type=str, default='2017-01-03', help='Trading date (YYYY-MM-DD)')
    parser.add_argument('--symbol', type=str, help='Specific symbol to analyze')
    parser.add_argument('--compare', action='store_true', help='Compare all symbols')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_symbols(args.data_dir, args.date)
    else:
        file_path = Path(args.data_dir) / "NBBO" / f"{args.date}.rda"
        analyze_nbbo_file(str(file_path), symbol=args.symbol)


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np
from pathlib import Path

# Load all parquets
parquets = list(Path('results/detailed').glob('*.parquet'))
summaries = []

for pq in parquets:
    parts = pq.stem.split('_')
    symbol = parts[0]
    date = parts[1]
    strategy = '_'.join(parts[2:])
    df = pd.read_parquet(pq)
    
    # Count fills properly
    fills = len(df[df['inventory'].diff() != 0])
    
    summaries.append({
        'symbol': symbol,
        'date': date,
        'strategy': strategy,
        'total_pnl': df['pnl'].iloc[-1],
        'total_fills': fills
    })

df = pd.DataFrame(summaries)

print('=' * 80)
print('STRATEGY SUMMARY')
print('=' * 80)
for strat in ['symmetric_baseline', 'microprice_only', 'ofi_ablation', 'ofi_full']:
    data = df[df['strategy'] == strat]['total_pnl']
    fills = df[df['strategy'] == strat]['total_fills']
    print(f'\n{strat}:')
    print(f'  Mean PnL: ${data.mean():.2f}')
    print(f'  Std Dev: ${data.std():.2f}')
    print(f'  N: {len(data)}')
    print(f'  Avg Fills: {fills.mean():.1f}')

print('\n' + '=' * 80)
print('IMPROVEMENTS vs BASELINE')
print('=' * 80)
baseline = df[df['strategy'] == 'symmetric_baseline'][['symbol', 'date', 'total_pnl']].rename(columns={'total_pnl': 'baseline_pnl'})

for strat in ['microprice_only', 'ofi_ablation', 'ofi_full']:
    strat_df = df[df['strategy'] == strat][['symbol', 'date', 'total_pnl']].rename(columns={'total_pnl': 'strat_pnl'})
    merged = baseline.merge(strat_df, on=['symbol', 'date'])
    
    # Filter out cases where baseline is exactly zero to avoid inf
    merged_valid = merged[merged['baseline_pnl'] != 0].copy()
    merged_valid['improvement'] = ((merged_valid['strat_pnl'] - merged_valid['baseline_pnl']) / merged_valid['baseline_pnl'].abs()) * 100
    
    # Also compute absolute improvement
    merged['abs_improvement'] = merged['strat_pnl'] - merged['baseline_pnl']
    
    print(f'\n{strat}:')
    print(f'  Mean Improvement: {merged_valid["improvement"].mean():.2f}%')
    print(f'  Std: {merged_valid["improvement"].std():.2f}%')
    print(f'  Min: {merged_valid["improvement"].min():.2f}%')
    print(f'  Max: {merged_valid["improvement"].max():.2f}%')
    print(f'  Win Rate: {(merged["abs_improvement"] > 0).sum()}/{len(merged)} ({100*(merged["abs_improvement"] > 0).sum()/len(merged):.0f}%)')
    print(f'  Mean Abs Improvement: ${merged["abs_improvement"].mean():.2f}')

print('\n' + '=' * 80)
print('BY SYMBOL (OFI_ABLATION)')
print('=' * 80)
baseline_sym = df[df['strategy'] == 'symmetric_baseline'][['symbol', 'date', 'total_pnl']].rename(columns={'total_pnl': 'baseline_pnl'})
ablation_sym = df[df['strategy'] == 'ofi_ablation'][['symbol', 'date', 'total_pnl']].rename(columns={'total_pnl': 'strat_pnl'})
merged_sym = baseline_sym.merge(ablation_sym, on=['symbol', 'date'])

for sym in ['AAPL', 'AMD', 'AMZN', 'MSFT', 'NVDA']:
    sym_data = merged_sym[merged_sym['symbol'] == sym].copy()
    sym_data_valid = sym_data[sym_data['baseline_pnl'] != 0].copy()
    sym_data_valid['improvement'] = ((sym_data_valid['strat_pnl'] - sym_data_valid['baseline_pnl']) / sym_data_valid['baseline_pnl'].abs()) * 100
    sym_data['abs_improvement'] = sym_data['strat_pnl'] - sym_data['baseline_pnl']
    
    wins = (sym_data['abs_improvement'] > 0).sum()
    total = len(sym_data)
    print(f'{sym}: {sym_data_valid["improvement"].mean():+.1f}% (win rate: {wins}/{total} = {100*wins/total:.0f}%)')

print('\n' + '=' * 80)
print('BY SYMBOL (OFI_FULL)')
print('=' * 80)
full_sym = df[df['strategy'] == 'ofi_full'][['symbol', 'date', 'total_pnl']].rename(columns={'total_pnl': 'strat_pnl'})
merged_full = baseline_sym.merge(full_sym, on=['symbol', 'date'])

for sym in ['AAPL', 'AMD', 'AMZN', 'MSFT', 'NVDA']:
    sym_data = merged_full[merged_full['symbol'] == sym].copy()
    sym_data_valid = sym_data[sym_data['baseline_pnl'] != 0].copy()
    sym_data_valid['improvement'] = ((sym_data_valid['strat_pnl'] - sym_data_valid['baseline_pnl']) / sym_data_valid['baseline_pnl'].abs()) * 100
    sym_data['abs_improvement'] = sym_data['strat_pnl'] - sym_data['baseline_pnl']
    
    wins = (sym_data['abs_improvement'] > 0).sum()
    total = len(sym_data)
    print(f'{sym}: {sym_data_valid["improvement"].mean():+.1f}% (win rate: {wins}/{total} = {100*wins/total:.0f}%)')

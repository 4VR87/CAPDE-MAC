from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data' / 'raw' / 'raw_run_metrics.csv'
OUTDIR = ROOT / 'data' / 'derived'

METRICS = [
    'mean_delay_ms','p95_delay_ms','p99_delay_ms','emergency_mean_delay_ms',
    'deadline_miss_ratio','packet_delivery_ratio','collision_ratio','energy_per_delivered_mj'
]


def ci95(x):
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 2:
        return np.nan
    return float(t.ppf(0.975, len(x)-1) * x.std(ddof=1) / np.sqrt(len(x)))


def rank_biserial_from_differences(d):
    d = np.asarray(d, dtype=float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    order = np.argsort(np.abs(d))
    ranks = np.empty(len(d), dtype=float)
    ranks[order] = np.arange(1, len(d)+1)
    pos = ranks[d > 0].sum(); neg = ranks[d < 0].sum()
    denom = pos + neg
    return float((pos - neg) / denom) if denom else 0.0


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    adj = np.empty_like(pvals)
    running = 0.0
    m = len(pvals)
    for rank, idx in enumerate(order):
        val = min(1.0, (m-rank) * pvals[idx])
        running = max(running, val)
        adj[idx] = running
    return adj


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW)
    rows=[]
    for (protocol, load), g in df.groupby(['protocol','load'], sort=True):
        row={'protocol':protocol,'load':load,'n_seeds':len(g)}
        for m in METRICS:
            row[m+'_mean']=g[m].mean()
            row[m+'_ci95_halfwidth']=ci95(g[m])
        rows.append(row)
    summary=pd.DataFrame(rows)
    summary.to_csv(OUTDIR/'summary_by_protocol_load.csv', index=False)

    # Paired Wilcoxon at load 1.8 for the two main baselines versus CAPDE-MAC.
    tests=[]
    pvals=[]
    for baseline in ['static','priority_csma','adt_mac','mdp_hymac']:
        a=df[(df.protocol=='capde') & (df.load==1.8)][['seed','mean_delay_ms']].sort_values('seed')
        b=df[(df.protocol==baseline) & (df.load==1.8)][['seed','mean_delay_ms']].sort_values('seed')
        m=a.merge(b,on='seed',suffixes=('_capde','_baseline'))
        diff=m.mean_delay_ms_baseline-m.mean_delay_ms_capde
        stat,p=wilcoxon(diff, alternative='two-sided', zero_method='wilcox')
        pvals.append(p)
        tests.append({
            'comparison':f'{baseline} - capde',
            'load':1.8,
            'median_reduction_ms':float(np.median(diff)),
            'wilcoxon_statistic':float(stat),
            'p_value_raw':float(p),
            'rank_biserial_correlation':rank_biserial_from_differences(diff),
        })
    adj=holm_adjust(pvals)
    for r,padj in zip(tests,adj): r['p_value_holm']=float(padj)
    pd.DataFrame(tests).to_csv(OUTDIR/'paired_tests_load_1p8.csv', index=False)
    print(summary.to_string(index=False))
    print('\nPaired tests:')
    print(pd.DataFrame(tests).to_string(index=False))

if __name__=='__main__':
    main()

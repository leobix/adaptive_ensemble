#!/usr/bin/env python3
"""
Summarize synthetic benchmark results and produce tables and plots.

This script recursively loads results CSVs produced by src/main_synthetic_parallel.jl
(one CSV per run stored under --results_dir), aggregates them into a single
DataFrame, and exports:

- Summary table by Method
- Best row per Method (smallest metric)
- Line plot of metric vs Bias_Drift_range per Method
- Bar plot of best metric per Method

Usage examples:

  python results_analysis/synthetic_results_report.py \
    --glob 'results/**/results_synthetic_*.csv' \
    --metric RMSE \
    --out results_summary

Dependencies: pandas, numpy, matplotlib, seaborn
  pip install pandas numpy matplotlib seaborn
"""

import argparse
import glob
import os
import re
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def parse_hparams_from_dir(path: str) -> Dict[str, Any]:
    """Parse hyperparameters encoded in results_dir names produced by the sweep script.

    Expected tokens (optional):
      synth_eta_<v>_lam_<v>_td<d>_tl<l>_tt<t>_ge<e>_glr<r>_gmd<d>_gml<l>_gnt<t>
    """
    hp: Dict[str, Any] = {}
    base = os.path.basename(path.rstrip(os.sep))
    # Tokenize by '_' and also support k=v patterns in future
    toks = base.split('_')
    try:
        # Simple linear scan
        for i, t in enumerate(toks):
            if t == 'eta' and i + 1 < len(toks):
                hp['hedge_eta'] = float(toks[i + 1])
            if t == 'lam' and i + 1 < len(toks):
                hp['rls_lambda'] = float(toks[i + 1])
            if t.startswith('td'):
                hp['tree_max_depth'] = int(t[2:])
            if t.startswith('tl'):
                hp['tree_min_leaf'] = int(t[2:])
            if t.startswith('tt'):
                hp['tree_num_thresholds'] = int(t[2:])
            if t.startswith('ge'):
                hp['gbrt_estimators'] = int(t[2:])
            if t.startswith('glr'):
                hp['gbrt_lr'] = float(t[3:])
            if t.startswith('gmd'):
                hp['gbrt_max_depth'] = int(t[3:])
            if t.startswith('gml'):
                hp['gbrt_min_leaf'] = int(t[3:])
            if t.startswith('gnt'):
                hp['gbrt_num_thresholds'] = int(t[3:])
    except Exception:
        pass
    return hp


def parse_m_seed_from_dir(path: str) -> Dict[str, Any]:
    """Extract N_models (m) and seed from directory tag like '.../m_<m>_seed_<seed>__...'."""
    base = os.path.basename(path.rstrip(os.sep))
    out: Dict[str, Any] = {}
    try:
        m = re.search(r"m_(\d+)", base)
        s = re.search(r"seed_(\d+)", base)
        if m:
            out['N_models'] = int(m.group(1))
        if s:
            out['seed'] = int(s.group(1))
    except Exception:
        pass
    return out


def load_results(glob_pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(glob_pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"No results found for pattern: {glob_pattern}")
    dfs: List[pd.DataFrame] = []
    for f in files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] Failed reading {f}: {e}")
            continue
        df['results_dir'] = os.path.dirname(f)
        df['results_file'] = os.path.basename(f)
        hps = parse_hparams_from_dir(df['results_dir'].iloc[0])
        # Also parse m and seed from dir name if present
        hps.update(parse_m_seed_from_dir(df['results_dir'].iloc[0]))
        for k, v in hps.items():
            df[k] = v
        dfs.append(df)
    if not dfs:
        raise RuntimeError("No readable result CSVs were loaded.")
    all_res = pd.concat(dfs, ignore_index=True)
    # Normalize column names if needed
    return all_res


def best_by_method(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    # Smaller is better for RMSE/MAE/MAPE
    if metric not in df.columns:
        raise KeyError(f"Metric '{metric}' not found in columns: {list(df.columns)}")
    idx = df.groupby('Method')[metric].idxmin()
    return df.loc[idx].sort_values(metric).reset_index(drop=True)


def best_by_method_seed(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Best per (Method, N_models, seed) by minimizing the metric."""
    required = {'Method', 'N_models', 'seed', metric}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns for best_by_method_seed: {missing}")
    idx = df.groupby(['Method', 'N_models', 'seed'])[metric].idxmin()
    return df.loc[idx].reset_index(drop=True)


def summary_by_method(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    ag = df.groupby('Method')[metric].agg(['mean', 'std', 'count']).reset_index()
    return ag.sort_values('mean')


def plot_metric_by_drift(df: pd.DataFrame, metric: str, out_png: str):
    # Aggregate across runs for the same Bias_Drift_range per Method
    if 'Bias_Drift_range' not in df.columns:
        print("[WARN] Bias_Drift_range not present; skipping bias-drift plot.")
        return
    g = df.groupby(['Method', 'Bias_Drift_range'])[metric].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=g, x='Bias_Drift_range', y=metric, hue='Method', marker='o')
    plt.title(f'{metric} vs Bias_Drift_range (mean across runs)')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_best_by_method(df_best: pd.DataFrame, metric: str, out_png: str):
    plt.figure(figsize=(10, 6))
    plot_df = df_best.sort_values(metric)
    sns.barplot(data=plot_df, x='Method', y=metric, color='#4477aa')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Best {metric} per Method')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description='Summarize synthetic results and plot.')
    ap.add_argument('--glob', default='results/**/results_synthetic_*.csv', help='Glob pattern to load results CSVs (single set mode)')
    ap.add_argument('--val_glob', default=None, help='Glob for validation results (seed-wise selection mode)')
    ap.add_argument('--test_glob', default=None, help='Glob for test results (seed-wise selection mode)')
    ap.add_argument('--metric', default='RMSE', choices=['RMSE', 'MAE', 'MAPE', 'R2', 'CVAR_05', 'CVAR_15'], help='Metric to optimize/plot')
    ap.add_argument('--out', default='results_summary', help='Output directory for tables and plots')
    ap.add_argument('--filter-methods', nargs='*', default=None, help='Optional subset of methods to keep (e.g., hedge gbrt tree rls)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Seed-wise selection mode if val_glob and test_glob are provided
    if args.val_glob and args.test_glob:
        df_val = load_results(args.val_glob)
        if 'Dataset' in df_val.columns:
            df_val = df_val[df_val['Dataset'] == 'synthetic'].copy()
        if args.filter_methods:
            df_val = df_val[df_val['Method'].isin(args.filter_methods)].copy()

        best_val = best_by_method_seed(df_val, args.metric)
        best_val_path = os.path.join(args.out, f'best_val_by_method_seed_{args.metric}.csv')
        best_val.to_csv(best_val_path, index=False)

        df_test = load_results(args.test_glob)
        if 'Dataset' in df_test.columns:
            df_test = df_test[df_test['Dataset'] == 'synthetic'].copy()
        if args.filter_methods:
            df_test = df_test[df_test['Method'].isin(args.filter_methods)].copy()

        # Robust join keys (include Method, N_models, seed and parsed hparams)
        join_keys = ['Method', 'N_models', 'seed',
                     'hedge_eta', 'rls_lambda',
                     'tree_max_depth', 'tree_min_leaf', 'tree_num_thresholds',
                     'gbrt_estimators', 'gbrt_lr', 'gbrt_max_depth', 'gbrt_min_leaf', 'gbrt_num_thresholds']
        # Some keys may be absent if not encoded; keep only present ones
        present = [k for k in join_keys if k in best_val.columns and k in df_test.columns]
        if not present:
            raise RuntimeError("No common join keys found between validation and test results.")

        best_test = best_val.merge(df_test, on=present, suffixes=('_val', '_test'))
        best_test_path = os.path.join(args.out, f'best_val_matched_test_{args.metric}.csv')
        best_test.to_csv(best_test_path, index=False)

        # Aggregate across seeds for final table by (Method, N_models)
        metric_test_col = f'{args.metric}_test'
        if metric_test_col not in best_test.columns:
            # Fallback: if suffixes not applied (identical column names), use args.metric
            metric_test_col = args.metric
        table = best_test.groupby(['Method', 'N_models'])[metric_test_col].agg(['mean', 'std', 'count']).reset_index()
        table_path = os.path.join(args.out, f'final_by_method_m_{args.metric}.csv')
        table.to_csv(table_path, index=False)

        # Plots on validation (averaged) and summary bar from test bests
        plot_metric_by_drift(df_val, args.metric, os.path.join(args.out, f'val_{args.metric}_vs_Bias_Drift_range.png'))
        # Best per method on test
        best_per_method_test = best_test.loc[best_test.groupby('Method')[metric_test_col].idxmin()].reset_index(drop=True)
        plot_best_by_method(best_per_method_test.rename(columns={metric_test_col: args.metric}), args.metric,
                            os.path.join(args.out, f'best_{args.metric}_per_method_test.png'))

        print('Wrote:')
        print(' -', best_val_path)
        print(' -', best_test_path)
        print(' -', table_path)
        print(' -', os.path.join(args.out, f'val_{args.metric}_vs_Bias_Drift_range.png'))
        print(' -', os.path.join(args.out, f'best_{args.metric}_per_method_test.png'))
        return

    # Fallback: single-set mode (original behavior)
    df = load_results(args.glob)
    if 'Dataset' in df.columns:
        df = df[df['Dataset'] == 'synthetic'].copy()
    if args.filter_methods:
        df = df[df['Method'].isin(args.filter_methods)].copy()

    merged_path = os.path.join(args.out, 'all_results.csv')
    df.to_csv(merged_path, index=False)

    summ = summary_by_method(df, args.metric)
    summ_path = os.path.join(args.out, f'summary_by_method_{args.metric}.csv')
    summ.to_csv(summ_path, index=False)

    best = best_by_method(df, args.metric)
    best_path = os.path.join(args.out, f'best_by_method_{args.metric}.csv')
    best.to_csv(best_path, index=False)

    plot_metric_by_drift(df, args.metric, os.path.join(args.out, f'{args.metric}_vs_Bias_Drift_range.png'))
    plot_best_by_method(best, args.metric, os.path.join(args.out, f'best_{args.metric}_per_method.png'))

    print('Wrote:')
    print(' -', merged_path)
    print(' -', summ_path)
    print(' -', best_path)
    print(' -', os.path.join(args.out, f'{args.metric}_vs_Bias_Drift_range.png'))
    print(' -', os.path.join(args.out, f'best_{args.metric}_per_method.png'))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""评估所有异常检测数据集对 LaGraph v5 的适配性"""
import pandas as pd
import numpy as np
import os

files = {
    'SWaT':         'dataset/anomaly_detect/data/swat.csv',
    'MSL':          'dataset/anomaly_detect/data/MSL.csv',
    'CMAPSS_FD001': 'dataset/anomaly_detect/data/CMAPSS_FD001.csv',
    'CMAPSS_FD002': 'dataset/anomaly_detect/data/CMAPSS_FD002.csv',
    'CMAPSS_FD003': 'dataset/anomaly_detect/data/CMAPSS_FD003.csv',
    'CMAPSS_FD004': 'dataset/anomaly_detect/data/CMAPSS_FD004.csv',
    'SKAB':         'dataset/anomaly_detect/data/SKAB_all.csv',
}

HEADER = f"{'Dataset':<16s} {'Features':>10s} {'HasLabel':>9s} {'Anom%':>8s} {'Length':>8s} {'Verdict'}"
print('=' * 75)
print(HEADER)
print('-' * 75)

for name, fp in files.items():
    if not os.path.exists(fp):
        print(f'{name:<16s} {"FILE MISSING":>10s}')
        continue
    
    raw = pd.read_csv(fp)
    cols = raw['cols'].unique()
    has_label = 'label' in cols
    
    feat_cols = [c for c in cols if c != 'label']
    n_feat = len(feat_cols)
    
    if has_label:
        labels_raw = raw[raw['cols'] == 'label']['data'].values
        n_anom = int((labels_raw > 0).sum())
        n_total = len(labels_raw)
        anom_pct = n_anom / n_total * 100 if n_total > 0 else 0
        
        issues = []
        if n_feat < 2:
            issues.append('single-var')
        if n_total < 100:
            issues.append(f'short({n_total})')
        if n_anom == 0:
            issues.append('zero-anom')
        verdict = 'OK' if not issues else 'NO:' + ','.join(issues)
    else:
        n_total = raw['date'].nunique()
        anom_pct = 0
        verdict = 'NO: no label col'
    
    print(f'{name:<16s} {n_feat:>10d} {str(has_label):>9s} {anom_pct:>7.2f}% {n_total:>8d} {verdict}')

print('=' * 75)

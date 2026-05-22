#!/usr/bin/env python3
"""
将 SKAB (Skoltech Anomaly Benchmark) 数据集转换为系统三列格式。
输入: dataset/anomaly_detect/data/SKAB/
输出: dataset/anomaly_detect/data/SKAB_all.csv

原始格式: 单列分号分隔
  datetime;Acc1;Acc24;Current;Pressure;Temp;Thermo;Voltage;Flow;
  [anomaly;changepoint — 仅故障文件]

输出格式: date, data, cols (三列长格式)
"""

import pandas as pd
import numpy as np
import os
import sys

SKAB_DIR = 'dataset/anomaly_detect/data/SKAB'
OUTPUT_FILE = 'dataset/anomaly_detect/data/SKAB_all.csv'

SENSOR_COLS = [
    'Accelerometer1RMS', 'Accelerometer2RMS', 'Current',
    'Pressure', 'Temperature', 'Thermocouple', 'Voltage',
    'Volume Flow RateRMS'
]
NUM_SENSORS = len(SENSOR_COLS)
SKIP_COLS = ['datetime', 'anomaly', 'changepoint']


def parse_skab_csv(filepath):
    """Parse a SKAB single-column semicolon-delimited CSV."""
    # Read all lines
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    if not lines:
        return None, None
    
    # Check header
    header = lines[0].strip().split(';')
    has_anomaly = 'anomaly' in header
    has_changepoint = 'changepoint' in header
    
    data_rows = []
    anomaly_rows = []
    
    for line in lines[1:]:
        parts = line.strip().split(';')
        if len(parts) < 1 + NUM_SENSORS:
            continue
        try:
            sensor_vals = [float(x) for x in parts[1:1+NUM_SENSORS]]
            data_rows.append(sensor_vals)
            if has_anomaly and len(parts) > 1+NUM_SENSORS:
                anomaly_rows.append(float(parts[1+NUM_SENSORS]))
            elif has_changepoint and len(parts) > 2+NUM_SENSORS:
                anomaly_rows.append(float(parts[2+NUM_SENSORS]))
        except (ValueError, IndexError):
            continue
    
    arr = np.array(data_rows, dtype=np.float32)
    if has_anomaly:
        anom_arr = np.array(anomaly_rows, dtype=np.float32)
    else:
        anom_arr = np.zeros(len(data_rows), dtype=np.float32)
    
    return arr, anom_arr


def convert_to_long_format(data_array, label_array):
    """Convert (T, C) array + (T,) labels to long format DataFrame."""
    T, C = data_array.shape
    records = []
    
    for t in range(T):
        for c in range(C):
            records.append({
                'date': t + 1,  # 1-indexed
                'data': data_array[t, c],
                'cols': SENSOR_COLS[c]
            })
        # Append label row
        records.append({
            'date': t + 1,
            'data': label_array[t],
            'cols': 'label'
        })
    
    return pd.DataFrame(records)


def main():
    all_parts = []
    total_steps = 0
    
    # Process subdirectories
    subdirs = ['anomaly-free', 'valve1', 'valve2', 'other']
    
    for sub in subdirs:
        subpath = os.path.join(SKAB_DIR, sub)
        if not os.path.isdir(subpath):
            print(f'  SKIP {sub}: not found')
            continue
        
        csv_files = sorted([
            f for f in os.listdir(subpath) if f.endswith('.csv')
        ])
        
        for csv_file in csv_files:
            filepath = os.path.join(subpath, csv_file)
            arr, label = parse_skab_csv(filepath)
            
            if arr is None:
                print(f'  SKIP {sub}/{csv_file}: empty')
                continue
            
            T = arr.shape[0]
            print(f'  {sub}/{csv_file}: {arr.shape} → {T*(NUM_SENSORS+1)} rows')
            
            df = convert_to_long_format(arr, label)
            df['date'] += total_steps  # offset dates for continuity
            all_parts.append(df)
            total_steps += T
    
    if not all_parts:
        print('ERROR: No data converted!')
        sys.exit(1)
    
    # Concatenate all parts
    full_df = pd.concat(all_parts, ignore_index=True)
    
    # SKAB labels are continuous (0~1), threshold to binary
    label_mask = full_df['cols'] == 'label'
    full_df.loc[label_mask, 'data'] = (full_df.loc[label_mask, 'data'] > 0.5).astype(int)
    
    full_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f'\n=== SKAB conversion complete ===')
    print(f'Output: {OUTPUT_FILE}')
    print(f'Shape: {full_df.shape}')
    print(f'Time steps: {total_steps}')
    print(f'Channels: {NUM_SENSORS}')
    n_anomalies = (full_df[label_mask]['data'] == 1).sum()
    n_total_labels = label_mask.sum()
    print(f'Label stats: {n_anomalies} anomalies / {n_total_labels} total ({100*n_anomalies/max(n_total_labels,1):.2f}%)')
    print(f'Samples: {full_df.iloc[:3000:9, :].to_string()}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TE (Tennessee Eastman) 数据集格式转换脚本
==========================================
将原始空格分隔的宽格式 .dat 文件转换为系统兼容的 CSV 长格式。

TE数据集结构：
  - 22个故障类型 (d00~d21)，其中d00为正常工况
  - train/ 目录：正常运行数据（标签全0）
  - test/ 目录：故障运行数据（标签全1）
  - 每个文件：52列传感器读数，空格分隔，无header

转换策略：
  1. 对每个故障类型，合并 train + test 为一个文件
  2. 前N_行（train）label=0，后N_行（test）label=1
  3. 输出CSV长格式：date, data, cols
  4. 同时在原data/目录存放，并在DETECT_META.csv中注册
"""

import os
import sys
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TE_TRAIN_DIR = os.path.join(PROJECT_ROOT, "dataset", "anomaly_detect", "data", "TE", "train")
TE_TEST_DIR = os.path.join(PROJECT_ROOT, "dataset", "anomaly_detect", "data", "TE", "test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "anomaly_detect", "data")
META_PATH = os.path.join(PROJECT_ROOT, "dataset", "anomaly_detect", "DETECT_META.csv")

# TE传感器名称（52个变量）
TE_VARIABLE_NAMES = [
    "XMEAS_1",  "XMEAS_2",  "XMEAS_3",  "XMEAS_4",  "XMEAS_5",
    "XMEAS_6",  "XMEAS_7",  "XMEAS_8",  "XMEAS_9",  "XMEAS_10",
    "XMEAS_11", "XMEAS_12", "XMEAS_13", "XMEAS_14", "XMEAS_15",
    "XMEAS_16", "XMEAS_17", "XMEAS_18", "XMEAS_19", "XMEAS_20",
    "XMEAS_21", "XMEAS_22", "XMV_1",    "XMV_2",    "XMV_3",
    "XMV_4",    "XMV_5",    "XMV_6",    "XMV_7",    "XMV_8",
    "XMV_9",    "XMV_10",   "XMV_11",   "XMV_12",   "XMEAS_23",
    "XMEAS_24", "XMEAS_25", "XMEAS_26", "XMEAS_27", "XMEAS_28",
    "XMEAS_29", "XMEAS_30", "XMEAS_31", "XMEAS_32", "XMEAS_33",
    "XMEAS_34", "XMEAS_35", "XMEAS_36", "XMEAS_37", "XMEAS_38",
    "XMEAS_39", "XMEAS_40",
]

# TE故障类型说明
TE_FAULT_DESCRIPTIONS = {
    0:  "Normal (no fault)",
    1:  "A/C Feed Ratio, B Composition Constant (Stream 4)",
    2:  "B Composition, A/C Ratio Constant (Stream 4)",
    3:  "D Feed Temperature (Stream 2)",
    4:  "Reactor Cooling Water Inlet Temperature",
    5:  "Condenser Cooling Water Inlet Temperature",
    6:  "A Feed Loss (Stream 1)",
    7:  "C Header Pressure Loss - Reduced Availability (Stream 4)",
    8:  "A, B, C Feed Composition (Stream 4)",
    9:  "D Feed Temperature (Stream 2)",
    10: "C Feed Temperature (Stream 4)",
    11: "Reactor Cooling Water Inlet Temperature",
    12: "Condenser Cooling Water Inlet Temperature",
    13: "Reaction Kinetics",
    14: "Reactor Cooling Water Valve",
    15: "Condenser Cooling Water Valve",
    16: "Unknown",
    17: "Unknown",
    18: "Unknown",
    19: "Unknown",
    20: "Unknown",
    21: "Valve Position Constant (Stream 4)",
}


def load_dat_file(filepath: str) -> np.ndarray:
    """
    加载空格分隔的 .dat 文件
    
    参数：
        filepath: .dat 文件路径
    
    返回：
        numpy array，形状为 (时间步, 52)
    """
    data = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 空格分隔
            row = [float(x) for x in line.split()]
            data.append(row)
    arr = np.array(data)
    logger.info("  Loaded %s: %d rows x %d cols", os.path.basename(filepath), arr.shape[0], arr.shape[1])
    return arr


def concatenate_train_test(fault_id: int) -> tuple:
    """
    合并train和test数据，构造标签
    
    参数：
        fault_id: 故障ID (0~21)
    
    返回：
        (combined_data, labels, train_len)
        combined_data: (总行数, 52) numpy array
        labels: (总行数,) 0=正常, 1=故障
        train_len: 训练（正常）部分长度
    """
    train_file = os.path.join(TE_TRAIN_DIR, f"d{fault_id:02d}.dat")
    test_file = os.path.join(TE_TEST_DIR, f"d{fault_id:02d}_te.dat")
    
    if not os.path.exists(train_file):
        raise FileNotFoundError(f"Training file not found: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    train_data = load_dat_file(train_file)
    test_data = load_dat_file(test_file)
    
    train_len = train_data.shape[0]
    test_len = test_data.shape[0]
    
    # 合并
    combined = np.vstack([train_data, test_data])
    
    # 构造标签：train全0，test全1
    labels = np.zeros(combined.shape[0], dtype=int)
    labels[train_len:] = 1
    
    logger.info("  Combined: %d train + %d test = %d total, anomaly ratio: %.2f%%",
                train_len, test_len, combined.shape[0],
                100 * labels.sum() / len(labels))
    
    return combined, labels, train_len


def convert_to_long_format(data: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    将宽格式数据转换为系统期望的CSV长格式
    
    系统期望格式 (来自 data/utils.py process_data_df):
      date, data, cols
      1, 2.1466, col_0
      2, 2.1466, col_0
      ...
      1, 0, label
      2, 0, label
      ...
    
    参数：
        data: (N, 52) numpy array，传感器数据
        labels: (N,) numpy array，标签
    
    返回：
        长格式DataFrame，列为 [date, data, cols]
    """
    n_timesteps, n_cols = data.shape
    rows = []
    
    # 添加传感器数据行
    for col_idx in range(n_cols):
        col_name = f"col_{col_idx}"  # 使用col_前缀，与系统一致
        for t in range(n_timesteps):
            rows.append({
                "date": t + 1,  # 时间步从1开始
                "data": data[t, col_idx],
                "cols": col_name,
            })
    
    # 添加标签行
    for t in range(n_timesteps):
        rows.append({
            "date": t + 1,
            "data": float(labels[t]),
            "cols": "label",
        })
    
    df = pd.DataFrame(rows)
    logger.info("  Long format: %d rows (%d sensors × %d timesteps + labels)",
                len(df), n_cols, n_timesteps)
    return df


def convert_fault(fault_id: int) -> str:
    """
    转换单个故障类型的所有数据
    
    参数：
        fault_id: 故障ID (0~21)
    
    返回：
        输出CSV文件名
    """
    fault_name = f"TE_d{fault_id:02d}"
    output_file = f"{fault_name}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_file)
    
    logger.info("Converting TE fault %d: %s", fault_id, TE_FAULT_DESCRIPTIONS.get(fault_id, "Unknown"))
    
    # 合并train+test
    combined_data, labels, train_len = concatenate_train_test(fault_id)
    
    # 转为长格式
    df_long = convert_to_long_format(combined_data, labels)
    
    # 保存
    df_long.to_csv(output_path, index=False)
    logger.info("  Saved to: %s", output_path)
    
    # 计算元数据
    n_timesteps = combined_data.shape[0]
    n_cols = combined_data.shape[1]
    
    return {
        "file_name": output_file,
        "fault_id": fault_id,
        "description": TE_FAULT_DESCRIPTIONS.get(fault_id, "Unknown"),
        "n_timesteps": n_timesteps,
        "n_features": n_cols,
        "train_len": train_len,
        "anomaly_ratio": float(labels.mean()),
    }


def update_metadata(meta_list: list):
    """
    更新 DETECT_META.csv
    
    参数：
        meta_list: 包含各数据集元信息的列表
    """
    # 读取现有元数据
    if os.path.exists(META_PATH):
        existing = pd.read_csv(META_PATH)
    else:
        existing = pd.DataFrame()
    
    # 构建新行
    new_rows = []
    for m in meta_list:
        new_rows.append({
            "file_name": m["file_name"],
            "freq": "other",
            "if_univariate": False,
            "size": "small",  # TE每文件约2000行，不算large
            "length": m["n_timesteps"],
            "trend": "",
            "seasonal": "",
            "stationary": "",
            "transition": "",
            "shifting": "",
            "correlation": "",
            "dataset_name": "TE",
            "type_value": m["description"],
            "train_lens": m["train_len"],
            "time_steps": "",
            "pattern": f"fault_{m['fault_id']}",
        })
    
    df_new = pd.DataFrame(new_rows)
    
    # 合并（去重：如果文件名已存在则更新）
    if not existing.empty:
        existing = existing[~existing["file_name"].isin(df_new["file_name"].values)]
        df_merged = pd.concat([existing, df_new], ignore_index=True)
    else:
        df_merged = df_new
    
    # 保存
    df_merged.to_csv(META_PATH, index=False)
    logger.info("Updated DETECT_META.csv with %d TE datasets", len(new_rows))
    
    return df_merged


def main():
    logger.info("=" * 60)
    logger.info("TE Dataset Conversion Tool")
    logger.info("=" * 60)
    
    meta_list = []
    success_count = 0
    fail_count = 0
    
    for fault_id in range(22):  # d00 ~ d21
        try:
            meta = convert_fault(fault_id)
            meta_list.append(meta)
            success_count += 1
        except FileNotFoundError as e:
            logger.warning("  SKIP fault %d: %s", fault_id, e)
            fail_count += 1
        except Exception as e:
            logger.error("  FAIL fault %d: %s", fault_id, e)
            fail_count += 1
    
    logger.info("-" * 60)
    logger.info("Conversion complete: %d success, %d failed", success_count, fail_count)
    
    if meta_list:
        update_metadata(meta_list)
        logger.info("Metadata updated successfully")
    
    logger.info("=" * 60)
    
    # 打印转换摘要
    print("\n📊 TE Dataset Conversion Summary")
    print("-" * 60)
    print(f"{'File':20s} {'Fault':6s} {'Samples':>8s} {'Train':>8s} {'Anom%':>8s}  Description")
    print("-" * 60)
    for m in meta_list:
        print(f"{m['file_name']:20s} d{m['fault_id']:02d}   {m['n_timesteps']:>8d} {m['train_len']:>8d} {m['anomaly_ratio']*100:>7.1f}%  {m['description'][:40]}")
    print("-" * 60)


if __name__ == "__main__":
    main()

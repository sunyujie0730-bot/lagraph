#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CMAPSS (Commercial Modular Aero-Propulsion System Simulation) 数据集格式转换脚本
===============================================================================
将 NASA CMAPSS 涡轮风扇发动机退化数据转换为系统兼容的 CSV 长格式。

CMAPSS数据集结构（FD001为例）：
  train_FD001.txt: 发动机从正常运行到故障的全生命周期数据
  test_FD001.txt:  截断的运行数据（需要预测RUL）
  RUL_FD001.txt:   测试集发动机的剩余使用寿命标签
  
  每行格式（空格分隔，26列）：
    unit_id, time_cycle, op_setting_1, op_setting_2, op_setting_3, 
    sensor_1 ~ sensor_21
  
  train_FD001: 100 engines × variable cycles, ~20,631 rows
  test_FD001:  100 engines × variable cycles
  
转换策略（用于异常检测）：
  核心思路：退化过程本身是一个逐渐偏离正常的过程
  
  方案：将早期运行周期标记为正常(label=0)，晚期标记为异常(label=1)
  
  具体做法：
  1. 对每台发动机，取其生命周期时间线
  2. 前 normal_ratio（默认70%）的周期标记为正常
  3. 后 1-normal_ratio（默认30%）的周期标记为异常
  4. 合并所有发动机的数据
  
  输出格式：CSV长格式 date, data, cols（与系统完全兼容）
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 路径配置
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMAPSS_DIR = os.path.join(PROJECT_ROOT, "dataset", "anomaly_detect", "data", "CMAPSSData")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "dataset", "anomaly_detect", "data")
META_PATH = os.path.join(PROJECT_ROOT, "dataset", "anomaly_detect", "DETECT_META.csv")

# CMAPSS列名
CMAPSS_COLUMNS = [
    "unit_id",        # 发动机编号
    "time_cycle",     # 运行周期
    "op_setting_1",   # 操作设置1
    "op_setting_2",   # 操作设置2
    "op_setting_3",   # 操作设置3
] + [f"sensor_{i}" for i in range(1, 22)]  # 21个传感器

# 剔除常值传感器（对异常检测无意义）
CONSTANT_SENSORS = [1, 5, 6, 10, 16, 18, 19]  # 索引1-based
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22) if i not in CONSTANT_SENSORS]
# 14个有效传感器

# 数据集子集配置
CMAPSS_SUBSETS = {
    "FD001": {
        "train": "train_FD001.txt",
        "test": "test_FD001.txt",
        "rul": "RUL_FD001.txt",
        "conditions": 1,  # 单一工况
        "faults": 1,      # 单一故障模式(HPC退化)
    },
    "FD002": {
        "train": "train_FD002.txt",
        "test": "test_FD002.txt",
        "rul": "RUL_FD002.txt",
        "conditions": 6,
        "faults": 1,
    },
    "FD003": {
        "train": "train_FD003.txt",
        "test": "test_FD003.txt",
        "rul": "RUL_FD003.txt",
        "conditions": 1,
        "faults": 2,
    },
    "FD004": {
        "train": "train_FD004.txt",
        "test": "test_FD004.txt",
        "rul": "RUL_FD004.txt",
        "conditions": 6,
        "faults": 2,
    },
}


def load_cmapss_file(filepath: str) -> pd.DataFrame:
    """
    加载CMAPSS空格分隔的数据文件
    
    参数：
        filepath: 文件路径
    
    返回：
        DataFrame，列名为CMAPSS_COLUMNS
    """
    df = pd.read_csv(
        filepath,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
    )
    logger.info("  Loaded %s: %d rows", os.path.basename(filepath), len(df))
    return df


def construct_anomaly_labels(
    train_df: pd.DataFrame,
    normal_ratio: float = 0.7,
    degradation_margin: float = 0.05,
) -> pd.DataFrame:
    """
    从训练数据构造异常检测标签
    
    策略：
      - 前 normal_ratio 的周期视为正常（early life）
      - degradation_margin 的过渡区
      - 后 (1 - normal_ratio - degradation_margin) 视为异常（degradation）
    
    参数：
        train_df: 训练数据DataFrame（含unit_id, time_cycle列）
        normal_ratio: 正常周期占比
        degradation_margin: 过渡区占比
    
    返回：
        添加了"label"列的DataFrame
    """
    df = train_df.copy()
    df["label"] = 0  # 默认正常
    
    for unit_id, group in df.groupby("unit_id"):
        max_cycle = group["time_cycle"].max()
        normal_threshold = int(max_cycle * normal_ratio)
        anomaly_threshold = int(max_cycle * (normal_ratio + degradation_margin))
        
        # 正常: 0 ~ normal_threshold
        # 过渡: normal_threshold ~ anomaly_threshold (不确定，标记为正常)
        # 异常: anomaly_threshold ~ max_cycle
        mask_anomaly = group["time_cycle"] > anomaly_threshold
        df.loc[group[mask_anomaly].index, "label"] = 1
    
    anomaly_count = df["label"].sum()
    total = len(df)
    logger.info("  Labels: %d normal, %d anomaly (%.1f%%)",
                total - anomaly_count, anomaly_count, 100 * anomaly_count / total)
    
    return df


def convert_to_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    将宽格式DataFrame转换为系统期望的CSV长格式
    
    系统期望格式:
      date, data, cols
      1, 2.1466, col_0
      ...
      1, 0, label
    
    参数：
        df: 含传感器数据和label列的DataFrame
    
    返回：
        长格式DataFrame，列为 [date, data, cols]
    """
    rows = []
    
    # 使用全局时间步索引（跨所有发动机）
    df_sorted = df.sort_values(["unit_id", "time_cycle"]).reset_index(drop=True)
    
    # 传感器数据列（去除unit_id和time_cycle）
    data_cols = [c for c in SENSOR_COLUMNS if c in df.columns]
    
    n_rows = len(df_sorted)
    
    # 添加传感器数据行
    for col_idx, col_name in enumerate(data_cols):
        col_alias = f"col_{col_idx}"
        for t in range(n_rows):
            rows.append({
                "date": t + 1,
                "data": float(df_sorted.iloc[t][col_name]),
                "cols": col_alias,
            })
    
    # 添加标签行
    if "label" in df_sorted.columns:
        for t in range(n_rows):
            rows.append({
                "date": t + 1,
                "data": float(df_sorted.iloc[t]["label"]),
                "cols": "label",
            })
    
    result = pd.DataFrame(rows)
    logger.info("  Long format: %d rows (%d sensors × %d timesteps + labels)",
                len(result), len(data_cols), n_rows)
    return result


def convert_subset(subset_name: str, normal_ratio: float = 0.7) -> dict:
    """
    转换单个CMAPSS子集
    
    参数：
        subset_name: 子集名称 (FD001~FD004)
        normal_ratio: 正常周期占比
    
    返回：
        包含元信息的字典
    """
    logger.info("Converting CMAPSS %s", subset_name)
    config = CMAPSS_SUBSETS[subset_name]
    
    train_path = os.path.join(CMAPSS_DIR, config["train"])
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found: {train_path}")
    
    # 加载训练数据
    train_df = load_cmapss_file(train_path)
    
    # 构造标签
    train_df = construct_anomaly_labels(train_df, normal_ratio=normal_ratio)
    
    # 获取统计信息
    n_engines = train_df["unit_id"].nunique()
    n_timesteps = len(train_df)
    n_features = len(SENSOR_COLUMNS)
    anomaly_ratio = train_df["label"].mean()
    normal_len = int(n_timesteps * (1 - anomaly_ratio))
    
    # 转为长格式
    df_long = convert_to_long_format(train_df)
    
    # 保存
    output_file = f"CMAPSS_{subset_name}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_file)
    df_long.to_csv(output_path, index=False)
    logger.info("  Saved to: %s", output_path)
    
    return {
        "file_name": output_file,
        "subset": subset_name,
        "description": (
            f"Turbofan Engine Degradation"
            f" ({config['conditions']} cond, {config['faults']} fault)"
        ),
        "n_engines": n_engines,
        "n_timesteps": n_timesteps,
        "n_features": n_features,
        "normal_len": normal_len,
        "anomaly_ratio": anomaly_ratio,
    }


def update_metadata(meta_list: list, existing_meta_path: str) -> pd.DataFrame:
    """
    更新 DETECT_META.csv
    
    参数：
        meta_list: 包含各数据集元信息的列表
        existing_meta_path: 现有元数据文件路径
    
    返回：
        合并后的元数据DataFrame
    """
    if os.path.exists(existing_meta_path):
        existing = pd.read_csv(existing_meta_path)
    else:
        existing = pd.DataFrame()
    
    new_rows = []
    for m in meta_list:
        new_rows.append({
            "file_name": m["file_name"],
            "freq": "other",
            "if_univariate": False,
            "size": "large" if m["n_timesteps"] > 10000 else "small",
            "length": m["n_timesteps"],
            "trend": "",
            "seasonal": "",
            "stationary": "",
            "transition": "",
            "shifting": "",
            "correlation": "",
            "dataset_name": f"CMAPSS_{m['subset']}",
            "type_value": m["description"],
            "train_lens": m["normal_len"],
            "time_steps": "",
            "pattern": f"engine_degradation_{m['n_engines']}engines",
        })
    
    df_new = pd.DataFrame(new_rows)
    
    if not existing.empty:
        existing = existing[~existing["file_name"].isin(df_new["file_name"].values)]
        df_merged = pd.concat([existing, df_new], ignore_index=True)
    else:
        df_merged = df_new
    
    df_merged.to_csv(existing_meta_path, index=False)
    logger.info("Updated %s with %d CMAPSS datasets", existing_meta_path, len(new_rows))
    
    return df_merged


def main():
    parser = argparse.ArgumentParser(
        description="Convert CMAPSS dataset to benchmark-compatible CSV format"
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["FD001", "FD002", "FD003", "FD004"],
        help="CMAPSS subsets to convert (default: all)",
    )
    parser.add_argument(
        "--normal-ratio",
        type=float,
        default=0.7,
        help="Ratio of early cycles treated as normal (default: 0.7)",
    )
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("CMAPSS Dataset Conversion Tool")
    logger.info("=" * 60)
    logger.info("Normal ratio: %.1f (first %.0f%% cycles = normal)", 
                args.normal_ratio, args.normal_ratio * 100)
    
    meta_list = []
    success_count = 0
    fail_count = 0
    
    for subset in args.subsets:
        if subset not in CMAPSS_SUBSETS:
            logger.warning("Unknown subset: %s, skipping", subset)
            fail_count += 1
            continue
        
        try:
            meta = convert_subset(subset, normal_ratio=args.normal_ratio)
            meta_list.append(meta)
            success_count += 1
        except FileNotFoundError as e:
            logger.warning("  SKIP %s: %s", subset, e)
            fail_count += 1
        except Exception as e:
            logger.error("  FAIL %s: %s", subset, e, exc_info=True)
            fail_count += 1
    
    logger.info("-" * 60)
    logger.info("Conversion complete: %d success, %d failed", success_count, fail_count)
    
    if meta_list:
        update_metadata(meta_list, META_PATH)
        logger.info("Metadata updated successfully")
    
    logger.info("=" * 60)
    
    # 打印转换摘要
    print("\n📊 CMAPSS Dataset Conversion Summary")
    print("-" * 75)
    print(f"{'File':22s} {'Engines':>7s} {'Samples':>8s} {'Features':>8s} {'Anom%':>7s}  Description")
    print("-" * 75)
    for m in meta_list:
        print(f"{m['file_name']:22s} {m['n_engines']:>7d} {m['n_timesteps']:>8d} "
              f"{m['n_features']:>8d} {m['anomaly_ratio']*100:>6.1f}%  {m['description']}")
    print("-" * 75)


if __name__ == "__main__":
    main()

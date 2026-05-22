#!/usr/bin/env python3
"""
诊断脚本 — 重建误差分布可视化（Step 1 of 验证实验）
===================================================

目的：
  验证模型是否能够区分正常/异常模式。
  通过可视化训练集和测试集的逐点重建误差分布，判断：
  - 正常模式的 MSE 是否集中在低值区域
  - 异常模式的 MSE 是否显著右偏/有长尾
  - 当前阈值选取是否合理

用法：
  cd /home/professor3/Lagraph/LaGraph
  python scripts/diagnose_recon_dist.py                          # 默认 swat.csv
  python scripts/diagnose_recon_dist.py --dataset MSL.csv        # 指定数据集
  python scripts/diagnose_recon_dist.py --epochs 10 --fast       # 快速测试

输出：
  result/diagnosis/<dataset>/recon_dist_<timestamp>/
    recon_distribution.png   — 训练/测试/异常的误差分布直方图
    score_ts.png             — 分数时间序列 + 真实标签
    diagnostic_report.json   — 诊断报告（含统计量）
    threshold_analysis.png   — 不同异常率下的阈值 vs 实际分数
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime

import matplotlib
matplotlib.use('Agg')  # 不依赖 GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ts_benchmark.baselines.self_impl.LaGraph.LaGraph import LaGraph
from ts_benchmark.baselines.utils import anomaly_detection_data_provider
from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource
from ts_benchmark.utils.data_processing import split_before

warnings.filterwarnings("ignore")

plt.rcParams.update({
    'figure.dpi': 120,
    'figure.figsize': (14, 8),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def load_dataset(dataset_name: str):
    """加载数据集并切分（Fixed模式：前80%训练，后20%测试）"""
    print(f"\n  [DATA] Loading dataset: {dataset_name}")

    data_src = LocalAnomalyDetectDataSource()
    meta = data_src.dataset.metadata
    row = meta[meta["file_name"] == dataset_name]
    if len(row) == 0:
        raise FileNotFoundError(f"Dataset {dataset_name} not found in DETECT_META.csv")

    # 加载数据（与 pipeline.py / run_single.py 一致的使用方式）
    data_src.load_series_list([dataset_name])
    data = data_src.dataset.get_series(dataset_name)

    # 分离特征和标签
    train_ratio = 0.8
    train_length = int(train_ratio * len(data))
    train, test = split_before(data, train_length)

    train_data = train.loc[:, train.columns != "label"]
    train_label = train.loc[:, ["label"]].values.flatten()

    test_data = test.loc[:, test.columns != "label"]
    test_label = test.loc[:, ["label"]].values.flatten()

    print(f"    Train: {len(train_data)} samples, "
          f"anomaly ratio: {train_label.mean()*100:.2f}%")
    print(f"    Test:  {len(test_data)} samples, "
          f"anomaly ratio: {test_label.mean()*100:.2f}%")
    print(f"    Features: {train_data.shape[1]}")

    return train_data, train_label, test_data, test_label


def compute_recon_scores(model, data_df: pd.DataFrame, batch_size: int = 64):
    """
    返回每个时间点的重建误差（MSE per point, max over channels）。
    直接调用模型的 detect_score 逻辑，但返回逐点误差而非聚合分数。
    """
    model.model.eval()
    scaled = pd.DataFrame(
        model.scaler.transform(data_df.values),
        columns=data_df.columns, index=data_df.index,
    )

    total_length = len(scaled)
    win_size = model.config.win_size

    loader = anomaly_detection_data_provider(
        scaled, batch_size=batch_size,
        win_size=win_size, step=1, mode="test",
    )

    window_scores_list = []
    with torch.no_grad():
        for i, (input_data, _) in enumerate(loader):
            input_data = input_data.float().to(model.device)
            scores_batch = model._detect_forward(input_data)  # (B, win_size)
            window_scores_list.append(scores_batch)
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

    if len(window_scores_list) == 0:
        return np.zeros(total_length)

    window_scores = np.concatenate(window_scores_list, axis=0)  # (N_windows, win_size)

    # 点级聚合
    point_scores = model._point_wise_aggregate(
        window_scores, win_size, total_length,
    )
    return point_scores


def diagnose(
    dataset_name: str = "swat.csv",
    train_epochs: int = 10,
    n_gpus: int = 1,
    output_dir: str = None,
):
    """主诊断流程"""
    t_start = time.time()

    # 1. 加载数据
    train_data, train_label, test_data, test_label = load_dataset(dataset_name)

    # 2. 训练模型
    print(f"\n  [TRAIN] Fitting LaGraph on {dataset_name} ({train_epochs} epochs)...")
    model = LaGraph(
        num_epochs=train_epochs,
        n_gpus=n_gpus,
        dataset_name=dataset_name,
    )
    model.detect_fit(train_data, test_data)

    # 3. 计算重建误差
    print(f"\n  [SCORE] Computing reconstruction errors...")
    train_scores = compute_recon_scores(model, train_data)
    test_scores = compute_recon_scores(model, test_data)

    # 对齐标签长度（point_scores 可能比原始数据短几个点，由于窗口边界）
    min_len_train = min(len(train_scores), len(train_label))
    min_len_test = min(len(test_scores), len(test_label))

    train_scores = train_scores[:min_len_train]
    train_label = train_label[:min_len_train]
    test_scores = test_scores[:min_len_test]
    test_label = test_label[:min_len_test]

    # 4. 分离正常/异常分数
    train_normal_scores = train_scores[train_label == 0]
    test_normal_scores = test_scores[test_label == 0]
    test_anomaly_scores = test_scores[test_label == 1]

    # 5. 计算统计量
    stats = {
        "dataset": dataset_name,
        "train_samples": len(train_scores),
        "test_samples": len(test_scores),
        "train_anomaly_ratio": float(train_label.mean()),
        "test_anomaly_ratio": float(test_label.mean()),
        "train_normal": {
            "n": int(len(train_normal_scores)),
            "mean": float(np.mean(train_normal_scores)),
            "std": float(np.std(train_normal_scores)),
            "p50": float(np.percentile(train_normal_scores, 50)),
            "p95": float(np.percentile(train_normal_scores, 95)),
            "p99": float(np.percentile(train_normal_scores, 99)),
            "max": float(np.max(train_normal_scores)),
        },
        "test_normal": {
            "n": int(len(test_normal_scores)),
            "mean": float(np.mean(test_normal_scores)),
            "std": float(np.std(test_normal_scores)),
            "p50": float(np.percentile(test_normal_scores, 50)),
            "p95": float(np.percentile(test_normal_scores, 95)),
            "p99": float(np.percentile(test_normal_scores, 99)),
            "max": float(np.max(test_normal_scores)),
        },
        "test_anomaly": {
            "n": int(len(test_anomaly_scores)),
            "mean": float(np.mean(test_anomaly_scores)),
            "std": float(np.std(test_anomaly_scores)),
            "p50": float(np.percentile(test_anomaly_scores, 50)),
            "p95": float(np.percentile(test_anomaly_scores, 95)),
            "p99": float(np.percentile(test_anomaly_scores, 99)),
            "max": float(np.max(test_anomaly_scores)),
        },
        "separation_metrics": {
            # 异常 vs 正常平均分的比值（越大越好）
            "anomaly_normal_mean_ratio": (
                float(np.mean(test_anomaly_scores)) /
                max(float(np.mean(test_normal_scores)), 1e-10)
            ),
            # 异常 P50 vs 正常 P95（理想 > 1）
            "anomaly_p50_vs_normal_p95": (
                float(np.percentile(test_anomaly_scores, 50)) /
                max(float(np.percentile(test_normal_scores, 95)), 1e-10)
            ),
            # 正常 P99 vs 异常 P50（理想 < 1）
            "normal_p99_vs_anomaly_p50": (
                float(np.percentile(test_normal_scores, 99)) /
                max(float(np.percentile(test_anomaly_scores, 50)), 1e-10)
            ),
        },
        "score_range": {
            "train": {
                "min": float(np.min(train_scores)),
                "max": float(np.max(train_scores)),
            },
            "test": {
                "min": float(np.min(test_scores)),
                "max": float(np.max(test_scores)),
            },
        },
    }

    print(f"\n  [DIAGNOSIS] Separation metrics:")
    for k, v in stats["separation_metrics"].items():
        print(f"    {k}: {v:.4f}")
    print(f"    Train score range: [{stats['score_range']['train']['min']:.6f}, "
          f"{stats['score_range']['train']['max']:.6f}]")
    print(f"    Test score range:  [{stats['score_range']['test']['min']:.6f}, "
          f"{stats['score_range']['test']['max']:.6f}]")

    # ============ 6. 可视化 ============
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join(ROOT_PATH, "result", "diagnosis",
                                  dataset_name.replace(".csv", ""), timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # --- Figure 1: 误差分布直方图 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1-A: 训练集正常 vs 测试集正常
    ax = axes[0, 0]
    if len(train_normal_scores) > 0:
        ax.hist(train_normal_scores, bins=80, alpha=0.6, density=True,
                label=f"Train Normal (n={len(train_normal_scores):,})",
                color='steelblue', edgecolor='white', linewidth=0.5)
    if len(test_normal_scores) > 0:
        ax.hist(test_normal_scores, bins=80, alpha=0.6, density=True,
                label=f"Test Normal (n={len(test_normal_scores):,})",
                color='mediumseagreen', edgecolor='white', linewidth=0.5)
    ax.set_xlabel("Reconstruction Error (L1 max per point)")
    ax.set_ylabel("Density")
    ax.set_title("Normal: Train vs Test Distribution")
    ax.legend(fontsize=9)

    # 1-B: 测试集正常 vs 测试集异常
    ax = axes[0, 1]
    if len(test_normal_scores) > 0:
        ax.hist(test_normal_scores, bins=80, alpha=0.6, density=True,
                label=f"Normal (n={len(test_normal_scores):,})",
                color='mediumseagreen', edgecolor='white', linewidth=0.5)
    if len(test_anomaly_scores) > 0:
        ax.hist(test_anomaly_scores, bins=80, alpha=0.6, density=True,
                label=f"Anomaly (n={len(test_anomaly_scores):,})",
                color='coral', edgecolor='white', linewidth=0.5)
    ax.set_xlabel("Reconstruction Error (L1 max per point)")
    ax.set_ylabel("Density")
    ax.set_title("Test: Normal vs Anomaly Distribution")
    ax.legend(fontsize=9)

    # 1-C: 所有分布的 CDF
    ax = axes[1, 0]
    for data, label, color in [
        (train_normal_scores, "Train Normal", "steelblue"),
        (test_normal_scores, "Test Normal", "mediumseagreen"),
        (test_anomaly_scores, "Test Anomaly", "coral"),
    ]:
        if len(data) > 0:
            sorted_data = np.sort(data)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cdf, label=label, color=color, linewidth=2)
    # 标记常用百分位
    for p in [95, 99]:
        ax.axvline(x=np.percentile(test_normal_scores, p) if len(test_normal_scores) > 0 else 0,
                   color='mediumseagreen', linestyle='--', alpha=0.5)
        ax.text(np.percentile(test_normal_scores, p) if len(test_normal_scores) > 0 else 0,
                0.5, f" P{p}", fontsize=8, color='mediumseagreen')
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Cumulative Probability")
    ax.set_title("CDF: Distribution Separation")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)

    # 1-D: 箱线图
    ax = axes[1, 1]
    box_data = []
    box_labels = []
    for d, l in [
        (train_normal_scores, "Train\nNormal"),
        (test_normal_scores, "Test\nNormal"),
        (test_anomaly_scores, "Test\nAnomaly"),
    ]:
        if len(d) > 0:
            # 采样，防止 boxplot 过慢
            if len(d) > 10000:
                d = np.random.choice(d, 10000, replace=False)
            box_data.append(d)
            box_labels.append(l)
    if box_data:
        bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True,
                        showfliers=False, widths=0.5)
        colors = ['steelblue', 'mediumseagreen', 'coral']
        for patch, color in zip(bp['boxes'], colors[:len(box_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
    ax.set_ylabel("Reconstruction Error")
    ax.set_title("Score Distribution (Box Plot, no outliers)")
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f"Reconstruction Error Distribution: {dataset_name}\n"
                 f"Train/Test split: 80/20  |  "
                 f"Train normal P99: {np.percentile(train_normal_scores, 99):.6f}  |  "
                 f"Test anomaly P50: {np.percentile(test_anomaly_scores, 50):.6f}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    dist_path = os.path.join(output_dir, "recon_distribution.png")
    plt.savefig(dist_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\n  [PLOT] Distribution → {dist_path}")

    # --- Figure 2: 时间序列分数 + 真实标签 ---
    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=False)

    # 2-A: 训练集
    ax = axes[0]
    ax.plot(train_scores, color='steelblue', linewidth=0.6, alpha=0.8, label="Recon Error")
    # 标记训练异常点
    anomaly_idx = np.where(train_label == 1)[0]
    if len(anomaly_idx) > 0:
        ax.scatter(anomaly_idx, train_scores[anomaly_idx],
                   color='red', s=4, alpha=0.4, label=f"Anomaly ({len(anomaly_idx)})")
    # 阈值线（P99）
    train_p99 = np.percentile(train_normal_scores, 99) if len(train_normal_scores) > 0 else 0
    ax.axhline(y=train_p99, color='orange', linestyle='--', alpha=0.7,
               label=f"P99={train_p99:.6f}")
    ax.set_title(f"Training Set — Recon Error ({dataset_name})")
    ax.set_ylabel("Score")
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(0, len(train_scores))

    # 2-B: 测试集
    ax = axes[1]
    ax.plot(test_scores, color='mediumseagreen', linewidth=0.6, alpha=0.8, label="Recon Error")
    anomaly_idx = np.where(test_label == 1)[0]
    if len(anomaly_idx) > 0:
        ax.scatter(anomaly_idx, test_scores[anomaly_idx],
                   color='red', s=4, alpha=0.4, label=f"Anomaly ({len(anomaly_idx)})")
    test_p99 = np.percentile(test_normal_scores, 99) if len(test_normal_scores) > 0 else 0
    ax.axhline(y=test_p99, color='orange', linestyle='--', alpha=0.7,
               label=f"P99={test_p99:.6f}")
    ax.set_title(f"Test Set — Recon Error")
    ax.set_ylabel("Score")
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(0, len(test_scores))

    # 2-C: 测试集异常区域放大（前 2000 点）
    ax = axes[2]
    zoom_end = min(2000, len(test_scores))
    ax.plot(test_scores[:zoom_end], color='mediumseagreen', linewidth=0.8, alpha=0.8, label="Recon Error")
    anomaly_idx = np.where(test_label[:zoom_end] == 1)[0]
    if len(anomaly_idx) > 0:
        ax.scatter(anomaly_idx, test_scores[anomaly_idx],
                   color='red', s=10, alpha=0.6, label=f"Anomaly ({len(anomaly_idx)})")
    ax.axhline(y=test_p99, color='orange', linestyle='--', alpha=0.7,
               label=f"P99={test_p99:.6f}")
    ax.set_title(f"Test Set (first {zoom_end} points) — Zoomed")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Score")
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlim(0, zoom_end)

    plt.tight_layout()
    ts_path = os.path.join(output_dir, "score_ts.png")
    plt.savefig(ts_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  [PLOT] Time series → {ts_path}")

    # --- Figure 3: 阈值分析 ---
    fig, ax = plt.subplots(figsize=(12, 6))

    anomaly_ratios = [0.1, 0.5, 1.0, 2, 3, 5, 10, 15, 20, 25]
    train_percentiles = []
    test_normal_percentiles = []
    test_anomaly_percentiles = []
    for r in anomaly_ratios:
        train_percentiles.append(np.percentile(train_scores, 100 - r))
        if len(test_normal_scores) > 0:
            test_normal_percentiles.append(np.percentile(test_normal_scores, 100 - r))
        else:
            test_normal_percentiles.append(0)
        if len(test_anomaly_scores) > 0:
            test_anomaly_percentiles.append(np.percentile(test_anomaly_scores, 100 - r))
        else:
            test_anomaly_percentiles.append(0)

    ax.plot(anomaly_ratios, train_percentiles, 'o-', color='steelblue',
            linewidth=2, markersize=6, label="Train Set Score Threshold")
    ax.plot(anomaly_ratios, test_normal_percentiles, 's-', color='mediumseagreen',
            linewidth=2, markersize=6, label="Test Normal Score Threshold")
    if len(test_anomaly_scores) > 0:
        ax.plot(anomaly_ratios, test_anomaly_percentiles, '^-', color='coral',
                linewidth=2, markersize=6, label="Test Anomaly Score Threshold")
    # 标出当前 config 中使用的 ratio
    default_ratios = [0.5, 1.0, 2, 5, 10, 15]
    for r in default_ratios:
        ax.axvline(x=r, color='gray', linestyle=':', alpha=0.4)

    ax.set_xlabel("Anomaly Ratio (percentile = 100 - ratio)")
    ax.set_ylabel("Score Threshold")
    ax.set_title("Threshold Analysis: Score at Different Anomaly Ratios")
    ax.set_xscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    thresh_path = os.path.join(output_dir, "threshold_analysis.png")
    plt.savefig(thresh_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  [PLOT] Threshold analysis → {thresh_path}")

    # 7. 保存诊断报告
    stats["elapsed_seconds"] = round(time.time() - t_start, 1)
    stats["output_dir"] = output_dir
    stats["files"] = {
        "distribution": "recon_distribution.png",
        "time_series": "score_ts.png",
        "threshold": "threshold_analysis.png",
    }

    report_path = os.path.join(output_dir, "diagnostic_report.json")
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  [REPORT] Diagnostic report → {report_path}")

    # 8. 打印诊断结论
    print(f"\n{'='*60}")
    print(f"  DIAGNOSTIC CONCLUSION")
    print(f"{'='*60}")
    sep = stats["separation_metrics"]
    if sep["anomaly_normal_mean_ratio"] > 2.0:
        print(f"  ✓ Anomaly/Normal mean ratio = {sep['anomaly_normal_mean_ratio']:.2f}x")
        print(f"    → 异常平均分明显高于正常平均分，模型能有效分离")
    elif sep["anomaly_normal_mean_ratio"] > 1.5:
        print(f"  ◐ Anomaly/Normal mean ratio = {sep['anomaly_normal_mean_ratio']:.2f}x")
        print(f"    → 有一定分离度，但需要调优阈值")
    else:
        print(f"  ✗ Anomaly/Normal mean ratio = {sep['anomaly_normal_mean_ratio']:.2f}x")
        print(f"    → 分离度不足！重建误差无法区分异常")

    if sep["anomaly_p50_vs_normal_p95"] > 1.0:
        print(f"  ✓ Anomaly P50 > Normal P95 ({sep['anomaly_p50_vs_normal_p95']:.2f}x)")
        print(f"    → 大部分异常点的分数高于 95% 的正常点")
    elif sep["anomaly_p50_vs_normal_p95"] > 0.5:
        print(f"  ◐ Anomaly P50 / Normal P95 = {sep['anomaly_p50_vs_normal_p95']:.2f}")
        print(f"    → 部分异常点可被识别，但存在重叠区域")
    else:
        print(f"  ✗ Anomaly P50 << Normal P95 ({sep['anomaly_p50_vs_normal_p95']:.2f})")
        print(f"    → 异常和正常分数重叠严重，detect_label 难以生效")

    print(f"\n  [INFO] Done. Results saved to: {output_dir}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="LaGraph 重建误差分布诊断")
    parser.add_argument("--dataset", type=str, default="swat.csv",
                        help="数据集文件名 (default: swat.csv)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="训练轮次 (default: 10)")
    parser.add_argument("--n-gpus", type=int, default=1,
                        help="GPU 数量 (default: 1)")
    parser.add_argument("--fast", action="store_true",
                        help="快速模式 (3 epochs)")
    args = parser.parse_args()

    if args.fast:
        args.epochs = 3

    diagnose(
        dataset_name=args.dataset,
        train_epochs=args.epochs,
        n_gpus=args.n_gpus,
    )


if __name__ == "__main__":
    main()

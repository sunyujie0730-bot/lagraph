#!/usr/bin/env python3
"""
诊断脚本 — 类别重平衡实验（Step 2 of 验证实验）
===================================================

目的：
  检验 LaGraph 在损失函数中加入异常样本权重 / Focal Loss 后，
  raw_f@1% 是否获得提升。
  
  若 weight=10/50/100 带来 raw_f@1% 提升 > 0.10，确认根因 D（类别不均衡）为主因。

假设：
  anomaly_weight=对异常的 attention 提升 → 更高重建误差惩罚 → 异常分离度改善

用法：
  cd /home/professor3/Lagraph/LaGraph
  python scripts/diagnose_reweight_loss.py                          # 默认 swat.csv
  python scripts/diagnose_reweight_loss.py --dataset MSL.csv        # 指定数据集
  python scripts/diagnose_reweight_loss.py --fast                   # 快速验证 (3 epochs)

输出：
  result/diagnosis/<dataset>/reweight_<timestamp>/
    reweight_report.json           — 各 weight 策略的评估结果
    reweight_comparison.png        — 不同 weight 策略的分离度对比
    raw_f_comparison.png           — raw_f@1% 随 anomaly_weight 的变化
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from copy import deepcopy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ts_benchmark.baselines.self_impl.LaGraph.LaGraph import LaGraph
from ts_benchmark.baselines.self_impl.LaGraph.gcn_model import SparseGCN
from ts_benchmark.baselines.utils import anomaly_detection_data_provider, train_val_split
from ts_benchmark.common.constant import ROOT_PATH
from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource
from ts_benchmark.utils.data_processing import split_before
from ts_benchmark.evaluation.metrics import get_metrics

warnings.filterwarnings("ignore")

plt.rcParams.update({
    'figure.dpi': 120,
    'figure.figsize': (14, 8),
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


# ==================== Focal Loss 实现 ====================
class FocalLoss(torch.nn.Module):
    """
    Focal Loss for binary classification-style anomaly detection.
    γ=2, α=0.75 的标准配置。
    
    公式: FL(p_t) = -α_t * (1-p_t)^γ * log(p_t)
    其中 p_t = p if y=1 else 1-p
    """
    def __init__(self, gamma=2.0, alpha=0.75, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # inputs: raw scores (not probabilities)
        # targets: 0/1 labels
        
        # 将重建误差分数转换为 pseudo-probability [0,1] via sigmoid
        probs = torch.sigmoid(inputs)
        
        pt = torch.where(targets == 1, probs, 1 - probs)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Focal loss
        loss = -alpha_t * (1 - pt) ** self.gamma * torch.log(pt + 1e-8)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


def load_dataset(dataset_name: str):
    """加载数据集并返回 raw 数据 + 标签"""
    data_src = LocalAnomalyDetectDataSource()
    meta = data_src.dataset.metadata
    row = meta[meta["file_name"] == dataset_name]
    if len(row) == 0:
        raise FileNotFoundError(f"Dataset {dataset_name} not found")
    
    pool = DataPool().get_pool()
    data = pool.get_series(dataset_name)
    
    train_ratio = 0.8
    train_length = int(train_ratio * len(data))
    train, test = split_before(data, train_length)
    
    train_data = train.loc[:, train.columns != "label"]
    train_label = train.loc[:, ["label"]].values.flatten()
    test_data = test.loc[:, test.columns != "label"]
    test_label = test.loc[:, ["label"]].values.flatten()
    
    return train_data, train_label, test_data, test_label


def compute_separation(model, data_df: pd.DataFrame, labels: np.ndarray):
    """
    计算模型在给定数据上的分离度指标。
    返回 anomaly/normal 的分数分布统计和 KS 距离。
    """
    from scipy.stats import ks_2samp
    
    model.model.eval()
    scaled = pd.DataFrame(
        model.scaler.transform(data_df.values),
        columns=data_df.columns, index=data_df.index,
    )
    
    total_length = len(scaled)
    win_size = model.config.win_size
    eval_batch_size = min(model.config.batch_size, 64)
    
    loader = anomaly_detection_data_provider(
        scaled, batch_size=eval_batch_size,
        win_size=win_size, step=1, mode="test",
    )
    
    window_list = []
    with torch.no_grad():
        for i, (input_data, _) in enumerate(loader):
            input_data = input_data.float().to(model.device)
            scores = model._detect_forward(input_data)
            window_list.append(scores)
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
    
    if not window_list:
        return {"error": "no windows"}
    
    windows = np.concatenate(window_list, axis=0)
    scores = model._point_wise_aggregate(windows, win_size, total_length)
    
    min_len = min(len(scores), len(labels))
    scores = scores[:min_len]
    labels = labels[:min_len]
    
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    ks_stat, ks_pvalue = ks_2samp(normal_scores, anomaly_scores)
    
    return {
        "n_normal": int(len(normal_scores)),
        "n_anomaly": int(len(anomaly_scores)),
        "normal_mean": float(np.mean(normal_scores)),
        "normal_std": float(np.std(normal_scores)),
        "normal_p50": float(np.percentile(normal_scores, 50)),
        "normal_p95": float(np.percentile(normal_scores, 95)),
        "normal_p99": float(np.percentile(normal_scores, 99)),
        "anomaly_mean": float(np.mean(anomaly_scores)),
        "anomaly_std": float(np.std(anomaly_scores)),
        "anomaly_p50": float(np.percentile(anomaly_scores, 50)),
        "anomaly_p95": float(np.percentile(anomaly_scores, 95)),
        "anomaly_p99": float(np.percentile(anomaly_scores, 99)),
        "anomaly_normal_mean_ratio": (
            float(np.mean(anomaly_scores)) / max(float(np.mean(normal_scores)), 1e-10)
        ),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
    }


def evaluate_raw_f(model, test_data: pd.DataFrame, test_label: np.ndarray, ratio: float = 1.0):
    """
    计算指定异常率下的 raw_f@ratio。
    使用 anomaly_detect.py 的 adjust_predicts 逻辑进行时间容差调整。
    """
    preds_dict, _ = model.detect_label(test_data)
    
    if ratio not in preds_dict:
        print(f"  [WARN] ratio {ratio} not in preds_dict keys: {list(preds_dict.keys())}")
        return 0.0
    
    preds = preds_dict[ratio]
    
    min_len = min(len(preds), len(test_label))
    preds = preds[:min_len]
    gt = test_label[:min_len]
    
    # 计算 raw metrics
    tp = np.sum((preds == 1) & (gt == 1))
    fp = np.sum((preds == 1) & (gt == 0))
    fn = np.sum((preds == 0) & (gt == 1))
    
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f_score = 2 * precision * recall / max(precision + recall, 1e-10)
    
    return {
        "f_score": float(f_score),
        "precision": float(precision),
        "recall": float(recall),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def diagnose_reweight(
    dataset_name: str = "swat.csv",
    train_epochs: int = 20,
    n_gpus: int = 1,
    output_dir: str = None,
):
    """主诊断流程"""
    t_start = time.time()
    
    # 1. 加载数据
    train_data, train_label, test_data, test_label = load_dataset(dataset_name)
    print(f"\n  [DATA] {dataset_name}: train={len(train_data)}, test={len(test_data)}")
    print(f"  [DATA] Train anomaly ratio: {train_label.mean()*100:.2f}%")
    print(f"  [DATA] Test anomaly ratio:  {test_label.mean()*100:.2f}%")
    
    train_anomaly_ratio = train_label.mean()
    
    # 2. 定义要测试的权重策略
    strategies = {
        "baseline (w=1)": {"anomaly_weight": 1.0, "use_focal": False, "focal_gamma": 0},
        "w=10": {"anomaly_weight": 10.0, "use_focal": False, "focal_gamma": 0},
        "w=50": {"anomaly_weight": 50.0, "use_focal": False, "focal_gamma": 0},
        "w=100": {"anomaly_weight": 100.0, "use_focal": False, "focal_gamma": 0},
        "focal (γ=2, α=0.75)": {"anomaly_weight": 1.0, "use_focal": True, "focal_gamma": 2.0},
    }
    
    results = {}
    
    for strategy_name, strat_params in strategies.items():
        print(f"\n{'='*60}")
        print(f"  Strategy: {strategy_name}")
        print(f"  Params: {strat_params}")
        print(f"{'='*60}")
        
        # 训练模型
        model = LaGraph(
            num_epochs=train_epochs,
            n_gpus=n_gpus,
            dataset_name=dataset_name,
        )
        
        # 保存异常参数供自定义训练使用
        model._anomaly_weight = strat_params["anomaly_weight"]
        model._use_focal = strat_params["use_focal"]
        model._focal_gamma = strat_params.get("focal_gamma", 2.0)
        
        # 使用自定义训练（带重加权损失）
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        model.scaler.fit(train_data_value.values)
        model._train_raw = train_data_value.copy()
        
        train_scaled = pd.DataFrame(
            model.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
        )
        valid_scaled = pd.DataFrame(
            model.scaler.transform(valid_data.values),
            columns=valid_data.columns,
        )
        
        # 单 GPU 训练（避免 DDP 开销）
        if n_gpus > 1:
            n_gpus = 1
        model.ngpu = 1
        model.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 固定种子保证 baseline 可重现
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # 用自定义训练，修改损失函数中的权重
        model._single_gpu_train_reweight(
            train_scaled, valid_scaled,
            anomaly_weight=strat_params["anomaly_weight"],
            use_focal=strat_params["use_focal"],
            focal_gamma=strat_params.get("focal_gamma", 2.0),
        )
        
        if model.early_stopping is not None and model.early_stopping.check_point is not None:
            model._get_raw_model().load_state_dict(model.early_stopping.check_point)
        model.trained = True
        
        # 计算训练集重建分数
        print(f"\n  [INFO] Computing train scores for threshold...")
        model._train_anomaly_scores, _ = model.detect_score(model._train_raw)
        
        # 评估分离度
        print(f"\n  [EVAL] Computing separation on test set...")
        sep = compute_separation(model, test_data, test_label)
        results[strategy_name] = {"separation": sep}
        
        print(f"    Anomaly/Normal mean ratio: {sep['anomaly_normal_mean_ratio']:.4f}")
        print(f"    KS statistic: {sep['ks_statistic']:.4f} (p={sep['ks_pvalue']:.4e})")
        
        # 评估 raw_f@1%
        print(f"  [EVAL] Computing raw_f@1%...")
        raw_f = evaluate_raw_f(model, test_data, test_label, ratio=1.0)
        results[strategy_name]["raw_f_1pct"] = raw_f
        print(f"    raw_f@1%: {raw_f['f_score']:.4f}")
        
        # 释放显存
        model._destroy_model_and_clean_cuda()
        del model
    
    # ============ 3. 可视化结果 ============
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = os.path.join(ROOT_PATH, "result", "diagnosis",
                                  dataset_name.replace(".csv", ""),
                                  f"reweight_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 3-A: 分离度对比
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    strategy_names = list(results.keys())
    
    # 异常/正常均值比
    ax = axes[0]
    ratios = [results[s]["separation"]["anomaly_normal_mean_ratio"] for s in strategy_names]
    colors = ['steelblue' if 'baseline' in s else 'coral' for s in strategy_names]
    bars = ax.bar(range(len(strategy_names)), ratios, color=colors, alpha=0.7, width=0.6)
    ax.axhline(y=2.0, color='green', linestyle='--', alpha=0.5, label='Good (2x)')
    ax.axhline(y=1.5, color='orange', linestyle='--', alpha=0.5, label='Fair (1.5x)')
    ax.set_xticks(range(len(strategy_names)))
    ax.set_xticklabels(strategy_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Anomaly/Normal Mean Ratio")
    ax.set_title("Separation Ratio by Strategy")
    ax.legend(fontsize=8)
    
    # KS 统计量
    ax = axes[1]
    ks_stats = [results[s]["separation"]["ks_statistic"] for s in strategy_names]
    bars = ax.bar(range(len(strategy_names)), ks_stats, color=colors, alpha=0.7, width=0.6)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Moderate')
    ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Strong')
    ax.set_xticks(range(len(strategy_names)))
    ax.set_xticklabels(strategy_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("KS Statistic")
    ax.set_title("Distribution Separation (KS Test)")
    ax.legend(fontsize=8)
    
    # raw_f@1%
    ax = axes[2]
    raw_fs = [results[s]["raw_f_1pct"]["f_score"] for s in strategy_names]
    bars = ax.bar(range(len(strategy_names)), raw_fs, color=colors, alpha=0.7, width=0.6)
    ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.5, label='Good (0.20)')
    ax.axhline(y=0.1, color='orange', linestyle='--', alpha=0.5, label='Fair (0.10)')
    for i, v in enumerate(raw_fs):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=8)
    ax.set_xticks(range(len(strategy_names)))
    ax.set_xticklabels(strategy_names, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("raw_f@1%")
    ax.set_title("Detection Performance at 1% Anomaly Ratio")
    ax.legend(fontsize=8)
    
    fig.suptitle(f"Reweight Experiment: {dataset_name}\n"
                 f"Train anomaly ratio: {train_anomaly_ratio*100:.2f}%",
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    comparison_path = os.path.join(output_dir, "reweight_comparison.png")
    plt.savefig(comparison_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\n  [PLOT] Comparison → {comparison_path}")
    
    # 3-B: raw_f@1% 随权重变化曲线
    fig, ax = plt.subplots(figsize=(10, 6))
    weights = [1, 10, 50, 100]
    weight_results = [
        results["baseline (w=1)"]["raw_f_1pct"]["f_score"],
        results["w=10"]["raw_f_1pct"]["f_score"],
        results["w=50"]["raw_f_1pct"]["f_score"],
        results["w=100"]["raw_f_1pct"]["f_score"],
    ]
    ax.plot(weights, weight_results, 'o-', color='coral', linewidth=2, markersize=8)
    ax.axhline(y=results["focal (γ=2, α=0.75)"]["raw_f_1pct"]["f_score"],
               color='purple', linestyle='--', alpha=0.7,
               label=f"Focal Loss: {results['focal (γ=2, α=0.75)']['raw_f_1pct']['f_score']:.4f}")
    ax.set_xscale('log')
    ax.set_xlabel("Anomaly Weight")
    ax.set_ylabel("raw_f@1%")
    ax.set_title("raw_f@1% vs Anomaly Weight (log scale)")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    for w, v in zip(weights, weight_results):
        ax.annotate(f"{v:.4f}", (w, v), textcoords="offset points",
                    xytext=(0, 10), ha='center', fontsize=9)
    
    raw_f_path = os.path.join(output_dir, "raw_f_comparison.png")
    plt.savefig(raw_f_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  [PLOT] Raw F comparison → {raw_f_path}")
    
    # 4. 保存报告
    stats = {
        "dataset": dataset_name,
        "train_epochs": train_epochs,
        "train_anomaly_ratio": float(train_anomaly_ratio),
        "test_anomaly_ratio": float(test_label.mean()),
        "strategies": {},
        "conclusion": {},
        "elapsed_seconds": round(time.time() - t_start, 1),
        "output_dir": output_dir,
    }
    
    for s in strategy_names:
        stats["strategies"][s] = results[s]
    
    # 结论
    baseline_f = results["baseline (w=1)"]["raw_f_1pct"]["f_score"]
    best_weight_f = max([results[s]["raw_f_1pct"]["f_score"] for s in strategy_names if "focal" not in s])
    focal_f = results["focal (γ=2, α=0.75)"]["raw_f_1pct"]["f_score"]
    
    stats["conclusion"] = {
        "baseline_raw_f_1pct": baseline_f,
        "best_weighted_raw_f_1pct": best_weight_f,
        "improvement_from_weighting": best_weight_f - baseline_f,
        "focal_raw_f_1pct": focal_f,
        "improvement_from_focal": focal_f - baseline_f,
        "root_cause_D_confirmed": (best_weight_f - baseline_f > 0.10) or (focal_f - baseline_f > 0.10),
        "verdict": (
            "✅ 根因 D（类别不均衡）确认"
            if (best_weight_f - baseline_f > 0.10)
            else (
                "◐ 类别不均衡有影响但非主因"
                if (best_weight_f - baseline_f > 0.05)
                else "❌ 类别不均衡不是主要瓶颈"
            )
        ),
    }
    
    report_path = os.path.join(output_dir, "reweight_report.json")
    with open(report_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"\n  [REPORT] → {report_path}")
    
    # 打印结论
    print(f"\n{'='*60}")
    print(f"  CONCLUSION — Root Cause D Test")
    print(f"{'='*60}")
    c = stats["conclusion"]
    print(f"  Baseline raw_f@1%:         {c['baseline_raw_f_1pct']:.4f}")
    print(f"  Best weighted raw_f@1%:    {c['best_weighted_raw_f_1pct']:.4f}")
    print(f"  Improvement (weighting):   {c['improvement_from_weighting']:+.4f}")
    print(f"  Focal Loss raw_f@1%:       {c['focal_raw_f_1pct']:.4f}")
    print(f"  Improvement (focal):        {c['improvement_from_focal']:+.4f}")
    print(f"  Verdict:                    {c['verdict']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="LaGraph 类别重平衡实验诊断 (Step 2 of 5)")
    parser.add_argument("--dataset", type=str, default="swat.csv",
                        help="数据集文件名 (default: swat.csv)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="训练轮次 (default: 20)")
    parser.add_argument("--n-gpus", type=int, default=1,
                        help="GPU 数量 (default: 1)")
    parser.add_argument("--fast", action="store_true",
                        help="快速模式 (3 epochs)")
    args = parser.parse_args()
    
    if args.fast:
        args.epochs = 3
    
    diagnose_reweight(
        dataset_name=args.dataset,
        train_epochs=args.epochs,
        n_gpus=args.n_gpus,
    )


if __name__ == "__main__":
    main()

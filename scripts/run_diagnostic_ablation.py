# -*- coding: utf-8 -*-
"""
LaGraph 诊断验证实验脚本
========================
优先级最高的验证实验：

[实验1] 重建误差分布可视化（正常vs异常直方图）
[实验2] 异常分数方差检查（确认分布塌陷）
[实验3] 单窗口对照实验（隔离多尺度融合影响）

用法:
  # 实验1+2: 标准诊断（SWAT）
  python scripts/run_diagnostic_ablation.py --epochs 100 --datasets swat.csv --n-gpus 1

  # 实验3: 单窗口模式（multi_scale_win_sizes=[100]）
  python scripts/run_diagnostic_ablation.py --epochs 100 --datasets swat.csv --n-gpus 1 --single-window

  # 无图模块基线
  python scripts/run_diagnostic_ablation.py --epochs 100 --datasets swat.csv --n-gpus 1 --no-graph

  # MSL 验证
  python scripts/run_diagnostic_ablation.py --epochs 100 --datasets MSL.csv --n-gpus 1

输出:
  result/diagnostic/{dataset}_{timestamp}/ 目录下的可视化结果和诊断日志
"""

import argparse
import json
import os
import sys
import time
import warnings
import pickle
import gc
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # 无GUI后端
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ts_benchmark.common.constant import ROOT_PATH, CONFIG_PATH
from ts_benchmark.data.data_source import LocalAnomalyDetectDataSource
from ts_benchmark.data.data_pool import DataPool
from ts_benchmark.baselines.self_impl.LaGraph.LaGraph import LaGraph
from ts_benchmark.baselines.utils import train_val_split

warnings.filterwarnings("ignore")


class DiagnosticRunner:
    """
    诊断运行器：
    1. 训练 LaGraph
    2. 收集训练集和测试集的重建误差
    3. 可视化正常vs异常误差分布
    4. 计算分数方差统计
    5. 输出诊断报告
    """

    def __init__(self, args):
        self.args = args
        self.dataset_name = args.datasets[0] if args.datasets else "swat.csv"
        self.output_dir = os.path.join(
            ROOT_PATH, "result", "diagnostic",
            f"{self.dataset_name.replace('.csv', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def load_dataset(self):
        """加载指定的数据集"""
        data_src = LocalAnomalyDetectDataSource()
        metadata = data_src.dataset.metadata
        row = metadata[metadata["file_name"] == self.dataset_name]
        if len(row) == 0:
            raise ValueError(f"Dataset {self.dataset_name} not found in metadata")

        print(f"\n{'='*60}")
        print(f"  加载数据集: {self.dataset_name}")
        print(f"{'='*60}")

        # 使用 DataPool 获取数据
        data_pool = DataPool()
        data_pool.load_data(
            data_source=data_src,
            data_set_name=["large_detect"],
            data_name_list=[self.dataset_name],
        )

        # 获取训练/测试数据
        data_config = {
            "data_set_name": ["large_detect"],
            "data_name_list": [self.dataset_name],
        }
        
        # 从 DataPool 获取数据项
        data_items = data_pool.get_data(data_config)
        print(f"  Data items: {len(data_items)}")

        train_data = None
        test_data = None
        test_labels = None

        for item in data_items:
            if item["data_type"] == "train":
                train_data = item["data"]
            elif item["data_type"] == "test":
                test_data = item["data"]
                test_labels = item.get("label", None)

        if train_data is None or test_data is None:
            # 回退：用原始数据文件加载
            print(f"  [WARN] DataPool didn't provide train/test split, loading raw...")
            data_dir = os.path.join(ROOT_PATH, "dataset", "anomaly_detect", "data")
            file_path = os.path.join(data_dir, self.dataset_name)
            if not os.path.exists(file_path):
                # 尝试搜索
                for root, dirs, files in os.walk(os.path.join(ROOT_PATH, "dataset")):
                    if self.dataset_name in files:
                        file_path = os.path.join(root, self.dataset_name)
                        break
            raw_data = pd.read_csv(file_path, index_col=0) if file_path else None
            if raw_data is None:
                raise FileNotFoundError(f"Cannot find {self.dataset_name} in dataset directory")
            
            # 简单分割
            n_total = len(raw_data)
            n_train = int(n_total * 0.8)
            train_data = raw_data.iloc[:n_train]
            test_data = raw_data.iloc[n_train:]

        print(f"  Train shape: {train_data.shape}")
        print(f"  Test shape:  {test_data.shape}")

        # 如果有标签，获取测试集异常标签
        if test_labels is not None and isinstance(test_labels, pd.DataFrame):
            test_labels = test_labels.values.flatten()
        elif test_labels is not None and isinstance(test_labels, np.ndarray):
            test_labels = test_labels.flatten()
        else:
            # 尝试从数据文件名推断异常区域（某些数据集最后一列是标签）
            if 'label' in train_data.columns:
                # 训练集含有标签列，需要排除
                label_col = 'label'
                train_data = train_data.drop(columns=[label_col])
            
            if 'label' in test_data.columns:
                # 测试集标签列
                test_labels = test_data['label'].values
                test_data = test_data.drop(columns=['label'])
            else:
                # 为测试数据生成随机标签（仅用于分布可视化）
                print(f"  [WARN] No test labels found, using sample-based labeling")
                test_labels = None

        if test_labels is not None:
            n_anomaly = np.sum(test_labels)
            n_normal = len(test_labels) - n_anomaly
            print(f"  Test labels: {n_anomaly} anomalies / {n_normal} normal ({n_anomaly/len(test_labels)*100:.1f}%)")

        return train_data, test_data, test_labels

    def run_training(self, train_data):
        """训练 LaGraph 模型"""
        print(f"\n{'='*60}")
        print(f"  训练 LaGraph v10")
        print(f"{'='*60}")

        # ★ 超参分析: batch_size
        #   时间序列异常检测中，win_size=100 的滑动窗口高度重叠，
        #   大 batch (2048) 产生冗余梯度，有效 batch 内信息量不到 50%。
        #   应使用 256：收敛更快、梯度更稳定、对异常更敏感。
        model_kwargs = {
            "num_epochs": self.args.epochs,
            "n_gpus": self.args.n_gpus,
            "batch_size": 256,
            "lr": 1e-4,
            "warmup_epochs": 10,  # ★ 从 5→10：VQ Bottleneck 需要足够预热周期
            "d_model": 128,
            "e_layers": 2,
            "n_heads": 4,
            "dataset_name": self.dataset_name.replace('.csv', ''),
        }

        # 单窗口模式：隔离多尺度融合影响
        if self.args.single_window:
            model_kwargs["multi_scale_win_sizes"] = [self.args.win_size or 100]
            model_kwargs["multi_scale_weights"] = [1.0]
            print(f"  [实验3] 单窗口模式: win_size={self.args.win_size or 100}")

        # 无图模块基线
        if self.args.no_graph:
            model_kwargs["use_graph"] = False
            print(f"  [实验2b] 无图模块基线模式")

        model = LaGraph(**model_kwargs)

        # 开始训练（detect_fit内部会split train/val）
        t0 = time.time()
        model.detect_fit(train_data, train_data.iloc[:100])  # test_data 参数在 detect_fit 实际未使用
        train_time = time.time() - t0
        print(f"\n  训练完成，耗时: {train_time:.1f}s")

        return model

    def collect_scores_and_visualize(self, model, train_data, test_data, test_labels):
        """收集重建误差，可视化分布，计算统计"""
        print(f"\n{'='*60}")
        print(f"  收集异常分数 + 分布分析")
        print(f"{'='*60}")

        # 1. 获取训练集分数
        print(f"\n  [1/4] 计算训练集重建误差...")
        train_scores, _ = model.detect_score(train_data)
        train_scores = train_scores.flatten()

        # 2. 获取测试集分数
        print(f"  [2/4] 计算测试集重建误差...")
        test_scores, _ = model.detect_score(test_data)
        test_scores = test_scores.flatten()

        # 3. 如果有标签，分离正常/异常分数
        normal_scores = None
        anomaly_scores = None
        if test_labels is not None:
            test_labels = test_labels.flatten()
            # 对齐长度（可能出现尾部截断）
            min_len = min(len(test_scores), len(test_labels))
            test_scores_aligned = test_scores[:min_len]
            test_labels_aligned = test_labels[:min_len]
            
            normal_scores = test_scores_aligned[test_labels_aligned == 0]
            anomaly_scores = test_scores_aligned[test_labels_aligned == 1]
            
            print(f"  Normal scores:   n={len(normal_scores)}, "
                  f"mean={normal_scores.mean():.6f}, std={normal_scores.std():.6f}")
            print(f"  Anomaly scores:  n={len(anomaly_scores)}, "
                  f"mean={anomaly_scores.mean():.6f}, std={anomaly_scores.std():.6f}")
        else:
            # 用训练集百分位数模拟标签
            print(f"  [WARN] No labels available; using percentile-based pseudo-labels")
            # 取训练集P95以上作为"伪异常"
            threshold = np.percentile(train_scores, 95)
            pseudo_anomaly = test_scores > threshold
            normal_scores = test_scores[~pseudo_anomaly]
            anomaly_scores = test_scores[pseudo_anomaly]
            test_labels_pseudo = pseudo_anomaly.astype(int)
            test_labels = test_labels_pseudo

        # 4. 计算分布统计
        normal_mean = normal_scores.mean() if normal_scores is not None else 0
        normal_std = normal_scores.std() if normal_scores is not None else 0
        anomaly_mean = anomaly_scores.mean() if anomaly_scores is not None else 0
        anomaly_std = anomaly_scores.std() if anomaly_scores is not None else 0

        # 分布重叠指标
        # Kolmogorov-Smirnov 距离（归一化差异）
        if normal_scores is not None and anomaly_scores is not None:
            ks_stat = self._compute_ks_stat(normal_scores, anomaly_scores)
            overlap_ratio = self._compute_overlap_ratio(normal_scores, anomaly_scores)
            # Bhattacharyya 距离
            bhatt_dist = self._compute_bhattacharyya(normal_scores, anomaly_scores)
        else:
            ks_stat = 0.0
            overlap_ratio = 1.0
            bhatt_dist = 0.0

        # 5. 分数分布塌陷检查
        all_scores = test_scores
        score_variance = np.var(all_scores)
        score_range = all_scores.max() - all_scores.min()
        score_cv = np.std(all_scores) / (np.mean(all_scores) + 1e-10)  # 变异系数
        score_skew = pd.Series(all_scores).skew()
        score_kurtosis = pd.Series(all_scores).kurtosis()

        # 分数分布的百分位分析
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        perc_values = np.percentile(all_scores, percentiles)

        print(f"\n  [3/4] 分数分布统计:")
        print(f"  方差:     {score_variance:.8f}")
        print(f"  范围:     [{all_scores.min():.6f}, {all_scores.max():.6f}]") 
        print(f"  变异系数: {score_cv:.4f}")
        print(f"  偏度:     {score_skew:.4f}")
        print(f"  峰度:     {score_kurtosis:.4f}")
        print(f"  百分位分布:")
        for p, v in zip(percentiles, perc_values):
            print(f"    P{p:2d}: {v:.6f}")
        print(f"  正常 vs 异常 KS统计量: {ks_stat:.4f}")
        print(f"  分布重叠率:         {overlap_ratio:.2%}")
        print(f"  Bhattacharyya距离:  {bhatt_dist:.4f}")

        # 6. 可视化
        print(f"\n  [4/4] 生成可视化...")
        self._plot_distributions(
            train_scores, normal_scores, anomaly_scores,
            all_scores, perc_values, percentiles,
            ks_stat, overlap_ratio, score_variance,
        )

        # 7. 生成诊断报告
        self._generate_report(
            train_scores, test_scores, test_labels,
            normal_scores, anomaly_scores,
            ks_stat, overlap_ratio, bhatt_dist,
            score_variance, score_cv, score_range,
        )

    def _compute_ks_stat(self, normal, anomaly):
        """计算 Kolmogorov-Smirnov 统计量"""
        # 简易 KS 统计：两分布累积分布函数的最大差异
        combined = np.concatenate([normal, anomaly])
        combined.sort()
        n1 = len(normal)
        n2 = len(anomaly)
        
        ecdf1 = np.searchsorted(normal, combined, side='right') / n1
        ecdf2 = np.searchsorted(anomaly, combined, side='right') / n2
        return np.max(np.abs(ecdf1 - ecdf2))

    def _compute_overlap_ratio(self, normal, anomaly):
        """计算分布重叠率"""
        # 使用直方图估计重叠面积
        all_vals = np.concatenate([normal, anomaly])
        bins = 100
        hist_range = (np.percentile(all_vals, 1), np.percentile(all_vals, 99))
        if hist_range[1] <= hist_range[0]:
            return 1.0
        
        hist_n, edges = np.histogram(normal, bins=bins, range=hist_range, density=True)
        hist_a, _ = np.histogram(anomaly, bins=bins, range=hist_range, density=True)
        
        overlap = np.sum(np.minimum(hist_n, hist_a)) * (edges[1] - edges[0])
        return overlap

    def _compute_bhattacharyya(self, normal, anomaly):
        """计算 Bhattacharyya 距离"""
        # 使用高斯假设的 Bhattacharyya 距离简化计算
        mean_n = np.mean(normal)
        mean_a = np.mean(anomaly)
        var_n = np.var(normal) + 1e-10
        var_a = np.var(anomaly) + 1e-10
        avg_var = (var_n + var_a) / 2
        
        db = 0.25 * np.log(0.25 * (var_n / var_a + var_a / var_n + 2))
        db += 0.25 * ((mean_n - mean_a) ** 2) / (var_n + var_a)
        return db

    def _plot_distributions(self, train_scores, normal_scores, anomaly_scores,
                           all_scores, perc_values, percentiles,
                           ks_stat, overlap_ratio, variance):
        """生成可视化图"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'LaGraph v10 诊断: {self.dataset_name}', fontsize=16, fontweight='bold')

        # 图1: 正常vs异常分数直方图
        ax = axes[0, 0]
        if normal_scores is not None and anomaly_scores is not None:
            bins = 80
            hist_range = (np.percentile(all_scores, 1), np.percentile(all_scores, 99))
            ax.hist(normal_scores, bins=bins, range=hist_range, alpha=0.6, 
                    label=f'Normal (n={len(normal_scores)})', color='blue', density=True)
            ax.hist(anomaly_scores, bins=bins, range=hist_range, alpha=0.6,
                    label=f'Anomaly (n={len(anomaly_scores)})', color='red', density=True)
            ax.set_title(f'Reconstruction Error Distribution\nKS={ks_stat:.4f} Overlap={overlap_ratio:.1%}')
        else:
            ax.hist(all_scores, bins=80, alpha=0.7, color='gray')
            ax.set_title('Test Scores Distribution (no labels)')
        ax.set_xlabel('Reconstruction Error')
        ax.set_ylabel('Density')
        ax.legend()

        # 图2: 训练集 vs 测试集分布对比
        ax = axes[0, 1]
        ax.hist(train_scores, bins=80, alpha=0.6, label=f'Train (n={len(train_scores)})', 
                color='green', density=True)
        ax.hist(all_scores, bins=80, alpha=0.6, label=f'Test (n={len(all_scores)})', 
                color='orange', density=True)
        ax.set_title('Train vs Test Distribution Overlap')
        ax.set_xlabel('Reconstruction Error')
        ax.set_ylabel('Density')
        ax.legend()

        # 图3: 分数序列 + 真实标签
        ax = axes[0, 2]
        ax.plot(all_scores, color='blue', alpha=0.7, linewidth=0.5, label='Score')
        if normal_scores is not None and anomaly_scores is not None and len(all_scores) == len(
                np.concatenate([np.zeros(len(normal_scores)), np.ones(len(anomaly_scores))]) if False else range(len(all_scores))):
            pass
        ax.axhline(y=np.percentile(all_scores, 95), color='red', linestyle='--', alpha=0.5, label='P95 threshold')
        ax.axhline(y=np.percentile(all_scores, 99), color='darkred', linestyle='--', alpha=0.5, label='P99 threshold')
        ax.set_title('Score Sequence')
        ax.set_xlabel('Time Point')
        ax.set_ylabel('Anomaly Score')
        ax.legend()

        # 图4: 百分位分布柱状图
        ax = axes[1, 0]
        bar_colors = ['green' if p <= 50 else 'yellow' if p <= 90 else 'red' for p in percentiles]
        bars = ax.bar(range(len(percentiles)), perc_values, color=bar_colors, alpha=0.7)
        ax.set_xticks(range(len(percentiles)))
        ax.set_xticklabels([f'P{p}' for p in percentiles])
        ax.set_title(f'Percentile Distribution\nVariance={variance:.8f}')
        ax.set_xlabel('Percentile')
        ax.set_ylabel('Score Value')
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, perc_values)):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{val:.4f}', ha='center', va='bottom', fontsize=7, rotation=45)

        # 图5: Q-Q 图 vs 正态分布（检查分布塌陷）
        ax = axes[1, 1]
        from scipy import stats
        stats.probplot(all_scores, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot vs Normal Distribution\n(Linearity → Gaussian-like collapse)')

        # 图6: 累积分布函数对比
        ax = axes[1, 2]
        if normal_scores is not None and anomaly_scores is not None:
            # 计算CDF
            sorted_n = np.sort(normal_scores)
            sorted_a = np.sort(anomaly_scores)
            cdf_n = np.arange(1, len(sorted_n)+1) / len(sorted_n)
            cdf_a = np.arange(1, len(sorted_a)+1) / len(sorted_a)
            ax.plot(sorted_n, cdf_n, label='Normal CDF', color='blue')
            ax.plot(sorted_a, cdf_a, label='Anomaly CDF', color='red')
            # 标注KS统计量位置
            ax.set_title(f'CDF Comparison (KS={ks_stat:.4f})')
        else:
            sorted_s = np.sort(all_scores)
            cdf_s = np.arange(1, len(sorted_s)+1) / len(sorted_s)
            ax.plot(sorted_s, cdf_s, color='gray')
            ax.set_title('Test CDF')
        ax.set_xlabel('Score')
        ax.set_ylabel('Cumulative Probability')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        
        # 保存
        save_path = os.path.join(self.output_dir, f'{self.dataset_name.replace(".csv","")}_diagnostic.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  可视化保存至: {save_path}")

    def _generate_report(self, train_scores, test_scores, test_labels,
                         normal_scores, anomaly_scores,
                         ks_stat, overlap_ratio, bhatt_dist,
                         score_variance, score_cv, score_range):
        """生成诊断报告"""
        # 点级指标计算
        if test_labels is not None:
            min_len = min(len(test_scores), len(test_labels))
            y_true = test_labels[:min_len].astype(int)
            y_score = test_scores[:min_len]
            
            # 在最佳阈值下的 raw_f
            best_f1 = 0
            best_thresh = 0
            for p in np.linspace(0.5, 99.5, 200):
                thresh = np.percentile(train_scores, p)
                y_pred = (y_score > thresh).astype(int)
                tp = np.sum((y_pred == 1) & (y_true == 1))
                fp = np.sum((y_pred == 1) & (y_true == 0))
                fn = np.sum((y_pred == 0) & (y_true == 1))
                precision = tp / (tp + fp + 1e-10)
                recall = tp / (tp + fn + 1e-10)
                f1 = 2 * precision * recall / (precision + recall + 1e-10)
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh
        else:
            best_f1 = 0
            best_thresh = 0

        # 分数塌陷诊断
        p99_p50_ratio = np.percentile(test_scores, 99) / (np.percentile(test_scores, 50) + 1e-10)
        p95_p50_ratio = np.percentile(test_scores, 95) / (np.percentile(test_scores, 50) + 1e-10)

        collapse_flag = "YES" if p99_p50_ratio < 2.0 else "NO"
        overlap_flag = "YES" if overlap_ratio > 0.30 else "NO"
        discrimination_flag = "GOOD" if ks_stat > 0.5 else ("POOR" if ks_stat > 0.3 else "CRITICAL")

        report = f"""
{'='*60}
  LaGraph v10 诊断验证报告
  数据集: {self.dataset_name}
  时间:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

【实验1】重建误差分布分析
{'-'*50}
  正常样本:
    数量:        {len(normal_scores) if normal_scores is not None else 'N/A'}
    均值:        {normal_scores.mean():.6f if normal_scores is not None else 'N/A'}
    标准差:      {normal_scores.std():.6f if normal_scores is not None else 'N/A'}
  
  异常样本:
    数量:        {len(anomaly_scores) if anomaly_scores is not None else 'N/A'}
    均值:        {anomaly_scores.mean():.6f if anomaly_scores is not None else 'N/A'}
    标准差:      {anomaly_scores.std():.6f if anomaly_scores is not None else 'N/A'}
  
  分布差异:
    KS 统计量:      {ks_stat:.4f}  ({'≥0.5=可区分' if ks_stat >= 0.5 else '<0.5=不可区分'})
    重叠率:         {overlap_ratio:.2%}  ({'≥30%=严重重叠' if overlap_ratio >= 0.3 else '<30%=可接受'})
    Bhattacharyya:  {bhatt_dist:.4f}  ({'≥0.5=可区分' if bhatt_dist >= 0.5 else '<0.5=高度重叠'})

  判定: {'❌ 分布高度重叠 - 重建范式失效' if overlap_ratio > 0.30 else '✅ 分布可区分'}

【实验2】异常分数分布塌陷检查
{'-'*50}
  方差:           {score_variance:.8f}
  变异系数(CV):   {score_cv:.4f}  ({'<0.1=严重塌陷' if score_cv < 0.1 else '>=0.1=正常'})
  范围:           [{test_scores.min():.6f}, {test_scores.max():.6f}]
  P99/P50 比值:   {p99_p50_ratio:.4f}  ({'<2.0=塌陷' if p99_p50_ratio < 2.0 else '>=2.0=正常'})
  P95/P50 比值:   {p95_p50_ratio:.4f}
  偏度:           {score_skew:.4f}
  峰度:           {score_kurtosis:.4f}

  分数塌陷判定: {collapse_flag}
  分布可判别性: {discrimination_flag}

【验证结论】
{'-'*50}
  重建误差分布重叠: {overlap_flag}
  分数分布塌陷:     {collapse_flag}
  在最佳阈值下的点级 F1: {best_f1:.4f}
  
  综合结论: {'模型需要转向对比/判别范式' if overlap_flag == 'YES' and collapse_flag == 'YES' else '模型有改进空间'}

【保存路径】
  报告:  {self.output_dir}/
  图片:  {os.path.join(self.output_dir, f"{self.dataset_name.replace('.csv','')}_diagnostic.png")}
{'='*60}
"""
        print(report)

        # 保存报告
        report_path = os.path.join(self.output_dir, f'{self.dataset_name.replace(".csv","")}_diagnostic_report.txt')
        with open(report_path, 'w') as f:
            f.write(report + "\n\n")
            # 保存分数原始数据
            np.savetxt(os.path.join(self.output_dir, 'train_scores.csv'), train_scores, delimiter=',')
            np.savetxt(os.path.join(self.output_dir, 'test_scores.csv'), test_scores, delimiter=',')
        print(f"  报告保存至: {report_path}")

    def run(self):
        """运行完整的诊断流程"""
        print(f"\n{'='*60}")
        print(f"  开始诊断验证实验")
        print(f"{'='*60}")
        print(f"  输出目录: {self.output_dir}")

        # 1. 加载数据
        train_data, test_data, test_labels = self.load_dataset()

        # 2. 训练模型
        model = self.run_training(train_data)

        # 3. 收集分数并诊断
        self.collect_scores_and_visualize(model, train_data, test_data, test_labels)

        print(f"\n{'='*60}")
        print(f"  诊断完成!")
        print(f"{'='*60}")
        print(f"  请查看 {self.output_dir}/ 下的结果")
        print()


def main():
    parser = argparse.ArgumentParser(description="LaGraph 诊断验证实验")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮次")
    parser.add_argument("--datasets", type=str, nargs="+", default=["swat.csv"],
                       help="数据集列表 (default: swat.csv)")
    parser.add_argument("--n-gpus", type=int, default=1, help="GPU 数量")
    parser.add_argument("--single-window", action="store_true",
                       help="实验3: 单窗口模式 (隔离多尺度融合)")
    parser.add_argument("--no-graph", action="store_true",
                       help="实验2b: 无图模块基线")
    parser.add_argument("--win-size", type=int, default=100,
                       help="单窗口模式的窗口大小 (default: 100)")
    args = parser.parse_args()

    runner = DiagnosticRunner(args)
    runner.run()


if __name__ == "__main__":
    main()

"""
LaGraph v11.4 SparseLaGraph — P0 五项修复合并版
=================================================

P0 修改（2026-05-20）:

1. ★ 术语诚实化：全局替换过时术语 → locality/adaptive_graph/refine/anomaly_boundary

2. ★ 推理阈值策略重构：POT 阈值估计 + 验证集自适应校准（根因3）

3. ★ 增加 AUC-ROC/AUPR 硬指标：评估报告系统（根因2）

4. ★ 多次运行与统计显著性：5-seed 实验框架（根因2）

5. ★ 中间结果可视化接口：特征/梯度/图结构可视化（根因5）

v10-v11.3 历史:
  v10: SparseLaGraph 重构（移除 FreqTower1D/Prototype/Contrastive/PredictionHead）
  v11: VQ Bottleneck + Dynamic Scale Selection
  v11.2: Ratio-Adaptive 门控 + 动态尺度选择
  v11.3: 术语修正（proximity/locality 统一）
  v11.4: P0 五项修复合并（术语、阈值、指标、多seed、可视化）
"""

import copy
import json
import os
import re
import socket
import subprocess
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats as scipy_stats
from torch import optim

from ts_benchmark.baselines.self_impl.LaGraph.gcn_model import SparseGCN
from ts_benchmark.baselines.utils import anomaly_detection_data_provider
from ts_benchmark.baselines.utils import train_val_split


# ======================== 默认超参数配置（v11.4）=======================
DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "win_size": 100,
    "patch_size": 16,
    "lr": 0.0001,
    "dropout": 0.25,
    "n_heads": 4,
    "e_layers": 2,
    "d_model": 128,
    "num_epochs": 100,
    "batch_size": 256,
    "eval_batch_size": 64,
    "enable_visualization_hooks": False,
    "patience": 15,
    "use_latest_checkpoint": False,
    "topk": 5,
    "anomaly_ratio": [0.5, 1.0, 2, 5, 10, 15],
    # --- 自适应图参数 ---
    "sparse_topk": None,
    "lambda_locality_l1": 0.001,   # 稀疏 L1 正则化系数
    "warmup_epochs": 10,
    # --- 多尺度异常评分参数 ---
    "multi_scale_win_sizes": [],
    "multi_scale_weights": [],
    # --- v11 VQ Bottleneck 参数 ---
    # --- architecture switches for controlled ablation ---
    "use_channel_graph": True,
    "use_temporal_graph": True,
    "use_dynamic_temporal_graph": False,
    "dynamic_temporal_residual_init": 0.1,
    "dynamic_temporal_topk": None,
    "dynamic_temporal_gate_mode": "global",
    "channel_graph_lr_scale": 0.1,
    "temporal_graph_lr_scale": 0.1,
    "use_vq_bypass": True,
    "use_multi_scale_scorer": False,
    "use_direct_vq_score": False,
    "lambda_vq": 0.1,
    "vq_cooldown_epochs": 10,
    "vq_score_weight": 0.3,
    "score_topk_k": None,
    "use_synthetic_anomaly_aux": False,
    "use_synthetic_anomaly_head": False,
    "use_synthetic_rca_head": False,
    "use_synthetic_score": False,
    "lambda_synthetic_anomaly": 0.0,
    "synthetic_aux_interval": 1,
    "use_synthetic_rca_loss": False,
    "lambda_synthetic_rca": 0.0,
    "synthetic_rca_margin": 0.2,
    "synthetic_rca_topk": 5,
    "synthetic_rca_min_roots": 1,
    "synthetic_rca_max_roots": 3,
    "synthetic_rca_bce_weight": 1.0,
    "synthetic_rca_rank_weight": 1.0,
    "synthetic_score_weight": 0.1,
    "synthetic_score_eps": 1e-6,
    "synthetic_min_len": 4,
    "synthetic_max_len": 20,
    "use_source_effect_synthetic": False,
    "lambda_source_effect": 0.0,
    "source_effect_interval": 4,
    "source_effect_min_len": 8,
    "source_effect_max_len": 30,
    "source_effect_min_roots": 1,
    "source_effect_max_roots": 2,
    "source_effect_neighbor_topk": 3,
    "source_effect_strength": 0.35,
    "source_effect_delay_max": 6,
    "source_effect_bce_weight": 1.0,
    "source_effect_rank_weight": 1.0,
    "source_effect_effect_rank_weight": 0.5,
    "source_effect_margin": 0.2,
    "use_channel_masked_modeling": False,
    "lambda_channel_masked": 0.0,
    "channel_mask_interval": 8,
    "channel_mask_ratio": 0.15,
    "channel_mask_min_channels": 1,
    "channel_mask_value": "zero",
    "use_parallel_graph_fusion": False,
    "graph_fusion_gate_mode": "sample",
    "graph_fusion_strategy": "parallel",
    "graph_fusion_residual_init": 0.1,
    "use_graph_shift_score": False,
    "graph_shift_score_weight": 0.1,
    "graph_shift_score_eps": 1e-6,
    "use_lagged_causal_graph": False,
    "causal_lags": [1, 2, 4],
    "causal_topk": 5,
    "causal_detach_backbone": True,
    "lambda_causal_mechanism": 0.0,
    "lambda_causal_sparse": 0.0,
    "use_causal_score": False,
    "causal_score_weight": 0.1,
    "causal_score_eps": 1e-6,
    "causal_score_mode": "residual",
    "causal_score_tail": "upper",
    "use_temporal_graph_regularization": False,
    "lambda_temporal_graph_smooth": 0.0,
    "lambda_temporal_graph_locality": 0.0,
    "reconstruction_loss_type": "mse",
    "smooth_l1_beta": 1.0,
    "mse_l1_alpha": 0.7,
    "mse_robust_alpha": 0.9,
    "charbonnier_eps": 1e-3,
    "lambda_temporal_diff_loss": 0.0,
    "use_robust_reconstruction_loss": False,
    "robust_loss_trim_ratio": 0.0,
    "robust_loss_min_weight": 0.2,
    "robust_loss_warmup_epochs": 10,
    "use_score_channel_normalization": False,
    "score_channel_norm_mode": "robust_z",
    "score_channel_norm_eps": 1e-6,
    "use_channel_corr_prior": False,
    "channel_corr_prior_weight": 0.0,
    "channel_corr_prior_bias": 0.0,
    "channel_corr_prior_topk": 5,
    "lambda_channel_prior_align": 0.0,
    "lambda_channel_mechanism": 0.0,
    "use_channel_mechanism_score": False,
    "channel_mechanism_score_weight": 0.1,
    "channel_mechanism_score_eps": 1e-6,
    "use_mechanism_coupled_decoder": False,
    "mechanism_coupling_init": 0.15,
    "use_mechanism_predictive_head": False,
    "mechanism_predictive_blend_init": 0.30,
    "use_source_gate": False,
    "source_gate_init": 0.20,
    "lambda_source_gate_sparse": 0.0,
    "use_state_aware_fusion": False,
    "state_aware_num_states": 4,
    "state_aware_graph_gate_init": 0.6,
    "state_aware_residual_init": 0.15,
    "lambda_state_balance": 0.0,
    "lambda_state_confidence": 0.0,
    # --- robust industrial sensor preprocessing ---
    "use_robust_input_preprocess": False,
    "input_clip_lower_quantile": 0.001,
    "input_clip_upper_quantile": 0.999,
    "input_clip_eps": 1e-12,
    # --- RTX 5070 single-GPU training path ---
    "dataloader_num_workers": 2,
    "dataloader_prefetch_factor": 2,
    # --- Affiliation-oriented inference shaping ---
    "score_aggregation": "mean",
    "score_aggregation_quantile": 0.9,
    "score_center_width": 1,
    "score_smoothing_window": 1,
    "score_smoothing_method": "mean",
    "use_event_persistence_score": False,
    "event_persistence_window": 9,
    "event_persistence_weight": 0.5,
    "event_persistence_eps": 1e-6,
    "prediction_fill_gap": 0,
    "prediction_min_len": 1,
    "prediction_dilate": 0,
    "export_rca": False,
    "rca_export_lite": False,
    "rca_export_top_k": 20,
    "rca_graph_weight": 0.0,
    "rca_mechanism_weight": 0.0,
    "rca_use_source_propagation": False,
    "rca_source_weight": 0.75,
    "rca_source_base_weight": 1.0,
    "rca_propagation_weight": 0.25,
    "rca_source_mechanism_weight": 0.0,
    "rca_causal_weight": 0.0,
    "rca_synthetic_weight": 0.0,
    "rca_source_gate_weight": 0.0,
    "rca_source_interaction_weight": 0.0,
    "rca_counterfactual_weight": 0.0,
    "rca_counterfactual_candidates": 12,
    "rca_counterfactual_max_windows": 32,
    "rca_counterfactual_batch_candidates": 4,
    "rca_counterfactual_baseline_window": 300,
    "rca_graph_direction": "outgoing",
    "rca_contrast_window": 0,
    "rca_contrast_weight": 0.0,
    "rca_mechanism_residual_window": 0,
    "rca_mechanism_residual_weight": 0.0,
    "rca_event_component_normalize": False,
    "rca_graph_penalty_weight": 0.0,
    "rca_event_head_ratio": 1.0,
    "rca_event_head_points": 0,
    "rca_onset_weight": 0.0,
    "rca_onset_baseline_window": 200,
    "rca_onset_z": 2.0,
    "rca_prediction_key": "pot",
    "rca_event_local_export": False,
    "rca_event_local_margin": 100,
    "rca_split_predicted_events": False,
    "rca_split_max_event_len": 120,
    "rca_split_stride": 60,
    # --- v11.4 P0-2: POT 阈值参数 ---
    "pot_risk": 1e-4,            # POT EVT 风险水平
    "pot_num_quantiles": 1000,   # POT 分位数数量
    # --- v11.4 P0-4: 多 seed 实验参数 ---
    "num_seeds": 5,              # 5-seed 实验
    "seed_base": 42,
}


# ======================== 辅助函数 ========================
def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m{int(s):02d}s"
    else:
        h, remainder = divmod(seconds, 3600)
        m, s = divmod(remainder, 60)
        return f"{int(h)}h{int(m):02d}m{int(s):02d}s"


def _format_params(n: int) -> str:
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def _get_gpu_info(device) -> str:
    if not torch.cuda.is_available():
        return "CPU"
    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx).replace("NVIDIA GeForce ", "")
    mem_total = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits", "-i", str(idx)],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            gpu_util = parts[0].strip() if len(parts) > 0 else "?"
            mem_used = int(parts[1].strip()) if len(parts) > 1 else 0
            temp = parts[2].strip() if len(parts) > 2 else "?"
            return f"{name} | Mem:{mem_used}/{mem_total:.0f}GB | Util:{gpu_util}% | {temp}°C"
    except Exception:
        pass
    return f"{name} | Mem:~/{mem_total:.0f}GB"


def _get_n_gpus():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0


def _pick_best_gpu() -> str:
    """选择剩余显存最多的 GPU"""
    if not torch.cuda.is_available():
        return "cpu"
    best_idx = 0
    best_free = 0
    for idx in range(torch.cuda.device_count()):
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            if free_bytes > best_free:
                best_free = free_bytes
                best_idx = idx
        except Exception:
            continue
    return f"cuda:{best_idx}"


# ══════════════════════════════════════════════════════════════════
#  ★ P0-2: POT (Peak Over Threshold) 阈值估计器
# ══════════════════════════════════════════════════════════════════

class POTThresholdEstimator:
    """
    基于极值理论 (EVT) 的 POT 阈值估计器（P0-2 新增）。

    原理:
      异常检测中，阈值选取直接影响 F1。POT 对分数尾部建模，
      采用广义帕累托分布 (GPD) 拟合超出阈值的分数，然后
      根据风险水平估计最优阈值。

    流程:
      1. 对训练分数排序，选取尾部 (如 top 0.1%) 作为"超出量"
      2. 用 GPD 拟合超出量：F(x) = 1 - (1 + ξ*(x - μ)/σ)^(-1/ξ)
      3. 根据 risk 分位数反推阈值

    ★ v11.4 新增: 验证集自适应校准
      在 POT 估计后，用验证集分数微调阈值:
      - 收集验证集上的分数分布
      - 寻找使"验证集异常比例 ≈ 期望异常比例"的阈值偏移
      - 将偏移量叠加到 POT 估计的基准阈值上
    """

    def __init__(self, risk: float = 1e-4, num_quantiles: int = 1000):
        self.risk = risk
        self.num_quantiles = num_quantiles
        self._baseline_threshold = None
        self._calibrated_threshold = None
        self._gpd_params = None

    def estimate(self, scores: np.ndarray) -> float:
        """
        从训练分数估计 POT 阈值（v11.4.2 增强数值稳定性）。

        Args:
            scores: (N,) 训练集上的异常分数

        Returns:
            threshold: POT 估计的阈值

        ★ v11.4.2 数值稳定性增强：
          1. 矩估计梯度裁剪：防止 xi 负值过大 → 公式爆炸
          2. 方差下界保护：var < 1e-12 时回退分位数法
          3. 阈值上界封顶：min(threshold, max_score)
          4. xi 边界保护：当 |xi| > 0.45 时使用 log 近似代替除法
          5. 超出量不足时的退避策略更平滑
        """
        scores = np.asarray(scores, dtype=np.float64).flatten()
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            return 0.0

        max_score = scores.max()
        min_score = scores.min()
        score_range = max_score - min_score

        # 1. 选取初始阈值 q (top risk 比例作为初始门槛)
        #    如果风险比过高，直接用 99.5 分位
        quantile = max(100 * (1 - self.risk), 50.0)  # 不低于中位数
        q = np.percentile(scores, quantile)

        # 2. 选取超出量
        exceedances = scores[scores > q]
        n = len(scores)
        n_exceed = len(exceedances)

        # ★ v11.4.2: 超出量不足时的退避策略
        if n_exceed < 5 or score_range < 1e-10:
            # 分数基本无变化或超出量太少 → 直接使用高分位
            threshold = np.percentile(scores, max(100 - 5 * self.risk * 100, 50))
            self._baseline_threshold = threshold
            self._gpd_params = None
            return threshold

        # 3. 拟合 GPD（增强数值稳定性）
        exceedances_shifted = exceedances - q
        mean = np.mean(exceedances_shifted)
        var = np.var(exceedances_shifted)

        # ★ v11.4.2: 方差下界保护
        if var < 1e-12 or mean < 1e-12:
            # 方差极小时矩估计不可靠，直接返回高分位
            threshold = np.percentile(scores, 100 * (1 - self.risk * 5))
            self._baseline_threshold = threshold
            self._gpd_params = None
            return threshold

        # ★ v11.4.2: 矩估计 + 梯度裁剪
        #   xi = 0.5 * (1 - mean²/var)
        #   当 var 很小而 mean 较大时，mean²/var 极大 → xi 极大负值 → 公式爆炸
        var_clipped = max(var, 1e-8 * max(1.0, mean**2))  # 防止极端比值
        xi_raw = 0.5 * (1 - mean**2 / var_clipped)
        # ★ v11.4.2: 更强的 xi 边界（GPD 要求 xi < 0.5 才有有限方差）
        xi = max(-0.45, min(xi_raw, 0.45))
        # ★ v11.4.2: sigma 边界保护
        sigma = max(0.5 * mean * (1 + xi), 1e-10)
        # sigma 不应超过 score_range 的 10 倍
        sigma = min(sigma, score_range * 10.0)

        self._gpd_params = {'xi': xi, 'sigma': sigma, 'q': q, 'xi_raw': float(xi_raw)}

        # 4. 根据 risk 计算阈值（★ v11.4.2: 增强数值稳定性）
        #   F^{-1}(p) = q + sigma/xi * ((n_exceed / (N * risk))^{-xi} - 1)
        p = self.risk
        ratio = n_exceed / n  # 超出量比例

        # ★ v11.4.2: 当 |xi| 接近 0 时，使用 L'Hospital 近似
        #   避免 xi→0 的除零问题
        if abs(xi) < 1e-4:
            # lim_{xi→0}: (r/p)^{-xi} ≈ 1 - xi * log(r/p)
            threshold = q + sigma * np.log(ratio / p)
        else:
            inner = (ratio / p) ** (-xi) - 1
            # ★ v11.4.2: inner 保护（防止极端值）
            inner = max(inner, -1.0)  # 不应当低于 -1
            threshold = q + sigma / xi * inner

        # ★ v11.4.2: 阈值上下界保护
        #   上界：不超过 max_score 的 2 倍（避免 EVT 外推过度）
        #   下界：不低于 q（初始尾部分位）
        threshold = max(threshold, q)
        threshold = min(threshold, max_score * 2.0)

        self._baseline_threshold = threshold
        return threshold

    def calibrate(self, val_scores: np.ndarray, target_ratio: float = 0.01) -> float:
        """
        用验证集分数校准 POT 阈值。

        Args:
            val_scores: (N,) 验证集上的分数
            target_ratio: 期望的异常比例

        Returns:
            calibrated_threshold: 校准后的阈值

        ★ Bugfix v11.4: 修复方向错误
          - 旧版: threshold = baseline - shift
            naive_ratio > target_ratio → shift > 0 → threshold 降低 → 更多异常 → 错误方向
          - 新版: threshold = baseline + shift
            naive_ratio > target_ratio → shift > 0 → threshold 提高 → 更少异常 → 正确方向
        """
        if self._baseline_threshold is None:
            raise RuntimeError("Must call estimate() before calibrate()")

        val_scores = np.asarray(val_scores, dtype=np.float64).flatten()
        val_scores = val_scores[np.isfinite(val_scores)]
        if len(val_scores) == 0:
            return self._baseline_threshold

        # 在验证集上通过 POT 基准阈值，计算实际异常比例
        naive_ratio = (val_scores > self._baseline_threshold).mean()

        # 计算比例偏差，调整阈值
        if naive_ratio > 0:
            ratio_bias = np.log(naive_ratio / max(target_ratio, 1e-8))
        else:
            ratio_bias = 0.0

        # 使用中位数绝对偏差 (MAD) 估计梯度
        mad = np.median(np.abs(val_scores - np.median(val_scores))) + 1e-10
        shift = ratio_bias * mad

        # ★ BUGFIX: naive_ratio > target_ratio → 阈值过低 → 需要提高阈值
        #   旧版: threshold = baseline - shift (方向错误)
        #   新版: threshold = baseline + shift (方向正确)
        threshold = self._baseline_threshold + shift
        self._calibrated_threshold = threshold
        return threshold

    def get_threshold(self) -> float:
        """获取最终阈值（优先校准后值）"""
        if self._calibrated_threshold is not None:
            return self._calibrated_threshold
        if self._baseline_threshold is not None:
            return self._baseline_threshold
        return 0.0


# ══════════════════════════════════════════════════════════════════
#  ★ P0-4: 多 seed 实验运行器
# ══════════════════════════════════════════════════════════════════

class MultiSeedExperimentRunner:
    """
    多 seed 实验运行器（P0-4 新增）。

    功能:
      1. 对同一数据集运行 LaGraph detect_label 多次（不同 seed）
      2. 收集每次运行的指标（AUC-ROC, AUPR, 各 ratio 的 F1）
      3. 计算均值和方差，输出统计显著性报告

    使用:
      runner = MultiSeedExperimentRunner(num_seeds=5, seed_base=42)
      results = runner.run(model_class, dataset)
    """

    def __init__(self, num_seeds: int = 5, seed_base: int = 42):
        self.num_seeds = num_seeds
        self.seed_base = seed_base
        self.results = {
            'auc_roc': [],
            'aupr': [],
            'f1_per_ratio': {},
        }

    def _compute_metrics(self, scores: np.ndarray, labels: np.ndarray) -> dict:
        """
        计算 AUC-ROC, AUPR, Best F1。

        Args:
            scores: (N,) 异常分数
            labels: (N,) 真实 0/1 标签

        Returns:
            metrics: {'auc_roc': float, 'aupr': float, 'best_f1': float}
        """
        scores = np.asarray(scores, dtype=np.float64).flatten()
        labels = np.asarray(labels, dtype=np.int32).flatten()

        # 过滤异常值
        valid = np.isfinite(scores)
        scores, labels = scores[valid], labels[valid]

        if len(np.unique(labels)) < 2:
            return {'auc_roc': 0.5, 'aupr': labels.mean(), 'best_f1': 0.0}

        try:
            auc_roc = roc_auc_score(labels, scores)
        except Exception:
            auc_roc = 0.5
        try:
            aupr = average_precision_score(labels, scores)
        except Exception:
            aupr = labels.mean()

        # 搜索最佳 F1
        thresholds = np.percentile(scores, np.linspace(99.9, 50, 500))
        best_f1 = 0.0
        for th in thresholds:
            pred = (scores > th).astype(int)
            tp = (pred * labels).sum()
            fp = pred.sum() - tp
            fn = labels.sum() - tp
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-10)
            best_f1 = max(best_f1, f1)

        return {'auc_roc': auc_roc, 'aupr': aupr, 'best_f1': best_f1}

    def run(self, model_instance, test_data: pd.DataFrame, test_labels: np.ndarray) -> dict:
        """
        对已训练的模型实例运行多次检测（仅检测阶段重复，模拟不同 seed 效果）。

        Args:
            model_instance: 已训练的 LaGraph 实例
            test_data: 测试数据
            test_labels: 真实标签 (N,)

        Returns:
            summary: 统计总结
        """
        for seed_idx in range(self.num_seeds):
            seed = self.seed_base + seed_idx
            np.random.seed(seed)
            torch.manual_seed(seed)

            preds, scores = model_instance.detect_label(test_data)

            # 对默认 ratio=1% 评估
            ratio_key = 1.0
            if ratio_key in preds:
                pred_labels = preds[ratio_key]
            else:
                # 取最接近的 ratio
                available = sorted(preds.keys())
                ratio_key = min(available, key=lambda x: abs(x - 1.0))
                pred_labels = preds[ratio_key]

            # 计算指标
            metrics = self._compute_metrics(scores, test_labels)
            self.results['auc_roc'].append(metrics['auc_roc'])
            self.results['aupr'].append(metrics['aupr'])

            for ratio, pred in preds.items():
                f1 = self._compute_f1(pred, test_labels)
                if ratio not in self.results['f1_per_ratio']:
                    self.results['f1_per_ratio'][ratio] = []
                self.results['f1_per_ratio'][ratio].append(f1)

        # 计算统计总结
        return self._summarize()

    def _compute_f1(self, pred: np.ndarray, labels: np.ndarray) -> float:
        pred = pred.flatten().astype(int)
        labels = labels.flatten().astype(int)
        tp = (pred * labels).sum()
        fp = pred.sum() - tp
        fn = labels.sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        return 2 * prec * rec / max(prec + rec, 1e-10)

    def _summarize(self) -> dict:
        """计算均值和置信区间。"""
        summary = {
            'num_seeds': self.num_seeds,
            'auc_roc': {
                'mean': float(np.mean(self.results['auc_roc'])),
                'std': float(np.std(self.results['auc_roc'], ddof=1)),
                'values': [float(v) for v in self.results['auc_roc']],
            },
            'aupr': {
                'mean': float(np.mean(self.results['aupr'])),
                'std': float(np.std(self.results['aupr'], ddof=1)),
                'values': [float(v) for v in self.results['aupr']],
            },
            'f1_per_ratio': {},
        }

        # 95% 置信区间
        for metric in ['auc_roc', 'aupr']:
            vals = self.results[metric]
            if len(vals) >= 2:
                se = np.std(vals, ddof=1) / np.sqrt(len(vals))
                ci = scipy_stats.t.ppf(0.975, len(vals) - 1) * se
                summary[metric]['ci_95'] = float(ci)
            else:
                summary[metric]['ci_95'] = 0.0

        for ratio, vals in self.results['f1_per_ratio'].items():
            summary['f1_per_ratio'][ratio] = {
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)),
                'values': [float(v) for v in vals],
            }

        return summary


# ══════════════════════════════════════════════════════════════════
#  ★ P0-5: 中间结果可视化接口
# ══════════════════════════════════════════════════════════════════

class VisualizationHook:
    """
    中间结果可视化接口（P0-5 新增）。

    功能:
      1. 注册 forward hook，捕获各层特征
      2. 提取图结构 (A_channel, A_temp)
      3. 计算梯度统计
      4. 保存为可序列化的 dict，供外部可视化

    用法:
      hook = VisualizationHook(model)
      hook.register_hooks()

      # 在某个 batch 前向后:
      vis_data = hook.extract()
      hook.save_json("vis_data.json")
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._hooks = []
        self._features = {}
        self._gradients = {}
        self._graph_matrices = {}

    def register_hooks(self):
        """注册 forward hook 到主要子模块。"""
        for name, module in self.model.named_modules():
            # 捕获 VQ bottleneck 输入/输出
            if 'vq_bottleneck' in name:
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )
            # 捕获通道图
            if 'channel_graph' in name:
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )
            # 捕获时序图
            if 'temporal_graph' in name or 'simplified_temporal' in name:
                self._hooks.append(
                    module.register_forward_hook(self._make_hook(name))
                )

    def _make_hook(self, name: str):
        def hook(module, input, output):
            self._features[name] = {
                'input_norm': float(torch.norm(input[0]).item()) if input else 0.0,
                'output_shape': list(output.shape) if torch.is_tensor(output) else None,
                'output_norm': float(torch.norm(output).item()) if torch.is_tensor(output) else 0.0,
            }
            # 记录图结构（如果是元组输出，可能包含 A 矩阵）
            if isinstance(output, (tuple, list)) and len(output) >= 2:
                for i, o in enumerate(output):
                    if torch.is_tensor(o) and o.dim() >= 2 and o.size(-1) == o.size(-2):
                        # 这是一个平方矩阵 (CxC 或 LxL) — 图邻接矩阵
                        key = f"{name}_A"
                        self._graph_matrices[key] = self._matrix_stats(o)
        return hook

    @staticmethod
    def _matrix_stats(mat: torch.Tensor) -> dict:
        """计算矩阵统计信息（不保存完整矩阵以避免 OOM）。"""
        mat = mat.detach().cpu()
        return {
            'shape': list(mat.shape),
            'mean': float(mat.mean().item()),
            'std': float(mat.std().item()),
            'sparsity': float((mat.abs() < 1e-6).float().mean().item()),
            'min': float(mat.min().item()),
            'max': float(mat.max().item()),
            'symmetry': float((mat - mat.transpose(-2, -1)).abs().mean().item()),
        }

    def extract(self) -> dict:
        """提取所有可视化的中间数据。"""
        data = {
            'features': dict(self._features),
            'graph_matrices': dict(self._graph_matrices),
            'gradients': {},
        }

        # 捕获梯度统计
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().cpu()
                data['gradients'][name] = {
                    'shape': list(grad.shape),
                    'mean': float(grad.mean().item()),
                    'std': float(grad.std().item()),
                    'norm': float(grad.norm().item()),
                    'sparsity': float((grad.abs() < 1e-8).float().mean().item()),
                }

        return data

    def save_to_json(self, data: dict, filepath: str):
        """保存可视化数据为 JSON。"""
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=_NumpyEncoder)
        return filepath

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


class _NumpyEncoder(json.JSONEncoder):
    """支持 numpy 类型序列化的 JSON 编码器。"""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


# ======================== 早停机制 ========================
class EarlyStopping:
    """
    早停机制，使用**相对阈值**（relative delta）。

    对比绝对 delta：
      - 绝对 delta=1e-4：对于 MSE loss=0.008，0.0001/0.008=1.25% — 太严格
      - 相对 delta=0.001：需要下降 ≥0.1% 才算 improvement

    ★ v10 fix: 使用相对 delta 替代绝对 delta。
    """
    def __init__(self, patience=7, verbose=False, delta=0, relative_delta=0.001):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_val_loss = np.Inf
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.relative_delta = relative_delta
        self.best_epoch = 0

    def __call__(self, val_loss, model, epoch):
        if self.best_score is None:
            self.best_score = -val_loss
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
            print(f"  [ES DBG] init: val_loss={val_loss:.8f}, best={val_loss:.8f}")
            return "initial"

        rel_improvement = (self.best_val_loss - val_loss) / max(self.best_val_loss, 1e-10)

        if rel_improvement >= self.relative_delta:
            self.best_score = -val_loss
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            self.best_epoch = epoch
            print(f"  [ES DBG] ★BEST epoch={epoch}: val_loss={val_loss:.8f} "
                  f"rel_impr={rel_improvement*100:.3f}%")
            return "improved"
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            print(f"  [ES DBG] NO↑ epoch={epoch}: val_loss={val_loss:.8f} "
                  f"rel_impr={rel_improvement*100:.4f}% cnt={self.counter}/{self.patience}")
            return "no_improve"

    def save_checkpoint(self, val_loss, model):
        self.val_loss_min = val_loss
        state_dict = model.state_dict()
        self.check_point = copy.deepcopy({k: v.cpu().clone() for k, v in state_dict.items()})


# ======================== 配置类 ========================
class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)
        for key, value in kwargs.items():
            setattr(self, key, value)


# ======================== LaGraph v11.4 主类 ========================
class LaGraph:
    """
    LaGraph v11.4 — P0 五项修复合并版

    ★ P0-1: 术语诚实化（locality/adaptive_graph 统一）
    ★ P0-2: POT 阈值估计 + 验证集自适应校准
    ★ P0-3: AUC-ROC/AUPR 硬指标报告
    ★ P0-4: 多 seed 实验框架
    ★ P0-5: 中间结果可视化接口
    """

    def __init__(self, **kwargs):
        super(LaGraph, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.scaler = StandardScaler()
        self._input_clip_lower = None
        self._input_clip_upper = None
        self.win_size = self.config.win_size

        # GPU 检测
        self.config_n_gpu = getattr(self.config, 'n_gpus', None)
        hardware_ngpu = _get_n_gpus()
        self.multi_gpu_requested = self.config_n_gpu is not None and self.config_n_gpu > 1
        if self.config_n_gpu == 0:
            self.ngpu = 0
        else:
            self.ngpu = 1 if hardware_ngpu >= 1 else 0

        if self.ngpu >= 1:
            self.device = torch.device(_pick_best_gpu())
        else:
            self.device = torch.device("cpu")
            self.ngpu = 0

        self.dataset_name = getattr(self.config, 'dataset_name', 'Unknown')

        self.early_stopping = None
        self.model = None
        self.optimizer = None
        self.train_loader = None
        self.valid_loader = None
        self.trained = False

        # ★ P0-5: 可视化 hook
        self._vis_hook = None

        # 阈值缓存
        self._train_raw = None
        self._train_anomaly_scores = None
        self._val_anomaly_scores = None
        self._last_channel_scores = None
        self._last_graph_channel_scores = None
        self._last_mechanism_channel_scores = None
        self._last_causal_channel_scores = None
        self._last_source_gate_channel_scores = None
        self._last_synthetic_rca_channel_scores = None
        self._last_channel_names = None
        self._channel_corr_prior = None

        # ★ P0-2: POT 阈值估计器
        self._pot_estimator = POTThresholdEstimator(
            risk=self.config.pot_risk,
            num_quantiles=self.config.pot_num_quantiles,
        )

    @staticmethod
    def _clean_cuda_memory(device=None):
        import gc
        if device is not None:
            devices = [device]
        else:
            devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
        for d in devices:
            try:
                torch.cuda.synchronize(d)
                with torch.cuda.device(d):
                    torch.cuda.empty_cache()
            except Exception:
                pass
        gc.collect()
        for _ in range(3):
            torch.cuda.empty_cache()

    @staticmethod
    def required_hyper_params() -> dict:
        return {}

    def __repr__(self) -> str:
        return "LaGraph-v11.4"

    def _should_load_best_checkpoint(self) -> bool:
        return (
            self.early_stopping is not None
            and self.early_stopping.check_point is not None
            and not bool(getattr(self.config, "use_latest_checkpoint", False))
        )

    def _fit_input_preprocessor(self, train_frame: pd.DataFrame) -> None:
        values = train_frame.values.astype(np.float64, copy=False)
        if getattr(self.config, "use_robust_input_preprocess", False):
            lower_q = float(getattr(self.config, "input_clip_lower_quantile", 0.001))
            upper_q = float(getattr(self.config, "input_clip_upper_quantile", 0.999))
            lower_q = min(max(lower_q, 0.0), 0.5)
            upper_q = min(max(upper_q, 0.5), 1.0)
            if lower_q >= upper_q:
                raise ValueError(
                    "input_clip_lower_quantile must be smaller than input_clip_upper_quantile"
                )
            self._input_clip_lower = np.nanquantile(values, lower_q, axis=0)
            self._input_clip_upper = np.nanquantile(values, upper_q, axis=0)
            eps = float(getattr(self.config, "input_clip_eps", 1e-12) or 1e-12)
            invalid = (self._input_clip_upper - self._input_clip_lower) < eps
            if np.any(invalid):
                self._input_clip_lower[invalid] = -np.inf
                self._input_clip_upper[invalid] = np.inf
            fit_values = np.clip(values, self._input_clip_lower, self._input_clip_upper)
            print(
                "\n  [InputPreprocess] Robust channel clipping enabled: "
                f"q=({lower_q:.4f}, {upper_q:.4f}), clipped_constant_channels={int(np.sum(invalid))}"
            )
        else:
            self._input_clip_lower = None
            self._input_clip_upper = None
            fit_values = values
        self.scaler.fit(fit_values)

    def _transform_input_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        values = frame.values.astype(np.float64, copy=False)
        if self._input_clip_lower is not None and self._input_clip_upper is not None:
            values = np.clip(values, self._input_clip_lower, self._input_clip_upper)
        return pd.DataFrame(
            self.scaler.transform(values),
            columns=frame.columns,
            index=frame.index,
        )

    def _get_raw_model(self):
        return self.model

    # ======================== 终端输出美化 ========================
    def _print_training_header(self, n_train, n_val, n_features, train_steps, total_params):
        W = 70
        print("╔" + "═" * W + "╗")
        title = f"  LaGraph v11.4 — {self.dataset_name}  "
        pad_total = W - len(title)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        print("║" + " " * pad_left + title + " " * pad_right + "║")
        print("╠" + "═" * W + "╣")

        gpu_str = _get_gpu_info(self.device)
        params_str = _format_params(total_params)

        lines = [
            f"  Dataset    │ {n_train:,} train + {n_val:,} val samples",
            f"  Features   │ {n_features} variables    │  Window  │ {self.config.win_size}",
            f"  Batch      │ {self.config.batch_size}",
            f"  Epochs     │ {self.config.num_epochs}                     │  d_model │ {self.config.d_model}",
            f"  Patience   │ {self.config.patience}                    │  e_layers│ {self.config.e_layers}",
            f"  Params     │ {params_str:>8s}               │  topk    │ {self.config.topk}",
            f"  Device     │ {gpu_str}",
        ]

        for line in lines:
            line = line + " " * (W - len(line)) if len(line) <= W else line[:W]
            print("║" + line + "║")

        LR_STR = f"  LR: {self.config.lr:.2e}  |  λ_l1: {self.config.lambda_locality_l1}  |  Dropout: {self.config.dropout}"
        print("╠" + "═" * W + "╣")
        print("║" + LR_STR + " " * (W - len(LR_STR)) + "║")
        print("╚" + "═" * W + "╝")
        print()

    def _print_progress_bar(self, current, total, width=30):
        ratio = current / total if total > 0 else 1.0
        filled = int(width * ratio)
        bar = "█" * filled + "░" * (width - filled)
        pct = ratio * 100
        return f"{bar}  {current:>{len(str(total))}}/{total} ({pct:5.1f}%)"

    def _print_epoch_result(self, epoch, epoch_time, avg_train_loss,
                            val_loss, lr, total_epochs, es_status, es_counter):
        bar = self._print_progress_bar(epoch, total_epochs, 25)
        time_str = _format_duration(epoch_time)
        remaining = (total_epochs - epoch) * epoch_time
        eta_str = _format_duration(remaining) if remaining > 0 else "—"

        print(f"\n  Epoch {epoch:>2}/{total_epochs:<2} {bar}  [{time_str:>7s}]")

        if es_status == "improved" or es_status == "initial":
            val_mark = "★ BEST"
        else:
            val_mark = f"▼ {es_counter}/{self.config.patience}"

        best_val = self.early_stopping.val_loss_min
        delta_val = val_loss - best_val
        if delta_val <= 0:
            delta_str = f"  Δ best: -{abs(delta_val):.6f} ↓"
        else:
            delta_str = f"  Δ best: +{delta_val:.6f} ↑"

        print(f"  Train Loss  {avg_train_loss:>10.6f}  │  Val Loss  {val_loss:>10.6f} {val_mark:<8s}  │  LR  {lr:.2e}")
        print(f"  Best Val    {best_val:>10.6f}{delta_str:>22s}  │  ETA  {eta_str:>8s}")
        print()

    # ======================== 参数保存 ========================
    def _save_best_params(self, total_params, total_time):
        from datetime import datetime
        from ts_benchmark.common.constant import ROOT_PATH
        params_dir = os.path.join(ROOT_PATH, "result", "params")
        os.makedirs(params_dir, exist_ok=True)

        hostname = socket.gethostname()
        pid = os.getpid()
        now = datetime.now()
        date_part = now.strftime("%Y-%m-%d_%H-%M-%S")
        micro_part = f"{now.microsecond:06d}"
        filename = f"LaGraph_params_{date_part}_{micro_part}.json"
        filepath = os.path.join(params_dir, filename)

        params_record = {
            "meta": {
                "model": "LaGraph-v11.4",
                "dataset": self.dataset_name,
                "timestamp": now.timestamp(),
                "hostname": hostname,
                "pid": pid,
                "saved_at": now.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "results": {
                "best_val_loss": float(self.early_stopping.val_loss_min),
                "best_epoch": self.early_stopping.best_epoch,
                "total_epochs_run": self.early_stopping.best_epoch,
                "total_train_time_seconds": round(total_time, 1),
                "total_train_time_human": _format_duration(total_time),
            },
            "hyper_params": {
                "win_size": self.config.win_size,
                "lr": self.config.lr,
                "dropout": self.config.dropout,
                "n_heads": self.config.n_heads,
                "e_layers": self.config.e_layers,
                "d_model": self.config.d_model,
                "num_epochs": self.config.num_epochs,
                "batch_size": self.config.batch_size,
                "patience": self.config.patience,
                "topk": self.config.topk,
                "anomaly_ratio": self.config.anomaly_ratio,
                "sparse_topk": getattr(self.config, 'sparse_topk', None),
                "lambda_locality_l1": self.config.lambda_locality_l1,
                "lambda_vq": getattr(self.config, "lambda_vq", None),
                "vq_cooldown_epochs": getattr(self.config, "vq_cooldown_epochs", None),
                "vq_score_weight": getattr(self.config, "vq_score_weight", None),
                "use_direct_vq_score": getattr(self.config, "use_direct_vq_score", None),
                "score_topk_k": getattr(self.config, "score_topk_k", None),
                "score_aggregation": getattr(self.config, "score_aggregation", None),
                "score_aggregation_quantile": getattr(self.config, "score_aggregation_quantile", None),
                "score_center_width": getattr(self.config, "score_center_width", None),
                "use_synthetic_anomaly_aux": getattr(self.config, "use_synthetic_anomaly_aux", None),
                "lambda_synthetic_anomaly": getattr(self.config, "lambda_synthetic_anomaly", None),
                "synthetic_aux_interval": getattr(self.config, "synthetic_aux_interval", None),
                "use_source_effect_synthetic": getattr(self.config, "use_source_effect_synthetic", None),
                "lambda_source_effect": getattr(self.config, "lambda_source_effect", None),
                "source_effect_interval": getattr(self.config, "source_effect_interval", None),
                "use_channel_masked_modeling": getattr(self.config, "use_channel_masked_modeling", None),
                "lambda_channel_masked": getattr(self.config, "lambda_channel_masked", None),
                "channel_mask_interval": getattr(self.config, "channel_mask_interval", None),
                "channel_mask_ratio": getattr(self.config, "channel_mask_ratio", None),
                "use_synthetic_score": getattr(self.config, "use_synthetic_score", None),
                "synthetic_score_weight": getattr(self.config, "synthetic_score_weight", None),
                "use_parallel_graph_fusion": getattr(self.config, "use_parallel_graph_fusion", None),
                "graph_fusion_gate_mode": getattr(self.config, "graph_fusion_gate_mode", None),
                "graph_fusion_strategy": getattr(self.config, "graph_fusion_strategy", None),
                "graph_fusion_residual_init": getattr(self.config, "graph_fusion_residual_init", None),
                "use_graph_shift_score": getattr(self.config, "use_graph_shift_score", None),
                "graph_shift_score_weight": getattr(self.config, "graph_shift_score_weight", None),
                "use_lagged_causal_graph": getattr(self.config, "use_lagged_causal_graph", None),
                "causal_lags": getattr(self.config, "causal_lags", None),
                "causal_topk": getattr(self.config, "causal_topk", None),
                "causal_detach_backbone": getattr(self.config, "causal_detach_backbone", None),
                "lambda_causal_mechanism": getattr(self.config, "lambda_causal_mechanism", None),
                "lambda_causal_sparse": getattr(self.config, "lambda_causal_sparse", None),
                "use_causal_score": getattr(self.config, "use_causal_score", None),
                "causal_score_mode": getattr(self.config, "causal_score_mode", None),
                "causal_score_tail": getattr(self.config, "causal_score_tail", None),
                "causal_score_weight": getattr(self.config, "causal_score_weight", None),
                "use_temporal_graph_regularization": getattr(self.config, "use_temporal_graph_regularization", None),
                "lambda_temporal_graph_smooth": getattr(self.config, "lambda_temporal_graph_smooth", None),
                "lambda_temporal_graph_locality": getattr(self.config, "lambda_temporal_graph_locality", None),
                "reconstruction_loss_type": getattr(self.config, "reconstruction_loss_type", None),
                "smooth_l1_beta": getattr(self.config, "smooth_l1_beta", None),
                "mse_l1_alpha": getattr(self.config, "mse_l1_alpha", None),
                "mse_robust_alpha": getattr(self.config, "mse_robust_alpha", None),
                "charbonnier_eps": getattr(self.config, "charbonnier_eps", None),
                "lambda_temporal_diff_loss": getattr(self.config, "lambda_temporal_diff_loss", None),
                "use_robust_reconstruction_loss": getattr(self.config, "use_robust_reconstruction_loss", None),
                "robust_loss_trim_ratio": getattr(self.config, "robust_loss_trim_ratio", None),
                "robust_loss_min_weight": getattr(self.config, "robust_loss_min_weight", None),
                "robust_loss_warmup_epochs": getattr(self.config, "robust_loss_warmup_epochs", None),
                "use_score_channel_normalization": getattr(self.config, "use_score_channel_normalization", None),
                "score_channel_norm_mode": getattr(self.config, "score_channel_norm_mode", None),
                "score_channel_norm_eps": getattr(self.config, "score_channel_norm_eps", None),
                "use_channel_corr_prior": getattr(self.config, "use_channel_corr_prior", None),
                "channel_corr_prior_weight": getattr(self.config, "channel_corr_prior_weight", None),
                "channel_corr_prior_bias": getattr(self.config, "channel_corr_prior_bias", None),
                "channel_corr_prior_topk": getattr(self.config, "channel_corr_prior_topk", None),
                "lambda_channel_prior_align": getattr(self.config, "lambda_channel_prior_align", None),
                "lambda_channel_mechanism": getattr(self.config, "lambda_channel_mechanism", None),
                "use_channel_mechanism_score": getattr(self.config, "use_channel_mechanism_score", None),
                "channel_mechanism_score_weight": getattr(self.config, "channel_mechanism_score_weight", None),
                "rca_mechanism_weight": getattr(self.config, "rca_mechanism_weight", None),
                "rca_use_source_propagation": getattr(self.config, "rca_use_source_propagation", None),
                "rca_source_weight": getattr(self.config, "rca_source_weight", None),
                "rca_source_base_weight": getattr(self.config, "rca_source_base_weight", None),
                "rca_propagation_weight": getattr(self.config, "rca_propagation_weight", None),
                "rca_source_mechanism_weight": getattr(self.config, "rca_source_mechanism_weight", None),
                "rca_causal_weight": getattr(self.config, "rca_causal_weight", None),
                "rca_onset_weight": getattr(self.config, "rca_onset_weight", None),
                "rca_onset_baseline_window": getattr(self.config, "rca_onset_baseline_window", None),
                "rca_onset_z": getattr(self.config, "rca_onset_z", None),
                "use_state_aware_fusion": getattr(self.config, "use_state_aware_fusion", None),
                "state_aware_num_states": getattr(self.config, "state_aware_num_states", None),
                "state_aware_graph_gate_init": getattr(self.config, "state_aware_graph_gate_init", None),
                "state_aware_residual_init": getattr(self.config, "state_aware_residual_init", None),
                "lambda_state_balance": getattr(self.config, "lambda_state_balance", None),
                "lambda_state_confidence": getattr(self.config, "lambda_state_confidence", None),
                "use_vq_bypass": getattr(self.config, "use_vq_bypass", None),
                "dynamic_temporal_residual_init": getattr(self.config, "dynamic_temporal_residual_init", None),
                "dynamic_temporal_topk": getattr(self.config, "dynamic_temporal_topk", None),
                "dynamic_temporal_gate_mode": getattr(self.config, "dynamic_temporal_gate_mode", None),
                "channel_graph_lr_scale": getattr(self.config, "channel_graph_lr_scale", None),
                "temporal_graph_lr_scale": getattr(self.config, "temporal_graph_lr_scale", None),
                "score_smoothing_window": getattr(self.config, "score_smoothing_window", None),
                "score_smoothing_method": getattr(self.config, "score_smoothing_method", None),
                "use_event_persistence_score": getattr(self.config, "use_event_persistence_score", None),
                "event_persistence_window": getattr(self.config, "event_persistence_window", None),
                "event_persistence_weight": getattr(self.config, "event_persistence_weight", None),
                "prediction_fill_gap": getattr(self.config, "prediction_fill_gap", None),
                "prediction_min_len": getattr(self.config, "prediction_min_len", None),
                "prediction_dilate": getattr(self.config, "prediction_dilate", None),
            },
            "model_info": {
                "total_params": total_params,
                "total_params_human": _format_params(total_params),
            },
            "environment": {
                "device": str(self.device),
                "gpu_info": _get_gpu_info(self.device),
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(params_record, f, indent=2, ensure_ascii=False)
        return filepath

    # ======================== 验证 ========================
    def vali(self, vali_loader):
        self.model.eval()
        loss_list = []
        with torch.inference_mode():
            for input_data, _ in vali_loader:
                input_data = input_data.float().to(self.device, non_blocking=True)
                rec, _, _, _, _, aux_losses, _ = self.model(input_data)
                loss = self._reconstruction_loss(rec, input_data)
                loss = self._add_temporal_difference_loss(loss, rec, input_data)
                # ★ P0-1: lambda_causal_l1 → lambda_locality_l1
                if aux_losses and 'sparse_loss' in aux_losses:
                    loss = loss + self.config.lambda_locality_l1 * aux_losses['sparse_loss']
                loss = self._add_temporal_graph_regularization(loss, aux_losses)
                loss_list.append(loss.item())
        return np.average(loss_list) if loss_list else 0.0


    def _reconstruction_loss(self, rec, target):
        loss_type = str(getattr(self.config, "reconstruction_loss_type", "mse") or "mse").lower()
        diff = rec - target
        if loss_type == "mse":
            elem_loss = diff.pow(2)
        elif loss_type == "mae":
            elem_loss = diff.abs()
        elif loss_type == "smooth_l1":
            beta = float(getattr(self.config, "smooth_l1_beta", 1.0) or 1.0)
            elem_loss = F.smooth_l1_loss(rec, target, reduction='none', beta=max(beta, 1e-6))
        elif loss_type == "log_cosh":
            abs_diff = diff.abs()
            elem_loss = abs_diff + F.softplus(-2.0 * abs_diff) - np.log(2.0)
        elif loss_type == "charbonnier":
            eps = float(getattr(self.config, "charbonnier_eps", 1e-3) or 1e-3)
            elem_loss = torch.sqrt(diff.pow(2) + eps * eps) - eps
        elif loss_type == "mse_mae":
            alpha = float(getattr(self.config, "mse_l1_alpha", 0.7) or 0.7)
            alpha = min(max(alpha, 0.0), 1.0)
            elem_loss = alpha * diff.pow(2) + (1.0 - alpha) * diff.abs()
        elif loss_type == "mse_log_cosh":
            alpha = float(getattr(self.config, "mse_robust_alpha", 0.9) or 0.9)
            alpha = min(max(alpha, 0.0), 1.0)
            abs_diff = diff.abs()
            robust_loss = abs_diff + F.softplus(-2.0 * abs_diff) - np.log(2.0)
            elem_loss = alpha * diff.pow(2) + (1.0 - alpha) * robust_loss
        elif loss_type == "mse_smooth_l1":
            alpha = float(getattr(self.config, "mse_robust_alpha", 0.9) or 0.9)
            alpha = min(max(alpha, 0.0), 1.0)
            beta = float(getattr(self.config, "smooth_l1_beta", 1.0) or 1.0)
            robust_loss = F.smooth_l1_loss(rec, target, reduction='none', beta=max(beta, 1e-6))
            elem_loss = alpha * diff.pow(2) + (1.0 - alpha) * robust_loss
        else:
            raise ValueError(
                f"Unsupported reconstruction_loss_type={loss_type!r}. "
                "Choose from mse, mae, smooth_l1, log_cosh, charbonnier, "
                "mse_mae, mse_log_cosh, mse_smooth_l1."
            )
        sample_loss = elem_loss.mean(dim=(1, 2))
        if not getattr(self.config, "use_robust_reconstruction_loss", False):
            return sample_loss.mean()

        epoch = getattr(self, "_current_train_epoch", 0)
        warmup = int(getattr(self.config, "robust_loss_warmup_epochs", 0) or 0)
        trim_ratio = float(getattr(self.config, "robust_loss_trim_ratio", 0.0) or 0.0)
        if epoch < warmup or trim_ratio <= 0.0 or sample_loss.numel() < 2:
            return sample_loss.mean()

        trim_ratio = min(max(trim_ratio, 0.0), 0.5)
        min_weight = float(getattr(self.config, "robust_loss_min_weight", 0.2) or 0.0)
        min_weight = min(max(min_weight, 0.0), 1.0)
        with torch.no_grad():
            cutoff = torch.quantile(sample_loss.detach(), 1.0 - trim_ratio)
            weights = (sample_loss.detach() <= cutoff).to(sample_loss.dtype)
            if min_weight > 0:
                weights = weights.clamp_min(min_weight)
        return (sample_loss * weights).sum() / weights.sum().clamp_min(1.0)

    def _add_temporal_difference_loss(self, loss, rec, target):
        weight = float(getattr(self.config, "lambda_temporal_diff_loss", 0.0) or 0.0)
        if weight <= 0 or rec.shape[1] < 2:
            return loss
        rec_diff = rec[:, 1:, :] - rec[:, :-1, :]
        target_diff = target[:, 1:, :] - target[:, :-1, :]
        return loss + weight * F.mse_loss(rec_diff, target_diff)

    @staticmethod
    def _build_channel_corr_prior(train_df: pd.DataFrame, topk: int = 5) -> np.ndarray:
        values = np.asarray(train_df.values, dtype=np.float32)
        corr = np.corrcoef(values, rowvar=False)
        corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)
        n_channels = corr.shape[0]
        topk = int(topk or 0)
        if topk > 0 and topk < n_channels:
            keep = np.zeros_like(corr, dtype=bool)
            idx = np.argpartition(-corr, kth=topk - 1, axis=1)[:, :topk]
            rows = np.arange(n_channels)[:, None]
            keep[rows, idx] = True
            corr = np.where(keep, corr, 0.0)
        row_sum = corr.sum(axis=1, keepdims=True)
        empty = row_sum.squeeze(-1) <= 1e-8
        corr = np.divide(corr, row_sum, out=np.zeros_like(corr), where=row_sum > 1e-8)
        if np.any(empty):
            corr[empty, :] = 1.0 / max(1, n_channels - 1)
            empty_rows = np.where(empty)[0]
            corr[empty_rows, empty_rows] = 0.0
            corr[empty, :] = corr[empty, :] / corr[empty, :].sum(axis=1, keepdims=True).clip(min=1e-8)
        return corr.astype(np.float32)


    def _add_temporal_graph_regularization(self, loss, aux_losses):
        if not aux_losses:
            return loss
        lambda_channel_prior = getattr(self.config, "lambda_channel_prior_align", 0.0)
        if lambda_channel_prior > 0 and 'channel_prior_align_loss' in aux_losses:
            loss = loss + lambda_channel_prior * aux_losses['channel_prior_align_loss']
        lambda_channel_mechanism = getattr(self.config, "lambda_channel_mechanism", 0.0)
        if lambda_channel_mechanism > 0 and 'channel_mechanism_loss' in aux_losses:
            loss = loss + lambda_channel_mechanism * aux_losses['channel_mechanism_loss']
        lambda_smooth = getattr(self.config, "lambda_temporal_graph_smooth", 0.0)
        lambda_locality = getattr(self.config, "lambda_temporal_graph_locality", 0.0)
        if lambda_smooth > 0 and 'temporal_graph_smooth_loss' in aux_losses:
            loss = loss + lambda_smooth * aux_losses['temporal_graph_smooth_loss']
        if lambda_locality > 0 and 'temporal_graph_locality_loss' in aux_losses:
            loss = loss + lambda_locality * aux_losses['temporal_graph_locality_loss']
        lambda_causal = getattr(self.config, "lambda_causal_mechanism", 0.0)
        lambda_causal_sparse = getattr(self.config, "lambda_causal_sparse", 0.0)
        if lambda_causal > 0 and 'causal_mechanism_loss' in aux_losses:
            loss = loss + lambda_causal * aux_losses['causal_mechanism_loss']
        if lambda_causal_sparse > 0 and 'causal_sparse_loss' in aux_losses:
            loss = loss + lambda_causal_sparse * aux_losses['causal_sparse_loss']
        lambda_source_gate_sparse = getattr(self.config, "lambda_source_gate_sparse", 0.0)
        if lambda_source_gate_sparse > 0 and 'source_gate_sparse_loss' in aux_losses:
            loss = loss + lambda_source_gate_sparse * aux_losses['source_gate_sparse_loss']
        lambda_state_balance = getattr(self.config, "lambda_state_balance", 0.0)
        lambda_state_confidence = getattr(self.config, "lambda_state_confidence", 0.0)
        if lambda_state_balance > 0 and 'state_balance_loss' in aux_losses:
            loss = loss + lambda_state_balance * aux_losses['state_balance_loss']
        if lambda_state_confidence > 0 and 'state_confidence_loss' in aux_losses:
            loss = loss + lambda_state_confidence * aux_losses['state_confidence_loss']
        return loss

    @torch.no_grad()
    def _make_synthetic_anomaly_batch(self, input_data: torch.Tensor):
        x = input_data.detach().clone()
        B, L, C = x.shape
        device = x.device
        mask = torch.zeros(B, L, device=device, dtype=torch.float32)
        channel_mask = torch.zeros(B, C, device=device, dtype=torch.float32)

        min_len = int(getattr(self.config, "synthetic_min_len", 4) or 4)
        max_len = int(getattr(self.config, "synthetic_max_len", 20) or 20)
        min_len = min(max(1, min_len), max(1, L))
        max_len = min(max(min_len, max_len), max(1, L))
        use_rca_roots = bool(getattr(self.config, "use_synthetic_rca_loss", False))
        min_roots = int(getattr(self.config, "synthetic_rca_min_roots", 1) or 1)
        max_roots = int(getattr(self.config, "synthetic_rca_max_roots", 3) or 3)
        min_roots = min(max(1, min_roots), C)
        max_roots = min(max(min_roots, max_roots), C)

        for b in range(B):
            n_segments = int(torch.randint(1, 3, (1,), device=device).item())
            fixed_root_channels = None
            if use_rca_roots:
                n_roots = int(torch.randint(min_roots, max_roots + 1, (1,), device=device).item())
                fixed_root_channels = torch.randperm(C, device=device)[:n_roots]
            for _ in range(n_segments):
                seg_len = int(torch.randint(min_len, max_len + 1, (1,), device=device).item())
                start_hi = max(1, L - seg_len + 1)
                start = int(torch.randint(0, start_hi, (1,), device=device).item())
                end = min(L, start + seg_len)

                if use_rca_roots:
                    ch = fixed_root_channels
                    n_channels = int(ch.numel())
                else:
                    frac = float(torch.empty((), device=device).uniform_(0.10, 0.35).item())
                    n_channels = min(C, max(1, int(round(C * frac))))
                    ch = torch.randperm(C, device=device)[:n_channels]
                typ = int(torch.randint(0, 7, (1,), device=device).item())
                scale = input_data[b, :, ch].std(dim=0).clamp_min(0.2)
                sign = torch.where(
                    torch.rand(n_channels, device=device) < 0.5,
                    -torch.ones(n_channels, device=device),
                    torch.ones(n_channels, device=device),
                )
                amp = sign * scale * torch.empty(n_channels, device=device).uniform_(1.5, 4.0)

                if typ == 0:
                    x[b, start:end, ch] = x[b, start:end, ch] + amp
                elif typ == 1:
                    ramp = torch.linspace(0.0, 1.0, end - start, device=device).unsqueeze(-1)
                    x[b, start:end, ch] = x[b, start:end, ch] + ramp * amp
                elif typ == 2:
                    x[b, start:end, ch] = x[b, start:start + 1, ch].expand(end - start, -1)
                elif typ == 3:
                    x[b, start:end, ch] = 0.0
                elif typ == 4:
                    factor = torch.empty(n_channels, device=device).uniform_(0.3, 2.5)
                    x[b, start:end, ch] = x[b, start:end, ch] * factor
                elif typ == 5 and end - start > 1:
                    perm = torch.randperm(end - start, device=device)
                    x[b, start:end, ch] = x[b, start:end, ch][perm]
                else:
                    spike_count = max(1, min(end - start, seg_len // 4))
                    local_idx = torch.randperm(end - start, device=device)[:spike_count] + start
                    x[b, local_idx[:, None], ch] = x[b, local_idx[:, None], ch] + amp

                mask[b, start:end] = 1.0
                channel_mask[b, ch] = 1.0

        return x, mask, channel_mask

    @torch.no_grad()
    def _make_source_effect_synthetic_batch(self, input_data: torch.Tensor):
        x = input_data.detach().clone()
        B, L, C = x.shape
        device = x.device
        event_mask = torch.zeros(B, L, device=device, dtype=torch.float32)
        source_channel_mask = torch.zeros(B, C, device=device, dtype=torch.float32)
        effect_channel_mask = torch.zeros(B, C, device=device, dtype=torch.float32)

        min_len = int(getattr(self.config, "source_effect_min_len", 8) or 8)
        max_len = int(getattr(self.config, "source_effect_max_len", 30) or 30)
        min_len = min(max(1, min_len), max(1, L))
        max_len = min(max(min_len, max_len), max(1, L))
        min_roots = int(getattr(self.config, "source_effect_min_roots", 1) or 1)
        max_roots = int(getattr(self.config, "source_effect_max_roots", 2) or 2)
        min_roots = min(max(1, min_roots), C)
        max_roots = min(max(min_roots, max_roots), C)
        neighbor_topk = min(max(0, int(getattr(self.config, "source_effect_neighbor_topk", 3) or 0)), C)
        effect_strength = float(getattr(self.config, "source_effect_strength", 0.35) or 0.35)
        delay_max = max(0, int(getattr(self.config, "source_effect_delay_max", 6) or 0))
        prior = getattr(self, "_channel_corr_prior", None)
        if prior is not None:
            prior = np.asarray(prior, dtype=np.float32)
            if prior.shape != (C, C):
                prior = None

        for b in range(B):
            n_roots = int(torch.randint(min_roots, max_roots + 1, (1,), device=device).item())
            roots = torch.randperm(C, device=device)[:n_roots]
            roots_cpu = [int(v) for v in roots.detach().cpu().tolist()]

            seg_len = int(torch.randint(min_len, max_len + 1, (1,), device=device).item())
            start_hi = max(1, L - seg_len + 1)
            start = int(torch.randint(0, start_hi, (1,), device=device).item())
            end = min(L, start + seg_len)

            scale = input_data[b, :, roots].std(dim=0).clamp_min(0.2)
            sign = torch.where(
                torch.rand(n_roots, device=device) < 0.5,
                -torch.ones(n_roots, device=device),
                torch.ones(n_roots, device=device),
            )
            amp = sign * scale * torch.empty(n_roots, device=device).uniform_(1.5, 4.0)
            typ = int(torch.randint(0, 4, (1,), device=device).item())
            if typ == 0:
                x[b, start:end, roots] = x[b, start:end, roots] + amp
            elif typ == 1:
                ramp = torch.linspace(0.0, 1.0, end - start, device=device).unsqueeze(-1)
                x[b, start:end, roots] = x[b, start:end, roots] + ramp * amp
            elif typ == 2:
                factor = torch.empty(n_roots, device=device).uniform_(0.3, 2.5)
                x[b, start:end, roots] = x[b, start:end, roots] * factor
            else:
                spike_count = max(1, min(end - start, seg_len // 4))
                local_idx = torch.randperm(end - start, device=device)[:spike_count] + start
                x[b, local_idx[:, None], roots] = x[b, local_idx[:, None], roots] + amp

            neighbors = []
            if prior is not None and neighbor_topk > 0:
                scores = prior[roots_cpu].max(axis=0)
                scores[roots_cpu] = 0.0
                top_idx = np.argsort(-scores)[:neighbor_topk]
                neighbors = [int(idx) for idx in top_idx if scores[idx] > 0]
            if not neighbors and neighbor_topk > 0:
                candidates = [idx for idx in range(C) if idx not in set(roots_cpu)]
                if candidates:
                    perm = torch.randperm(len(candidates), device=device)[:neighbor_topk].detach().cpu().tolist()
                    neighbors = [candidates[int(idx)] for idx in perm]

            if neighbors:
                effects = torch.as_tensor(neighbors, device=device, dtype=torch.long)
                max_delay = min(delay_max, max(0, end - start - 1))
                delay = int(torch.randint(1, max_delay + 2, (1,), device=device).item()) if max_delay > 0 else 0
                eff_start = min(end, start + delay)
                if eff_start < end:
                    eff_scale = input_data[b, :, effects].std(dim=0).clamp_min(0.2)
                    eff_sign = torch.where(
                        torch.rand(effects.numel(), device=device) < 0.5,
                        -torch.ones(effects.numel(), device=device),
                        torch.ones(effects.numel(), device=device),
                    )
                    eff_amp = eff_sign * eff_scale * torch.empty(effects.numel(), device=device).uniform_(0.8, 2.0)
                    eff_amp = eff_amp * effect_strength
                    eff_ramp = torch.linspace(0.0, 1.0, end - eff_start, device=device).unsqueeze(-1)
                    x[b, eff_start:end, effects] = x[b, eff_start:end, effects] + eff_ramp * eff_amp
                    effect_channel_mask[b, effects] = 1.0

            event_mask[b, start:end] = 1.0
            source_channel_mask[b, roots] = 1.0
            effect_channel_mask[b, roots] = 0.0

        return x, event_mask, source_channel_mask, effect_channel_mask

    def _synthetic_anomaly_aux_loss(self, input_data, normal_aux_losses, batch_idx=None):
        use_aux = bool(getattr(self.config, "use_synthetic_anomaly_aux", False))
        use_rca = bool(getattr(self.config, "use_synthetic_rca_loss", False))
        if not use_aux and not use_rca:
            return input_data.new_tensor(0.0)
        interval = int(getattr(self.config, "synthetic_aux_interval", 1) or 1)
        if batch_idx is not None and interval > 1 and batch_idx % interval != 0:
            return input_data.new_tensor(0.0)
        lambda_synth = float(getattr(self.config, "lambda_synthetic_anomaly", 0.0) or 0.0)
        lambda_rca = float(getattr(self.config, "lambda_synthetic_rca", 0.0) or 0.0)
        if (lambda_synth <= 0 and lambda_rca <= 0) or not normal_aux_losses:
            return input_data.new_tensor(0.0)
        normal_logits = normal_aux_losses.get("synthetic_logits")
        normal_rca_logits = normal_aux_losses.get("synthetic_rca_logits")

        synth_data, synth_mask, synth_channel_mask = self._make_synthetic_anomaly_batch(input_data)
        synth_rec, _, _, _, _, synth_aux, _ = self.model(synth_data)
        synth_logits = synth_aux.get("synthetic_logits") if synth_aux else None
        synth_rca_logits = synth_aux.get("synthetic_rca_logits") if synth_aux else None
        total_loss = input_data.new_tensor(0.0)

        if use_aux and lambda_synth > 0 and normal_logits is not None and synth_logits is not None:
            normal_target = torch.zeros_like(normal_logits)
            normal_loss = F.binary_cross_entropy_with_logits(normal_logits, normal_target)

            pos = synth_mask.sum().clamp_min(1.0)
            neg = (synth_mask.numel() - synth_mask.sum()).clamp_min(1.0)
            pos_weight = (neg / pos).clamp(1.0, 20.0)
            synth_loss = F.binary_cross_entropy_with_logits(
                synth_logits, synth_mask, pos_weight=pos_weight,
            )
            total_loss = total_loss + lambda_synth * (synth_loss + 0.25 * normal_loss)

        if use_rca and lambda_rca > 0:
            bce_weight = float(getattr(self.config, "synthetic_rca_bce_weight", 1.0) or 1.0)
            rank_weight = float(getattr(self.config, "synthetic_rca_rank_weight", 1.0) or 1.0)
            rca_loss = input_data.new_tensor(0.0)
            if synth_rca_logits is not None:
                pos = synth_channel_mask.sum().clamp_min(1.0)
                neg = (synth_channel_mask.numel() - synth_channel_mask.sum()).clamp_min(1.0)
                pos_weight = (neg / pos).clamp(1.0, 20.0)
                bce_loss = F.binary_cross_entropy_with_logits(
                    synth_rca_logits,
                    synth_channel_mask,
                    pos_weight=pos_weight,
                )
                if normal_rca_logits is not None:
                    normal_rca_target = torch.zeros_like(normal_rca_logits)
                    bce_loss = bce_loss + 0.25 * F.binary_cross_entropy_with_logits(
                        normal_rca_logits,
                        normal_rca_target,
                    )
                rca_loss = rca_loss + bce_weight * bce_loss
                rca_loss = rca_loss + rank_weight * self._synthetic_rca_ranking_loss(
                    torch.sigmoid(synth_rca_logits),
                    synth_channel_mask,
                )
            else:
                channel_err = None
                if synth_aux:
                    channel_err = synth_aux.get("channel_mechanism_error")
                if channel_err is None:
                    channel_err = F.l1_loss(synth_rec, synth_data, reduction="none")
                channel_scores = channel_err.mean(dim=1)
                rca_loss = rca_loss + self._synthetic_rca_ranking_loss(
                    channel_scores,
                    synth_channel_mask,
                )
            total_loss = total_loss + lambda_rca * rca_loss

        return total_loss

    def _source_effect_ranking_loss(self, channel_scores, source_mask, effect_mask):
        margin = float(getattr(self.config, "source_effect_margin", 0.2) or 0.2)
        losses = []
        for b in range(channel_scores.shape[0]):
            roots = source_mask[b] > 0.5
            non_roots = ~roots
            if roots.sum() == 0 or non_roots.sum() == 0:
                continue
            pos_score = channel_scores[b, roots].mean()
            neg_scores = channel_scores[b, non_roots]
            k = min(5, neg_scores.numel())
            hard_neg = torch.topk(neg_scores, k=k).values.mean()
            losses.append(F.relu(channel_scores.new_tensor(margin) + hard_neg - pos_score))

            effects = (effect_mask[b] > 0.5) & non_roots
            if effects.sum() > 0:
                effect_scores = channel_scores[b, effects]
                k_eff = min(3, effect_scores.numel())
                hard_effect = torch.topk(effect_scores, k=k_eff).values.mean()
                losses.append(F.relu(channel_scores.new_tensor(margin) + hard_effect - pos_score))
        if not losses:
            return channel_scores.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _weighted_channel_bce(self, probs, target):
        probs = probs.clamp(1e-5, 1.0 - 1e-5)
        pos = target.sum().clamp_min(1.0)
        neg = (target.numel() - target.sum()).clamp_min(1.0)
        pos_weight = (neg / pos).clamp(1.0, 20.0)
        loss = -(pos_weight * target * torch.log(probs) + (1.0 - target) * torch.log(1.0 - probs))
        return loss.mean()

    def _channel_masked_modeling_loss(self, input_data, batch_idx=None):
        if not bool(getattr(self.config, "use_channel_masked_modeling", False)):
            return input_data.new_tensor(0.0)
        lambda_channel_masked = float(getattr(self.config, "lambda_channel_masked", 0.0) or 0.0)
        if lambda_channel_masked <= 0:
            return input_data.new_tensor(0.0)
        interval = int(getattr(self.config, "channel_mask_interval", 8) or 8)
        if batch_idx is not None and interval > 1 and batch_idx % interval != 0:
            return input_data.new_tensor(0.0)

        B, L, C = input_data.shape
        ratio = float(getattr(self.config, "channel_mask_ratio", 0.15) or 0.15)
        min_channels = int(getattr(self.config, "channel_mask_min_channels", 1) or 1)
        n_mask = min(C, max(min_channels, int(round(C * ratio))))
        if n_mask <= 0:
            return input_data.new_tensor(0.0)

        masked = input_data.detach().clone()
        channel_mask = torch.zeros(B, C, device=input_data.device, dtype=input_data.dtype)
        mask_value = str(getattr(self.config, "channel_mask_value", "zero") or "zero").lower()
        for b in range(B):
            channels = torch.randperm(C, device=input_data.device)[:n_mask]
            channel_mask[b, channels] = 1.0
            if mask_value == "mean":
                fill = input_data[b].mean(dim=0, keepdim=True)[:, channels]
                masked[b, :, channels] = fill.expand(L, -1)
            else:
                masked[b, :, channels] = 0.0

        rec, _, _, _, _, _, _ = self.model(masked)
        mask = channel_mask[:, None, :]
        denom = mask.sum().clamp_min(1.0) * max(1, L)
        masked_loss = F.smooth_l1_loss(
            rec * mask,
            input_data * mask,
            reduction="sum",
        ) / denom
        return lambda_channel_masked * masked_loss

    def _source_effect_synthetic_loss(self, input_data, batch_idx=None):
        if not bool(getattr(self.config, "use_source_effect_synthetic", False)):
            return input_data.new_tensor(0.0)
        interval = int(getattr(self.config, "source_effect_interval", 4) or 4)
        if batch_idx is not None and interval > 1 and batch_idx % interval != 0:
            return input_data.new_tensor(0.0)
        lambda_source_effect = float(getattr(self.config, "lambda_source_effect", 0.0) or 0.0)
        if lambda_source_effect <= 0:
            return input_data.new_tensor(0.0)

        synth_data, event_mask, source_mask, effect_mask = self._make_source_effect_synthetic_batch(input_data)
        synth_rec, _, _, _, _, synth_aux, _ = self.model(synth_data)
        if not synth_aux:
            return input_data.new_tensor(0.0)

        mask_sum = event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        gate_prob = synth_aux.get("source_gate_prob")
        gate_channel_scores = None
        if gate_prob is not None:
            gate_channel_scores = (gate_prob * event_mask.unsqueeze(-1)).sum(dim=1) / mask_sum

        source_score = synth_aux.get("source_gate_score")
        if source_score is not None:
            channel_scores = (source_score * event_mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        else:
            channel_err = F.l1_loss(synth_rec, synth_data, reduction="none")
            channel_scores = (channel_err * event_mask.unsqueeze(-1)).sum(dim=1) / mask_sum

        bce_weight = float(getattr(self.config, "source_effect_bce_weight", 1.0) or 1.0)
        rank_weight = float(getattr(self.config, "source_effect_rank_weight", 1.0) or 1.0)
        effect_rank_weight = float(getattr(self.config, "source_effect_effect_rank_weight", 0.5) or 0.5)

        total = input_data.new_tensor(0.0)
        if gate_channel_scores is not None:
            total = total + bce_weight * self._weighted_channel_bce(gate_channel_scores, source_mask)
        if rank_weight > 0:
            total = total + rank_weight * self._synthetic_rca_ranking_loss(channel_scores, source_mask)
        if effect_rank_weight > 0:
            total = total + effect_rank_weight * self._source_effect_ranking_loss(
                channel_scores,
                source_mask,
                effect_mask,
            )
        return lambda_source_effect * total

    def _synthetic_rca_ranking_loss(self, channel_scores, channel_mask):
        margin = float(getattr(self.config, "synthetic_rca_margin", 0.2) or 0.2)
        hard_topk = int(getattr(self.config, "synthetic_rca_topk", 5) or 5)
        losses = []
        for b in range(channel_scores.shape[0]):
            roots = channel_mask[b] > 0.5
            non_roots = ~roots
            if roots.sum() == 0 or non_roots.sum() == 0:
                continue
            pos_score = channel_scores[b, roots].mean()
            neg_scores = channel_scores[b, non_roots]
            k = min(max(1, hard_topk), neg_scores.numel())
            hard_neg = torch.topk(neg_scores, k=k).values.mean()
            losses.append(F.relu(channel_scores.new_tensor(margin) + hard_neg - pos_score))
        if not losses:
            return channel_scores.new_tensor(0.0)
        return torch.stack(losses).mean()

    @torch.no_grad()
    def _fit_score_channel_stats(self, train_data: pd.DataFrame):
        if train_data is None or self.model is None:
            return
        raw_model = self._get_raw_model()
        if not hasattr(raw_model, "set_score_channel_stats"):
            return

        print("\n  [ScoreNorm] Fitting channel-wise reconstruction error stats...")
        if self._should_load_best_checkpoint():
            raw_model.load_state_dict(self.early_stopping.check_point)
        self.model.to(self.device)
        self.model.eval()

        scaled_data = self._transform_input_frame(train_data)
        loader = anomaly_detection_data_provider(
            scaled_data,
            batch_size=min(self.config.batch_size, 64),
            win_size=self.config.win_size,
            step=1,
            mode="test",
            num_workers=0,
        )

        channel_errors = []
        for input_data, _ in loader:
            input_data = input_data.float().to(self.device)
            rec, _, _, _, _, _, _ = self.model(input_data)
            err = torch.abs(rec - input_data).mean(dim=1)
            channel_errors.append(err.cpu().numpy())

        if not channel_errors:
            return

        errors = np.concatenate(channel_errors, axis=0)
        center = np.median(errors, axis=0)
        q25 = np.percentile(errors, 25, axis=0)
        q75 = np.percentile(errors, 75, axis=0)
        scale = q75 - q25
        eps = float(getattr(self.config, "score_channel_norm_eps", 1e-6) or 1e-6)
        fallback = np.maximum(np.abs(center), eps)
        scale = np.where(scale > eps, scale, fallback)
        raw_model.set_score_channel_stats(center, scale)
        print(
            "  [ScoreNorm] center median="
            f"{float(np.median(center)):.6f}, scale median={float(np.median(scale)):.6f}"
        )

    @torch.no_grad()
    def _fit_graph_shift_stats(self, train_data: pd.DataFrame):
        if train_data is None or self.model is None:
            return
        raw_model = self._get_raw_model()
        if not hasattr(raw_model, "set_graph_shift_stats"):
            return

        print("\n  [GraphShift] Fitting normal channel-graph statistics...")
        if self._should_load_best_checkpoint():
            raw_model.load_state_dict(self.early_stopping.check_point)
        self.model.to(self.device)
        self.model.eval()

        scaled_data = self._transform_input_frame(train_data)

        def _loader():
            return anomaly_detection_data_provider(
                scaled_data,
                batch_size=min(self.config.batch_size, 64),
                win_size=self.config.win_size,
                step=1,
                mode="test",
                num_workers=0,
            )

        graph_sum = None
        n_graphs = 0
        for input_data, _ in _loader():
            input_data = input_data.float().to(self.device)
            _, A_adaptive, _, _, _, _, _ = self.model(input_data)
            A_cpu = A_adaptive.detach().cpu()
            graph_sum = A_cpu.sum(dim=0) if graph_sum is None else graph_sum + A_cpu.sum(dim=0)
            n_graphs += A_cpu.shape[0]

        if graph_sum is None or n_graphs == 0:
            return

        center = graph_sum / float(n_graphs)
        center_dev = center.to(self.device)
        shifts = []
        for input_data, _ in _loader():
            input_data = input_data.float().to(self.device)
            _, A_adaptive, _, _, _, _, _ = self.model(input_data)
            shift = (A_adaptive - center_dev).abs().mean(dim=(1, 2))
            shifts.append(shift.detach().cpu().numpy())

        if not shifts:
            return

        shifts = np.concatenate(shifts, axis=0)
        score_center = float(np.median(shifts))
        q25 = float(np.percentile(shifts, 25))
        q75 = float(np.percentile(shifts, 75))
        eps = float(getattr(self.config, "graph_shift_score_eps", 1e-6) or 1e-6)
        score_scale = max(q75 - q25, float(np.std(shifts)), abs(score_center), eps)
        raw_model.set_graph_shift_stats(center.numpy(), score_center, score_scale)
        print(
            "  [GraphShift] shift median="
            f"{score_center:.6f}, scale={score_scale:.6f}, graphs={n_graphs}"
        )

    @torch.no_grad()
    def _fit_causal_score_stats(self, train_data: pd.DataFrame):
        if train_data is None or self.model is None:
            return
        raw_model = self._get_raw_model()
        if not hasattr(raw_model, "set_causal_score_stats"):
            return

        print("\n  [CausalLag] Fitting lagged-mechanism score statistics...")
        if self._should_load_best_checkpoint():
            raw_model.load_state_dict(self.early_stopping.check_point)
        self.model.to(self.device)
        self.model.eval()

        scaled_data = self._transform_input_frame(train_data)
        loader = anomaly_detection_data_provider(
            scaled_data,
            batch_size=min(self.config.batch_size, 64),
            win_size=self.config.win_size,
            step=1,
            mode="test",
            num_workers=0,
        )

        scores = []
        for input_data, _ in loader:
            input_data = input_data.float().to(self.device)
            _, _, _, _, _, aux_losses, _ = self.model(input_data)
            causal_score = aux_losses.get("causal_score") if aux_losses else None
            if causal_score is not None:
                scores.append(causal_score.detach().cpu().numpy().reshape(-1))

        if not scores:
            return

        scores = np.concatenate(scores, axis=0)
        score_center = float(np.median(scores))
        q25 = float(np.percentile(scores, 25))
        q75 = float(np.percentile(scores, 75))
        eps = float(getattr(self.config, "causal_score_eps", 1e-6) or 1e-6)
        score_scale = max(q75 - q25, float(np.std(scores)), abs(score_center), eps)
        raw_model.set_causal_score_stats(score_center, score_scale)
        print(
            "  [CausalLag] score median="
            f"{score_center:.6f}, scale={score_scale:.6f}"
        )

    # ======================== 训练（单卡）=======================
    @torch.no_grad()
    def _fit_channel_mechanism_score_stats(self, train_data: pd.DataFrame):
        if train_data is None or self.model is None:
            return
        raw_model = self._get_raw_model()
        if not hasattr(raw_model, "set_channel_mechanism_score_stats"):
            return

        print("\n  [ChannelMechanism] Fitting normal mechanism-violation score statistics...")
        if self._should_load_best_checkpoint():
            raw_model.load_state_dict(self.early_stopping.check_point)
        self.model.to(self.device)
        self.model.eval()

        scaled_data = self._transform_input_frame(train_data)
        loader = anomaly_detection_data_provider(
            scaled_data,
            batch_size=min(self.config.batch_size, 64),
            win_size=self.config.win_size,
            step=1,
            mode="test",
            num_workers=0,
        )

        scores = []
        for input_data, _ in loader:
            input_data = input_data.float().to(self.device)
            _, _, _, _, _, aux_losses, _ = self.model(input_data)
            mechanism_score = aux_losses.get("channel_mechanism_score") if aux_losses else None
            if mechanism_score is not None:
                scores.append(mechanism_score.detach().cpu().numpy().reshape(-1))

        if not scores:
            return

        scores = np.concatenate(scores, axis=0)
        score_center = float(np.median(scores))
        q25 = float(np.percentile(scores, 25))
        q75 = float(np.percentile(scores, 75))
        eps = float(getattr(self.config, "channel_mechanism_score_eps", 1e-6) or 1e-6)
        score_scale = max(q75 - q25, float(np.std(scores)), abs(score_center), eps)
        raw_model.set_channel_mechanism_score_stats(score_center, score_scale)
        print(
            "  [ChannelMechanism] score median="
            f"{score_center:.6f}, scale={score_scale:.6f}"
        )

    @torch.no_grad()
    def _fit_synthetic_score_stats(self, train_data: pd.DataFrame):
        if train_data is None or self.model is None:
            return
        raw_model = self._get_raw_model()
        if not hasattr(raw_model, "set_synthetic_score_stats"):
            return

        print("\n  [SynthAux] Fitting synthetic-head score statistics...")
        if self._should_load_best_checkpoint():
            raw_model.load_state_dict(self.early_stopping.check_point)
        self.model.to(self.device)
        self.model.eval()

        scaled_data = self._transform_input_frame(train_data)
        loader = anomaly_detection_data_provider(
            scaled_data,
            batch_size=min(self.config.batch_size, 64),
            win_size=self.config.win_size,
            step=1,
            mode="test",
            num_workers=0,
        )

        scores = []
        for input_data, _ in loader:
            input_data = input_data.float().to(self.device)
            _, _, _, _, _, aux_losses, _ = self.model(input_data)
            logits = aux_losses.get("synthetic_logits") if aux_losses else None
            if logits is not None:
                scores.append(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1))

        if not scores:
            return

        scores = np.concatenate(scores, axis=0)
        score_center = float(np.median(scores))
        q25 = float(np.percentile(scores, 25))
        q75 = float(np.percentile(scores, 75))
        eps = float(getattr(self.config, "synthetic_score_eps", 1e-6) or 1e-6)
        score_scale = max(q75 - q25, float(np.std(scores)), abs(score_center), eps)
        raw_model.set_synthetic_score_stats(score_center, score_scale)
        print(
            "  [SynthAux] score median="
            f"{score_center:.6f}, scale={score_scale:.6f}"
        )

    def _destroy_model_and_clean_cuda(self):
        import gc
        if hasattr(self, 'model') and self.model is not None:
            for p in self.model.parameters():
                if p.device.type == 'cuda':
                    p.data = p.data.cpu()
            del self.model
            self.model = None
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            del self.optimizer
            self.optimizer = None
        if hasattr(self, 'train_loader') and self.train_loader is not None:
            del self.train_loader
            self.train_loader = None
        if hasattr(self, 'valid_loader') and self.valid_loader is not None:
            del self.valid_loader
            self.valid_loader = None
        if hasattr(self, 'thre_loader') and self.thre_loader is not None:
            del self.thre_loader
            self.thre_loader = None
        if hasattr(self, 'early_stopping') and self.early_stopping is not None:
            del self.early_stopping
            self.early_stopping = None
        gc.collect()
        for i in range(torch.cuda.device_count()):
            try:
                torch.cuda.synchronize(f'cuda:{i}')
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats(i)
            except Exception:
                pass
        gc.collect()
        for _ in range(3):
            torch.cuda.empty_cache()

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        self.config.input_c = train_data.shape[1]
        self.config.output_c = train_data.shape[1]

        self._destroy_model_and_clean_cuda()

        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self._fit_input_preprocessor(train_data_value)

        self._train_raw = train_data_value.copy()

        train_scaled = self._transform_input_frame(train_data_value)
        valid_scaled = self._transform_input_frame(valid_data)

        if self.multi_gpu_requested:
            print(
                f"\n  [INFO] n_gpus={self.config_n_gpu} was requested, "
                "but DDP has been removed; using the single-GPU training path."
            )

        self._single_gpu_train(train_scaled, valid_scaled)

        if self._should_load_best_checkpoint():
            self._get_raw_model().load_state_dict(self.early_stopping.check_point)

        self.trained = True

        if getattr(self.config, "use_score_channel_normalization", False):
            self._fit_score_channel_stats(self._train_raw)

        if getattr(self.config, "use_graph_shift_score", False):
            self._fit_graph_shift_stats(self._train_raw)

        if getattr(self.config, "use_causal_score", False):
            self._fit_causal_score_stats(self._train_raw)

        if getattr(self.config, "use_channel_mechanism_score", False):
            self._fit_channel_mechanism_score_stats(self._train_raw)

        if getattr(self.config, "use_synthetic_score", False):
            self._fit_synthetic_score_stats(self._train_raw)

        # ★ v10 fix: 训练集缓存分数
        if self._train_raw is not None:
            print(f"\n  [INFO] Computing training reconstruction scores for threshold...")
            self._train_anomaly_scores, _ = self.detect_score(self._train_raw)
            print(f"  [INFO] Train scores: min={self._train_anomaly_scores.min():.6f}, "
                  f"max={self._train_anomaly_scores.max():.6f}, "
                  f"mean={self._train_anomaly_scores.mean():.6f}")

        # ★ P0-2: 同时缓存验证集分数用于阈值校准
        print(f"\n  [INFO] Computing validation reconstruction scores for POT calibration...")
        self._val_anomaly_scores, _ = self.detect_score(valid_data)
        print(f"  [INFO] Val scores: min={self._val_anomaly_scores.min():.6f}, "
              f"max={self._val_anomaly_scores.max():.6f}, "
              f"mean={self._val_anomaly_scores.mean():.6f}")

        # ★ P0-2: POT 阈值估计 + 验证集校准
        if self._train_anomaly_scores is not None:
            pot_threshold = self._pot_estimator.estimate(self._train_anomaly_scores)
            print(f"\n  [POT] Baseline threshold: {pot_threshold:.6f}")

            if self._val_anomaly_scores is not None:
                # 用验证集校准（目标异常比例取 anomaly_ratio 的中位数）
                target_ratio = np.median(self.config.anomaly_ratio) / 100.0
                calibrated = self._pot_estimator.calibrate(
                    self._val_anomaly_scores, target_ratio=target_ratio
                )
                print(f"  [POT] Calibrated threshold: {calibrated:.6f} "
                      f"(target_ratio={target_ratio*100:.1f}%)")

        # ★ P0-5: 设置可视化 hook
        if self.model is not None and bool(getattr(self.config, "enable_visualization_hooks", False)):
            self._vis_hook = VisualizationHook(self.model)
            self._vis_hook.register_hooks()
            print(f"\n  [VIS] Visualization hooks registered")

    def _single_gpu_train(self, train_df, val_df):
        """单 GPU 训练（v11.4 版）"""
        from ts_benchmark.baselines.utils import anomaly_detection_data_provider as adp

        self.train_loader = adp(
            train_df, batch_size=self.config.batch_size,
            win_size=self.config.win_size, step=1, mode="train",
            num_workers=getattr(self.config, 'dataloader_num_workers', 0),
            prefetch_factor=getattr(self.config, 'dataloader_prefetch_factor', 2),
        )
        self.valid_loader = adp(
            val_df, batch_size=self.config.batch_size,
            win_size=self.config.win_size, step=1, mode="val",
            num_workers=getattr(self.config, 'dataloader_num_workers', 0),
            prefetch_factor=getattr(self.config, 'dataloader_prefetch_factor', 2),
        )

        self.model = SparseGCN(
            win_size=self.config.win_size,
            enc_in=self.config.input_c,
            c_out=self.config.output_c,
            dropout=self.config.dropout,
            n_heads=self.config.n_heads,
            d_model=self.config.d_model,
            e_layers=self.config.e_layers,
            patch_size=self.config.patch_size,
            channel=self.config.input_c,
            topk=self.config.topk,
            sparse_topk=self.config.sparse_topk,
            use_channel_graph=getattr(self.config, "use_channel_graph", True),
            use_temporal_graph=getattr(self.config, "use_temporal_graph", True),
            use_dynamic_temporal_graph=getattr(self.config, "use_dynamic_temporal_graph", False),
            dynamic_temporal_residual_init=getattr(self.config, "dynamic_temporal_residual_init", 0.1),
            dynamic_temporal_topk=getattr(self.config, "dynamic_temporal_topk", None),
            dynamic_temporal_gate_mode=getattr(self.config, "dynamic_temporal_gate_mode", "global"),
            use_vq_bypass=getattr(self.config, "use_vq_bypass", True),
            use_multi_scale_scorer=getattr(self.config, "use_multi_scale_scorer", False),
            use_direct_vq_score=getattr(self.config, "use_direct_vq_score", False),
            vq_score_weight=getattr(self.config, "vq_score_weight", 0.3),
            score_topk_k=getattr(self.config, "score_topk_k", None),
            use_synthetic_anomaly_head=getattr(self.config, "use_synthetic_anomaly_head", False),
            use_synthetic_rca_head=getattr(self.config, "use_synthetic_rca_head", False),
            use_synthetic_score=getattr(self.config, "use_synthetic_score", False),
            synthetic_score_weight=getattr(self.config, "synthetic_score_weight", 0.1),
            synthetic_score_eps=getattr(self.config, "synthetic_score_eps", 1e-6),
            use_parallel_graph_fusion=getattr(self.config, "use_parallel_graph_fusion", False),
            graph_fusion_gate_mode=getattr(self.config, "graph_fusion_gate_mode", "sample"),
            graph_fusion_strategy=getattr(self.config, "graph_fusion_strategy", "parallel"),
            graph_fusion_residual_init=getattr(self.config, "graph_fusion_residual_init", 0.1),
            use_graph_shift_score=getattr(self.config, "use_graph_shift_score", False),
            graph_shift_score_weight=getattr(self.config, "graph_shift_score_weight", 0.1),
            graph_shift_score_eps=getattr(self.config, "graph_shift_score_eps", 1e-6),
            use_lagged_causal_graph=getattr(self.config, "use_lagged_causal_graph", False),
            causal_lags=getattr(self.config, "causal_lags", [1, 2, 4]),
            causal_topk=getattr(self.config, "causal_topk", 5),
            causal_detach_backbone=getattr(self.config, "causal_detach_backbone", True),
            use_causal_score=getattr(self.config, "use_causal_score", False),
            causal_score_weight=getattr(self.config, "causal_score_weight", 0.1),
            causal_score_eps=getattr(self.config, "causal_score_eps", 1e-6),
            causal_score_mode=getattr(self.config, "causal_score_mode", "residual"),
            causal_score_tail=getattr(self.config, "causal_score_tail", "upper"),
            use_temporal_graph_regularization=getattr(self.config, "use_temporal_graph_regularization", False),
            use_score_channel_normalization=getattr(self.config, "use_score_channel_normalization", False),
            score_channel_norm_mode=getattr(self.config, "score_channel_norm_mode", "robust_z"),
            score_channel_norm_eps=getattr(self.config, "score_channel_norm_eps", 1e-6),
            use_channel_corr_prior=getattr(self.config, "use_channel_corr_prior", False),
            channel_corr_prior_weight=getattr(self.config, "channel_corr_prior_weight", 0.0),
            channel_corr_prior_bias=getattr(self.config, "channel_corr_prior_bias", 0.0),
            lambda_channel_prior_align=getattr(self.config, "lambda_channel_prior_align", 0.0),
            lambda_channel_mechanism=getattr(self.config, "lambda_channel_mechanism", 0.0),
            use_channel_mechanism_score=getattr(self.config, "use_channel_mechanism_score", False),
            channel_mechanism_score_weight=getattr(self.config, "channel_mechanism_score_weight", 0.1),
            channel_mechanism_score_eps=getattr(self.config, "channel_mechanism_score_eps", 1e-6),
            use_mechanism_coupled_decoder=getattr(self.config, "use_mechanism_coupled_decoder", False),
            mechanism_coupling_init=getattr(self.config, "mechanism_coupling_init", 0.15),
            use_mechanism_predictive_head=getattr(self.config, "use_mechanism_predictive_head", False),
            mechanism_predictive_blend_init=getattr(self.config, "mechanism_predictive_blend_init", 0.30),
            use_source_gate=getattr(self.config, "use_source_gate", False),
            source_gate_init=getattr(self.config, "source_gate_init", 0.20),
            use_state_aware_fusion=getattr(self.config, "use_state_aware_fusion", False),
            state_aware_num_states=getattr(self.config, "state_aware_num_states", 4),
            state_aware_graph_gate_init=getattr(self.config, "state_aware_graph_gate_init", 0.6),
            state_aware_residual_init=getattr(self.config, "state_aware_residual_init", 0.15),
        )
        self.model.to(self.device)

        if getattr(self.config, "use_channel_corr_prior", False):
            prior = self._build_channel_corr_prior(
                train_df,
                topk=getattr(self.config, "channel_corr_prior_topk", 5),
            )
            self._channel_corr_prior = prior
            self.model.set_channel_static_prior(prior)
            print(
                f"  [ChannelPrior] normal correlation prior set "
                f"(topk={getattr(self.config, 'channel_corr_prior_topk', 5)}, "
                f"weight={getattr(self.config, 'channel_corr_prior_weight', 0.0)}, "
                f"bias={getattr(self.config, 'channel_corr_prior_bias', 0.0)})"
            )

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        train_steps = len(self.train_loader)
        n_train = len(train_df)
        n_val = len(val_df)
        n_features = self.config.input_c

        self._print_training_header(
            n_train=n_train, n_val=n_val, n_features=n_features,
            train_steps=train_steps, total_params=total_params,
        )

        if not torch.cuda.is_available():
            print("  WARNING: CUDA NOT available - running on CPU")
            print()

        self.early_stopping = EarlyStopping(
            patience=self.config.patience, verbose=True, relative_delta=0.001,
        )

        channel_graph_params = []
        temporal_graph_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'channel_graph' in name:
                channel_graph_params.append(param)
            elif 'temporal_graph' in name:
                temporal_graph_params.append(param)
            else:
                other_params.append(param)

        weight_decay_v10 = getattr(self.config, 'weight_decay', 1e-4)
        channel_lr_scale = getattr(self.config, 'channel_graph_lr_scale', 0.1)
        temporal_lr_scale = getattr(self.config, 'temporal_graph_lr_scale', 0.1)

        param_groups = [
            {'params': other_params, 'lr': self.config.lr, 'param_names': ['other']},
        ]
        if channel_graph_params:
            param_groups.append({
                'params': channel_graph_params,
                'lr': self.config.lr * channel_lr_scale,
                'param_names': ['channel_graph'],
            })
        if temporal_graph_params:
            param_groups.append({
                'params': temporal_graph_params,
                'lr': self.config.lr * temporal_lr_scale,
                'param_names': ['temporal_graph'],
            })
        self.optimizer = optim.Adam(
            param_groups, lr=self.config.lr, weight_decay=weight_decay_v10,
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=self.config.warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.num_epochs - self.config.warmup_epochs),
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs],
        )

        train_start_time = time.time()
        REPORT_INTERVAL = 500

        def _freeze_except_vq(model, freeze=True):
            for name, p in model.named_parameters():
                if 'vq_bottleneck' not in name:
                    p.requires_grad = not freeze
                else:
                    p.requires_grad = True

        vq_cooldown_epochs = getattr(self.config, 'vq_cooldown_epochs', 10)
        lambda_vq = getattr(self.config, 'lambda_vq', 0.1)
        use_vq_bypass = getattr(self.config, 'use_vq_bypass', True)

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            epoch_losses = []
            self._current_train_epoch = epoch

            warmup_alpha = min(1.0, epoch / max(1, self.config.num_epochs * 0.1))
            self.model.set_warmup_progress(warmup_alpha)

            if use_vq_bypass and epoch == 0:
                _freeze_except_vq(self.model, freeze=True)
            elif use_vq_bypass and epoch == vq_cooldown_epochs:
                _freeze_except_vq(self.model, freeze=False)
                if hasattr(self.model, "_apply_architecture_switches"):
                    self.model._apply_architecture_switches()

            self.model.train()
            self.optimizer.zero_grad()

            for i, (input_data, _) in enumerate(self.train_loader):
                input_data = input_data.float().to(self.device, non_blocking=True)
                rec, _, _, _, _, aux_losses, _ = self.model(input_data)

                loss = self._reconstruction_loss(rec, input_data)
                loss = self._add_temporal_difference_loss(loss, rec, input_data)

                # ★ P0-1: lambda_causal_l1 → lambda_locality_l1
                if aux_losses and 'sparse_loss' in aux_losses:
                    loss = loss + self.config.lambda_locality_l1 * aux_losses['sparse_loss']
                loss = self._add_temporal_graph_regularization(loss, aux_losses)

                if use_vq_bypass and aux_losses and 'vq_loss' in aux_losses and epoch < vq_cooldown_epochs:
                    loss = aux_losses['vq_loss']
                else:
                    if use_vq_bypass and aux_losses and 'vq_loss' in aux_losses:
                        loss = loss + lambda_vq * aux_losses['vq_loss']
                    loss = loss + self._synthetic_anomaly_aux_loss(input_data, aux_losses, batch_idx=i)
                    loss = loss + self._source_effect_synthetic_loss(input_data, batch_idx=i)
                    loss = loss + self._channel_masked_modeling_loss(input_data, batch_idx=i)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_losses.append(loss.item())

                if (i + 1) % REPORT_INTERVAL == 0:
                    bar = self._print_progress_bar(i + 1, train_steps, 30)
                    gpu_str = _get_gpu_info(self.device)
                    print(f"  [batch] {bar}  Loss: {np.mean(epoch_losses[-REPORT_INTERVAL:]):.6f}  |  {gpu_str}")

            avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0
            epoch_time = time.time() - epoch_start
            val_loss = self.vali(self.valid_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            es_status = self.early_stopping(val_loss, self.model, epoch + 1)

            self._print_epoch_result(
                epoch=epoch + 1, epoch_time=epoch_time,
                avg_train_loss=avg_train_loss, val_loss=val_loss,
                lr=current_lr, total_epochs=self.config.num_epochs,
                es_status=es_status, es_counter=self.early_stopping.counter,
            )

            if self.early_stopping.early_stop:
                print(f"  ⏹ Early stopping at epoch {epoch + 1} (best: {self.early_stopping.best_epoch})")
                print()
                break

            scheduler.step()

        total_time = time.time() - train_start_time
        saved_path = self._save_best_params(total_params=total_params, total_time=total_time)

        self._save_train_history(total_params=total_params, total_time=total_time)

        print("─" * 70)
        print(f"  [OK] Training Complete!")
        print(f"     Best Val Loss: {self.early_stopping.val_loss_min:.6f}  @  Epoch {self.early_stopping.best_epoch}")
        print(f"     Total Time:    {_format_duration(total_time)}")
        print(f"     Device:        {_get_gpu_info(self.device)}")
        print(f"     Best params saved to: {saved_path}")
        print("─" * 70)
        print()

    def _single_gpu_train_reweight(
        self, train_df, val_df,
        anomaly_weight: float = 1.0,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
    ):
        from ts_benchmark.utils.data_processing import split_before
        from ts_benchmark.baselines.utils import anomaly_detection_data_provider as adp

        self.train_loader = adp(
            train_df, batch_size=self.config.batch_size,
            win_size=self.config.win_size, step=1, mode="train",
        )
        self.valid_loader = adp(
            val_df, batch_size=self.config.batch_size,
            win_size=self.config.win_size, step=1, mode="val",
        )

        self.model = SparseGCN(
            win_size=self.config.win_size,
            enc_in=self.config.input_c,
            c_out=self.config.output_c,
            dropout=self.config.dropout,
            n_heads=self.config.n_heads,
            d_model=self.config.d_model,
            e_layers=self.config.e_layers,
            patch_size=self.config.patch_size,
            channel=self.config.input_c,
            topk=self.config.topk,
            sparse_topk=self.config.sparse_topk,
            use_channel_graph=getattr(self.config, "use_channel_graph", True),
            use_temporal_graph=getattr(self.config, "use_temporal_graph", True),
            use_dynamic_temporal_graph=getattr(self.config, "use_dynamic_temporal_graph", False),
            dynamic_temporal_residual_init=getattr(self.config, "dynamic_temporal_residual_init", 0.1),
            dynamic_temporal_topk=getattr(self.config, "dynamic_temporal_topk", None),
            dynamic_temporal_gate_mode=getattr(self.config, "dynamic_temporal_gate_mode", "global"),
            use_vq_bypass=getattr(self.config, "use_vq_bypass", True),
            use_multi_scale_scorer=getattr(self.config, "use_multi_scale_scorer", False),
            use_direct_vq_score=getattr(self.config, "use_direct_vq_score", False),
            vq_score_weight=getattr(self.config, "vq_score_weight", 0.3),
            score_topk_k=getattr(self.config, "score_topk_k", None),
            use_synthetic_anomaly_head=getattr(self.config, "use_synthetic_anomaly_head", False),
            use_synthetic_rca_head=getattr(self.config, "use_synthetic_rca_head", False),
            use_synthetic_score=getattr(self.config, "use_synthetic_score", False),
            synthetic_score_weight=getattr(self.config, "synthetic_score_weight", 0.1),
            synthetic_score_eps=getattr(self.config, "synthetic_score_eps", 1e-6),
            use_parallel_graph_fusion=getattr(self.config, "use_parallel_graph_fusion", False),
            graph_fusion_gate_mode=getattr(self.config, "graph_fusion_gate_mode", "sample"),
            graph_fusion_strategy=getattr(self.config, "graph_fusion_strategy", "parallel"),
            graph_fusion_residual_init=getattr(self.config, "graph_fusion_residual_init", 0.1),
            use_graph_shift_score=getattr(self.config, "use_graph_shift_score", False),
            graph_shift_score_weight=getattr(self.config, "graph_shift_score_weight", 0.1),
            graph_shift_score_eps=getattr(self.config, "graph_shift_score_eps", 1e-6),
            use_lagged_causal_graph=getattr(self.config, "use_lagged_causal_graph", False),
            causal_lags=getattr(self.config, "causal_lags", [1, 2, 4]),
            causal_topk=getattr(self.config, "causal_topk", 5),
            causal_detach_backbone=getattr(self.config, "causal_detach_backbone", True),
            use_causal_score=getattr(self.config, "use_causal_score", False),
            causal_score_weight=getattr(self.config, "causal_score_weight", 0.1),
            causal_score_eps=getattr(self.config, "causal_score_eps", 1e-6),
            causal_score_mode=getattr(self.config, "causal_score_mode", "residual"),
            causal_score_tail=getattr(self.config, "causal_score_tail", "upper"),
            use_temporal_graph_regularization=getattr(self.config, "use_temporal_graph_regularization", False),
            use_score_channel_normalization=getattr(self.config, "use_score_channel_normalization", False),
            score_channel_norm_mode=getattr(self.config, "score_channel_norm_mode", "robust_z"),
            score_channel_norm_eps=getattr(self.config, "score_channel_norm_eps", 1e-6),
            lambda_channel_mechanism=getattr(self.config, "lambda_channel_mechanism", 0.0),
            use_channel_mechanism_score=getattr(self.config, "use_channel_mechanism_score", False),
            channel_mechanism_score_weight=getattr(self.config, "channel_mechanism_score_weight", 0.1),
            channel_mechanism_score_eps=getattr(self.config, "channel_mechanism_score_eps", 1e-6),
            use_mechanism_coupled_decoder=getattr(self.config, "use_mechanism_coupled_decoder", False),
            mechanism_coupling_init=getattr(self.config, "mechanism_coupling_init", 0.15),
            use_mechanism_predictive_head=getattr(self.config, "use_mechanism_predictive_head", False),
            mechanism_predictive_blend_init=getattr(self.config, "mechanism_predictive_blend_init", 0.30),
            use_source_gate=getattr(self.config, "use_source_gate", False),
            source_gate_init=getattr(self.config, "source_gate_init", 0.20),
            use_state_aware_fusion=getattr(self.config, "use_state_aware_fusion", False),
            state_aware_num_states=getattr(self.config, "state_aware_num_states", 4),
            state_aware_graph_gate_init=getattr(self.config, "state_aware_graph_gate_init", 0.6),
            state_aware_residual_init=getattr(self.config, "state_aware_residual_init", 0.15),
        )
        self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        train_steps = len(self.train_loader)
        n_train = len(train_df)
        n_val = len(val_df)
        n_features = self.config.input_c

        self._print_training_header(
            n_train=n_train, n_val=n_val, n_features=n_features,
            train_steps=train_steps, total_params=total_params,
        )

        self.early_stopping = EarlyStopping(
            patience=self.config.patience, verbose=True, relative_delta=0.001,
        )

        channel_graph_params = []
        temporal_graph_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if 'channel_graph' in name:
                channel_graph_params.append(param)
            elif 'temporal_graph' in name:
                temporal_graph_params.append(param)
            else:
                other_params.append(param)

        channel_lr_scale = getattr(self.config, 'channel_graph_lr_scale', 0.1)
        temporal_lr_scale = getattr(self.config, 'temporal_graph_lr_scale', 0.1)
        param_groups = [{'params': other_params, 'lr': self.config.lr}]
        if channel_graph_params:
            param_groups.append({
                'params': channel_graph_params,
                'lr': self.config.lr * channel_lr_scale,
            })
        if temporal_graph_params:
            param_groups.append({
                'params': temporal_graph_params,
                'lr': self.config.lr * temporal_lr_scale,
            })

        self.optimizer = optim.Adam(
            param_groups, lr=self.config.lr, weight_decay=1e-4,
        )

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0,
            total_iters=self.config.warmup_epochs,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.num_epochs - self.config.warmup_epochs),
            eta_min=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_epochs],
        )

        train_start_time = time.time()

        if use_focal:
            print(f"  [WARN] FocalLoss has been removed (P0). Falling back to weighted MSE.")
        print(f"\n  [REWEIGHT] anomaly_weight={anomaly_weight}")

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()
            epoch_losses = []
            self._current_train_epoch = epoch

            self.model.train()
            self.optimizer.zero_grad()

            for i, (input_data, labels) in enumerate(self.train_loader):
                input_data = input_data.float().to(self.device)
                labels = labels.float().to(self.device)

                rec, _, _, _, _, aux_losses, _ = self.model(input_data)

                loss = self._reconstruction_loss(rec, input_data)
                loss = self._add_temporal_difference_loss(loss, rec, input_data)

                # ★ P0-1: lambda_causal_l1 → lambda_locality_l1
                if aux_losses and 'sparse_loss' in aux_losses:
                    loss = loss + self.config.lambda_locality_l1 * aux_losses['sparse_loss']
                loss = self._add_temporal_graph_regularization(loss, aux_losses)
                loss = loss + self._source_effect_synthetic_loss(input_data, batch_idx=i)
                loss = loss + self._channel_masked_modeling_loss(input_data, batch_idx=i)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_losses.append(loss.item())

            avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0
            epoch_time = time.time() - epoch_start
            val_loss = self.vali(self.valid_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            es_status = self.early_stopping(val_loss, self.model, epoch + 1)

            self._print_epoch_result(
                epoch=epoch + 1, epoch_time=epoch_time,
                avg_train_loss=avg_train_loss, val_loss=val_loss,
                lr=current_lr, total_epochs=self.config.num_epochs,
                es_status=es_status, es_counter=self.early_stopping.counter,
            )

            if self.early_stopping.early_stop:
                break

            scheduler.step()

        total_time = time.time() - train_start_time
        self._save_best_params(total_params=total_params, total_time=total_time)
        self._save_train_history(total_params=total_params, total_time=total_time)

        print("─" * 70)
        print(f"  [OK] Reweight Training Complete!")
        print(f"     Best Val Loss: {self.early_stopping.val_loss_min:.6f}")
        print("─" * 70)
        print()

    def _save_train_history(self, total_params, total_time):
        from datetime import datetime
        from ts_benchmark.common.constant import ROOT_PATH

        if self.early_stopping is None:
            return

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        history_dir = os.path.join(ROOT_PATH, "result", "experiments", f"{self.dataset_name}_{timestamp}")
        os.makedirs(history_dir, exist_ok=True)

        history = {
            "dataset": self.dataset_name,
            "model": "LaGraph-v11.4",
            "config": {
                "d_model": self.config.d_model,
                "e_layers": self.config.e_layers,
                "n_heads": self.config.n_heads,
                "batch_size": self.config.batch_size,
                "lr": self.config.lr,
                "num_epochs": self.config.num_epochs,
                "patience": self.config.patience,
                "warmup_epochs": self.config.warmup_epochs,
                "lambda_vq": getattr(self.config, "lambda_vq", None),
                "vq_cooldown_epochs": getattr(self.config, "vq_cooldown_epochs", None),
                "vq_score_weight": getattr(self.config, "vq_score_weight", None),
                "use_direct_vq_score": getattr(self.config, "use_direct_vq_score", None),
                "score_topk_k": getattr(self.config, "score_topk_k", None),
                "use_parallel_graph_fusion": getattr(self.config, "use_parallel_graph_fusion", None),
                "graph_fusion_gate_mode": getattr(self.config, "graph_fusion_gate_mode", None),
                "graph_fusion_strategy": getattr(self.config, "graph_fusion_strategy", None),
                "graph_fusion_residual_init": getattr(self.config, "graph_fusion_residual_init", None),
                "use_graph_shift_score": getattr(self.config, "use_graph_shift_score", None),
                "graph_shift_score_weight": getattr(self.config, "graph_shift_score_weight", None),
                "use_lagged_causal_graph": getattr(self.config, "use_lagged_causal_graph", None),
                "causal_lags": getattr(self.config, "causal_lags", None),
                "causal_topk": getattr(self.config, "causal_topk", None),
                "causal_detach_backbone": getattr(self.config, "causal_detach_backbone", None),
                "lambda_causal_mechanism": getattr(self.config, "lambda_causal_mechanism", None),
                "lambda_causal_sparse": getattr(self.config, "lambda_causal_sparse", None),
                "use_causal_score": getattr(self.config, "use_causal_score", None),
                "causal_score_mode": getattr(self.config, "causal_score_mode", None),
                "causal_score_tail": getattr(self.config, "causal_score_tail", None),
                "causal_score_weight": getattr(self.config, "causal_score_weight", None),
                "use_temporal_graph_regularization": getattr(self.config, "use_temporal_graph_regularization", None),
                "lambda_temporal_graph_smooth": getattr(self.config, "lambda_temporal_graph_smooth", None),
                "lambda_temporal_graph_locality": getattr(self.config, "lambda_temporal_graph_locality", None),
                "reconstruction_loss_type": getattr(self.config, "reconstruction_loss_type", None),
                "smooth_l1_beta": getattr(self.config, "smooth_l1_beta", None),
                "mse_l1_alpha": getattr(self.config, "mse_l1_alpha", None),
                "mse_robust_alpha": getattr(self.config, "mse_robust_alpha", None),
                "charbonnier_eps": getattr(self.config, "charbonnier_eps", None),
                "lambda_temporal_diff_loss": getattr(self.config, "lambda_temporal_diff_loss", None),
                "use_robust_reconstruction_loss": getattr(self.config, "use_robust_reconstruction_loss", None),
                "robust_loss_trim_ratio": getattr(self.config, "robust_loss_trim_ratio", None),
                "robust_loss_min_weight": getattr(self.config, "robust_loss_min_weight", None),
                "robust_loss_warmup_epochs": getattr(self.config, "robust_loss_warmup_epochs", None),
                "use_score_channel_normalization": getattr(self.config, "use_score_channel_normalization", None),
                "score_channel_norm_mode": getattr(self.config, "score_channel_norm_mode", None),
                "score_channel_norm_eps": getattr(self.config, "score_channel_norm_eps", None),
                "use_state_aware_fusion": getattr(self.config, "use_state_aware_fusion", None),
                "state_aware_num_states": getattr(self.config, "state_aware_num_states", None),
                "state_aware_graph_gate_init": getattr(self.config, "state_aware_graph_gate_init", None),
                "state_aware_residual_init": getattr(self.config, "state_aware_residual_init", None),
                "lambda_state_balance": getattr(self.config, "lambda_state_balance", None),
                "lambda_state_confidence": getattr(self.config, "lambda_state_confidence", None),
                "dataloader_num_workers": getattr(self.config, "dataloader_num_workers", None),
                "dataloader_prefetch_factor": getattr(self.config, "dataloader_prefetch_factor", None),
                "use_channel_graph": getattr(self.config, "use_channel_graph", None),
                "use_temporal_graph": getattr(self.config, "use_temporal_graph", None),
                "use_dynamic_temporal_graph": getattr(self.config, "use_dynamic_temporal_graph", None),
                "dynamic_temporal_residual_init": getattr(self.config, "dynamic_temporal_residual_init", None),
                "dynamic_temporal_topk": getattr(self.config, "dynamic_temporal_topk", None),
                "dynamic_temporal_gate_mode": getattr(self.config, "dynamic_temporal_gate_mode", None),
                "channel_graph_lr_scale": getattr(self.config, "channel_graph_lr_scale", None),
                "temporal_graph_lr_scale": getattr(self.config, "temporal_graph_lr_scale", None),
                "use_vq_bypass": getattr(self.config, "use_vq_bypass", None),
                "use_multi_scale_scorer": getattr(self.config, "use_multi_scale_scorer", None),
                "score_aggregation": getattr(self.config, "score_aggregation", None),
                "score_aggregation_quantile": getattr(self.config, "score_aggregation_quantile", None),
                "score_center_width": getattr(self.config, "score_center_width", None),
                "use_synthetic_anomaly_aux": getattr(self.config, "use_synthetic_anomaly_aux", None),
                "lambda_synthetic_anomaly": getattr(self.config, "lambda_synthetic_anomaly", None),
                "synthetic_aux_interval": getattr(self.config, "synthetic_aux_interval", None),
                "use_source_effect_synthetic": getattr(self.config, "use_source_effect_synthetic", None),
                "lambda_source_effect": getattr(self.config, "lambda_source_effect", None),
                "source_effect_interval": getattr(self.config, "source_effect_interval", None),
                "use_synthetic_score": getattr(self.config, "use_synthetic_score", None),
                "synthetic_score_weight": getattr(self.config, "synthetic_score_weight", None),
                "score_smoothing_window": getattr(self.config, "score_smoothing_window", None),
                "score_smoothing_method": getattr(self.config, "score_smoothing_method", None),
                "use_event_persistence_score": getattr(self.config, "use_event_persistence_score", None),
                "event_persistence_window": getattr(self.config, "event_persistence_window", None),
                "event_persistence_weight": getattr(self.config, "event_persistence_weight", None),
                "prediction_fill_gap": getattr(self.config, "prediction_fill_gap", None),
                "prediction_min_len": getattr(self.config, "prediction_min_len", None),
                "prediction_dilate": getattr(self.config, "prediction_dilate", None),
                # ★ P0-1: lambda_causal_l1 → lambda_locality_l1
                "lambda_locality_l1": self.config.lambda_locality_l1,
                "dropout": self.config.dropout,
                "win_size": self.config.win_size,
            },
            "total_params": total_params,
            "total_params_human": _format_params(total_params),
            "total_time_seconds": round(total_time, 1),
            "total_time_human": _format_duration(total_time),
            "best_val_loss": float(self.early_stopping.val_loss_min),
            "best_epoch": self.early_stopping.best_epoch,
            "early_stopped": self.early_stopping.early_stop,
        }

        filepath = os.path.join(history_dir, "train_history.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
        print(f"  [History] Training history -> {filepath}")

    # ════════════════════════════════════════════════════════════════
    #  检测阶段
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _point_wise_aggregate(
        window_scores: np.ndarray,
        win_size: int,
        total_length: int,
        method: str = "mean",
        quantile: float = 0.9,
        center_width: int = 1,
    ) -> np.ndarray:
        window_scores = np.asarray(window_scores, dtype=np.float32)
        if window_scores.ndim != 2 or len(window_scores) == 0:
            return np.zeros(total_length, dtype=np.float32)

        N_windows = len(window_scores)
        win_size = min(int(win_size), window_scores.shape[1])
        total_length = int(total_length)
        method = (method or "mean").lower()

        def aggregate_mean() -> np.ndarray:
            point_scores = np.zeros(total_length, dtype=np.float64)
            point_counts = np.zeros(total_length, dtype=np.float64)
            for offset in range(win_size):
                n = min(N_windows, total_length - offset)
                if n <= 0:
                    break
                point_scores[offset:offset + n] += window_scores[:n, offset]
                point_counts[offset:offset + n] += 1.0
            return np.divide(
                point_scores, point_counts,
                out=np.zeros_like(point_scores),
                where=point_counts > 0,
            )

        if method == "mean":
            return aggregate_mean().astype(np.float32)

        if method == "max":
            point_scores = np.full(total_length, -np.inf, dtype=np.float64)
            point_counts = np.zeros(total_length, dtype=np.float64)
            for offset in range(win_size):
                n = min(N_windows, total_length - offset)
                if n <= 0:
                    break
                sl = slice(offset, offset + n)
                point_scores[sl] = np.maximum(point_scores[sl], window_scores[:n, offset])
                point_counts[sl] += 1.0
            fallback = aggregate_mean()
            point_scores = np.where(point_counts > 0, point_scores, fallback)
            return point_scores.astype(np.float32)

        if method in {"q75", "q90", "q95", "quantile"}:
            if method == "q75":
                q = 0.75
            elif method == "q90":
                q = 0.90
            elif method == "q95":
                q = 0.95
            else:
                q = float(quantile)
            q = min(max(q, 0.0), 1.0)
            values = np.full((total_length, win_size), np.nan, dtype=np.float32)
            for offset in range(win_size):
                n = min(N_windows, total_length - offset)
                if n <= 0:
                    break
                values[offset:offset + n, offset] = window_scores[:n, offset]
            return np.nanquantile(values, q, axis=1).astype(np.float32)

        if method in {"center", "last"}:
            fallback = aggregate_mean()
            point_scores = np.zeros(total_length, dtype=np.float64)
            point_counts = np.zeros(total_length, dtype=np.float64)
            if method == "last":
                offsets = [win_size - 1]
            else:
                width = max(1, int(center_width or 1))
                center = win_size // 2
                start = max(0, center - width // 2)
                end = min(win_size, start + width)
                offsets = list(range(start, end))
            for offset in offsets:
                n = min(N_windows, total_length - offset)
                if n <= 0:
                    continue
                point_scores[offset:offset + n] += window_scores[:n, offset]
                point_counts[offset:offset + n] += 1.0
            point_scores = np.divide(
                point_scores, point_counts,
                out=fallback.astype(np.float64, copy=True),
                where=point_counts > 0,
            )
            return point_scores.astype(np.float32)

        raise ValueError(
            f"Unsupported score_aggregation={method!r}. "
            "Choose from mean, max, q75, q90, q95, quantile, center, last."
        )

    def _aggregate_window_scores(self, window_scores: np.ndarray, total_length: int) -> np.ndarray:
        return self._point_wise_aggregate(
            window_scores,
            self.config.win_size,
            total_length,
            method=getattr(self.config, "score_aggregation", "mean"),
            quantile=getattr(self.config, "score_aggregation_quantile", 0.9),
            center_width=getattr(self.config, "score_center_width", 1),
        )

    def _smooth_scores_for_detection(self, scores: np.ndarray) -> np.ndarray:
        window = int(getattr(self.config, "score_smoothing_window", 1) or 1)
        if window <= 1:
            return scores.astype(np.float32, copy=False)
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(scores.astype(np.float64), (pad, pad), mode="edge")
        method = getattr(self.config, "score_smoothing_method", "mean")
        if method == "max":
            smoothed = np.empty_like(scores, dtype=np.float64)
            for i in range(len(scores)):
                smoothed[i] = padded[i:i + window].max()
        else:
            kernel = np.ones(window, dtype=np.float64) / float(window)
            smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed.astype(np.float32)

    def _apply_event_persistence_score(self, scores: np.ndarray) -> np.ndarray:
        if not getattr(self.config, "use_event_persistence_score", False):
            return scores.astype(np.float32, copy=False)

        scores64 = scores.astype(np.float64, copy=False)
        window = int(getattr(self.config, "event_persistence_window", 9) or 9)
        weight = float(getattr(self.config, "event_persistence_weight", 0.5) or 0.5)
        eps = float(getattr(self.config, "event_persistence_eps", 1e-6) or 1e-6)
        if window <= 1 or weight <= 0:
            return scores64.astype(np.float32)
        if window % 2 == 0:
            window += 1

        median = np.median(scores64)
        mad = np.median(np.abs(scores64 - median))
        robust_scale = max(1.4826 * mad, float(np.std(scores64)), eps)
        positive = np.maximum((scores64 - median) / robust_scale, 0.0)

        pad = window // 2
        padded = np.pad(positive, (pad, pad), mode="edge")
        kernel = np.ones(window, dtype=np.float64) / float(window)
        support = np.convolve(padded, kernel, mode="valid")
        support = support / (support + 1.0)

        return (scores64 * (1.0 + weight * support)).astype(np.float32)

    @staticmethod
    def _binary_segments(pred: np.ndarray, value: int):
        n = len(pred)
        start = None
        for i in range(n + 1):
            cur = pred[i] if i < n else 1 - value
            if cur == value and start is None:
                start = i
            elif cur != value and start is not None:
                yield start, i
                start = None

    def _shape_prediction_segments(self, pred: np.ndarray) -> np.ndarray:
        pred = pred.astype(np.int32, copy=True)
        min_len = int(getattr(self.config, "prediction_min_len", 1) or 1)
        fill_gap = int(getattr(self.config, "prediction_fill_gap", 0) or 0)
        dilate = int(getattr(self.config, "prediction_dilate", 0) or 0)

        if min_len > 1:
            for start, end in list(self._binary_segments(pred, 1)):
                if end - start < min_len:
                    pred[start:end] = 0

        if fill_gap > 0:
            for start, end in list(self._binary_segments(pred, 0)):
                if start > 0 and end < len(pred) and end - start <= fill_gap:
                    pred[start:end] = 1

        if dilate > 0 and pred.any():
            shaped = pred.copy()
            for start, end in self._binary_segments(pred, 1):
                shaped[max(0, start - dilate):min(len(pred), end + dilate)] = 1
            pred = shaped

        return pred

    @torch.no_grad()
    def _detect_forward(self, input_data):
        score, vq_score = self.model.multi_scale_forward(input_data)
        return score.cpu().numpy()

    @torch.no_grad()
    def _detect_forward_with_channels(self, input_data):
        score, _ = self.model.multi_scale_forward(input_data)
        raw_model = self._get_raw_model()
        x_rec, A_adaptive, _, _, _, aux_losses, _ = raw_model(input_data)
        channel_err = F.l1_loss(x_rec, input_data, reduction="none")
        if hasattr(raw_model, "_normalize_score_error"):
            channel_err = raw_model._normalize_score_error(channel_err)
        graph_err = self._graph_propagated_channel_error(channel_err, A_adaptive)
        mechanism_err = None
        if aux_losses:
            mechanism_err = aux_losses.get("channel_mechanism_error")
        if mechanism_err is None:
            mechanism_err = torch.zeros_like(channel_err)
        elif hasattr(raw_model, "_normalize_channel_mechanism_score"):
            mechanism_err = raw_model._normalize_channel_mechanism_score(mechanism_err)
        causal_err = None
        if aux_losses:
            causal_err = aux_losses.get("causal_channel_error")
        if causal_err is None:
            causal_err = torch.zeros_like(channel_err)
        source_gate_err = None
        if aux_losses:
            source_gate_err = aux_losses.get("source_gate_score")
        if source_gate_err is None:
            source_gate_err = torch.zeros_like(channel_err)
        synthetic_rca_window_scores = None
        if aux_losses:
            synthetic_rca_logits = aux_losses.get("synthetic_rca_logits")
            if synthetic_rca_logits is not None:
                synthetic_rca_window_scores = torch.sigmoid(synthetic_rca_logits)
        if synthetic_rca_window_scores is None:
            synthetic_rca_window_scores = torch.zeros(
                channel_err.shape[0],
                channel_err.shape[-1],
                device=channel_err.device,
                dtype=channel_err.dtype,
            )
        return (
            score.cpu().numpy(),
            channel_err.cpu().numpy(),
            graph_err.cpu().numpy(),
            mechanism_err.cpu().numpy(),
            causal_err.cpu().numpy(),
            source_gate_err.cpu().numpy(),
            synthetic_rca_window_scores.cpu().numpy(),
        )

    def _graph_propagated_channel_error(self, channel_err, A_adaptive):
        weight = float(getattr(self.config, "rca_graph_weight", 0.0) or 0.0)
        if weight <= 0.0 or A_adaptive is None:
            return torch.zeros_like(channel_err)

        A = A_adaptive.detach().clamp_min(0.0)
        A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        direction = str(getattr(self.config, "rca_graph_direction", "outgoing") or "outgoing")
        if direction == "incoming":
            return torch.bmm(channel_err, A)
        if direction == "both":
            outgoing = torch.bmm(channel_err, A.transpose(1, 2))
            incoming = torch.bmm(channel_err, A)
            return 0.5 * (outgoing + incoming)
        return torch.bmm(channel_err, A.transpose(1, 2))

    @staticmethod
    def _add_window_channel_scores(channel_sums, point_counts, window_channels, start_index, update_counts=True):
        if window_channels is None or len(window_channels) == 0:
            return
        n_windows, win_size, _ = window_channels.shape
        total_length = channel_sums.shape[0]
        for offset in range(win_size):
            point_start = start_index + offset
            if point_start >= total_length:
                break
            n = min(n_windows, total_length - point_start)
            if n <= 0:
                continue
            sl = slice(point_start, point_start + n)
            channel_sums[sl] += window_channels[:n, offset, :]
            if update_counts:
                point_counts[sl] += 1.0

    @staticmethod
    def _add_window_channel_score_group(channel_sums_list, point_counts, window_channels_list, start_index):
        pairs = [
            (channel_sums, window_channels)
            for channel_sums, window_channels in zip(channel_sums_list, window_channels_list)
            if channel_sums is not None and window_channels is not None and len(window_channels) > 0
        ]
        if not pairs:
            return
        n_windows, win_size, _ = pairs[0][1].shape
        total_length = pairs[0][0].shape[0]
        for offset in range(win_size):
            point_start = start_index + offset
            if point_start >= total_length:
                break
            n = min(n_windows, total_length - point_start)
            if n <= 0:
                continue
            sl = slice(point_start, point_start + n)
            for channel_sums, window_channels in pairs:
                channel_sums[sl] += window_channels[:n, offset, :]
            point_counts[sl] += 1.0

    @staticmethod
    def _add_window_constant_channel_scores(channel_diff, window_channels, start_index, win_size):
        if channel_diff is None or window_channels is None or len(window_channels) == 0:
            return
        n_windows, _ = window_channels.shape
        total_length = channel_diff.shape[0] - 1
        starts = start_index + np.arange(n_windows)
        valid = starts < total_length
        if not np.any(valid):
            return
        starts = starts[valid]
        ends = np.minimum(starts + int(win_size), total_length)
        values = window_channels[valid]
        np.add.at(channel_diff, starts, values)
        np.add.at(channel_diff, ends, -values)

    @staticmethod
    def _merge_intervals(intervals):
        intervals = sorted((int(s), int(e)) for s, e in intervals if int(e) > int(s))
        if not intervals:
            return []
        merged = [list(intervals[0])]
        for start, end in intervals[1:]:
            if start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        return [(start, end) for start, end in merged]

    def _compute_event_local_rca_channel_scores(self, test_data, labels, pred_mask=None, rca_offset=0):
        scaled_test = self._transform_input_frame(test_data)
        total_length, n_channels = scaled_test.shape
        win_size = int(self.config.win_size)
        if total_length <= 0 or win_size <= 0:
            return False

        event_segments = []
        event_segments.extend(self._label_segments(np.asarray(labels).astype(int)))
        if pred_mask is not None:
            event_segments.extend(self._label_segments(np.asarray(pred_mask).astype(int)))
        if not event_segments:
            return False

        margin = int(getattr(self.config, "rca_event_local_margin", win_size) or 0)
        point_regions = []
        for start, end in event_segments:
            full_start = int(start) + int(rca_offset)
            full_end = int(end) + int(rca_offset)
            point_regions.append(
                (
                    max(0, full_start - margin),
                    min(total_length, full_end + margin),
                )
            )
        point_regions = self._merge_intervals(point_regions)

        max_start = max(0, total_length - win_size)
        start_regions = []
        for start, end in point_regions:
            start_min = max(0, start - win_size + 1)
            start_max = min(max_start, end - 1)
            if start_max >= start_min:
                start_regions.append((start_min, start_max + 1))
        start_regions = self._merge_intervals(start_regions)
        if not start_regions:
            return False

        channel_sums = np.zeros((total_length, n_channels), dtype=np.float64)
        graph_channel_sums = np.zeros_like(channel_sums)
        mechanism_channel_sums = np.zeros_like(channel_sums)
        causal_channel_sums = np.zeros_like(channel_sums)
        source_gate_channel_sums = np.zeros_like(channel_sums)
        synthetic_rca_channel_diff = np.zeros((total_length + 1, n_channels), dtype=np.float64)
        channel_counts = np.zeros(total_length, dtype=np.float64)

        eval_batch_size = min(
            int(self.config.batch_size),
            int(getattr(self.config, "eval_batch_size", 64) or 64),
        )
        self.model.eval()
        for start_region, end_region in start_regions:
            cursor = int(start_region)
            while cursor < int(end_region):
                batch_end = min(cursor + eval_batch_size, int(end_region))
                windows = np.stack(
                    [scaled_test[start:start + win_size] for start in range(cursor, batch_end)],
                    axis=0,
                )
                input_data = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
                (
                    _,
                    channel_err,
                    graph_channel_err,
                    mechanism_channel_err,
                    causal_channel_err,
                    source_gate_channel_err,
                    synthetic_rca_channel_err,
                ) = self._detect_forward_with_channels(input_data)
                self._add_window_channel_score_group(
                    [
                        channel_sums,
                        graph_channel_sums,
                        mechanism_channel_sums,
                        causal_channel_sums,
                        source_gate_channel_sums,
                    ],
                    channel_counts,
                    [
                        channel_err,
                        graph_channel_err,
                        mechanism_channel_err,
                        causal_channel_err,
                        source_gate_channel_err,
                    ],
                    cursor,
                )
                self._add_window_constant_channel_scores(
                    synthetic_rca_channel_diff,
                    synthetic_rca_channel_err,
                    cursor,
                    win_size,
                )
                cursor = batch_end

        self._last_channel_scores = np.divide(
            channel_sums,
            channel_counts[:, None],
            out=np.zeros_like(channel_sums),
            where=channel_counts[:, None] > 0,
        ).astype(np.float32)
        self._last_graph_channel_scores = np.divide(
            graph_channel_sums,
            channel_counts[:, None],
            out=np.zeros_like(graph_channel_sums),
            where=channel_counts[:, None] > 0,
        ).astype(np.float32)
        self._last_mechanism_channel_scores = np.divide(
            mechanism_channel_sums,
            channel_counts[:, None],
            out=np.zeros_like(mechanism_channel_sums),
            where=channel_counts[:, None] > 0,
        ).astype(np.float32)
        self._last_causal_channel_scores = np.divide(
            causal_channel_sums,
            channel_counts[:, None],
            out=np.zeros_like(causal_channel_sums),
            where=channel_counts[:, None] > 0,
        ).astype(np.float32)
        self._last_source_gate_channel_scores = np.divide(
            source_gate_channel_sums,
            channel_counts[:, None],
            out=np.zeros_like(source_gate_channel_sums),
            where=channel_counts[:, None] > 0,
        ).astype(np.float32)
        synthetic_rca_channel_sums = np.cumsum(synthetic_rca_channel_diff[:-1], axis=0)
        self._last_synthetic_rca_channel_scores = np.divide(
            synthetic_rca_channel_sums,
            channel_counts[:, None],
            out=np.zeros_like(synthetic_rca_channel_sums),
            where=channel_counts[:, None] > 0,
        ).astype(np.float32)
        self._last_channel_names = list(test_data.columns)
        covered = int((channel_counts > 0).sum())
        print(
            f"  [RCA] Event-local channel scoring: regions={len(point_regions)}, "
            f"windows={sum(e - s for s, e in start_regions):,}, covered_points={covered:,}/{total_length:,}"
        )
        return True

    @torch.no_grad()
    def _compute_counterfactual_channel_scores(
        self,
        test_data,
        labels,
        pred_mask,
        rca_offset,
        candidate_channel_scores,
        event_head_ratio,
        event_head_points,
    ):
        scaled_test = self._transform_input_frame(test_data)
        values = scaled_test.values.astype(np.float32, copy=False)
        total_length, n_channels = values.shape
        label_length = int(len(labels))
        if total_length <= 0 or label_length <= 0 or n_channels <= 0:
            return np.zeros((label_length, n_channels), dtype=np.float32)

        segments = []
        seen = set()
        for mask in (labels, pred_mask):
            if mask is None:
                continue
            for start, end in self._label_segments(np.asarray(mask).reshape(-1).astype(int)):
                key = (int(start), int(end))
                if key not in seen and end > start:
                    seen.add(key)
                    segments.append(key)
        if not segments:
            return np.zeros((label_length, n_channels), dtype=np.float32)

        win_size = int(self.config.win_size)
        max_start = max(0, total_length - win_size)
        candidate_topk = min(
            n_channels,
            max(1, int(getattr(self.config, "rca_counterfactual_candidates", 12) or 12)),
        )
        max_windows = max(1, int(getattr(self.config, "rca_counterfactual_max_windows", 32) or 32))
        batch_candidates = max(1, int(getattr(self.config, "rca_counterfactual_batch_candidates", 4) or 4))
        baseline_window = max(1, int(getattr(self.config, "rca_counterfactual_baseline_window", 300) or 300))
        output = np.zeros((label_length, n_channels), dtype=np.float32)
        window_offsets = np.arange(win_size, dtype=np.int64)
        total_windows = 0

        self.model.eval()
        for start, end in segments:
            score_start, score_end = self._rca_event_score_bounds(
                start,
                end,
                event_head_ratio,
                event_head_points,
            )
            score_start = max(0, min(label_length, int(score_start)))
            if score_start >= label_length:
                continue
            score_end = max(score_start + 1, min(label_length, int(score_end)))
            full_score_start = int(rca_offset) + score_start
            full_score_end = int(rca_offset) + score_end
            if full_score_start >= total_length or full_score_end <= 0:
                continue
            full_score_start = max(0, full_score_start)
            full_score_end = min(total_length, full_score_end)
            if full_score_end <= full_score_start:
                continue

            window_start_min = max(0, full_score_start - win_size + 1)
            window_start_max = min(max_start, full_score_end - 1)
            if window_start_max < window_start_min:
                continue
            starts = np.arange(window_start_min, window_start_max + 1, dtype=np.int64)
            if len(starts) > max_windows:
                sampled = np.linspace(0, len(starts) - 1, max_windows)
                starts = starts[np.unique(np.round(sampled).astype(np.int64))]
            if len(starts) == 0:
                continue

            point_index = starts[:, None] + window_offsets[None, :]
            event_mask = (point_index >= full_score_start) & (point_index < full_score_end)
            valid_rows = event_mask.any(axis=1)
            starts = starts[valid_rows]
            event_mask = event_mask[valid_rows]
            if len(starts) == 0:
                continue

            windows = np.stack([values[s:s + win_size] for s in starts], axis=0)
            input_data = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
            base_window_scores = self._detect_forward(input_data)
            mask_float = event_mask.astype(np.float32)
            denom = np.maximum(mask_float.sum(axis=1), 1.0)
            base_event_scores = (base_window_scores * mask_float).sum(axis=1) / denom

            event_candidate_scores = candidate_channel_scores[score_start:score_end].mean(axis=0)
            candidates = np.argsort(-event_candidate_scores)[:candidate_topk].astype(np.int64)
            baseline_start = max(0, full_score_start - baseline_window)
            if baseline_start < full_score_start:
                baseline_values = np.median(values[baseline_start:full_score_start], axis=0)
            else:
                baseline_values = np.zeros(n_channels, dtype=np.float32)

            n_windows = len(windows)
            event_scores = np.zeros(n_channels, dtype=np.float32)
            for cursor in range(0, len(candidates), batch_candidates):
                cand_batch = candidates[cursor:cursor + batch_candidates]
                cf_windows = np.repeat(windows, len(cand_batch), axis=0).copy()
                for local_idx, channel_idx in enumerate(cand_batch):
                    block = cf_windows[local_idx * n_windows:(local_idx + 1) * n_windows, :, channel_idx]
                    block[event_mask] = baseline_values[channel_idx]
                cf_input = torch.as_tensor(cf_windows, dtype=torch.float32, device=self.device)
                cf_window_scores = self._detect_forward(cf_input)
                cf_window_scores = cf_window_scores.reshape(len(cand_batch), n_windows, win_size)
                cf_event_scores = (cf_window_scores * mask_float[None, :, :]).sum(axis=2) / denom[None, :]
                drops = np.maximum(base_event_scores[None, :] - cf_event_scores, 0.0).mean(axis=1)
                event_scores[cand_batch] = drops.astype(np.float32)

            if np.any(event_scores > 0):
                output[score_start:score_end] = np.maximum(output[score_start:score_end], event_scores)
            total_windows += int(len(starts) * max(1, len(candidates)))

        print(
            f"  [RCA] Counterfactual channel scoring: events={len(segments)}, "
            f"candidate_window_evals={total_windows:,}, topk={candidate_topk}, max_windows={max_windows}"
        )
        return output

    def detect_score(self, train: pd.DataFrame) -> np.ndarray:
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call detect_fit first.")
        if self._should_load_best_checkpoint():
            self._get_raw_model().load_state_dict(self.early_stopping.check_point)

        import gc
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        gc.collect()
        for _ in range(2):
            torch.cuda.empty_cache()
        self.model.to(self.device)

        eval_batch_size = min(
            int(self.config.batch_size),
            int(getattr(self.config, "eval_batch_size", 64) or 64),
        )
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info(self.device)[0] / 1024**3
            if free_gb < 2.0:
                eval_batch_size = min(eval_batch_size, 32)
            elif free_gb < 4.0:
                eval_batch_size = min(eval_batch_size, 48)

        scaled_data = self._transform_input_frame(train)
        total_length = len(scaled_data)

        self.model.eval()
        window_scores_list = []

        loader = anomaly_detection_data_provider(
            scaled_data, batch_size=eval_batch_size,
            win_size=self.config.win_size, step=1, mode="test",
            num_workers=0,
        )

        for i, (input_data, labels) in enumerate(loader):
            input_data = input_data.float().to(self.device)
            scores_batch = self._detect_forward(input_data)
            window_scores_list.append(scores_batch)
            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()

        if len(window_scores_list) == 0:
            dummy = np.zeros(total_length)
            return dummy, dummy

        window_scores = np.concatenate(window_scores_list, axis=0)

        point_scores = self._aggregate_window_scores(window_scores, total_length)
        point_scores = self._apply_event_persistence_score(point_scores)
        point_scores = self._smooth_scores_for_detection(point_scores)

        return point_scores, point_scores

    def detect_label(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        检测并返回异常标签（多异常率）。

        ★ P0-2: 支持 POT 阈值估计（通过 self._pot_estimator）
        ★ P0-3: 如果 test_data 真实标签可通过外部获取，调用 evaluate() 计算指标

        Returns:
            preds: {ratio: np.ndarray(N,)} — 各异常率下的 0/1 标签
            test_energy: np.ndarray(N,) — 点级异常分数
        """
        if not self.trained:
            raise RuntimeError("Model not trained yet. Call detect_fit first.")
        if self._should_load_best_checkpoint():
            self._get_raw_model().load_state_dict(self.early_stopping.check_point)
        self.model.to(self.device)

        eval_batch_size = min(
            int(self.config.batch_size),
            int(getattr(self.config, "eval_batch_size", 64) or 64),
        )
        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info(self.device)[0] / 1024**3
            if free_gb < 2.0:
                eval_batch_size = min(eval_batch_size, 32)
            elif free_gb < 4.0:
                eval_batch_size = min(eval_batch_size, 48)

        scaled_test = self._transform_input_frame(test_data)
        total_length = len(scaled_test)

        self.model.eval()

        test_loader = anomaly_detection_data_provider(
            scaled_test, batch_size=eval_batch_size,
            win_size=self.config.win_size, step=1, mode="test",
            num_workers=0,
        )

        test_window_list = []
        export_rca = bool(getattr(self.config, "export_rca", False))
        event_local_rca = export_rca and bool(getattr(self.config, "rca_event_local_export", False))
        dense_rca_export = export_rca and not event_local_rca
        channel_sums = None
        graph_channel_sums = None
        mechanism_channel_sums = None
        causal_channel_sums = None
        source_gate_channel_sums = None
        synthetic_rca_channel_diff = None
        channel_counts = None
        window_cursor = 0
        if dense_rca_export:
            channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            graph_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            mechanism_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            causal_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            source_gate_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            synthetic_rca_channel_diff = np.zeros((total_length + 1, scaled_test.shape[1]), dtype=np.float64)
            channel_counts = np.zeros(total_length, dtype=np.float64)

        for i, (input_data, labels) in enumerate(test_loader):
            input_data = input_data.float().to(self.device)
            if dense_rca_export:
                (
                    cri,
                    channel_err,
                    graph_channel_err,
                    mechanism_channel_err,
                    causal_channel_err,
                    source_gate_channel_err,
                    synthetic_rca_channel_err,
                ) = self._detect_forward_with_channels(input_data)
                self._add_window_channel_score_group(
                    [
                        channel_sums,
                        graph_channel_sums,
                        mechanism_channel_sums,
                        causal_channel_sums,
                        source_gate_channel_sums,
                    ],
                    channel_counts,
                    [
                        channel_err,
                        graph_channel_err,
                        mechanism_channel_err,
                        causal_channel_err,
                        source_gate_channel_err,
                    ],
                    window_cursor,
                )
                self._add_window_constant_channel_scores(
                    synthetic_rca_channel_diff,
                    synthetic_rca_channel_err,
                    window_cursor,
                    int(channel_err.shape[1]),
                )
                window_cursor += int(channel_err.shape[0])
            else:
                cri = self._detect_forward(input_data)
            test_window_list.append(cri)
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()

        if len(test_window_list) == 0:
            dummy = np.zeros(total_length, dtype=np.int32)
            return {r: dummy for r in self.config.anomaly_ratio}, np.zeros(total_length)

        if dense_rca_export and channel_sums is not None:
            self._last_channel_scores = np.divide(
                channel_sums,
                channel_counts[:, None],
                out=np.zeros_like(channel_sums),
                where=channel_counts[:, None] > 0,
            ).astype(np.float32)
            self._last_graph_channel_scores = np.divide(
                graph_channel_sums,
                channel_counts[:, None],
                out=np.zeros_like(graph_channel_sums),
                where=channel_counts[:, None] > 0,
            ).astype(np.float32)
            self._last_mechanism_channel_scores = np.divide(
                mechanism_channel_sums,
                channel_counts[:, None],
                out=np.zeros_like(mechanism_channel_sums),
                where=channel_counts[:, None] > 0,
            ).astype(np.float32)
            self._last_causal_channel_scores = np.divide(
                causal_channel_sums,
                channel_counts[:, None],
                out=np.zeros_like(causal_channel_sums),
                where=channel_counts[:, None] > 0,
            ).astype(np.float32)
            self._last_source_gate_channel_scores = np.divide(
                source_gate_channel_sums,
                channel_counts[:, None],
                out=np.zeros_like(source_gate_channel_sums),
                where=channel_counts[:, None] > 0,
            ).astype(np.float32)
            synthetic_rca_channel_sums = np.cumsum(synthetic_rca_channel_diff[:-1], axis=0)
            self._last_synthetic_rca_channel_scores = np.divide(
                synthetic_rca_channel_sums,
                channel_counts[:, None],
                out=np.zeros_like(synthetic_rca_channel_sums),
                where=channel_counts[:, None] > 0,
            ).astype(np.float32)
            self._last_channel_names = list(test_data.columns)
        else:
            self._last_channel_scores = None
            self._last_graph_channel_scores = None
            self._last_mechanism_channel_scores = None
            self._last_causal_channel_scores = None
            self._last_source_gate_channel_scores = None
            self._last_synthetic_rca_channel_scores = None
            self._last_channel_names = list(test_data.columns) if event_local_rca else None

        test_windows = np.concatenate(test_window_list, axis=0)
        test_energy = self._aggregate_window_scores(test_windows, total_length)
        test_energy = self._apply_event_persistence_score(test_energy)
        test_energy = self._smooth_scores_for_detection(test_energy)

        # === 步骤 2：阈值选取（★ P0-2: 优先使用 POT 阈值）===
        pot_threshold = self._pot_estimator.get_threshold()

        if pot_threshold > 0 and self._train_anomaly_scores is not None:
            # 使用 POT 估计的阈值作为基准
            threshold_source = self._train_anomaly_scores
            print(f"  [POT] Using POT-estimated threshold={pot_threshold:.6f}")
        elif self._train_anomaly_scores is not None and len(self._train_anomaly_scores) > 0:
            threshold_source = self._train_anomaly_scores
            print(f"  [INFO] Using cached train scores for threshold")
        else:
            print(f"  [WARN] Train scores not available, falling back to test scores")
            threshold_source = test_energy

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        print(f"\n  [DIAGNOSTIC] detect_label analysis:")
        print(f"  [DIAGNOSTIC]   test_energy: n={len(test_energy)}, "
              f"min={test_energy.min():.6f}, max={test_energy.max():.6f}, "
              f"mean={test_energy.mean():.6f}")

        preds = {}
        for ratio in self.config.anomaly_ratio:
            # ★ BUGFIX v11.4.1: 每个 ratio 独立使用百分位数阈值
            #   旧版: max(pot_threshold, np.percentile(..., 100 - ratio))
            #   问题: pot_threshold ≈ 99.99th percentile 始终大于所有 ratio 百分位
            #         导致所有 ratio 共享同一极端的 POT 阈值 → 所有预测完全相同
            #   新版: per-ratio 直接使用百分位数阈值，POT 阈值仅用于 ratio=None 的默认输出
            threshold = np.percentile(threshold_source, 100 - ratio)

            pred = (test_energy > threshold).astype(int)
            preds[ratio] = self._shape_prediction_segments(pred)
            anom_frac = preds[ratio].mean() * 100
            print(f"  [DIAGNOSTIC]   ratio={ratio:>5.1f}%: threshold={threshold:.6f}, "
                  f"anomaly_frac={anom_frac:.2f}%")

        # ★ BUGFIX: POT 阈值仅用于默认检测（ratio=None），不覆盖 per-ratio 结果
        if pot_threshold > 0:
            pred_pot = self._shape_prediction_segments(
                (test_energy > pot_threshold).astype(int)
            )
            preds[None] = pred_pot
            print(f"  [DIAGNOSTIC]   POT default: threshold={pot_threshold:.6f}, "
                  f"anomaly_frac={pred_pot.mean() * 100:.2f}%")

        # ★ P0-5: 如果可视化 hook 已注册，捕获中间数据
        if self._vis_hook is not None:
            try:
                vis_data = self._vis_hook.extract()
                from datetime import datetime
                from ts_benchmark.common.constant import ROOT_PATH
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                vis_dir = os.path.join(ROOT_PATH, "result", "diagnosis", self.dataset_name, ts)
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, "vis_data.json")
                self._vis_hook.save_to_json(vis_data, vis_path)
                print(f"\n  [VIS] Visualization data -> {vis_path}")
            except Exception as e:
                print(f"\n  [VIS WARN] Failed to extract visualization data: {e}")

        return preds, test_energy

    # ════════════════════════════════════════════════════════════════
    #  ★ P0-3: 评估报告系统（AUC-ROC / AUPR）
    # ════════════════════════════════════════════════════════════════

    @staticmethod
    def _label_segments(mask):
        mask = np.asarray(mask).astype(bool)
        segments = []
        in_segment = False
        for idx, value in enumerate(mask):
            if value and not in_segment:
                start = idx
                in_segment = True
            elif in_segment and not value:
                segments.append((start, idx))
                in_segment = False
        if in_segment:
            segments.append((start, len(mask)))
        return segments

    @staticmethod
    def _root_cause_group_name(feature_name):
        if not isinstance(feature_name, str):
            return feature_name
        name = feature_name.strip()
        if re.match(r"^P\d+_", name):
            return name.split("_", 1)[0]
        wadi_match = re.match(r"^([123])(?:[A-Z])?_", name)
        if wadi_match:
            return f"WADI_P{wadi_match.group(1)}"
        match = re.search(r"(\d{3})", name)
        if match:
            return f"P{match.group(1)[0]}"
        return name

    @staticmethod
    def _rca_prediction_key_name(key):
        if key is None:
            return "pot"
        return str(key)

    def _select_rca_prediction_mask(self, predict_labels, length):
        if predict_labels is None:
            return None, None
        if not isinstance(predict_labels, dict):
            pred = np.asarray(predict_labels).reshape(-1).astype(int)
            pred = pred[:length]
            if len(pred) < length:
                pred = np.pad(pred, (0, length - len(pred)), mode="constant")
            return "single", pred

        requested = str(getattr(self.config, "rca_prediction_key", "pot") or "pot")
        selected_key = None
        if requested.lower() == "pot" and None in predict_labels:
            selected_key = None
        else:
            for key in predict_labels.keys():
                if self._rca_prediction_key_name(key) == requested:
                    selected_key = key
                    break
        if selected_key is None and None in predict_labels:
            selected_key = None
        elif selected_key is None and predict_labels:
            selected_key = next(iter(predict_labels.keys()))

        pred = np.asarray(predict_labels[selected_key]).reshape(-1).astype(int)
        pred = pred[:length]
        if len(pred) < length:
            pred = np.pad(pred, (0, length - len(pred)), mode="constant")
        return self._rca_prediction_key_name(selected_key), pred

    @staticmethod
    def _normalize_prediction_mask(prediction, length):
        pred = np.asarray(prediction).reshape(-1).astype(int)
        pred = pred[:length]
        if len(pred) < length:
            pred = np.pad(pred, (0, length - len(pred)), mode="constant")
        return pred

    def _rca_predicted_segments(self, pred_mask):
        """Return local predicted RCA windows without changing detection labels."""
        segments = self._label_segments(pred_mask)
        if not bool(getattr(self.config, "rca_split_predicted_events", False)):
            return segments

        max_len = max(1, int(getattr(self.config, "rca_split_max_event_len", 120) or 120))
        stride = max(1, int(getattr(self.config, "rca_split_stride", max_len) or max_len))
        split_segments = []
        for start, end in segments:
            length = int(end) - int(start)
            if length <= max_len:
                split_segments.append((int(start), int(end)))
                continue

            cursor = int(start)
            while cursor < int(end):
                window_end = min(int(end), cursor + max_len)
                if window_end > cursor:
                    split_segments.append((cursor, window_end))
                if window_end >= int(end):
                    break
                cursor += stride
            if split_segments and split_segments[-1][1] < int(end):
                split_segments.append((max(int(start), int(end) - max_len), int(end)))

        # Remove exact duplicates that can occur when stride/window align at the tail.
        deduped = []
        seen = set()
        for segment in split_segments:
            if segment not in seen:
                deduped.append(segment)
                seen.add(segment)
        return deduped

    @staticmethod
    def _event_onset_scores(score_matrix, start, end, baseline_window=200, onset_z=2.0):
        """Score variables that cross their normal local baseline earlier within an event."""
        if score_matrix is None or end <= start or start <= 0:
            width = score_matrix.shape[1] if score_matrix is not None and score_matrix.ndim == 2 else 0
            return np.zeros(width, dtype=np.float32)

        event_scores = score_matrix[start:end]
        if event_scores.size == 0:
            return np.zeros(score_matrix.shape[1], dtype=np.float32)

        baseline_start = max(0, start - max(1, int(baseline_window)))
        baseline_scores = score_matrix[baseline_start:start]
        if baseline_scores.size == 0:
            return np.zeros(score_matrix.shape[1], dtype=np.float32)

        center = np.median(baseline_scores, axis=0)
        mad = 1.4826 * np.median(np.abs(baseline_scores - center), axis=0)
        std = np.std(baseline_scores, axis=0)
        scale = np.where(mad > 1e-6, mad, std)
        scale = np.maximum(scale, 1e-6)

        threshold = center + float(onset_z) * scale
        above = event_scores >= threshold
        has_onset = above.any(axis=0)
        first_idx = np.argmax(above, axis=0)
        length = max(1, event_scores.shape[0])
        early_factor = np.where(has_onset, 1.0 - (first_idx / float(length)), 0.0)
        peak_delta = np.maximum(event_scores.max(axis=0) - center, 0.0) / scale
        onset_scores = np.where(has_onset, early_factor * np.log1p(peak_delta), 0.0)
        return np.nan_to_num(onset_scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def _normalize_event_component(values):
        """Normalize one event-level RCA component across channels to avoid scale domination."""
        arr = np.asarray(values, dtype=np.float32)
        if arr.size == 0:
            return arr
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        lo = float(np.min(arr))
        hi = float(np.max(arr))
        span = hi - lo
        if span <= 1e-8:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - lo) / span).astype(np.float32)

    def _build_rca_events(
        self,
        segments,
        channel_scores,
        base_channel_scores,
        graph_channel_scores,
        mechanism_channel_scores,
        feature_names,
        contrast_window,
        contrast_weight,
        mechanism_residual_window,
        mechanism_residual_weight,
        event_head_ratio=1.0,
        event_head_points=0,
        max_channel_ranking=None,
        source_channel_scores=None,
        propagation_channel_scores=None,
        causal_channel_scores=None,
        source_gate_channel_scores=None,
        synthetic_channel_scores=None,
        counterfactual_channel_scores=None,
        use_source_propagation=False,
        graph_weight=0.0,
        mechanism_weight=0.0,
        source_weight=0.75,
        source_base_weight=1.0,
        propagation_weight=0.25,
        source_mechanism_weight=0.0,
        causal_weight=0.0,
        synthetic_weight=0.0,
        source_gate_weight=0.0,
        source_interaction_weight=0.0,
        counterfactual_weight=0.0,
        onset_weight=0.0,
        onset_baseline_window=200,
        onset_z=2.0,
        event_component_normalize=False,
        graph_penalty_weight=0.0,
        hierarchical_mode="off",
        hierarchical_group_topk=0,
        hierarchical_group_boost=0.0,
        hierarchical_outside_penalty=0.0,
        hierarchical_group_aggregation="max",
    ):
        events = []
        for event_id, (start, end) in enumerate(segments, start=1):
            if end <= start:
                continue
            score_start, score_end = self._rca_event_score_bounds(
                start,
                end,
                event_head_ratio,
                event_head_points,
            )
            event_raw_scores = channel_scores[score_start:score_end].mean(axis=0)
            event_contrast_scores = np.zeros_like(event_raw_scores)
            if contrast_window > 0 and contrast_weight > 0.0 and start > 0:
                baseline_start = max(0, start - contrast_window)
                baseline_scores = channel_scores[baseline_start:start].mean(axis=0)
                event_contrast_scores = np.maximum(event_raw_scores - baseline_scores, 0.0)
            event_scores = event_raw_scores + contrast_weight * event_contrast_scores
            event_base_scores = base_channel_scores[score_start:score_end].mean(axis=0)
            event_graph_scores = graph_channel_scores[score_start:score_end].mean(axis=0)
            event_mechanism_scores = mechanism_channel_scores[score_start:score_end].mean(axis=0)
            event_source_scores = (
                source_channel_scores[score_start:score_end].mean(axis=0)
                if source_channel_scores is not None
                else event_raw_scores
            )
            event_propagation_scores = (
                propagation_channel_scores[score_start:score_end].mean(axis=0)
                if propagation_channel_scores is not None
                else event_graph_scores
            )
            event_causal_scores = (
                causal_channel_scores[score_start:score_end].mean(axis=0)
                if causal_channel_scores is not None
                else np.zeros_like(event_raw_scores)
            )
            event_source_gate_scores = (
                source_gate_channel_scores[score_start:score_end].mean(axis=0)
                if source_gate_channel_scores is not None
                else np.zeros_like(event_raw_scores)
            )
            event_synthetic_scores = (
                synthetic_channel_scores[score_start:score_end].mean(axis=0)
                if synthetic_channel_scores is not None
                else np.zeros_like(event_raw_scores)
            )
            event_counterfactual_scores = (
                counterfactual_channel_scores[score_start:score_end].mean(axis=0)
                if counterfactual_channel_scores is not None
                else np.zeros_like(event_raw_scores)
            )
            onset_source_scores = source_channel_scores if source_channel_scores is not None else base_channel_scores
            event_onset_scores = np.zeros_like(event_raw_scores)
            if onset_weight > 0.0:
                event_onset_scores = self._event_onset_scores(
                    onset_source_scores,
                    start,
                    end,
                    baseline_window=onset_baseline_window,
                    onset_z=onset_z,
                )
                event_scores = event_scores + onset_weight * event_onset_scores
            event_mechanism_residual_scores = np.zeros_like(event_mechanism_scores)
            if mechanism_residual_window > 0 and mechanism_residual_weight > 0.0 and start > 0:
                baseline_start = max(0, start - mechanism_residual_window)
                baseline_mechanism_scores = mechanism_channel_scores[baseline_start:start].mean(axis=0)
                event_mechanism_residual_scores = np.maximum(
                    event_mechanism_scores - baseline_mechanism_scores,
                    0.0,
                )
                event_scores = event_scores + mechanism_residual_weight * event_mechanism_residual_scores
            if event_component_normalize:
                base_norm = self._normalize_event_component(event_base_scores)
                graph_norm = self._normalize_event_component(event_graph_scores)
                mechanism_norm = self._normalize_event_component(event_mechanism_scores)
                causal_norm = self._normalize_event_component(event_causal_scores)
                source_gate_norm = self._normalize_event_component(event_source_gate_scores)
                synthetic_norm = self._normalize_event_component(event_synthetic_scores)
                counterfactual_norm = self._normalize_event_component(event_counterfactual_scores)
                onset_norm = self._normalize_event_component(event_onset_scores)
                mechanism_residual_norm = self._normalize_event_component(event_mechanism_residual_scores)
                contrast_norm = self._normalize_event_component(event_contrast_scores)
                source_norm = (
                    source_base_weight * base_norm
                    + source_mechanism_weight * mechanism_norm
                    + causal_weight * causal_norm
                    + source_gate_weight * source_gate_norm
                    + synthetic_weight * synthetic_norm
                    + counterfactual_weight * counterfactual_norm
                )
                if source_interaction_weight > 0.0:
                    source_evidence_norm = np.maximum(onset_norm, mechanism_residual_norm)
                    source_norm = source_norm + source_interaction_weight * base_norm * source_evidence_norm
                if use_source_propagation:
                    event_scores = source_weight * source_norm + propagation_weight * graph_norm
                else:
                    event_scores = (
                        base_norm
                        + graph_weight * graph_norm
                        + mechanism_weight * mechanism_norm
                        + source_gate_weight * source_gate_norm
                        + synthetic_weight * synthetic_norm
                        + counterfactual_weight * counterfactual_norm
                    )
                event_scores = (
                    event_scores
                    + onset_weight * onset_norm
                    + mechanism_residual_weight * mechanism_residual_norm
                    + contrast_weight * contrast_norm
                    - graph_penalty_weight * graph_norm
                )
            group_values_by_name = {}
            for name, value in zip(feature_names, event_scores):
                group = self._root_cause_group_name(name)
                group_values_by_name.setdefault(group, []).append(float(value))
            aggregation = str(hierarchical_group_aggregation or "max").lower()
            group_scores = {}
            for group, values in group_values_by_name.items():
                values = sorted(values, reverse=True)
                if aggregation == "mean":
                    group_scores[group] = float(np.mean(values))
                elif aggregation == "topk_mean":
                    k = max(1, int(hierarchical_group_topk or 3))
                    group_scores[group] = float(np.mean(values[: min(k, len(values))]))
                else:
                    group_scores[group] = float(values[0])
            sorted_groups = sorted(group_scores.items(), key=lambda item: item[1], reverse=True)
            group_rank = {name: int(rank + 1) for rank, (name, _) in enumerate(sorted_groups)}
            group_values = np.asarray([value for _, value in sorted_groups], dtype=np.float64)
            if group_values.size and float(group_values.max() - group_values.min()) > 1e-12:
                group_norm = {
                    name: float((value - group_values.min()) / (group_values.max() - group_values.min()))
                    for name, value in sorted_groups
                }
            else:
                group_norm = {name: 0.0 for name, _ in sorted_groups}

            hierarchical_mode = str(hierarchical_mode or "off").lower()
            hierarchical_scores = np.asarray(event_scores, dtype=np.float64).copy()
            topk = max(0, int(hierarchical_group_topk or 0))
            if hierarchical_mode == "soft":
                top_groups = set(group_rank)
                if topk > 0:
                    top_groups = {name for name, rank in group_rank.items() if rank <= topk}
                for idx, name in enumerate(feature_names):
                    group = self._root_cause_group_name(name)
                    hierarchical_scores[idx] += float(hierarchical_group_boost) * group_norm.get(group, 0.0)
                    if topk > 0 and group not in top_groups:
                        hierarchical_scores[idx] -= float(hierarchical_outside_penalty)
                order = np.argsort(-hierarchical_scores)
            elif hierarchical_mode == "strict":
                order = np.asarray(
                    sorted(
                        range(len(feature_names)),
                        key=lambda idx: (
                            group_rank.get(self._root_cause_group_name(feature_names[idx]), 10**9),
                            -float(event_scores[idx]),
                        ),
                    ),
                    dtype=np.int64,
                )
            else:
                order = np.argsort(-event_scores)

            within_group_rank = {}
            for group, _ in sorted_groups:
                indices = [
                    idx
                    for idx, name in enumerate(feature_names)
                    if self._root_cause_group_name(name) == group
                ]
                indices = sorted(indices, key=lambda idx: float(event_scores[idx]), reverse=True)
                for rank, idx in enumerate(indices, start=1):
                    within_group_rank[idx] = int(rank)

            channel_ranking = [
                {
                    "rank": int(rank + 1),
                    "name": feature_names[idx],
                    "score": float(event_scores[idx]),
                    "hierarchical_score": float(hierarchical_scores[idx]),
                    "group": self._root_cause_group_name(feature_names[idx]),
                    "group_rank": int(group_rank.get(self._root_cause_group_name(feature_names[idx]), 0)),
                    "group_score": float(group_scores.get(self._root_cause_group_name(feature_names[idx]), 0.0)),
                    "within_group_rank": int(within_group_rank.get(idx, 0)),
                    "base_score": float(event_base_scores[idx]),
                    "graph_score": float(event_graph_scores[idx]),
                    "mechanism_score": float(event_mechanism_scores[idx]),
                    "source_score": float(event_source_scores[idx]),
                    "propagation_score": float(event_propagation_scores[idx]),
                    "causal_score": float(event_causal_scores[idx]),
                    "source_gate_score": float(event_source_gate_scores[idx]),
                    "synthetic_rca_score": float(event_synthetic_scores[idx]),
                    "counterfactual_score": float(event_counterfactual_scores[idx]),
                    "onset_score": float(event_onset_scores[idx]),
                    "mechanism_residual_score": float(event_mechanism_residual_scores[idx]),
                    "contrast_score": float(event_contrast_scores[idx]),
                }
                for rank, idx in enumerate(order)
            ]
            if max_channel_ranking is not None:
                channel_ranking = channel_ranking[:max(1, int(max_channel_ranking))]
            group_ranking = [
                {"rank": int(rank + 1), "name": name, "score": float(value)}
                for rank, (name, value) in enumerate(sorted_groups)
            ]
            events.append(
                {
                    "event_id": event_id,
                    "start": int(start),
                    "end": int(end),
                    "length": int(end - start),
                    "score_start": int(score_start),
                    "score_end": int(score_end),
                    "score_length": int(score_end - score_start),
                    "top_channels": channel_ranking[:20],
                    "channel_ranking": channel_ranking,
                    "group_ranking": group_ranking,
                }
            )
        return events

    @staticmethod
    def _rca_event_score_bounds(start, end, event_head_ratio=1.0, event_head_points=0):
        length = max(0, int(end) - int(start))
        if length <= 0:
            return int(start), int(end)
        head_len = length
        ratio = float(event_head_ratio if event_head_ratio is not None else 1.0)
        if 0.0 < ratio < 1.0:
            head_len = min(head_len, int(np.ceil(length * ratio)))
        points = int(event_head_points or 0)
        if points > 0:
            head_len = min(head_len, points)
        head_len = max(1, head_len)
        return int(start), int(start) + head_len

    def export_root_cause_report(self, series_name, test_data, test_label, predict_labels=None, scores=None):
        if not bool(getattr(self.config, "export_rca", False)):
            return None
        event_local_rca = bool(getattr(self.config, "rca_event_local_export", False))

        labels = test_label.to_numpy().reshape(-1).astype(int)
        rca_offset = 0
        try:
            from ts_benchmark.common.constant import ANOMALY_DETECT_DATASET_PATH
            meta_path = os.path.join(ANOMALY_DETECT_DATASET_PATH, "DETECT_META.csv")
            meta_df = pd.read_csv(meta_path)
            row = meta_df.loc[meta_df["file_name"] == series_name]
            if not row.empty and "train_lens" in row.columns and pd.notna(row.iloc[0]["train_lens"]):
                candidate_offset = int(row.iloc[0]["train_lens"])
                if 0 < candidate_offset < len(labels):
                    rca_offset = candidate_offset
        except Exception:
            rca_offset = 0

        if rca_offset > 0 and len(labels) == len(test_data):
            labels = labels[rca_offset:]
        elif len(labels) > len(test_data):
            labels = labels[-len(test_data):]
        elif len(labels) < len(test_data):
            labels = np.pad(labels, (0, len(test_data) - len(labels)), mode="constant")
        score_slice = slice(rca_offset, rca_offset + len(labels)) if rca_offset > 0 else slice(0, len(labels))
        pred_key, pred_mask = self._select_rca_prediction_mask(predict_labels, len(labels))
        if pred_mask is not None and rca_offset > 0:
            requested_prediction = None
            if isinstance(predict_labels, dict):
                for key, prediction in predict_labels.items():
                    if self._rca_prediction_key_name(key) == pred_key:
                        requested_prediction = prediction
                        break
            elif predict_labels is not None:
                requested_prediction = predict_labels
            if requested_prediction is not None:
                raw_mask = self._normalize_prediction_mask(requested_prediction, len(test_data))
                pred_mask = raw_mask[rca_offset:rca_offset + len(labels)]

        if (self._last_channel_scores is None or self._last_channel_names is None) and event_local_rca:
            self._compute_event_local_rca_channel_scores(
                test_data,
                labels,
                pred_mask=pred_mask,
                rca_offset=rca_offset,
            )
        if self._last_channel_scores is None or self._last_channel_names is None:
            return None

        base_channel_scores = self._last_channel_scores[score_slice]
        graph_channel_scores = self._last_graph_channel_scores
        if graph_channel_scores is None:
            graph_channel_scores = np.zeros_like(base_channel_scores)
        else:
            graph_channel_scores = graph_channel_scores[score_slice]
        mechanism_channel_scores = getattr(self, "_last_mechanism_channel_scores", None)
        if mechanism_channel_scores is None:
            mechanism_channel_scores = np.zeros_like(base_channel_scores)
        else:
            mechanism_channel_scores = mechanism_channel_scores[score_slice]
        causal_channel_scores = getattr(self, "_last_causal_channel_scores", None)
        if causal_channel_scores is None:
            causal_channel_scores = np.zeros_like(base_channel_scores)
        else:
            causal_channel_scores = causal_channel_scores[score_slice]
        source_gate_channel_scores = getattr(self, "_last_source_gate_channel_scores", None)
        if source_gate_channel_scores is None:
            source_gate_channel_scores = np.zeros_like(base_channel_scores)
        else:
            source_gate_channel_scores = source_gate_channel_scores[score_slice]
        synthetic_channel_scores = getattr(self, "_last_synthetic_rca_channel_scores", None)
        if synthetic_channel_scores is None:
            synthetic_channel_scores = np.zeros_like(base_channel_scores)
        else:
            synthetic_channel_scores = synthetic_channel_scores[score_slice]
        def _cfg_float(name, default):
            value = getattr(self.config, name, default)
            return float(default if value is None else value)

        def _cfg_int(name, default):
            value = getattr(self.config, name, default)
            return int(default if value is None else value)

        graph_weight = _cfg_float("rca_graph_weight", 0.0)
        mechanism_weight = _cfg_float("rca_mechanism_weight", 0.0)
        use_source_propagation = bool(getattr(self.config, "rca_use_source_propagation", False))
        source_weight = _cfg_float("rca_source_weight", 0.75)
        source_base_weight = _cfg_float("rca_source_base_weight", 1.0)
        propagation_weight = _cfg_float("rca_propagation_weight", 0.25)
        source_mechanism_weight = _cfg_float("rca_source_mechanism_weight", 0.0)
        causal_weight = _cfg_float("rca_causal_weight", 0.0)
        synthetic_weight = _cfg_float("rca_synthetic_weight", 0.0)
        source_gate_weight = _cfg_float("rca_source_gate_weight", 0.0)
        source_interaction_weight = _cfg_float("rca_source_interaction_weight", 0.0)
        counterfactual_weight = _cfg_float("rca_counterfactual_weight", 0.0)
        contrast_window = _cfg_int("rca_contrast_window", 0)
        contrast_weight = _cfg_float("rca_contrast_weight", 0.0)
        mechanism_residual_window = _cfg_int("rca_mechanism_residual_window", 0)
        mechanism_residual_weight = _cfg_float("rca_mechanism_residual_weight", 0.0)
        event_head_ratio = _cfg_float("rca_event_head_ratio", 1.0)
        event_head_points = _cfg_int("rca_event_head_points", 0)
        onset_weight = _cfg_float("rca_onset_weight", 0.0)
        onset_baseline_window = _cfg_int("rca_onset_baseline_window", 200)
        onset_z = _cfg_float("rca_onset_z", 2.0)
        event_component_normalize = bool(getattr(self.config, "rca_event_component_normalize", False))
        graph_penalty_weight = _cfg_float("rca_graph_penalty_weight", 0.0)
        hierarchical_mode = str(getattr(self.config, "rca_hierarchical_mode", "off") or "off")
        hierarchical_group_topk = _cfg_int("rca_hierarchical_group_topk", 0)
        hierarchical_group_boost = _cfg_float("rca_hierarchical_group_boost", 0.0)
        hierarchical_outside_penalty = _cfg_float("rca_hierarchical_outside_penalty", 0.0)
        hierarchical_group_aggregation = str(
            getattr(self.config, "rca_hierarchical_group_aggregation", "max") or "max"
        )
        export_lite = bool(getattr(self.config, "rca_export_lite", False))
        export_top_k = int(getattr(self.config, "rca_export_top_k", 20) or 20)
        max_channel_ranking = export_top_k if export_lite else None
        source_channel_scores = None
        propagation_channel_scores = None
        counterfactual_channel_scores = np.zeros_like(base_channel_scores)
        counterfactual_candidate_scores = (
            source_base_weight * base_channel_scores
            + source_mechanism_weight * mechanism_channel_scores
            + causal_weight * causal_channel_scores
            + source_gate_weight * source_gate_channel_scores
            + synthetic_weight * synthetic_channel_scores
        )
        if counterfactual_weight > 0.0:
            counterfactual_channel_scores = self._compute_counterfactual_channel_scores(
                test_data,
                labels,
                pred_mask,
                rca_offset,
                counterfactual_candidate_scores,
                event_head_ratio,
                event_head_points,
            )
        if use_source_propagation:
            source_channel_scores = (
                source_base_weight * base_channel_scores
                + source_mechanism_weight * mechanism_channel_scores
                + causal_weight * causal_channel_scores
                + source_gate_weight * source_gate_channel_scores
                + synthetic_weight * synthetic_channel_scores
                + counterfactual_weight * counterfactual_channel_scores
            )
            propagation_channel_scores = graph_channel_scores
            channel_scores = (
                source_weight * source_channel_scores
                + propagation_weight * propagation_channel_scores
            )
        else:
            channel_scores = (
                base_channel_scores
                + graph_weight * graph_channel_scores
                + mechanism_weight * mechanism_channel_scores
                + source_gate_weight * source_gate_channel_scores
                + synthetic_weight * synthetic_channel_scores
                + counterfactual_weight * counterfactual_channel_scores
            )
        feature_names = list(self._last_channel_names)
        events = self._build_rca_events(
            self._label_segments(labels),
            channel_scores,
            base_channel_scores,
            graph_channel_scores,
            mechanism_channel_scores,
            feature_names,
            contrast_window,
            contrast_weight,
            mechanism_residual_window,
            mechanism_residual_weight,
            event_head_ratio,
            event_head_points,
            max_channel_ranking=max_channel_ranking,
            source_channel_scores=source_channel_scores,
            propagation_channel_scores=propagation_channel_scores,
            causal_channel_scores=causal_channel_scores,
            source_gate_channel_scores=source_gate_channel_scores,
            synthetic_channel_scores=synthetic_channel_scores,
            counterfactual_channel_scores=counterfactual_channel_scores,
            use_source_propagation=use_source_propagation,
            graph_weight=graph_weight,
            mechanism_weight=mechanism_weight,
            source_weight=source_weight,
            source_base_weight=source_base_weight,
            propagation_weight=propagation_weight,
            source_mechanism_weight=source_mechanism_weight,
            causal_weight=causal_weight,
            synthetic_weight=synthetic_weight,
            source_gate_weight=source_gate_weight,
            source_interaction_weight=source_interaction_weight,
            counterfactual_weight=counterfactual_weight,
            onset_weight=onset_weight,
            onset_baseline_window=onset_baseline_window,
            onset_z=onset_z,
            event_component_normalize=event_component_normalize,
            graph_penalty_weight=graph_penalty_weight,
            hierarchical_mode=hierarchical_mode,
            hierarchical_group_topk=hierarchical_group_topk,
            hierarchical_group_boost=hierarchical_group_boost,
            hierarchical_outside_penalty=hierarchical_outside_penalty,
            hierarchical_group_aggregation=hierarchical_group_aggregation,
        )
        predicted_events_by_key = {}
        if export_lite:
            if pred_mask is not None:
                predicted_events_by_key[pred_key] = self._build_rca_events(
                    self._rca_predicted_segments(pred_mask),
                    channel_scores,
                    base_channel_scores,
                    graph_channel_scores,
                    mechanism_channel_scores,
                    feature_names,
                    contrast_window,
                    contrast_weight,
                    mechanism_residual_window,
                    mechanism_residual_weight,
                    event_head_ratio,
                    event_head_points,
                    max_channel_ranking=max_channel_ranking,
                    source_channel_scores=source_channel_scores,
                    propagation_channel_scores=propagation_channel_scores,
                    causal_channel_scores=causal_channel_scores,
                    source_gate_channel_scores=source_gate_channel_scores,
                    synthetic_channel_scores=synthetic_channel_scores,
                    counterfactual_channel_scores=counterfactual_channel_scores,
                    use_source_propagation=use_source_propagation,
                    graph_weight=graph_weight,
                    mechanism_weight=mechanism_weight,
                    source_weight=source_weight,
                    source_base_weight=source_base_weight,
                    propagation_weight=propagation_weight,
                    source_mechanism_weight=source_mechanism_weight,
                    causal_weight=causal_weight,
                    synthetic_weight=synthetic_weight,
                    source_gate_weight=source_gate_weight,
                    source_interaction_weight=source_interaction_weight,
                    counterfactual_weight=counterfactual_weight,
                    onset_weight=onset_weight,
                    onset_baseline_window=onset_baseline_window,
                    onset_z=onset_z,
                    event_component_normalize=event_component_normalize,
                    graph_penalty_weight=graph_penalty_weight,
                    hierarchical_mode=hierarchical_mode,
                    hierarchical_group_topk=hierarchical_group_topk,
                    hierarchical_group_boost=hierarchical_group_boost,
                    hierarchical_outside_penalty=hierarchical_outside_penalty,
                    hierarchical_group_aggregation=hierarchical_group_aggregation,
                )
        elif isinstance(predict_labels, dict):
            for key, prediction in predict_labels.items():
                key_name = self._rca_prediction_key_name(key)
                mask = self._normalize_prediction_mask(prediction, len(labels))
                if rca_offset > 0:
                    raw_mask = self._normalize_prediction_mask(prediction, len(test_data))
                    mask = raw_mask[rca_offset:rca_offset + len(labels)]
                predicted_events_by_key[key_name] = self._build_rca_events(
                    self._rca_predicted_segments(mask),
                    channel_scores,
                    base_channel_scores,
                    graph_channel_scores,
                    mechanism_channel_scores,
                    feature_names,
                    contrast_window,
                    contrast_weight,
                    mechanism_residual_window,
                    mechanism_residual_weight,
                    event_head_ratio,
                    event_head_points,
                    max_channel_ranking=max_channel_ranking,
                    source_channel_scores=source_channel_scores,
                    propagation_channel_scores=propagation_channel_scores,
                    causal_channel_scores=causal_channel_scores,
                    source_gate_channel_scores=source_gate_channel_scores,
                    synthetic_channel_scores=synthetic_channel_scores,
                    counterfactual_channel_scores=counterfactual_channel_scores,
                    use_source_propagation=use_source_propagation,
                    graph_weight=graph_weight,
                    mechanism_weight=mechanism_weight,
                    source_weight=source_weight,
                    source_base_weight=source_base_weight,
                    propagation_weight=propagation_weight,
                    source_mechanism_weight=source_mechanism_weight,
                    causal_weight=causal_weight,
                    synthetic_weight=synthetic_weight,
                    source_gate_weight=source_gate_weight,
                    source_interaction_weight=source_interaction_weight,
                    counterfactual_weight=counterfactual_weight,
                    onset_weight=onset_weight,
                    onset_baseline_window=onset_baseline_window,
                    onset_z=onset_z,
                    event_component_normalize=event_component_normalize,
                    graph_penalty_weight=graph_penalty_weight,
                    hierarchical_mode=hierarchical_mode,
                    hierarchical_group_topk=hierarchical_group_topk,
                    hierarchical_group_boost=hierarchical_group_boost,
                    hierarchical_outside_penalty=hierarchical_outside_penalty,
                    hierarchical_group_aggregation=hierarchical_group_aggregation,
                )
        elif predict_labels is not None:
            mask = self._normalize_prediction_mask(predict_labels, len(labels))
            if rca_offset > 0:
                raw_mask = self._normalize_prediction_mask(predict_labels, len(test_data))
                mask = raw_mask[rca_offset:rca_offset + len(labels)]
            predicted_events_by_key["single"] = self._build_rca_events(
                self._rca_predicted_segments(mask),
                channel_scores,
                base_channel_scores,
                graph_channel_scores,
                mechanism_channel_scores,
                feature_names,
                contrast_window,
                contrast_weight,
                mechanism_residual_window,
                mechanism_residual_weight,
                event_head_ratio,
                event_head_points,
                max_channel_ranking=max_channel_ranking,
                source_channel_scores=source_channel_scores,
                propagation_channel_scores=propagation_channel_scores,
                causal_channel_scores=causal_channel_scores,
                source_gate_channel_scores=source_gate_channel_scores,
                synthetic_channel_scores=synthetic_channel_scores,
                counterfactual_channel_scores=counterfactual_channel_scores,
                use_source_propagation=use_source_propagation,
                graph_weight=graph_weight,
                mechanism_weight=mechanism_weight,
                source_weight=source_weight,
                source_base_weight=source_base_weight,
                propagation_weight=propagation_weight,
                source_mechanism_weight=source_mechanism_weight,
                causal_weight=causal_weight,
                synthetic_weight=synthetic_weight,
                source_gate_weight=source_gate_weight,
                source_interaction_weight=source_interaction_weight,
                counterfactual_weight=counterfactual_weight,
                onset_weight=onset_weight,
                onset_baseline_window=onset_baseline_window,
                onset_z=onset_z,
                event_component_normalize=event_component_normalize,
                graph_penalty_weight=graph_penalty_weight,
                hierarchical_mode=hierarchical_mode,
                hierarchical_group_topk=hierarchical_group_topk,
                hierarchical_group_boost=hierarchical_group_boost,
                hierarchical_outside_penalty=hierarchical_outside_penalty,
                hierarchical_group_aggregation=hierarchical_group_aggregation,
            )
        predicted_events = predicted_events_by_key.get(pred_key, [])
        if not predicted_events and pred_mask is not None:
            predicted_events = self._build_rca_events(
                self._rca_predicted_segments(pred_mask),
                channel_scores,
                base_channel_scores,
                graph_channel_scores,
                mechanism_channel_scores,
                feature_names,
                contrast_window,
                contrast_weight,
                mechanism_residual_window,
                mechanism_residual_weight,
                event_head_ratio,
                event_head_points,
                max_channel_ranking=max_channel_ranking,
                source_channel_scores=source_channel_scores,
                propagation_channel_scores=propagation_channel_scores,
                causal_channel_scores=causal_channel_scores,
                source_gate_channel_scores=source_gate_channel_scores,
                synthetic_channel_scores=synthetic_channel_scores,
                counterfactual_channel_scores=counterfactual_channel_scores,
                use_source_propagation=use_source_propagation,
                graph_weight=graph_weight,
                mechanism_weight=mechanism_weight,
                source_weight=source_weight,
                source_base_weight=source_base_weight,
                propagation_weight=propagation_weight,
                source_mechanism_weight=source_mechanism_weight,
                causal_weight=causal_weight,
                synthetic_weight=synthetic_weight,
                source_gate_weight=source_gate_weight,
                source_interaction_weight=source_interaction_weight,
                counterfactual_weight=counterfactual_weight,
                onset_weight=onset_weight,
                onset_baseline_window=onset_baseline_window,
                onset_z=onset_z,
                event_component_normalize=event_component_normalize,
                graph_penalty_weight=graph_penalty_weight,
                hierarchical_mode=hierarchical_mode,
                hierarchical_group_topk=hierarchical_group_topk,
                hierarchical_group_boost=hierarchical_group_boost,
                hierarchical_outside_penalty=hierarchical_outside_penalty,
                hierarchical_group_aggregation=hierarchical_group_aggregation,
            )

        from datetime import datetime
        from ts_benchmark.common.constant import ROOT_PATH

        safe_name = os.path.splitext(os.path.basename(str(series_name)))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(ROOT_PATH, "result", "rca", safe_name)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{timestamp}_rca.json")
        score_method = (
            "event-level source/propagation RCA: "
            "source=(weighted base residual + mechanism prior deviation + lagged causal deviation + model source-gate score + synthetic responsibility + event-local counterfactual responsibility), "
            "propagation=graph-propagated residual, "
            "onset=early local-baseline crossing"
            if use_source_propagation
            else "mean channel-wise normalized reconstruction error plus graph-propagated, mechanism-violation, and local-contrast attribution"
        )
        payload = {
            "series_name": series_name,
            "dataset_name": self.dataset_name,
            "score_method": score_method,
            "rca_export_lite": export_lite,
            "rca_export_top_k": export_top_k if export_lite else None,
            "rca_graph_weight": graph_weight,
            "rca_mechanism_weight": mechanism_weight,
            "rca_use_source_propagation": use_source_propagation,
            "rca_source_weight": source_weight,
            "rca_source_base_weight": source_base_weight,
            "rca_propagation_weight": propagation_weight,
            "rca_source_mechanism_weight": source_mechanism_weight,
            "rca_causal_weight": causal_weight,
            "rca_synthetic_weight": synthetic_weight,
            "rca_source_gate_weight": source_gate_weight,
            "rca_source_interaction_weight": source_interaction_weight,
            "rca_counterfactual_weight": counterfactual_weight,
            "rca_counterfactual_candidates": _cfg_int("rca_counterfactual_candidates", 12),
            "rca_counterfactual_max_windows": _cfg_int("rca_counterfactual_max_windows", 32),
            "rca_counterfactual_batch_candidates": _cfg_int("rca_counterfactual_batch_candidates", 4),
            "rca_counterfactual_baseline_window": _cfg_int("rca_counterfactual_baseline_window", 300),
            "rca_offset": int(rca_offset),
            "rca_graph_direction": str(getattr(self.config, "rca_graph_direction", "outgoing") or "outgoing"),
            "rca_contrast_window": contrast_window,
            "rca_contrast_weight": contrast_weight,
            "rca_mechanism_residual_window": mechanism_residual_window,
            "rca_mechanism_residual_weight": mechanism_residual_weight,
            "rca_event_component_normalize": event_component_normalize,
            "rca_graph_penalty_weight": graph_penalty_weight,
            "rca_hierarchical_mode": hierarchical_mode,
            "rca_hierarchical_group_topk": hierarchical_group_topk,
            "rca_hierarchical_group_boost": hierarchical_group_boost,
            "rca_hierarchical_outside_penalty": hierarchical_outside_penalty,
            "rca_hierarchical_group_aggregation": hierarchical_group_aggregation,
            "rca_event_head_ratio": event_head_ratio,
            "rca_event_head_points": event_head_points,
            "rca_onset_weight": onset_weight,
            "rca_onset_baseline_window": onset_baseline_window,
            "rca_onset_z": onset_z,
            "rca_prediction_key": pred_key,
            "rca_split_predicted_events": bool(getattr(self.config, "rca_split_predicted_events", False)),
            "rca_split_max_event_len": _cfg_int("rca_split_max_event_len", 120),
            "rca_split_stride": _cfg_int("rca_split_stride", 60),
            "feature_names": feature_names,
            "events": events,
            "predicted_events": predicted_events,
            "predicted_events_by_key": predicted_events_by_key,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"  [RCA] Root-cause ranking -> {output_path}")
        return output_path

    def evaluate(self, test_data: pd.DataFrame, test_labels: np.ndarray,
                 save_path: str = None) -> dict:
        """
        完整评估报告：AUC-ROC + AUPR + 各 ratio F1。

        Args:
            test_data: 测试数据
            test_labels: 真实 0/1 标签 (N,)
            save_path: 保存路径（可选）

        Returns:
            report: 包含所有指标的 dict
        """
        preds, scores = self.detect_label(test_data)
        scores = np.asarray(scores, dtype=np.float64).flatten()
        labels = np.asarray(test_labels, dtype=np.int32).flatten()

        # 对齐长度
        min_len = min(len(scores), len(labels))
        scores, labels = scores[:min_len], labels[:min_len]

        # 计算指标
        report = {
            'dataset': self.dataset_name,
            'model': 'LaGraph-v11.4',
            'n_samples': len(scores),
            'anomaly_ratio_gt': float(labels.mean() * 100),
        }

        # AUC-ROC
        if len(np.unique(labels)) >= 2:
            try:
                report['auc_roc'] = float(roc_auc_score(labels, scores))
            except Exception:
                report['auc_roc'] = 0.5
        else:
            report['auc_roc'] = 0.5

        # AUPR
        try:
            report['aupr'] = float(average_precision_score(labels, scores))
        except Exception:
            report['aupr'] = float(labels.mean())

        # 各 ratio 的 F1
        report['f1_per_ratio'] = {}
        for ratio, pred in preds.items():
            pred = pred[:min_len].flatten().astype(int)
            tp = (pred * labels).sum()
            fp = pred.sum() - tp
            fn = labels.sum() - tp
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-10)
            report['f1_per_ratio'][ratio] = {
                'f1': float(f1),
                'precision': float(prec),
                'recall': float(rec),
                'threshold': None,  # 由 detect_label 决定
            }

        # 输出报告
        print(f"\n{'='*60}")
        print(f"  ★ Evaluation Report: {self.dataset_name}")
        print(f"{'='*60}")
        print(f"  AUC-ROC:  {report['auc_roc']:.4f}")
        print(f"  AUPR:     {report['aupr']:.4f}")
        print(f"  GT Anomaly Ratio: {report['anomaly_ratio_gt']:.2f}%")
        for ratio, f1_data in report['f1_per_ratio'].items():
            print(f"  F1@{ratio}%: {f1_data['f1']:.4f}")
        print(f"{'='*60}\n")

        if save_path:
            import json
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2, cls=_NumpyEncoder)
            print(f"  [Report] Saved to {save_path}")

        return report

    # ════════════════════════════════════════════════════════════════
    #  ★ P0-4: 多 seed 实验运行接口
    # ════════════════════════════════════════════════════════════════

    def run_multi_seed_experiment(self, train_data: pd.DataFrame,
                                   test_data: pd.DataFrame,
                                   test_labels: np.ndarray,
                                   num_seeds: int = None) -> dict:
        """
        多次运行训练+检测（不同 seed），统计指标。

        Args:
            train_data: 训练数据
            test_data: 测试数据
            test_labels: 真实标签
            num_seeds: 运行次数（默认 self.config.num_seeds）

        Returns:
            summary: 统计总结
        """
        if num_seeds is None:
            num_seeds = self.config.num_seeds

        all_results = []
        for seed_idx in range(num_seeds):
            seed = self.config.seed_base + seed_idx
            print(f"\n{'='*60}")
            print(f"  Seed {seed_idx + 1}/{num_seeds} (base={seed})")
            print(f"{'='*60}")

            # 设置随机种子
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            # 创建新模型实例
            model = LaGraph(**{
                k: getattr(self.config, k)
                for k in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS
            })
            model.config.dataset_name = self.dataset_name

            # 训练
            model.detect_fit(train_data, test_data)

            # 评估
            report = model.evaluate(test_data, test_labels)
            all_results.append(report)

        # 汇总统计
        summary = self._aggregate_seed_results(all_results)
        return summary

    def _aggregate_seed_results(self, results: list) -> dict:
        """聚合多次 seed 实验的结果。"""
        if not results:
            return {}

        metrics = ['auc_roc', 'aupr']
        summary = {
            'num_seeds': len(results),
            'dataset': self.dataset_name,
        }

        for metric in metrics:
            values = [r.get(metric, 0.0) for r in results]
            summary[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values, ddof=1)),
                'values': [float(v) for v in values],
            }
            if len(values) >= 2:
                se = np.std(values, ddof=1) / np.sqrt(len(values))
                ci = scipy_stats.t.ppf(0.975, len(values) - 1) * se
                summary[metric]['ci_95'] = float(ci)
            else:
                summary[metric]['ci_95'] = 0.0

        # F1 聚合
        f1_ratios = set()
        for r in results:
            f1_ratios.update(r.get('f1_per_ratio', {}).keys())
        summary['f1_per_ratio'] = {}
        for ratio in sorted(f1_ratios):
            f1_vals = [r['f1_per_ratio'].get(ratio, {}).get('f1', 0.0) for r in results]
            summary['f1_per_ratio'][ratio] = {
                'mean': float(np.mean(f1_vals)),
                'std': float(np.std(f1_vals, ddof=1)),
            }

        # 打印统计报告
        print(f"\n{'='*60}")
        print(f"  ★ Multi-Seed Summary ({len(results)} seeds)")
        print(f"{'='*60}")
        for metric in metrics:
            m = summary[metric]
            print(f"  {metric.upper():>8s}: {m['mean']:.4f} ± {m['std']:.4f} "
                  f"(95% CI: ±{m.get('ci_95', 0):.4f})")
        print(f"{'='*60}\n")

        return summary

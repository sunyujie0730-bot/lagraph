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
import math
import os
import random
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
from torch.utils.data import DataLoader, Subset

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
    "debug_loss_breakdown": False,
    "debug_loss_log_path": "",
    "debug_loss_max_batches": 0,
    "debug_train_max_batches": 0,
    "patience": 15,
    "checkpoint_policy": "best_val_loss",
    "use_latest_checkpoint": False,
    "use_rca_aware_checkpoint": False,
    "rca_checkpoint_normalize": False,
    "rca_checkpoint_proxy_weight": 0.0,
    "rca_checkpoint_proxy_batches": 2,
    "rca_checkpoint_min_epoch": 1,
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
    "channel_graph_evidence_only": False,
    "channel_graph_type": "adaptive_symmetric",
    "directed_graph_parent_topk": 5,
    "directed_graph_embedding_dim": 8,
    "directed_graph_use_signed_transfer": False,
    "directed_graph_transfer_mode": "low_rank",
    "directed_graph_transfer_rank": 8,
    "directed_graph_transfer_scale": 2.0,
    "use_channel_graph_reliability": False,
    "channel_graph_role": "shared",
    "use_multilag_graph_propagation": False,
    "graph_propagation_lags": [0, 1, 2, 4, 8],
    "graph_propagation_lag_mode": "static",
    "graph_propagation_alignment_temperature": 0.2,
    "channel_corr_prior_selection_axis": "source",
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
    "lambda_masked_rca_head": 0.0,
    "masked_rca_bce_weight": 1.0,
    "masked_rca_rank_weight": 1.0,
    "synthetic_rca_margin": 0.2,
    "synthetic_rca_topk": 5,
    "synthetic_rca_positive_aggregation": "mean",
    "synthetic_rca_positive_topk": 1,
    "synthetic_rca_positive_min_weight": 0.0,
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
    "source_effect_aux_batch_size": 0,
    "source_effect_min_len": 8,
    "source_effect_max_len": 30,
    "source_effect_min_roots": 1,
    "source_effect_max_roots": 2,
    "source_effect_neighbor_topk": 3,
    "source_effect_use_channel_prior": False,
    "source_effect_prior_topk": 5,
    "source_effect_strength": 0.35,
    "source_effect_delay_max": 6,
    "source_effect_bce_weight": 1.0,
    "source_effect_rank_weight": 1.0,
    "source_effect_effect_rank_weight": 0.5,
    "source_effect_onset_rank_weight": 0.0,
    "source_effect_graph_alignment_weight": 0.0,
    "source_effect_graph_reverse_weight": 0.5,
    "source_effect_margin": 0.2,
    "source_effect_onset_margin": 0.2,
    "source_effect_specificity_weight": 0.0,
    "source_effect_consistency_weight": 0.0,
    "source_effect_consistency_rank_weight": 0.75,
    "source_effect_consistency_effect_rank_weight": 0.75,
    "source_effect_consistency_effect_suppress_weight": 0.25,
    "source_effect_consistency_gate_align_weight": 0.20,
    "source_effect_consistency_margin": 0.15,
    "lambda_source_effect_rca_head": 0.0,
    "source_effect_rca_head_bce_weight": 1.0,
    "source_effect_rca_head_rank_weight": 1.0,
    "use_root_score_head": False,
    "root_score_head_mode": "mlp",
    "root_score_detach_features": True,
    "root_response_penalty_init": 1.0,
    "root_response_confidence_discount": 0.75,
    "root_response_use_innovation_split": False,
    "use_pairwise_root_response_head": False,
    "pairwise_root_response_rank": 8,
    "pairwise_root_response_graph_weight": 0.75,
    "pairwise_root_response_reward_init": 0.25,
    "pairwise_root_response_penalty_init": 0.50,
    "pairwise_root_response_logit_weight": 1.0,
    "lambda_pairwise_root_response": 0.0,
    "lambda_source_effect_root_score": 0.0,
    "root_score_bce_weight": 1.0,
    "root_score_rank_weight": 1.0,
    "root_score_effect_suppress_weight": 0.5,
    "root_response_source_branch_weight": 0.0,
    "root_response_response_branch_weight": 0.0,
    "root_response_response_suppress_weight": 0.5,
    "use_event_responsibility_head": False,
    "event_responsibility_hidden": 16,
    "event_responsibility_detach_features": False,
    "lambda_event_responsibility": 0.0,
    "event_responsibility_loss_mode": "bce",
    "event_responsibility_bce_balance": True,
    "event_responsibility_ce_weight": 1.0,
    "event_responsibility_rank_weight": 0.5,
    "event_responsibility_effect_suppress_weight": 0.25,
    "event_responsibility_entropy_weight": 0.0,
    "use_event_route_head": False,
    "event_route_hidden": 16,
    "event_route_init": 0.55,
    "event_route_detach_inputs": True,
    "event_route_response_penalty": 0.0,
    "lambda_event_route": 0.0,
    "event_route_rank_weight": 1.0,
    "event_route_effect_suppress_weight": 0.25,
    "event_route_pairwise_effect_weight": 0.50,
    "event_route_pairwise_effect_margin": 0.15,
    "event_route_alpha_weight": 0.25,
    "rca_event_route_weight": 0.0,
    "use_response_suppressor_head": False,
    "response_suppressor_hidden": 16,
    "response_suppressor_detach_inputs": True,
    "lambda_response_suppressor": 0.0,
    "response_suppressor_bce_weight": 1.0,
    "response_suppressor_rank_weight": 0.5,
    "response_suppressor_source_leak_weight": 0.5,
    "use_evidence_fusion_head": False,
    "evidence_fusion_hidden": 16,
    "evidence_fusion_detach_inputs": False,
    "evidence_fusion_use_graph_response": False,
    "evidence_fusion_loss_mode": "bce",
    "evidence_fusion_export_mode": "sigmoid",
    "lambda_evidence_fusion": 0.0,
    "evidence_fusion_bce_weight": 1.0,
    "evidence_fusion_rank_weight": 1.0,
    "evidence_fusion_effect_suppress_weight": 0.25,
    "evidence_fusion_pairwise_effect_weight": 0.0,
    "evidence_fusion_pairwise_effect_margin": 0.2,
    "evidence_fusion_entropy_weight": 0.0,
    "use_source_interaction_head": False,
    "source_interaction_detach_inputs": True,
    "lambda_source_interaction_head": 0.0,
    "source_interaction_head_bce_weight": 1.0,
    "source_interaction_head_rank_weight": 1.0,
    "source_interaction_head_effect_suppress_weight": 0.25,
    "source_interaction_head_pairwise_effect_weight": 0.5,
    "source_interaction_head_pairwise_effect_margin": 0.15,
    "use_source_consistency_head": False,
    "source_consistency_detach_inputs": True,
    "lambda_source_consistency_head": 0.0,
    "source_consistency_head_bce_weight": 0.75,
    "source_consistency_head_rank_weight": 1.0,
    "source_consistency_head_effect_suppress_weight": 0.25,
    "source_consistency_head_pairwise_effect_weight": 0.50,
    "source_consistency_head_pairwise_effect_margin": 0.15,
    "lambda_source_bottleneck": 0.0,
    "source_bottleneck_bce_weight": 1.0,
    "source_bottleneck_rank_weight": 0.5,
    "source_bottleneck_effect_suppress_weight": 0.5,
    "source_bottleneck_effect_margin_weight": 0.0,
    "source_bottleneck_effect_margin": 0.10,
    "source_bottleneck_specificity_weight": 0.0,
    "use_channel_masked_modeling": False,
    "lambda_channel_masked": 0.0,
    "channel_mask_interval": 8,
    "channel_mask_aux_batch_size": 0,
    "channel_mask_ratio": 0.15,
    "channel_mask_min_channels": 1,
    "channel_mask_value": "zero",
    "use_interventional_channel_masking": False,
    "lambda_interventional_source_bce": 0.0,
    "lambda_interventional_source_rank": 0.0,
    "lambda_interventional_graph_support": 0.0,
    "interventional_rank_signal": "source_gate",
    "interventional_graph_support_eps": 1e-6,
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
    "use_causal_response_evidence": False,
    "lambda_causal_response": 0.0,
    "causal_response_margin": 0.10,
    "use_strict_cross_mechanism": False,
    "strict_cross_lags": [1, 3, 6, 12],
    "strict_cross_topk": 5,
    "strict_cross_detach_backbone": True,
    "strict_cross_use_channel_prior": False,
    "use_sps_role_head": False,
    "sps_role_hidden": 16,
    "sps_role_detach_features": False,
    "use_bounded_sps_fusion": False,
    "bounded_sps_max_correction": 0.25,
    "sps_lr_scale": 10.0,
    "lambda_strict_cross_mechanism": 0.0,
    "lambda_strict_cross_sparse": 0.0,
    "lambda_sps_source": 0.0,
    "lambda_sps_response": 0.0,
    "lambda_sps_separation": 0.0,
    "sps_separation_margin": 0.10,
    "use_sps_teacher": False,
    "use_sps_residual_synthetic": False,
    "lambda_residual_sps": 0.0,
    "residual_sps_aux_batch_size": 0,
    "residual_sps_source_weight": 1.0,
    "residual_sps_response_weight": 0.5,
    "residual_sps_separation_weight": 0.5,
    "sps_teacher_lags": [1, 3, 6, 12],
    "sps_teacher_topk": 3,
    "sps_teacher_ridge": 0.10,
    "sps_teacher_max_samples": 30000,
    "sps_teacher_max_gain": 0.65,
    "sps_teacher_response_ratio": 0.15,
    "sps_teacher_clip": 8.0,
    "sps_teacher_onset_len": 2,
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
    "score_stats_max_samples": 2_000_000,
    "score_stats_max_batches": 128,
    "score_stats_num_workers": 0,
    "use_mechanism_coupled_decoder": False,
    "mechanism_coupling_init": 0.15,
    "use_source_preserving_decoder": False,
    "source_preserving_init": 0.65,
    "source_preserving_detach_gate": True,
    "use_mechanism_residual_feedback": False,
    "mechanism_feedback_init": 0.10,
    "mechanism_feedback_detach": True,
    "mechanism_feedback_norm": "sample_l1",
    "mechanism_feedback_clip": 3.0,
    "use_mechanism_predictive_head": False,
    "mechanism_predictive_blend_init": 0.30,
    "use_source_gate": False,
    "source_gate_init": 0.20,
    "use_onset_aware_source_gate": False,
    "source_gate_onset_window": 8,
    "source_gate_onset_weight": 0.5,
    "use_dual_expert_fusion": False,
    "dual_expert_hidden": 16,
    "dual_expert_graph_init": 0.5,
    "dual_expert_detach_inputs": True,
    "lambda_dual_expert_selection": 0.0,
    "dual_expert_selection_temperature": 0.25,
    "dual_expert_balance_weight": 0.0,
    "dual_expert_min_usage": 0.10,
    "dual_expert_propagated_event_prob": 1.0,
    "dual_expert_mode_target_weight": 0.0,
    "use_source_expert_branch": False,
    "source_expert_hidden": 16,
    "source_expert_graph_init": 0.5,
    "source_expert_detach_gate_inputs": True,
    "source_expert_gradient_checkpoint": False,
    "source_expert_gate_scope": "time",
    "lambda_source_expert": 0.0,
    "lambda_source_expert_gate": 0.0,
    "source_expert_effect_suppress_weight": 0.25,
    "source_expert_pairwise_effect_weight": 0.50,
    "source_expert_pairwise_effect_margin": 0.15,
    "use_channel_temporal_corefinement": False,
    "corefinement_init": 0.10,
    "corefinement_detach_first_pass": True,
    "use_source_aware_corefinement": False,
    "source_aware_corefinement_init": 0.15,
    "source_aware_corefinement_detach_gate": True,
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
    "inference_dataloader_num_workers": 0,
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
    "rca_source_score_weight": 0.0,
    "rca_source_consensus_weight": 0.0,
    "rca_onset_consensus_weight": 0.0,
    "rca_source_consensus_mode": "sqrt_bg",
    "rca_causal_weight": 0.0,
    "rca_synthetic_weight": 0.0,
    "rca_source_gate_weight": 0.0,
    "rca_source_gate_evidence_guard_weight": 0.0,
    "rca_source_gate_evidence_guard_floor": 0.0,
    "rca_root_score_weight": 0.0,
    "rca_event_responsibility_weight": 0.0,
    "rca_root_score_signal": "prob",
    "rca_root_score_pooling": "mean",
    "rca_root_score_head_ratio": 0.30,
    "rca_root_score_head_points": 30,
    "rca_root_score_top_quantile": 0.80,
    "rca_event_responsibility_pooling": "mean",
    "rca_event_responsibility_head_ratio": 0.30,
    "rca_event_responsibility_head_points": 30,
    "rca_event_responsibility_top_quantile": 0.80,
    "rca_evidence_fusion_weight": 0.0,
    "rca_evidence_fusion_pooling": "mean",
    "rca_evidence_fusion_head_ratio": 0.30,
    "rca_evidence_fusion_head_points": 30,
    "rca_evidence_fusion_top_quantile": 0.80,
    "rca_source_interaction_head_weight": 0.0,
    "rca_source_interaction_head_pooling": "head_mean",
    "rca_source_interaction_head_ratio": 0.30,
    "rca_source_interaction_head_points": 30,
    "rca_source_interaction_head_top_quantile": 0.80,
    "rca_source_consistency_head_weight": 0.0,
    "rca_source_consistency_head_pooling": "head_mean",
    "rca_source_consistency_head_ratio": 0.30,
    "rca_source_consistency_head_points": 30,
    "rca_source_consistency_head_top_quantile": 0.80,
    "rca_source_interaction_weight": 0.0,
    "rca_source_innovation_weight": 0.0,
    "rca_source_innovation_mode": "series",
    "rca_source_innovation_neighbor_weight": 1.0,
    "rca_source_innovation_lead_points": 1,
    "rca_mechanism_guided_source_weight": 0.0,
    "rca_adaptive_mechanism_gate_weight": 0.0,
    "rca_adaptive_mechanism_gate_floor": 0.05,
    "rca_adaptive_mechanism_gate_mode": "source_onset",
    "rca_adaptive_mechanism_selection_weight": 0.0,
    "rca_adaptive_mechanism_selection_floor": 0.05,
    "rca_adaptive_mechanism_selection_mode": "source_onset",
    "rca_conservative_mechanism_weight": 0.0,
    "rca_conservative_mechanism_support_floor": 0.20,
    "rca_conservative_mechanism_candidate_topk": 0,
    "rca_conservative_mechanism_support_mode": "source_onset",
    "rca_source_gated_mechanism_weight": 0.0,
    "rca_source_gated_mechanism_support_floor": 0.20,
    "rca_source_gated_mechanism_candidate_topk": 0,
    "rca_source_gated_mechanism_support_mode": "source_onset",
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
    "rca_event_base_weight": 1.0,
    "rca_export_score_distribution": "raw",
    "rca_export_score_distribution_top_m": 0,
    "rca_export_score_distribution_power": 2.0,
    "rca_export_score_distribution_tau": 1.0,
    "rca_export_responsibility_score": False,
    "rca_export_responsibility_ranking_weight": 0.0,
    "rca_export_responsibility_pre_weight": 0.0,
    "rca_export_responsibility_base_weight": 0.0,
    "rca_export_responsibility_source_gate_weight": 0.0,
    "rca_export_responsibility_onset_weight": 0.0,
    "rca_export_responsibility_mechanism_residual_weight": 0.0,
    "rca_export_responsibility_root_weight": 0.0,
    "rca_export_responsibility_event_weight": 0.0,
    "rca_export_responsibility_evidence_fusion_weight": 0.0,
    "rca_export_responsibility_source_interaction_weight": 0.0,
    "rca_export_responsibility_graph_low_weight": 0.0,
    "rca_export_responsibility_response_suppressor_low_weight": 0.0,
    "rca_responsibility_rerank": False,
    "rca_graph_penalty_weight": 0.0,
    "rca_response_suppressor_weight": 0.0,
    "rca_response_suppressor_source_guard_mode": "source_onset",
    "rca_response_suppressor_source_guard_floor": 0.0,
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
    "rca_align_event_onset": False,
    "rca_align_event_onset_baseline_window": 300,
    "rca_align_event_onset_z": 2.0,
    "rca_align_event_onset_quantile": 0.90,
    "rca_event_specificity_weight": 0.0,
    "rca_event_specificity_top_k": 1,
    "rca_event_specificity_threshold": 0.35,
    "rca_event_specificity_min_events": 20,
    "rca_event_specificity_source_guard_weight": 0.0,
    "rca_event_specificity_source_guard_floor": 0.0,
    "rca_event_specificity_keep_top_k": 0,
    "rca_event_specificity_fill_secondary_top_k": 0,
    "rca_event_specificity_secondary_base_weight": 0.0,
    "rca_event_specificity_secondary_onset_weight": 0.0,
    "rca_event_specificity_secondary_source_gate_weight": 0.0,
    "rca_event_specificity_secondary_mechanism_residual_weight": 0.0,
    "rca_event_specificity_secondary_source_interaction_weight": 0.0,
    "rca_event_specificity_secondary_source_consistency_head_weight": 0.0,
    "rca_event_specificity_secondary_evidence_fusion_weight": 0.0,
    "rca_adaptive_evidence_rerank": False,
    "rca_adaptive_evidence_top_k": 20,
    "rca_adaptive_evidence_keep_top_k": 1,
    "rca_adaptive_evidence_fill_top_k": 5,
    "rca_adaptive_evidence_gate": "source_top_old_rank",
    "rca_adaptive_evidence_source_old_rank_threshold": 1,
    "rca_adaptive_evidence_source_margin_threshold": 0.0,
    "rca_adaptive_evidence_source_base_weight": 0.45,
    "rca_adaptive_evidence_source_onset_weight": 0.25,
    "rca_adaptive_evidence_source_gate_weight": 1.0,
    "rca_adaptive_evidence_source_mechanism_residual_weight": 0.25,
    "rca_adaptive_evidence_source_interaction_weight": 0.25,
    "rca_adaptive_evidence_fallback_original_weight": 1.0,
    "rca_adaptive_evidence_fallback_base_weight": 0.45,
    "rca_adaptive_evidence_fallback_onset_weight": 0.20,
    "rca_adaptive_evidence_fallback_mechanism_residual_weight": 0.0,
    "rca_adaptive_evidence_fallback_root_weight": 0.0,
    "rca_adaptive_evidence_fallback_event_weight": 0.0,
    "rca_adaptive_evidence_fallback_evidence_fusion_weight": 0.0,
    "rca_adaptive_evidence_fallback_source_interaction_weight": 0.0,
    "rca_adaptive_evidence_fallback_source_consistency_weight": 0.0,
    "rca_adaptive_evidence_fallback_graph_low_weight": 0.0,
    "rca_adaptive_evidence_fallback_response_suppressor_low_weight": 0.0,
    "rca_topk_rerank": False,
    "rca_topk_rerank_k": 5,
    "rca_topk_rerank_original_weight": 1.0,
    "rca_topk_rerank_base_weight": 0.0,
    "rca_topk_rerank_group_weight": 0.0,
    "rca_topk_rerank_onset_weight": 0.0,
    "rca_topk_rerank_source_gate_weight": 0.0,
    "rca_topk_rerank_mechanism_residual_weight": 0.0,
    "rca_topk_rerank_source_interaction_weight": 0.0,
    "rca_topk_rerank_source_consistency_head_weight": 0.0,
    "rca_topk_rerank_graph_penalty_weight": 0.0,
    "rca_topk_rerank_component_scope": "candidate",
    "rca_topk_rerank_keep_primary_top_k": 0,
    "rca_topk_rerank_fill_secondary_top_k": 0,
    "rca_topk_rerank_secondary_original_weight": 0.0,
    "rca_topk_rerank_secondary_base_weight": 0.0,
    "rca_topk_rerank_secondary_group_weight": 0.0,
    "rca_topk_rerank_secondary_onset_weight": 0.0,
    "rca_topk_rerank_secondary_source_gate_weight": 0.0,
    "rca_topk_rerank_secondary_mechanism_residual_weight": 0.0,
    "rca_topk_rerank_secondary_source_interaction_weight": 0.0,
    "rca_topk_rerank_secondary_source_consistency_head_weight": 0.0,
    "rca_topk_rerank_secondary_graph_penalty_weight": 0.0,
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
        self.last_epoch = 0
        self.best_monitor_value = np.Inf
        self.monitor_name = "val_loss"

    def __call__(self, val_loss, model, epoch, selection_score=None, selection_note=None):
        self.last_epoch = epoch
        monitor_value = val_loss if selection_score is None else float(selection_score)
        monitor_name = "val_loss" if selection_score is None else "rca_aware_score"
        if selection_note:
            print(f"  [RCA CKPT] {monitor_name}={monitor_value:.8f}, {selection_note}")
        if self.best_score is None:
            self.best_score = -monitor_value
            self.best_val_loss = val_loss
            self.best_monitor_value = monitor_value
            self.monitor_name = monitor_name
            self.save_checkpoint(val_loss, model)
            self.best_epoch = epoch
            print(f"  [ES DBG] init: val_loss={val_loss:.8f}, best={val_loss:.8f}")
            return "initial"

        rel_improvement = (
            self.best_monitor_value - monitor_value
        ) / max(abs(self.best_monitor_value), 1e-10)

        if rel_improvement >= self.relative_delta:
            self.best_score = -monitor_value
            self.best_val_loss = val_loss
            self.best_monitor_value = monitor_value
            self.monitor_name = monitor_name
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
        self._last_root_score_channel_scores = None
        self._last_event_responsibility_channel_scores = None
        self._last_event_route_channel_scores = None
        self._last_response_suppressor_channel_scores = None
        self._last_evidence_fusion_channel_scores = None
        self._last_source_interaction_head_channel_scores = None
        self._last_source_consistency_head_channel_scores = None
        self._last_synthetic_rca_channel_scores = None
        self._last_channel_names = None
        self._channel_corr_prior = None
        self._sps_teacher = None

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
        checkpoint_dir = os.path.join(ROOT_PATH, "result", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            checkpoint_dir, f"LaGraph_checkpoint_{date_part}_{micro_part}.pt"
        )

        checkpoint_state = None
        if self.early_stopping is not None:
            checkpoint_state = getattr(self.early_stopping, "check_point", None)
        if checkpoint_state is not None:
            torch.save(
                {
                    "model_state_dict": checkpoint_state,
                    "config": dict(vars(self.config)),
                    "scaler": self.scaler,
                    "input_clip_lower": self._input_clip_lower,
                    "input_clip_upper": self._input_clip_upper,
                    "dataset_name": self.dataset_name,
                    "best_val_loss": float(self.early_stopping.val_loss_min),
                    "best_epoch": int(self.early_stopping.best_epoch),
                    "last_epoch": int(getattr(self.early_stopping, "last_epoch", 0)),
                    "best_monitor_value": float(
                        getattr(self.early_stopping, "best_monitor_value", np.nan)
                    ),
                    "monitor_name": getattr(self.early_stopping, "monitor_name", "val_loss"),
                },
                checkpoint_path,
            )
        else:
            checkpoint_path = None

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
                "best_monitor_value": float(
                    getattr(self.early_stopping, "best_monitor_value", np.nan)
                ),
                "monitor_name": getattr(self.early_stopping, "monitor_name", "val_loss"),
                "best_epoch": self.early_stopping.best_epoch,
                "selected_checkpoint_epoch": self.early_stopping.best_epoch,
                "total_epochs_run": int(getattr(self.early_stopping, "last_epoch", 0)),
                "max_epochs": int(getattr(self.config, "num_epochs", 0)),
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
                "checkpoint_policy": getattr(self.config, "checkpoint_policy", None),
                "use_rca_aware_checkpoint": getattr(
                    self.config, "use_rca_aware_checkpoint", None
                ),
                "rca_checkpoint_normalize": getattr(
                    self.config, "rca_checkpoint_normalize", None
                ),
                "rca_checkpoint_proxy_weight": getattr(
                    self.config, "rca_checkpoint_proxy_weight", None
                ),
                "rca_checkpoint_proxy_batches": getattr(
                    self.config, "rca_checkpoint_proxy_batches", None
                ),
                "rca_checkpoint_min_epoch": getattr(
                    self.config, "rca_checkpoint_min_epoch", None
                ),
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
                "source_effect_use_channel_prior": getattr(
                    self.config, "source_effect_use_channel_prior", None
                ),
                "source_effect_prior_topk": getattr(self.config, "source_effect_prior_topk", None),
                "source_effect_onset_rank_weight": getattr(
                    self.config, "source_effect_onset_rank_weight", None
                ),
                "source_effect_graph_alignment_weight": getattr(
                    self.config, "source_effect_graph_alignment_weight", None
                ),
                "source_effect_graph_reverse_weight": getattr(
                    self.config, "source_effect_graph_reverse_weight", None
                ),
                "source_effect_consistency_weight": getattr(
                    self.config, "source_effect_consistency_weight", None
                ),
                "source_effect_consistency_rank_weight": getattr(
                    self.config, "source_effect_consistency_rank_weight", None
                ),
                "source_effect_consistency_effect_rank_weight": getattr(
                    self.config,
                    "source_effect_consistency_effect_rank_weight",
                    None,
                ),
                "source_effect_consistency_effect_suppress_weight": getattr(
                    self.config,
                    "source_effect_consistency_effect_suppress_weight",
                    None,
                ),
                "source_effect_consistency_gate_align_weight": getattr(
                    self.config,
                    "source_effect_consistency_gate_align_weight",
                    None,
                ),
                "source_effect_consistency_margin": getattr(
                    self.config, "source_effect_consistency_margin", None
                ),
                "lambda_source_effect_rca_head": getattr(
                    self.config, "lambda_source_effect_rca_head", None
                ),
                "source_effect_rca_head_bce_weight": getattr(
                    self.config, "source_effect_rca_head_bce_weight", None
                ),
                "source_effect_rca_head_rank_weight": getattr(
                    self.config, "source_effect_rca_head_rank_weight", None
                ),
                "use_root_score_head": getattr(self.config, "use_root_score_head", None),
                "root_score_head_mode": getattr(self.config, "root_score_head_mode", None),
                "root_score_detach_features": getattr(
                    self.config, "root_score_detach_features", None
                ),
                "root_response_penalty_init": getattr(
                    self.config, "root_response_penalty_init", None
                ),
                "root_response_confidence_discount": getattr(
                    self.config, "root_response_confidence_discount", None
                ),
                "root_response_use_innovation_split": getattr(
                    self.config, "root_response_use_innovation_split", None
                ),
                "use_pairwise_root_response_head": getattr(
                    self.config, "use_pairwise_root_response_head", None
                ),
                "pairwise_root_response_rank": getattr(
                    self.config, "pairwise_root_response_rank", None
                ),
                "pairwise_root_response_graph_weight": getattr(
                    self.config, "pairwise_root_response_graph_weight", None
                ),
                "pairwise_root_response_reward_init": getattr(
                    self.config, "pairwise_root_response_reward_init", None
                ),
                "pairwise_root_response_penalty_init": getattr(
                    self.config, "pairwise_root_response_penalty_init", None
                ),
                "pairwise_root_response_logit_weight": getattr(
                    self.config, "pairwise_root_response_logit_weight", None
                ),
                "lambda_pairwise_root_response": getattr(
                    self.config, "lambda_pairwise_root_response", None
                ),
                "lambda_source_effect_root_score": getattr(
                    self.config, "lambda_source_effect_root_score", None
                ),
                "root_score_bce_weight": getattr(self.config, "root_score_bce_weight", None),
                "root_score_rank_weight": getattr(self.config, "root_score_rank_weight", None),
                "root_score_effect_suppress_weight": getattr(
                    self.config, "root_score_effect_suppress_weight", None
                ),
                "root_response_source_branch_weight": getattr(
                    self.config, "root_response_source_branch_weight", None
                ),
                "root_response_response_branch_weight": getattr(
                    self.config, "root_response_response_branch_weight", None
                ),
                "root_response_response_suppress_weight": getattr(
                    self.config, "root_response_response_suppress_weight", None
                ),
                "use_event_responsibility_head": getattr(
                    self.config, "use_event_responsibility_head", None
                ),
                "lambda_event_responsibility": getattr(
                    self.config, "lambda_event_responsibility", None
                ),
                "event_responsibility_loss_mode": getattr(
                    self.config, "event_responsibility_loss_mode", None
                ),
                "event_responsibility_bce_balance": getattr(
                    self.config, "event_responsibility_bce_balance", None
                ),
                "event_responsibility_ce_weight": getattr(
                    self.config, "event_responsibility_ce_weight", None
                ),
                "event_responsibility_rank_weight": getattr(
                    self.config, "event_responsibility_rank_weight", None
                ),
                "event_responsibility_effect_suppress_weight": getattr(
                    self.config, "event_responsibility_effect_suppress_weight", None
                ),
                "use_response_suppressor_head": getattr(
                    self.config, "use_response_suppressor_head", None
                ),
                "lambda_response_suppressor": getattr(
                    self.config, "lambda_response_suppressor", None
                ),
                "response_suppressor_detach_inputs": getattr(
                    self.config, "response_suppressor_detach_inputs", None
                ),
                "use_evidence_fusion_head": getattr(
                    self.config, "use_evidence_fusion_head", None
                ),
                "evidence_fusion_use_graph_response": getattr(
                    self.config, "evidence_fusion_use_graph_response", None
                ),
                "lambda_evidence_fusion": getattr(
                    self.config, "lambda_evidence_fusion", None
                ),
                "evidence_fusion_loss_mode": getattr(
                    self.config, "evidence_fusion_loss_mode", None
                ),
                "evidence_fusion_export_mode": getattr(
                    self.config, "evidence_fusion_export_mode", None
                ),
                "evidence_fusion_bce_weight": getattr(
                    self.config, "evidence_fusion_bce_weight", None
                ),
                "evidence_fusion_rank_weight": getattr(
                    self.config, "evidence_fusion_rank_weight", None
                ),
                "evidence_fusion_effect_suppress_weight": getattr(
                    self.config, "evidence_fusion_effect_suppress_weight", None
                ),
                "rca_evidence_fusion_weight": getattr(
                    self.config, "rca_evidence_fusion_weight", None
                ),
                "rca_evidence_fusion_pooling": getattr(
                    self.config, "rca_evidence_fusion_pooling", None
                ),
                "rca_evidence_fusion_head_ratio": getattr(
                    self.config, "rca_evidence_fusion_head_ratio", None
                ),
                "rca_evidence_fusion_head_points": getattr(
                    self.config, "rca_evidence_fusion_head_points", None
                ),
                "rca_evidence_fusion_top_quantile": getattr(
                    self.config, "rca_evidence_fusion_top_quantile", None
                ),
                "use_source_interaction_head": getattr(
                    self.config, "use_source_interaction_head", None
                ),
                "source_interaction_detach_inputs": getattr(
                    self.config, "source_interaction_detach_inputs", None
                ),
                "lambda_source_interaction_head": getattr(
                    self.config, "lambda_source_interaction_head", None
                ),
                "source_interaction_head_bce_weight": getattr(
                    self.config, "source_interaction_head_bce_weight", None
                ),
                "source_interaction_head_rank_weight": getattr(
                    self.config, "source_interaction_head_rank_weight", None
                ),
                "source_interaction_head_effect_suppress_weight": getattr(
                    self.config,
                    "source_interaction_head_effect_suppress_weight",
                    None,
                ),
                "source_interaction_head_pairwise_effect_weight": getattr(
                    self.config,
                    "source_interaction_head_pairwise_effect_weight",
                    None,
                ),
                "source_interaction_head_pairwise_effect_margin": getattr(
                    self.config,
                    "source_interaction_head_pairwise_effect_margin",
                    None,
                ),
                "rca_source_interaction_head_weight": getattr(
                    self.config, "rca_source_interaction_head_weight", None
                ),
                "rca_source_interaction_head_pooling": getattr(
                    self.config, "rca_source_interaction_head_pooling", None
                ),
                "rca_source_interaction_head_ratio": getattr(
                    self.config, "rca_source_interaction_head_ratio", None
                ),
                "rca_source_interaction_head_points": getattr(
                    self.config, "rca_source_interaction_head_points", None
                ),
                "rca_source_interaction_head_top_quantile": getattr(
                    self.config, "rca_source_interaction_head_top_quantile", None
                ),
                "use_source_consistency_head": getattr(
                    self.config, "use_source_consistency_head", None
                ),
                "source_consistency_detach_inputs": getattr(
                    self.config, "source_consistency_detach_inputs", None
                ),
                "lambda_source_consistency_head": getattr(
                    self.config, "lambda_source_consistency_head", None
                ),
                "source_consistency_head_bce_weight": getattr(
                    self.config, "source_consistency_head_bce_weight", None
                ),
                "source_consistency_head_rank_weight": getattr(
                    self.config, "source_consistency_head_rank_weight", None
                ),
                "source_consistency_head_effect_suppress_weight": getattr(
                    self.config,
                    "source_consistency_head_effect_suppress_weight",
                    None,
                ),
                "source_consistency_head_pairwise_effect_weight": getattr(
                    self.config,
                    "source_consistency_head_pairwise_effect_weight",
                    None,
                ),
                "source_consistency_head_pairwise_effect_margin": getattr(
                    self.config,
                    "source_consistency_head_pairwise_effect_margin",
                    None,
                ),
                "rca_source_consistency_head_weight": getattr(
                    self.config, "rca_source_consistency_head_weight", None
                ),
                "rca_source_consistency_head_pooling": getattr(
                    self.config, "rca_source_consistency_head_pooling", None
                ),
                "rca_source_consistency_head_ratio": getattr(
                    self.config, "rca_source_consistency_head_ratio", None
                ),
                "rca_source_consistency_head_points": getattr(
                    self.config, "rca_source_consistency_head_points", None
                ),
                "rca_source_consistency_head_top_quantile": getattr(
                    self.config, "rca_source_consistency_head_top_quantile", None
                ),
                "rca_event_responsibility_weight": getattr(
                    self.config, "rca_event_responsibility_weight", None
                ),
                "rca_event_responsibility_pooling": getattr(
                    self.config, "rca_event_responsibility_pooling", None
                ),
                "rca_event_responsibility_head_ratio": getattr(
                    self.config, "rca_event_responsibility_head_ratio", None
                ),
                "rca_event_responsibility_head_points": getattr(
                    self.config, "rca_event_responsibility_head_points", None
                ),
                "rca_event_responsibility_top_quantile": getattr(
                    self.config, "rca_event_responsibility_top_quantile", None
                ),
                "rca_root_score_pooling": getattr(self.config, "rca_root_score_pooling", None),
                "rca_root_score_head_ratio": getattr(self.config, "rca_root_score_head_ratio", None),
                "rca_root_score_head_points": getattr(self.config, "rca_root_score_head_points", None),
                "rca_root_score_top_quantile": getattr(
                    self.config, "rca_root_score_top_quantile", None
                ),
                "lambda_source_bottleneck": getattr(self.config, "lambda_source_bottleneck", None),
                "source_bottleneck_bce_weight": getattr(
                    self.config, "source_bottleneck_bce_weight", None
                ),
                "source_bottleneck_rank_weight": getattr(
                    self.config, "source_bottleneck_rank_weight", None
                ),
                "source_bottleneck_effect_suppress_weight": getattr(
                    self.config, "source_bottleneck_effect_suppress_weight", None
                ),
                "use_channel_masked_modeling": getattr(self.config, "use_channel_masked_modeling", None),
                "lambda_channel_masked": getattr(self.config, "lambda_channel_masked", None),
                "channel_mask_interval": getattr(self.config, "channel_mask_interval", None),
                "channel_mask_ratio": getattr(self.config, "channel_mask_ratio", None),
                "lambda_masked_rca_head": getattr(self.config, "lambda_masked_rca_head", None),
                "masked_rca_bce_weight": getattr(self.config, "masked_rca_bce_weight", None),
                "masked_rca_rank_weight": getattr(self.config, "masked_rca_rank_weight", None),
                "use_mechanism_residual_feedback": getattr(
                    self.config, "use_mechanism_residual_feedback", None
                ),
                "mechanism_feedback_init": getattr(self.config, "mechanism_feedback_init", None),
                "mechanism_feedback_detach": getattr(
                    self.config, "mechanism_feedback_detach", None
                ),
                "mechanism_feedback_norm": getattr(self.config, "mechanism_feedback_norm", None),
                "mechanism_feedback_clip": getattr(self.config, "mechanism_feedback_clip", None),
                "use_interventional_channel_masking": getattr(
                    self.config, "use_interventional_channel_masking", None
                ),
                "lambda_interventional_source_bce": getattr(
                    self.config, "lambda_interventional_source_bce", None
                ),
                "lambda_interventional_source_rank": getattr(
                    self.config, "lambda_interventional_source_rank", None
                ),
                "lambda_interventional_graph_support": getattr(
                    self.config, "lambda_interventional_graph_support", None
                ),
                "interventional_rank_signal": getattr(self.config, "interventional_rank_signal", None),
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
                "rca_source_score_weight": getattr(self.config, "rca_source_score_weight", None),
                "rca_root_score_weight": getattr(self.config, "rca_root_score_weight", None),
                "rca_root_score_signal": getattr(self.config, "rca_root_score_signal", None),
                "rca_source_consensus_weight": getattr(self.config, "rca_source_consensus_weight", None),
                "rca_onset_consensus_weight": getattr(self.config, "rca_onset_consensus_weight", None),
                "rca_source_consensus_mode": getattr(self.config, "rca_source_consensus_mode", None),
                "rca_propagation_weight": getattr(self.config, "rca_propagation_weight", None),
                "rca_source_mechanism_weight": getattr(self.config, "rca_source_mechanism_weight", None),
                "rca_causal_weight": getattr(self.config, "rca_causal_weight", None),
                "rca_mechanism_guided_source_weight": getattr(self.config, "rca_mechanism_guided_source_weight", None),
                "rca_source_innovation_weight": getattr(self.config, "rca_source_innovation_weight", None),
                "rca_source_innovation_mode": getattr(self.config, "rca_source_innovation_mode", None),
                "rca_source_innovation_neighbor_weight": getattr(
                    self.config, "rca_source_innovation_neighbor_weight", None
                ),
                "rca_source_innovation_lead_points": getattr(
                    self.config, "rca_source_innovation_lead_points", None
                ),
                "rca_onset_weight": getattr(self.config, "rca_onset_weight", None),
                "rca_onset_baseline_window": getattr(self.config, "rca_onset_baseline_window", None),
                "rca_onset_z": getattr(self.config, "rca_onset_z", None),
                "rca_event_specificity_keep_top_k": getattr(
                    self.config, "rca_event_specificity_keep_top_k", None
                ),
                "rca_event_specificity_fill_secondary_top_k": getattr(
                    self.config, "rca_event_specificity_fill_secondary_top_k", None
                ),
                "rca_event_specificity_secondary_base_weight": getattr(
                    self.config, "rca_event_specificity_secondary_base_weight", None
                ),
                "rca_event_specificity_secondary_onset_weight": getattr(
                    self.config, "rca_event_specificity_secondary_onset_weight", None
                ),
                "rca_event_specificity_secondary_source_gate_weight": getattr(
                    self.config, "rca_event_specificity_secondary_source_gate_weight", None
                ),
                "rca_event_specificity_secondary_mechanism_residual_weight": getattr(
                    self.config,
                    "rca_event_specificity_secondary_mechanism_residual_weight",
                    None,
                ),
                "rca_event_specificity_secondary_source_interaction_weight": getattr(
                    self.config,
                    "rca_event_specificity_secondary_source_interaction_weight",
                    None,
                ),
                "rca_event_specificity_secondary_source_consistency_head_weight": getattr(
                    self.config,
                    "rca_event_specificity_secondary_source_consistency_head_weight",
                    None,
                ),
                "rca_event_specificity_secondary_evidence_fusion_weight": getattr(
                    self.config,
                    "rca_event_specificity_secondary_evidence_fusion_weight",
                    None,
                ),
                "rca_adaptive_evidence_rerank": getattr(
                    self.config, "rca_adaptive_evidence_rerank", None
                ),
                "rca_adaptive_evidence_top_k": getattr(
                    self.config, "rca_adaptive_evidence_top_k", None
                ),
                "rca_adaptive_evidence_keep_top_k": getattr(
                    self.config, "rca_adaptive_evidence_keep_top_k", None
                ),
                "rca_adaptive_evidence_fill_top_k": getattr(
                    self.config, "rca_adaptive_evidence_fill_top_k", None
                ),
                "rca_adaptive_evidence_gate": getattr(
                    self.config, "rca_adaptive_evidence_gate", None
                ),
                "rca_adaptive_evidence_source_old_rank_threshold": getattr(
                    self.config,
                    "rca_adaptive_evidence_source_old_rank_threshold",
                    None,
                ),
                "rca_adaptive_evidence_source_margin_threshold": getattr(
                    self.config,
                    "rca_adaptive_evidence_source_margin_threshold",
                    None,
                ),
                "rca_adaptive_evidence_source_base_weight": getattr(
                    self.config, "rca_adaptive_evidence_source_base_weight", None
                ),
                "rca_adaptive_evidence_source_onset_weight": getattr(
                    self.config, "rca_adaptive_evidence_source_onset_weight", None
                ),
                "rca_adaptive_evidence_source_gate_weight": getattr(
                    self.config, "rca_adaptive_evidence_source_gate_weight", None
                ),
                "rca_adaptive_evidence_source_mechanism_residual_weight": getattr(
                    self.config,
                    "rca_adaptive_evidence_source_mechanism_residual_weight",
                    None,
                ),
                "rca_adaptive_evidence_source_interaction_weight": getattr(
                    self.config,
                    "rca_adaptive_evidence_source_interaction_weight",
                    None,
                ),
                "rca_adaptive_evidence_fallback_original_weight": getattr(
                    self.config,
                    "rca_adaptive_evidence_fallback_original_weight",
                    None,
                ),
                "rca_adaptive_evidence_fallback_base_weight": getattr(
                    self.config, "rca_adaptive_evidence_fallback_base_weight", None
                ),
                "rca_adaptive_evidence_fallback_onset_weight": getattr(
                    self.config, "rca_adaptive_evidence_fallback_onset_weight", None
                ),
                "rca_adaptive_evidence_fallback_mechanism_residual_weight": getattr(
                    self.config,
                    "rca_adaptive_evidence_fallback_mechanism_residual_weight",
                    None,
                ),
                "rca_adaptive_evidence_fallback_root_weight": getattr(
                    self.config, "rca_adaptive_evidence_fallback_root_weight", None
                ),
                "rca_adaptive_evidence_fallback_event_weight": getattr(
                    self.config, "rca_adaptive_evidence_fallback_event_weight", None
                ),
                "rca_adaptive_evidence_fallback_evidence_fusion_weight": getattr(
                    self.config,
                    "rca_adaptive_evidence_fallback_evidence_fusion_weight",
                    None,
                ),
                "rca_adaptive_evidence_fallback_source_interaction_weight": getattr(
                    self.config,
                    "rca_adaptive_evidence_fallback_source_interaction_weight",
                    None,
                ),
                "rca_adaptive_evidence_fallback_source_consistency_weight": getattr(
                    self.config,
                    "rca_adaptive_evidence_fallback_source_consistency_weight",
                    None,
                ),
                "rca_adaptive_evidence_fallback_graph_low_weight": getattr(
                    self.config,
                    "rca_adaptive_evidence_fallback_graph_low_weight",
                    None,
                ),
                "rca_adaptive_evidence_fallback_response_suppressor_low_weight": getattr(
                    self.config,
                    "rca_adaptive_evidence_fallback_response_suppressor_low_weight",
                    None,
                ),
                "rca_topk_rerank": getattr(self.config, "rca_topk_rerank", None),
                "rca_topk_rerank_k": getattr(self.config, "rca_topk_rerank_k", None),
                "rca_topk_rerank_original_weight": getattr(
                    self.config, "rca_topk_rerank_original_weight", None
                ),
                "rca_topk_rerank_base_weight": getattr(
                    self.config, "rca_topk_rerank_base_weight", None
                ),
                "rca_topk_rerank_group_weight": getattr(
                    self.config, "rca_topk_rerank_group_weight", None
                ),
                "rca_topk_rerank_onset_weight": getattr(
                    self.config, "rca_topk_rerank_onset_weight", None
                ),
                "rca_topk_rerank_source_gate_weight": getattr(
                    self.config, "rca_topk_rerank_source_gate_weight", None
                ),
                "rca_topk_rerank_mechanism_residual_weight": getattr(
                    self.config, "rca_topk_rerank_mechanism_residual_weight", None
                ),
                "rca_topk_rerank_source_interaction_weight": getattr(
                    self.config, "rca_topk_rerank_source_interaction_weight", None
                ),
                "rca_topk_rerank_source_consistency_head_weight": getattr(
                    self.config,
                    "rca_topk_rerank_source_consistency_head_weight",
                    None,
                ),
                "rca_topk_rerank_graph_penalty_weight": getattr(
                    self.config, "rca_topk_rerank_graph_penalty_weight", None
                ),
                "rca_topk_rerank_component_scope": getattr(
                    self.config, "rca_topk_rerank_component_scope", None
                ),
                "rca_topk_rerank_keep_primary_top_k": getattr(
                    self.config, "rca_topk_rerank_keep_primary_top_k", None
                ),
                "rca_topk_rerank_fill_secondary_top_k": getattr(
                    self.config, "rca_topk_rerank_fill_secondary_top_k", None
                ),
                "rca_topk_rerank_secondary_original_weight": getattr(
                    self.config, "rca_topk_rerank_secondary_original_weight", None
                ),
                "rca_topk_rerank_secondary_base_weight": getattr(
                    self.config, "rca_topk_rerank_secondary_base_weight", None
                ),
                "rca_topk_rerank_secondary_group_weight": getattr(
                    self.config, "rca_topk_rerank_secondary_group_weight", None
                ),
                "rca_topk_rerank_secondary_onset_weight": getattr(
                    self.config, "rca_topk_rerank_secondary_onset_weight", None
                ),
                "rca_topk_rerank_secondary_source_gate_weight": getattr(
                    self.config, "rca_topk_rerank_secondary_source_gate_weight", None
                ),
                "rca_topk_rerank_secondary_mechanism_residual_weight": getattr(
                    self.config, "rca_topk_rerank_secondary_mechanism_residual_weight", None
                ),
                "rca_topk_rerank_secondary_source_interaction_weight": getattr(
                    self.config, "rca_topk_rerank_secondary_source_interaction_weight", None
                ),
                "rca_topk_rerank_secondary_source_consistency_head_weight": getattr(
                    self.config,
                    "rca_topk_rerank_secondary_source_consistency_head_weight",
                    None,
                ),
                "rca_topk_rerank_secondary_graph_penalty_weight": getattr(
                    self.config, "rca_topk_rerank_secondary_graph_penalty_weight", None
                ),
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
                "channel_graph_evidence_only": getattr(
                    self.config, "channel_graph_evidence_only", None
                ),
                "channel_graph_type": getattr(self.config, "channel_graph_type", None),
                "directed_graph_parent_topk": getattr(
                    self.config, "directed_graph_parent_topk", None
                ),
                "directed_graph_embedding_dim": getattr(
                    self.config, "directed_graph_embedding_dim", None
                ),
                "directed_graph_use_signed_transfer": getattr(
                    self.config, "directed_graph_use_signed_transfer", None
                ),
                "directed_graph_transfer_mode": getattr(
                    self.config, "directed_graph_transfer_mode", None
                ),
                "directed_graph_transfer_rank": getattr(
                    self.config, "directed_graph_transfer_rank", None
                ),
                "directed_graph_transfer_scale": getattr(
                    self.config, "directed_graph_transfer_scale", None
                ),
                "use_channel_graph_reliability": getattr(
                    self.config, "use_channel_graph_reliability", None
                ),
                "channel_graph_role": getattr(self.config, "channel_graph_role", None),
                "use_multilag_graph_propagation": getattr(
                    self.config, "use_multilag_graph_propagation", None
                ),
                "graph_propagation_lags": getattr(
                    self.config, "graph_propagation_lags", None
                ),
                "graph_propagation_lag_mode": getattr(
                    self.config, "graph_propagation_lag_mode", None
                ),
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
            "artifacts": {
                "checkpoint_path": checkpoint_path,
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

    def _compute_rca_checkpoint_proxy(self, vali_loader):
        if not bool(getattr(self.config, "use_rca_aware_checkpoint", False)):
            return None
        max_batches = int(getattr(self.config, "rca_checkpoint_proxy_batches", 2) or 0)
        if max_batches <= 0:
            return None

        was_training = bool(self.model.training)
        py_random_state = random.getstate()
        np_random_state = np.random.get_state()
        torch_random_state = torch.random.get_rng_state()
        cuda_random_states = (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        )

        self.model.eval()
        reciprocal_ranks = []
        hit1 = []
        margins = []
        try:
            with torch.inference_mode():
                for batch_idx, (input_data, _) in enumerate(vali_loader):
                    if batch_idx >= max_batches:
                        break
                    input_data = input_data.float().to(self.device, non_blocking=True)
                    (
                        synth_data,
                        event_mask,
                        source_mask,
                        _effect_mask,
                        source_onset_mask,
                        _effect_time_mask,
                        _propagated_event_mask,
                    ) = self._make_source_effect_synthetic_batch(input_data)
                    synth_rec, _, _, _, _, synth_aux, _ = self.model(synth_data)
                    source_score = None
                    if synth_aux:
                        source_score = synth_aux.get("source_gate_score")
                        if source_score is None:
                            source_score = synth_aux.get("channel_mechanism_error")
                    if source_score is None:
                        source_score = F.l1_loss(synth_rec, synth_data, reduction="none")

                    if source_score.dim() == 3:
                        focus_mask = source_onset_mask
                        if focus_mask.sum() <= 0:
                            focus_mask = event_mask
                        denom = focus_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
                        channel_scores = (
                            source_score * focus_mask.unsqueeze(-1).to(dtype=source_score.dtype)
                        ).sum(dim=1) / denom
                    elif source_score.dim() == 2:
                        channel_scores = source_score
                    else:
                        continue

                    B, C = channel_scores.shape
                    ranks_template = torch.arange(C, device=channel_scores.device)
                    for b in range(B):
                        roots = source_mask[b] > 0.5
                        non_roots = ~roots
                        if roots.sum() == 0 or non_roots.sum() == 0:
                            continue
                        order = torch.argsort(channel_scores[b], descending=True)
                        rank_pos = torch.empty_like(order)
                        rank_pos[order] = ranks_template
                        best_rank = rank_pos[roots].min().float() + 1.0
                        reciprocal_ranks.append(float((1.0 / best_rank).detach().cpu().item()))
                        hit1.append(float(best_rank.detach().cpu().item() <= 1.0))
                        pos_score = channel_scores[b, roots].mean()
                        neg_score = channel_scores[b, non_roots].max()
                        margins.append(float((pos_score - neg_score).detach().cpu().item()))
        finally:
            random.setstate(py_random_state)
            np.random.set_state(np_random_state)
            torch.random.set_rng_state(torch_random_state)
            if cuda_random_states is not None:
                torch.cuda.set_rng_state_all(cuda_random_states)
            if was_training:
                self.model.train()

        if not reciprocal_ranks:
            return None
        return {
            "source_mrr": float(np.mean(reciprocal_ranks)),
            "source_hit1": float(np.mean(hit1)),
            "source_margin": float(np.mean(margins)) if margins else 0.0,
            "num_synthetic_events": int(len(reciprocal_ranks)),
        }


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
    def _build_channel_corr_prior(
        train_df: pd.DataFrame,
        topk: int = 5,
        selection_axis: str = "source",
    ) -> np.ndarray:
        values = np.asarray(train_df.values, dtype=np.float32)
        corr = np.corrcoef(values, rowvar=False)
        corr = np.nan_to_num(np.abs(corr), nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(corr, 0.0)
        n_channels = corr.shape[0]
        topk = int(topk or 0)
        selection_axis = str(selection_axis or "source").lower()
        if selection_axis not in {"source", "target"}:
            raise ValueError(
                f"Unsupported channel correlation selection_axis={selection_axis!r}"
            )
        if topk > 0 and topk < n_channels:
            keep = np.zeros_like(corr, dtype=bool)
            if selection_axis == "target":
                idx = np.argpartition(-corr, kth=topk - 1, axis=0)[:topk, :]
                cols = np.arange(n_channels)[None, :]
                keep[idx, cols] = True
            else:
                idx = np.argpartition(-corr, kth=topk - 1, axis=1)[:, :topk]
                rows = np.arange(n_channels)[:, None]
                keep[rows, idx] = True
            corr = np.where(keep, corr, 0.0)
        norm_axis = 0 if selection_axis == "target" else 1
        sums = corr.sum(axis=norm_axis, keepdims=True)
        corr = np.divide(corr, sums, out=np.zeros_like(corr), where=sums > 1e-8)
        if selection_axis == "source":
            empty = sums.squeeze(-1) <= 1e-8
            if np.any(empty):
                corr[empty, :] = 1.0 / max(1, n_channels - 1)
                empty_rows = np.where(empty)[0]
                corr[empty_rows, empty_rows] = 0.0
                corr[empty, :] = corr[empty, :] / corr[empty, :].sum(
                    axis=1, keepdims=True
                ).clip(min=1e-8)
        return corr.astype(np.float32)

    def _fit_sps_teacher(self, train_df: pd.DataFrame):
        """Fit a frozen sparse lagged teacher on normal training data only."""
        self._sps_teacher = None
        if not bool(getattr(self.config, "use_sps_teacher", False)):
            return
        if train_df is None or len(train_df) < 4:
            return

        values = np.asarray(train_df.values, dtype=np.float64)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        n_steps, n_channels = values.shape
        lags = tuple(
            sorted(
                {
                    int(lag)
                    for lag in getattr(self.config, "sps_teacher_lags", [1, 3, 6, 12])
                    if int(lag) > 0
                }
            )
        ) or (1,)
        max_lag = max(lags)
        if n_steps <= max_lag + 4:
            print("  [SPS] skipped frozen propagation teacher: insufficient normal samples")
            return

        max_samples = max(100, int(getattr(self.config, "sps_teacher_max_samples", 30000)))
        indices = np.arange(max_lag, n_steps)
        if indices.size > max_samples:
            stride = int(np.ceil(indices.size / max_samples))
            indices = indices[::stride]

        design_dim = n_channels * len(lags)
        xtx = np.zeros((design_dim, design_dim), dtype=np.float64)
        xty = np.zeros((design_dim, n_channels), dtype=np.float64)
        chunk_size = 2048
        for start in range(0, indices.size, chunk_size):
            idx = indices[start:start + chunk_size]
            design = np.concatenate([values[idx - lag] for lag in lags], axis=1)
            target = values[idx]
            xtx += design.T @ design
            xty += design.T @ target

        ridge = max(float(getattr(self.config, "sps_teacher_ridge", 0.10)), 1e-8)
        try:
            coef = np.linalg.solve(xtx + ridge * np.eye(design_dim), xty)
        except np.linalg.LinAlgError:
            coef = np.linalg.pinv(xtx + ridge * np.eye(design_dim)) @ xty
        weights = coef.reshape(len(lags), n_channels, n_channels)
        for lag_idx in range(len(lags)):
            np.fill_diagonal(weights[lag_idx], 0.0)

        topk = min(
            max(1, int(getattr(self.config, "sps_teacher_topk", 3))),
            max(1, n_channels - 1),
        )
        aggregate = np.abs(weights).sum(axis=0)
        keep = np.zeros_like(aggregate, dtype=bool)
        top_idx = np.argpartition(-aggregate, kth=topk - 1, axis=0)[:topk, :]
        keep[top_idx, np.arange(n_channels)[None, :]] = True
        weights *= keep[None, :, :]

        max_gain = max(float(getattr(self.config, "sps_teacher_max_gain", 0.65)), 1e-6)
        outgoing_gain = np.abs(weights).sum(axis=(0, 2))
        scale = np.minimum(1.0, max_gain / np.maximum(outgoing_gain, 1e-8))
        weights *= scale[None, :, None]

        edge_sets = []
        for lag_idx, lag in enumerate(lags):
            source_idx, target_idx = np.nonzero(np.abs(weights[lag_idx]) > 1e-10)
            edge_sets.append(
                {
                    "lag": int(lag),
                    "source": source_idx.astype(np.int64),
                    "target": target_idx.astype(np.int64),
                    "weight": weights[lag_idx, source_idx, target_idx].astype(np.float32),
                }
            )
        edge_count = sum(item["weight"].size for item in edge_sets)
        self._sps_teacher = {"lags": lags, "edges": edge_sets}
        print(
            f"  [SPS] frozen sparse lag teacher fitted "
            f"(samples={indices.size}, lags={list(lags)}, edges={edge_count})"
        )


    def _make_teacher_source_effect_synthetic_batch(self, input_data: torch.Tensor):
        """Generate labelled source-response events from the frozen SPS teacher."""
        teacher = self._sps_teacher
        if teacher is None:
            return self._make_source_effect_synthetic_batch(input_data)

        B, L, C = input_data.shape
        device = input_data.device
        dtype = input_data.dtype
        source_delta = torch.zeros_like(input_data)
        source_channel_mask = torch.zeros(B, C, device=device, dtype=dtype)
        source_onset_mask = torch.zeros(B, L, device=device, dtype=dtype)

        min_len = min(max(2, int(getattr(self.config, "source_effect_min_len", 8))), L)
        max_len = min(
            max(min_len, int(getattr(self.config, "source_effect_max_len", 30))),
            L,
        )
        min_roots = min(max(1, int(getattr(self.config, "source_effect_min_roots", 1))), C)
        max_roots = min(
            max(min_roots, int(getattr(self.config, "source_effect_max_roots", 2))), C
        )

        for batch_idx in range(B):
            n_roots = int(
                torch.randint(min_roots, max_roots + 1, (1,), device=device).item()
            )
            roots = torch.randperm(C, device=device)[:n_roots]
            seg_len = int(
                torch.randint(min_len, max_len + 1, (1,), device=device).item()
            )
            start = int(torch.randint(0, max(1, L - seg_len + 1), (1,), device=device).item())
            end = min(L, start + seg_len)
            scale = input_data[batch_idx, :, roots].std(dim=0).clamp_min(0.2)
            sign = torch.where(
                torch.rand(n_roots, device=device) < 0.5,
                -torch.ones(n_roots, device=device, dtype=dtype),
                torch.ones(n_roots, device=device, dtype=dtype),
            )
            amplitude = sign * scale * torch.empty(
                n_roots, device=device, dtype=dtype
            ).uniform_(1.5, 3.5)
            shape = int(torch.randint(0, 3, (1,), device=device).item())
            if shape == 0:
                source_delta[batch_idx, start:end, roots] = amplitude
            elif shape == 1:
                ramp = torch.linspace(0.5, 1.0, end - start, device=device, dtype=dtype)
                source_delta[batch_idx, start:end, roots] = ramp.unsqueeze(-1) * amplitude
            else:
                pulse_count = max(1, min(end - start, seg_len // 4))
                pulse_idx = torch.randperm(end - start, device=device)[:pulse_count] + start
                source_delta[batch_idx, pulse_idx[:, None], roots] = amplitude
            source_channel_mask[batch_idx, roots] = 1.0
            onset_len = max(1, int(getattr(self.config, "sps_teacher_onset_len", 2)))
            onset_end = min(end, start + onset_len)
            source_onset_mask[batch_idx, start:onset_end] = 1.0

        total_delta = source_delta.clone()
        for time_idx in range(L):
            propagated_at_t = torch.zeros(B, C, device=device, dtype=dtype)
            for edge_set in teacher["edges"]:
                lag = edge_set["lag"]
                if time_idx < lag or edge_set["weight"].size == 0:
                    continue
                source_idx = torch.as_tensor(edge_set["source"], device=device, dtype=torch.long)
                target_idx = torch.as_tensor(edge_set["target"], device=device, dtype=torch.long)
                weight = torch.as_tensor(edge_set["weight"], device=device, dtype=dtype)
                messages = total_delta[:, time_idx - lag, :].index_select(1, source_idx)
                messages = messages * weight.unsqueeze(0)
                propagated_at_t.scatter_add_(
                    1,
                    target_idx.unsqueeze(0).expand(B, -1),
                    messages,
                )
            total_delta[:, time_idx, :] = total_delta[:, time_idx, :] + propagated_at_t

        clip = max(float(getattr(self.config, "sps_teacher_clip", 8.0)), 1.0)
        total_delta = total_delta.clamp(-clip, clip)
        propagated_delta = total_delta - source_delta
        root_amplitude = source_delta.abs().amax(dim=(1, 2)).clamp_min(0.2)
        response_ratio = max(
            float(getattr(self.config, "sps_teacher_response_ratio", 0.15)), 1e-4
        )
        response_threshold = root_amplitude * response_ratio
        propagated_peak = propagated_delta.abs().amax(dim=1)
        effect_channel_mask = (
            propagated_peak > response_threshold.unsqueeze(-1)
        ).to(dtype=dtype) * (1.0 - source_channel_mask)
        effect_energy = (
            propagated_delta.abs() * effect_channel_mask.unsqueeze(1)
        ).sum(dim=-1)
        effect_time_mask = (
            effect_energy > response_threshold.unsqueeze(-1)
        ).to(dtype=dtype)
        event_mask = torch.maximum(
            (source_delta.abs().sum(dim=-1) > 0).to(dtype=dtype),
            effect_time_mask,
        )
        propagated_event_mask = (effect_channel_mask.sum(dim=-1) > 0).to(dtype=dtype)

        return (
            input_data.detach() + total_delta,
            event_mask,
            source_channel_mask,
            effect_channel_mask,
            source_onset_mask,
            effect_time_mask,
            propagated_event_mask,
        )


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
        lambda_strict_cross = getattr(self.config, "lambda_strict_cross_mechanism", 0.0)
        lambda_strict_cross_sparse = getattr(self.config, "lambda_strict_cross_sparse", 0.0)
        if lambda_strict_cross > 0 and 'strict_cross_mechanism_loss' in aux_losses:
            loss = loss + lambda_strict_cross * aux_losses['strict_cross_mechanism_loss']
        if lambda_strict_cross_sparse > 0 and 'strict_cross_sparse_loss' in aux_losses:
            loss = loss + lambda_strict_cross_sparse * aux_losses['strict_cross_sparse_loss']
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
        source_onset_mask = torch.zeros(B, L, device=device, dtype=torch.float32)
        effect_time_mask = torch.zeros(B, L, device=device, dtype=torch.float32)
        propagated_event_mask = torch.zeros(B, device=device, dtype=torch.float32)

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
        propagated_event_prob = min(
            max(
                float(
                    getattr(self.config, "dual_expert_propagated_event_prob", 1.0)
                    or 0.0
                ),
                0.0,
            ),
            1.0,
        )
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
            source_onset_end = end
            use_propagated_event = bool(
                torch.rand((), device=device).item() < propagated_event_prob
            )
            if use_propagated_event and prior is not None and neighbor_topk > 0:
                scores = prior[roots_cpu].max(axis=0)
                scores[roots_cpu] = 0.0
                top_idx = np.argsort(-scores)[:neighbor_topk]
                neighbors = [int(idx) for idx in top_idx if scores[idx] > 0]
            if use_propagated_event and not neighbors and neighbor_topk > 0:
                candidates = [idx for idx in range(C) if idx not in set(roots_cpu)]
                if candidates:
                    perm = torch.randperm(len(candidates), device=device)[:neighbor_topk].detach().cpu().tolist()
                    neighbors = [candidates[int(idx)] for idx in perm]

            if neighbors:
                effects = torch.as_tensor(neighbors, device=device, dtype=torch.long)
                max_delay = min(delay_max, max(0, end - start - 1))
                delay = int(torch.randint(1, max_delay + 2, (1,), device=device).item()) if max_delay > 0 else 0
                eff_start = min(end, start + delay)
                source_onset_end = eff_start if eff_start > start else min(end, start + max(1, (end - start) // 3))
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
                    effect_time_mask[b, eff_start:end] = 1.0
                    propagated_event_mask[b] = 1.0

            event_mask[b, start:end] = 1.0
            source_onset_mask[b, start:max(start + 1, source_onset_end)] = 1.0
            source_channel_mask[b, roots] = 1.0
            effect_channel_mask[b, roots] = 0.0

        return (
            x,
            event_mask,
            source_channel_mask,
            effect_channel_mask,
            source_onset_mask,
            effect_time_mask,
            propagated_event_mask,
        )

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

    def _source_effect_onset_ranking_loss(self, onset_scores, effect_scores, source_mask, effect_mask):
        margin = float(
            getattr(
                self.config,
                "source_effect_onset_margin",
                getattr(self.config, "source_effect_margin", 0.2),
            )
            or 0.2
        )
        losses = []
        for b in range(onset_scores.shape[0]):
            roots = source_mask[b] > 0.5
            non_roots = ~roots
            if roots.sum() == 0 or non_roots.sum() == 0:
                continue

            pos_score = onset_scores[b, roots].mean()
            onset_neg = onset_scores[b, non_roots]
            if onset_neg.numel() > 0:
                k = min(5, onset_neg.numel())
                hard_onset_neg = torch.topk(onset_neg, k=k).values.mean()
                losses.append(F.relu(onset_scores.new_tensor(margin) + hard_onset_neg - pos_score))

            effects = (effect_mask[b] > 0.5) & non_roots
            if effects.sum() > 0:
                effect_neg = effect_scores[b, effects]
                k_eff = min(3, effect_neg.numel())
                hard_effect_neg = torch.topk(effect_neg, k=k_eff).values.mean()
                losses.append(F.relu(onset_scores.new_tensor(margin) + hard_effect_neg - pos_score))
        if not losses:
            return onset_scores.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _source_effect_pairwise_margin_loss(
        self,
        source_scores,
        effect_scores,
        source_mask,
        effect_mask,
        margin,
    ):
        margin = max(float(margin), 0.0)
        losses = []
        for b in range(source_scores.shape[0]):
            roots = source_mask[b] > 0.5
            effects = (effect_mask[b] > 0.5) & (~roots)
            if roots.sum() == 0 or effects.sum() == 0:
                continue
            pos_score = source_scores[b, roots].mean()
            hard_effect = torch.topk(
                effect_scores[b, effects],
                k=min(3, int(effects.sum().item())),
            ).values.mean()
            losses.append(
                F.relu(source_scores.new_tensor(margin) + hard_effect - pos_score)
            )
        if not losses:
            return source_scores.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _effect_over_source_margin_loss(
        self,
        effect_scores,
        source_scores,
        source_mask,
        effect_mask,
        margin,
    ):
        """Require delayed effect variables to gain more cross-channel support."""
        margin = max(float(margin), 0.0)
        losses = []
        for b in range(effect_scores.shape[0]):
            roots = source_mask[b] > 0.5
            effects = (effect_mask[b] > 0.5) & (~roots)
            if roots.sum() == 0 or effects.sum() == 0:
                continue
            source_support = source_scores[b, roots].mean()
            effect_support = torch.topk(
                effect_scores[b, effects],
                k=min(3, int(effects.sum().item())),
            ).values.mean()
            losses.append(
                F.relu(effect_scores.new_tensor(margin) + source_support - effect_support)
            )
        if not losses:
            return effect_scores.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _weighted_channel_bce(self, probs, target):
        probs = probs.clamp(1e-5, 1.0 - 1e-5)
        pos = target.sum().clamp_min(1.0)
        neg = (target.numel() - target.sum()).clamp_min(1.0)
        pos_weight = (neg / pos).clamp(1.0, 20.0)
        loss = -(pos_weight * target * torch.log(probs) + (1.0 - target) * torch.log(1.0 - probs))
        return loss.mean()

    def _loss_debug_enabled(self):
        return bool(getattr(self.config, "debug_loss_breakdown", False))

    def _loss_debug_batch_allowed(self):
        if not self._loss_debug_enabled():
            return False
        max_batches = int(getattr(self.config, "debug_loss_max_batches", 0) or 0)
        batch_idx = int(getattr(self, "_current_train_batch", 0) or 0)
        return max_batches <= 0 or batch_idx < max_batches

    def _begin_loss_debug_epoch(self, epoch):
        if not self._loss_debug_enabled():
            return
        self._loss_debug_epoch = int(epoch)
        self._loss_debug_components = {}

    def _record_loss_debug(self, name, value):
        if not self._loss_debug_batch_allowed():
            return
        if value is None:
            return
        if torch.is_tensor(value):
            value = float(value.detach().mean().cpu().item())
        else:
            value = float(value)
        components = getattr(self, "_loss_debug_components", None)
        if components is None:
            components = {}
            self._loss_debug_components = components
        components.setdefault(name, []).append(value)

    def _finish_loss_debug_epoch(self, epoch):
        if not self._loss_debug_enabled():
            return
        components = getattr(self, "_loss_debug_components", {}) or {}
        if not components:
            return
        summary = {
            "epoch": int(epoch),
            "dataset": str(getattr(self, "dataset_name", "Unknown")),
            "components": {},
        }
        for name, values in sorted(components.items()):
            arr = np.asarray(values, dtype=np.float64)
            summary["components"][name] = {
                "count": int(arr.size),
                "mean": float(arr.mean()) if arr.size else 0.0,
                "max": float(arr.max()) if arr.size else 0.0,
                "min": float(arr.min()) if arr.size else 0.0,
            }
        log_path = str(getattr(self.config, "debug_loss_log_path", "") or "")
        if log_path:
            log_dir = os.path.dirname(log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        compact = ", ".join(
            f"{name}={stats['mean']:.4g}"
            for name, stats in summary["components"].items()
            if stats["count"] > 0
        )
        print(f"  [loss-debug] epoch={epoch} {compact}")

    def _source_bottleneck_loss(
        self,
        gate_prob,
        source_mask,
        effect_mask,
        source_onset_mask,
        effect_time_mask,
        event_mask,
    ):
        weight = float(getattr(self.config, "lambda_source_bottleneck", 0.0) or 0.0)
        if weight <= 0 or gate_prob is None:
            return source_mask.new_tensor(0.0)

        bce_weight = float(getattr(self.config, "source_bottleneck_bce_weight", 1.0) or 1.0)
        rank_weight = float(getattr(self.config, "source_bottleneck_rank_weight", 0.5) or 0.5)
        effect_suppress_weight = float(
            getattr(self.config, "source_bottleneck_effect_suppress_weight", 0.5) or 0.5
        )
        effect_margin_weight = float(
            getattr(self.config, "source_bottleneck_effect_margin_weight", 0.0)
            or 0.0
        )
        effect_margin = float(
            getattr(self.config, "source_bottleneck_effect_margin", 0.10) or 0.10
        )
        specificity_weight = float(
            getattr(self.config, "source_bottleneck_specificity_weight", 0.0) or 0.0
        )

        gate_prob = gate_prob.clamp(1e-5, 1.0 - 1e-5)
        event_focus = event_mask.unsqueeze(-1).to(dtype=gate_prob.dtype)
        source_target = (
            source_onset_mask.unsqueeze(-1).to(dtype=gate_prob.dtype)
            * source_mask.unsqueeze(1).to(dtype=gate_prob.dtype)
        )

        total = gate_prob.new_tensor(0.0)
        if bce_weight > 0:
            focus = event_focus.expand_as(gate_prob)
            pos = (source_target * focus).sum().clamp_min(1.0)
            neg = ((1.0 - source_target) * focus).sum().clamp_min(1.0)
            pos_weight = (neg / pos).clamp(1.0, 20.0)
            bce = -(
                pos_weight * source_target * torch.log(gate_prob)
                + (1.0 - source_target) * torch.log(1.0 - gate_prob)
            )
            bce = (bce * focus).sum() / focus.sum().clamp_min(1.0)
            total = total + bce_weight * bce
            self._record_loss_debug("source_bottleneck_bce_raw", bce)
            self._record_loss_debug("source_bottleneck_bce_inner_weighted", bce_weight * bce)

        onset_gate_scores = None
        if rank_weight > 0 or specificity_weight > 0 or effect_margin_weight > 0:
            onset_sum = source_onset_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            onset_gate_scores = (
                gate_prob * source_onset_mask.unsqueeze(-1).to(dtype=gate_prob.dtype)
            ).sum(dim=1) / onset_sum
        if rank_weight > 0 and onset_gate_scores is not None:
            rank_loss = self._synthetic_rca_ranking_loss(
                onset_gate_scores,
                source_mask,
            )
            total = total + rank_weight * rank_loss
            self._record_loss_debug("source_bottleneck_rank_raw", rank_loss)
            self._record_loss_debug("source_bottleneck_rank_inner_weighted", rank_weight * rank_loss)

        if specificity_weight > 0 and onset_gate_scores is not None and onset_gate_scores.shape[0] > 1:
            pred_freq = onset_gate_scores.mean(dim=0)
            target_freq = source_mask.to(dtype=gate_prob.dtype).mean(dim=0)
            pred_dist = pred_freq / pred_freq.sum().clamp_min(1e-6)
            target_dist = target_freq / target_freq.sum().clamp_min(1e-6)
            specificity_loss = F.mse_loss(pred_dist, target_dist, reduction="sum")
            total = total + specificity_weight * specificity_loss
            self._record_loss_debug("source_bottleneck_specificity_raw", specificity_loss)
            self._record_loss_debug(
                "source_bottleneck_specificity_inner_weighted",
                specificity_weight * specificity_loss,
            )

        effect_gate_scores = None
        if (
            (effect_suppress_weight > 0 or effect_margin_weight > 0)
            and effect_mask.sum() > 0
            and effect_time_mask.sum() > 0
        ):
            effect_sum = effect_time_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            effect_gate_scores = (
                gate_prob * effect_time_mask.unsqueeze(-1).to(dtype=gate_prob.dtype)
            ).sum(dim=1) / effect_sum
        if effect_suppress_weight > 0 and effect_gate_scores is not None:
            suppress = (
                effect_gate_scores * effect_mask.to(dtype=gate_prob.dtype)
            ).sum() / effect_mask.sum().clamp_min(1.0)
            total = total + effect_suppress_weight * suppress
            self._record_loss_debug("source_bottleneck_effect_suppress_raw", suppress)
            self._record_loss_debug(
                "source_bottleneck_effect_suppress_inner_weighted",
                effect_suppress_weight * suppress,
            )
        if (
            effect_margin_weight > 0
            and onset_gate_scores is not None
            and effect_gate_scores is not None
        ):
            source_mask_float = source_mask.to(dtype=gate_prob.dtype)
            effect_mask_bool = effect_mask.to(dtype=torch.bool)
            valid = (source_mask.sum(dim=1) > 0) & (effect_mask.sum(dim=1) > 0)
            if valid.any():
                source_count = source_mask_float.sum(dim=1).clamp_min(1.0)
                source_mean = (onset_gate_scores * source_mask_float).sum(dim=1) / source_count
                masked_effect = effect_gate_scores.masked_fill(~effect_mask_bool, -1.0)
                effect_top = masked_effect.max(dim=1).values
                margin_loss = F.relu(effect_top - source_mean + effect_margin)
                margin_loss = margin_loss[valid].mean()
                total = total + effect_margin_weight * margin_loss
                self._record_loss_debug(
                    "source_bottleneck_effect_margin_raw",
                    margin_loss,
                )
                self._record_loss_debug(
                    "source_bottleneck_effect_margin_inner_weighted",
                    effect_margin_weight * margin_loss,
                )

        scaled = weight * total
        self._record_loss_debug("source_bottleneck_total_inner", total)
        self._record_loss_debug("source_bottleneck_total_scaled", scaled)
        return scaled

    def _source_effect_consistency_loss(
        self,
        gate_prob,
        onset_signal,
        mechanism_signal,
        source_mask,
        effect_mask,
        source_onset_mask,
        effect_time_mask,
        onset_sum,
        effect_time_sum,
    ):
        weight = float(getattr(self.config, "source_effect_consistency_weight", 0.0) or 0.0)
        if weight <= 0 or gate_prob is None or mechanism_signal is None:
            return source_mask.new_tensor(0.0)
        if source_mask.sum() <= 0:
            return source_mask.new_tensor(0.0)

        dtype = gate_prob.dtype
        eps = 1e-6
        onset_signal = torch.clamp_min(onset_signal.to(dtype=dtype), 0.0)
        mechanism_signal = torch.clamp_min(mechanism_signal.to(dtype=dtype), 0.0)
        source_time = source_onset_mask.to(dtype=dtype)
        effect_time = effect_time_mask.to(dtype=dtype)
        onset_denom = onset_sum.to(dtype=dtype).clamp_min(1.0)
        effect_denom = effect_time_sum.to(dtype=dtype).clamp_min(1.0)

        def _time_mean(values, mask, denom):
            return torch.einsum("blc,bl->bc", values, mask) / denom

        source_gate = _time_mean(gate_prob, source_time, onset_denom).clamp(0.0, 1.0)
        effect_gate = _time_mean(gate_prob, effect_time, effect_denom).clamp(0.0, 1.0)
        source_onset = _time_mean(onset_signal, source_time, onset_denom)
        effect_onset = _time_mean(onset_signal, effect_time, effect_denom)
        source_mechanism = _time_mean(mechanism_signal, source_time, onset_denom)
        effect_mechanism = _time_mean(mechanism_signal, effect_time, effect_denom)

        def _shared_norm(source_values, effect_values):
            denom = torch.maximum(
                source_values.detach().amax(dim=1, keepdim=True),
                effect_values.detach().amax(dim=1, keepdim=True),
            ).clamp_min(eps)
            return source_values / denom, effect_values / denom

        source_onset_norm, effect_onset_norm = _shared_norm(source_onset, effect_onset)
        source_mech_norm, effect_mech_norm = _shared_norm(source_mechanism, effect_mechanism)
        source_support = torch.sqrt(
            torch.clamp_min(source_onset_norm * source_mech_norm, 0.0) + eps
        )
        effect_support = torch.sqrt(
            torch.clamp_min(effect_onset_norm * effect_mech_norm, 0.0) + eps
        )
        source_consistency = source_gate * source_support
        effect_consistency = effect_gate * effect_support

        total = source_consistency.new_tensor(0.0)
        rank_weight = float(
            getattr(self.config, "source_effect_consistency_rank_weight", 0.75) or 0.0
        )
        if rank_weight > 0:
            rank_loss = self._synthetic_rca_ranking_loss(source_consistency, source_mask)
            total = total + rank_weight * rank_loss
            self._record_loss_debug("source_effect_consistency_rank_raw", rank_loss)

        effect_rank_weight = float(
            getattr(self.config, "source_effect_consistency_effect_rank_weight", 0.75)
            or 0.0
        )
        if effect_rank_weight > 0 and effect_mask.sum() > 0:
            effect_rank_loss = self._source_effect_pairwise_margin_loss(
                source_consistency,
                effect_consistency,
                source_mask,
                effect_mask,
                float(
                    getattr(self.config, "source_effect_consistency_margin", 0.15)
                    or 0.15
                ),
            )
            total = total + effect_rank_weight * effect_rank_loss
            self._record_loss_debug(
                "source_effect_consistency_effect_rank_raw",
                effect_rank_loss,
            )

        suppress_weight = float(
            getattr(self.config, "source_effect_consistency_effect_suppress_weight", 0.25)
            or 0.0
        )
        if suppress_weight > 0 and effect_mask.sum() > 0:
            suppress = (
                effect_consistency * effect_mask.to(dtype=effect_consistency.dtype)
            ).sum() / effect_mask.sum().clamp_min(1.0)
            total = total + suppress_weight * suppress
            self._record_loss_debug("source_effect_consistency_effect_suppress_raw", suppress)

        align_weight = float(
            getattr(self.config, "source_effect_consistency_gate_align_weight", 0.20)
            or 0.0
        )
        if align_weight > 0:
            root_mask = source_mask.to(dtype=source_gate.dtype)
            align = (
                F.relu(source_support.detach() - source_gate) * root_mask
            ).sum() / root_mask.sum().clamp_min(1.0)
            total = total + align_weight * align
            self._record_loss_debug("source_effect_consistency_gate_align_raw", align)

        scaled = weight * total
        self._record_loss_debug("source_effect_consistency_inner", total)
        self._record_loss_debug("source_effect_consistency_scaled", scaled)
        return scaled

    def _channel_masked_modeling_loss(self, input_data, batch_idx=None):
        if not bool(getattr(self.config, "use_channel_masked_modeling", False)):
            return input_data.new_tensor(0.0)
        lambda_channel_masked = float(getattr(self.config, "lambda_channel_masked", 0.0) or 0.0)
        if lambda_channel_masked <= 0:
            return input_data.new_tensor(0.0)
        interval = int(getattr(self.config, "channel_mask_interval", 8) or 8)
        if batch_idx is not None and interval > 1 and batch_idx % interval != 0:
            return input_data.new_tensor(0.0)
        aux_batch_size = int(
            getattr(self.config, "channel_mask_aux_batch_size", 0) or 0
        )
        if 0 < aux_batch_size < input_data.shape[0]:
            input_data = input_data[:aux_batch_size]

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

        rec, A_adaptive, _, _, _, aux_losses, _ = self.model(masked)
        mask = channel_mask[:, None, :]
        denom = mask.sum().clamp_min(1.0) * max(1, L)
        masked_loss = F.smooth_l1_loss(
            rec * mask,
            input_data * mask,
            reduction="sum",
        ) / denom
        total_loss = lambda_channel_masked * masked_loss

        lambda_masked_rca = float(getattr(self.config, "lambda_masked_rca_head", 0.0) or 0.0)
        if aux_losses and lambda_masked_rca > 0:
            masked_rca_logits = aux_losses.get("synthetic_rca_logits")
            if masked_rca_logits is not None:
                pos = channel_mask.sum().clamp_min(1.0)
                neg = (channel_mask.numel() - channel_mask.sum()).clamp_min(1.0)
                pos_weight = (neg / pos).clamp(1.0, 20.0)
                bce = F.binary_cross_entropy_with_logits(
                    masked_rca_logits,
                    channel_mask,
                    pos_weight=pos_weight,
                )
                prob = torch.sigmoid(masked_rca_logits)
                rank = self._synthetic_rca_ranking_loss(prob, channel_mask)
                bce_weight = float(getattr(self.config, "masked_rca_bce_weight", 1.0) or 1.0)
                rank_weight = float(getattr(self.config, "masked_rca_rank_weight", 1.0) or 1.0)
                total_loss = total_loss + lambda_masked_rca * (
                    bce_weight * bce + rank_weight * rank
                )

        if not bool(getattr(self.config, "use_interventional_channel_masking", False)):
            return total_loss

        lambda_source_bce = float(getattr(self.config, "lambda_interventional_source_bce", 0.0) or 0.0)
        lambda_source_rank = float(getattr(self.config, "lambda_interventional_source_rank", 0.0) or 0.0)
        lambda_graph_support = float(getattr(self.config, "lambda_interventional_graph_support", 0.0) or 0.0)

        if aux_losses and lambda_source_bce > 0:
            gate_prob = aux_losses.get("source_gate_prob")
            if gate_prob is not None:
                gate_prob = gate_prob.mean(dim=1).clamp(1e-5, 1.0 - 1e-5)
                total_loss = total_loss + lambda_source_bce * self._weighted_channel_bce(
                    gate_prob,
                    channel_mask,
                )

        if aux_losses and lambda_source_rank > 0:
            rank_signal = str(getattr(self.config, "interventional_rank_signal", "source_gate") or "source_gate").lower()
            if rank_signal == "mechanism":
                source_score = aux_losses.get("channel_mechanism_error")
            elif rank_signal == "reconstruction":
                source_score = F.l1_loss(rec, input_data, reduction="none")
            else:
                source_score = aux_losses.get("source_gate_score")
                if source_score is None:
                    source_score = aux_losses.get("channel_mechanism_error")
            if source_score is not None:
                if source_score.dim() == 3:
                    source_score = source_score.mean(dim=1)
                total_loss = total_loss + lambda_source_rank * self._synthetic_rca_ranking_loss(
                    source_score,
                    channel_mask,
                )

        if lambda_graph_support > 0 and A_adaptive is not None:
            A = A_adaptive.clamp_min(0.0)
            if A.dim() == 3 and A.shape[-1] == C and A.shape[-2] == C:
                eye = torch.eye(C, device=A.device, dtype=A.dtype).unsqueeze(0)
                A = A * (1.0 - eye)
                unmasked_sources = (1.0 - channel_mask).to(dtype=A.dtype).unsqueeze(-1)
                incoming_support = (A * unmasked_sources).sum(dim=1).clamp_min(
                    float(getattr(self.config, "interventional_graph_support_eps", 1e-6) or 1e-6)
                )
                graph_support_loss = -(
                    channel_mask.to(dtype=A.dtype) * torch.log(incoming_support)
                ).sum() / channel_mask.sum().clamp_min(1.0)
                total_loss = total_loss + lambda_graph_support * graph_support_loss

        return total_loss

    def _source_effect_synthetic_loss(self, input_data, batch_idx=None, base_aux=None):
        if not bool(getattr(self.config, "use_source_effect_synthetic", False)):
            return input_data.new_tensor(0.0)
        interval = int(getattr(self.config, "source_effect_interval", 4) or 4)
        if batch_idx is not None and interval > 1 and batch_idx % interval != 0:
            return input_data.new_tensor(0.0)
        aux_batch_size = int(
            getattr(self.config, "source_effect_aux_batch_size", 0) or 0
        )
        if 0 < aux_batch_size < input_data.shape[0]:
            input_data = input_data[:aux_batch_size]
        lambda_source_effect = float(getattr(self.config, "lambda_source_effect", 0.0) or 0.0)
        if lambda_source_effect <= 0:
            return input_data.new_tensor(0.0)

        make_sps_batch = (
            self._make_teacher_source_effect_synthetic_batch
            if bool(getattr(self.config, "use_sps_teacher", False))
            else self._make_source_effect_synthetic_batch
        )
        (
            synth_data,
            event_mask,
            source_mask,
            effect_mask,
            source_onset_mask,
            effect_time_mask,
            propagated_event_mask,
        ) = make_sps_batch(input_data)
        synth_rec, _, _, _, _, synth_aux, _ = self.model(
            synth_data,
            return_root_score=(
                bool(getattr(self.config, "use_root_score_head", False))
                or bool(getattr(self.config, "use_evidence_fusion_head", False))
                or bool(getattr(self.config, "use_response_suppressor_head", False))
                or bool(getattr(self.config, "use_source_interaction_head", False))
                or bool(getattr(self.config, "use_source_consistency_head", False))
                or bool(getattr(self.config, "use_event_route_head", False))
            ),
        )
        if not synth_aux:
            return input_data.new_tensor(0.0)

        mask_sum = event_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        onset_sum = source_onset_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        effect_time_sum = effect_time_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        gate_prob = synth_aux.get("source_gate_prob")
        synth_rca_logits = synth_aux.get("synthetic_rca_logits")
        root_score_logits = synth_aux.get("root_score_logits")
        root_response_source_evidence = synth_aux.get("root_response_source_evidence")
        root_response_response_evidence = synth_aux.get("root_response_response_evidence")
        root_response_pairwise_relation = synth_aux.get("root_response_pairwise_relation")
        event_responsibility_logits = synth_aux.get("event_responsibility_logits")
        event_route_score = synth_aux.get("event_route_score")
        event_route_alpha = synth_aux.get("event_route_alpha")
        dual_expert_graph_gate = synth_aux.get("dual_expert_graph_gate")
        dual_expert_independent_evidence = synth_aux.get(
            "dual_expert_independent_evidence"
        )
        dual_expert_graph_evidence = synth_aux.get("dual_expert_graph_evidence")
        response_suppressor_logits = synth_aux.get("response_suppressor_logits")
        evidence_fusion_logits = synth_aux.get("evidence_fusion_logits")
        evidence_fusion_weights = synth_aux.get("evidence_fusion_weights")
        source_interaction_logits = synth_aux.get("source_interaction_logits")
        source_consistency_logits = synth_aux.get("source_consistency_logits")
        causal_innovation = synth_aux.get("causal_channel_error")
        causal_response_support = synth_aux.get("causal_response_support")
        strict_cross_gain = synth_aux.get("strict_cross_response_gain")
        sps_root_logits = synth_aux.get("sps_root_logits")
        sps_response_logits = synth_aux.get("sps_response_logits")
        gate_channel_scores = None
        if gate_prob is not None:
            gate_channel_scores = (gate_prob * event_mask.unsqueeze(-1)).sum(dim=1) / mask_sum

        channel_err = F.l1_loss(synth_rec, synth_data, reduction="none")
        if channel_err.shape[1] > 1:
            prev_channel_err = torch.cat(
                [
                    torch.zeros_like(channel_err[:, :1, :]),
                    channel_err[:, :-1, :],
                ],
                dim=1,
            )
        else:
            prev_channel_err = torch.zeros_like(channel_err)
        onset_signal = torch.relu(channel_err - prev_channel_err)
        mechanism_signal = synth_aux.get("channel_mechanism_error")

        source_score = synth_aux.get("source_gate_score")
        if source_score is not None:
            channel_scores = (source_score * event_mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        else:
            channel_scores = (channel_err * event_mask.unsqueeze(-1)).sum(dim=1) / mask_sum

        if source_score is not None:
            time_channel_scores = source_score
        else:
            time_channel_scores = channel_err
        onset_channel_scores = (
            time_channel_scores * source_onset_mask.unsqueeze(-1)
        ).sum(dim=1) / onset_sum
        effect_channel_scores = (
            time_channel_scores * effect_time_mask.unsqueeze(-1)
        ).sum(dim=1) / effect_time_sum

        def _configured_weight(name, default):
            value = getattr(self.config, name, default)
            return float(default if value is None else value)

        # A weight of zero is meaningful for the SPS-only profile.  Do not use
        # ``value or default`` here because it silently reactivates old losses.
        bce_weight = _configured_weight("source_effect_bce_weight", 1.0)
        rank_weight = _configured_weight("source_effect_rank_weight", 1.0)
        effect_rank_weight = _configured_weight("source_effect_effect_rank_weight", 0.5)
        onset_rank_weight = _configured_weight("source_effect_onset_rank_weight", 0.0)
        specificity_weight = _configured_weight("source_effect_specificity_weight", 0.0)
        rca_head_weight = float(getattr(self.config, "lambda_source_effect_rca_head", 0.0) or 0.0)
        root_score_weight = float(getattr(self.config, "lambda_source_effect_root_score", 0.0) or 0.0)
        pairwise_root_response_weight = float(
            getattr(self.config, "lambda_pairwise_root_response", 0.0) or 0.0
        )
        event_responsibility_weight = float(
            getattr(self.config, "lambda_event_responsibility", 0.0) or 0.0
        )
        response_suppressor_weight = float(
            getattr(self.config, "lambda_response_suppressor", 0.0) or 0.0
        )
        evidence_fusion_weight = float(
            getattr(self.config, "lambda_evidence_fusion", 0.0) or 0.0
        )
        source_interaction_head_weight = float(
            getattr(self.config, "lambda_source_interaction_head", 0.0) or 0.0
        )
        source_consistency_head_weight = float(
            getattr(self.config, "lambda_source_consistency_head", 0.0) or 0.0
        )
        event_route_weight = float(
            getattr(self.config, "lambda_event_route", 0.0) or 0.0
        )
        graph_alignment_weight = float(
            getattr(self.config, "source_effect_graph_alignment_weight", 0.0) or 0.0
        )
        graph_reverse_weight = float(
            getattr(self.config, "source_effect_graph_reverse_weight", 0.5) or 0.0
        )

        total = input_data.new_tensor(0.0)
        sps_source_weight = float(
            getattr(self.config, "lambda_sps_source", 0.0) or 0.0
        )
        sps_response_weight = float(
            getattr(self.config, "lambda_sps_response", 0.0) or 0.0
        )
        sps_separation_weight = float(
            getattr(self.config, "lambda_sps_separation", 0.0) or 0.0
        )
        if sps_root_logits is not None and sps_source_weight > 0:
            event_root_logits = (
                sps_root_logits
                * source_onset_mask.unsqueeze(-1).to(dtype=sps_root_logits.dtype)
            ).sum(dim=1) / onset_sum.to(dtype=sps_root_logits.dtype)
            source_target = source_mask.to(dtype=sps_root_logits.dtype)
            source_target = source_target / source_target.sum(dim=-1, keepdim=True).clamp_min(1.0)
            source_loss = -(
                source_target * F.log_softmax(event_root_logits, dim=-1)
            ).sum(dim=-1).mean()
            source_rank = self._synthetic_rca_ranking_loss(
                event_root_logits,
                source_mask,
            )
            source_total = source_loss + source_rank
            total = total + sps_source_weight * source_total
            self._record_loss_debug("sps_source_ce", source_loss)
            self._record_loss_debug("sps_source_rank", source_rank)
            source_prob = torch.softmax(event_root_logits, dim=-1)
            self._record_loss_debug(
                "sps_source_logit_std", event_root_logits.detach().std(dim=-1).mean()
            )
            self._record_loss_debug(
                "sps_source_max_probability", source_prob.detach().amax(dim=-1).mean()
            )
            if strict_cross_gain is not None:
                strict_self_error = synth_aux.get("strict_cross_self_error")
                if strict_self_error is not None:
                    primitive_innovation = (
                        strict_self_error
                        * source_onset_mask.unsqueeze(-1).to(dtype=strict_self_error.dtype)
                    ).sum(dim=1) / onset_sum.to(dtype=strict_self_error.dtype)
                    self._record_loss_debug(
                        "sps_input_innovation_rank",
                        self._synthetic_rca_ranking_loss(primitive_innovation, source_mask),
                    )
                    if strict_self_error.shape[1] > 1:
                        primitive_previous = torch.cat(
                            [
                                torch.zeros_like(strict_self_error[:, :1, :]),
                                strict_self_error[:, :-1, :],
                            ],
                            dim=1,
                        )
                        primitive_onset = (strict_self_error - primitive_previous).clamp_min(0.0)
                        primitive_onset = (
                            primitive_onset
                            * source_onset_mask.unsqueeze(-1).to(dtype=strict_self_error.dtype)
                        ).sum(dim=1) / onset_sum.to(dtype=strict_self_error.dtype)
                        self._record_loss_debug(
                            "sps_input_onset_rank",
                            self._synthetic_rca_ranking_loss(primitive_onset, source_mask),
                        )
            self._record_loss_debug(
                "sps_source_weighted", sps_source_weight * source_total
            )

        if sps_response_logits is not None and sps_response_weight > 0:
            event_response_logits = (
                sps_response_logits
                * effect_time_mask.unsqueeze(-1).to(dtype=sps_response_logits.dtype)
            ).sum(dim=1) / effect_time_sum.to(dtype=sps_response_logits.dtype)
            response_loss = self._weighted_channel_bce(
                torch.sigmoid(event_response_logits),
                effect_mask,
            )
            total = total + sps_response_weight * response_loss
            self._record_loss_debug("sps_response_bce", response_loss)
            self._record_loss_debug(
                "sps_response_weighted", sps_response_weight * response_loss
            )

        if strict_cross_gain is not None and sps_separation_weight > 0:
            effect_gain = (
                strict_cross_gain
                * effect_time_mask.unsqueeze(-1).to(dtype=strict_cross_gain.dtype)
            ).sum(dim=1) / effect_time_sum.to(dtype=strict_cross_gain.dtype)
            source_gain = (
                strict_cross_gain
                * source_onset_mask.unsqueeze(-1).to(dtype=strict_cross_gain.dtype)
            ).sum(dim=1) / onset_sum.to(dtype=strict_cross_gain.dtype)
            separation_loss = self._effect_over_source_margin_loss(
                effect_gain,
                source_gain,
                source_mask,
                effect_mask,
                getattr(self.config, "sps_separation_margin", 0.10),
            )
            total = total + sps_separation_weight * separation_loss
            self._record_loss_debug("sps_gain_separation", separation_loss)
            self._record_loss_debug(
                "sps_gain_separation_weighted",
                sps_separation_weight * separation_loss,
            )

        causal_response_weight = float(
            getattr(self.config, "lambda_causal_response", 0.0) or 0.0
        )
        if (
            causal_response_weight > 0
            and causal_innovation is not None
            and causal_response_support is not None
        ):
            source_innovation = (
                causal_innovation
                * source_onset_mask.unsqueeze(-1).to(dtype=causal_innovation.dtype)
            ).sum(dim=1) / onset_sum.to(dtype=causal_innovation.dtype)
            effect_support = (
                causal_response_support
                * effect_time_mask.unsqueeze(-1).to(dtype=causal_response_support.dtype)
            ).sum(dim=1) / effect_time_sum.to(dtype=causal_response_support.dtype)
            source_support = (
                causal_response_support
                * source_onset_mask.unsqueeze(-1).to(dtype=causal_response_support.dtype)
            ).sum(dim=1) / onset_sum.to(dtype=causal_response_support.dtype)
            innovation_rank = self._synthetic_rca_ranking_loss(
                source_innovation,
                source_mask,
            )
            response_rank = self._effect_over_source_margin_loss(
                effect_support,
                source_support,
                source_mask,
                effect_mask,
                getattr(self.config, "causal_response_margin", 0.10),
            )
            causal_response_loss = innovation_rank + response_rank
            total = total + causal_response_weight * causal_response_loss
            self._record_loss_debug("causal_response_innovation_rank", innovation_rank)
            self._record_loss_debug("causal_response_effect_rank", response_rank)
            self._record_loss_debug(
                "causal_response_inner_weighted",
                causal_response_weight * causal_response_loss,
            )
        dual_expert_selection_weight = float(
            getattr(self.config, "lambda_dual_expert_selection", 0.0) or 0.0
        )
        if (
            dual_expert_selection_weight > 0
            and dual_expert_graph_gate is not None
            and dual_expert_independent_evidence is not None
            and dual_expert_graph_evidence is not None
        ):
            def _expert_source_effect_quality(evidence):
                source_scores = (
                    evidence
                    * source_onset_mask.unsqueeze(-1).to(dtype=evidence.dtype)
                ).sum(dim=1) / onset_sum.to(dtype=evidence.dtype)
                effect_scores = (
                    evidence
                    * effect_time_mask.unsqueeze(-1).to(dtype=evidence.dtype)
                ).sum(dim=1) / effect_time_sum.to(dtype=evidence.dtype)
                joined = torch.cat([source_scores, effect_scores], dim=-1)
                center = joined.mean(dim=-1, keepdim=True)
                scale = joined.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
                source_scores = (source_scores - center) / scale
                effect_scores = (effect_scores - center) / scale
                quality = []
                margin = 0.15
                for b in range(source_scores.shape[0]):
                    roots = source_mask[b] > 0.5
                    non_roots = ~roots
                    if roots.sum() == 0 or non_roots.sum() == 0:
                        quality.append(source_scores.new_tensor(0.0))
                        continue
                    root_score = source_scores[b, roots].mean()
                    hard_non_root = torch.topk(
                        source_scores[b, non_roots],
                        k=min(5, int(non_roots.sum().item())),
                    ).values.mean()
                    source_rank = F.relu(
                        source_scores.new_tensor(margin) + hard_non_root - root_score
                    )
                    effects = (effect_mask[b] > 0.5) & non_roots
                    if effects.sum() > 0:
                        hard_effect = torch.topk(
                            effect_scores[b, effects],
                            k=min(3, int(effects.sum().item())),
                        ).values.mean()
                        response_rank = F.relu(
                            source_scores.new_tensor(margin) + hard_effect - root_score
                        )
                    else:
                        response_rank = source_scores.new_tensor(0.0)
                    quality.append(source_rank + 0.5 * response_rank)
                return torch.stack(quality)

            independent_quality = _expert_source_effect_quality(
                dual_expert_independent_evidence
            )
            graph_quality = _expert_source_effect_quality(dual_expert_graph_evidence)
            temperature = max(
                float(
                    getattr(
                        self.config,
                        "dual_expert_selection_temperature",
                        0.25,
                    )
                    or 0.25
                ),
                1e-3,
            )
            quality_graph_target = torch.softmax(
                torch.stack([-independent_quality, -graph_quality], dim=-1)
                / temperature,
                dim=-1,
            )[:, 1].detach()
            mode_target_weight = min(
                max(
                    float(
                        getattr(
                            self.config,
                            "dual_expert_mode_target_weight",
                            0.0,
                        )
                        or 0.0
                    ),
                    0.0,
                ),
                1.0,
            )
            graph_target = (
                mode_target_weight
                * propagated_event_mask.to(dtype=quality_graph_target.dtype)
                + (1.0 - mode_target_weight) * quality_graph_target
            ).detach()
            graph_gate_source = (
                dual_expert_graph_gate.squeeze(-1)
                * source_onset_mask.to(dtype=dual_expert_graph_gate.dtype)
            ).sum(dim=1) / onset_sum.to(dtype=dual_expert_graph_gate.dtype).squeeze(-1)
            selection_loss = F.binary_cross_entropy(
                graph_gate_source.clamp(1e-5, 1.0 - 1e-5),
                graph_target.to(dtype=graph_gate_source.dtype),
            )
            total = total + dual_expert_selection_weight * selection_loss
            self._record_loss_debug(
                "dual_expert_selection_raw",
                selection_loss,
            )
            self._record_loss_debug(
                "dual_expert_selection_inner_weighted",
                dual_expert_selection_weight * selection_loss,
            )
            self._record_loss_debug(
                "dual_expert_graph_target_mean",
                graph_target.mean(),
            )
            self._record_loss_debug(
                "dual_expert_quality_graph_target_mean",
                quality_graph_target.mean(),
            )
            self._record_loss_debug(
                "dual_expert_propagated_event_mean",
                propagated_event_mask.mean(),
            )
            self._record_loss_debug(
                "dual_expert_independent_quality_mean",
                independent_quality.detach().mean(),
            )
            self._record_loss_debug(
                "dual_expert_graph_quality_mean",
                graph_quality.detach().mean(),
            )

            balance_weight = float(
                getattr(self.config, "dual_expert_balance_weight", 0.0) or 0.0
            )
            if balance_weight > 0:
                min_usage = min(
                    max(
                        float(
                            getattr(self.config, "dual_expert_min_usage", 0.10)
                            or 0.10
                        ),
                        0.0,
                    ),
                    0.5,
                )
                usage = dual_expert_graph_gate.mean()
                balance_loss = (
                    F.relu(usage.new_tensor(min_usage) - usage).pow(2)
                    + F.relu(usage - usage.new_tensor(1.0 - min_usage)).pow(2)
                )
                total = total + balance_weight * balance_loss
                self._record_loss_debug("dual_expert_balance_raw", balance_loss)
                self._record_loss_debug(
                    "dual_expert_balance_inner_weighted",
                    balance_weight * balance_loss,
                )
        source_expert_logits = synth_aux.get("source_expert_logits")
        source_expert_graph_gate = synth_aux.get("source_expert_graph_gate")
        source_expert_weight = float(
            getattr(self.config, "lambda_source_expert", 0.0) or 0.0
        )
        if source_expert_logits is not None and source_expert_weight > 0:
            source_expert_prob = torch.sigmoid(source_expert_logits)
            source_expert_scores = (
                source_expert_prob
                * source_onset_mask.unsqueeze(-1).to(dtype=source_expert_prob.dtype)
            ).sum(dim=1) / onset_sum.to(dtype=source_expert_prob.dtype)
            source_expert_bce = self._weighted_channel_bce(
                source_expert_scores,
                source_mask,
            )
            source_expert_rank = self._synthetic_rca_ranking_loss(
                source_expert_scores,
                source_mask,
            )
            source_expert_total = source_expert_bce + source_expert_rank
            effect_scores = None
            if effect_mask.sum() > 0 and effect_time_mask.sum() > 0:
                effect_scores = (
                    source_expert_prob
                    * effect_time_mask.unsqueeze(-1).to(dtype=source_expert_prob.dtype)
                ).sum(dim=1) / effect_time_sum.to(dtype=source_expert_prob.dtype)
                suppress_weight = float(
                    getattr(
                        self.config,
                        "source_expert_effect_suppress_weight",
                        0.25,
                    )
                    or 0.0
                )
                if suppress_weight > 0:
                    source_expert_suppress = (
                        effect_scores * effect_mask.to(dtype=effect_scores.dtype)
                    ).sum() / effect_mask.sum().clamp_min(1.0)
                    source_expert_total = (
                        source_expert_total
                        + suppress_weight * source_expert_suppress
                    )
                    self._record_loss_debug(
                        "source_expert_effect_suppress_raw",
                        source_expert_suppress,
                    )
                pairwise_weight = float(
                    getattr(
                        self.config,
                        "source_expert_pairwise_effect_weight",
                        0.50,
                    )
                    or 0.0
                )
                if pairwise_weight > 0:
                    source_expert_pairwise = self._source_effect_pairwise_margin_loss(
                        source_expert_scores,
                        effect_scores,
                        source_mask,
                        effect_mask,
                        float(
                            getattr(
                                self.config,
                                "source_expert_pairwise_effect_margin",
                                0.15,
                            )
                            or 0.15
                        ),
                    )
                    source_expert_total = (
                        source_expert_total
                        + pairwise_weight * source_expert_pairwise
                    )
                    self._record_loss_debug(
                        "source_expert_pairwise_effect_raw",
                        source_expert_pairwise,
                    )
            total = total + source_expert_weight * source_expert_total
            self._record_loss_debug("source_expert_bce_raw", source_expert_bce)
            self._record_loss_debug("source_expert_rank_raw", source_expert_rank)
            self._record_loss_debug(
                "source_expert_inner_weighted",
                source_expert_weight * source_expert_total,
            )

        source_expert_gate_weight = float(
            getattr(self.config, "lambda_source_expert_gate", 0.0) or 0.0
        )
        if source_expert_graph_gate is not None and source_expert_gate_weight > 0:
            gate_scope = str(
                getattr(self.config, "source_expert_gate_scope", "time") or "time"
            ).lower()
            if gate_scope == "event":
                source_expert_gate_onset = source_expert_graph_gate.reshape(
                    source_expert_graph_gate.shape[0],
                    -1,
                ).mean(dim=1)
            else:
                source_expert_gate_onset = (
                    source_expert_graph_gate.squeeze(-1)
                    * source_onset_mask.to(dtype=source_expert_graph_gate.dtype)
                ).sum(dim=1) / onset_sum.to(dtype=source_expert_graph_gate.dtype).squeeze(-1)
            source_expert_gate_loss = F.binary_cross_entropy(
                source_expert_gate_onset.clamp(1e-5, 1.0 - 1e-5),
                propagated_event_mask.to(dtype=source_expert_gate_onset.dtype),
            )
            total = total + source_expert_gate_weight * source_expert_gate_loss
            self._record_loss_debug(
                "source_expert_gate_raw",
                source_expert_gate_loss,
            )
            self._record_loss_debug(
                "source_expert_gate_inner_weighted",
                source_expert_gate_weight * source_expert_gate_loss,
            )
            self._record_loss_debug(
                "source_expert_gate_mean",
                source_expert_gate_onset.detach().mean(),
            )
            self._record_loss_debug(
                "source_expert_gate_target_mean",
                propagated_event_mask.mean(),
            )
        dense_graph = synth_aux.get("channel_graph_dense")
        if (
            graph_alignment_weight > 0
            and dense_graph is not None
            and effect_mask.sum() > 0
        ):
            source_parent = source_mask.to(dtype=dense_graph.dtype).unsqueeze(-1)
            effect_target = effect_mask.to(dtype=dense_graph.dtype)
            source_to_target = (dense_graph * source_parent).sum(dim=1)
            forward_loss = -(
                effect_target * torch.log(source_to_target.clamp_min(1e-6))
            ).sum() / effect_target.sum().clamp_min(1.0)
            graph_alignment = forward_loss
            self._record_loss_debug("source_effect_graph_forward_raw", forward_loss)

            if graph_reverse_weight > 0:
                effect_parent = effect_mask.to(dtype=dense_graph.dtype).unsqueeze(-1)
                source_target = source_mask.to(dtype=dense_graph.dtype)
                reverse_to_source = (dense_graph * effect_parent).sum(dim=1)
                reverse_loss = -(
                    source_target
                    * torch.log((1.0 - reverse_to_source).clamp_min(1e-6))
                ).sum() / source_target.sum().clamp_min(1.0)
                graph_alignment = graph_alignment + graph_reverse_weight * reverse_loss
                self._record_loss_debug("source_effect_graph_reverse_raw", reverse_loss)

            total = total + graph_alignment_weight * graph_alignment
            self._record_loss_debug(
                "source_effect_graph_inner_weighted",
                graph_alignment_weight * graph_alignment,
            )
        if gate_channel_scores is not None:
            gate_bce = self._weighted_channel_bce(gate_channel_scores, source_mask)
            total = total + bce_weight * gate_bce
            self._record_loss_debug("source_effect_gate_bce_raw", gate_bce)
            self._record_loss_debug("source_effect_gate_bce_inner_weighted", bce_weight * gate_bce)
            bottleneck_loss = self._source_bottleneck_loss(
                gate_prob,
                source_mask,
                effect_mask,
                source_onset_mask,
                effect_time_mask,
                event_mask,
            )
            total = total + bottleneck_loss
            self._record_loss_debug("source_effect_bottleneck_inner_scaled", bottleneck_loss)
            if specificity_weight > 0 and gate_channel_scores.shape[0] > 1:
                pred_freq = gate_channel_scores.clamp_min(0.0).mean(dim=0)
                target_freq = source_mask.to(dtype=gate_channel_scores.dtype).mean(dim=0)
                pred_dist = pred_freq / pred_freq.sum().clamp_min(1e-6)
                target_dist = target_freq / target_freq.sum().clamp_min(1e-6)
                gate_specificity = F.mse_loss(
                    pred_dist,
                    target_dist,
                    reduction="sum",
                )
                total = total + specificity_weight * gate_specificity
                self._record_loss_debug("source_effect_specificity_raw", gate_specificity)
                self._record_loss_debug(
                    "source_effect_specificity_inner_weighted",
                    specificity_weight * gate_specificity,
                )
        consistency_loss = self._source_effect_consistency_loss(
            gate_prob,
            onset_signal,
            mechanism_signal,
            source_mask,
            effect_mask,
            source_onset_mask,
            effect_time_mask,
            onset_sum,
            effect_time_sum,
        )
        total = total + consistency_loss
        if synth_rca_logits is not None and rca_head_weight > 0:
            pos = source_mask.sum().clamp_min(1.0)
            neg = (source_mask.numel() - source_mask.sum()).clamp_min(1.0)
            pos_weight = (neg / pos).clamp(1.0, 20.0)
            rca_head_bce = F.binary_cross_entropy_with_logits(
                synth_rca_logits,
                source_mask,
                pos_weight=pos_weight,
            )
            rca_head_prob = torch.sigmoid(synth_rca_logits)
            rca_head_rank = self._synthetic_rca_ranking_loss(rca_head_prob, source_mask)
            rca_head_bce_weight = float(
                getattr(self.config, "source_effect_rca_head_bce_weight", 1.0) or 1.0
            )
            rca_head_rank_weight = float(
                getattr(self.config, "source_effect_rca_head_rank_weight", 1.0) or 1.0
            )
            total = total + rca_head_weight * (
                rca_head_bce_weight * rca_head_bce
                + rca_head_rank_weight * rca_head_rank
            )
            self._record_loss_debug("source_effect_rca_head_bce_raw", rca_head_bce)
            self._record_loss_debug("source_effect_rca_head_rank_raw", rca_head_rank)
            self._record_loss_debug(
                "source_effect_rca_head_inner_weighted",
                rca_head_weight
                * (rca_head_bce_weight * rca_head_bce + rca_head_rank_weight * rca_head_rank),
            )
        if root_score_logits is not None and root_score_weight > 0:
            root_target = (
                source_onset_mask.unsqueeze(-1).to(dtype=root_score_logits.dtype)
                * source_mask.unsqueeze(1).to(dtype=root_score_logits.dtype)
            )
            event_focus = event_mask.unsqueeze(-1).to(dtype=root_score_logits.dtype).expand_as(root_score_logits)
            root_bce_weight = float(getattr(self.config, "root_score_bce_weight", 1.0) or 1.0)
            root_rank_weight = float(getattr(self.config, "root_score_rank_weight", 1.0) or 1.0)
            root_effect_suppress_weight = float(
                getattr(self.config, "root_score_effect_suppress_weight", 0.5) or 0.0
            )
            pos = (root_target * event_focus).sum().clamp_min(1.0)
            neg = ((1.0 - root_target) * event_focus).sum().clamp_min(1.0)
            pos_weight = (neg / pos).clamp(1.0, 20.0)
            root_bce = F.binary_cross_entropy_with_logits(
                root_score_logits,
                root_target,
                pos_weight=pos_weight,
                reduction="none",
            )
            root_bce = (root_bce * event_focus).sum() / event_focus.sum().clamp_min(1.0)

            root_prob = torch.sigmoid(root_score_logits)
            root_onset_scores = (
                root_prob * source_onset_mask.unsqueeze(-1).to(dtype=root_prob.dtype)
            ).sum(dim=1) / onset_sum
            root_rank = self._synthetic_rca_ranking_loss(root_onset_scores, source_mask)
            root_total = root_bce_weight * root_bce + root_rank_weight * root_rank
            if root_effect_suppress_weight > 0 and effect_mask.sum() > 0 and effect_time_mask.sum() > 0:
                root_effect_scores = (
                    root_prob * effect_time_mask.unsqueeze(-1).to(dtype=root_prob.dtype)
                ).sum(dim=1) / effect_time_sum
                effect_suppress = (
                    root_effect_scores * effect_mask.to(dtype=root_prob.dtype)
                ).sum() / effect_mask.sum().clamp_min(1.0)
                root_total = root_total + root_effect_suppress_weight * effect_suppress
            total = total + root_score_weight * root_total
            self._record_loss_debug("source_effect_root_score_bce_raw", root_bce)
            self._record_loss_debug("source_effect_root_score_rank_raw", root_rank)
            self._record_loss_debug(
                "source_effect_root_score_inner_weighted",
                root_score_weight * root_total,
            )
        if (
            root_response_pairwise_relation is not None
            and pairwise_root_response_weight > 0
            and effect_mask.sum() > 0
            and source_mask.sum() > 0
        ):
            relation = root_response_pairwise_relation.clamp_min(1e-8)
            src = source_mask.to(dtype=relation.dtype)
            eff = effect_mask.to(dtype=relation.dtype)
            source_count = src.sum(dim=1).clamp_min(1.0)
            effect_count = eff.sum(dim=1).clamp_min(1.0)
            source_to_effect = (relation * src.unsqueeze(-1) * eff.unsqueeze(1)).sum(
                dim=(1, 2)
            ) / source_count
            incoming_from_source = (relation * src.unsqueeze(-1)).sum(dim=1)
            effect_coverage = (incoming_from_source * eff).sum(dim=1) / effect_count
            valid = ((src.sum(dim=1) > 0) & (eff.sum(dim=1) > 0)).to(dtype=relation.dtype)
            source_to_effect = source_to_effect.clamp(max=1.0)
            effect_coverage = effect_coverage.clamp(max=1.0)
            pair_loss = -(
                torch.log(source_to_effect.clamp_min(1e-6))
                + torch.log(effect_coverage.clamp_min(1e-6))
            )
            pair_loss = (pair_loss * valid).sum() / valid.sum().clamp_min(1.0)
            total = total + pairwise_root_response_weight * pair_loss
            self._record_loss_debug("pairwise_root_response_raw", pair_loss)
            self._record_loss_debug(
                "pairwise_root_response_inner_weighted",
                pairwise_root_response_weight * pair_loss,
            )
        if event_responsibility_logits is not None and event_responsibility_weight > 0:
            focus_mask = source_onset_mask
            focus_sum = onset_sum
            if focus_mask.sum() <= 0:
                focus_mask = event_mask
                focus_sum = mask_sum
            responsibility_logits = (
                event_responsibility_logits
                * focus_mask.unsqueeze(-1).to(dtype=event_responsibility_logits.dtype)
            ).sum(dim=1) / focus_sum
            target = source_mask.to(dtype=responsibility_logits.dtype)
            loss_mode = str(
                getattr(self.config, "event_responsibility_loss_mode", "bce") or "bce"
            ).lower()
            if loss_mode in {"softmax", "softmax_ce", "ce"}:
                target_dist = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
                log_q = F.log_softmax(responsibility_logits, dim=-1)
                q = torch.softmax(responsibility_logits, dim=-1)
                resp_primary = -(target_dist * log_q).sum(dim=1).mean()
            else:
                q = torch.sigmoid(responsibility_logits)
                bce_raw = F.binary_cross_entropy_with_logits(
                    responsibility_logits,
                    target,
                    reduction="none",
                )
                if bool(getattr(self.config, "event_responsibility_bce_balance", True)):
                    pos_count = target.sum().clamp_min(1.0)
                    neg_count = (1.0 - target).sum().clamp_min(1.0)
                    numel = float(target.numel())
                    pos_weight = numel / (2.0 * pos_count)
                    neg_weight = numel / (2.0 * neg_count)
                    bce_weight = torch.where(target > 0.5, pos_weight, neg_weight)
                    resp_primary = (bce_raw * bce_weight).mean()
                else:
                    resp_primary = bce_raw.mean()
            resp_rank = self._synthetic_rca_ranking_loss(q, source_mask)
            resp_total = (
                float(getattr(self.config, "event_responsibility_ce_weight", 1.0) or 1.0)
                * resp_primary
                + float(getattr(self.config, "event_responsibility_rank_weight", 0.5) or 0.0)
                * resp_rank
            )
            suppress_weight = float(
                getattr(self.config, "event_responsibility_effect_suppress_weight", 0.25)
                or 0.0
            )
            if suppress_weight > 0 and effect_mask.sum() > 0 and effect_time_mask.sum() > 0:
                resp_prob_time = torch.sigmoid(event_responsibility_logits)
                resp_effect_scores = (
                    resp_prob_time * effect_time_mask.unsqueeze(-1).to(dtype=resp_prob_time.dtype)
                ).sum(dim=1) / effect_time_sum
                resp_effect_suppress = (
                    resp_effect_scores * effect_mask.to(dtype=resp_effect_scores.dtype)
                ).sum() / effect_mask.sum().clamp_min(1.0)
                resp_total = resp_total + suppress_weight * resp_effect_suppress
                self._record_loss_debug(
                    "event_responsibility_effect_suppress_raw",
                    resp_effect_suppress,
                )
            entropy_weight = float(
                getattr(self.config, "event_responsibility_entropy_weight", 0.0) or 0.0
            )
            if entropy_weight > 0:
                entropy = -(q.clamp_min(1e-8) * q.clamp_min(1e-8).log()).sum(dim=-1)
                entropy = entropy.mean() / max(math.log(max(2, q.shape[-1])), 1e-8)
                resp_total = resp_total + entropy_weight * entropy
                self._record_loss_debug("event_responsibility_entropy_raw", entropy)
            total = total + event_responsibility_weight * resp_total
            self._record_loss_debug("event_responsibility_primary_raw", resp_primary)
            self._record_loss_debug("event_responsibility_rank_raw", resp_rank)
            self._record_loss_debug(
                "event_responsibility_inner_weighted",
                event_responsibility_weight * resp_total,
            )
        if response_suppressor_logits is not None and response_suppressor_weight > 0:
            response_focus = effect_time_mask
            response_focus_sum = effect_time_sum
            if response_focus.sum() <= 0:
                response_focus = event_mask
                response_focus_sum = mask_sum
            response_logits = (
                response_suppressor_logits
                * response_focus.unsqueeze(-1).to(dtype=response_suppressor_logits.dtype)
            ).sum(dim=1) / response_focus_sum.to(dtype=response_suppressor_logits.dtype)
            response_target = effect_mask.to(dtype=response_logits.dtype)
            pos = response_target.sum().clamp_min(1.0)
            neg = (response_target.numel() - response_target.sum()).clamp_min(1.0)
            pos_weight = (neg / pos).clamp(1.0, 20.0)
            response_bce = F.binary_cross_entropy_with_logits(
                response_logits,
                response_target,
                pos_weight=pos_weight,
            )
            response_prob = torch.sigmoid(response_logits)
            response_rank = self._synthetic_rca_ranking_loss(
                response_prob,
                effect_mask,
            )
            response_total = (
                float(getattr(self.config, "response_suppressor_bce_weight", 1.0) or 1.0)
                * response_bce
                + float(getattr(self.config, "response_suppressor_rank_weight", 0.5) or 0.0)
                * response_rank
            )
            leak_weight = float(
                getattr(self.config, "response_suppressor_source_leak_weight", 0.5)
                or 0.0
            )
            if leak_weight > 0 and source_mask.sum() > 0 and source_onset_mask.sum() > 0:
                leak_prob_time = torch.sigmoid(response_suppressor_logits)
                source_leak_scores = (
                    leak_prob_time
                    * source_onset_mask.unsqueeze(-1).to(dtype=leak_prob_time.dtype)
                ).sum(dim=1) / onset_sum.to(dtype=leak_prob_time.dtype)
                source_leak = (
                    source_leak_scores * source_mask.to(dtype=leak_prob_time.dtype)
                ).sum() / source_mask.sum().clamp_min(1.0)
                response_total = response_total + leak_weight * source_leak
                self._record_loss_debug(
                    "response_suppressor_source_leak_raw",
                    source_leak,
                )
            total = total + response_suppressor_weight * response_total
            self._record_loss_debug("response_suppressor_bce_raw", response_bce)
            self._record_loss_debug("response_suppressor_rank_raw", response_rank)
            self._record_loss_debug(
                "response_suppressor_inner_weighted",
                response_suppressor_weight * response_total,
            )
        if evidence_fusion_logits is not None and evidence_fusion_weight > 0:
            focus_mask = source_onset_mask
            focus_sum = onset_sum
            if focus_mask.sum() <= 0:
                focus_mask = event_mask
                focus_sum = mask_sum
            fusion_logits = (
                evidence_fusion_logits
                * focus_mask.unsqueeze(-1).to(dtype=evidence_fusion_logits.dtype)
            ).sum(dim=1) / focus_sum
            target = source_mask.to(dtype=fusion_logits.dtype)
            fusion_loss_mode = str(
                getattr(self.config, "evidence_fusion_loss_mode", "bce") or "bce"
            ).lower()
            if fusion_loss_mode in {"softmax", "softmax_ce", "ce", "competition"}:
                target_dist = target / target.sum(dim=1, keepdim=True).clamp_min(1.0)
                log_q = F.log_softmax(fusion_logits, dim=-1)
                fusion_primary = -(target_dist * log_q).sum(dim=1).mean()
                fusion_prob = torch.softmax(fusion_logits, dim=-1)
            else:
                pos_count = target.sum().clamp_min(1.0)
                neg_count = (1.0 - target).sum().clamp_min(1.0)
                pos_weight = (neg_count / pos_count).clamp(1.0, 20.0)
                fusion_primary = F.binary_cross_entropy_with_logits(
                    fusion_logits,
                    target,
                    pos_weight=pos_weight,
                )
                fusion_prob = torch.sigmoid(fusion_logits)
            fusion_rank = self._synthetic_rca_ranking_loss(fusion_prob, source_mask)
            fusion_total = (
                float(getattr(self.config, "evidence_fusion_bce_weight", 1.0) or 1.0)
                * fusion_primary
                + float(getattr(self.config, "evidence_fusion_rank_weight", 1.0) or 0.0)
                * fusion_rank
            )
            suppress_weight = float(
                getattr(self.config, "evidence_fusion_effect_suppress_weight", 0.25)
                or 0.0
            )
            if suppress_weight > 0 and effect_mask.sum() > 0 and effect_time_mask.sum() > 0:
                if fusion_loss_mode in {"softmax", "softmax_ce", "ce", "competition"}:
                    fusion_prob_time = torch.softmax(evidence_fusion_logits, dim=-1)
                else:
                    fusion_prob_time = torch.sigmoid(evidence_fusion_logits)
                fusion_effect_scores = (
                    fusion_prob_time * effect_time_mask.unsqueeze(-1).to(dtype=fusion_prob_time.dtype)
                ).sum(dim=1) / effect_time_sum
                fusion_effect_suppress = (
                    fusion_effect_scores * effect_mask.to(dtype=fusion_effect_scores.dtype)
                ).sum() / effect_mask.sum().clamp_min(1.0)
                fusion_total = fusion_total + suppress_weight * fusion_effect_suppress
                self._record_loss_debug(
                    "evidence_fusion_effect_suppress_raw",
                    fusion_effect_suppress,
                )
            pairwise_effect_weight = float(
                getattr(self.config, "evidence_fusion_pairwise_effect_weight", 0.0)
                or 0.0
            )
            if (
                pairwise_effect_weight > 0
                and effect_mask.sum() > 0
                and effect_time_mask.sum() > 0
            ):
                source_time = source_onset_mask.to(dtype=evidence_fusion_logits.dtype)
                effect_time = effect_time_mask.to(dtype=evidence_fusion_logits.dtype)
                fusion_source_scores = torch.einsum(
                    "blc,bl->bc",
                    evidence_fusion_logits,
                    source_time,
                ) / onset_sum.to(dtype=evidence_fusion_logits.dtype)
                fusion_effect_scores = torch.einsum(
                    "blc,bl->bc",
                    evidence_fusion_logits,
                    effect_time,
                ) / effect_time_sum.to(dtype=evidence_fusion_logits.dtype)
                pairwise_effect_margin = float(
                    getattr(
                        self.config,
                        "evidence_fusion_pairwise_effect_margin",
                        0.2,
                    )
                    or 0.2
                )
                fusion_pairwise_effect = self._source_effect_pairwise_margin_loss(
                    fusion_source_scores,
                    fusion_effect_scores,
                    source_mask,
                    effect_mask,
                    pairwise_effect_margin,
                )
                fusion_total = (
                    fusion_total + pairwise_effect_weight * fusion_pairwise_effect
                )
                self._record_loss_debug(
                    "evidence_fusion_pairwise_effect_raw",
                    fusion_pairwise_effect,
                )
            entropy_weight = float(
                getattr(self.config, "evidence_fusion_entropy_weight", 0.0) or 0.0
            )
            if (
                entropy_weight > 0
                and evidence_fusion_weights is not None
                and evidence_fusion_weights.shape[-1] > 1
            ):
                weights = evidence_fusion_weights.clamp_min(1e-8)
                weight_entropy = -(weights * weights.log()).sum(dim=-1)
                weight_entropy = weight_entropy / max(math.log(weights.shape[-1]), 1e-8)
                focused_entropy = (
                    weight_entropy
                    * focus_mask.unsqueeze(-1).to(dtype=weight_entropy.dtype)
                ).sum() / (
                    focus_mask.sum().clamp_min(1.0) * max(1, weight_entropy.shape[-1])
                )
                fusion_total = fusion_total + entropy_weight * focused_entropy
                self._record_loss_debug("evidence_fusion_weight_entropy_raw", focused_entropy)
            total = total + evidence_fusion_weight * fusion_total
            self._record_loss_debug("evidence_fusion_primary_raw", fusion_primary)
            self._record_loss_debug("evidence_fusion_rank_raw", fusion_rank)
            self._record_loss_debug(
                "evidence_fusion_inner_weighted",
                evidence_fusion_weight * fusion_total,
            )
        if source_interaction_logits is not None and source_interaction_head_weight > 0:
            interaction_target = (
                source_onset_mask.unsqueeze(-1).to(dtype=source_interaction_logits.dtype)
                * source_mask.unsqueeze(1).to(dtype=source_interaction_logits.dtype)
            )
            interaction_focus = event_mask.unsqueeze(-1).to(
                dtype=source_interaction_logits.dtype
            ).expand_as(source_interaction_logits)
            pos = (interaction_target * interaction_focus).sum().clamp_min(1.0)
            neg = ((1.0 - interaction_target) * interaction_focus).sum().clamp_min(1.0)
            pos_weight = (neg / pos).clamp(1.0, 20.0)
            interaction_bce = F.binary_cross_entropy_with_logits(
                source_interaction_logits,
                interaction_target,
                pos_weight=pos_weight,
                reduction="none",
            )
            interaction_bce = (
                interaction_bce * interaction_focus
            ).sum() / interaction_focus.sum().clamp_min(1.0)
            interaction_prob = torch.sigmoid(source_interaction_logits)
            interaction_onset_scores = (
                interaction_prob
                * source_onset_mask.unsqueeze(-1).to(dtype=interaction_prob.dtype)
            ).sum(dim=1) / onset_sum
            interaction_rank = self._synthetic_rca_ranking_loss(
                interaction_onset_scores,
                source_mask,
            )
            interaction_total = (
                float(getattr(self.config, "source_interaction_head_bce_weight", 1.0) or 1.0)
                * interaction_bce
                + float(getattr(self.config, "source_interaction_head_rank_weight", 1.0) or 0.0)
                * interaction_rank
            )
            suppress_weight = float(
                getattr(self.config, "source_interaction_head_effect_suppress_weight", 0.25)
                or 0.0
            )
            if suppress_weight > 0 and effect_mask.sum() > 0 and effect_time_mask.sum() > 0:
                interaction_effect_scores = (
                    interaction_prob
                    * effect_time_mask.unsqueeze(-1).to(dtype=interaction_prob.dtype)
                ).sum(dim=1) / effect_time_sum
                interaction_effect_suppress = (
                    interaction_effect_scores * effect_mask.to(dtype=interaction_prob.dtype)
                ).sum() / effect_mask.sum().clamp_min(1.0)
                interaction_total = (
                    interaction_total + suppress_weight * interaction_effect_suppress
                )
                self._record_loss_debug(
                    "source_interaction_head_effect_suppress_raw",
                    interaction_effect_suppress,
                )
            pairwise_effect_weight = float(
                getattr(self.config, "source_interaction_head_pairwise_effect_weight", 0.5)
                or 0.0
            )
            if pairwise_effect_weight > 0 and effect_mask.sum() > 0 and effect_time_mask.sum() > 0:
                source_time = source_onset_mask.to(dtype=source_interaction_logits.dtype)
                effect_time = effect_time_mask.to(dtype=source_interaction_logits.dtype)
                interaction_source_scores = torch.einsum(
                    "blc,bl->bc",
                    source_interaction_logits,
                    source_time,
                ) / onset_sum.to(dtype=source_interaction_logits.dtype)
                interaction_effect_scores = torch.einsum(
                    "blc,bl->bc",
                    source_interaction_logits,
                    effect_time,
                ) / effect_time_sum.to(dtype=source_interaction_logits.dtype)
                interaction_pairwise = self._source_effect_pairwise_margin_loss(
                    interaction_source_scores,
                    interaction_effect_scores,
                    source_mask,
                    effect_mask,
                    float(
                        getattr(
                            self.config,
                            "source_interaction_head_pairwise_effect_margin",
                            0.15,
                        )
                        or 0.15
                    ),
                )
                interaction_total = (
                    interaction_total + pairwise_effect_weight * interaction_pairwise
                )
                self._record_loss_debug(
                    "source_interaction_head_pairwise_effect_raw",
                    interaction_pairwise,
                )
            total = total + source_interaction_head_weight * interaction_total
            self._record_loss_debug("source_interaction_head_bce_raw", interaction_bce)
            self._record_loss_debug("source_interaction_head_rank_raw", interaction_rank)
            self._record_loss_debug(
                "source_interaction_head_inner_weighted",
                source_interaction_head_weight * interaction_total,
            )
        if source_consistency_logits is not None and source_consistency_head_weight > 0:
            consistency_target = (
                source_onset_mask.unsqueeze(-1).to(dtype=source_consistency_logits.dtype)
                * source_mask.unsqueeze(1).to(dtype=source_consistency_logits.dtype)
            )
            consistency_focus = event_mask.unsqueeze(-1).to(
                dtype=source_consistency_logits.dtype
            ).expand_as(source_consistency_logits)
            pos = (consistency_target * consistency_focus).sum().clamp_min(1.0)
            neg = ((1.0 - consistency_target) * consistency_focus).sum().clamp_min(1.0)
            pos_weight = (neg / pos).clamp(1.0, 20.0)
            consistency_bce = F.binary_cross_entropy_with_logits(
                source_consistency_logits,
                consistency_target,
                pos_weight=pos_weight,
                reduction="none",
            )
            consistency_bce = (
                consistency_bce * consistency_focus
            ).sum() / consistency_focus.sum().clamp_min(1.0)
            consistency_prob = torch.sigmoid(source_consistency_logits)
            consistency_onset_scores = (
                consistency_prob
                * source_onset_mask.unsqueeze(-1).to(dtype=consistency_prob.dtype)
            ).sum(dim=1) / onset_sum
            consistency_rank = self._synthetic_rca_ranking_loss(
                consistency_onset_scores,
                source_mask,
            )
            consistency_total = (
                float(getattr(self.config, "source_consistency_head_bce_weight", 0.75) or 0.0)
                * consistency_bce
                + float(getattr(self.config, "source_consistency_head_rank_weight", 1.0) or 0.0)
                * consistency_rank
            )
            suppress_weight = float(
                getattr(self.config, "source_consistency_head_effect_suppress_weight", 0.25)
                or 0.0
            )
            if suppress_weight > 0 and effect_mask.sum() > 0 and effect_time_mask.sum() > 0:
                consistency_effect_scores = (
                    consistency_prob
                    * effect_time_mask.unsqueeze(-1).to(dtype=consistency_prob.dtype)
                ).sum(dim=1) / effect_time_sum
                consistency_effect_suppress = (
                    consistency_effect_scores * effect_mask.to(dtype=consistency_prob.dtype)
                ).sum() / effect_mask.sum().clamp_min(1.0)
                consistency_total = (
                    consistency_total + suppress_weight * consistency_effect_suppress
                )
                self._record_loss_debug(
                    "source_consistency_head_effect_suppress_raw",
                    consistency_effect_suppress,
                )
            pairwise_effect_weight = float(
                getattr(self.config, "source_consistency_head_pairwise_effect_weight", 0.50)
                or 0.0
            )
            if pairwise_effect_weight > 0 and effect_mask.sum() > 0 and effect_time_mask.sum() > 0:
                source_time = source_onset_mask.to(dtype=source_consistency_logits.dtype)
                effect_time = effect_time_mask.to(dtype=source_consistency_logits.dtype)
                consistency_source_scores = torch.einsum(
                    "blc,bl->bc",
                    source_consistency_logits,
                    source_time,
                ) / onset_sum.to(dtype=source_consistency_logits.dtype)
                consistency_effect_scores = torch.einsum(
                    "blc,bl->bc",
                    source_consistency_logits,
                    effect_time,
                ) / effect_time_sum.to(dtype=source_consistency_logits.dtype)
                consistency_pairwise = self._source_effect_pairwise_margin_loss(
                    consistency_source_scores,
                    consistency_effect_scores,
                    source_mask,
                    effect_mask,
                    float(
                        getattr(
                            self.config,
                            "source_consistency_head_pairwise_effect_margin",
                            0.15,
                        )
                        or 0.15
                    ),
                )
                consistency_total = (
                    consistency_total + pairwise_effect_weight * consistency_pairwise
                )
                self._record_loss_debug(
                    "source_consistency_head_pairwise_effect_raw",
                    consistency_pairwise,
                )
            total = total + source_consistency_head_weight * consistency_total
            self._record_loss_debug("source_consistency_head_bce_raw", consistency_bce)
            self._record_loss_debug("source_consistency_head_rank_raw", consistency_rank)
            self._record_loss_debug(
                "source_consistency_head_inner_weighted",
                source_consistency_head_weight * consistency_total,
            )
        if event_route_score is not None and event_route_weight > 0:
            route_source_scores = (
                event_route_score
                * source_onset_mask.unsqueeze(-1).to(dtype=event_route_score.dtype)
            ).sum(dim=1) / onset_sum.to(dtype=event_route_score.dtype)
            route_rank_weight = float(
                getattr(self.config, "event_route_rank_weight", 1.0) or 0.0
            )
            route_total = route_source_scores.new_tensor(0.0)
            route_rank = self._synthetic_rca_ranking_loss(
                route_source_scores,
                source_mask,
            )
            route_total = route_total + route_rank_weight * route_rank
            self._record_loss_debug("event_route_rank_raw", route_rank)

            route_effect_scores = None
            if effect_mask.sum() > 0 and effect_time_mask.sum() > 0:
                route_effect_scores = (
                    event_route_score
                    * effect_time_mask.unsqueeze(-1).to(dtype=event_route_score.dtype)
                ).sum(dim=1) / effect_time_sum.to(dtype=event_route_score.dtype)
                suppress_weight = float(
                    getattr(self.config, "event_route_effect_suppress_weight", 0.25)
                    or 0.0
                )
                if suppress_weight > 0:
                    route_effect_suppress = (
                        route_effect_scores
                        * effect_mask.to(dtype=route_effect_scores.dtype)
                    ).sum() / effect_mask.sum().clamp_min(1.0)
                    route_total = route_total + suppress_weight * route_effect_suppress
                    self._record_loss_debug(
                        "event_route_effect_suppress_raw",
                        route_effect_suppress,
                    )
                pairwise_weight = float(
                    getattr(self.config, "event_route_pairwise_effect_weight", 0.50)
                    or 0.0
                )
                if pairwise_weight > 0:
                    route_pairwise = self._source_effect_pairwise_margin_loss(
                        route_source_scores,
                        route_effect_scores,
                        source_mask,
                        effect_mask,
                        float(
                            getattr(
                                self.config,
                                "event_route_pairwise_effect_margin",
                                0.15,
                            )
                            or 0.15
                        ),
                    )
                    route_total = route_total + pairwise_weight * route_pairwise
                    self._record_loss_debug(
                        "event_route_pairwise_effect_raw",
                        route_pairwise,
                    )

            alpha_weight = float(
                getattr(self.config, "event_route_alpha_weight", 0.25) or 0.0
            )
            if alpha_weight > 0 and event_route_alpha is not None:
                alpha = event_route_alpha.view(-1).clamp(1e-5, 1.0 - 1e-5)
                if dense_graph is not None:
                    graph = dense_graph.detach().clamp_min(0.0)
                    src = source_mask.to(dtype=graph.dtype)
                    eff = effect_mask.to(dtype=graph.dtype)
                    incoming = (graph * src.unsqueeze(-1)).sum(dim=1)
                    effect_count = eff.sum(dim=1).clamp_min(1.0)
                    alpha_target = (incoming * eff).sum(dim=1) / effect_count
                    has_effect = (effect_mask.sum(dim=1) > 0).to(dtype=alpha_target.dtype)
                    alpha_target = (alpha_target * has_effect).clamp(0.0, 1.0)
                else:
                    alpha_target = torch.zeros_like(alpha)
                alpha_loss = F.binary_cross_entropy(alpha, alpha_target.to(dtype=alpha.dtype))
                route_total = route_total + alpha_weight * alpha_loss
                self._record_loss_debug("event_route_alpha_raw", alpha_loss)
                self._record_loss_debug("event_route_alpha_mean", alpha.detach().mean())

            total = total + event_route_weight * route_total
            self._record_loss_debug(
                "event_route_inner_weighted",
                event_route_weight * route_total,
            )
        source_branch_weight = float(
            getattr(self.config, "root_response_source_branch_weight", 0.0) or 0.0
        )
        response_branch_weight = float(
            getattr(self.config, "root_response_response_branch_weight", 0.0) or 0.0
        )
        response_suppress_weight = float(
            getattr(self.config, "root_response_response_suppress_weight", 0.5) or 0.0
        )
        if root_response_source_evidence is not None and source_branch_weight > 0:
            source_branch_logits = (
                root_response_source_evidence
                * source_onset_mask.unsqueeze(-1).to(dtype=root_response_source_evidence.dtype)
            ).sum(dim=1) / onset_sum
            pos = source_mask.sum().clamp_min(1.0)
            neg = (source_mask.numel() - source_mask.sum()).clamp_min(1.0)
            pos_weight = (neg / pos).clamp(1.0, 20.0)
            source_branch_bce = F.binary_cross_entropy_with_logits(
                source_branch_logits,
                source_mask,
                pos_weight=pos_weight,
            )
            source_branch_rank = self._synthetic_rca_ranking_loss(
                torch.sigmoid(source_branch_logits),
                source_mask,
            )
            source_branch_loss = source_branch_bce + source_branch_rank
            total = total + source_branch_weight * source_branch_loss
            self._record_loss_debug("root_response_source_branch_bce_raw", source_branch_bce)
            self._record_loss_debug("root_response_source_branch_rank_raw", source_branch_rank)
            self._record_loss_debug(
                "root_response_source_branch_inner_weighted",
                source_branch_weight * source_branch_loss,
            )
        if (
            root_response_response_evidence is not None
            and response_branch_weight > 0
            and effect_mask.sum() > 0
            and effect_time_mask.sum() > 0
        ):
            response_effect_scores = (
                root_response_response_evidence
                * effect_time_mask.unsqueeze(-1).to(dtype=root_response_response_evidence.dtype)
            ).sum(dim=1) / effect_time_sum
            response_branch_rank = self._synthetic_rca_ranking_loss(
                response_effect_scores,
                effect_mask,
            )
            response_source_scores = (
                root_response_response_evidence
                * source_onset_mask.unsqueeze(-1).to(dtype=root_response_response_evidence.dtype)
            ).sum(dim=1) / onset_sum
            response_source_suppress = (
                response_source_scores * source_mask.to(dtype=response_source_scores.dtype)
            ).sum() / source_mask.sum().clamp_min(1.0)
            response_branch_loss = (
                response_branch_rank
                + response_suppress_weight * response_source_suppress
            )
            total = total + response_branch_weight * response_branch_loss
            self._record_loss_debug("root_response_response_branch_rank_raw", response_branch_rank)
            self._record_loss_debug(
                "root_response_response_source_suppress_raw",
                response_source_suppress,
            )
            self._record_loss_debug(
                "root_response_response_branch_inner_weighted",
                response_branch_weight * response_branch_loss,
            )
        if rank_weight > 0:
            rank_loss = self._synthetic_rca_ranking_loss(channel_scores, source_mask)
            total = total + rank_weight * rank_loss
            self._record_loss_debug("source_effect_rank_raw", rank_loss)
            self._record_loss_debug("source_effect_rank_inner_weighted", rank_weight * rank_loss)
        if effect_rank_weight > 0:
            effect_rank_loss = self._source_effect_ranking_loss(
                channel_scores,
                source_mask,
                effect_mask,
            )
            total = total + effect_rank_weight * effect_rank_loss
            self._record_loss_debug("source_effect_effect_rank_raw", effect_rank_loss)
            self._record_loss_debug(
                "source_effect_effect_rank_inner_weighted",
                effect_rank_weight * effect_rank_loss,
            )
        if onset_rank_weight > 0:
            onset_rank_loss = self._source_effect_onset_ranking_loss(
                onset_channel_scores,
                effect_channel_scores,
                source_mask,
                effect_mask,
            )
            total = total + onset_rank_weight * onset_rank_loss
            self._record_loss_debug("source_effect_onset_rank_raw", onset_rank_loss)
            self._record_loss_debug(
                "source_effect_onset_rank_inner_weighted",
                onset_rank_weight * onset_rank_loss,
            )
        scaled_total = lambda_source_effect * total
        self._record_loss_debug("source_effect_total_inner", total)
        self._record_loss_debug("source_effect_total_scaled", scaled_total)
        return scaled_total

    def _residual_sps_auxiliary_loss(self, input_data, base_aux, batch_idx=None):
        """Train source/response roles in residual space without replacing SPS."""
        if not bool(getattr(self.config, "use_sps_residual_synthetic", False)):
            return input_data.new_tensor(0.0)
        loss_weight = float(getattr(self.config, "lambda_residual_sps", 0.0) or 0.0)
        if loss_weight <= 0:
            return input_data.new_tensor(0.0)
        interval = int(getattr(self.config, "source_effect_interval", 4) or 4)
        if batch_idx is not None and interval > 1 and batch_idx % interval != 0:
            return input_data.new_tensor(0.0)
        residual = (base_aux or {}).get("strict_cross_input")
        if residual is None:
            return input_data.new_tensor(0.0)
        aux_batch_size = int(
            getattr(self.config, "residual_sps_aux_batch_size", 0) or 0
        )
        if 0 < aux_batch_size < residual.shape[0]:
            residual = residual[:aux_batch_size]

        make_batch = (
            self._make_teacher_source_effect_synthetic_batch
            if bool(getattr(self.config, "use_sps_teacher", False))
            else self._make_source_effect_synthetic_batch
        )
        (
            synth_residual,
            _,
            source_mask,
            effect_mask,
            source_onset_mask,
            effect_time_mask,
            _,
        ) = make_batch(residual)
        role_aux = self._get_raw_model().forward_sps_roles_from_residual(
            synth_residual
        )
        root_logits = role_aux.get("sps_root_logits")
        response_logits = role_aux.get("sps_response_logits")
        response_gain = role_aux.get("strict_cross_response_gain")
        if root_logits is None or response_logits is None:
            return input_data.new_tensor(0.0)

        onset_sum = source_onset_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        effect_sum = effect_time_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        event_root_logits = (
            root_logits
            * source_onset_mask.unsqueeze(-1).to(dtype=root_logits.dtype)
        ).sum(dim=1) / onset_sum.to(dtype=root_logits.dtype)
        source_target = source_mask.to(dtype=root_logits.dtype)
        source_target = source_target / source_target.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1.0)
        source_ce = -(
            source_target * F.log_softmax(event_root_logits, dim=-1)
        ).sum(dim=-1).mean()
        source_rank = self._synthetic_rca_ranking_loss(
            event_root_logits,
            source_mask,
        )
        source_weight = float(
            getattr(self.config, "residual_sps_source_weight", 1.0) or 0.0
        )
        total = source_weight * (source_ce + source_rank)

        event_response_logits = (
            response_logits
            * effect_time_mask.unsqueeze(-1).to(dtype=response_logits.dtype)
        ).sum(dim=1) / effect_sum.to(dtype=response_logits.dtype)
        response_loss = self._weighted_channel_bce(
            torch.sigmoid(event_response_logits),
            effect_mask,
        )
        response_weight = float(
            getattr(self.config, "residual_sps_response_weight", 0.5) or 0.0
        )
        total = total + response_weight * response_loss

        separation_loss = input_data.new_tensor(0.0)
        separation_weight = float(
            getattr(self.config, "residual_sps_separation_weight", 0.5) or 0.0
        )
        if response_gain is not None and separation_weight > 0:
            effect_gain = (
                response_gain
                * effect_time_mask.unsqueeze(-1).to(dtype=response_gain.dtype)
            ).sum(dim=1) / effect_sum.to(dtype=response_gain.dtype)
            source_gain = (
                response_gain
                * source_onset_mask.unsqueeze(-1).to(dtype=response_gain.dtype)
            ).sum(dim=1) / onset_sum.to(dtype=response_gain.dtype)
            separation_loss = self._effect_over_source_margin_loss(
                effect_gain,
                source_gain,
                source_mask,
                effect_mask,
                getattr(self.config, "sps_separation_margin", 0.10),
            )
            total = total + separation_weight * separation_loss

        scaled = loss_weight * total
        self._record_loss_debug("residual_sps_source_ce", source_ce)
        self._record_loss_debug("residual_sps_source_rank", source_rank)
        self._record_loss_debug("residual_sps_response_bce", response_loss)
        self._record_loss_debug("residual_sps_gain_separation", separation_loss)
        self._record_loss_debug("residual_sps_total_scaled", scaled)
        source_prob = torch.softmax(event_root_logits, dim=-1)
        self._record_loss_debug(
            "residual_sps_max_probability",
            source_prob.detach().amax(dim=-1).mean(),
        )
        fusion_scale = (base_aux or {}).get("sps_fusion_scale")
        if fusion_scale is not None:
            self._record_loss_debug("residual_sps_fusion_scale", fusion_scale)
        return scaled

    def _synthetic_rca_ranking_loss(self, channel_scores, channel_mask):
        margin = float(getattr(self.config, "synthetic_rca_margin", 0.2) or 0.2)
        hard_topk = int(getattr(self.config, "synthetic_rca_topk", 5) or 5)
        positive_aggregation = str(
            getattr(self.config, "synthetic_rca_positive_aggregation", "mean") or "mean"
        ).lower()
        positive_topk = int(getattr(self.config, "synthetic_rca_positive_topk", 1) or 1)
        positive_min_weight = float(
            getattr(self.config, "synthetic_rca_positive_min_weight", 0.0) or 0.0
        )
        positive_min_weight = min(max(positive_min_weight, 0.0), 1.0)
        losses = []
        for b in range(channel_scores.shape[0]):
            roots = channel_mask[b] > 0.5
            non_roots = ~roots
            if roots.sum() == 0 or non_roots.sum() == 0:
                continue
            pos_scores = channel_scores[b, roots]
            if positive_aggregation in {"min", "weakest"}:
                pos_score = pos_scores.min()
            elif positive_aggregation in {"bottomk", "bottom", "weak_mean"}:
                pos_k = min(max(1, positive_topk), pos_scores.numel())
                pos_score = torch.topk(pos_scores, k=pos_k, largest=False).values.mean()
            elif positive_aggregation in {"blend", "mean_bottomk", "mean_weak"}:
                pos_k = min(max(1, positive_topk), pos_scores.numel())
                weak_score = torch.topk(pos_scores, k=pos_k, largest=False).values.mean()
                pos_score = (1.0 - positive_min_weight) * pos_scores.mean() + positive_min_weight * weak_score
            else:
                pos_score = pos_scores.mean()
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
            num_workers=getattr(self.config, "dataloader_num_workers", 0),
            prefetch_factor=getattr(self.config, "dataloader_prefetch_factor", 2),
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
                num_workers=getattr(self.config, "dataloader_num_workers", 0),
                prefetch_factor=getattr(self.config, "dataloader_prefetch_factor", 2),
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
            num_workers=getattr(self.config, "dataloader_num_workers", 0),
            prefetch_factor=getattr(self.config, "dataloader_prefetch_factor", 2),
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

    def _make_score_stats_loader(self, scaled_data: pd.DataFrame, tag: str):
        batch_size = min(self.config.batch_size, 64)
        stats_num_workers = int(getattr(self.config, "score_stats_num_workers", 0) or 0)
        max_batches = int(getattr(self.config, "score_stats_max_batches", 128) or 0)
        base_loader = anomaly_detection_data_provider(
            scaled_data,
            batch_size=batch_size,
            win_size=self.config.win_size,
            step=1,
            mode="test",
            num_workers=0,
            prefetch_factor=getattr(self.config, "dataloader_prefetch_factor", 2),
        )
        dataset = base_loader.dataset
        total_windows = len(dataset)
        sampled_windows = total_windows
        if max_batches > 0 and total_windows > max_batches * batch_size:
            sampled_windows = max_batches * batch_size
            sample_idx = np.linspace(
                0,
                total_windows - 1,
                num=sampled_windows,
                dtype=np.int64,
            ).tolist()
            dataset = Subset(dataset, sample_idx)

        loader_kwargs = dict(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=stats_num_workers,
            pin_memory=True,
            persistent_workers=True if stats_num_workers > 0 else False,
            drop_last=False,
        )
        if stats_num_workers > 0:
            loader_kwargs["prefetch_factor"] = getattr(self.config, "dataloader_prefetch_factor", 2)
        loader = DataLoader(**loader_kwargs)
        print(
            f"  [{tag}] stats windows={sampled_windows:,}/{total_windows:,}, "
            f"batches={len(loader)}, workers={stats_num_workers}"
        )
        return loader, stats_num_workers

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
        loader, _ = self._make_score_stats_loader(scaled_data, "ChannelMechanism")

        scores = []
        max_samples = int(getattr(self.config, "score_stats_max_samples", 2_000_000) or 2_000_000)
        per_batch_samples = max(1, int(math.ceil(max_samples / max(len(loader), 1))))
        for input_data, _ in loader:
            input_data = input_data.float().to(self.device)
            _, _, _, _, _, aux_losses, _ = self.model(input_data)
            mechanism_score = aux_losses.get("channel_mechanism_score") if aux_losses else None
            if mechanism_score is not None:
                batch_scores = mechanism_score.detach().cpu().numpy().reshape(-1)
                if 0 < per_batch_samples < batch_scores.size:
                    sample_idx = np.linspace(
                        0,
                        batch_scores.size - 1,
                        num=per_batch_samples,
                        dtype=np.int64,
                    )
                    batch_scores = batch_scores[sample_idx]
                scores.append(batch_scores)

        if not scores:
            return

        scores = np.concatenate(scores, axis=0)
        if max_samples > 0 and scores.size > max_samples:
            sample_idx = np.linspace(0, scores.size - 1, num=max_samples, dtype=np.int64)
            scores = scores[sample_idx]
        score_center = float(np.median(scores))
        q25 = float(np.percentile(scores, 25))
        q75 = float(np.percentile(scores, 75))
        eps = float(getattr(self.config, "channel_mechanism_score_eps", 1e-6) or 1e-6)
        score_scale = max(q75 - q25, float(np.std(scores)), abs(score_center), eps)
        raw_model.set_channel_mechanism_score_stats(score_center, score_scale)
        print(
            "  [ChannelMechanism] score median="
            f"{score_center:.6f}, scale={score_scale:.6f}, samples={scores.size:,}"
        )

    @torch.no_grad()
    def _fit_channel_graph_reliability(self, train_data: pd.DataFrame):
        if train_data is None or self.model is None:
            return
        raw_model = self._get_raw_model()
        if not hasattr(raw_model, "set_channel_graph_reliability"):
            return

        print("\n  [ChannelGraph] Calibrating per-channel graph reliability on normal windows...")
        if self._should_load_best_checkpoint():
            raw_model.load_state_dict(self.early_stopping.check_point)
        self.model.to(self.device)
        self.model.eval()

        scaled_data = self._transform_input_frame(train_data)
        loader, _ = self._make_score_stats_loader(scaled_data, "ChannelGraphReliability")
        squared_error_sum = None
        target_variance_sum = None
        window_count = 0
        for input_data, _ in loader:
            input_data = input_data.float().to(self.device)
            _, _, _, _, _, aux_losses, _ = self.model(input_data)
            if not aux_losses:
                continue
            mechanism_error = aux_losses.get("channel_mechanism_error")
            target_variance = aux_losses.get("channel_graph_target_variance")
            if mechanism_error is None or target_variance is None:
                continue
            batch_squared_error = mechanism_error.pow(2).mean(dim=1).sum(dim=0)
            batch_target_variance = target_variance.sum(dim=0)
            if squared_error_sum is None:
                squared_error_sum = batch_squared_error.double()
                target_variance_sum = batch_target_variance.double()
            else:
                squared_error_sum += batch_squared_error.double()
                target_variance_sum += batch_target_variance.double()
            window_count += int(input_data.shape[0])

        if squared_error_sum is None or window_count <= 0:
            print("  [ChannelGraph] Reliability calibration skipped: no mechanism evidence.")
            return

        eps = 1e-8
        nmse = squared_error_sum / target_variance_sum.clamp_min(eps)
        reliability = (1.0 - nmse).clamp(0.0, 1.0).float()
        raw_model.set_channel_graph_reliability(reliability)
        rel_cpu = reliability.detach().cpu().numpy()
        print(
            "  [ChannelGraph] reliability "
            f"min={rel_cpu.min():.4f}, median={np.median(rel_cpu):.4f}, "
            f"mean={rel_cpu.mean():.4f}, max={rel_cpu.max():.4f}, "
            f">0.5={int((rel_cpu > 0.5).sum())}/{rel_cpu.size}, windows={window_count:,}"
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
        loader, _ = self._make_score_stats_loader(scaled_data, "SynthAux")

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
        self._fit_sps_teacher(train_scaled)

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

        if getattr(self.config, "use_channel_graph_reliability", False):
            self._fit_channel_graph_reliability(self._train_raw)

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
            channel_graph_evidence_only=getattr(
                self.config, "channel_graph_evidence_only", False
            ),
            channel_graph_type=getattr(
                self.config, "channel_graph_type", "adaptive_symmetric"
            ),
            directed_graph_parent_topk=getattr(
                self.config, "directed_graph_parent_topk", 5
            ),
            directed_graph_embedding_dim=getattr(
                self.config, "directed_graph_embedding_dim", 8
            ),
            directed_graph_use_signed_transfer=getattr(
                self.config, "directed_graph_use_signed_transfer", False
            ),
            directed_graph_transfer_mode=getattr(
                self.config, "directed_graph_transfer_mode", "low_rank"
            ),
            directed_graph_transfer_rank=getattr(
                self.config, "directed_graph_transfer_rank", 8
            ),
            directed_graph_transfer_scale=getattr(
                self.config, "directed_graph_transfer_scale", 2.0
            ),
            use_channel_graph_reliability=getattr(
                self.config, "use_channel_graph_reliability", False
            ),
            channel_graph_role=getattr(self.config, "channel_graph_role", "shared"),
            use_multilag_graph_propagation=getattr(
                self.config, "use_multilag_graph_propagation", False
            ),
            graph_propagation_lags=getattr(
                self.config, "graph_propagation_lags", (0, 1, 2, 4, 8)
            ),
            graph_propagation_lag_mode=getattr(
                self.config, "graph_propagation_lag_mode", "static"
            ),
            graph_propagation_alignment_temperature=getattr(
                self.config, "graph_propagation_alignment_temperature", 0.2
            ),
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
            use_causal_response_evidence=getattr(
                self.config, "use_causal_response_evidence", False
            ),
            use_strict_cross_mechanism=getattr(
                self.config, "use_strict_cross_mechanism", False
            ),
            strict_cross_lags=getattr(
                self.config, "strict_cross_lags", [1, 3, 6, 12]
            ),
            strict_cross_topk=getattr(self.config, "strict_cross_topk", 5),
            strict_cross_detach_backbone=getattr(
                self.config, "strict_cross_detach_backbone", True
            ),
            strict_cross_use_channel_prior=getattr(
                self.config, "strict_cross_use_channel_prior", False
            ),
            use_sps_role_head=getattr(self.config, "use_sps_role_head", False),
            sps_role_hidden=getattr(self.config, "sps_role_hidden", 16),
            sps_role_detach_features=getattr(
                self.config, "sps_role_detach_features", False
            ),
            use_bounded_sps_fusion=getattr(
                self.config, "use_bounded_sps_fusion", False
            ),
            bounded_sps_max_correction=getattr(
                self.config, "bounded_sps_max_correction", 0.25
            ),
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
            use_source_preserving_decoder=getattr(
                self.config, "use_source_preserving_decoder", False
            ),
            source_preserving_init=getattr(self.config, "source_preserving_init", 0.65),
            source_preserving_detach_gate=getattr(
                self.config, "source_preserving_detach_gate", True
            ),
            use_mechanism_residual_feedback=getattr(
                self.config, "use_mechanism_residual_feedback", False
            ),
            mechanism_feedback_init=getattr(self.config, "mechanism_feedback_init", 0.10),
            mechanism_feedback_detach=getattr(self.config, "mechanism_feedback_detach", True),
            mechanism_feedback_norm=getattr(self.config, "mechanism_feedback_norm", "sample_l1"),
            mechanism_feedback_clip=getattr(self.config, "mechanism_feedback_clip", 3.0),
            use_mechanism_predictive_head=getattr(self.config, "use_mechanism_predictive_head", False),
            mechanism_predictive_blend_init=getattr(self.config, "mechanism_predictive_blend_init", 0.30),
            use_source_gate=getattr(self.config, "use_source_gate", False),
            source_gate_init=getattr(self.config, "source_gate_init", 0.20),
            use_onset_aware_source_gate=getattr(
                self.config,
                "use_onset_aware_source_gate",
                False,
            ),
            source_gate_onset_window=getattr(
                self.config,
                "source_gate_onset_window",
                8,
            ),
            source_gate_onset_weight=getattr(
                self.config,
                "source_gate_onset_weight",
                0.5,
            ),
            use_dual_expert_fusion=getattr(
                self.config,
                "use_dual_expert_fusion",
                False,
            ),
            dual_expert_hidden=getattr(self.config, "dual_expert_hidden", 16),
            dual_expert_graph_init=getattr(
                self.config,
                "dual_expert_graph_init",
                0.5,
            ),
            dual_expert_detach_inputs=getattr(
                self.config,
                "dual_expert_detach_inputs",
                True,
            ),
            use_source_expert_branch=getattr(
                self.config,
                "use_source_expert_branch",
                False,
            ),
            source_expert_hidden=getattr(self.config, "source_expert_hidden", 16),
            source_expert_graph_init=getattr(
                self.config,
                "source_expert_graph_init",
                0.5,
            ),
            source_expert_detach_gate_inputs=getattr(
                self.config,
                "source_expert_detach_gate_inputs",
                True,
            ),
            source_expert_gradient_checkpoint=getattr(
                self.config,
                "source_expert_gradient_checkpoint",
                False,
            ),
            source_expert_gate_scope=getattr(
                self.config,
                "source_expert_gate_scope",
                "time",
            ),
            use_root_score_head=getattr(self.config, "use_root_score_head", False),
            root_score_head_mode=getattr(self.config, "root_score_head_mode", "mlp"),
            root_score_detach_features=getattr(self.config, "root_score_detach_features", True),
            root_response_penalty_init=getattr(self.config, "root_response_penalty_init", 1.0),
            root_response_confidence_discount=getattr(
                self.config,
                "root_response_confidence_discount",
                0.75,
            ),
            root_response_use_innovation_split=getattr(
                self.config,
                "root_response_use_innovation_split",
                False,
            ),
            use_pairwise_root_response_head=getattr(
                self.config,
                "use_pairwise_root_response_head",
                False,
            ),
            pairwise_root_response_rank=getattr(
                self.config,
                "pairwise_root_response_rank",
                8,
            ),
            pairwise_root_response_graph_weight=getattr(
                self.config,
                "pairwise_root_response_graph_weight",
                0.75,
            ),
            pairwise_root_response_reward_init=getattr(
                self.config,
                "pairwise_root_response_reward_init",
                0.25,
            ),
            pairwise_root_response_penalty_init=getattr(
                self.config,
                "pairwise_root_response_penalty_init",
                0.50,
            ),
            pairwise_root_response_logit_weight=getattr(
                self.config,
                "pairwise_root_response_logit_weight",
                1.0,
            ),
            use_event_responsibility_head=getattr(
                self.config,
                "use_event_responsibility_head",
                False,
            ),
            event_responsibility_hidden=getattr(
                self.config,
                "event_responsibility_hidden",
                16,
            ),
            event_responsibility_detach_features=getattr(
                self.config,
                "event_responsibility_detach_features",
                False,
            ),
            use_event_route_head=getattr(
                self.config,
                "use_event_route_head",
                False,
            ),
            event_route_hidden=getattr(self.config, "event_route_hidden", 16),
            event_route_init=getattr(self.config, "event_route_init", 0.55),
            event_route_detach_inputs=getattr(
                self.config,
                "event_route_detach_inputs",
                True,
            ),
            event_route_response_penalty=getattr(
                self.config,
                "event_route_response_penalty",
                0.0,
            ),
            use_response_suppressor_head=getattr(
                self.config,
                "use_response_suppressor_head",
                False,
            ),
            response_suppressor_hidden=getattr(
                self.config,
                "response_suppressor_hidden",
                16,
            ),
            response_suppressor_detach_inputs=getattr(
                self.config,
                "response_suppressor_detach_inputs",
                True,
            ),
            use_evidence_fusion_head=getattr(
                self.config,
                "use_evidence_fusion_head",
                False,
            ),
            evidence_fusion_hidden=getattr(
                self.config,
                "evidence_fusion_hidden",
                16,
            ),
            evidence_fusion_detach_inputs=getattr(
                self.config,
                "evidence_fusion_detach_inputs",
                False,
            ),
            evidence_fusion_use_graph_response=getattr(
                self.config,
                "evidence_fusion_use_graph_response",
                False,
            ),
            use_source_interaction_head=getattr(
                self.config,
                "use_source_interaction_head",
                False,
            ),
            source_interaction_detach_inputs=getattr(
                self.config,
                "source_interaction_detach_inputs",
                True,
            ),
            use_source_consistency_head=getattr(
                self.config,
                "use_source_consistency_head",
                False,
            ),
            source_consistency_detach_inputs=getattr(
                self.config,
                "source_consistency_detach_inputs",
                True,
            ),
            use_channel_temporal_corefinement=getattr(
                self.config, "use_channel_temporal_corefinement", False
            ),
            corefinement_init=getattr(self.config, "corefinement_init", 0.10),
            corefinement_detach_first_pass=getattr(
                self.config, "corefinement_detach_first_pass", True
            ),
            use_source_aware_corefinement=getattr(
                self.config, "use_source_aware_corefinement", False
            ),
            source_aware_corefinement_init=getattr(
                self.config, "source_aware_corefinement_init", 0.15
            ),
            source_aware_corefinement_detach_gate=getattr(
                self.config, "source_aware_corefinement_detach_gate", True
            ),
            use_state_aware_fusion=getattr(self.config, "use_state_aware_fusion", False),
            state_aware_num_states=getattr(self.config, "state_aware_num_states", 4),
            state_aware_graph_gate_init=getattr(self.config, "state_aware_graph_gate_init", 0.6),
            state_aware_residual_init=getattr(self.config, "state_aware_residual_init", 0.15),
        )
        self.model.to(self.device)

        use_model_channel_prior = bool(getattr(self.config, "use_channel_corr_prior", False))
        use_source_effect_prior = bool(
            getattr(self.config, "use_source_effect_synthetic", False)
            and getattr(self.config, "source_effect_use_channel_prior", False)
        )
        use_strict_cross_prior = bool(
            getattr(self.config, "use_strict_cross_mechanism", False)
            and getattr(self.config, "strict_cross_use_channel_prior", False)
        )
        if use_model_channel_prior or use_source_effect_prior or use_strict_cross_prior:
            prior_topk = getattr(self.config, "channel_corr_prior_topk", 5)
            if use_source_effect_prior and not use_model_channel_prior:
                prior_topk = getattr(self.config, "source_effect_prior_topk", prior_topk)
            if use_model_channel_prior:
                model_prior = self._build_channel_corr_prior(
                    train_df,
                    topk=getattr(self.config, "channel_corr_prior_topk", 5),
                    selection_axis=getattr(
                        self.config, "channel_corr_prior_selection_axis", "source"
                    ),
                )
                self.model.set_channel_static_prior(model_prior)
                print(
                    f"  [ChannelPrior] normal correlation prior set "
                    f"(topk={getattr(self.config, 'channel_corr_prior_topk', 5)}, "
                    f"weight={getattr(self.config, 'channel_corr_prior_weight', 0.0)}, "
                    f"axis={getattr(self.config, 'channel_corr_prior_selection_axis', 'source')}, "
                    f"bias={getattr(self.config, 'channel_corr_prior_bias', 0.0)})"
                )
            if use_strict_cross_prior:
                strict_prior = self._build_channel_corr_prior(
                    train_df,
                    topk=getattr(self.config, "channel_corr_prior_topk", 5),
                    selection_axis="target",
                )
                self.model.set_strict_cross_static_prior(strict_prior)
                print(
                    f"  [StrictCrossPrior] normal correlation prior set "
                    f"(topk={getattr(self.config, 'channel_corr_prior_topk', 5)})"
                )
            if use_source_effect_prior:
                source_prior = self._build_channel_corr_prior(
                    train_df,
                    topk=getattr(self.config, "source_effect_prior_topk", prior_topk),
                    selection_axis="source",
                )
                self._channel_corr_prior = source_prior
                print(
                    f"  [SourceEffectPrior] normal correlation prior set for synthetic "
                    f"source-effect sampling only (topk={prior_topk})"
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
        sps_params = []
        other_params = []
        for name, param in self.model.named_parameters():
            if (
                name.startswith("strict_cross_mechanism")
                or name.startswith("sps_role_head")
                or name.startswith("sps_fusion_scale_raw")
            ):
                sps_params.append(param)
            elif 'channel_graph' in name:
                channel_graph_params.append(param)
            elif 'temporal_graph' in name:
                temporal_graph_params.append(param)
            else:
                other_params.append(param)

        weight_decay_v10 = getattr(self.config, 'weight_decay', 1e-4)
        channel_lr_scale = getattr(self.config, 'channel_graph_lr_scale', 0.1)
        temporal_lr_scale = getattr(self.config, 'temporal_graph_lr_scale', 0.1)
        sps_lr_scale = max(float(getattr(self.config, 'sps_lr_scale', 10.0)), 1.0)

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
        if sps_params:
            param_groups.append({
                'params': sps_params,
                'lr': self.config.lr * sps_lr_scale,
                'param_names': ['strict_cross_sps'],
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
            self._begin_loss_debug_epoch(epoch + 1)

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
            debug_train_max_batches = int(getattr(self.config, "debug_train_max_batches", 0) or 0)

            for i, (input_data, _) in enumerate(self.train_loader):
                if debug_train_max_batches > 0 and i >= debug_train_max_batches:
                    break
                self._current_train_batch = i
                input_data = input_data.float().to(self.device, non_blocking=True)
                rec, _, _, _, _, aux_losses, _ = self.model(input_data)

                loss = self._reconstruction_loss(rec, input_data)
                self._record_loss_debug("reconstruction_loss", loss)
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
                        self._record_loss_debug("vq_loss_scaled", lambda_vq * aux_losses['vq_loss'])
                    loss = loss + self._synthetic_anomaly_aux_loss(input_data, aux_losses, batch_idx=i)
                    loss = loss + self._source_effect_synthetic_loss(
                        input_data,
                        batch_idx=i,
                        base_aux=aux_losses,
                    )
                    loss = loss + self._residual_sps_auxiliary_loss(
                        input_data,
                        aux_losses,
                        batch_idx=i,
                    )
                    loss = loss + self._channel_masked_modeling_loss(input_data, batch_idx=i)

                self._record_loss_debug("total_train_loss", loss)
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
            self._finish_loss_debug_epoch(epoch + 1)
            val_loss = self.vali(self.valid_loader)
            current_lr = self.optimizer.param_groups[0]["lr"]

            selection_score = None
            selection_note = None
            if (
                bool(getattr(self.config, "use_rca_aware_checkpoint", False))
                and (epoch + 1) >= int(getattr(self.config, "rca_checkpoint_min_epoch", 1) or 1)
            ):
                proxy = self._compute_rca_checkpoint_proxy(self.valid_loader)
                if proxy is not None:
                    proxy_weight = float(
                        getattr(self.config, "rca_checkpoint_proxy_weight", 0.0) or 0.0
                    )
                    if bool(getattr(self.config, "rca_checkpoint_normalize", False)):
                        if not hasattr(self, "_rca_checkpoint_val_ref"):
                            self._rca_checkpoint_val_ref = max(abs(float(val_loss)), 1e-10)
                        val_norm = float(val_loss) / max(float(self._rca_checkpoint_val_ref), 1e-10)
                        proxy_norm = min(max(float(proxy["source_mrr"]), 0.0), 1.0)
                        selection_score = val_norm - proxy_weight * proxy_norm
                        selection_note = (
                            f"val_norm={val_norm:.4f}, "
                            f"proxy_weight={proxy_weight:.3f}, "
                        )
                    else:
                        selection_score = val_loss - proxy_weight * proxy["source_mrr"]
                        selection_note = ""
                    selection_note = (
                        selection_note +
                        f"source_mrr={proxy['source_mrr']:.4f}, "
                        f"source_hit1={proxy['source_hit1']:.4f}, "
                        f"source_margin={proxy['source_margin']:.4f}, "
                        f"n={proxy['num_synthetic_events']}"
                    )

            es_status = self.early_stopping(
                val_loss,
                self.model,
                epoch + 1,
                selection_score=selection_score,
                selection_note=selection_note,
            )

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
        if self._should_load_best_checkpoint():
            self._get_raw_model().load_state_dict(self.early_stopping.check_point)
            print(
                f"  [Checkpoint] Using validation-selected checkpoint: "
                f"epoch {self.early_stopping.best_epoch}/"
                f"{getattr(self.early_stopping, 'last_epoch', self.config.num_epochs)} "
                f"(monitor={getattr(self.early_stopping, 'monitor_name', 'val_loss')})"
            )
        elif bool(getattr(self.config, "use_latest_checkpoint", False)):
            print("  [Checkpoint] Using latest epoch checkpoint for evaluation.")

        saved_path = self._save_best_params(total_params=total_params, total_time=total_time)

        self._save_train_history(total_params=total_params, total_time=total_time)

        print("─" * 70)
        print(f"  [OK] Training Complete!")
        print(f"     Best Val Loss: {self.early_stopping.val_loss_min:.6f}  @  Epoch {self.early_stopping.best_epoch}")
        print(f"     Epochs Run:    {getattr(self.early_stopping, 'last_epoch', self.config.num_epochs)} / {self.config.num_epochs}")
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
            channel_graph_evidence_only=getattr(
                self.config, "channel_graph_evidence_only", False
            ),
            channel_graph_type=getattr(
                self.config, "channel_graph_type", "adaptive_symmetric"
            ),
            directed_graph_parent_topk=getattr(
                self.config, "directed_graph_parent_topk", 5
            ),
            directed_graph_embedding_dim=getattr(
                self.config, "directed_graph_embedding_dim", 8
            ),
            directed_graph_use_signed_transfer=getattr(
                self.config, "directed_graph_use_signed_transfer", False
            ),
            directed_graph_transfer_mode=getattr(
                self.config, "directed_graph_transfer_mode", "low_rank"
            ),
            directed_graph_transfer_rank=getattr(
                self.config, "directed_graph_transfer_rank", 8
            ),
            directed_graph_transfer_scale=getattr(
                self.config, "directed_graph_transfer_scale", 2.0
            ),
            use_channel_graph_reliability=getattr(
                self.config, "use_channel_graph_reliability", False
            ),
            channel_graph_role=getattr(self.config, "channel_graph_role", "shared"),
            use_multilag_graph_propagation=getattr(
                self.config, "use_multilag_graph_propagation", False
            ),
            graph_propagation_lags=getattr(
                self.config, "graph_propagation_lags", (0, 1, 2, 4, 8)
            ),
            graph_propagation_lag_mode=getattr(
                self.config, "graph_propagation_lag_mode", "static"
            ),
            graph_propagation_alignment_temperature=getattr(
                self.config, "graph_propagation_alignment_temperature", 0.2
            ),
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
            use_causal_response_evidence=getattr(
                self.config, "use_causal_response_evidence", False
            ),
            use_strict_cross_mechanism=getattr(
                self.config, "use_strict_cross_mechanism", False
            ),
            strict_cross_lags=getattr(
                self.config, "strict_cross_lags", [1, 3, 6, 12]
            ),
            strict_cross_topk=getattr(self.config, "strict_cross_topk", 5),
            strict_cross_detach_backbone=getattr(
                self.config, "strict_cross_detach_backbone", True
            ),
            strict_cross_use_channel_prior=getattr(
                self.config, "strict_cross_use_channel_prior", False
            ),
            use_sps_role_head=getattr(self.config, "use_sps_role_head", False),
            sps_role_hidden=getattr(self.config, "sps_role_hidden", 16),
            sps_role_detach_features=getattr(
                self.config, "sps_role_detach_features", False
            ),
            use_bounded_sps_fusion=getattr(
                self.config, "use_bounded_sps_fusion", False
            ),
            bounded_sps_max_correction=getattr(
                self.config, "bounded_sps_max_correction", 0.25
            ),
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
            use_source_preserving_decoder=getattr(
                self.config, "use_source_preserving_decoder", False
            ),
            source_preserving_init=getattr(self.config, "source_preserving_init", 0.65),
            source_preserving_detach_gate=getattr(
                self.config, "source_preserving_detach_gate", True
            ),
            use_mechanism_residual_feedback=getattr(
                self.config, "use_mechanism_residual_feedback", False
            ),
            mechanism_feedback_init=getattr(self.config, "mechanism_feedback_init", 0.10),
            mechanism_feedback_detach=getattr(self.config, "mechanism_feedback_detach", True),
            mechanism_feedback_norm=getattr(self.config, "mechanism_feedback_norm", "sample_l1"),
            mechanism_feedback_clip=getattr(self.config, "mechanism_feedback_clip", 3.0),
            use_mechanism_predictive_head=getattr(self.config, "use_mechanism_predictive_head", False),
            mechanism_predictive_blend_init=getattr(self.config, "mechanism_predictive_blend_init", 0.30),
            use_source_gate=getattr(self.config, "use_source_gate", False),
            source_gate_init=getattr(self.config, "source_gate_init", 0.20),
            use_onset_aware_source_gate=getattr(
                self.config,
                "use_onset_aware_source_gate",
                False,
            ),
            source_gate_onset_window=getattr(
                self.config,
                "source_gate_onset_window",
                8,
            ),
            source_gate_onset_weight=getattr(
                self.config,
                "source_gate_onset_weight",
                0.5,
            ),
            use_dual_expert_fusion=getattr(
                self.config,
                "use_dual_expert_fusion",
                False,
            ),
            dual_expert_hidden=getattr(self.config, "dual_expert_hidden", 16),
            dual_expert_graph_init=getattr(
                self.config,
                "dual_expert_graph_init",
                0.5,
            ),
            dual_expert_detach_inputs=getattr(
                self.config,
                "dual_expert_detach_inputs",
                True,
            ),
            use_source_expert_branch=getattr(
                self.config,
                "use_source_expert_branch",
                False,
            ),
            source_expert_hidden=getattr(self.config, "source_expert_hidden", 16),
            source_expert_graph_init=getattr(
                self.config,
                "source_expert_graph_init",
                0.5,
            ),
            source_expert_detach_gate_inputs=getattr(
                self.config,
                "source_expert_detach_gate_inputs",
                True,
            ),
            source_expert_gradient_checkpoint=getattr(
                self.config,
                "source_expert_gradient_checkpoint",
                False,
            ),
            source_expert_gate_scope=getattr(
                self.config,
                "source_expert_gate_scope",
                "time",
            ),
            use_root_score_head=getattr(self.config, "use_root_score_head", False),
            root_score_head_mode=getattr(self.config, "root_score_head_mode", "mlp"),
            root_score_detach_features=getattr(self.config, "root_score_detach_features", True),
            root_response_penalty_init=getattr(self.config, "root_response_penalty_init", 1.0),
            root_response_confidence_discount=getattr(
                self.config,
                "root_response_confidence_discount",
                0.75,
            ),
            root_response_use_innovation_split=getattr(
                self.config,
                "root_response_use_innovation_split",
                False,
            ),
            use_pairwise_root_response_head=getattr(
                self.config,
                "use_pairwise_root_response_head",
                False,
            ),
            pairwise_root_response_rank=getattr(
                self.config,
                "pairwise_root_response_rank",
                8,
            ),
            pairwise_root_response_graph_weight=getattr(
                self.config,
                "pairwise_root_response_graph_weight",
                0.75,
            ),
            pairwise_root_response_reward_init=getattr(
                self.config,
                "pairwise_root_response_reward_init",
                0.25,
            ),
            pairwise_root_response_penalty_init=getattr(
                self.config,
                "pairwise_root_response_penalty_init",
                0.50,
            ),
            pairwise_root_response_logit_weight=getattr(
                self.config,
                "pairwise_root_response_logit_weight",
                1.0,
            ),
            use_event_responsibility_head=getattr(
                self.config,
                "use_event_responsibility_head",
                False,
            ),
            event_responsibility_hidden=getattr(
                self.config,
                "event_responsibility_hidden",
                16,
            ),
            event_responsibility_detach_features=getattr(
                self.config,
                "event_responsibility_detach_features",
                False,
            ),
            use_event_route_head=getattr(
                self.config,
                "use_event_route_head",
                False,
            ),
            event_route_hidden=getattr(self.config, "event_route_hidden", 16),
            event_route_init=getattr(self.config, "event_route_init", 0.55),
            event_route_detach_inputs=getattr(
                self.config,
                "event_route_detach_inputs",
                True,
            ),
            event_route_response_penalty=getattr(
                self.config,
                "event_route_response_penalty",
                0.0,
            ),
            use_response_suppressor_head=getattr(
                self.config,
                "use_response_suppressor_head",
                False,
            ),
            response_suppressor_hidden=getattr(
                self.config,
                "response_suppressor_hidden",
                16,
            ),
            response_suppressor_detach_inputs=getattr(
                self.config,
                "response_suppressor_detach_inputs",
                True,
            ),
            use_evidence_fusion_head=getattr(
                self.config,
                "use_evidence_fusion_head",
                False,
            ),
            evidence_fusion_hidden=getattr(
                self.config,
                "evidence_fusion_hidden",
                16,
            ),
            evidence_fusion_detach_inputs=getattr(
                self.config,
                "evidence_fusion_detach_inputs",
                False,
            ),
            evidence_fusion_use_graph_response=getattr(
                self.config,
                "evidence_fusion_use_graph_response",
                False,
            ),
            use_source_interaction_head=getattr(
                self.config,
                "use_source_interaction_head",
                False,
            ),
            source_interaction_detach_inputs=getattr(
                self.config,
                "source_interaction_detach_inputs",
                True,
            ),
            use_source_consistency_head=getattr(
                self.config,
                "use_source_consistency_head",
                False,
            ),
            source_consistency_detach_inputs=getattr(
                self.config,
                "source_consistency_detach_inputs",
                True,
            ),
            use_channel_temporal_corefinement=getattr(
                self.config, "use_channel_temporal_corefinement", False
            ),
            corefinement_init=getattr(self.config, "corefinement_init", 0.10),
            corefinement_detach_first_pass=getattr(
                self.config, "corefinement_detach_first_pass", True
            ),
            use_source_aware_corefinement=getattr(
                self.config, "use_source_aware_corefinement", False
            ),
            source_aware_corefinement_init=getattr(
                self.config, "source_aware_corefinement_init", 0.15
            ),
            source_aware_corefinement_detach_gate=getattr(
                self.config, "source_aware_corefinement_detach_gate", True
            ),
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
            self._begin_loss_debug_epoch(epoch + 1)

            self.model.train()
            self.optimizer.zero_grad()
            debug_train_max_batches = int(getattr(self.config, "debug_train_max_batches", 0) or 0)

            for i, (input_data, labels) in enumerate(self.train_loader):
                if debug_train_max_batches > 0 and i >= debug_train_max_batches:
                    break
                self._current_train_batch = i
                input_data = input_data.float().to(self.device)
                labels = labels.float().to(self.device)

                rec, _, _, _, _, aux_losses, _ = self.model(input_data)

                loss = self._reconstruction_loss(rec, input_data)
                self._record_loss_debug("reconstruction_loss", loss)
                loss = self._add_temporal_difference_loss(loss, rec, input_data)

                # ★ P0-1: lambda_causal_l1 → lambda_locality_l1
                if aux_losses and 'sparse_loss' in aux_losses:
                    loss = loss + self.config.lambda_locality_l1 * aux_losses['sparse_loss']
                loss = self._add_temporal_graph_regularization(loss, aux_losses)
                loss = loss + self._source_effect_synthetic_loss(
                    input_data,
                    batch_idx=i,
                    base_aux=aux_losses,
                )
                loss = loss + self._residual_sps_auxiliary_loss(
                    input_data,
                    aux_losses,
                    batch_idx=i,
                )
                loss = loss + self._channel_masked_modeling_loss(input_data, batch_idx=i)

                self._record_loss_debug("total_train_loss", loss)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_losses.append(loss.item())

            avg_train_loss = np.mean(epoch_losses) if epoch_losses else 0
            epoch_time = time.time() - epoch_start
            self._finish_loss_debug_epoch(epoch + 1)
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
        if self._should_load_best_checkpoint():
            self._get_raw_model().load_state_dict(self.early_stopping.check_point)
            print(
                f"  [Checkpoint] Using validation-selected checkpoint: "
                f"epoch {self.early_stopping.best_epoch}/"
                f"{getattr(self.early_stopping, 'last_epoch', self.config.num_epochs)} "
                f"(monitor={getattr(self.early_stopping, 'monitor_name', 'val_loss')})"
            )
        elif bool(getattr(self.config, "use_latest_checkpoint", False)):
            print("  [Checkpoint] Using latest epoch checkpoint for evaluation.")

        self._save_best_params(total_params=total_params, total_time=total_time)
        self._save_train_history(total_params=total_params, total_time=total_time)

        print("─" * 70)
        print(f"  [OK] Reweight Training Complete!")
        print(f"     Best Val Loss: {self.early_stopping.val_loss_min:.6f}")
        print(f"     Epochs Run:    {getattr(self.early_stopping, 'last_epoch', self.config.num_epochs)} / {self.config.num_epochs}")
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
                "checkpoint_policy": getattr(self.config, "checkpoint_policy", None),
                "use_rca_aware_checkpoint": getattr(self.config, "use_rca_aware_checkpoint", None),
                "rca_checkpoint_normalize": getattr(self.config, "rca_checkpoint_normalize", None),
                "rca_checkpoint_proxy_weight": getattr(
                    self.config, "rca_checkpoint_proxy_weight", None
                ),
                "rca_checkpoint_proxy_batches": getattr(
                    self.config, "rca_checkpoint_proxy_batches", None
                ),
                "rca_checkpoint_min_epoch": getattr(self.config, "rca_checkpoint_min_epoch", None),
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
                "channel_graph_evidence_only": getattr(
                    self.config, "channel_graph_evidence_only", None
                ),
                "channel_graph_type": getattr(self.config, "channel_graph_type", None),
                "directed_graph_parent_topk": getattr(
                    self.config, "directed_graph_parent_topk", None
                ),
                "directed_graph_embedding_dim": getattr(
                    self.config, "directed_graph_embedding_dim", None
                ),
                "directed_graph_use_signed_transfer": getattr(
                    self.config, "directed_graph_use_signed_transfer", None
                ),
                "directed_graph_transfer_mode": getattr(
                    self.config, "directed_graph_transfer_mode", None
                ),
                "directed_graph_transfer_rank": getattr(
                    self.config, "directed_graph_transfer_rank", None
                ),
                "directed_graph_transfer_scale": getattr(
                    self.config, "directed_graph_transfer_scale", None
                ),
                "use_channel_graph_reliability": getattr(
                    self.config, "use_channel_graph_reliability", None
                ),
                "channel_graph_role": getattr(self.config, "channel_graph_role", None),
                "use_multilag_graph_propagation": getattr(
                    self.config, "use_multilag_graph_propagation", None
                ),
                "graph_propagation_lags": getattr(
                    self.config, "graph_propagation_lags", None
                ),
                "graph_propagation_lag_mode": getattr(
                    self.config, "graph_propagation_lag_mode", None
                ),
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
                "use_root_score_head": getattr(self.config, "use_root_score_head", None),
                "root_score_head_mode": getattr(self.config, "root_score_head_mode", None),
                "root_score_detach_features": getattr(
                    self.config, "root_score_detach_features", None
                ),
                "root_response_penalty_init": getattr(
                    self.config, "root_response_penalty_init", None
                ),
                "root_response_confidence_discount": getattr(
                    self.config, "root_response_confidence_discount", None
                ),
                "root_response_use_innovation_split": getattr(
                    self.config, "root_response_use_innovation_split", None
                ),
                "use_pairwise_root_response_head": getattr(
                    self.config, "use_pairwise_root_response_head", None
                ),
                "pairwise_root_response_rank": getattr(
                    self.config, "pairwise_root_response_rank", None
                ),
                "pairwise_root_response_graph_weight": getattr(
                    self.config, "pairwise_root_response_graph_weight", None
                ),
                "pairwise_root_response_reward_init": getattr(
                    self.config, "pairwise_root_response_reward_init", None
                ),
                "pairwise_root_response_penalty_init": getattr(
                    self.config, "pairwise_root_response_penalty_init", None
                ),
                "pairwise_root_response_logit_weight": getattr(
                    self.config, "pairwise_root_response_logit_weight", None
                ),
                "lambda_pairwise_root_response": getattr(
                    self.config, "lambda_pairwise_root_response", None
                ),
                "lambda_source_effect_root_score": getattr(
                    self.config, "lambda_source_effect_root_score", None
                ),
                "root_score_bce_weight": getattr(self.config, "root_score_bce_weight", None),
                "root_score_rank_weight": getattr(self.config, "root_score_rank_weight", None),
                "root_score_effect_suppress_weight": getattr(
                    self.config, "root_score_effect_suppress_weight", None
                ),
                "root_response_source_branch_weight": getattr(
                    self.config, "root_response_source_branch_weight", None
                ),
                "root_response_response_branch_weight": getattr(
                    self.config, "root_response_response_branch_weight", None
                ),
                "root_response_response_suppress_weight": getattr(
                    self.config, "root_response_response_suppress_weight", None
                ),
                "use_event_responsibility_head": getattr(
                    self.config, "use_event_responsibility_head", None
                ),
                "lambda_event_responsibility": getattr(
                    self.config, "lambda_event_responsibility", None
                ),
                "event_responsibility_loss_mode": getattr(
                    self.config, "event_responsibility_loss_mode", None
                ),
                "event_responsibility_bce_balance": getattr(
                    self.config, "event_responsibility_bce_balance", None
                ),
                "event_responsibility_ce_weight": getattr(
                    self.config, "event_responsibility_ce_weight", None
                ),
                "event_responsibility_rank_weight": getattr(
                    self.config, "event_responsibility_rank_weight", None
                ),
                "event_responsibility_effect_suppress_weight": getattr(
                    self.config, "event_responsibility_effect_suppress_weight", None
                ),
                "rca_event_responsibility_weight": getattr(
                    self.config, "rca_event_responsibility_weight", None
                ),
                "rca_event_responsibility_pooling": getattr(
                    self.config, "rca_event_responsibility_pooling", None
                ),
                "rca_event_responsibility_head_ratio": getattr(
                    self.config, "rca_event_responsibility_head_ratio", None
                ),
                "rca_event_responsibility_head_points": getattr(
                    self.config, "rca_event_responsibility_head_points", None
                ),
                "rca_event_responsibility_top_quantile": getattr(
                    self.config, "rca_event_responsibility_top_quantile", None
                ),
                "rca_root_score_weight": getattr(self.config, "rca_root_score_weight", None),
                "rca_root_score_signal": getattr(self.config, "rca_root_score_signal", None),
                "rca_root_score_pooling": getattr(self.config, "rca_root_score_pooling", None),
                "rca_root_score_head_ratio": getattr(self.config, "rca_root_score_head_ratio", None),
                "rca_root_score_head_points": getattr(self.config, "rca_root_score_head_points", None),
                "rca_root_score_top_quantile": getattr(
                    self.config, "rca_root_score_top_quantile", None
                ),
                "rca_source_innovation_weight": getattr(self.config, "rca_source_innovation_weight", None),
                "rca_source_innovation_mode": getattr(self.config, "rca_source_innovation_mode", None),
                "rca_source_innovation_neighbor_weight": getattr(
                    self.config, "rca_source_innovation_neighbor_weight", None
                ),
                "rca_source_innovation_lead_points": getattr(
                    self.config, "rca_source_innovation_lead_points", None
                ),
                "rca_topk_rerank": getattr(self.config, "rca_topk_rerank", None),
                "rca_topk_rerank_k": getattr(self.config, "rca_topk_rerank_k", None),
                "rca_topk_rerank_original_weight": getattr(
                    self.config, "rca_topk_rerank_original_weight", None
                ),
                "rca_topk_rerank_base_weight": getattr(
                    self.config, "rca_topk_rerank_base_weight", None
                ),
                "rca_topk_rerank_group_weight": getattr(
                    self.config, "rca_topk_rerank_group_weight", None
                ),
                "rca_topk_rerank_onset_weight": getattr(
                    self.config, "rca_topk_rerank_onset_weight", None
                ),
                "rca_topk_rerank_source_gate_weight": getattr(
                    self.config, "rca_topk_rerank_source_gate_weight", None
                ),
                "rca_topk_rerank_mechanism_residual_weight": getattr(
                    self.config, "rca_topk_rerank_mechanism_residual_weight", None
                ),
                "rca_topk_rerank_source_interaction_weight": getattr(
                    self.config, "rca_topk_rerank_source_interaction_weight", None
                ),
                "rca_topk_rerank_source_consistency_head_weight": getattr(
                    self.config,
                    "rca_topk_rerank_source_consistency_head_weight",
                    None,
                ),
                "rca_topk_rerank_graph_penalty_weight": getattr(
                    self.config, "rca_topk_rerank_graph_penalty_weight", None
                ),
                "rca_topk_rerank_component_scope": getattr(
                    self.config, "rca_topk_rerank_component_scope", None
                ),
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
            "best_monitor_value": float(
                getattr(self.early_stopping, "best_monitor_value", np.nan)
            ),
            "monitor_name": getattr(self.early_stopping, "monitor_name", "val_loss"),
            "best_epoch": self.early_stopping.best_epoch,
            "selected_checkpoint_epoch": self.early_stopping.best_epoch,
            "total_epochs_run": int(getattr(self.early_stopping, "last_epoch", 0)),
            "max_epochs": int(getattr(self.config, "num_epochs", 0)),
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
        x_rec, A_adaptive, _, _, _, aux_losses, _ = raw_model(
            input_data,
            return_root_score=(
                bool(getattr(self.config, "use_root_score_head", False))
                or bool(getattr(self.config, "use_evidence_fusion_head", False))
                or bool(getattr(self.config, "use_response_suppressor_head", False))
                or bool(getattr(self.config, "use_source_interaction_head", False))
                or bool(getattr(self.config, "use_source_consistency_head", False))
                or bool(getattr(self.config, "use_event_route_head", False))
            ),
        )
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
        root_score_err = None
        if aux_losses:
            root_score_signal = str(
                getattr(self.config, "rca_root_score_signal", "prob") or "prob"
            ).lower()
            if root_score_signal in {"source_confidence", "confidence", "source_prob"}:
                root_score_err = aux_losses.get("root_response_source_confidence")
            elif root_score_signal in {"source_evidence", "source"}:
                source_evidence = aux_losses.get("root_response_source_evidence")
                if source_evidence is not None:
                    root_score_err = torch.sigmoid(source_evidence)
            elif root_score_signal in {"logit", "logits", "raw_logit"}:
                root_score_err = aux_losses.get("root_score_logits")
            if root_score_err is None:
                root_score_err = aux_losses.get("root_score_prob")
        if root_score_err is None:
            root_score_err = torch.zeros_like(channel_err)
        event_responsibility_err = None
        if aux_losses:
            event_responsibility_logits = aux_losses.get("event_responsibility_logits")
            if event_responsibility_logits is not None:
                event_responsibility_err = torch.softmax(event_responsibility_logits, dim=-1)
        if event_responsibility_err is None:
            event_responsibility_err = torch.zeros_like(channel_err)
        event_route_err = None
        if aux_losses:
            event_route_score = aux_losses.get("event_route_score")
            if event_route_score is not None:
                event_route_err = event_route_score
        if event_route_err is None:
            event_route_err = torch.zeros_like(channel_err)
        response_suppressor_err = None
        if aux_losses:
            response_suppressor_logits = aux_losses.get("response_suppressor_logits")
            if response_suppressor_logits is not None:
                response_suppressor_err = torch.sigmoid(response_suppressor_logits)
        if response_suppressor_err is None:
            response_suppressor_err = torch.zeros_like(channel_err)
        evidence_fusion_err = None
        if aux_losses:
            evidence_fusion_logits = aux_losses.get("evidence_fusion_logits")
            if evidence_fusion_logits is not None:
                evidence_fusion_export_mode = str(
                    getattr(self.config, "evidence_fusion_export_mode", "sigmoid")
                    or "sigmoid"
                ).lower()
                if evidence_fusion_export_mode in {
                    "softmax",
                    "responsibility",
                    "competition",
                }:
                    evidence_fusion_err = torch.softmax(evidence_fusion_logits, dim=-1)
                else:
                    evidence_fusion_err = torch.sigmoid(evidence_fusion_logits)
        if evidence_fusion_err is None:
            evidence_fusion_err = torch.zeros_like(channel_err)
        source_interaction_head_err = None
        if aux_losses:
            source_interaction_logits = aux_losses.get("source_interaction_logits")
            if source_interaction_logits is not None:
                source_interaction_head_err = torch.sigmoid(source_interaction_logits)
        if source_interaction_head_err is None:
            source_interaction_head_err = torch.zeros_like(channel_err)
        source_consistency_head_err = None
        if aux_losses:
            source_consistency_logits = aux_losses.get("source_consistency_logits")
            if source_consistency_logits is not None:
                source_consistency_head_err = torch.sigmoid(source_consistency_logits)
        if source_consistency_head_err is None:
            source_consistency_head_err = torch.zeros_like(channel_err)
        return (
            score.cpu().numpy(),
            channel_err.cpu().numpy(),
            graph_err.cpu().numpy(),
            mechanism_err.cpu().numpy(),
            causal_err.cpu().numpy(),
            source_gate_err.cpu().numpy(),
            synthetic_rca_window_scores.cpu().numpy(),
            root_score_err.cpu().numpy(),
            event_responsibility_err.cpu().numpy(),
            event_route_err.cpu().numpy(),
            response_suppressor_err.cpu().numpy(),
            evidence_fusion_err.cpu().numpy(),
            source_interaction_head_err.cpu().numpy(),
            source_consistency_head_err.cpu().numpy(),
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
        root_score_channel_sums = np.zeros_like(channel_sums)
        event_responsibility_channel_sums = np.zeros_like(channel_sums)
        event_route_channel_sums = np.zeros_like(channel_sums)
        response_suppressor_channel_sums = np.zeros_like(channel_sums)
        evidence_fusion_channel_sums = np.zeros_like(channel_sums)
        source_interaction_head_channel_sums = np.zeros_like(channel_sums)
        source_consistency_head_channel_sums = np.zeros_like(channel_sums)
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
                    root_score_channel_err,
                    event_responsibility_channel_err,
                    event_route_channel_err,
                    response_suppressor_channel_err,
                    evidence_fusion_channel_err,
                    source_interaction_head_channel_err,
                    source_consistency_head_channel_err,
                ) = self._detect_forward_with_channels(input_data)
                self._add_window_channel_score_group(
                    [
                        channel_sums,
                        graph_channel_sums,
                        mechanism_channel_sums,
                        causal_channel_sums,
                        source_gate_channel_sums,
                        root_score_channel_sums,
                        event_responsibility_channel_sums,
                        event_route_channel_sums,
                        response_suppressor_channel_sums,
                        evidence_fusion_channel_sums,
                        source_interaction_head_channel_sums,
                        source_consistency_head_channel_sums,
                    ],
                    channel_counts,
                    [
                        channel_err,
                        graph_channel_err,
                        mechanism_channel_err,
                        causal_channel_err,
                        source_gate_channel_err,
                        root_score_channel_err,
                        event_responsibility_channel_err,
                        event_route_channel_err,
                        response_suppressor_channel_err,
                        evidence_fusion_channel_err,
                        source_interaction_head_channel_err,
                        source_consistency_head_channel_err,
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
        self._last_root_score_channel_scores = np.divide(
            root_score_channel_sums,
            channel_counts[:, None],
            out=np.zeros_like(root_score_channel_sums),
            where=channel_counts[:, None] > 0,
        ).astype(np.float32)
        self._last_event_responsibility_channel_scores = np.divide(
            event_responsibility_channel_sums,
            channel_counts[:, None],
            out=np.zeros_like(event_responsibility_channel_sums),
            where=channel_counts[:, None] > 0,
        ).astype(np.float32)
        self._last_event_route_channel_scores = np.divide(
            event_route_channel_sums,
            channel_counts[:, None],
            out=np.zeros_like(event_route_channel_sums),
            where=channel_counts[:, None] > 0,
        ).astype(np.float32)
        self._last_response_suppressor_channel_scores = np.divide(
            response_suppressor_channel_sums,
            channel_counts[:, None],
            out=np.zeros_like(response_suppressor_channel_sums),
            where=channel_counts[:, None] > 0,
        ).astype(np.float32)
        self._last_evidence_fusion_channel_scores = np.divide(
            evidence_fusion_channel_sums,
            channel_counts[:, None],
            out=np.zeros_like(evidence_fusion_channel_sums),
            where=channel_counts[:, None] > 0,
        ).astype(np.float32)
        self._last_source_interaction_head_channel_scores = np.divide(
            source_interaction_head_channel_sums,
            channel_counts[:, None],
            out=np.zeros_like(source_interaction_head_channel_sums),
            where=channel_counts[:, None] > 0,
        ).astype(np.float32)
        self._last_source_consistency_head_channel_scores = np.divide(
            source_consistency_head_channel_sums,
            channel_counts[:, None],
            out=np.zeros_like(source_consistency_head_channel_sums),
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

    def _inference_dataloader_num_workers(self) -> int:
        return max(0, int(getattr(self.config, "inference_dataloader_num_workers", 0) or 0))

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
        aggregation_method = str(getattr(self.config, "score_aggregation", "mean") or "mean").lower()
        stream_mean = aggregation_method == "mean"
        window_scores_list = [] if not stream_mean else None
        if stream_mean:
            point_scores_sum = np.zeros(total_length, dtype=np.float64)
            point_scores_count = np.zeros(total_length, dtype=np.float64)
            next_window_start = 0

        loader = anomaly_detection_data_provider(
            scaled_data, batch_size=eval_batch_size,
            win_size=self.config.win_size, step=1, mode="test",
            num_workers=self._inference_dataloader_num_workers(),
            prefetch_factor=getattr(self.config, "dataloader_prefetch_factor", 2),
        )

        for i, (input_data, labels) in enumerate(loader):
            input_data = input_data.float().to(self.device)
            scores_batch = self._detect_forward(input_data)
            if stream_mean:
                scores_batch = np.asarray(scores_batch, dtype=np.float32)
                n_windows = scores_batch.shape[0]
                win_size = min(int(self.config.win_size), scores_batch.shape[1])
                for offset in range(win_size):
                    start = next_window_start + offset
                    n = min(n_windows, total_length - start)
                    if n <= 0:
                        break
                    point_scores_sum[start:start + n] += scores_batch[:n, offset]
                    point_scores_count[start:start + n] += 1.0
                next_window_start += n_windows
            else:
                window_scores_list.append(scores_batch)
            if (i + 1) % 5 == 0:
                torch.cuda.empty_cache()

        if stream_mean:
            point_scores = np.divide(
                point_scores_sum,
                point_scores_count,
                out=np.zeros_like(point_scores_sum),
                where=point_scores_count > 0,
            ).astype(np.float32)
            point_scores = self._apply_event_persistence_score(point_scores)
            point_scores = self._smooth_scores_for_detection(point_scores)
            return point_scores, point_scores

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
            num_workers=self._inference_dataloader_num_workers(),
            prefetch_factor=getattr(self.config, "dataloader_prefetch_factor", 2),
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
        root_score_channel_sums = None
        event_responsibility_channel_sums = None
        event_route_channel_sums = None
        response_suppressor_channel_sums = None
        evidence_fusion_channel_sums = None
        source_interaction_head_channel_sums = None
        source_consistency_head_channel_sums = None
        synthetic_rca_channel_diff = None
        channel_counts = None
        window_cursor = 0
        if dense_rca_export:
            channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            graph_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            mechanism_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            causal_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            source_gate_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            root_score_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            event_responsibility_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            event_route_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            response_suppressor_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            evidence_fusion_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            source_interaction_head_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
            source_consistency_head_channel_sums = np.zeros((total_length, scaled_test.shape[1]), dtype=np.float64)
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
                    root_score_channel_err,
                    event_responsibility_channel_err,
                    event_route_channel_err,
                    response_suppressor_channel_err,
                    evidence_fusion_channel_err,
                    source_interaction_head_channel_err,
                    source_consistency_head_channel_err,
                ) = self._detect_forward_with_channels(input_data)
                self._add_window_channel_score_group(
                    [
                        channel_sums,
                        graph_channel_sums,
                        mechanism_channel_sums,
                        causal_channel_sums,
                        source_gate_channel_sums,
                        root_score_channel_sums,
                        event_responsibility_channel_sums,
                        event_route_channel_sums,
                        response_suppressor_channel_sums,
                        evidence_fusion_channel_sums,
                        source_interaction_head_channel_sums,
                        source_consistency_head_channel_sums,
                    ],
                    channel_counts,
                    [
                        channel_err,
                        graph_channel_err,
                        mechanism_channel_err,
                        causal_channel_err,
                        source_gate_channel_err,
                        root_score_channel_err,
                        event_responsibility_channel_err,
                        event_route_channel_err,
                        response_suppressor_channel_err,
                        evidence_fusion_channel_err,
                        source_interaction_head_channel_err,
                        source_consistency_head_channel_err,
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
            self._last_root_score_channel_scores = np.divide(
                root_score_channel_sums,
                channel_counts[:, None],
                out=np.zeros_like(root_score_channel_sums),
                where=channel_counts[:, None] > 0,
            ).astype(np.float32)
            self._last_event_responsibility_channel_scores = np.divide(
                event_responsibility_channel_sums,
                channel_counts[:, None],
                out=np.zeros_like(event_responsibility_channel_sums),
                where=channel_counts[:, None] > 0,
            ).astype(np.float32)
            self._last_event_route_channel_scores = np.divide(
                event_route_channel_sums,
                channel_counts[:, None],
                out=np.zeros_like(event_route_channel_sums),
                where=channel_counts[:, None] > 0,
            ).astype(np.float32)
            self._last_response_suppressor_channel_scores = np.divide(
                response_suppressor_channel_sums,
                channel_counts[:, None],
                out=np.zeros_like(response_suppressor_channel_sums),
                where=channel_counts[:, None] > 0,
            ).astype(np.float32)
            self._last_evidence_fusion_channel_scores = np.divide(
                evidence_fusion_channel_sums,
                channel_counts[:, None],
                out=np.zeros_like(evidence_fusion_channel_sums),
                where=channel_counts[:, None] > 0,
            ).astype(np.float32)
            self._last_source_interaction_head_channel_scores = np.divide(
                source_interaction_head_channel_sums,
                channel_counts[:, None],
                out=np.zeros_like(source_interaction_head_channel_sums),
                where=channel_counts[:, None] > 0,
            ).astype(np.float32)
            self._last_source_consistency_head_channel_scores = np.divide(
                source_consistency_head_channel_sums,
                channel_counts[:, None],
                out=np.zeros_like(source_consistency_head_channel_sums),
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
            self._last_root_score_channel_scores = None
            self._last_event_responsibility_channel_scores = None
            self._last_event_route_channel_scores = None
            self._last_response_suppressor_channel_scores = None
            self._last_evidence_fusion_channel_scores = None
            self._last_source_interaction_head_channel_scores = None
            self._last_source_consistency_head_channel_scores = None
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
    def _event_onset_profile(score_matrix, start, end, baseline_window=200, onset_z=2.0):
        """Return onset strength, first crossing index, and crossing mask for one event."""
        if score_matrix is None or end <= start or start <= 0:
            width = score_matrix.shape[1] if score_matrix is not None and score_matrix.ndim == 2 else 0
            return (
                np.zeros(width, dtype=np.float32),
                np.full(width, np.inf, dtype=np.float32),
                np.zeros(width, dtype=bool),
            )

        event_scores = score_matrix[start:end]
        if event_scores.size == 0:
            width = score_matrix.shape[1]
            return (
                np.zeros(width, dtype=np.float32),
                np.full(width, np.inf, dtype=np.float32),
                np.zeros(width, dtype=bool),
            )

        baseline_start = max(0, start - max(1, int(baseline_window)))
        baseline_scores = score_matrix[baseline_start:start]
        if baseline_scores.size == 0:
            width = score_matrix.shape[1]
            return (
                np.zeros(width, dtype=np.float32),
                np.full(width, np.inf, dtype=np.float32),
                np.zeros(width, dtype=bool),
            )

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
        onset_scores = np.nan_to_num(onset_scores, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        first_idx = np.where(has_onset, first_idx.astype(np.float32), np.inf).astype(np.float32)
        return onset_scores, first_idx, has_onset

    @staticmethod
    def _event_onset_scores(score_matrix, start, end, baseline_window=200, onset_z=2.0):
        """Score variables that cross their normal local baseline earlier within an event."""
        onset_scores, _, _ = LaGraph._event_onset_profile(
            score_matrix,
            start,
            end,
            baseline_window=baseline_window,
            onset_z=onset_z,
        )
        return onset_scores

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

    @staticmethod
    def _event_score_distribution(
        scores,
        order,
        mode="raw",
        top_m=0,
        power=2.0,
        tau=1.0,
    ):
        """Convert event ranking logits to a sparse responsibility distribution."""
        arr = np.asarray(scores, dtype=np.float64)
        if arr.size == 0:
            return arr
        mode = str(mode or "raw").lower()
        if mode in {"raw", "none", "linear"}:
            return arr.copy()
        ranking = np.asarray(order, dtype=np.int64)
        if ranking.size == 0:
            ranking = np.argsort(-arr)
        keep_count = int(top_m or 0)
        if keep_count <= 0:
            keep = ranking
        else:
            keep = ranking[: min(keep_count, ranking.size)]
        out = np.zeros_like(arr, dtype=np.float64)
        if keep.size == 0:
            return out
        kept_scores = arr[keep]
        if mode == "softmax":
            tau = max(float(tau or 1.0), 1e-6)
            shifted = kept_scores / tau
            shifted = shifted - float(np.max(shifted))
            weights = np.exp(shifted)
        elif mode in {"positive_l1", "responsibility", "sparse_l1"}:
            min_score = float(np.min(kept_scores))
            power = max(float(power or 1.0), 1e-6)
            weights = np.maximum(kept_scores - min_score, 0.0) ** power
        else:
            return arr.copy()
        total = float(np.sum(weights))
        if total <= 1e-12:
            out[keep] = 1.0 / float(keep.size)
        else:
            out[keep] = weights / total
        return out

    @staticmethod
    def _source_innovation_scores(base_scores, prior, neighbor_weight=1.0):
        """Residual unexplained by normal-prior neighbors; high values are source-like."""
        scores = np.asarray(base_scores, dtype=np.float32)
        if scores.ndim != 2 or scores.size == 0:
            return np.zeros_like(scores, dtype=np.float32)
        prior_arr = np.array(prior, dtype=np.float32, copy=True) if prior is not None else None
        if prior_arr is None or prior_arr.shape != (scores.shape[1], scores.shape[1]):
            return np.zeros_like(scores, dtype=np.float32)
        prior_arr = np.nan_to_num(prior_arr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(prior_arr, 0.0)
        row_sum = prior_arr.sum(axis=1, keepdims=True)
        prior_arr = np.divide(
            prior_arr,
            row_sum,
            out=np.zeros_like(prior_arr, dtype=np.float32),
            where=row_sum > 1e-8,
        )
        support = scores @ prior_arr.T
        innovation = scores - max(float(neighbor_weight), 0.0) * support
        return np.nan_to_num(
            np.maximum(innovation, 0.0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

    @staticmethod
    def _event_directional_source_innovation_scores(
        score_matrix,
        prior,
        start,
        end,
        baseline_window=200,
        onset_z=2.0,
        neighbor_weight=1.0,
        lead_points=1,
    ):
        """Onset residual unexplained by normal-prior neighbors that become abnormal earlier."""
        if score_matrix is None or prior is None:
            width = score_matrix.shape[1] if score_matrix is not None and getattr(score_matrix, "ndim", 0) == 2 else 0
            return np.zeros(width, dtype=np.float32)
        scores = np.asarray(score_matrix, dtype=np.float32)
        if scores.ndim != 2 or scores.size == 0:
            return np.zeros(scores.shape[1] if scores.ndim == 2 else 0, dtype=np.float32)
        prior_arr = np.array(prior, dtype=np.float32, copy=True)
        if prior_arr.shape != (scores.shape[1], scores.shape[1]):
            return np.zeros(scores.shape[1], dtype=np.float32)

        onset_scores, first_idx, has_onset = LaGraph._event_onset_profile(
            scores,
            start,
            end,
            baseline_window=baseline_window,
            onset_z=onset_z,
        )
        if onset_scores.size == 0:
            return onset_scores

        prior_arr = np.nan_to_num(prior_arr, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(prior_arr, 0.0)
        row_sum = prior_arr.sum(axis=1, keepdims=True)
        prior_arr = np.divide(
            prior_arr,
            row_sum,
            out=np.zeros_like(prior_arr, dtype=np.float32),
            where=row_sum > 1e-8,
        )

        lead = max(0, int(lead_points))
        earlier = (first_idx[None, :] + float(lead)) <= first_idx[:, None]
        earlier = earlier & has_onset[None, :] & has_onset[:, None]
        directional_prior = prior_arr * earlier.astype(np.float32)
        support_sum = directional_prior.sum(axis=1, keepdims=True)
        directional_prior = np.divide(
            directional_prior,
            support_sum,
            out=np.zeros_like(directional_prior, dtype=np.float32),
            where=support_sum > 1e-8,
        )
        predecessor_support = directional_prior @ onset_scores
        innovation = onset_scores - max(float(neighbor_weight), 0.0) * predecessor_support
        return np.nan_to_num(
            np.maximum(innovation, 0.0),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

    def _apply_topk_root_rerank(
        self,
        order,
        event_scores,
        feature_names,
        event_base_scores,
        group_scores,
        event_onset_scores,
        event_source_gate_scores,
        event_mechanism_residual_scores,
        event_source_consistency_head_scores,
        event_graph_scores,
    ):
        """Conservatively reorder only the current Top-K RCA candidates."""
        if not bool(getattr(self.config, "rca_topk_rerank", False)):
            return order, np.asarray(event_scores, dtype=np.float64), {}

        order_list = [int(idx) for idx in np.asarray(order).tolist()]
        top_k = int(getattr(self.config, "rca_topk_rerank_k", 5) or 5)
        top_k = min(max(2, top_k), len(order_list))
        if top_k < 2:
            return order, np.asarray(event_scores, dtype=np.float64), {}

        candidate_indices = order_list[:top_k]

        event_scores = np.asarray(event_scores, dtype=np.float64)
        event_base_scores = np.asarray(event_base_scores, dtype=np.float64)
        event_onset_scores = np.asarray(event_onset_scores, dtype=np.float64)
        event_source_gate_scores = np.asarray(event_source_gate_scores, dtype=np.float64)
        event_mechanism_residual_scores = np.asarray(
            event_mechanism_residual_scores,
            dtype=np.float64,
        )
        event_source_consistency_head_scores = np.asarray(
            event_source_consistency_head_scores,
            dtype=np.float64,
        )
        event_graph_scores = np.asarray(event_graph_scores, dtype=np.float64)
        source_interaction_scores = event_base_scores * np.maximum(
            event_onset_scores,
            event_mechanism_residual_scores,
        )
        all_group_values = np.asarray(
            [
                float(group_scores.get(self._root_cause_group_name(name), 0.0))
                for name in feature_names
            ],
            dtype=np.float64,
        )
        component_scope = str(
            getattr(self.config, "rca_topk_rerank_component_scope", "candidate") or "candidate"
        ).lower()
        use_event_scope = component_scope in {"event", "global", "full"}

        def _candidate_norm(values):
            arr = np.asarray([values[idx] for idx in candidate_indices], dtype=np.float64)
            return self._normalize_event_component(arr)

        def _event_norm(values):
            arr = self._normalize_event_component(np.asarray(values, dtype=np.float64))
            return np.asarray([arr[idx] for idx in candidate_indices], dtype=np.float64)

        normalizer = _event_norm if use_event_scope else _candidate_norm
        group_norm = normalizer(all_group_values)
        original_norm = normalizer(event_scores)
        base_norm = normalizer(event_base_scores)
        onset_norm = normalizer(event_onset_scores)
        source_gate_norm = normalizer(event_source_gate_scores)
        mechanism_residual_norm = normalizer(event_mechanism_residual_scores)
        source_interaction_norm = normalizer(source_interaction_scores)
        source_consistency_head_norm = normalizer(event_source_consistency_head_scores)
        graph_norm = normalizer(event_graph_scores)

        def _weighted_signal(prefix):
            original_weight = float(
                getattr(self.config, f"{prefix}_original_weight", 1.0) or 0.0
            )
            base_weight = float(getattr(self.config, f"{prefix}_base_weight", 0.0) or 0.0)
            group_weight = float(getattr(self.config, f"{prefix}_group_weight", 0.0) or 0.0)
            onset_weight = float(getattr(self.config, f"{prefix}_onset_weight", 0.0) or 0.0)
            source_gate_weight = float(
                getattr(self.config, f"{prefix}_source_gate_weight", 0.0) or 0.0
            )
            mechanism_residual_weight = float(
                getattr(self.config, f"{prefix}_mechanism_residual_weight", 0.0) or 0.0
            )
            source_interaction_weight = float(
                getattr(self.config, f"{prefix}_source_interaction_weight", 0.0) or 0.0
            )
            source_consistency_head_weight = float(
                getattr(self.config, f"{prefix}_source_consistency_head_weight", 0.0)
                or 0.0
            )
            graph_penalty_weight = float(
                getattr(self.config, f"{prefix}_graph_penalty_weight", 0.0) or 0.0
            )
            return (
                original_weight * original_norm
                + base_weight * base_norm
                + group_weight * group_norm
                + onset_weight * onset_norm
                + source_gate_weight * source_gate_norm
                + mechanism_residual_weight * mechanism_residual_norm
                + source_interaction_weight * source_interaction_norm
                + source_consistency_head_weight * source_consistency_head_norm
                - graph_penalty_weight * graph_norm
            )

        rerank_signal = _weighted_signal("rca_topk_rerank")
        primary_pairs = sorted(
            zip(candidate_indices, rerank_signal.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
        primary_top = [idx for idx, _ in primary_pairs]

        keep_primary_top_k = int(
            getattr(self.config, "rca_topk_rerank_keep_primary_top_k", 0) or 0
        )
        fill_secondary_top_k = int(
            getattr(self.config, "rca_topk_rerank_fill_secondary_top_k", 0) or 0
        )
        secondary_scores = None
        secondary_rank = {}
        if keep_primary_top_k > 0 and fill_secondary_top_k > keep_primary_top_k:
            secondary_signal = _weighted_signal("rca_topk_rerank_secondary")
            secondary_pairs = sorted(
                zip(candidate_indices, secondary_signal.tolist()),
                key=lambda item: item[1],
                reverse=True,
            )
            secondary_scores = {
                int(idx): float(score)
                for idx, score in secondary_pairs
            }
            secondary_rank = {
                int(idx): int(rank)
                for rank, (idx, _) in enumerate(secondary_pairs, start=1)
            }
            keep_primary_top_k = min(max(0, keep_primary_top_k), top_k)
            fill_secondary_top_k = min(max(keep_primary_top_k, fill_secondary_top_k), top_k)
            reranked_top = []
            for idx in primary_top[:keep_primary_top_k]:
                if idx not in reranked_top:
                    reranked_top.append(idx)
            for idx, _ in secondary_pairs:
                if len(reranked_top) >= fill_secondary_top_k:
                    break
                if idx not in reranked_top:
                    reranked_top.append(idx)
            for idx in primary_top:
                if idx not in reranked_top:
                    reranked_top.append(idx)
        else:
            reranked_top = primary_top

        adjusted_scores = event_scores.copy()
        original_top_scores = sorted(
            [float(event_scores[idx]) for idx in candidate_indices],
            reverse=True,
        )
        for idx, score in zip(reranked_top, original_top_scores):
            adjusted_scores[idx] = score

        rerank_info = {
            int(idx): {
                "topk_rerank_score": float(score),
                "topk_rerank_primary_score": float(score),
                "topk_rerank_secondary_score": float(
                    secondary_scores.get(int(idx), 0.0) if secondary_scores is not None else 0.0
                ),
                "topk_rerank_original_rank": int(candidate_indices.index(idx) + 1),
                "topk_rerank_secondary_rank": int(secondary_rank.get(int(idx), 0)),
            }
            for idx, score in primary_pairs
        }
        final_order = np.asarray(reranked_top + order_list[top_k:], dtype=np.int64)
        return final_order, adjusted_scores, rerank_info

    def _aggregate_root_score_event_component(
        self,
        root_score_channel_scores,
        score_start,
        score_end,
        root_score_pooling="mean",
        head_ratio=0.30,
        head_points=30,
        top_quantile=0.80,
    ):
        scores = np.asarray(root_score_channel_scores[score_start:score_end], dtype=np.float64)
        if scores.size == 0:
            fallback = np.asarray(root_score_channel_scores, dtype=np.float64)
            width = fallback.shape[1] if fallback.ndim == 2 else 0
            return np.zeros(width, dtype=np.float64)
        mode = str(root_score_pooling or "mean").lower()
        if mode in {"early", "early_mean", "head", "head_mean"}:
            length = scores.shape[0]
            ratio = min(max(float(head_ratio or 1.0), 1e-6), 1.0)
            take = int(np.ceil(length * ratio))
            fixed_points = int(head_points or 0)
            if fixed_points > 0:
                take = min(take, fixed_points)
            else:
                take = min(length, take)
            take = max(1, take)
            return scores[:take].mean(axis=0)
        if mode in {"top_quantile", "topq", "quantile_mean"}:
            q = min(max(float(top_quantile or 0.80), 0.0), 0.999)
            thresholds = np.quantile(scores, q, axis=0, keepdims=True)
            mask = scores >= thresholds
            counts = mask.sum(axis=0).clip(min=1)
            return (scores * mask).sum(axis=0) / counts
        if mode in {"max", "peak"}:
            return scores.max(axis=0)
        return scores.mean(axis=0)

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
        root_score_channel_scores=None,
        root_score_pooling="mean",
        root_score_head_ratio=0.30,
        root_score_head_points=30,
        root_score_top_quantile=0.80,
        source_innovation_channel_scores=None,
        synthetic_channel_scores=None,
        event_responsibility_channel_scores=None,
        event_responsibility_pooling="mean",
        event_responsibility_head_ratio=0.30,
        event_responsibility_head_points=30,
        event_responsibility_top_quantile=0.80,
        event_route_channel_scores=None,
        response_suppressor_channel_scores=None,
        evidence_fusion_channel_scores=None,
        evidence_fusion_pooling="mean",
        evidence_fusion_head_ratio=0.30,
        evidence_fusion_head_points=30,
        evidence_fusion_top_quantile=0.80,
        source_interaction_head_channel_scores=None,
        source_interaction_head_pooling="head_mean",
        source_interaction_head_ratio=0.30,
        source_interaction_head_points=30,
        source_interaction_head_top_quantile=0.80,
        source_consistency_head_channel_scores=None,
        source_consistency_head_pooling="head_mean",
        source_consistency_head_ratio=0.30,
        source_consistency_head_points=30,
        source_consistency_head_top_quantile=0.80,
        counterfactual_channel_scores=None,
        use_source_propagation=False,
        graph_weight=0.0,
        mechanism_weight=0.0,
        source_weight=0.75,
        source_base_weight=1.0,
        propagation_weight=0.25,
        source_mechanism_weight=0.0,
        source_score_weight=0.0,
        source_consensus_weight=0.0,
        onset_consensus_weight=0.0,
        source_consensus_mode="sqrt_bg",
        causal_weight=0.0,
        synthetic_weight=0.0,
        event_responsibility_weight=0.0,
        event_route_weight=0.0,
        response_suppressor_weight=0.0,
        response_suppressor_source_guard_mode="source_onset",
        response_suppressor_source_guard_floor=0.0,
        evidence_fusion_weight=0.0,
        source_interaction_head_weight=0.0,
        source_consistency_head_weight=0.0,
        source_gate_weight=0.0,
        source_gate_evidence_guard_weight=0.0,
        source_gate_evidence_guard_floor=0.0,
        root_score_weight=0.0,
        source_interaction_weight=0.0,
        source_innovation_weight=0.0,
        source_innovation_mode="series",
        source_innovation_neighbor_weight=1.0,
        source_innovation_lead_points=1,
        mechanism_guided_source_weight=0.0,
        adaptive_mechanism_gate_weight=0.0,
        adaptive_mechanism_gate_floor=0.05,
        adaptive_mechanism_gate_mode="source_onset",
        adaptive_mechanism_selection_weight=0.0,
        adaptive_mechanism_selection_floor=0.05,
        adaptive_mechanism_selection_mode="source_onset",
        conservative_mechanism_weight=0.0,
        conservative_mechanism_support_floor=0.20,
        conservative_mechanism_candidate_topk=0,
        conservative_mechanism_support_mode="source_onset",
        source_gated_mechanism_weight=0.0,
        source_gated_mechanism_support_floor=0.20,
        source_gated_mechanism_candidate_topk=0,
        source_gated_mechanism_support_mode="source_onset",
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
        event_base_weight = float(getattr(self.config, "rca_event_base_weight", 1.0) or 1.0)
        export_score_distribution = str(
            getattr(self.config, "rca_export_score_distribution", "raw") or "raw"
        ).lower()
        export_score_distribution_top_m = max(
            0,
            int(getattr(self.config, "rca_export_score_distribution_top_m", 0) or 0),
        )
        export_score_distribution_power = max(
            float(
                getattr(self.config, "rca_export_score_distribution_power", 2.0)
                or 2.0
            ),
            1e-6,
        )
        export_score_distribution_tau = max(
            float(
                getattr(self.config, "rca_export_score_distribution_tau", 1.0)
                or 1.0
            ),
            1e-6,
        )
        export_responsibility_score = bool(
            getattr(self.config, "rca_export_responsibility_score", False)
        )
        export_responsibility_ranking_weight = float(
            getattr(self.config, "rca_export_responsibility_ranking_weight", 0.0)
            or 0.0
        )
        export_responsibility_pre_weight = float(
            getattr(self.config, "rca_export_responsibility_pre_weight", 0.0)
            or 0.0
        )
        export_responsibility_base_weight = float(
            getattr(self.config, "rca_export_responsibility_base_weight", 0.0)
            or 0.0
        )
        export_responsibility_source_gate_weight = float(
            getattr(
                self.config,
                "rca_export_responsibility_source_gate_weight",
                0.0,
            )
            or 0.0
        )
        export_responsibility_onset_weight = float(
            getattr(self.config, "rca_export_responsibility_onset_weight", 0.0)
            or 0.0
        )
        export_responsibility_mechanism_residual_weight = float(
            getattr(
                self.config,
                "rca_export_responsibility_mechanism_residual_weight",
                0.0,
            )
            or 0.0
        )
        export_responsibility_root_weight = float(
            getattr(self.config, "rca_export_responsibility_root_weight", 0.0)
            or 0.0
        )
        export_responsibility_event_weight = float(
            getattr(self.config, "rca_export_responsibility_event_weight", 0.0)
            or 0.0
        )
        export_responsibility_evidence_fusion_weight = float(
            getattr(
                self.config,
                "rca_export_responsibility_evidence_fusion_weight",
                0.0,
            )
            or 0.0
        )
        export_responsibility_source_interaction_weight = float(
            getattr(
                self.config,
                "rca_export_responsibility_source_interaction_weight",
                0.0,
            )
            or 0.0
        )
        export_responsibility_graph_low_weight = float(
            getattr(self.config, "rca_export_responsibility_graph_low_weight", 0.0)
            or 0.0
        )
        export_responsibility_response_suppressor_low_weight = float(
            getattr(
                self.config,
                "rca_export_responsibility_response_suppressor_low_weight",
                0.0,
            )
            or 0.0
        )
        responsibility_rerank = bool(
            getattr(self.config, "rca_responsibility_rerank", False)
        )
        align_event_onset = bool(getattr(self.config, "rca_align_event_onset", False))
        align_baseline_window = int(
            getattr(self.config, "rca_align_event_onset_baseline_window", 300) or 300
        )
        align_onset_z = float(getattr(self.config, "rca_align_event_onset_z", 2.0) or 2.0)
        align_quantile = float(getattr(self.config, "rca_align_event_onset_quantile", 0.90) or 0.90)
        for event_id, (start, end) in enumerate(segments, start=1):
            if end <= start:
                continue
            event_anchor_start = int(start)
            if align_event_onset:
                event_anchor_start = self._align_rca_event_onset(
                    base_channel_scores,
                    start,
                    end,
                    baseline_window=align_baseline_window,
                    onset_z=align_onset_z,
                    quantile=align_quantile,
                )
            score_start, score_end = self._rca_event_score_bounds(
                event_anchor_start,
                end,
                event_head_ratio,
                event_head_points,
            )
            event_raw_scores = channel_scores[score_start:score_end].mean(axis=0)
            event_contrast_scores = np.zeros_like(event_raw_scores)
            if contrast_window > 0 and contrast_weight > 0.0 and event_anchor_start > 0:
                baseline_start = max(0, event_anchor_start - contrast_window)
                baseline_scores = channel_scores[baseline_start:event_anchor_start].mean(axis=0)
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
            event_source_gate_raw_scores = np.asarray(
                event_source_gate_scores,
                dtype=np.float64,
            ).copy()
            event_root_scores = (
                self._aggregate_root_score_event_component(
                    root_score_channel_scores,
                    score_start,
                    score_end,
                    root_score_pooling=root_score_pooling,
                    head_ratio=root_score_head_ratio,
                    head_points=root_score_head_points,
                    top_quantile=root_score_top_quantile,
                )
                if root_score_channel_scores is not None
                else np.zeros_like(event_raw_scores)
            )
            onset_source_scores = source_channel_scores if source_channel_scores is not None else base_channel_scores
            event_source_innovation_scores = (
                source_innovation_channel_scores[score_start:score_end].mean(axis=0)
                if source_innovation_channel_scores is not None
                else np.zeros_like(event_raw_scores)
            )
            if (
                source_innovation_weight > 0.0
                and str(source_innovation_mode or "series").lower() in {"onset_directional", "directional_onset", "temporal"}
            ):
                event_source_innovation_scores = self._event_directional_source_innovation_scores(
                    onset_source_scores,
                    getattr(self, "_channel_corr_prior", None),
                    start,
                    end,
                    baseline_window=onset_baseline_window,
                    onset_z=onset_z,
                    neighbor_weight=source_innovation_neighbor_weight,
                    lead_points=source_innovation_lead_points,
                )
            event_synthetic_scores = (
                synthetic_channel_scores[score_start:score_end].mean(axis=0)
                if synthetic_channel_scores is not None
                else np.zeros_like(event_raw_scores)
            )
            event_responsibility_scores = (
                self._aggregate_root_score_event_component(
                    event_responsibility_channel_scores,
                    score_start,
                    score_end,
                    root_score_pooling=event_responsibility_pooling,
                    head_ratio=event_responsibility_head_ratio,
                    head_points=event_responsibility_head_points,
                    top_quantile=event_responsibility_top_quantile,
                )
                if event_responsibility_channel_scores is not None
                else np.zeros_like(event_raw_scores)
            )
            event_route_scores = (
                event_route_channel_scores[score_start:score_end].mean(axis=0)
                if event_route_channel_scores is not None
                else np.zeros_like(event_raw_scores)
            )
            event_response_suppressor_scores = (
                response_suppressor_channel_scores[score_start:score_end].mean(axis=0)
                if response_suppressor_channel_scores is not None
                else np.zeros_like(event_raw_scores)
            )
            event_evidence_fusion_scores = (
                self._aggregate_root_score_event_component(
                    evidence_fusion_channel_scores,
                    score_start,
                    score_end,
                    root_score_pooling=evidence_fusion_pooling,
                    head_ratio=evidence_fusion_head_ratio,
                    head_points=evidence_fusion_head_points,
                    top_quantile=evidence_fusion_top_quantile,
                )
                if evidence_fusion_channel_scores is not None
                else np.zeros_like(event_raw_scores)
            )
            event_source_interaction_head_scores = (
                self._aggregate_root_score_event_component(
                    source_interaction_head_channel_scores,
                    score_start,
                    score_end,
                    root_score_pooling=source_interaction_head_pooling,
                    head_ratio=source_interaction_head_ratio,
                    head_points=source_interaction_head_points,
                    top_quantile=source_interaction_head_top_quantile,
                )
                if source_interaction_head_channel_scores is not None
                else np.zeros_like(event_raw_scores)
            )
            event_source_consistency_head_scores = (
                self._aggregate_root_score_event_component(
                    source_consistency_head_channel_scores,
                    score_start,
                    score_end,
                    root_score_pooling=source_consistency_head_pooling,
                    head_ratio=source_consistency_head_ratio,
                    head_points=source_consistency_head_points,
                    top_quantile=source_consistency_head_top_quantile,
                )
                if source_consistency_head_channel_scores is not None
                else np.zeros_like(event_raw_scores)
            )
            event_counterfactual_scores = (
                counterfactual_channel_scores[score_start:score_end].mean(axis=0)
                if counterfactual_channel_scores is not None
                else np.zeros_like(event_raw_scores)
            )
            event_onset_scores = np.zeros_like(event_raw_scores)
            if onset_weight > 0.0:
                event_onset_scores = self._event_onset_scores(
                    onset_source_scores,
                    event_anchor_start,
                    end,
                    baseline_window=onset_baseline_window,
                    onset_z=onset_z,
                )
                event_scores = event_scores + onset_weight * event_onset_scores
            event_mechanism_residual_scores = np.zeros_like(event_mechanism_scores)
            needs_mechanism_residual = (
                mechanism_residual_weight > 0.0
                or mechanism_guided_source_weight > 0.0
                or source_interaction_weight > 0.0
                or adaptive_mechanism_gate_weight > 0.0
                or adaptive_mechanism_selection_weight > 0.0
                or conservative_mechanism_weight > 0.0
                or source_gated_mechanism_weight > 0.0
            )
            if mechanism_residual_window > 0 and needs_mechanism_residual and event_anchor_start > 0:
                baseline_start = max(0, event_anchor_start - mechanism_residual_window)
                baseline_mechanism_scores = mechanism_channel_scores[baseline_start:event_anchor_start].mean(axis=0)
                event_mechanism_residual_scores = np.maximum(
                    event_mechanism_scores - baseline_mechanism_scores,
                    0.0,
                )
                if mechanism_residual_weight > 0.0:
                    event_scores = event_scores + mechanism_residual_weight * event_mechanism_residual_scores
            event_mechanism_guided_source_scores = np.zeros_like(event_raw_scores)
            event_adaptive_mechanism_gate_scores = np.zeros_like(event_raw_scores)
            event_adaptive_mechanism_scores = np.zeros_like(event_raw_scores)
            event_adaptive_mechanism_selection_gate_scores = np.zeros_like(event_raw_scores)
            event_adaptive_mechanism_selection_scores = np.zeros_like(event_raw_scores)
            event_conservative_mechanism_gate_scores = np.zeros_like(event_raw_scores)
            event_conservative_mechanism_scores = np.zeros_like(event_raw_scores)
            event_source_gated_mechanism_gate_scores = np.zeros_like(event_raw_scores)
            event_source_gated_mechanism_scores = np.zeros_like(event_raw_scores)
            event_response_suppressor_guard_scores = np.zeros_like(event_raw_scores)
            event_response_suppressor_penalty_scores = np.zeros_like(event_raw_scores)
            event_source_gate_evidence_guard_scores = np.ones_like(event_raw_scores)
            event_source_gate_calibrated_scores = np.asarray(
                event_source_gate_scores,
                dtype=np.float64,
            ).copy()
            if event_component_normalize:
                base_norm = self._normalize_event_component(event_base_scores)
                graph_norm = self._normalize_event_component(event_graph_scores)
                mechanism_norm = self._normalize_event_component(event_mechanism_scores)
                causal_norm = self._normalize_event_component(event_causal_scores)
                source_gate_norm = self._normalize_event_component(event_source_gate_scores)
                root_score_norm = self._normalize_event_component(event_root_scores)
                source_innovation_norm = self._normalize_event_component(event_source_innovation_scores)
                synthetic_norm = self._normalize_event_component(event_synthetic_scores)
                responsibility_norm = self._normalize_event_component(event_responsibility_scores)
                event_route_norm = self._normalize_event_component(event_route_scores)
                response_suppressor_norm = self._normalize_event_component(
                    event_response_suppressor_scores
                )
                evidence_fusion_norm = self._normalize_event_component(event_evidence_fusion_scores)
                source_interaction_head_norm = self._normalize_event_component(
                    event_source_interaction_head_scores
                )
                source_consistency_head_norm = self._normalize_event_component(
                    event_source_consistency_head_scores
                )
                counterfactual_norm = self._normalize_event_component(event_counterfactual_scores)
                source_score_norm = self._normalize_event_component(event_source_scores)
                onset_norm = self._normalize_event_component(event_onset_scores)
                mechanism_residual_norm = self._normalize_event_component(event_mechanism_residual_scores)
                contrast_norm = self._normalize_event_component(event_contrast_scores)
                guard_weight = min(
                    max(float(source_gate_evidence_guard_weight or 0.0), 0.0),
                    1.0,
                )
                if guard_weight > 0.0:
                    learned_source_support = np.maximum.reduce(
                        [
                            onset_norm,
                            root_score_norm,
                            responsibility_norm,
                            evidence_fusion_norm,
                            source_consistency_head_norm,
                        ]
                    )
                    anchor_support = np.sqrt(
                        np.maximum(base_norm * learned_source_support, 0.0)
                    )
                    learned_source_support = np.maximum(
                        learned_source_support,
                        anchor_support,
                    )
                    guard_floor = min(
                        max(float(source_gate_evidence_guard_floor or 0.0), 0.0),
                        1.0,
                    )
                    source_gate_guard = (
                        guard_floor + (1.0 - guard_floor) * learned_source_support
                    )
                    source_gate_norm = (
                        (1.0 - guard_weight) * source_gate_norm
                        + guard_weight * source_gate_norm * source_gate_guard
                    )
                    event_source_gate_evidence_guard_scores = source_gate_guard
                    event_source_gate_calibrated_scores = source_gate_norm
                    event_source_gate_scores = source_gate_norm
                mechanism_source_evidence_norm = np.maximum(mechanism_norm, mechanism_residual_norm)
                if source_gate_weight > 0.0:
                    mechanism_source_evidence_norm = np.maximum(
                        mechanism_source_evidence_norm,
                        source_gate_norm,
                    )
                event_mechanism_guided_source_scores = base_norm * mechanism_source_evidence_norm
                if adaptive_mechanism_gate_weight > 0.0:
                    mode = str(adaptive_mechanism_gate_mode or "source_onset").lower()
                    if mode == "source_gate":
                        mechanism_support_norm = source_gate_norm
                    elif mode == "onset":
                        mechanism_support_norm = onset_norm
                    elif mode == "base":
                        mechanism_support_norm = base_norm
                    elif mode == "base_source_onset":
                        mechanism_support_norm = np.maximum.reduce([base_norm, source_gate_norm, onset_norm])
                    else:
                        mechanism_support_norm = np.maximum(source_gate_norm, onset_norm)
                    raw_gate = np.sqrt(
                        np.maximum(mechanism_source_evidence_norm * mechanism_support_norm, 0.0)
                    )
                    floor = min(max(float(adaptive_mechanism_gate_floor or 0.0), 0.0), 1.0)
                    event_adaptive_mechanism_gate_scores = floor + (1.0 - floor) * raw_gate
                    event_adaptive_mechanism_scores = (
                        base_norm
                        * mechanism_source_evidence_norm
                        * event_adaptive_mechanism_gate_scores
                    )
                direct_source_norm = (
                    source_base_weight * base_norm
                    + source_score_weight * source_score_norm
                    + causal_weight * causal_norm
                    + source_gate_weight * source_gate_norm
                    + root_score_weight * root_score_norm
                    + source_innovation_weight * source_innovation_norm
                    + synthetic_weight * synthetic_norm
                    + event_responsibility_weight * responsibility_norm
                    + event_route_weight * event_route_norm
                    + evidence_fusion_weight * evidence_fusion_norm
                    + source_interaction_head_weight * source_interaction_head_norm
                    + source_consistency_head_weight * source_consistency_head_norm
                    + counterfactual_weight * counterfactual_norm
                )
                if source_interaction_weight > 0.0:
                    direct_source_norm = (
                        direct_source_norm
                        + source_interaction_weight * base_norm * np.maximum(onset_norm, source_gate_norm)
                    )
                if adaptive_mechanism_selection_weight > 0.0:
                    mode = str(adaptive_mechanism_selection_mode or "source_onset").lower()
                    if mode == "source_gate":
                        selection_support_norm = source_gate_norm
                    elif mode == "onset":
                        selection_support_norm = onset_norm
                    elif mode == "base":
                        selection_support_norm = base_norm
                    elif mode == "base_source_onset":
                        selection_support_norm = np.maximum.reduce([base_norm, source_gate_norm, onset_norm])
                    else:
                        selection_support_norm = np.maximum(source_gate_norm, onset_norm)
                    raw_selection_gate = np.sqrt(
                        np.maximum(mechanism_source_evidence_norm * selection_support_norm, 0.0)
                    )
                    selection_floor = min(
                        max(float(adaptive_mechanism_selection_floor or 0.0), 0.0),
                        1.0,
                    )
                    event_adaptive_mechanism_selection_gate_scores = (
                        selection_floor + (1.0 - selection_floor) * raw_selection_gate
                    )
                    mechanism_selected_norm = (
                        source_base_weight * base_norm
                        + mechanism_source_evidence_norm
                        + mechanism_guided_source_weight * event_mechanism_guided_source_scores
                    )
                    event_adaptive_mechanism_selection_scores = (
                        (1.0 - event_adaptive_mechanism_selection_gate_scores) * direct_source_norm
                        + event_adaptive_mechanism_selection_gate_scores * mechanism_selected_norm
                    )
                source_norm = (
                    source_base_weight * base_norm
                    + source_score_weight * source_score_norm
                    + source_mechanism_weight * mechanism_norm
                    + causal_weight * causal_norm
                    + source_gate_weight * source_gate_norm
                    + root_score_weight * root_score_norm
                    + source_innovation_weight * source_innovation_norm
                    + mechanism_guided_source_weight * event_mechanism_guided_source_scores
                    + adaptive_mechanism_gate_weight * event_adaptive_mechanism_scores
                    + synthetic_weight * synthetic_norm
                    + event_responsibility_weight * responsibility_norm
                    + event_route_weight * event_route_norm
                    + evidence_fusion_weight * evidence_fusion_norm
                    + source_interaction_head_weight * source_interaction_head_norm
                    + source_consistency_head_weight * source_consistency_head_norm
                    + counterfactual_weight * counterfactual_norm
                )
                if source_interaction_weight > 0.0:
                    source_evidence_norm = np.maximum(onset_norm, mechanism_residual_norm)
                    source_norm = source_norm + source_interaction_weight * base_norm * source_evidence_norm
                if adaptive_mechanism_selection_weight > 0.0:
                    selection_weight = min(max(float(adaptive_mechanism_selection_weight), 0.0), 1.0)
                    source_norm = (
                        (1.0 - selection_weight) * source_norm
                        + selection_weight * event_adaptive_mechanism_selection_scores
                    )
                if conservative_mechanism_weight > 0.0:
                    mode = str(conservative_mechanism_support_mode or "source_onset").lower()
                    if mode == "source_gate":
                        conservative_support_norm = source_gate_norm
                    elif mode == "onset":
                        conservative_support_norm = onset_norm
                    elif mode == "source_score":
                        conservative_support_norm = source_score_norm
                    elif mode == "base":
                        conservative_support_norm = base_norm
                    elif mode == "base_source_onset":
                        conservative_support_norm = np.maximum.reduce(
                            [base_norm, source_score_norm, source_gate_norm, onset_norm]
                        )
                    else:
                        conservative_support_norm = np.maximum.reduce(
                            [source_score_norm, source_gate_norm, onset_norm]
                        )
                    floor = min(
                        max(float(conservative_mechanism_support_floor or 0.0), 0.0),
                        1.0,
                    )
                    if floor >= 1.0:
                        support_gate = np.zeros_like(conservative_support_norm)
                    else:
                        support_gate = np.clip(
                            (conservative_support_norm - floor) / max(1.0 - floor, 1e-12),
                            0.0,
                            1.0,
                        )
                    candidate_topk = max(0, int(conservative_mechanism_candidate_topk or 0))
                    if candidate_topk > 0 and direct_source_norm.size > candidate_topk:
                        candidate_order = np.argsort(-direct_source_norm)[:candidate_topk]
                        candidate_mask = np.zeros_like(support_gate)
                        candidate_mask[candidate_order] = 1.0
                        support_gate = support_gate * candidate_mask
                    event_conservative_mechanism_gate_scores = support_gate
                    event_conservative_mechanism_scores = (
                        mechanism_source_evidence_norm
                        * support_gate
                        * np.maximum(source_score_norm, base_norm)
                    )
                    source_norm = (
                        source_norm
                        + conservative_mechanism_weight * event_conservative_mechanism_scores
                    )
                if source_gated_mechanism_weight > 0.0:
                    mode = str(source_gated_mechanism_support_mode or "source_onset").lower()
                    if mode == "source_gate":
                        gated_support_norm = source_gate_norm
                    elif mode == "onset":
                        gated_support_norm = onset_norm
                    elif mode == "source_score":
                        gated_support_norm = source_score_norm
                    elif mode == "base":
                        gated_support_norm = base_norm
                    elif mode == "base_source_onset":
                        gated_support_norm = np.sqrt(
                            np.maximum(
                                base_norm
                                * np.maximum.reduce([source_score_norm, source_gate_norm, onset_norm]),
                                0.0,
                            )
                        )
                    else:
                        gated_support_norm = np.sqrt(
                            np.maximum(source_score_norm * np.maximum(source_gate_norm, onset_norm), 0.0)
                        )
                    floor = min(
                        max(float(source_gated_mechanism_support_floor or 0.0), 0.0),
                        1.0,
                    )
                    if floor >= 1.0:
                        gated_support = np.zeros_like(gated_support_norm)
                    else:
                        gated_support = np.clip(
                            (gated_support_norm - floor) / max(1.0 - floor, 1e-12),
                            0.0,
                            1.0,
                        )
                    candidate_topk = max(0, int(source_gated_mechanism_candidate_topk or 0))
                    direct_source_norm = np.asarray(direct_source_norm, dtype=np.float64)
                    if candidate_topk > 0 and direct_source_norm.size > candidate_topk:
                        candidate_order = np.argsort(-direct_source_norm)[:candidate_topk]
                        candidate_mask = np.zeros_like(gated_support)
                        candidate_mask[candidate_order] = 1.0
                        gated_support = gated_support * candidate_mask
                    event_source_gated_mechanism_gate_scores = gated_support
                    source_anchor_norm = np.maximum.reduce([base_norm, source_score_norm, source_gate_norm])
                    event_source_gated_mechanism_scores = (
                        mechanism_source_evidence_norm * gated_support * source_anchor_norm
                    )
                    source_norm = (
                        source_norm
                        + source_gated_mechanism_weight * event_source_gated_mechanism_scores
                    )
                if use_source_propagation:
                    event_scores = source_weight * source_norm + propagation_weight * graph_norm
                else:
                    event_scores = (
                        event_base_weight * base_norm
                        + graph_weight * graph_norm
                        + mechanism_weight * mechanism_norm
                        + source_gate_weight * source_gate_norm
                        + root_score_weight * root_score_norm
                        + source_innovation_weight * source_innovation_norm
                        + synthetic_weight * synthetic_norm
                        + event_responsibility_weight * responsibility_norm
                        + event_route_weight * event_route_norm
                        + evidence_fusion_weight * evidence_fusion_norm
                        + source_interaction_head_weight * source_interaction_head_norm
                        + source_consistency_head_weight * source_consistency_head_norm
                        + counterfactual_weight * counterfactual_norm
                    )
                event_scores = (
                    event_scores
                    + onset_weight * onset_norm
                    + mechanism_residual_weight * mechanism_residual_norm
                    + contrast_weight * contrast_norm
                    - graph_penalty_weight * graph_norm
                )
                if response_suppressor_weight > 0.0:
                    guard_mode = str(
                        response_suppressor_source_guard_mode or "source_onset"
                    ).lower()
                    if guard_mode == "source_gate":
                        source_guard_norm = source_gate_norm
                    elif guard_mode == "onset":
                        source_guard_norm = onset_norm
                    elif guard_mode == "source_score":
                        source_guard_norm = source_score_norm
                    elif guard_mode == "base":
                        source_guard_norm = base_norm
                    elif guard_mode == "base_source_onset":
                        source_guard_norm = np.maximum.reduce(
                            [base_norm, source_score_norm, source_gate_norm, onset_norm]
                        )
                    else:
                        source_guard_norm = np.maximum.reduce(
                            [source_score_norm, source_gate_norm, onset_norm]
                        )
                    guard_floor = min(
                        max(float(response_suppressor_source_guard_floor or 0.0), 0.0),
                        1.0,
                    )
                    if guard_floor >= 1.0:
                        source_guard = np.zeros_like(source_guard_norm)
                    else:
                        source_guard = np.clip(
                            (source_guard_norm - guard_floor)
                            / max(1.0 - guard_floor, 1e-12),
                            0.0,
                            1.0,
                        )
                    event_response_suppressor_guard_scores = source_guard
                    event_response_suppressor_penalty_scores = (
                        response_suppressor_norm * (1.0 - source_guard)
                    )
                    event_scores = (
                        event_scores
                        - response_suppressor_weight
                        * event_response_suppressor_penalty_scores
                    )
                if source_consensus_weight > 0.0 or onset_consensus_weight > 0.0:
                    mode = str(source_consensus_mode or "sqrt_bg").lower()
                    if mode == "base":
                        consensus_norm = base_norm
                    elif mode == "graph":
                        consensus_norm = graph_norm
                    elif mode == "max_bg":
                        consensus_norm = np.maximum(base_norm, graph_norm)
                    elif mode == "min_bg":
                        consensus_norm = np.minimum(base_norm, graph_norm)
                    elif mode == "gate":
                        consensus_norm = source_gate_norm
                    elif mode == "none":
                        consensus_norm = np.ones_like(base_norm)
                    else:
                        consensus_norm = np.sqrt(np.maximum(base_norm * graph_norm, 0.0))
                    event_scores = (
                        event_scores
                        + source_consensus_weight * source_score_norm * consensus_norm
                        + onset_consensus_weight * onset_norm * consensus_norm
                    )
            elif mechanism_guided_source_weight > 0.0:
                base_norm = self._normalize_event_component(event_base_scores)
                mechanism_norm = self._normalize_event_component(event_mechanism_scores)
                mechanism_residual_norm = self._normalize_event_component(event_mechanism_residual_scores)
                source_gate_norm = self._normalize_event_component(event_source_gate_scores)
                source_innovation_norm = self._normalize_event_component(event_source_innovation_scores)
                mechanism_source_evidence_norm = np.maximum(mechanism_norm, mechanism_residual_norm)
                if source_gate_weight > 0.0:
                    mechanism_source_evidence_norm = np.maximum(
                        mechanism_source_evidence_norm,
                        source_gate_norm,
                    )
                event_mechanism_guided_source_scores = base_norm * mechanism_source_evidence_norm
                event_scores = (
                    event_scores
                    + mechanism_guided_source_weight * event_mechanism_guided_source_scores
                    + source_innovation_weight * source_innovation_norm
                )
            if response_suppressor_weight > 0.0 and not event_component_normalize:
                response_suppressor_norm = self._normalize_event_component(
                    event_response_suppressor_scores
                )
                guard_mode = str(
                    response_suppressor_source_guard_mode or "source_onset"
                ).lower()
                source_score_norm = self._normalize_event_component(event_source_scores)
                source_gate_norm = self._normalize_event_component(event_source_gate_scores)
                onset_norm = self._normalize_event_component(event_onset_scores)
                base_norm = self._normalize_event_component(event_base_scores)
                if guard_mode == "source_gate":
                    source_guard_norm = source_gate_norm
                elif guard_mode == "onset":
                    source_guard_norm = onset_norm
                elif guard_mode == "source_score":
                    source_guard_norm = source_score_norm
                elif guard_mode == "base":
                    source_guard_norm = base_norm
                elif guard_mode == "base_source_onset":
                    source_guard_norm = np.maximum.reduce(
                        [base_norm, source_score_norm, source_gate_norm, onset_norm]
                    )
                else:
                    source_guard_norm = np.maximum.reduce(
                        [source_score_norm, source_gate_norm, onset_norm]
                    )
                guard_floor = min(
                    max(float(response_suppressor_source_guard_floor or 0.0), 0.0),
                    1.0,
                )
                if guard_floor >= 1.0:
                    source_guard = np.zeros_like(source_guard_norm)
                else:
                    source_guard = np.clip(
                        (source_guard_norm - guard_floor)
                        / max(1.0 - guard_floor, 1e-12),
                        0.0,
                        1.0,
                    )
                event_response_suppressor_guard_scores = source_guard
                event_response_suppressor_penalty_scores = (
                    response_suppressor_norm * (1.0 - source_guard)
                )
                event_scores = (
                    event_scores
                    - response_suppressor_weight
                    * event_response_suppressor_penalty_scores
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

            pre_rerank_scores = np.asarray(event_scores, dtype=np.float64).copy()
            order, ranking_scores, topk_rerank_info = self._apply_topk_root_rerank(
                order,
                event_scores,
                feature_names,
                event_base_scores,
                group_scores,
                event_onset_scores,
                event_source_gate_scores,
                event_mechanism_residual_scores,
                event_source_consistency_head_scores,
                event_graph_scores,
            )
            ranking_scores = np.asarray(ranking_scores, dtype=np.float64)
            distribution_input_scores = ranking_scores
            responsibility_scores_for_order = None
            rank_before_responsibility = {
                int(idx): int(rank + 1) for rank, idx in enumerate(order)
            }
            if export_responsibility_score:
                event_source_interaction_export_scores = (
                    event_source_interaction_head_scores
                    if np.max(np.abs(event_source_interaction_head_scores)) > 1e-12
                    else event_base_scores
                    * np.maximum(event_onset_scores, event_mechanism_residual_scores)
                )
                distribution_input_scores = (
                    export_responsibility_ranking_weight
                    * self._normalize_event_component(ranking_scores)
                    + export_responsibility_pre_weight
                    * self._normalize_event_component(pre_rerank_scores)
                    + export_responsibility_base_weight
                    * self._normalize_event_component(event_base_scores)
                    + export_responsibility_source_gate_weight
                    * self._normalize_event_component(event_source_gate_scores)
                    + export_responsibility_onset_weight
                    * self._normalize_event_component(event_onset_scores)
                    + export_responsibility_mechanism_residual_weight
                    * self._normalize_event_component(event_mechanism_residual_scores)
                    + export_responsibility_root_weight
                    * self._normalize_event_component(event_root_scores)
                    + export_responsibility_event_weight
                    * self._normalize_event_component(event_responsibility_scores)
                    + export_responsibility_evidence_fusion_weight
                    * self._normalize_event_component(event_evidence_fusion_scores)
                    + export_responsibility_source_interaction_weight
                    * self._normalize_event_component(
                        event_source_interaction_export_scores
                    )
                    - export_responsibility_graph_low_weight
                    * self._normalize_event_component(event_graph_scores)
                    - export_responsibility_response_suppressor_low_weight
                    * self._normalize_event_component(event_response_suppressor_scores)
                )
                responsibility_scores_for_order = np.asarray(
                    distribution_input_scores,
                    dtype=np.float64,
                )
                if responsibility_rerank:
                    order = np.argsort(-responsibility_scores_for_order)
            final_order_scores = (
                responsibility_scores_for_order
                if responsibility_rerank and responsibility_scores_for_order is not None
                else ranking_scores
            )
            export_scores = self._event_score_distribution(
                distribution_input_scores,
                order,
                mode=export_score_distribution,
                top_m=export_score_distribution_top_m,
                power=export_score_distribution_power,
                tau=export_score_distribution_tau,
            )

            within_group_rank = {}
            for group, _ in sorted_groups:
                indices = [
                    idx
                    for idx, name in enumerate(feature_names)
                    if self._root_cause_group_name(name) == group
                ]
                indices = sorted(indices, key=lambda idx: float(final_order_scores[idx]), reverse=True)
                for rank, idx in enumerate(indices, start=1):
                    within_group_rank[idx] = int(rank)

            channel_ranking = [
                {
                    "rank": int(rank + 1),
                    "name": feature_names[idx],
                    "score": float(export_scores[idx]),
                    "ranking_score": float(ranking_scores[idx]),
                    "final_rank_score": float(final_order_scores[idx]),
                    "export_responsibility_score": float(
                        responsibility_scores_for_order[idx]
                        if responsibility_scores_for_order is not None
                        else 0.0
                    ),
                    "rank_before_responsibility_rerank": int(
                        rank_before_responsibility.get(int(idx), rank + 1)
                    ),
                    "responsibility_rerank_applied": bool(
                        responsibility_rerank and responsibility_scores_for_order is not None
                    ),
                    "pre_rerank_score": float(pre_rerank_scores[idx]),
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
                    "source_gate_raw_score": float(event_source_gate_raw_scores[idx]),
                    "source_gate_score": float(event_source_gate_scores[idx]),
                    "source_gate_evidence_guard_score": float(
                        event_source_gate_evidence_guard_scores[idx]
                    ),
                    "source_gate_calibrated_score": float(
                        event_source_gate_calibrated_scores[idx]
                    ),
                    "root_score": float(event_root_scores[idx]),
                    "source_innovation_score": float(event_source_innovation_scores[idx]),
                    "event_responsibility_score": float(event_responsibility_scores[idx]),
                    "event_route_score": float(event_route_scores[idx]),
                    "response_suppressor_score": float(event_response_suppressor_scores[idx]),
                    "response_suppressor_guard_score": float(
                        event_response_suppressor_guard_scores[idx]
                    ),
                    "response_suppressor_penalty_score": float(
                        event_response_suppressor_penalty_scores[idx]
                    ),
                    "evidence_fusion_score": float(event_evidence_fusion_scores[idx]),
                    "source_interaction_head_score": float(
                        event_source_interaction_head_scores[idx]
                    ),
                    "source_consistency_head_score": float(
                        event_source_consistency_head_scores[idx]
                    ),
                    "mechanism_guided_source_score": float(event_mechanism_guided_source_scores[idx]),
                    "adaptive_mechanism_gate_score": float(event_adaptive_mechanism_gate_scores[idx]),
                    "adaptive_mechanism_score": float(event_adaptive_mechanism_scores[idx]),
                    "adaptive_mechanism_selection_gate_score": float(
                        event_adaptive_mechanism_selection_gate_scores[idx]
                    ),
                    "adaptive_mechanism_selection_score": float(
                        event_adaptive_mechanism_selection_scores[idx]
                    ),
                    "conservative_mechanism_gate_score": float(
                        event_conservative_mechanism_gate_scores[idx]
                    ),
                    "conservative_mechanism_score": float(event_conservative_mechanism_scores[idx]),
                    "source_gated_mechanism_gate_score": float(
                        event_source_gated_mechanism_gate_scores[idx]
                    ),
                    "source_gated_mechanism_score": float(
                        event_source_gated_mechanism_scores[idx]
                    ),
                    "synthetic_rca_score": float(event_synthetic_scores[idx]),
                    "counterfactual_score": float(event_counterfactual_scores[idx]),
                    "onset_score": float(event_onset_scores[idx]),
                    "mechanism_residual_score": float(event_mechanism_residual_scores[idx]),
                    "contrast_score": float(event_contrast_scores[idx]),
                    "topk_rerank_applied": bool(idx in topk_rerank_info),
                    "topk_rerank_score": float(
                        topk_rerank_info.get(int(idx), {}).get("topk_rerank_score", 0.0)
                    ),
                    "topk_rerank_primary_score": float(
                        topk_rerank_info.get(int(idx), {}).get("topk_rerank_primary_score", 0.0)
                    ),
                    "topk_rerank_secondary_score": float(
                        topk_rerank_info.get(int(idx), {}).get("topk_rerank_secondary_score", 0.0)
                    ),
                    "topk_rerank_original_rank": int(
                        topk_rerank_info.get(int(idx), {}).get("topk_rerank_original_rank", 0)
                    ),
                    "topk_rerank_secondary_rank": int(
                        topk_rerank_info.get(int(idx), {}).get("topk_rerank_secondary_rank", 0)
                    ),
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
                    "aligned_start": int(event_anchor_start),
                    "score_start": int(score_start),
                    "score_end": int(score_end),
                    "score_length": int(score_end - score_start),
                    "top_channels": channel_ranking[:20],
                    "channel_ranking": channel_ranking,
                    "group_ranking": group_ranking,
                }
            )
        return events

    def _apply_rca_event_specificity_suppression(
        self,
        events,
        weight=0.0,
        top_k=1,
        threshold=0.35,
        min_events=20,
        source_guard_weight=0.0,
        source_guard_floor=0.0,
        keep_top_k=0,
        fill_secondary_top_k=0,
        secondary_base_weight=0.0,
        secondary_onset_weight=0.0,
        secondary_source_gate_weight=0.0,
        secondary_mechanism_residual_weight=0.0,
        secondary_source_interaction_weight=0.0,
        secondary_source_consistency_head_weight=0.0,
        secondary_evidence_fusion_weight=0.0,
    ):
        """Down-rank channels that behave like generic responders across many predicted events."""
        weight = float(weight or 0.0)
        if weight <= 0.0 or not events:
            return events
        top_k = max(1, int(top_k or 1))
        min_events = max(1, int(min_events or 1))
        source_guard_weight = min(max(float(source_guard_weight or 0.0), 0.0), 1.0)
        source_guard_floor = min(max(float(source_guard_floor or 0.0), 0.0), 1.0)
        keep_top_k = max(0, int(keep_top_k or 0))
        fill_secondary_top_k = max(0, int(fill_secondary_top_k or 0))
        secondary_base_weight = float(secondary_base_weight or 0.0)
        secondary_onset_weight = float(secondary_onset_weight or 0.0)
        secondary_source_gate_weight = float(secondary_source_gate_weight or 0.0)
        secondary_mechanism_residual_weight = float(secondary_mechanism_residual_weight or 0.0)
        secondary_source_interaction_weight = float(secondary_source_interaction_weight or 0.0)
        secondary_source_consistency_head_weight = float(
            secondary_source_consistency_head_weight or 0.0
        )
        secondary_evidence_fusion_weight = float(secondary_evidence_fusion_weight or 0.0)
        use_secondary_fill = (
            keep_top_k > 0
            and fill_secondary_top_k > keep_top_k
            and (
                abs(secondary_base_weight)
                + abs(secondary_onset_weight)
                + abs(secondary_source_gate_weight)
                + abs(secondary_mechanism_residual_weight)
                + abs(secondary_source_interaction_weight)
                + abs(secondary_source_consistency_head_weight)
                + abs(secondary_evidence_fusion_weight)
            )
            > 0.0
        )
        valid_events = [event for event in events if event.get("channel_ranking")]
        if len(valid_events) < min_events:
            return events

        top_counts = {}
        for event in valid_events:
            seen = set()
            for item in event.get("channel_ranking", [])[:top_k]:
                name = item.get("name")
                if not name or name in seen:
                    continue
                top_counts[name] = top_counts.get(name, 0) + 1
                seen.add(name)
        if not top_counts:
            return events

        total_events = float(len(valid_events))
        max_rate = max(top_counts.values()) / total_events
        threshold = float(threshold if threshold is not None else 0.0)
        if max_rate < threshold:
            return events

        for event in events:
            ranking = event.get("channel_ranking", [])
            if not ranking:
                continue
            raw_scores = np.asarray(
                [
                    float(
                        item.get(
                            "ranking_score",
                            item.get("raw_score", item.get("score", 0.0)),
                        )
                    )
                    for item in ranking
                ],
                dtype=np.float64,
            )
            if raw_scores.size and float(raw_scores.max() - raw_scores.min()) > 1e-12:
                base_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
            else:
                base_scores = np.zeros_like(raw_scores, dtype=np.float64)
            source_guard_scores = np.asarray(
                [float(item.get("source_gate_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            if (
                source_guard_weight > 0.0
                and source_guard_scores.size
                and float(source_guard_scores.max() - source_guard_scores.min()) > 1e-12
            ):
                source_guard_norm = (
                    (source_guard_scores - source_guard_scores.min())
                    / (source_guard_scores.max() - source_guard_scores.min())
                )
            else:
                source_guard_norm = np.zeros_like(raw_scores, dtype=np.float64)

            adjusted_items = []
            for idx, item in enumerate(ranking):
                name = item.get("name", "")
                nuisance_rate = float(top_counts.get(name, 0)) / total_events
                nuisance_penalty = weight * nuisance_rate
                if source_guard_weight > 0.0:
                    guarded_source = max(0.0, float(source_guard_norm[idx]) - source_guard_floor)
                    guarded_source = guarded_source / max(1.0 - source_guard_floor, 1e-12)
                    nuisance_penalty *= max(0.0, 1.0 - source_guard_weight * guarded_source)
                specificity_score = float(base_scores[idx]) - nuisance_penalty
                new_item = dict(item)
                new_item.setdefault("raw_rank", int(item.get("rank", idx + 1)))
                new_item.setdefault(
                    "raw_score",
                    float(item.get("ranking_score", item.get("score", 0.0))),
                )
                new_item["specificity_base_score"] = float(base_scores[idx])
                new_item["specificity_adjusted_score"] = specificity_score
                new_item["specificity_source_guard_score"] = float(source_guard_norm[idx])
                new_item["specificity_source_guard_weight"] = float(source_guard_weight)
                new_item["nuisance_rate"] = nuisance_rate
                new_item["nuisance_penalty"] = nuisance_penalty
                new_item["score"] = specificity_score
                adjusted_items.append(new_item)

            adjusted_items = sorted(
                adjusted_items,
                key=lambda item: float(item.get("specificity_adjusted_score", item.get("score", 0.0))),
                reverse=True,
            )
            if use_secondary_fill and len(adjusted_items) > keep_top_k:
                fill_limit = min(fill_secondary_top_k, len(adjusted_items))
                protected_count = min(keep_top_k, len(adjusted_items))
                base_arr = np.asarray(
                    [float(item.get("base_score", 0.0)) for item in adjusted_items],
                    dtype=np.float64,
                )
                onset_arr = np.asarray(
                    [float(item.get("onset_score", 0.0)) for item in adjusted_items],
                    dtype=np.float64,
                )
                source_gate_arr = np.asarray(
                    [float(item.get("source_gate_score", 0.0)) for item in adjusted_items],
                    dtype=np.float64,
                )
                mechanism_arr = np.asarray(
                    [float(item.get("mechanism_residual_score", 0.0)) for item in adjusted_items],
                    dtype=np.float64,
                )
                evidence_fusion_arr = np.asarray(
                    [float(item.get("evidence_fusion_score", 0.0)) for item in adjusted_items],
                    dtype=np.float64,
                )
                consistency_arr = np.asarray(
                    [
                        float(item.get("source_consistency_head_score", 0.0))
                        for item in adjusted_items
                    ],
                    dtype=np.float64,
                )
                interaction_arr = base_arr * np.maximum(onset_arr, mechanism_arr)
                secondary_scores = (
                    secondary_base_weight * self._normalize_event_component(base_arr)
                    + secondary_onset_weight * self._normalize_event_component(onset_arr)
                    + secondary_source_gate_weight * self._normalize_event_component(source_gate_arr)
                    + secondary_mechanism_residual_weight * self._normalize_event_component(mechanism_arr)
                    + secondary_source_interaction_weight
                    * self._normalize_event_component(interaction_arr)
                    + secondary_source_consistency_head_weight
                    * self._normalize_event_component(consistency_arr)
                    + secondary_evidence_fusion_weight
                    * self._normalize_event_component(evidence_fusion_arr)
                )
                secondary_pairs = sorted(
                    enumerate(secondary_scores.tolist()),
                    key=lambda pair: pair[1],
                    reverse=True,
                )
                for idx, score in enumerate(secondary_scores.tolist()):
                    adjusted_items[idx]["specificity_secondary_score"] = float(score)
                    adjusted_items[idx]["specificity_secondary_rank"] = 0
                    adjusted_items[idx]["specificity_post_fill_protected"] = idx < protected_count
                for rank, (idx, _) in enumerate(secondary_pairs, start=1):
                    adjusted_items[idx]["specificity_secondary_rank"] = int(rank)

                filled_items = list(adjusted_items[:protected_count])
                selected_names = {str(item.get("name", "")) for item in filled_items}
                for idx, _ in secondary_pairs:
                    if len(filled_items) >= fill_limit:
                        break
                    item = adjusted_items[idx]
                    name = str(item.get("name", ""))
                    if name in selected_names:
                        continue
                    item["specificity_post_fill_selected"] = True
                    filled_items.append(item)
                    selected_names.add(name)
                for item in adjusted_items:
                    name = str(item.get("name", ""))
                    if name in selected_names:
                        continue
                    filled_items.append(item)
                    selected_names.add(name)
                adjusted_items = filled_items

            for rank, item in enumerate(adjusted_items, start=1):
                item["rank"] = int(rank)
            event["channel_ranking"] = adjusted_items
            event["top_channels"] = adjusted_items[:20]
            event["event_specificity_applied"] = True
            event["event_specificity_top_k"] = int(top_k)
            event["event_specificity_max_rate"] = float(max_rate)
            event["event_specificity_keep_top_k"] = int(keep_top_k)
            event["event_specificity_fill_secondary_top_k"] = int(fill_secondary_top_k)
        return events

    def _apply_rca_adaptive_evidence_rerank(
        self,
        events,
        enabled=False,
        top_k=20,
        keep_top_k=1,
        fill_top_k=5,
        gate="source_top_old_rank",
        source_old_rank_threshold=1,
        source_margin_threshold=0.0,
        source_base_weight=0.45,
        source_onset_weight=0.25,
        source_gate_weight=1.0,
        source_mechanism_residual_weight=0.25,
        source_interaction_weight=0.25,
        fallback_original_weight=1.0,
        fallback_base_weight=0.45,
        fallback_onset_weight=0.20,
        fallback_mechanism_residual_weight=0.0,
        fallback_root_weight=0.0,
        fallback_event_weight=0.0,
        fallback_evidence_fusion_weight=0.0,
        fallback_source_interaction_weight=0.0,
        fallback_source_consistency_weight=0.0,
        fallback_graph_low_weight=0.0,
        fallback_response_suppressor_low_weight=0.0,
    ):
        """Choose source-evidence fill only when it agrees with fallback evidence."""
        if not enabled or not events:
            return events
        top_k = max(1, int(top_k or 20))
        keep_top_k = max(0, int(keep_top_k or 0))
        fill_top_k = max(1, int(fill_top_k or top_k))
        source_old_rank_threshold = max(1, int(source_old_rank_threshold or 1))
        source_margin_threshold = float(source_margin_threshold or 0.0)
        gate = str(gate or "source_top_old_rank").lower()

        source_base_weight = float(source_base_weight or 0.0)
        source_onset_weight = float(source_onset_weight or 0.0)
        source_gate_weight = float(source_gate_weight or 0.0)
        source_mechanism_residual_weight = float(source_mechanism_residual_weight or 0.0)
        source_interaction_weight = float(source_interaction_weight or 0.0)
        fallback_original_weight = float(fallback_original_weight or 0.0)
        fallback_base_weight = float(fallback_base_weight or 0.0)
        fallback_onset_weight = float(fallback_onset_weight or 0.0)
        fallback_mechanism_residual_weight = float(fallback_mechanism_residual_weight or 0.0)
        fallback_root_weight = float(fallback_root_weight or 0.0)
        fallback_event_weight = float(fallback_event_weight or 0.0)
        fallback_evidence_fusion_weight = float(fallback_evidence_fusion_weight or 0.0)
        fallback_source_interaction_weight = float(fallback_source_interaction_weight or 0.0)
        fallback_source_consistency_weight = float(fallback_source_consistency_weight or 0.0)
        fallback_graph_low_weight = float(fallback_graph_low_weight or 0.0)
        fallback_response_suppressor_low_weight = float(
            fallback_response_suppressor_low_weight or 0.0
        )

        for event in events:
            ranking = event.get("channel_ranking", [])
            if not ranking:
                continue
            candidate_count = min(top_k, len(ranking))
            if candidate_count <= 1:
                continue

            base_arr = np.asarray(
                [float(item.get("base_score", item.get("score", 0.0))) for item in ranking],
                dtype=np.float64,
            )
            onset_arr = np.asarray(
                [float(item.get("onset_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            source_gate_arr = np.asarray(
                [float(item.get("source_gate_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            mechanism_arr = np.asarray(
                [float(item.get("mechanism_residual_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            root_arr = np.asarray(
                [float(item.get("root_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            event_arr = np.asarray(
                [
                    float(item.get("event_responsibility_score", 0.0))
                    for item in ranking
                ],
                dtype=np.float64,
            )
            evidence_fusion_arr = np.asarray(
                [float(item.get("evidence_fusion_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            source_interaction_head_arr = np.asarray(
                [
                    float(item.get("source_interaction_head_score", 0.0))
                    for item in ranking
                ],
                dtype=np.float64,
            )
            source_consistency_arr = np.asarray(
                [
                    float(item.get("source_consistency_head_score", 0.0))
                    for item in ranking
                ],
                dtype=np.float64,
            )
            graph_arr = np.asarray(
                [float(item.get("graph_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            response_suppressor_arr = np.asarray(
                [
                    float(item.get("response_suppressor_score", 0.0))
                    for item in ranking
                ],
                dtype=np.float64,
            )
            original_arr = np.asarray(
                [
                    1.0
                    / max(
                        int(
                            item.get(
                                "topk_rerank_original_rank",
                                item.get("raw_rank", item.get("rank", idx + 1)),
                            )
                        )
                        or idx + 1,
                        1,
                    )
                    for idx, item in enumerate(ranking)
                ],
                dtype=np.float64,
            )
            interaction_arr = base_arr * np.maximum.reduce(
                [source_gate_arr, onset_arr, mechanism_arr]
            )
            learned_interaction_arr = (
                source_interaction_head_arr
                if np.max(np.abs(source_interaction_head_arr)) > 1e-12
                else base_arr * np.maximum(onset_arr, mechanism_arr)
            )

            source_scores = (
                source_base_weight * self._normalize_event_component(base_arr)
                + source_onset_weight * self._normalize_event_component(onset_arr)
                + source_gate_weight * self._normalize_event_component(source_gate_arr)
                + source_mechanism_residual_weight * self._normalize_event_component(mechanism_arr)
                + source_interaction_weight * self._normalize_event_component(interaction_arr)
            )
            fallback_scores = (
                fallback_original_weight * self._normalize_event_component(original_arr)
                + fallback_base_weight * self._normalize_event_component(base_arr)
                + fallback_onset_weight * self._normalize_event_component(onset_arr)
                + fallback_mechanism_residual_weight
                * self._normalize_event_component(mechanism_arr)
                + fallback_root_weight * self._normalize_event_component(root_arr)
                + fallback_event_weight * self._normalize_event_component(event_arr)
                + fallback_evidence_fusion_weight
                * self._normalize_event_component(evidence_fusion_arr)
                + fallback_source_interaction_weight
                * self._normalize_event_component(learned_interaction_arr)
                + fallback_source_consistency_weight
                * self._normalize_event_component(source_consistency_arr)
                - fallback_graph_low_weight * self._normalize_event_component(graph_arr)
                - fallback_response_suppressor_low_weight
                * self._normalize_event_component(response_suppressor_arr)
            )

            candidate_indices = list(range(candidate_count))
            source_order = sorted(
                candidate_indices,
                key=lambda idx: (float(source_scores[idx]), -idx),
                reverse=True,
            )
            fallback_order = sorted(
                candidate_indices,
                key=lambda idx: (float(fallback_scores[idx]), -idx),
                reverse=True,
            )
            if not source_order or not fallback_order:
                continue
            source_top_idx = int(source_order[0])
            fallback_rank_of_source_top = int(fallback_order.index(source_top_idx) + 1)
            source_rank_of_exported_top = (
                int(source_order.index(0) + 1) if 0 in source_order else candidate_count + 1
            )
            top3_overlap = len(set(source_order[:3]) & set(fallback_order[:3])) / float(
                max(min(3, len(candidate_indices)), 1)
            )
            source_margin = 0.0
            if len(source_order) > 1:
                source_margin = float(source_scores[source_order[0]] - source_scores[source_order[1]])

            if gate in {"source_top_old_rank", "source_top_fallback_rank"}:
                use_source = fallback_rank_of_source_top <= source_old_rank_threshold
                gate_value = float(fallback_rank_of_source_top)
            elif gate in {"exported_top_source_rank", "current_top_source_rank"}:
                use_source = source_rank_of_exported_top <= source_old_rank_threshold
                gate_value = float(source_rank_of_exported_top)
            elif gate in {"top3_overlap", "overlap"}:
                use_source = top3_overlap >= (1.0 / 3.0)
                gate_value = float(top3_overlap)
            elif gate in {"source_margin"}:
                use_source = source_margin >= source_margin_threshold
                gate_value = float(source_margin)
            else:
                use_source = fallback_rank_of_source_top <= source_old_rank_threshold
                gate_value = float(fallback_rank_of_source_top)

            if use_source:
                protected_count = min(keep_top_k, candidate_count)
                fill_limit = min(max(fill_top_k, protected_count), candidate_count)
                new_order = list(range(protected_count))
                selected = set(new_order)
                for idx in source_order:
                    if len(new_order) >= fill_limit:
                        break
                    if idx in selected:
                        continue
                    new_order.append(idx)
                    selected.add(idx)
                for idx in candidate_indices:
                    if idx not in selected:
                        new_order.append(idx)
                        selected.add(idx)
                final_scores = source_scores
            else:
                new_order = fallback_order + list(range(candidate_count, len(ranking)))
                selected = set(new_order)
                for idx in range(len(ranking)):
                    if idx not in selected:
                        new_order.append(idx)
                        selected.add(idx)
                final_scores = fallback_scores

            if use_source:
                selected = set(new_order)
                for idx in range(candidate_count, len(ranking)):
                    if idx not in selected:
                        new_order.append(idx)
                        selected.add(idx)

            adjusted_items = []
            for rank, idx in enumerate(new_order, start=1):
                item = dict(ranking[idx])
                item["rank"] = int(rank)
                item["adaptive_evidence_source_score"] = float(source_scores[idx])
                item["adaptive_evidence_fallback_score"] = float(fallback_scores[idx])
                item["adaptive_evidence_used_source"] = bool(use_source)
                item["adaptive_evidence_gate_value"] = float(gate_value)
                item["score"] = float(final_scores[idx])
                adjusted_items.append(item)
            event["channel_ranking"] = adjusted_items
            event["top_channels"] = adjusted_items[:20]
            event["adaptive_evidence_rerank_applied"] = True
            event["adaptive_evidence_used_source"] = bool(use_source)
            event["adaptive_evidence_gate"] = gate
            event["adaptive_evidence_gate_value"] = float(gate_value)
            event["adaptive_evidence_source_top_fallback_rank"] = fallback_rank_of_source_top
            event["adaptive_evidence_exported_top_source_rank"] = source_rank_of_exported_top
            event["adaptive_evidence_top3_overlap"] = float(top3_overlap)
            event["adaptive_evidence_source_margin"] = float(source_margin)
        return events

    def _refresh_rca_export_scores(
        self,
        events,
        *,
        export_responsibility_score=False,
        export_score_distribution="raw",
        export_score_distribution_top_m=0,
        export_score_distribution_power=2.0,
        export_score_distribution_tau=1.0,
        export_responsibility_ranking_weight=0.0,
        export_responsibility_pre_weight=0.0,
        export_responsibility_base_weight=0.0,
        export_responsibility_source_gate_weight=0.0,
        export_responsibility_onset_weight=0.0,
        export_responsibility_mechanism_residual_weight=0.0,
        export_responsibility_root_weight=0.0,
        export_responsibility_event_weight=0.0,
        export_responsibility_evidence_fusion_weight=0.0,
        export_responsibility_source_interaction_weight=0.0,
        export_responsibility_graph_low_weight=0.0,
        export_responsibility_response_suppressor_low_weight=0.0,
        responsibility_rerank=False,
    ):
        if not events or not bool(export_responsibility_score):
            return events
        mode = str(export_score_distribution or "raw").lower()
        for event in events:
            ranking = event.get("channel_ranking", [])
            if not ranking:
                continue
            ranking_arr = np.asarray(
                [
                    float(
                        item.get(
                            "ranking_score",
                            item.get("raw_score", item.get("score", 0.0)),
                        )
                    )
                    for item in ranking
                ],
                dtype=np.float64,
            )
            pre_arr = np.asarray(
                [float(item.get("pre_rerank_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            base_arr = np.asarray(
                [float(item.get("base_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            source_gate_arr = np.asarray(
                [float(item.get("source_gate_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            onset_arr = np.asarray(
                [float(item.get("onset_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            mechanism_arr = np.asarray(
                [float(item.get("mechanism_residual_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            root_arr = np.asarray(
                [float(item.get("root_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            event_arr = np.asarray(
                [
                    float(item.get("event_responsibility_score", 0.0))
                    for item in ranking
                ],
                dtype=np.float64,
            )
            evidence_fusion_arr = np.asarray(
                [float(item.get("evidence_fusion_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            source_interaction_head_arr = np.asarray(
                [
                    float(item.get("source_interaction_head_score", 0.0))
                    for item in ranking
                ],
                dtype=np.float64,
            )
            source_interaction_arr = (
                source_interaction_head_arr
                if np.max(np.abs(source_interaction_head_arr)) > 1e-12
                else base_arr * np.maximum(onset_arr, mechanism_arr)
            )
            graph_arr = np.asarray(
                [float(item.get("graph_score", 0.0)) for item in ranking],
                dtype=np.float64,
            )
            response_suppressor_arr = np.asarray(
                [
                    float(item.get("response_suppressor_score", 0.0))
                    for item in ranking
                ],
                dtype=np.float64,
            )
            responsibility_scores = (
                float(export_responsibility_ranking_weight or 0.0)
                * self._normalize_event_component(ranking_arr)
                + float(export_responsibility_pre_weight or 0.0)
                * self._normalize_event_component(pre_arr)
                + float(export_responsibility_base_weight or 0.0)
                * self._normalize_event_component(base_arr)
                + float(export_responsibility_source_gate_weight or 0.0)
                * self._normalize_event_component(source_gate_arr)
                + float(export_responsibility_onset_weight or 0.0)
                * self._normalize_event_component(onset_arr)
                + float(export_responsibility_mechanism_residual_weight or 0.0)
                * self._normalize_event_component(mechanism_arr)
                + float(export_responsibility_root_weight or 0.0)
                * self._normalize_event_component(root_arr)
                + float(export_responsibility_event_weight or 0.0)
                * self._normalize_event_component(event_arr)
                + float(export_responsibility_evidence_fusion_weight or 0.0)
                * self._normalize_event_component(evidence_fusion_arr)
                + float(export_responsibility_source_interaction_weight or 0.0)
                * self._normalize_event_component(source_interaction_arr)
                - float(export_responsibility_graph_low_weight or 0.0)
                * self._normalize_event_component(graph_arr)
                - float(export_responsibility_response_suppressor_low_weight or 0.0)
                * self._normalize_event_component(response_suppressor_arr)
            )
            export_scores = self._event_score_distribution(
                responsibility_scores,
                np.arange(len(ranking), dtype=np.int64),
                mode=mode,
                top_m=export_score_distribution_top_m,
                power=export_score_distribution_power,
                tau=export_score_distribution_tau,
            )
            if bool(responsibility_rerank):
                rerank_order = np.argsort(-responsibility_scores)
                ranking = [ranking[int(idx)] for idx in rerank_order]
                responsibility_scores = responsibility_scores[rerank_order]
                export_scores = export_scores[rerank_order]
            for idx, item in enumerate(ranking):
                item.setdefault("rank_before_responsibility_rerank", item.get("rank", idx + 1))
                item["score"] = float(export_scores[idx])
                item["export_responsibility_score"] = float(
                    responsibility_scores[idx]
                )
                item["final_rank_score"] = float(responsibility_scores[idx])
                item["responsibility_rerank_applied"] = bool(responsibility_rerank)
                item["rank"] = int(idx + 1)
            event["channel_ranking"] = ranking
            event["top_channels"] = ranking[:20]
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

    @staticmethod
    def _align_rca_event_onset(
        score_matrix,
        start,
        end,
        baseline_window=300,
        onset_z=2.0,
        quantile=0.90,
    ):
        """Align RCA aggregation to the first local residual burst inside an event."""
        if score_matrix is None or end <= start or start <= 0:
            return int(start)
        scores = np.asarray(score_matrix, dtype=np.float32)
        if scores.ndim != 2 or scores.size == 0:
            return int(start)

        baseline_start = max(0, int(start) - max(1, int(baseline_window)))
        baseline_scores = scores[baseline_start:int(start)]
        event_scores = scores[int(start):int(end)]
        if baseline_scores.size == 0 or event_scores.size == 0:
            return int(start)

        q = min(max(float(quantile), 0.0), 1.0)
        baseline_signal = np.quantile(baseline_scores, q, axis=1)
        event_signal = np.quantile(event_scores, q, axis=1)
        center = float(np.median(baseline_signal))
        mad = 1.4826 * float(np.median(np.abs(baseline_signal - center)))
        std = float(np.std(baseline_signal))
        scale = max(mad if mad > 1e-6 else std, 1e-6)
        threshold = center + max(0.0, float(onset_z)) * scale

        hits = np.flatnonzero(event_signal >= threshold)
        if hits.size == 0:
            return int(start)
        aligned = int(start) + int(hits[0])
        return min(max(int(start), aligned), int(end) - 1)

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
        root_score_channel_scores = getattr(self, "_last_root_score_channel_scores", None)
        if root_score_channel_scores is None:
            root_score_channel_scores = np.zeros_like(base_channel_scores)
        else:
            root_score_channel_scores = root_score_channel_scores[score_slice]
        event_responsibility_channel_scores = getattr(
            self,
            "_last_event_responsibility_channel_scores",
            None,
        )
        if event_responsibility_channel_scores is None:
            event_responsibility_channel_scores = np.zeros_like(base_channel_scores)
        else:
            event_responsibility_channel_scores = event_responsibility_channel_scores[score_slice]
        event_route_channel_scores = getattr(
            self,
            "_last_event_route_channel_scores",
            None,
        )
        if event_route_channel_scores is None:
            event_route_channel_scores = np.zeros_like(base_channel_scores)
        else:
            event_route_channel_scores = event_route_channel_scores[score_slice]
        response_suppressor_channel_scores = getattr(
            self,
            "_last_response_suppressor_channel_scores",
            None,
        )
        if response_suppressor_channel_scores is None:
            response_suppressor_channel_scores = np.zeros_like(base_channel_scores)
        else:
            response_suppressor_channel_scores = response_suppressor_channel_scores[score_slice]
        evidence_fusion_channel_scores = getattr(
            self,
            "_last_evidence_fusion_channel_scores",
            None,
        )
        if evidence_fusion_channel_scores is None:
            evidence_fusion_channel_scores = np.zeros_like(base_channel_scores)
        else:
            evidence_fusion_channel_scores = evidence_fusion_channel_scores[score_slice]
        source_interaction_head_channel_scores = getattr(
            self,
            "_last_source_interaction_head_channel_scores",
            None,
        )
        if source_interaction_head_channel_scores is None:
            source_interaction_head_channel_scores = np.zeros_like(base_channel_scores)
        else:
            source_interaction_head_channel_scores = source_interaction_head_channel_scores[score_slice]
        source_consistency_head_channel_scores = getattr(
            self,
            "_last_source_consistency_head_channel_scores",
            None,
        )
        if source_consistency_head_channel_scores is None:
            source_consistency_head_channel_scores = np.zeros_like(base_channel_scores)
        else:
            source_consistency_head_channel_scores = source_consistency_head_channel_scores[score_slice]
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
        source_score_weight = _cfg_float("rca_source_score_weight", 0.0)
        source_consensus_weight = _cfg_float("rca_source_consensus_weight", 0.0)
        onset_consensus_weight = _cfg_float("rca_onset_consensus_weight", 0.0)
        source_consensus_mode = str(getattr(self.config, "rca_source_consensus_mode", "sqrt_bg") or "sqrt_bg")
        causal_weight = _cfg_float("rca_causal_weight", 0.0)
        synthetic_weight = _cfg_float("rca_synthetic_weight", 0.0)
        event_responsibility_weight = _cfg_float("rca_event_responsibility_weight", 0.0)
        event_route_weight = _cfg_float("rca_event_route_weight", 0.0)
        source_gate_weight = _cfg_float("rca_source_gate_weight", 0.0)
        source_gate_evidence_guard_weight = _cfg_float(
            "rca_source_gate_evidence_guard_weight",
            0.0,
        )
        source_gate_evidence_guard_floor = _cfg_float(
            "rca_source_gate_evidence_guard_floor",
            0.0,
        )
        root_score_weight = _cfg_float("rca_root_score_weight", 0.0)
        root_score_signal = str(getattr(self.config, "rca_root_score_signal", "prob") or "prob")
        root_score_pooling = str(getattr(self.config, "rca_root_score_pooling", "mean") or "mean")
        root_score_head_ratio = _cfg_float("rca_root_score_head_ratio", 0.30)
        root_score_head_points = _cfg_int("rca_root_score_head_points", 30)
        root_score_top_quantile = _cfg_float("rca_root_score_top_quantile", 0.80)
        event_responsibility_pooling = str(
            getattr(self.config, "rca_event_responsibility_pooling", "mean") or "mean"
        )
        event_responsibility_head_ratio = _cfg_float("rca_event_responsibility_head_ratio", 0.30)
        event_responsibility_head_points = _cfg_int("rca_event_responsibility_head_points", 30)
        event_responsibility_top_quantile = _cfg_float("rca_event_responsibility_top_quantile", 0.80)
        evidence_fusion_weight = _cfg_float("rca_evidence_fusion_weight", 0.0)
        evidence_fusion_pooling = str(
            getattr(self.config, "rca_evidence_fusion_pooling", "mean") or "mean"
        )
        evidence_fusion_head_ratio = _cfg_float("rca_evidence_fusion_head_ratio", 0.30)
        evidence_fusion_head_points = _cfg_int("rca_evidence_fusion_head_points", 30)
        evidence_fusion_top_quantile = _cfg_float("rca_evidence_fusion_top_quantile", 0.80)
        source_interaction_head_weight = _cfg_float("rca_source_interaction_head_weight", 0.0)
        source_interaction_head_pooling = str(
            getattr(self.config, "rca_source_interaction_head_pooling", "head_mean")
            or "head_mean"
        )
        source_interaction_head_ratio = _cfg_float("rca_source_interaction_head_ratio", 0.30)
        source_interaction_head_points = _cfg_int("rca_source_interaction_head_points", 30)
        source_interaction_head_top_quantile = _cfg_float(
            "rca_source_interaction_head_top_quantile",
            0.80,
        )
        source_consistency_head_weight = _cfg_float("rca_source_consistency_head_weight", 0.0)
        source_consistency_head_pooling = str(
            getattr(self.config, "rca_source_consistency_head_pooling", "head_mean")
            or "head_mean"
        )
        source_consistency_head_ratio = _cfg_float("rca_source_consistency_head_ratio", 0.30)
        source_consistency_head_points = _cfg_int("rca_source_consistency_head_points", 30)
        source_consistency_head_top_quantile = _cfg_float(
            "rca_source_consistency_head_top_quantile",
            0.80,
        )
        source_interaction_weight = _cfg_float("rca_source_interaction_weight", 0.0)
        source_innovation_weight = _cfg_float("rca_source_innovation_weight", 0.0)
        source_innovation_mode = str(
            getattr(self.config, "rca_source_innovation_mode", "series") or "series"
        ).lower()
        source_innovation_neighbor_weight = _cfg_float("rca_source_innovation_neighbor_weight", 1.0)
        source_innovation_lead_points = _cfg_int("rca_source_innovation_lead_points", 1)
        mechanism_guided_source_weight = _cfg_float("rca_mechanism_guided_source_weight", 0.0)
        adaptive_mechanism_gate_weight = _cfg_float("rca_adaptive_mechanism_gate_weight", 0.0)
        adaptive_mechanism_gate_floor = _cfg_float("rca_adaptive_mechanism_gate_floor", 0.05)
        adaptive_mechanism_gate_mode = str(
            getattr(self.config, "rca_adaptive_mechanism_gate_mode", "source_onset") or "source_onset"
        )
        adaptive_mechanism_selection_weight = _cfg_float(
            "rca_adaptive_mechanism_selection_weight",
            0.0,
        )
        adaptive_mechanism_selection_floor = _cfg_float(
            "rca_adaptive_mechanism_selection_floor",
            0.05,
        )
        adaptive_mechanism_selection_mode = str(
            getattr(self.config, "rca_adaptive_mechanism_selection_mode", "source_onset")
            or "source_onset"
        )
        conservative_mechanism_weight = _cfg_float("rca_conservative_mechanism_weight", 0.0)
        conservative_mechanism_support_floor = _cfg_float(
            "rca_conservative_mechanism_support_floor",
            0.20,
        )
        conservative_mechanism_candidate_topk = _cfg_int(
            "rca_conservative_mechanism_candidate_topk",
            0,
        )
        conservative_mechanism_support_mode = str(
            getattr(self.config, "rca_conservative_mechanism_support_mode", "source_onset")
            or "source_onset"
        )
        source_gated_mechanism_weight = _cfg_float("rca_source_gated_mechanism_weight", 0.0)
        source_gated_mechanism_support_floor = _cfg_float(
            "rca_source_gated_mechanism_support_floor",
            0.20,
        )
        source_gated_mechanism_candidate_topk = _cfg_int(
            "rca_source_gated_mechanism_candidate_topk",
            0,
        )
        source_gated_mechanism_support_mode = str(
            getattr(self.config, "rca_source_gated_mechanism_support_mode", "source_onset")
            or "source_onset"
        )
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
        event_base_weight = _cfg_float("rca_event_base_weight", 1.0)
        export_score_distribution = str(
            getattr(self.config, "rca_export_score_distribution", "raw") or "raw"
        ).lower()
        export_score_distribution_top_m = _cfg_int(
            "rca_export_score_distribution_top_m",
            0,
        )
        export_score_distribution_power = _cfg_float(
            "rca_export_score_distribution_power",
            2.0,
        )
        export_score_distribution_tau = _cfg_float(
            "rca_export_score_distribution_tau",
            1.0,
        )
        export_responsibility_score = bool(
            getattr(self.config, "rca_export_responsibility_score", False)
        )
        export_responsibility_ranking_weight = _cfg_float(
            "rca_export_responsibility_ranking_weight",
            0.0,
        )
        export_responsibility_pre_weight = _cfg_float(
            "rca_export_responsibility_pre_weight",
            0.0,
        )
        export_responsibility_base_weight = _cfg_float(
            "rca_export_responsibility_base_weight",
            0.0,
        )
        export_responsibility_source_gate_weight = _cfg_float(
            "rca_export_responsibility_source_gate_weight",
            0.0,
        )
        export_responsibility_onset_weight = _cfg_float(
            "rca_export_responsibility_onset_weight",
            0.0,
        )
        export_responsibility_mechanism_residual_weight = _cfg_float(
            "rca_export_responsibility_mechanism_residual_weight",
            0.0,
        )
        export_responsibility_root_weight = _cfg_float(
            "rca_export_responsibility_root_weight",
            0.0,
        )
        export_responsibility_event_weight = _cfg_float(
            "rca_export_responsibility_event_weight",
            0.0,
        )
        export_responsibility_evidence_fusion_weight = _cfg_float(
            "rca_export_responsibility_evidence_fusion_weight",
            0.0,
        )
        export_responsibility_source_interaction_weight = _cfg_float(
            "rca_export_responsibility_source_interaction_weight",
            0.0,
        )
        export_responsibility_graph_low_weight = _cfg_float(
            "rca_export_responsibility_graph_low_weight",
            0.0,
        )
        export_responsibility_response_suppressor_low_weight = _cfg_float(
            "rca_export_responsibility_response_suppressor_low_weight",
            0.0,
        )
        responsibility_rerank = bool(
            getattr(self.config, "rca_responsibility_rerank", False)
        )
        graph_penalty_weight = _cfg_float("rca_graph_penalty_weight", 0.0)
        response_suppressor_weight = _cfg_float("rca_response_suppressor_weight", 0.0)
        response_suppressor_source_guard_mode = str(
            getattr(
                self.config,
                "rca_response_suppressor_source_guard_mode",
                "source_onset",
            )
            or "source_onset"
        )
        response_suppressor_source_guard_floor = _cfg_float(
            "rca_response_suppressor_source_guard_floor",
            0.0,
        )
        hierarchical_mode = str(getattr(self.config, "rca_hierarchical_mode", "off") or "off")
        hierarchical_group_topk = _cfg_int("rca_hierarchical_group_topk", 0)
        hierarchical_group_boost = _cfg_float("rca_hierarchical_group_boost", 0.0)
        hierarchical_outside_penalty = _cfg_float("rca_hierarchical_outside_penalty", 0.0)
        hierarchical_group_aggregation = str(
            getattr(self.config, "rca_hierarchical_group_aggregation", "max") or "max"
        )
        event_specificity_weight = _cfg_float("rca_event_specificity_weight", 0.0)
        event_specificity_top_k = _cfg_int("rca_event_specificity_top_k", 1)
        event_specificity_threshold = _cfg_float("rca_event_specificity_threshold", 0.35)
        event_specificity_min_events = _cfg_int("rca_event_specificity_min_events", 20)
        event_specificity_source_guard_weight = _cfg_float(
            "rca_event_specificity_source_guard_weight",
            0.0,
        )
        event_specificity_source_guard_floor = _cfg_float(
            "rca_event_specificity_source_guard_floor",
            0.0,
        )
        event_specificity_keep_top_k = _cfg_int("rca_event_specificity_keep_top_k", 0)
        event_specificity_fill_secondary_top_k = _cfg_int(
            "rca_event_specificity_fill_secondary_top_k",
            0,
        )
        event_specificity_secondary_base_weight = _cfg_float(
            "rca_event_specificity_secondary_base_weight",
            0.0,
        )
        event_specificity_secondary_onset_weight = _cfg_float(
            "rca_event_specificity_secondary_onset_weight",
            0.0,
        )
        event_specificity_secondary_source_gate_weight = _cfg_float(
            "rca_event_specificity_secondary_source_gate_weight",
            0.0,
        )
        event_specificity_secondary_mechanism_residual_weight = _cfg_float(
            "rca_event_specificity_secondary_mechanism_residual_weight",
            0.0,
        )
        event_specificity_secondary_source_interaction_weight = _cfg_float(
            "rca_event_specificity_secondary_source_interaction_weight",
            0.0,
        )
        event_specificity_secondary_source_consistency_head_weight = _cfg_float(
            "rca_event_specificity_secondary_source_consistency_head_weight",
            0.0,
        )
        event_specificity_secondary_evidence_fusion_weight = _cfg_float(
            "rca_event_specificity_secondary_evidence_fusion_weight",
            0.0,
        )
        adaptive_evidence_rerank = bool(
            getattr(self.config, "rca_adaptive_evidence_rerank", False)
        )
        adaptive_evidence_top_k = _cfg_int("rca_adaptive_evidence_top_k", 20)
        adaptive_evidence_keep_top_k = _cfg_int("rca_adaptive_evidence_keep_top_k", 1)
        adaptive_evidence_fill_top_k = _cfg_int("rca_adaptive_evidence_fill_top_k", 5)
        adaptive_evidence_gate = str(
            getattr(self.config, "rca_adaptive_evidence_gate", "source_top_old_rank")
            or "source_top_old_rank"
        )
        adaptive_evidence_source_old_rank_threshold = _cfg_int(
            "rca_adaptive_evidence_source_old_rank_threshold",
            1,
        )
        adaptive_evidence_source_margin_threshold = _cfg_float(
            "rca_adaptive_evidence_source_margin_threshold",
            0.0,
        )
        adaptive_evidence_source_base_weight = _cfg_float(
            "rca_adaptive_evidence_source_base_weight",
            0.45,
        )
        adaptive_evidence_source_onset_weight = _cfg_float(
            "rca_adaptive_evidence_source_onset_weight",
            0.25,
        )
        adaptive_evidence_source_gate_weight = _cfg_float(
            "rca_adaptive_evidence_source_gate_weight",
            1.0,
        )
        adaptive_evidence_source_mechanism_residual_weight = _cfg_float(
            "rca_adaptive_evidence_source_mechanism_residual_weight",
            0.25,
        )
        adaptive_evidence_source_interaction_weight = _cfg_float(
            "rca_adaptive_evidence_source_interaction_weight",
            0.25,
        )
        adaptive_evidence_fallback_original_weight = _cfg_float(
            "rca_adaptive_evidence_fallback_original_weight",
            1.0,
        )
        adaptive_evidence_fallback_base_weight = _cfg_float(
            "rca_adaptive_evidence_fallback_base_weight",
            0.45,
        )
        adaptive_evidence_fallback_onset_weight = _cfg_float(
            "rca_adaptive_evidence_fallback_onset_weight",
            0.20,
        )
        adaptive_evidence_fallback_mechanism_residual_weight = _cfg_float(
            "rca_adaptive_evidence_fallback_mechanism_residual_weight",
            0.0,
        )
        adaptive_evidence_fallback_root_weight = _cfg_float(
            "rca_adaptive_evidence_fallback_root_weight",
            0.0,
        )
        adaptive_evidence_fallback_event_weight = _cfg_float(
            "rca_adaptive_evidence_fallback_event_weight",
            0.0,
        )
        adaptive_evidence_fallback_evidence_fusion_weight = _cfg_float(
            "rca_adaptive_evidence_fallback_evidence_fusion_weight",
            0.0,
        )
        adaptive_evidence_fallback_source_interaction_weight = _cfg_float(
            "rca_adaptive_evidence_fallback_source_interaction_weight",
            0.0,
        )
        adaptive_evidence_fallback_source_consistency_weight = _cfg_float(
            "rca_adaptive_evidence_fallback_source_consistency_weight",
            0.0,
        )
        adaptive_evidence_fallback_graph_low_weight = _cfg_float(
            "rca_adaptive_evidence_fallback_graph_low_weight",
            0.0,
        )
        adaptive_evidence_fallback_response_suppressor_low_weight = _cfg_float(
            "rca_adaptive_evidence_fallback_response_suppressor_low_weight",
            0.0,
        )
        export_lite = bool(getattr(self.config, "rca_export_lite", False))
        export_top_k = int(getattr(self.config, "rca_export_top_k", 20) or 20)
        max_channel_ranking = export_top_k if export_lite else None
        source_channel_scores = None
        propagation_channel_scores = None
        source_innovation_channel_scores = None
        counterfactual_channel_scores = None
        directional_innovation_modes = {"onset_directional", "directional_onset", "temporal"}
        if source_innovation_weight > 0.0 and source_innovation_mode not in directional_innovation_modes:
            source_innovation_channel_scores = self._source_innovation_scores(
                base_channel_scores,
                getattr(self, "_channel_corr_prior", None),
                neighbor_weight=source_innovation_neighbor_weight,
            )
        if counterfactual_weight > 0.0:
            counterfactual_candidate_scores = (
                source_base_weight * base_channel_scores
                + source_mechanism_weight * mechanism_channel_scores
                + causal_weight * causal_channel_scores
                + source_gate_weight * source_gate_channel_scores
                + root_score_weight * root_score_channel_scores
                + event_responsibility_weight * event_responsibility_channel_scores
                + event_route_weight * event_route_channel_scores
                + evidence_fusion_weight * evidence_fusion_channel_scores
                + source_consistency_head_weight * source_consistency_head_channel_scores
                + source_innovation_weight * (
                    source_innovation_channel_scores
                    if source_innovation_channel_scores is not None
                    else 0.0
                )
                + synthetic_weight * synthetic_channel_scores
            )
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
                + root_score_weight * root_score_channel_scores
                + event_responsibility_weight * event_responsibility_channel_scores
                + event_route_weight * event_route_channel_scores
                + evidence_fusion_weight * evidence_fusion_channel_scores
                + source_consistency_head_weight * source_consistency_head_channel_scores
                + source_innovation_weight * (
                    source_innovation_channel_scores
                    if source_innovation_channel_scores is not None
                    else 0.0
                )
                + synthetic_weight * synthetic_channel_scores
            )
            if counterfactual_channel_scores is not None:
                source_channel_scores = (
                    source_channel_scores
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
                + root_score_weight * root_score_channel_scores
                + event_responsibility_weight * event_responsibility_channel_scores
                + event_route_weight * event_route_channel_scores
                + evidence_fusion_weight * evidence_fusion_channel_scores
                + source_consistency_head_weight * source_consistency_head_channel_scores
                + source_innovation_weight * (
                    source_innovation_channel_scores
                    if source_innovation_channel_scores is not None
                    else 0.0
                )
                + synthetic_weight * synthetic_channel_scores
            )
            if counterfactual_channel_scores is not None:
                channel_scores = (
                    channel_scores
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
            root_score_channel_scores=root_score_channel_scores,
            root_score_pooling=root_score_pooling,
            root_score_head_ratio=root_score_head_ratio,
            root_score_head_points=root_score_head_points,
            root_score_top_quantile=root_score_top_quantile,
            source_innovation_channel_scores=source_innovation_channel_scores,
            synthetic_channel_scores=synthetic_channel_scores,
            event_responsibility_channel_scores=event_responsibility_channel_scores,
            event_responsibility_pooling=event_responsibility_pooling,
            event_responsibility_head_ratio=event_responsibility_head_ratio,
            event_responsibility_head_points=event_responsibility_head_points,
            event_responsibility_top_quantile=event_responsibility_top_quantile,
            event_route_channel_scores=event_route_channel_scores,
            response_suppressor_channel_scores=response_suppressor_channel_scores,
            evidence_fusion_channel_scores=evidence_fusion_channel_scores,
            evidence_fusion_pooling=evidence_fusion_pooling,
            evidence_fusion_head_ratio=evidence_fusion_head_ratio,
            evidence_fusion_head_points=evidence_fusion_head_points,
            evidence_fusion_top_quantile=evidence_fusion_top_quantile,
            source_interaction_head_channel_scores=source_interaction_head_channel_scores,
            source_interaction_head_pooling=source_interaction_head_pooling,
            source_interaction_head_ratio=source_interaction_head_ratio,
            source_interaction_head_points=source_interaction_head_points,
            source_interaction_head_top_quantile=source_interaction_head_top_quantile,
            source_consistency_head_channel_scores=source_consistency_head_channel_scores,
            source_consistency_head_pooling=source_consistency_head_pooling,
            source_consistency_head_ratio=source_consistency_head_ratio,
            source_consistency_head_points=source_consistency_head_points,
            source_consistency_head_top_quantile=source_consistency_head_top_quantile,
            counterfactual_channel_scores=counterfactual_channel_scores,
            use_source_propagation=use_source_propagation,
            graph_weight=graph_weight,
            mechanism_weight=mechanism_weight,
            source_weight=source_weight,
            source_base_weight=source_base_weight,
            propagation_weight=propagation_weight,
            source_mechanism_weight=source_mechanism_weight,
            source_score_weight=source_score_weight,
            source_consensus_weight=source_consensus_weight,
            onset_consensus_weight=onset_consensus_weight,
            source_consensus_mode=source_consensus_mode,
            causal_weight=causal_weight,
            synthetic_weight=synthetic_weight,
            event_responsibility_weight=event_responsibility_weight,
            event_route_weight=event_route_weight,
            response_suppressor_weight=response_suppressor_weight,
            response_suppressor_source_guard_mode=response_suppressor_source_guard_mode,
            response_suppressor_source_guard_floor=response_suppressor_source_guard_floor,
            evidence_fusion_weight=evidence_fusion_weight,
            source_interaction_head_weight=source_interaction_head_weight,
            source_consistency_head_weight=source_consistency_head_weight,
            source_gate_weight=source_gate_weight,
            source_gate_evidence_guard_weight=source_gate_evidence_guard_weight,
            source_gate_evidence_guard_floor=source_gate_evidence_guard_floor,
            root_score_weight=root_score_weight,
            source_interaction_weight=source_interaction_weight,
            source_innovation_weight=source_innovation_weight,
            source_innovation_mode=source_innovation_mode,
            source_innovation_neighbor_weight=source_innovation_neighbor_weight,
            source_innovation_lead_points=source_innovation_lead_points,
            mechanism_guided_source_weight=mechanism_guided_source_weight,
            adaptive_mechanism_gate_weight=adaptive_mechanism_gate_weight,
            adaptive_mechanism_gate_floor=adaptive_mechanism_gate_floor,
            adaptive_mechanism_gate_mode=adaptive_mechanism_gate_mode,
            adaptive_mechanism_selection_weight=adaptive_mechanism_selection_weight,
            adaptive_mechanism_selection_floor=adaptive_mechanism_selection_floor,
            adaptive_mechanism_selection_mode=adaptive_mechanism_selection_mode,
            conservative_mechanism_weight=conservative_mechanism_weight,
            conservative_mechanism_support_floor=conservative_mechanism_support_floor,
            conservative_mechanism_candidate_topk=conservative_mechanism_candidate_topk,
            conservative_mechanism_support_mode=conservative_mechanism_support_mode,
            source_gated_mechanism_weight=source_gated_mechanism_weight,
            source_gated_mechanism_support_floor=source_gated_mechanism_support_floor,
            source_gated_mechanism_candidate_topk=source_gated_mechanism_candidate_topk,
            source_gated_mechanism_support_mode=source_gated_mechanism_support_mode,
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
            lite_predictions = {}
            if isinstance(predict_labels, dict):
                for key, prediction in predict_labels.items():
                    key_name = self._rca_prediction_key_name(key)
                    mask = self._normalize_prediction_mask(prediction, len(labels))
                    if rca_offset > 0:
                        raw_mask = self._normalize_prediction_mask(prediction, len(test_data))
                        mask = raw_mask[rca_offset:rca_offset + len(labels)]
                    lite_predictions[key_name] = mask
            elif pred_mask is not None:
                lite_predictions[pred_key] = pred_mask
            for key_name, mask in lite_predictions.items():
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
                    root_score_channel_scores=root_score_channel_scores,
                    root_score_pooling=root_score_pooling,
                    root_score_head_ratio=root_score_head_ratio,
                    root_score_head_points=root_score_head_points,
                    root_score_top_quantile=root_score_top_quantile,
                    source_innovation_channel_scores=source_innovation_channel_scores,
                    synthetic_channel_scores=synthetic_channel_scores,
                    event_responsibility_channel_scores=event_responsibility_channel_scores,
                    event_responsibility_pooling=event_responsibility_pooling,
                    event_responsibility_head_ratio=event_responsibility_head_ratio,
                    event_responsibility_head_points=event_responsibility_head_points,
                    event_responsibility_top_quantile=event_responsibility_top_quantile,
                    event_route_channel_scores=event_route_channel_scores,
                    response_suppressor_channel_scores=response_suppressor_channel_scores,
                    evidence_fusion_channel_scores=evidence_fusion_channel_scores,
                    evidence_fusion_pooling=evidence_fusion_pooling,
                    evidence_fusion_head_ratio=evidence_fusion_head_ratio,
                    evidence_fusion_head_points=evidence_fusion_head_points,
                    evidence_fusion_top_quantile=evidence_fusion_top_quantile,
                    source_interaction_head_channel_scores=source_interaction_head_channel_scores,
                    source_interaction_head_pooling=source_interaction_head_pooling,
                    source_interaction_head_ratio=source_interaction_head_ratio,
                    source_interaction_head_points=source_interaction_head_points,
                    source_interaction_head_top_quantile=source_interaction_head_top_quantile,
                    source_consistency_head_channel_scores=source_consistency_head_channel_scores,
                    source_consistency_head_pooling=source_consistency_head_pooling,
                    source_consistency_head_ratio=source_consistency_head_ratio,
                    source_consistency_head_points=source_consistency_head_points,
                    source_consistency_head_top_quantile=source_consistency_head_top_quantile,
                    counterfactual_channel_scores=counterfactual_channel_scores,
                    use_source_propagation=use_source_propagation,
                    graph_weight=graph_weight,
                    mechanism_weight=mechanism_weight,
                    source_weight=source_weight,
                    source_base_weight=source_base_weight,
                    propagation_weight=propagation_weight,
                    source_mechanism_weight=source_mechanism_weight,
                    source_score_weight=source_score_weight,
                    source_consensus_weight=source_consensus_weight,
                    onset_consensus_weight=onset_consensus_weight,
                    source_consensus_mode=source_consensus_mode,
                    causal_weight=causal_weight,
                    synthetic_weight=synthetic_weight,
                    event_responsibility_weight=event_responsibility_weight,
                    event_route_weight=event_route_weight,
                    response_suppressor_weight=response_suppressor_weight,
                    response_suppressor_source_guard_mode=response_suppressor_source_guard_mode,
                    response_suppressor_source_guard_floor=response_suppressor_source_guard_floor,
                    evidence_fusion_weight=evidence_fusion_weight,
                    source_interaction_head_weight=source_interaction_head_weight,
                    source_consistency_head_weight=source_consistency_head_weight,
                    source_gate_weight=source_gate_weight,
                    source_gate_evidence_guard_weight=source_gate_evidence_guard_weight,
                    source_gate_evidence_guard_floor=source_gate_evidence_guard_floor,
                    root_score_weight=root_score_weight,
                    source_interaction_weight=source_interaction_weight,
                    source_innovation_weight=source_innovation_weight,
                    source_innovation_mode=source_innovation_mode,
                    source_innovation_neighbor_weight=source_innovation_neighbor_weight,
                    source_innovation_lead_points=source_innovation_lead_points,
                    mechanism_guided_source_weight=mechanism_guided_source_weight,
                    adaptive_mechanism_gate_weight=adaptive_mechanism_gate_weight,
                    adaptive_mechanism_gate_floor=adaptive_mechanism_gate_floor,
                    adaptive_mechanism_gate_mode=adaptive_mechanism_gate_mode,
                    adaptive_mechanism_selection_weight=adaptive_mechanism_selection_weight,
                    adaptive_mechanism_selection_floor=adaptive_mechanism_selection_floor,
                    adaptive_mechanism_selection_mode=adaptive_mechanism_selection_mode,
                    conservative_mechanism_weight=conservative_mechanism_weight,
                    conservative_mechanism_support_floor=conservative_mechanism_support_floor,
                    conservative_mechanism_candidate_topk=conservative_mechanism_candidate_topk,
                    conservative_mechanism_support_mode=conservative_mechanism_support_mode,
                    source_gated_mechanism_weight=source_gated_mechanism_weight,
                    source_gated_mechanism_support_floor=source_gated_mechanism_support_floor,
                    source_gated_mechanism_candidate_topk=source_gated_mechanism_candidate_topk,
                    source_gated_mechanism_support_mode=source_gated_mechanism_support_mode,
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
                    root_score_channel_scores=root_score_channel_scores,
                    root_score_pooling=root_score_pooling,
                    root_score_head_ratio=root_score_head_ratio,
                    root_score_head_points=root_score_head_points,
                    root_score_top_quantile=root_score_top_quantile,
                    source_innovation_channel_scores=source_innovation_channel_scores,
                    synthetic_channel_scores=synthetic_channel_scores,
                    event_responsibility_channel_scores=event_responsibility_channel_scores,
                    event_responsibility_pooling=event_responsibility_pooling,
                    event_responsibility_head_ratio=event_responsibility_head_ratio,
                    event_responsibility_head_points=event_responsibility_head_points,
                    event_responsibility_top_quantile=event_responsibility_top_quantile,
                    event_route_channel_scores=event_route_channel_scores,
                    response_suppressor_channel_scores=response_suppressor_channel_scores,
                    evidence_fusion_channel_scores=evidence_fusion_channel_scores,
                    evidence_fusion_pooling=evidence_fusion_pooling,
                    evidence_fusion_head_ratio=evidence_fusion_head_ratio,
                    evidence_fusion_head_points=evidence_fusion_head_points,
                    evidence_fusion_top_quantile=evidence_fusion_top_quantile,
                    source_interaction_head_channel_scores=source_interaction_head_channel_scores,
                    source_interaction_head_pooling=source_interaction_head_pooling,
                    source_interaction_head_ratio=source_interaction_head_ratio,
                    source_interaction_head_points=source_interaction_head_points,
                    source_interaction_head_top_quantile=source_interaction_head_top_quantile,
                    source_consistency_head_channel_scores=source_consistency_head_channel_scores,
                    source_consistency_head_pooling=source_consistency_head_pooling,
                    source_consistency_head_ratio=source_consistency_head_ratio,
                    source_consistency_head_points=source_consistency_head_points,
                    source_consistency_head_top_quantile=source_consistency_head_top_quantile,
                    counterfactual_channel_scores=counterfactual_channel_scores,
                    use_source_propagation=use_source_propagation,
                    graph_weight=graph_weight,
                    mechanism_weight=mechanism_weight,
                    source_weight=source_weight,
                    source_base_weight=source_base_weight,
                    propagation_weight=propagation_weight,
                    source_mechanism_weight=source_mechanism_weight,
                    source_score_weight=source_score_weight,
                    source_consensus_weight=source_consensus_weight,
                    onset_consensus_weight=onset_consensus_weight,
                    source_consensus_mode=source_consensus_mode,
                    causal_weight=causal_weight,
                    synthetic_weight=synthetic_weight,
                    event_responsibility_weight=event_responsibility_weight,
                    event_route_weight=event_route_weight,
                    response_suppressor_weight=response_suppressor_weight,
                    response_suppressor_source_guard_mode=response_suppressor_source_guard_mode,
                    response_suppressor_source_guard_floor=response_suppressor_source_guard_floor,
                    evidence_fusion_weight=evidence_fusion_weight,
                    source_interaction_head_weight=source_interaction_head_weight,
                    source_consistency_head_weight=source_consistency_head_weight,
                    source_gate_weight=source_gate_weight,
                    source_gate_evidence_guard_weight=source_gate_evidence_guard_weight,
                    source_gate_evidence_guard_floor=source_gate_evidence_guard_floor,
                    root_score_weight=root_score_weight,
                    source_interaction_weight=source_interaction_weight,
                    source_innovation_weight=source_innovation_weight,
                    source_innovation_mode=source_innovation_mode,
                    source_innovation_neighbor_weight=source_innovation_neighbor_weight,
                    source_innovation_lead_points=source_innovation_lead_points,
                    mechanism_guided_source_weight=mechanism_guided_source_weight,
                    adaptive_mechanism_gate_weight=adaptive_mechanism_gate_weight,
                    adaptive_mechanism_gate_floor=adaptive_mechanism_gate_floor,
                    adaptive_mechanism_gate_mode=adaptive_mechanism_gate_mode,
                    adaptive_mechanism_selection_weight=adaptive_mechanism_selection_weight,
                    adaptive_mechanism_selection_floor=adaptive_mechanism_selection_floor,
                    adaptive_mechanism_selection_mode=adaptive_mechanism_selection_mode,
                    conservative_mechanism_weight=conservative_mechanism_weight,
                    conservative_mechanism_support_floor=conservative_mechanism_support_floor,
                    conservative_mechanism_candidate_topk=conservative_mechanism_candidate_topk,
                    conservative_mechanism_support_mode=conservative_mechanism_support_mode,
                    source_gated_mechanism_weight=source_gated_mechanism_weight,
                    source_gated_mechanism_support_floor=source_gated_mechanism_support_floor,
                    source_gated_mechanism_candidate_topk=source_gated_mechanism_candidate_topk,
                    source_gated_mechanism_support_mode=source_gated_mechanism_support_mode,
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
                root_score_channel_scores=root_score_channel_scores,
                root_score_pooling=root_score_pooling,
                root_score_head_ratio=root_score_head_ratio,
                root_score_head_points=root_score_head_points,
                root_score_top_quantile=root_score_top_quantile,
                source_innovation_channel_scores=source_innovation_channel_scores,
                synthetic_channel_scores=synthetic_channel_scores,
                event_responsibility_channel_scores=event_responsibility_channel_scores,
                event_responsibility_pooling=event_responsibility_pooling,
                event_responsibility_head_ratio=event_responsibility_head_ratio,
                event_responsibility_head_points=event_responsibility_head_points,
                event_responsibility_top_quantile=event_responsibility_top_quantile,
                event_route_channel_scores=event_route_channel_scores,
                response_suppressor_channel_scores=response_suppressor_channel_scores,
                evidence_fusion_channel_scores=evidence_fusion_channel_scores,
                evidence_fusion_pooling=evidence_fusion_pooling,
                evidence_fusion_head_ratio=evidence_fusion_head_ratio,
                evidence_fusion_head_points=evidence_fusion_head_points,
                evidence_fusion_top_quantile=evidence_fusion_top_quantile,
                source_interaction_head_channel_scores=source_interaction_head_channel_scores,
                source_interaction_head_pooling=source_interaction_head_pooling,
                source_interaction_head_ratio=source_interaction_head_ratio,
                source_interaction_head_points=source_interaction_head_points,
                source_interaction_head_top_quantile=source_interaction_head_top_quantile,
                source_consistency_head_channel_scores=source_consistency_head_channel_scores,
                source_consistency_head_pooling=source_consistency_head_pooling,
                source_consistency_head_ratio=source_consistency_head_ratio,
                source_consistency_head_points=source_consistency_head_points,
                source_consistency_head_top_quantile=source_consistency_head_top_quantile,
                counterfactual_channel_scores=counterfactual_channel_scores,
                use_source_propagation=use_source_propagation,
                graph_weight=graph_weight,
                mechanism_weight=mechanism_weight,
                source_weight=source_weight,
                source_base_weight=source_base_weight,
                propagation_weight=propagation_weight,
                source_mechanism_weight=source_mechanism_weight,
                source_score_weight=source_score_weight,
                source_consensus_weight=source_consensus_weight,
                onset_consensus_weight=onset_consensus_weight,
                source_consensus_mode=source_consensus_mode,
                causal_weight=causal_weight,
                synthetic_weight=synthetic_weight,
                event_responsibility_weight=event_responsibility_weight,
                event_route_weight=event_route_weight,
                response_suppressor_weight=response_suppressor_weight,
                response_suppressor_source_guard_mode=response_suppressor_source_guard_mode,
                response_suppressor_source_guard_floor=response_suppressor_source_guard_floor,
                evidence_fusion_weight=evidence_fusion_weight,
                source_interaction_head_weight=source_interaction_head_weight,
                source_consistency_head_weight=source_consistency_head_weight,
                source_gate_weight=source_gate_weight,
                source_gate_evidence_guard_weight=source_gate_evidence_guard_weight,
                source_gate_evidence_guard_floor=source_gate_evidence_guard_floor,
                root_score_weight=root_score_weight,
                source_interaction_weight=source_interaction_weight,
                source_innovation_weight=source_innovation_weight,
                source_innovation_mode=source_innovation_mode,
                source_innovation_neighbor_weight=source_innovation_neighbor_weight,
                source_innovation_lead_points=source_innovation_lead_points,
                mechanism_guided_source_weight=mechanism_guided_source_weight,
                adaptive_mechanism_gate_weight=adaptive_mechanism_gate_weight,
                adaptive_mechanism_gate_floor=adaptive_mechanism_gate_floor,
                adaptive_mechanism_gate_mode=adaptive_mechanism_gate_mode,
                adaptive_mechanism_selection_weight=adaptive_mechanism_selection_weight,
                adaptive_mechanism_selection_floor=adaptive_mechanism_selection_floor,
                adaptive_mechanism_selection_mode=adaptive_mechanism_selection_mode,
                conservative_mechanism_weight=conservative_mechanism_weight,
                conservative_mechanism_support_floor=conservative_mechanism_support_floor,
                conservative_mechanism_candidate_topk=conservative_mechanism_candidate_topk,
                conservative_mechanism_support_mode=conservative_mechanism_support_mode,
                source_gated_mechanism_weight=source_gated_mechanism_weight,
                source_gated_mechanism_support_floor=source_gated_mechanism_support_floor,
                source_gated_mechanism_candidate_topk=source_gated_mechanism_candidate_topk,
                source_gated_mechanism_support_mode=source_gated_mechanism_support_mode,
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
        if event_specificity_weight > 0.0 and predicted_events_by_key:
            for key_name, key_events in list(predicted_events_by_key.items()):
                predicted_events_by_key[key_name] = self._apply_rca_event_specificity_suppression(
                    key_events,
                    weight=event_specificity_weight,
                    top_k=event_specificity_top_k,
                    threshold=event_specificity_threshold,
                    min_events=event_specificity_min_events,
                    source_guard_weight=event_specificity_source_guard_weight,
                    source_guard_floor=event_specificity_source_guard_floor,
                    keep_top_k=event_specificity_keep_top_k,
                    fill_secondary_top_k=event_specificity_fill_secondary_top_k,
                    secondary_base_weight=event_specificity_secondary_base_weight,
                    secondary_onset_weight=event_specificity_secondary_onset_weight,
                    secondary_source_gate_weight=event_specificity_secondary_source_gate_weight,
                    secondary_mechanism_residual_weight=(
                        event_specificity_secondary_mechanism_residual_weight
                    ),
                    secondary_source_interaction_weight=(
                        event_specificity_secondary_source_interaction_weight
                    ),
                    secondary_source_consistency_head_weight=(
                        event_specificity_secondary_source_consistency_head_weight
                    ),
                    secondary_evidence_fusion_weight=(
                        event_specificity_secondary_evidence_fusion_weight
                    ),
                )
        if adaptive_evidence_rerank and predicted_events_by_key:
            for key_name, key_events in list(predicted_events_by_key.items()):
                predicted_events_by_key[key_name] = self._apply_rca_adaptive_evidence_rerank(
                    key_events,
                    enabled=adaptive_evidence_rerank,
                    top_k=adaptive_evidence_top_k,
                    keep_top_k=adaptive_evidence_keep_top_k,
                    fill_top_k=adaptive_evidence_fill_top_k,
                    gate=adaptive_evidence_gate,
                    source_old_rank_threshold=adaptive_evidence_source_old_rank_threshold,
                    source_margin_threshold=adaptive_evidence_source_margin_threshold,
                    source_base_weight=adaptive_evidence_source_base_weight,
                    source_onset_weight=adaptive_evidence_source_onset_weight,
                    source_gate_weight=adaptive_evidence_source_gate_weight,
                    source_mechanism_residual_weight=(
                        adaptive_evidence_source_mechanism_residual_weight
                    ),
                    source_interaction_weight=adaptive_evidence_source_interaction_weight,
                    fallback_original_weight=adaptive_evidence_fallback_original_weight,
                    fallback_base_weight=adaptive_evidence_fallback_base_weight,
                    fallback_onset_weight=adaptive_evidence_fallback_onset_weight,
                    fallback_mechanism_residual_weight=(
                        adaptive_evidence_fallback_mechanism_residual_weight
                    ),
                    fallback_root_weight=adaptive_evidence_fallback_root_weight,
                    fallback_event_weight=adaptive_evidence_fallback_event_weight,
                    fallback_evidence_fusion_weight=(
                        adaptive_evidence_fallback_evidence_fusion_weight
                    ),
                    fallback_source_interaction_weight=(
                        adaptive_evidence_fallback_source_interaction_weight
                    ),
                    fallback_source_consistency_weight=(
                        adaptive_evidence_fallback_source_consistency_weight
                    ),
                    fallback_graph_low_weight=adaptive_evidence_fallback_graph_low_weight,
                    fallback_response_suppressor_low_weight=(
                        adaptive_evidence_fallback_response_suppressor_low_weight
                    ),
                )
        if predicted_events_by_key:
            for key_name, key_events in list(predicted_events_by_key.items()):
                predicted_events_by_key[key_name] = self._refresh_rca_export_scores(
                    key_events,
                    export_responsibility_score=export_responsibility_score,
                    export_score_distribution=export_score_distribution,
                    export_score_distribution_top_m=export_score_distribution_top_m,
                    export_score_distribution_power=export_score_distribution_power,
                    export_score_distribution_tau=export_score_distribution_tau,
                    export_responsibility_ranking_weight=(
                        export_responsibility_ranking_weight
                    ),
                    export_responsibility_pre_weight=export_responsibility_pre_weight,
                    export_responsibility_base_weight=export_responsibility_base_weight,
                    export_responsibility_source_gate_weight=(
                        export_responsibility_source_gate_weight
                    ),
                    export_responsibility_onset_weight=(
                        export_responsibility_onset_weight
                    ),
                    export_responsibility_mechanism_residual_weight=(
                        export_responsibility_mechanism_residual_weight
                    ),
                    export_responsibility_root_weight=(
                        export_responsibility_root_weight
                    ),
                    export_responsibility_event_weight=(
                        export_responsibility_event_weight
                    ),
                    export_responsibility_evidence_fusion_weight=(
                        export_responsibility_evidence_fusion_weight
                    ),
                    export_responsibility_source_interaction_weight=(
                        export_responsibility_source_interaction_weight
                    ),
                    export_responsibility_graph_low_weight=(
                        export_responsibility_graph_low_weight
                    ),
                    export_responsibility_response_suppressor_low_weight=(
                        export_responsibility_response_suppressor_low_weight
                    ),
                    responsibility_rerank=responsibility_rerank,
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
                root_score_channel_scores=root_score_channel_scores,
                root_score_pooling=root_score_pooling,
                root_score_head_ratio=root_score_head_ratio,
                root_score_head_points=root_score_head_points,
                root_score_top_quantile=root_score_top_quantile,
                source_innovation_channel_scores=source_innovation_channel_scores,
                synthetic_channel_scores=synthetic_channel_scores,
                event_responsibility_channel_scores=event_responsibility_channel_scores,
                event_responsibility_pooling=event_responsibility_pooling,
                event_responsibility_head_ratio=event_responsibility_head_ratio,
                event_responsibility_head_points=event_responsibility_head_points,
                event_responsibility_top_quantile=event_responsibility_top_quantile,
                event_route_channel_scores=event_route_channel_scores,
                response_suppressor_channel_scores=response_suppressor_channel_scores,
                evidence_fusion_channel_scores=evidence_fusion_channel_scores,
                evidence_fusion_pooling=evidence_fusion_pooling,
                evidence_fusion_head_ratio=evidence_fusion_head_ratio,
                evidence_fusion_head_points=evidence_fusion_head_points,
                evidence_fusion_top_quantile=evidence_fusion_top_quantile,
                source_interaction_head_channel_scores=source_interaction_head_channel_scores,
                source_interaction_head_pooling=source_interaction_head_pooling,
                source_interaction_head_ratio=source_interaction_head_ratio,
                source_interaction_head_points=source_interaction_head_points,
                source_interaction_head_top_quantile=source_interaction_head_top_quantile,
                source_consistency_head_channel_scores=source_consistency_head_channel_scores,
                source_consistency_head_pooling=source_consistency_head_pooling,
                source_consistency_head_ratio=source_consistency_head_ratio,
                source_consistency_head_points=source_consistency_head_points,
                source_consistency_head_top_quantile=source_consistency_head_top_quantile,
                counterfactual_channel_scores=counterfactual_channel_scores,
                use_source_propagation=use_source_propagation,
                graph_weight=graph_weight,
                mechanism_weight=mechanism_weight,
                source_weight=source_weight,
                source_base_weight=source_base_weight,
                propagation_weight=propagation_weight,
                source_mechanism_weight=source_mechanism_weight,
                source_score_weight=source_score_weight,
                source_consensus_weight=source_consensus_weight,
                onset_consensus_weight=onset_consensus_weight,
                source_consensus_mode=source_consensus_mode,
                causal_weight=causal_weight,
                synthetic_weight=synthetic_weight,
                event_responsibility_weight=event_responsibility_weight,
                event_route_weight=event_route_weight,
                response_suppressor_weight=response_suppressor_weight,
                response_suppressor_source_guard_mode=response_suppressor_source_guard_mode,
                response_suppressor_source_guard_floor=response_suppressor_source_guard_floor,
                evidence_fusion_weight=evidence_fusion_weight,
                source_interaction_head_weight=source_interaction_head_weight,
                source_consistency_head_weight=source_consistency_head_weight,
                source_gate_weight=source_gate_weight,
                source_gate_evidence_guard_weight=source_gate_evidence_guard_weight,
                source_gate_evidence_guard_floor=source_gate_evidence_guard_floor,
                root_score_weight=root_score_weight,
                source_interaction_weight=source_interaction_weight,
                source_innovation_weight=source_innovation_weight,
                source_innovation_mode=source_innovation_mode,
                source_innovation_neighbor_weight=source_innovation_neighbor_weight,
                source_innovation_lead_points=source_innovation_lead_points,
                mechanism_guided_source_weight=mechanism_guided_source_weight,
                adaptive_mechanism_gate_weight=adaptive_mechanism_gate_weight,
                adaptive_mechanism_gate_floor=adaptive_mechanism_gate_floor,
                adaptive_mechanism_gate_mode=adaptive_mechanism_gate_mode,
                adaptive_mechanism_selection_weight=adaptive_mechanism_selection_weight,
                adaptive_mechanism_selection_floor=adaptive_mechanism_selection_floor,
                adaptive_mechanism_selection_mode=adaptive_mechanism_selection_mode,
                conservative_mechanism_weight=conservative_mechanism_weight,
                conservative_mechanism_support_floor=conservative_mechanism_support_floor,
                conservative_mechanism_candidate_topk=conservative_mechanism_candidate_topk,
                conservative_mechanism_support_mode=conservative_mechanism_support_mode,
                source_gated_mechanism_weight=source_gated_mechanism_weight,
                source_gated_mechanism_support_floor=source_gated_mechanism_support_floor,
                source_gated_mechanism_candidate_topk=source_gated_mechanism_candidate_topk,
                source_gated_mechanism_support_mode=source_gated_mechanism_support_mode,
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
            if event_specificity_weight > 0.0:
                predicted_events = self._apply_rca_event_specificity_suppression(
                    predicted_events,
                    weight=event_specificity_weight,
                    top_k=event_specificity_top_k,
                    threshold=event_specificity_threshold,
                    min_events=event_specificity_min_events,
                    source_guard_weight=event_specificity_source_guard_weight,
                    source_guard_floor=event_specificity_source_guard_floor,
                    keep_top_k=event_specificity_keep_top_k,
                    fill_secondary_top_k=event_specificity_fill_secondary_top_k,
                    secondary_base_weight=event_specificity_secondary_base_weight,
                    secondary_onset_weight=event_specificity_secondary_onset_weight,
                    secondary_source_gate_weight=event_specificity_secondary_source_gate_weight,
                    secondary_mechanism_residual_weight=(
                        event_specificity_secondary_mechanism_residual_weight
                    ),
                    secondary_source_interaction_weight=(
                        event_specificity_secondary_source_interaction_weight
                    ),
                    secondary_source_consistency_head_weight=(
                        event_specificity_secondary_source_consistency_head_weight
                    ),
                    secondary_evidence_fusion_weight=(
                        event_specificity_secondary_evidence_fusion_weight
                    ),
                )
            if adaptive_evidence_rerank:
                predicted_events = self._apply_rca_adaptive_evidence_rerank(
                    predicted_events,
                    enabled=adaptive_evidence_rerank,
                    top_k=adaptive_evidence_top_k,
                    keep_top_k=adaptive_evidence_keep_top_k,
                    fill_top_k=adaptive_evidence_fill_top_k,
                    gate=adaptive_evidence_gate,
                    source_old_rank_threshold=adaptive_evidence_source_old_rank_threshold,
                    source_margin_threshold=adaptive_evidence_source_margin_threshold,
                    source_base_weight=adaptive_evidence_source_base_weight,
                    source_onset_weight=adaptive_evidence_source_onset_weight,
                    source_gate_weight=adaptive_evidence_source_gate_weight,
                    source_mechanism_residual_weight=(
                        adaptive_evidence_source_mechanism_residual_weight
                    ),
                    source_interaction_weight=adaptive_evidence_source_interaction_weight,
                fallback_original_weight=adaptive_evidence_fallback_original_weight,
                fallback_base_weight=adaptive_evidence_fallback_base_weight,
                fallback_onset_weight=adaptive_evidence_fallback_onset_weight,
                fallback_mechanism_residual_weight=(
                    adaptive_evidence_fallback_mechanism_residual_weight
                ),
                fallback_root_weight=adaptive_evidence_fallback_root_weight,
                fallback_event_weight=adaptive_evidence_fallback_event_weight,
                fallback_evidence_fusion_weight=(
                    adaptive_evidence_fallback_evidence_fusion_weight
                ),
                fallback_source_interaction_weight=(
                    adaptive_evidence_fallback_source_interaction_weight
                ),
                fallback_source_consistency_weight=(
                    adaptive_evidence_fallback_source_consistency_weight
                ),
                fallback_graph_low_weight=adaptive_evidence_fallback_graph_low_weight,
                fallback_response_suppressor_low_weight=(
                    adaptive_evidence_fallback_response_suppressor_low_weight
                ),
            )
            predicted_events = self._refresh_rca_export_scores(
                predicted_events,
                export_responsibility_score=export_responsibility_score,
                export_score_distribution=export_score_distribution,
                export_score_distribution_top_m=export_score_distribution_top_m,
                export_score_distribution_power=export_score_distribution_power,
                export_score_distribution_tau=export_score_distribution_tau,
                export_responsibility_ranking_weight=export_responsibility_ranking_weight,
                export_responsibility_pre_weight=export_responsibility_pre_weight,
                export_responsibility_base_weight=export_responsibility_base_weight,
                export_responsibility_source_gate_weight=(
                    export_responsibility_source_gate_weight
                ),
                export_responsibility_onset_weight=export_responsibility_onset_weight,
                export_responsibility_mechanism_residual_weight=(
                    export_responsibility_mechanism_residual_weight
                ),
                export_responsibility_root_weight=export_responsibility_root_weight,
                export_responsibility_event_weight=export_responsibility_event_weight,
                export_responsibility_evidence_fusion_weight=(
                    export_responsibility_evidence_fusion_weight
                ),
                export_responsibility_source_interaction_weight=(
                    export_responsibility_source_interaction_weight
                ),
                export_responsibility_graph_low_weight=(
                    export_responsibility_graph_low_weight
                ),
                export_responsibility_response_suppressor_low_weight=(
                    export_responsibility_response_suppressor_low_weight
                ),
                responsibility_rerank=responsibility_rerank,
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
            "source=(weighted base residual + learned RootScore + fused source score + learned source-consistency head + consensus-gated source evidence + mechanism prior deviation + source innovation + mechanism-guided source interaction + lagged causal deviation + model source-gate score + synthetic responsibility + event-local counterfactual responsibility), "
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
            "rca_source_score_weight": source_score_weight,
            "rca_source_consensus_weight": source_consensus_weight,
            "rca_onset_consensus_weight": onset_consensus_weight,
            "rca_source_consensus_mode": source_consensus_mode,
            "rca_causal_weight": causal_weight,
            "rca_synthetic_weight": synthetic_weight,
            "rca_source_gate_weight": source_gate_weight,
            "rca_source_gate_evidence_guard_weight": (
                source_gate_evidence_guard_weight
            ),
            "rca_source_gate_evidence_guard_floor": (
                source_gate_evidence_guard_floor
            ),
            "rca_root_score_weight": root_score_weight,
            "rca_root_score_signal": root_score_signal,
            "rca_root_score_pooling": root_score_pooling,
            "rca_root_score_head_ratio": root_score_head_ratio,
            "rca_root_score_head_points": root_score_head_points,
            "rca_root_score_top_quantile": root_score_top_quantile,
            "rca_source_interaction_weight": source_interaction_weight,
            "rca_source_interaction_head_weight": source_interaction_head_weight,
            "rca_source_interaction_head_pooling": source_interaction_head_pooling,
            "rca_source_interaction_head_ratio": source_interaction_head_ratio,
            "rca_source_interaction_head_points": source_interaction_head_points,
            "rca_source_interaction_head_top_quantile": (
                source_interaction_head_top_quantile
            ),
            "rca_source_consistency_head_weight": source_consistency_head_weight,
            "rca_source_consistency_head_pooling": source_consistency_head_pooling,
            "rca_source_consistency_head_ratio": source_consistency_head_ratio,
            "rca_source_consistency_head_points": source_consistency_head_points,
            "rca_source_consistency_head_top_quantile": (
                source_consistency_head_top_quantile
            ),
            "rca_source_innovation_weight": source_innovation_weight,
            "rca_source_innovation_mode": source_innovation_mode,
            "rca_source_innovation_neighbor_weight": source_innovation_neighbor_weight,
            "rca_source_innovation_lead_points": source_innovation_lead_points,
            "rca_mechanism_guided_source_weight": mechanism_guided_source_weight,
            "rca_adaptive_mechanism_gate_weight": adaptive_mechanism_gate_weight,
            "rca_adaptive_mechanism_gate_floor": adaptive_mechanism_gate_floor,
            "rca_adaptive_mechanism_gate_mode": adaptive_mechanism_gate_mode,
            "rca_adaptive_mechanism_selection_weight": adaptive_mechanism_selection_weight,
            "rca_adaptive_mechanism_selection_floor": adaptive_mechanism_selection_floor,
            "rca_adaptive_mechanism_selection_mode": adaptive_mechanism_selection_mode,
            "rca_conservative_mechanism_weight": conservative_mechanism_weight,
            "rca_conservative_mechanism_support_floor": conservative_mechanism_support_floor,
            "rca_conservative_mechanism_candidate_topk": conservative_mechanism_candidate_topk,
            "rca_conservative_mechanism_support_mode": conservative_mechanism_support_mode,
            "rca_source_gated_mechanism_weight": source_gated_mechanism_weight,
            "rca_source_gated_mechanism_support_floor": source_gated_mechanism_support_floor,
            "rca_source_gated_mechanism_candidate_topk": source_gated_mechanism_candidate_topk,
            "rca_source_gated_mechanism_support_mode": source_gated_mechanism_support_mode,
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
            "rca_event_base_weight": event_base_weight,
            "rca_export_score_distribution": export_score_distribution,
            "rca_export_score_distribution_top_m": export_score_distribution_top_m,
            "rca_export_score_distribution_power": export_score_distribution_power,
            "rca_export_score_distribution_tau": export_score_distribution_tau,
            "rca_export_responsibility_score": export_responsibility_score,
            "rca_export_responsibility_ranking_weight": (
                export_responsibility_ranking_weight
            ),
            "rca_export_responsibility_pre_weight": export_responsibility_pre_weight,
            "rca_export_responsibility_base_weight": export_responsibility_base_weight,
            "rca_export_responsibility_source_gate_weight": (
                export_responsibility_source_gate_weight
            ),
            "rca_export_responsibility_onset_weight": (
                export_responsibility_onset_weight
            ),
            "rca_export_responsibility_mechanism_residual_weight": (
                export_responsibility_mechanism_residual_weight
            ),
            "rca_export_responsibility_root_weight": (
                export_responsibility_root_weight
            ),
            "rca_export_responsibility_event_weight": (
                export_responsibility_event_weight
            ),
            "rca_export_responsibility_evidence_fusion_weight": (
                export_responsibility_evidence_fusion_weight
            ),
            "rca_export_responsibility_source_interaction_weight": (
                export_responsibility_source_interaction_weight
            ),
            "rca_export_responsibility_graph_low_weight": (
                export_responsibility_graph_low_weight
            ),
            "rca_export_responsibility_response_suppressor_low_weight": (
                export_responsibility_response_suppressor_low_weight
            ),
            "rca_responsibility_rerank": responsibility_rerank,
            "rca_graph_penalty_weight": graph_penalty_weight,
            "rca_response_suppressor_weight": response_suppressor_weight,
            "rca_response_suppressor_source_guard_mode": (
                response_suppressor_source_guard_mode
            ),
            "rca_response_suppressor_source_guard_floor": (
                response_suppressor_source_guard_floor
            ),
            "rca_hierarchical_mode": hierarchical_mode,
            "rca_hierarchical_group_topk": hierarchical_group_topk,
            "rca_hierarchical_group_boost": hierarchical_group_boost,
            "rca_hierarchical_outside_penalty": hierarchical_outside_penalty,
            "rca_hierarchical_group_aggregation": hierarchical_group_aggregation,
            "rca_event_specificity_weight": event_specificity_weight,
            "rca_event_specificity_top_k": event_specificity_top_k,
            "rca_event_specificity_threshold": event_specificity_threshold,
            "rca_event_specificity_min_events": event_specificity_min_events,
            "rca_event_specificity_source_guard_weight": event_specificity_source_guard_weight,
            "rca_event_specificity_source_guard_floor": event_specificity_source_guard_floor,
            "rca_event_specificity_keep_top_k": event_specificity_keep_top_k,
            "rca_event_specificity_fill_secondary_top_k": event_specificity_fill_secondary_top_k,
            "rca_event_specificity_secondary_base_weight": event_specificity_secondary_base_weight,
            "rca_event_specificity_secondary_onset_weight": event_specificity_secondary_onset_weight,
            "rca_event_specificity_secondary_source_gate_weight": (
                event_specificity_secondary_source_gate_weight
            ),
            "rca_event_specificity_secondary_mechanism_residual_weight": (
                event_specificity_secondary_mechanism_residual_weight
            ),
            "rca_event_specificity_secondary_source_interaction_weight": (
                event_specificity_secondary_source_interaction_weight
            ),
            "rca_event_specificity_secondary_source_consistency_head_weight": (
                event_specificity_secondary_source_consistency_head_weight
            ),
            "rca_event_specificity_secondary_evidence_fusion_weight": (
                event_specificity_secondary_evidence_fusion_weight
            ),
            "rca_adaptive_evidence_rerank": adaptive_evidence_rerank,
            "rca_adaptive_evidence_top_k": adaptive_evidence_top_k,
            "rca_adaptive_evidence_keep_top_k": adaptive_evidence_keep_top_k,
            "rca_adaptive_evidence_fill_top_k": adaptive_evidence_fill_top_k,
            "rca_adaptive_evidence_gate": adaptive_evidence_gate,
            "rca_adaptive_evidence_source_old_rank_threshold": (
                adaptive_evidence_source_old_rank_threshold
            ),
            "rca_adaptive_evidence_source_margin_threshold": (
                adaptive_evidence_source_margin_threshold
            ),
            "rca_adaptive_evidence_source_base_weight": adaptive_evidence_source_base_weight,
            "rca_adaptive_evidence_source_onset_weight": adaptive_evidence_source_onset_weight,
            "rca_adaptive_evidence_source_gate_weight": adaptive_evidence_source_gate_weight,
            "rca_adaptive_evidence_source_mechanism_residual_weight": (
                adaptive_evidence_source_mechanism_residual_weight
            ),
            "rca_adaptive_evidence_source_interaction_weight": (
                adaptive_evidence_source_interaction_weight
            ),
            "rca_adaptive_evidence_fallback_original_weight": (
                adaptive_evidence_fallback_original_weight
            ),
            "rca_adaptive_evidence_fallback_base_weight": adaptive_evidence_fallback_base_weight,
            "rca_adaptive_evidence_fallback_onset_weight": adaptive_evidence_fallback_onset_weight,
            "rca_adaptive_evidence_fallback_mechanism_residual_weight": (
                adaptive_evidence_fallback_mechanism_residual_weight
            ),
            "rca_adaptive_evidence_fallback_root_weight": (
                adaptive_evidence_fallback_root_weight
            ),
            "rca_adaptive_evidence_fallback_event_weight": (
                adaptive_evidence_fallback_event_weight
            ),
            "rca_adaptive_evidence_fallback_evidence_fusion_weight": (
                adaptive_evidence_fallback_evidence_fusion_weight
            ),
            "rca_adaptive_evidence_fallback_source_interaction_weight": (
                adaptive_evidence_fallback_source_interaction_weight
            ),
            "rca_adaptive_evidence_fallback_source_consistency_weight": (
                adaptive_evidence_fallback_source_consistency_weight
            ),
            "rca_adaptive_evidence_fallback_graph_low_weight": (
                adaptive_evidence_fallback_graph_low_weight
            ),
            "rca_adaptive_evidence_fallback_response_suppressor_low_weight": (
                adaptive_evidence_fallback_response_suppressor_low_weight
            ),
            "rca_topk_rerank": bool(getattr(self.config, "rca_topk_rerank", False)),
            "rca_topk_rerank_k": _cfg_int("rca_topk_rerank_k", 5),
            "rca_topk_rerank_original_weight": _cfg_float(
                "rca_topk_rerank_original_weight",
                1.0,
            ),
            "rca_topk_rerank_base_weight": _cfg_float("rca_topk_rerank_base_weight", 0.0),
            "rca_topk_rerank_group_weight": _cfg_float("rca_topk_rerank_group_weight", 0.0),
            "rca_topk_rerank_onset_weight": _cfg_float("rca_topk_rerank_onset_weight", 0.0),
            "rca_topk_rerank_source_gate_weight": _cfg_float(
                "rca_topk_rerank_source_gate_weight",
                0.0,
            ),
            "rca_topk_rerank_mechanism_residual_weight": _cfg_float(
                "rca_topk_rerank_mechanism_residual_weight",
                0.0,
            ),
            "rca_topk_rerank_source_interaction_weight": _cfg_float(
                "rca_topk_rerank_source_interaction_weight",
                0.0,
            ),
            "rca_topk_rerank_source_consistency_head_weight": _cfg_float(
                "rca_topk_rerank_source_consistency_head_weight",
                0.0,
            ),
            "rca_topk_rerank_graph_penalty_weight": _cfg_float(
                "rca_topk_rerank_graph_penalty_weight",
                0.0,
            ),
            "rca_topk_rerank_component_scope": str(
                getattr(self.config, "rca_topk_rerank_component_scope", "candidate") or "candidate"
            ),
            "rca_topk_rerank_keep_primary_top_k": _cfg_int(
                "rca_topk_rerank_keep_primary_top_k",
                0,
            ),
            "rca_topk_rerank_fill_secondary_top_k": _cfg_int(
                "rca_topk_rerank_fill_secondary_top_k",
                0,
            ),
            "rca_topk_rerank_secondary_original_weight": _cfg_float(
                "rca_topk_rerank_secondary_original_weight",
                0.0,
            ),
            "rca_topk_rerank_secondary_base_weight": _cfg_float(
                "rca_topk_rerank_secondary_base_weight",
                0.0,
            ),
            "rca_topk_rerank_secondary_group_weight": _cfg_float(
                "rca_topk_rerank_secondary_group_weight",
                0.0,
            ),
            "rca_topk_rerank_secondary_onset_weight": _cfg_float(
                "rca_topk_rerank_secondary_onset_weight",
                0.0,
            ),
            "rca_topk_rerank_secondary_source_gate_weight": _cfg_float(
                "rca_topk_rerank_secondary_source_gate_weight",
                0.0,
            ),
            "rca_topk_rerank_secondary_mechanism_residual_weight": _cfg_float(
                "rca_topk_rerank_secondary_mechanism_residual_weight",
                0.0,
            ),
            "rca_topk_rerank_secondary_source_interaction_weight": _cfg_float(
                "rca_topk_rerank_secondary_source_interaction_weight",
                0.0,
            ),
            "rca_topk_rerank_secondary_source_consistency_head_weight": _cfg_float(
                "rca_topk_rerank_secondary_source_consistency_head_weight",
                0.0,
            ),
            "rca_topk_rerank_secondary_graph_penalty_weight": _cfg_float(
                "rca_topk_rerank_secondary_graph_penalty_weight",
                0.0,
            ),
            "rca_event_head_ratio": event_head_ratio,
            "rca_event_head_points": event_head_points,
            "rca_align_event_onset": bool(getattr(self.config, "rca_align_event_onset", False)),
            "rca_align_event_onset_baseline_window": _cfg_int("rca_align_event_onset_baseline_window", 300),
            "rca_align_event_onset_z": _cfg_float("rca_align_event_onset_z", 2.0),
            "rca_align_event_onset_quantile": _cfg_float("rca_align_event_onset_quantile", 0.90),
            "rca_onset_weight": onset_weight,
            "rca_onset_baseline_window": onset_baseline_window,
            "rca_onset_z": onset_z,
            "rca_prediction_key": pred_key,
            "rca_split_predicted_events": bool(getattr(self.config, "rca_split_predicted_events", False)),
            "rca_split_max_event_len": _cfg_int("rca_split_max_event_len", 120),
            "rca_split_stride": _cfg_int("rca_split_stride", 60),
            "use_source_effect_synthetic": bool(getattr(self.config, "use_source_effect_synthetic", False)),
            "lambda_source_effect": _cfg_float("lambda_source_effect", 0.0),
            "source_effect_interval": _cfg_int("source_effect_interval", 4),
            "source_effect_use_channel_prior": bool(
                getattr(self.config, "source_effect_use_channel_prior", False)
            ),
            "source_effect_prior_topk": _cfg_int("source_effect_prior_topk", 5),
            "source_effect_onset_rank_weight": _cfg_float("source_effect_onset_rank_weight", 0.0),
            "source_effect_graph_alignment_weight": _cfg_float(
                "source_effect_graph_alignment_weight", 0.0
            ),
            "source_effect_graph_reverse_weight": _cfg_float(
                "source_effect_graph_reverse_weight", 0.5
            ),
            "source_effect_consistency_weight": _cfg_float(
                "source_effect_consistency_weight",
                0.0,
            ),
            "source_effect_consistency_rank_weight": _cfg_float(
                "source_effect_consistency_rank_weight",
                0.75,
            ),
            "source_effect_consistency_effect_rank_weight": _cfg_float(
                "source_effect_consistency_effect_rank_weight",
                0.75,
            ),
            "source_effect_consistency_effect_suppress_weight": _cfg_float(
                "source_effect_consistency_effect_suppress_weight",
                0.25,
            ),
            "source_effect_consistency_gate_align_weight": _cfg_float(
                "source_effect_consistency_gate_align_weight",
                0.20,
            ),
            "source_effect_consistency_margin": _cfg_float(
                "source_effect_consistency_margin",
                0.15,
            ),
            "source_effect_specificity_weight": _cfg_float("source_effect_specificity_weight", 0.0),
            "lambda_source_effect_rca_head": _cfg_float("lambda_source_effect_rca_head", 0.0),
            "source_effect_rca_head_bce_weight": _cfg_float(
                "source_effect_rca_head_bce_weight",
                1.0,
            ),
            "source_effect_rca_head_rank_weight": _cfg_float(
                "source_effect_rca_head_rank_weight",
                1.0,
            ),
            "use_source_gate": bool(getattr(self.config, "use_source_gate", False)),
            "source_gate_init": _cfg_float("source_gate_init", 0.20),
            "use_onset_aware_source_gate": bool(
                getattr(self.config, "use_onset_aware_source_gate", False)
            ),
            "source_gate_onset_window": _cfg_int("source_gate_onset_window", 8),
            "source_gate_onset_weight": _cfg_float("source_gate_onset_weight", 0.5),
            "use_root_score_head": bool(getattr(self.config, "use_root_score_head", False)),
            "lambda_source_effect_root_score": _cfg_float("lambda_source_effect_root_score", 0.0),
            "root_score_bce_weight": _cfg_float("root_score_bce_weight", 1.0),
            "root_score_rank_weight": _cfg_float("root_score_rank_weight", 1.0),
            "root_score_effect_suppress_weight": _cfg_float(
                "root_score_effect_suppress_weight",
                0.5,
            ),
            "root_response_source_branch_weight": _cfg_float(
                "root_response_source_branch_weight",
                0.0,
            ),
            "root_response_confidence_discount": _cfg_float(
                "root_response_confidence_discount",
                0.75,
            ),
            "root_response_use_innovation_split": bool(
                getattr(self.config, "root_response_use_innovation_split", False)
            ),
            "use_pairwise_root_response_head": bool(
                getattr(self.config, "use_pairwise_root_response_head", False)
            ),
            "pairwise_root_response_rank": _cfg_int("pairwise_root_response_rank", 8),
            "pairwise_root_response_graph_weight": _cfg_float(
                "pairwise_root_response_graph_weight",
                0.75,
            ),
            "pairwise_root_response_reward_init": _cfg_float(
                "pairwise_root_response_reward_init",
                0.25,
            ),
            "pairwise_root_response_penalty_init": _cfg_float(
                "pairwise_root_response_penalty_init",
                0.50,
            ),
            "pairwise_root_response_logit_weight": _cfg_float(
                "pairwise_root_response_logit_weight",
                1.0,
            ),
            "lambda_pairwise_root_response": _cfg_float(
                "lambda_pairwise_root_response",
                0.0,
            ),
            "root_response_response_branch_weight": _cfg_float(
                "root_response_response_branch_weight",
                0.0,
            ),
            "root_response_response_suppress_weight": _cfg_float(
                "root_response_response_suppress_weight",
                0.5,
            ),
            "use_event_responsibility_head": bool(
                getattr(self.config, "use_event_responsibility_head", False)
            ),
            "lambda_event_responsibility": _cfg_float("lambda_event_responsibility", 0.0),
            "event_responsibility_loss_mode": str(
                getattr(self.config, "event_responsibility_loss_mode", "bce") or "bce"
            ),
            "event_responsibility_bce_balance": bool(
                getattr(self.config, "event_responsibility_bce_balance", True)
            ),
            "event_responsibility_ce_weight": _cfg_float(
                "event_responsibility_ce_weight",
                1.0,
            ),
            "event_responsibility_rank_weight": _cfg_float(
                "event_responsibility_rank_weight",
                0.5,
            ),
            "event_responsibility_effect_suppress_weight": _cfg_float(
                "event_responsibility_effect_suppress_weight",
                0.25,
            ),
            "use_event_route_head": bool(getattr(self.config, "use_event_route_head", False)),
            "event_route_hidden": _cfg_int("event_route_hidden", 16),
            "event_route_init": _cfg_float("event_route_init", 0.55),
            "event_route_detach_inputs": bool(
                getattr(self.config, "event_route_detach_inputs", True)
            ),
            "event_route_response_penalty": _cfg_float(
                "event_route_response_penalty",
                0.0,
            ),
            "lambda_event_route": _cfg_float("lambda_event_route", 0.0),
            "event_route_rank_weight": _cfg_float("event_route_rank_weight", 1.0),
            "event_route_effect_suppress_weight": _cfg_float(
                "event_route_effect_suppress_weight",
                0.25,
            ),
            "event_route_pairwise_effect_weight": _cfg_float(
                "event_route_pairwise_effect_weight",
                0.50,
            ),
            "event_route_pairwise_effect_margin": _cfg_float(
                "event_route_pairwise_effect_margin",
                0.15,
            ),
            "event_route_alpha_weight": _cfg_float("event_route_alpha_weight", 0.25),
            "use_response_suppressor_head": bool(
                getattr(self.config, "use_response_suppressor_head", False)
            ),
            "response_suppressor_hidden": _cfg_int("response_suppressor_hidden", 16),
            "response_suppressor_detach_inputs": bool(
                getattr(self.config, "response_suppressor_detach_inputs", True)
            ),
            "lambda_response_suppressor": _cfg_float("lambda_response_suppressor", 0.0),
            "response_suppressor_bce_weight": _cfg_float(
                "response_suppressor_bce_weight",
                1.0,
            ),
            "response_suppressor_rank_weight": _cfg_float(
                "response_suppressor_rank_weight",
                0.5,
            ),
            "response_suppressor_source_leak_weight": _cfg_float(
                "response_suppressor_source_leak_weight",
                0.5,
            ),
            "rca_event_responsibility_weight": event_responsibility_weight,
            "rca_event_route_weight": event_route_weight,
            "rca_event_responsibility_pooling": event_responsibility_pooling,
            "rca_event_responsibility_head_ratio": event_responsibility_head_ratio,
            "rca_event_responsibility_head_points": event_responsibility_head_points,
            "rca_event_responsibility_top_quantile": event_responsibility_top_quantile,
            "use_evidence_fusion_head": bool(
                getattr(self.config, "use_evidence_fusion_head", False)
            ),
            "evidence_fusion_hidden": _cfg_int("evidence_fusion_hidden", 16),
            "evidence_fusion_detach_inputs": bool(
                getattr(self.config, "evidence_fusion_detach_inputs", False)
            ),
            "evidence_fusion_use_graph_response": bool(
                getattr(self.config, "evidence_fusion_use_graph_response", False)
            ),
            "lambda_evidence_fusion": _cfg_float("lambda_evidence_fusion", 0.0),
            "evidence_fusion_loss_mode": str(
                getattr(self.config, "evidence_fusion_loss_mode", "bce") or "bce"
            ),
            "evidence_fusion_export_mode": str(
                getattr(self.config, "evidence_fusion_export_mode", "sigmoid")
                or "sigmoid"
            ),
            "evidence_fusion_bce_weight": _cfg_float(
                "evidence_fusion_bce_weight",
                1.0,
            ),
            "evidence_fusion_rank_weight": _cfg_float(
                "evidence_fusion_rank_weight",
                1.0,
            ),
            "evidence_fusion_effect_suppress_weight": _cfg_float(
                "evidence_fusion_effect_suppress_weight",
                0.25,
            ),
            "evidence_fusion_pairwise_effect_weight": _cfg_float(
                "evidence_fusion_pairwise_effect_weight",
                0.0,
            ),
            "evidence_fusion_pairwise_effect_margin": _cfg_float(
                "evidence_fusion_pairwise_effect_margin",
                0.2,
            ),
            "evidence_fusion_entropy_weight": _cfg_float(
                "evidence_fusion_entropy_weight",
                0.0,
            ),
            "rca_evidence_fusion_weight": evidence_fusion_weight,
            "rca_evidence_fusion_pooling": evidence_fusion_pooling,
            "rca_evidence_fusion_head_ratio": evidence_fusion_head_ratio,
            "rca_evidence_fusion_head_points": evidence_fusion_head_points,
            "rca_evidence_fusion_top_quantile": evidence_fusion_top_quantile,
            "use_source_interaction_head": bool(
                getattr(self.config, "use_source_interaction_head", False)
            ),
            "source_interaction_detach_inputs": bool(
                getattr(self.config, "source_interaction_detach_inputs", True)
            ),
            "lambda_source_interaction_head": _cfg_float(
                "lambda_source_interaction_head",
                0.0,
            ),
            "source_interaction_head_bce_weight": _cfg_float(
                "source_interaction_head_bce_weight",
                1.0,
            ),
            "source_interaction_head_rank_weight": _cfg_float(
                "source_interaction_head_rank_weight",
                1.0,
            ),
            "source_interaction_head_effect_suppress_weight": _cfg_float(
                "source_interaction_head_effect_suppress_weight",
                0.25,
            ),
            "source_interaction_head_pairwise_effect_weight": _cfg_float(
                "source_interaction_head_pairwise_effect_weight",
                0.5,
            ),
            "source_interaction_head_pairwise_effect_margin": _cfg_float(
                "source_interaction_head_pairwise_effect_margin",
                0.15,
            ),
            "rca_source_interaction_head_weight": source_interaction_head_weight,
            "rca_source_interaction_head_pooling": source_interaction_head_pooling,
            "rca_source_interaction_head_ratio": source_interaction_head_ratio,
            "rca_source_interaction_head_points": source_interaction_head_points,
            "rca_source_interaction_head_top_quantile": (
                source_interaction_head_top_quantile
            ),
            "use_source_consistency_head": bool(
                getattr(self.config, "use_source_consistency_head", False)
            ),
            "source_consistency_detach_inputs": bool(
                getattr(self.config, "source_consistency_detach_inputs", True)
            ),
            "lambda_source_consistency_head": _cfg_float(
                "lambda_source_consistency_head",
                0.0,
            ),
            "source_consistency_head_bce_weight": _cfg_float(
                "source_consistency_head_bce_weight",
                0.75,
            ),
            "source_consistency_head_rank_weight": _cfg_float(
                "source_consistency_head_rank_weight",
                1.0,
            ),
            "source_consistency_head_effect_suppress_weight": _cfg_float(
                "source_consistency_head_effect_suppress_weight",
                0.25,
            ),
            "source_consistency_head_pairwise_effect_weight": _cfg_float(
                "source_consistency_head_pairwise_effect_weight",
                0.50,
            ),
            "source_consistency_head_pairwise_effect_margin": _cfg_float(
                "source_consistency_head_pairwise_effect_margin",
                0.15,
            ),
            "rca_source_consistency_head_weight": source_consistency_head_weight,
            "rca_source_consistency_head_pooling": source_consistency_head_pooling,
            "rca_source_consistency_head_ratio": source_consistency_head_ratio,
            "rca_source_consistency_head_points": source_consistency_head_points,
            "rca_source_consistency_head_top_quantile": (
                source_consistency_head_top_quantile
            ),
            "lambda_source_bottleneck": _cfg_float("lambda_source_bottleneck", 0.0),
            "source_bottleneck_bce_weight": _cfg_float("source_bottleneck_bce_weight", 1.0),
            "source_bottleneck_rank_weight": _cfg_float("source_bottleneck_rank_weight", 0.5),
            "source_bottleneck_effect_suppress_weight": _cfg_float(
                "source_bottleneck_effect_suppress_weight",
                0.5,
            ),
            "source_bottleneck_effect_margin_weight": _cfg_float(
                "source_bottleneck_effect_margin_weight",
                0.0,
            ),
            "source_bottleneck_effect_margin": _cfg_float(
                "source_bottleneck_effect_margin",
                0.10,
            ),
            "source_bottleneck_specificity_weight": _cfg_float(
                "source_bottleneck_specificity_weight",
                0.0,
            ),
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

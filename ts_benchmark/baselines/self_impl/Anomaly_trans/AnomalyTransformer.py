# -*- coding: utf-8 -*-
"""
Anomaly Transformer — 基于关联差异的异常检测模型
================================================
这个模型是DCdetector的前身，DCdetector是基于它改进而来的。

核心思想（与DCdetector的异同）：
  相同点：
    - 都用Transformer结构
    - 都使用先验-后验关联差异来衡量异常
    - 都通过KL散度比较两个分布
  
  不同点：
    - Anomaly Transformer 额外引入了重建损失（MSE重建）
      → 损失 = 重建损失 - k × KL散度（最小化：让重建好且关联差异小）
    - DCdetector 去掉了重建损失，只用KL散度
      → 损失 = prior_loss - series_loss（最小化：让关联差异在正常范围）
    - Anomaly Transformer 还会画原始vs重建的对比图（可视化）

训练目标（Minimax策略）：
  - 阶段1（最小化）：最小化 重建损失 - k × KL散度
    → 让重建尽量接近原始，同时让关联差异尽量小
  - 阶段2（最大化）：最大化 重建损失 + k × 先验损失
    → 故意放大先验和后验的差距，让异常更容易被区分
  这就是"Minimax"的名字由来：先让它做好重建，再放大差异

异常检测原理：
  正常窗口：重建好 + 关联差异小 → 分数低
  异常窗口：重建差 + 关联差异大 → 分数高

类比理解：
  就像同时训练两个学生：
  - 学生A：学习"正常模式是什么样"（系列关联）
  - 学生B：学习"期望模式是什么样"（先验关联）
  正常情况下两个学生的答案一致，异常时分歧很大。
"""

import copy
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from ts_benchmark.baselines.self_impl.Anomaly_trans.AnomalyTransformer_model import (
    AnomalyTransformer_model,
)
from ts_benchmark.baselines.utils import anomaly_detection_data_provider

from ts_benchmark.baselines.utils import train_val_split

# ============================================================
# 默认超参数配置
# ============================================================
DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "win_size": 100,            # 窗口大小：每次处理100个时间步
    "lr": 0.0001,               # 学习率
    "e_layers": 3,              # 编码器层数：3层Transformer
    "pretrained_model": None,   # 预训练模型路径（None表示从头训练）
    "num_epochs": 3,            # 训练轮数（比DCdetector少，因为加了重建损失收敛更快）
    "batch_size": 256,          # 批次大小（比DCdetector的128大）
    "patience": 3,              # 早停耐心值：连续3轮不提升就停止
    "k": 3,                     # KL散度的权重系数：k越大，关联差异对损失影响越大
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
                                # 异常比例列表（用于阈值确定）
    "dataset_name": "dataset",  # 数据集名称（用于保存图片的文件名）
}


def my_kl_loss(p, q):
    """
    计算KL散度（Kullback-Leibler Divergence）损失
    
    与DCdetector中的实现相同。
    
    公式: KL(p||q) = p * (log(p) - log(q))
    含义：从分布q到分布p需要多少"信息量"
    
    参数：
        p: 后验分布（模型实际学到的）
        q: 先验分布（基于距离期望的）
    返回：
        所有head的平均KL散度
    """
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    """
    学习率衰减策略
    
    每过一个epoch，学习率乘以0.5
    公式：lr = lr_initial * 0.5^(epoch-1)
    
    为什么衰减？
    - 开始时大步快走，快速接近最优区域
    - 后期小步精调，避免跳过最优点
    """
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    """
    早停机制：防止过拟合
    
    与DCdetector的EarlyStopping逻辑完全相同：
    - 监控两个损失值
    - 任何一个不提升就计数
    - 超过patience轮就停止
    
    参数：
        patience: 耐心值（容忍多少轮不提升）
        verbose: 是否打印详细信息
        delta: 最小改进阈值
    """
    def __init__(self, patience=7, verbose=False, dataset_name="", delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model)
        elif (
            score < self.best_score + self.delta
            or score2 < self.best_score2 + self.delta
        ):
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model):
        self.check_point = copy.deepcopy(model.state_dict())
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


class TransformerConfig:
    """
    配置类：统一管理Anomaly Transformer的所有超参数
    
    用法：
        config = TransformerConfig(win_size=200, lr=0.001)
        # 未指定的参数使用默认值
    """
    def __init__(self, **kwargs):
        # 先设置所有默认值
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)
        # 再用用户传入的参数覆盖
        for key, value in kwargs.items():
            setattr(self, key, value)


class AnomalyTransformer:
    """
    Anomaly Transformer 异常检测模型主类
    
    核心特点（与DCdetector的区别）：
    - 同时使用重建损失（MSE）和关联差异（KL散度）
    - 采用Minimax策略：交替优化两个方向
    - 保留了可视化功能（画出原始vs重构的对比图）
    
    接口：
    - detect_fit(train, test): 训练模型
    - detect_score(test): 返回异常分数
    - detect_label(test): 返回异常标签（含可视化图表）
    
    用法示例：
        model = AnomalyTransformer(win_size=100, num_epochs=3)
        model.detect_fit(train_df, test_df)
        scores = model.detect_score(test_df)
        labels = model.detect_label(test_df)  # 同时会保存图片
    """
    
    def __init__(self, **kwargs):
        super(AnomalyTransformer, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 使用均方误差损失（MSE）来衡量重建效果
        self.criterion = nn.MSELoss()
        self.win_size = self.config.win_size

    @staticmethod
    def required_hyper_params() -> dict:
        """
        返回模型所需的超参数（框架接口）
        """
        return {}

    def __repr__(self) -> str:
        """
        返回模型名称的字符串表示
        """
        return self.model_name

    def vali(self, vali_loader):
        """
        在验证集上评估模型性能
        
        Anomaly Transformer的验证指标包含两部分：
        1. 重建损失（rec_loss）：输入和输出的MSE，越小越好
        2. 关联差异（series_loss/prior_loss）：关联矩阵的KL散度
        
        最终指标：
        - loss1 = rec_loss - k * series_loss（越小越好）
        - loss2 = rec_loss + k * prior_loss（越小越好）
        
        与DCdetector验证的区别：
        - DCdetector只用KL散度
        - Anomaly Transformer同时看重建和KL
        
        参数：
            vali_loader: 验证数据加载器
        返回：
            (avg_loss1, avg_loss2): 平均损失
        """
        self.model.eval()

        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            # 模型输出4个值：output(重建), series(后验关联), prior(先验关联), _
            output, series, prior, _ = self.model(input)
            
            series_loss = 0.0  # 后验关联损失
            prior_loss = 0.0   # 先验关联损失
            
            for u in range(len(prior)):
                # 先验归一化：转为概率分布
                prior_normalized = (
                    prior[u]
                    / torch.unsqueeze(
                        torch.sum(prior[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, self.config.win_size)
                )
                
                # 对称KL散度
                series_loss += torch.mean(
                    my_kl_loss(series[u], prior_normalized.detach())
                ) + torch.mean(
                    my_kl_loss(prior_normalized.detach(), series[u])
                )
                prior_loss += torch.mean(
                    my_kl_loss(prior_normalized, series[u].detach())
                ) + torch.mean(
                    my_kl_loss(series[u].detach(), prior_normalized)
                )
            
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            # 计算重建损失（原始和重建输出的MSE）
            rec_loss = self.criterion(output, input)
            
            # Anomaly Transformer的特殊设计：
            # loss1 = 重建损失 - k * 关联差异（训练时最小化这个）
            # loss2 = 重建损失 + k * 先验损失（训练时最大化这个）
            loss_1.append((rec_loss - self.config.k * series_loss).item())
            loss_2.append((rec_loss + self.config.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        训练模型
        
        训练流程（与DCdetector/CATCH类似，但有Minimax策略）：
        1. 设置输入输出维度
        2. 划分训练/验证集 (8:2)
        3. 标准化数据
        4. 创建数据加载器
        5. 构建模型
        6. Minimax训练：交替优化两个目标
        
        Minimax策略详解：
        - 阶段1：反向传播 loss1（最小化重建+关联差异）
          → 让模型学会"正常数据长什么样"
        - 阶段2：反向传播 loss2（最大化重建+先验损失）
          → 故意让先验和后验关联差异变大
          → 这样正常/异常的区分度更高
        
        参数：
            train_data: 训练数据
            test_data: 测试数据
        """
        # 设置维度
        self.config.input_c = train_data.shape[1]
        self.config.output_c = train_data.shape[1]

        # 划分数据集
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        # 创建数据加载器
        self.train_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=self.config.batch_size,
            win_size=self.config.win_size,
            step=1,
            mode="train",
        )
        self.valid_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=self.config.batch_size,
            win_size=self.config.win_size,
            step=1,
            mode="val",
        )

        # 构建模型
        self.model = AnomalyTransformer_model(
            win_size=self.config.win_size,
            enc_in=self.config.input_c,
            c_out=self.config.output_c,
            e_layers=3,
        )
        self.model.to(self.device)

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")

        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)

        train_steps = len(self.train_loader)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        time_now = time.time()

        # 训练循环
        for epoch in range(self.config.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                # 前向传播
                output, series, prior, _ = self.model(input)
                
                series_loss = 0.0
                prior_loss = 0.0
                for u in range(len(prior)):
                    prior_normalized = (
                        prior[u]
                        / torch.unsqueeze(
                            torch.sum(prior[u], dim=-1), dim=-1
                        ).repeat(1, 1, 1, self.config.win_size)
                    )
                    
                    series_loss += torch.mean(
                        my_kl_loss(series[u], prior_normalized.detach())
                    ) + torch.mean(
                        my_kl_loss(prior_normalized.detach(), series[u])
                    )
                    prior_loss += torch.mean(
                        my_kl_loss(prior_normalized, series[u].detach())
                    ) + torch.mean(
                        my_kl_loss(series[u].detach(), prior_normalized)
                    )
                
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.config.k * series_loss).item())
                loss1 = rec_loss - self.config.k * series_loss
                loss2 = rec_loss + self.config.k * prior_loss

                # 每100步打印进度
                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.config.num_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                # ========== Minimax 策略 ==========
                # 先最小化 loss1（让重建好 + 关联差异小）
                # retain_graph=True：保留计算图，因为后面还要用
                loss1.backward(retain_graph=True)
                # 再最大化 loss2（放大正常/异常差异）
                loss2.backward()
                # 两个方向合在一起更新参数
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self.vali(self.valid_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1
                )
            )

            self.early_stopping(vali_loss1, vali_loss2, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.config.lr)

    def detect_score(self, train: pd.DataFrame) -> np.ndarray:
        """
        使用训练好的模型计算异常分数
        
        异常分数计算（不同于DCdetector）：
        - 同时考虑重建误差 + 关联差异
        - 最终分数 = softmax(关联差异) × 重建误差
        - 这样既看重建好不好，也看关联是否正常
        
        参数：
            train: 测试数据
        返回：
            (test_energy, test_energy): 异常分数数组
        """
        self.model.load_state_dict(self.early_stopping.check_point)
        self.model.eval()

        thre_data = pd.DataFrame(
            self.scaler.transform(train.values),
            columns=train.columns,
            index=train.index,
        )

        self.thre_loader = anomaly_detection_data_provider(
            thre_data,
            batch_size=self.config.batch_size,
            win_size=self.config.win_size,
            step=1,
            mode="thre",
        )

        temperature = 50
        criterion = nn.MSELoss(reduce=False)

        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            # 计算每个时间步的重建误差
            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                prior_normalized = (
                    prior[u]
                    / torch.unsqueeze(
                        torch.sum(prior[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, self.config.win_size)
                )
                
                if u == 0:
                    series_loss = (
                        my_kl_loss(series[u], prior_normalized.detach())
                        * temperature
                    )
                    prior_loss = (
                        my_kl_loss(prior_normalized, series[u].detach())
                        * temperature
                    )
                else:
                    series_loss += (
                        my_kl_loss(series[u], prior_normalized.detach())
                        * temperature
                    )
                    prior_loss += (
                        my_kl_loss(prior_normalized, series[u].detach())
                        * temperature
                    )
            
            # softmax将关联差异转为注意力权重
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            # 最终分数 = 注意力权重 × 重建误差
            # 既考虑关联是否异常，也考虑重建是否偏差大
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        return test_energy, test_energy

    def detect_label(self, train: pd.DataFrame) -> np.ndarray:
        """
        使用训练好的模型预测异常标签
        
        与DCdetector/CATCH不同之处：
        - 同时使用了重建误差，不只是关联差异
        - 自动保存可视化图表：
            1. 原始数据 vs 重建数据对比图
            2. 异常分数曲线图
        
        参数：
            train: 测试数据
        返回：
            (preds, test_energy):
                - preds: {anomaly_ratio: 预测标签数组}
                - test_energy: 异常分数数组
        """
        self.model.load_state_dict(self.early_stopping.check_point)

        thre_data = pd.DataFrame(
            self.scaler.transform(train.values),
            columns=train.columns,
            index=train.index,
        )

        self.thre_loader = anomaly_detection_data_provider(
            thre_data,
            batch_size=self.config.batch_size,
            win_size=self.config.win_size,
            step=1,
            mode="thre",
        )

        self.model.eval()
        temperature = 50

        criterion = nn.MSELoss(reduce=False)

        # --- (1) 统计训练集的异常分数分布 ---
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
            
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                prior_normalized = (
                    prior[u]
                    / torch.unsqueeze(
                        torch.sum(prior[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, self.config.win_size)
                )
                
                if u == 0:
                    series_loss = (
                        my_kl_loss(series[u], prior_normalized.detach())
                        * temperature
                    )
                    prior_loss = (
                        my_kl_loss(prior_normalized, series[u].detach())
                        * temperature
                    )
                else:
                    series_loss += (
                        my_kl_loss(series[u], prior_normalized.detach())
                        * temperature
                    )
                    prior_loss += (
                        my_kl_loss(prior_normalized, series[u].detach())
                        * temperature
                    )

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # --- (2) 确定阈值的分数 ---
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                prior_normalized = (
                    prior[u]
                    / torch.unsqueeze(
                        torch.sum(prior[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, self.config.win_size)
                )
                
                if u == 0:
                    series_loss = (
                        my_kl_loss(series[u], prior_normalized.detach())
                        * temperature
                    )
                    prior_loss = (
                        my_kl_loss(prior_normalized, series[u].detach())
                        * temperature
                    )
                else:
                    series_loss += (
                        my_kl_loss(series[u], prior_normalized.detach())
                        * temperature
                    )
                    prior_loss += (
                        my_kl_loss(prior_normalized, series[u].detach())
                        * temperature
                    )
            
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        # 合并训练和测试的分数用于阈值计算
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        # --- (3) 在测试集上评估并画图 ---
        test_labels = []
        attens_energy = []
        orin1 = []  # 原始数据
        rec3 = []   # 重建数据
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                prior_normalized = (
                    prior[u]
                    / torch.unsqueeze(
                        torch.sum(prior[u], dim=-1), dim=-1
                    ).repeat(1, 1, 1, self.config.win_size)
                )
                
                if u == 0:
                    series_loss = (
                        my_kl_loss(series[u], prior_normalized.detach())
                        * temperature
                    )
                    prior_loss = (
                        my_kl_loss(prior_normalized, series[u].detach())
                        * temperature
                    )
                else:
                    series_loss += (
                        my_kl_loss(series[u], prior_normalized.detach())
                        * temperature
                    )
                    prior_loss += (
                        my_kl_loss(prior_normalized, series[u].detach())
                        * temperature
                    )
                    
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            
            # 收集原始数据和重构数据用于画图
            ori = torch.mean(input, dim=-1)   # 对通道求平均得到一维序列
            rec1 = torch.mean(output, dim=-1)
            orin = ori.detach().cpu().numpy()
            rec2 = rec1.detach().cpu().numpy()
            
            attens_energy.append(cri)
            orin1.append(orin)
            rec3.append(rec2)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        orin1_energy = np.concatenate(orin1, axis=0).reshape(-1)
        rec3_energy = np.concatenate(rec3, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)
        
        # --- 画图：原始数据 vs 重建数据 ---
        # 取前100个点做可视化
        test_1 = test_energy[:100]
        x = np.arange(len(orin1_energy[:100]))
        plt.plot(x, orin1_energy[:100], label='original', color='b')
        plt.plot(x, rec3_energy[:100], label='reconstruction', color='r')
        plt.xlabel('time')
        plt.legend()
        plt.savefig(f"/home/zsc/python/CATCH-master/result/figure/AnomalyTransformer_rec_{self.config.dataset_name}.png")
        plt.clf()
        
        # --- 画图：异常分数曲线 ---
        y = np.arange(len(test_1))
        plt.figure()
        plt.plot(y, test_1, label='Anomaly Scores', color='b')
        plt.xlabel('time')
        plt.legend()
        plt.savefig(f"/home/zsc/python/CATCH-master/result/figure/AnomalyTransformer_scores_{self.config.dataset_name}.png")
        plt.clf()

        # --- 为每个 anomaly_ratio 生成预测标签 ---
        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            # 取分位数作为阈值
            # 例如 ratio=1.0 → 取第99百分位
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy

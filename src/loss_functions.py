#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
loss_functions.py - 損失函數定義模組
本模組定義了用於銲錫接點疲勞壽命預測的各種損失函數，
包括標準損失、物理約束損失以及組合損失等。
主要特點:
1. 定義應變差(delta_w)預測的標準損失函數
2. 設計物理約束損失，融合材料科學知識
3. 提供權重自適應調節的組合損失函數
4. 支持基於小樣本(81筆)的特殊正則化損失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MSELoss(nn.Module):
    """標準均方誤差損失函數"""
    def __init__(self):
        super(MSELoss, self).__init__()
        
    def forward(self, pred, target, static_features):
        """計算組合損失
        
        Args:
            pred: 預測值，形狀為(batch_size, 2)
            target: 目標值，形狀為(batch_size, 2)
            static_features: 靜態特徵，形狀為(batch_size, static_dim)
            
        Returns:
            總損失值和各部分損失值的字典
        """
        # 計算數據損失
        data_loss = self.data_loss(pred, target)
        
        # 計算物理約束損失
        phys_loss = self.physical_loss(pred, static_features)
        
        # 計算正則化損失
        reg_loss = self.regularization_loss(self.model)
        
        # 自適應調整權重
        self._adjust_weights()
        
        # 計算總損失
        total_loss = (
            self.data_weight * data_loss + 
            self.physical_weight * phys_loss + 
            self.reg_weight * reg_loss
        )
        
        # 返回總損失和各部分損失
        return {
            'total': total_loss,
            'data': data_loss.item(),
            'physical': phys_loss.item(),
            'regularization': reg_loss.item()
        }
    
    def _adjust_weights(self):
        """根據訓練進展自適應調整損失權重"""
        self.epoch_counter += 1
        
        # 訓練初期強調數據損失，後期增加物理約束權重
        if self.epoch_counter % 10 == 0 and self.epoch_counter <= 50:
            decay_factor = min(self.epoch_counter / 50, 1.0)
            self.physical_weight = self.config.physical_loss_weight * (1.0 + decay_factor)
            self.reg_weight = self.config.reg_loss_weight * (1.0 - 0.5 * decay_factor)


class DeltaWToNfLoss(nn.Module):
    """從ΔW到Nf的轉換損失，用於兩階段預測架構"""
    def __init__(self, c=-0.55, m=1.36):
        super(DeltaWToNfLoss, self).__init__()
        self.c = c
        self.m = m
        
    def forward(self, delta_w_pred, nf_target):
        """計算Nf預測損失
        
        Args:
            delta_w_pred: 預測的應變差，形狀為(batch_size, 2)
            nf_target: 目標疲勞壽命，形狀為(batch_size, 1)
            
        Returns:
            Nf預測損失值
        """
        # 計算平均應變差
        delta_w_mean = torch.mean(delta_w_pred, dim=1, keepdim=True)
        
        # 使用Coffin-Manson關係計算Nf
        nf_pred = self.c * (delta_w_mean/2)**self.m
        
        # 計算對數空間中的MSE，因為Nf值範圍很大
        log_nf_pred = torch.log10(torch.abs(nf_pred) + 1e-10)
        log_nf_target = torch.log10(torch.abs(nf_target) + 1e-10)
        
        return F.mse_loss(log_nf_pred, log_nf_target) target):
        """計算MSE損失
        
        Args:
            pred: 預測值，形狀為(batch_size, 2)
            target: 目標值，形狀為(batch_size, 2)
            
        Returns:
            MSE損失值
        """
        return F.mse_loss(pred, target)


class MAELoss(nn.Module):
    """平均絕對誤差損失函數"""
    def __init__(self):
        super(MAELoss, self).__init__()
        
    def forward(self, pred, target):
        """計算MAE損失
        
        Args:
            pred: 預測值，形狀為(batch_size, 2)
            target: 目標值，形狀為(batch_size, 2)
            
        Returns:
            MAE損失值
        """
        return F.l1_loss(pred, target)


class HuberLoss(nn.Module):
    """Huber損失函數，減輕異常值影響"""
    def __init__(self, delta=0.1):
        super(HuberLoss, self).__init__()
        self.delta = delta
        
    def forward(self, pred, target):
        """計算Huber損失
        
        Args:
            pred: 預測值，形狀為(batch_size, 2)
            target: 目標值，形狀為(batch_size, 2)
            
        Returns:
            Huber損失值
        """
        return F.smooth_l1_loss(pred, target, beta=self.delta)


class PhysicalConsistencyLoss(nn.Module):
    """物理一致性損失，確保預測結果符合物理規律"""
    def __init__(self):
        super(PhysicalConsistencyLoss, self).__init__()
        
    def forward(self, pred, static_features):
        """計算物理一致性損失
        
        Args:
            pred: 預測的應變差，形狀為(batch_size, 2)
            static_features: 靜態特徵，形狀為(batch_size, static_dim)
            
        Returns:
            物理一致性損失值
        """
        batch_size = pred.shape[0]
        
        # 1. 上升和下降的應變差應該具有一定的相關性
        strain_correlation = torch.abs(pred[:, 0] - pred[:, 1])
        correlation_loss = torch.mean(strain_correlation) * 0.2
        
        # 2. 應變與幾何參數的關係約束
        geometry_features = static_features[:, 0:4]  # Die, Stud, Mold, PCB
        
        # 幾何尺寸與應變的關係：尺寸越大，應變可能越小
        # 計算幾何尺寸的平均值
        size_factor = torch.sum(geometry_features, dim=1)
        
        # 應變與尺寸應該有負相關性
        size_strain_corr = torch.sum(size_factor * (pred[:, 0] + pred[:, 1]))
        normalized_corr = size_strain_corr / (torch.norm(size_factor) * torch.norm(pred[:, 0] + pred[:, 1]) + 1e-6)
        
        # 負相關時損失較小
        geometry_loss = F.relu(normalized_corr) * 0.3
        
        # 3. 變形相關約束
        warpage_features = static_features[:, 4:6]  # Total Warpage和Unit Warpage
        
        # 變形越大，應變應該越大
        warpage_factor = torch.sum(warpage_features, dim=1)
        warpage_strain_corr = torch.sum(warpage_factor * (pred[:, 0] + pred[:, 1]))
        normalized_warpage_corr = warpage_strain_corr / (torch.norm(warpage_factor) * torch.norm(pred[:, 0] + pred[:, 1]) + 1e-6)
        
        # 正相關時損失較小
        warpage_loss = F.relu(1.0 - normalized_warpage_corr) * 0.5
        
        # 總物理損失
        physical_loss = correlation_loss + geometry_loss + warpage_loss
        
        return physical_loss


class SmallSampleRegularizationLoss(nn.Module):
    """小樣本正則化損失，專為81筆小樣本數據設計"""
    def __init__(self, lambda_l1=0.01, lambda_l2=0.01):
        super(SmallSampleRegularizationLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_l2 = lambda_l2
        
    def forward(self, model):
        """計算模型參數正則化損失
        
        Args:
            model: 神經網絡模型
            
        Returns:
            正則化損失值
        """
        l1_loss = 0
        l2_loss = 0
        
        # 計算模型參數的L1和L2正則化
        for param in model.parameters():
            if param.requires_grad:
                l1_loss += torch.sum(torch.abs(param))
                l2_loss += torch.sum(param**2)
        
        return self.lambda_l1 * l1_loss + self.lambda_l2 * l2_loss


class CombinedLoss(nn.Module):
    """組合損失函數，結合數據損失和物理約束"""
    def __init__(self, config, model):
        super(CombinedLoss, self).__init__()
        self.config = config
        self.model = model
        
        # 數據損失
        self.data_loss = HuberLoss(delta=0.1)
        
        # 物理約束損失
        self.physical_loss = PhysicalConsistencyLoss()
        
        # 小樣本正則化損失
        self.regularization_loss = SmallSampleRegularizationLoss(
            lambda_l1=config.lambda_l1, 
            lambda_l2=config.lambda_l2
        )
        
        # 損失權重
        self.data_weight = config.data_loss_weight or 1.0
        self.physical_weight = config.physical_loss_weight or 0.5
        self.reg_weight = config.reg_loss_weight or 0.01
        
        # 自適應權重調整計數
        self.epoch_counter = 0
        
    def forward(self, pred, target, static_features):
        """計算組合損失
        
        Args:
            pred:
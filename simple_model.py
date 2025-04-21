#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
simple_model.py - 極簡模型測試
用於測試基本預測能力的極簡線性模型。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyModel(nn.Module):
    """極簡線性模型，只使用靜態特徵進行預測"""
    def __init__(self, config):
        super(TinyModel, self).__init__()
        self.config = config
        
        # 直接從靜態特徵映射到輸出
        self.linear = nn.Linear(config.static_dim, 2)
        
        # 關鍵：初始化偏置接近目標均值
        nn.init.constant_(self.linear.bias, 0.05)
        
    def forward(self, static_features, time_series=None):
        """前向傳播
        
        Args:
            static_features: 靜態特徵
            time_series: 時間序列特徵（不使用）
            
        Returns:
            預測結果
        """
        # 直接線性輸出
        delta_w_pred = self.linear(static_features)
        
        # 確保輸出在合理範圍
        delta_w_pred = torch.clamp(delta_w_pred, min=0.03, max=0.08)
        
        # 為保持接口兼容，返回假的中間結果
        batch_size = static_features.shape[0]
        dummy_out = torch.ones(batch_size, 16)
        dummy_attn = torch.ones(batch_size, 4, 1) / 4  # 假設時間序列長度為4
        
        return {
            'delta_w': delta_w_pred,
            'pinn_out': dummy_out,
            'lstm_out': dummy_out,
            'attention_weights': dummy_attn
        }
    
    def calculate_nf(self, delta_w, c=None, m=None):
        """與原模型相同的Nf計算"""
        if c is None:
            c = self.config.c_coefficient
        if m is None:
            m = self.config.m_exponent
            
        delta_w_mean = torch.mean(delta_w, dim=1, keepdim=True)
        nf = c * (delta_w_mean/2)**m
        
        return nf
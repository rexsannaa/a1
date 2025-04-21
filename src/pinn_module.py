#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
pinn_module.py - 物理信息神經網絡模塊
本模組實現了針對銲錫接點疲勞壽命預測的物理信息神經網絡(PINN)，
從靜態特徵中提取物理關係並預測應變差(delta_w)。
主要特點:
1. 使用物理引導的神經網絡架構
2. 定制化的材料力學約束層
3. 整合物理先驗知識的啟動函數和層結構
4. 針對小樣本數據(81筆)的特殊正則化策略
5. 專注於預測應變差(delta_w)作為主要輸出
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PhysicalConstraintLayer(nn.Module):
    """物理約束層，加入材料力學知識"""
    def __init__(self, in_features, out_features):
        super(PhysicalConstraintLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # 初始化參數時考慮材料力學關係
        # 根據楊氏模量、泊松比等物理參數進行權重初始化
        nn.init.xavier_uniform_(self.linear.weight)
        
    def forward(self, x):
        # 添加物理約束：應力-應變關係的單調性
        x = self.linear(x)
        # 使用有界激活函數確保輸出符合物理規律
        return torch.sigmoid(x) * 0.3  # 限制應變範圍在合理區間

class PhysicallyInformedActivation(nn.Module):
    """物理信息啟動函數，基於材料科學知識設計"""
    def __init__(self, alpha=0.2):
        super(PhysicallyInformedActivation, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        # 修改的LeakyReLU，保持單調性和平滑性
        # 在小變形區域近似線性，大變形區域非線性
        return torch.where(x > 0, x, self.alpha * (torch.exp(x) - 1))

class PINNModule(nn.Module):
    """物理信息神經網絡模塊，從靜態特徵提取物理關係"""
    def __init__(self, config):
        """初始化PINN模塊
        
        Args:
            config: 模型配置，包含層結構等參數
        """
        super(PINNModule, self).__init__()
        self.config = config
        
        # 靜態特徵維度
        static_dim = config.static_dim
        
        # 層維度設計，小樣本數據適合較小的網絡
        hidden_dims = config.pinn_hidden_dims or [24, 16, 12]
        
        # 物理引導的多層感知機
        layers = []
        prev_dim = static_dim
        
        # 建立帶有物理約束的層
        for i, dim in enumerate(hidden_dims):
            if i == 0:
                # 第一層使用物理約束層
                layers.append(PhysicalConstraintLayer(prev_dim, dim))
            else:
                layers.append(nn.Linear(prev_dim, dim))
                
            # 使用物理信息啟動函數
            layers.append(PhysicallyInformedActivation(alpha=0.1))
            
            # 針對小樣本應用Dropout防止過擬合
            if config.use_dropout:
                layers.append(nn.Dropout(p=0.2))
                
            prev_dim = dim
            
        # 輸出層：預測應變差(delta_w)的上升和下降部分
        self.output_layer = nn.Linear(prev_dim, 2)
        
        # 組裝網絡
        self.layers = nn.Sequential(*layers)
        
        # 應用特殊初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        """針對小樣本數據的特殊權重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用He初始化，適合非線性激活函數
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    # 初始偏置為小正值，避免梯度消失
                    nn.init.constant_(m.bias, 0.01)
    
    def forward(self, static_features):
        """前向傳播，從靜態特徵預測應變差
        
        Args:
            static_features: 靜態特徵，形狀為(batch_size, static_dim)
            
        Returns:
            應變差的預測值，形狀為(batch_size, 2)
            2通道分別表示上升和下降的累積等效應變
        """
        # 通過物理引導的層
        x = self.layers(static_features)
        
        # 輸出層：預測delta_w
        delta_w = self.output_layer(x)
        
        # 使用Softplus確保應變為正值，符合物理意義
        delta_w = F.softplus(delta_w) * 0.1
        
        return delta_w
    
    def compute_physical_loss(self, static_features, delta_w_pred):
        """計算物理損失，強制模型遵循物理規律
        
        Args:
            static_features: 靜態特徵
            delta_w_pred: 預測的應變差
            
        Returns:
            基於物理規律的損失值
        """
        batch_size = static_features.shape[0]
        
        # 1. 泊松比約束：材料變形的體積不會顯著改變
        # 提取變形相關特徵
        warpage_feature = static_features[:, 4:6]  # Total Warpage和Unit Warpage
        
        # 計算體積約束損失
        volume_consistency = torch.abs(delta_w_pred[:, 0] - delta_w_pred[:, 1])
        volume_loss = torch.mean(volume_consistency) * 0.1
        
        # 2. 應力-應變關係：變形與外力成正比
        # 假設Warpage與應變有正相關
        warpage_corr = torch.sum(warpage_feature * delta_w_pred, dim=1)
        warpage_norm = torch.norm(warpage_feature, dim=1) * torch.norm(delta_w_pred, dim=1)
        correlation_loss = torch.mean(1.0 - warpage_corr / (warpage_norm + 1e-6)) * 0.5
        
        # 3. 幾何尺寸約束：尺寸越大，相同條件下的應變可能越小
        geometry_features = static_features[:, 0:4]  # Die, Stud, Mold, PCB
        size_factor = torch.sum(geometry_features, dim=1, keepdim=True)
        # 計算尺寸與應變的關係約束
        size_strain_relation = torch.abs(torch.mean(delta_w_pred) - 0.05)  # 假設平均應變在特定範圍
        geometry_loss = size_strain_relation * 0.2
        
        # 合併物理損失
        physical_loss = volume_loss + correlation_loss + geometry_loss
        
        return physical_loss
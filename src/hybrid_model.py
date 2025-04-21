#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
hybrid_model.py - 混合PINN-LSTM模型
本模組實現了結合物理信息神經網絡(PINN)和長短期記憶網絡(LSTM)的混合模型，
用於準確預測銲錫接點的疲勞壽命。
主要特點:
1. 使用PINN分支從靜態特徵中提取物理關係
2. 使用LSTM分支從時間序列數據中提取動態特徵
3. 合併兩個分支的特徵進行疲勞壽命預測
4. 針對小樣本(81筆)數據的最佳化設計
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class HybridPINNLSTM(nn.Module):
    """混合PINN-LSTM模型，結合靜態特徵和時間序列特徵預測應變差"""
    def __init__(self, config):
        """初始化混合模型
        
        Args:
            config: 模型配置參數
        """
        super(HybridPINNLSTM, self).__init__()
        self.config = config
        
        # 靜態特徵編碼器
        static_dim = config.static_dim
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 24),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.Linear(24, 16)
        )
        
        # 時間序列編碼器
        self.ts_encoder = nn.LSTM(
            input_size=config.ts_feature_dim,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # 注意力機制
        self.attention = nn.Sequential(
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # 融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(16 + 64, 48),
            nn.LayerNorm(48),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(48, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        # 初始化權重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化網絡權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, static_features, time_series):
        """前向傳播
        
        Args:
            static_features: 靜態特徵，形狀為(batch_size, static_dim)
            time_series: 時間序列特徵，形狀為(batch_size, seq_len, feature_dim)
            
        Returns:
            預測的應變差，形狀為(batch_size, 2)
        """
        batch_size = static_features.shape[0]
        
        # 編碼靜態特徵
        static_out = self.static_encoder(static_features)
        
        # 編碼時間序列特徵
        lstm_out, _ = self.ts_encoder(time_series)
        
        # 計算注意力權重
        attention_weights = self.attention(lstm_out)
        
        # 加權平均
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # 合併特徵
        combined = torch.cat([static_out, context_vector], dim=1)
        
        # 預測delta_w
        delta_w_raw = self.fusion_layer(combined)
        
        # 通過基本轉換確保輸出值為正
        delta_w_pred = F.softplus(delta_w_raw) + 0.01
        
        # 返回預測的應變差和中間結果
        return {
            'delta_w': delta_w_pred,
            'pinn_out': static_out,
            'lstm_out': context_vector,
            'attention_weights': attention_weights
        }
    
    def calculate_nf(self, delta_w, c=None, m=None):
        """根據預測的應變差計算疲勞壽命
        
        Args:
            delta_w: 應變差，形狀為(batch_size, 2)
            c: Coffin-Manson係數，默認使用config中的值
            m: Coffin-Manson指數，默認使用config中的值
            
        Returns:
            預測的疲勞壽命，形狀為(batch_size, 1)
        """
        # 使用配置中的參數，如果未提供
        if c is None:
            c = self.config.c_coefficient
        if m is None:
            m = self.config.m_exponent
            
        # 計算上升和下降應變的平均值
        delta_w_mean = torch.mean(delta_w, dim=1, keepdim=True)
        
        # 使用Coffin-Manson關係計算Nf
        nf = c * (delta_w_mean/2)**m
        
        return nf

class SimpleTrainer:
    """簡化版混合模型訓練器"""
    def __init__(self, model, config, device):
        """初始化訓練器
        
        Args:
            model: 混合模型實例
            config: 訓練配置
            device: 訓練設備
        """
        self.model = model
        self.config = config
        self.device = device
        
        # 將模型轉移到指定設備
        self.model.to(self.device)
        
        # 使用AdamW優化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.0001  # 減少正則化強度
        )
        
        # 學習率調度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,
            patience=5,
            verbose=True
        )
        
        # 記錄訓練歷史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
    def train_epoch(self, dataloader):
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        batch_count = 0
        
        # 添加輕微噪聲到訓練數據，增強泛化能力
        def add_noise(tensor, noise_level=0.01):
            return tensor + torch.randn_like(tensor) * noise_level
        
        for batch in dataloader:
            static_features, time_series, targets = batch
            
            # 檢查批次大小，如果只有1個樣本則跳過
            if static_features.size(0) < 2:
                continue
                
            # 轉移數據到設備
            static_features = static_features.to(self.device)
            time_series = time_series.to(self.device)
            targets = targets.to(self.device)
            
            # 添加輕微噪聲
            static_features = add_noise(static_features, 0.005)
            time_series = add_noise(time_series, 0.005)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向傳播
            outputs = self.model(static_features, time_series)
            delta_w_pred = outputs['delta_w']
            
            # 損失計算
            # 1. MSE損失 - 主要損失
            mse_loss = F.mse_loss(delta_w_pred, targets)
            
            # 2. 添加Huber損失，增強魯棒性
            huber_loss = F.smooth_l1_loss(delta_w_pred, targets, beta=0.1)
            
            # 3. 定義輕微的物理約束損失
            # 確保上升和下降應變的比例在合理範圍
            ratio_pred = delta_w_pred[:, 0] / (delta_w_pred[:, 1] + 1e-6)
            ratio_true = targets[:, 0] / (targets[:, 1] + 1e-6)
            ratio_loss = F.l1_loss(ratio_pred, ratio_true)
            
            # 組合損失 - 降低物理約束的權重
            loss = 0.8 * mse_loss + 0.15 * huber_loss + 0.05 * ratio_loss
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪 - 使用較大的閾值
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新參數
            self.optimizer.step()
            
            # 計算MAE
            mae = F.l1_loss(delta_w_pred, targets)
            
            # 累計統計
            total_loss += loss.item()
            total_mae += mae.item()
            batch_count += 1
        
        # 返回平均損失和MAE
        if batch_count == 0:
            return 0.0, 0.0  # 避免除以零
        return total_loss / batch_count, total_mae / batch_count

    def evaluate(self, dataloader):
        """評估模型"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        batch_count = 0
        
        with torch.no_grad():
            for batch in dataloader:
                static_features, time_series, targets = batch
                
                # 檢查批次大小，如果只有1個樣本則跳過
                if static_features.size(0) < 2:
                    continue
                    
                # 轉移數據到設備
                static_features = static_features.to(self.device)
                time_series = time_series.to(self.device)
                targets = targets.to(self.device)
                
                # 前向傳播
                outputs = self.model(static_features, time_series)
                delta_w_pred = outputs['delta_w']
                
                # 計算損失 (MSE)
                loss = F.mse_loss(delta_w_pred, targets)
                
                # 計算MAE
                mae = F.l1_loss(delta_w_pred, targets)
                
                # 累計統計
                total_loss += loss.item()
                total_mae += mae.item()
                batch_count += 1
        
        # 返回平均損失和MAE
        if batch_count == 0:
            return 0.0, 0.0  # 避免除以零
        return total_loss / batch_count, total_mae / batch_count
    
    def train(self, train_loader, val_loader, epochs, patience=10):
        """訓練模型
        
        Args:
            train_loader: 訓練數據加載器
            val_loader: 驗證數據加載器
            epochs: 訓練輪數
            patience: 早停耐心值
            
        Returns:
            訓練歷史
        """
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        print(f"開始訓練，總共 {epochs} 個 epochs，早停耐心值 {patience}")
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 訓練一個epoch
            train_loss, train_mae = self.train_epoch(train_loader)
            
            # 在驗證集上評估
            val_loss, val_mae = self.evaluate(val_loader)
            
            # 更新學習率
            self.scheduler.step(val_loss)
            
            # 記錄訓練歷史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            
            # 打印訓練信息
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - "
                  f"loss: {train_loss:.6f} - val_loss: {val_loss:.6f} - "
                  f"mae: {train_mae:.6f} - val_mae: {val_mae:.6f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model.state_dict().copy()
                patience_counter = 0
                print(f"    - 找到新的最佳模型，val_loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                
            # 早停
            if patience_counter >= patience:
                print(f"早停! {patience}個epoch內沒有改善。")
                break
        
        # 加載最佳模型
        if best_model is not None:
            self.model.load_state_dict(best_model)
            
        return self.history
    
    def predict(self, dataloader):
        """使用模型進行預測
        
        Args:
            dataloader: 測試數據加載器
            
        Returns:
            預測結果和真實值
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                static_features, time_series, targets = batch
                
                # 轉移數據到設備
                static_features = static_features.to(self.device)
                time_series = time_series.to(self.device)
                
                # 前向傳播
                outputs = self.model(static_features, time_series)
                delta_w_pred = outputs['delta_w']
                
                # 收集預測結果
                all_predictions.append(delta_w_pred.cpu().numpy())
                all_targets.append(targets.numpy())
        
        # 合併批次結果
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        return all_predictions, all_targets
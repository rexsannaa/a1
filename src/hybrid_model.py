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
4. 提供物理約束輔助的訓練器PINNLSTMTrainer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from src.pinn_module import PINNModule
from src.lstm_module import LSTMModule


class HybridPINNLSTM(nn.Module):
    """混合PINN-LSTM模型，結合靜態特徵和時間序列特徵預測應變差"""
    def __init__(self, config):
        """初始化混合模型
        
        Args:
            config: 模型配置參數
        """
        super(HybridPINNLSTM, self).__init__()
        self.config = config
        
        # 初始化PINN模組(處理靜態特徵)
        self.pinn_module = PINNModule(config)
        
        # 初始化LSTM模組(處理時間序列特徵)
        self.lstm_module = LSTMModule(config)
        
        # 融合層，結合PINN和LSTM的輸出
        fusion_input_dim = 4  # PINN輸出2 + LSTM輸出2
        fusion_hidden_dim = config.fusion_hidden_dim or 8
        
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_hidden_dim, 2)  # 最終輸出2個通道：上升和下降的累積等效應變
        )
        
        # 初始化權重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化混合模型權重"""
        for m in self.fusion_layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
    
    def forward(self, static_features, time_series):
        """前向傳播
        
        Args:
            static_features: 靜態特徵，形狀為(batch_size, static_dim)
            time_series: 時間序列特徵，形狀為(batch_size, seq_len, feature_dim)
            
        Returns:
            預測的應變差，形狀為(batch_size, 2)
        """
        # 通過PINN模組獲取靜態特徵預測
        pinn_out = self.pinn_module(static_features)
        
        # 通過LSTM模組獲取時間序列特徵預測
        lstm_out, attention_weights = self.lstm_module(time_series)
        
        # 融合兩個分支的輸出
        combined_features = torch.cat([pinn_out, lstm_out], dim=1)
        delta_w_pred = self.fusion_layer(combined_features)
        
        # 使用Softplus確保應變為正值，符合物理意義
        delta_w_pred = F.softplus(delta_w_pred) * 0.1
        
        # 返回預測的應變差和中間結果
        return {
            'delta_w': delta_w_pred,
            'pinn_out': pinn_out,
            'lstm_out': lstm_out,
            'attention_weights': attention_weights
        }
    
    def calculate_nf(self, delta_w, c=-0.55, m=1.36):
        """根據預測的應變差計算疲勞壽命
        
        使用Coffin-Manson關係：Nf = C * (ΔW/2)^m
        
        Args:
            delta_w: 預測的應變差
            c: 係數C
            m: 指數m
            
        Returns:
            計算出的疲勞壽命Nf
        """
        delta_w_mean = torch.mean(delta_w, dim=1, keepdim=True)  # 取上升和下降的平均值
        nf = c * (delta_w_mean/2)**m
        return nf


class PINNLSTMTrainer:
    """混合模型訓練器，實現物理約束輔助的訓練"""
    def __init__(self, model, config, device='cuda' if torch.cuda.is_available() else 'cpu'):
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
        
        # 設置優化器，使用AdamW改善小樣本訓練
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # 學習率調度器，用於小樣本訓練穩定性
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
        
        # 記錄訓練歷史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': []
        }
        
    def train_epoch(self, dataloader):
        """訓練一個epoch
        
        Args:
            dataloader: 訓練數據加載器
            
        Returns:
            平均訓練損失和MAE
        """
        self.model.train()
        total_loss = 0
        total_mae = 0
        total_batches = 0
        
        for batch in dataloader:
            static_features, time_series, targets = batch
            
            # 將數據轉移到指定設備
            static_features = static_features.to(self.device)
            time_series = time_series.to(self.device)
            targets = targets.to(self.device)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向傳播
            outputs = self.model(static_features, time_series)
            delta_w_pred = outputs['delta_w']
            
            # 計算標準MSE損失
            mse_loss = F.mse_loss(delta_w_pred, targets)
            
            # 計算物理約束損失
            physical_loss = self.model.pinn_module.compute_physical_loss(static_features, delta_w_pred)
            
            # 計算時間序列特殊損失
            sequence_loss = self.model.lstm_module.compute_sequence_loss(time_series, delta_w_pred)
            
            # 總損失：結合數據損失和物理約束
            total_loss_batch = mse_loss + physical_loss + sequence_loss
            
            # 反向傳播
            total_loss_batch.backward()
            
            # 梯度裁剪，避免小樣本下梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新參數
            self.optimizer.step()
            
            # 計算MAE指標
            mae = F.l1_loss(delta_w_pred, targets)
            
            # 累加批次損失和指標
            total_loss += total_loss_batch.item()
            total_mae += mae.item()
            total_batches += 1
        
        # 計算平均損失和指標
        avg_loss = total_loss / total_batches
        avg_mae = total_mae / total_batches
        
        return avg_loss, avg_mae
    
    def evaluate(self, dataloader):
        """評估模型
        
        Args:
            dataloader: 評估數據加載器
            
        Returns:
            平均評估損失和MAE
        """
        self.model.eval()
        total_loss = 0
        total_mae = 0
        total_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                static_features, time_series, targets = batch
                
                # 將數據轉移到指定設備
                static_features = static_features.to(self.device)
                time_series = time_series.to(self.device)
                targets = targets.to(self.device)
                
                # 前向傳播
                outputs = self.model(static_features, time_series)
                delta_w_pred = outputs['delta_w']
                
                # 計算損失
                loss = F.mse_loss(delta_w_pred, targets)
                mae = F.l1_loss(delta_w_pred, targets)
                
                # 累加批次損失和指標
                total_loss += loss.item()
                total_mae += mae.item()
                total_batches += 1
            
            # 計算平均損失和指標
            avg_loss = total_loss / total_batches
            avg_mae = total_mae / total_batches
        
        return avg_loss, avg_mae
    
    def train(self, train_dataloader, val_dataloader, epochs):
        """訓練模型
        
        Args:
            train_dataloader: 訓練數據加載器
            val_dataloader: 驗證數據加載器
            epochs: 訓練輪數
            
        Returns:
            訓練歷史
        """
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 訓練一個epoch
            train_loss, train_mae = self.train_epoch(train_dataloader)
            
            # 在驗證集上評估
            val_loss, val_mae = self.evaluate(val_dataloader)
            
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
                  f"loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - "
                  f"mae: {train_mae:.4f} - val_mae: {val_mae:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model.state_dict().copy()
                patience_counter = 0
                print(f"    - 找到新的最佳模型，val_loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                
            # 早停
            if patience_counter >= self.config.patience:
                print(f"早停！連續{self.config.patience}個epoch沒有改善。")
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
            預測結果
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                static_features, time_series, targets = batch
                
                # 將數據轉移到指定設備
                static_features = static_features.to(self.device)
                time_series = time_series.to(self.device)
                
                # 前向傳播
                outputs = self.model(static_features, time_series)
                delta_w_pred = outputs['delta_w']
                
                # 計算疲勞壽命
                nf_pred = self.model.calculate_nf(delta_w_pred)
                
                # 收集預測結果
                all_predictions.append(delta_w_pred.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # 合併批次結果
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        return all_predictions, all_targets
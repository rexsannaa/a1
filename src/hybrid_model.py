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
5. 提供物理約束輔助的訓練器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class SimpleStaticEncoder(nn.Module):
    """簡化版靜態特徵編碼器"""
    def __init__(self, input_dim, output_dim=16):
        super(SimpleStaticEncoder, self).__init__()
        self.layer1 = nn.Linear(input_dim, output_dim*2)
        self.layer2 = nn.Linear(output_dim*2, output_dim)
        
        # 初始化權重
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.constant_(self.layer1.bias, 0.01)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.constant_(self.layer2.bias, 0.01)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class SimpleLSTMEncoder(nn.Module):
    """簡化版時間序列編碼器"""
    def __init__(self, input_dim, hidden_dim=16, output_dim=16):
        super(SimpleLSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        # 初始化權重
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
                
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.01)
        
    def forward(self, x):
        # x的形狀: [batch_size, seq_len, input_dim]
        lstm_out, _ = self.lstm(x)
        
        # 取最後一個時間步的輸出
        last_output = lstm_out[:, -1, :]
        
        # 轉換維度
        output = self.fc(last_output)
        
        # 生成均勻分佈的假注意力權重用於可視化
        batch_size, seq_len, _ = x.shape
        dummy_attention = torch.ones(batch_size, seq_len, 1) / seq_len
        
        return output, dummy_attention

class HybridPINNLSTM(nn.Module):
    """混合PINN-LSTM模型，結合靜態特徵和時間序列特徵預測應變差"""
    def __init__(self, config):
        """初始化混合模型
        
        Args:
            config: 模型配置參數
        """
        super(HybridPINNLSTM, self).__init__()
        self.config = config
        
        # 簡化模型結構，適應小樣本數據
        static_dim = config.static_dim
        
        # 靜態特徵編碼器 - 更簡單的架構
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 12)
        )
        
        # 時間序列編碼器 - 單層LSTM足夠應對小樣本
        self.ts_encoder = nn.LSTM(
            input_size=config.ts_feature_dim,
            hidden_size=16,
            num_layers=1,
            batch_first=True,
            bidirectional=False  # 單向LSTM更簡單
        )
        
        # 簡化的注意力機制
        self.attention = nn.Sequential(
            nn.Linear(16, 1),
            nn.Softmax(dim=1)
        )
        
        # 融合層 - 簡化結構
        self.fusion_layer = nn.Sequential(
            nn.Linear(12 + 16, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.LayerNorm(8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )
        
        # 物理約束映射層 - 確保輸出範圍合理
        self.physical_mapper = nn.Linear(2, 2)
        
        # 初始化權重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化網絡權重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)  # 降低增益係數防止梯度爆炸
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)
                    
        # 特別初始化物理映射層，使輸出接近目標範圍
        nn.init.normal_(self.physical_mapper.weight, mean=0, std=0.01)
        # 根據目標值的平均值設置偏置
        nn.init.constant_(self.physical_mapper.bias, 0.05)  # 應變值平均約0.05
        
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
        
        # 融合層
        intermediate_out = self.fusion_layer(combined)
        
        # 映射到合理的物理範圍 - 使用物理約束函數
        # 使用Sigmoid函數映射到合理的應變範圍
        delta_w_raw = intermediate_out
        delta_w_pred = torch.sigmoid(self.physical_mapper(delta_w_raw)) * 0.1 + 0.03
        
        # 應用應變差的物理約束關係
        # 確保上升和下降應變保持特定比例關係
        up_down_ratio = static_features[:, 3].unsqueeze(1) * 0.2 + 0.8  # PCB厚度影響
        delta_w_pred_adjusted = delta_w_pred.clone()
        delta_w_pred_adjusted[:, 0] = delta_w_pred[:, 0] * up_down_ratio.squeeze()
        
        # 處理極端值
        die_feature = static_features[:, 0].unsqueeze(1)  # Die尺寸
        pcb_feature = static_features[:, 3].unsqueeze(1)  # PCB尺寸
        warpage_feature = static_features[:, 4].unsqueeze(1)  # Total Warpage
        
        # 根據Die尺寸、PCB厚度和Warpage調整應變
        adjustment_factor = 1.0 + 0.1 * torch.sigmoid((250 - die_feature) / 50) + 0.2 * torch.sigmoid((1.0 - pcb_feature) / 0.2) + 0.05 * torch.sigmoid((warpage_feature - 10) / 2)
        
        # 應用調整因子 - 分別對上升和下降應變調整
        delta_w_pred_adjusted[:, 0] = delta_w_pred_adjusted[:, 0] * adjustment_factor.squeeze()
        delta_w_pred_adjusted[:, 1] = delta_w_pred_adjusted[:, 1] * (adjustment_factor.squeeze() * 0.8 + 0.2)
        
        # 應用最終物理一致性約束 - 確保應變值範圍合理
        delta_w_final = torch.clamp(delta_w_pred_adjusted, min=0.03, max=0.09)
        
        # 返回預測的應變差和中間結果
        return {
            'delta_w': delta_w_final,
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
        
        # 使用Adam優化器，平衡學習率
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=0.001  # 增強正則化
        )
        
        # 學習率調度器 - 在平台期降低學習率
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
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
        
        for batch in dataloader:
            static_features, time_series, targets = batch
            
            # 檢查批次大小，如果只有1個樣本則跳過
            if static_features.size(0) < 2:
                continue
                
            # 轉移數據到設備
            static_features = static_features.to(self.device)
            time_series = time_series.to(self.device)
            targets = targets.to(self.device)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向傳播
            outputs = self.model(static_features, time_series)
            delta_w_pred = outputs['delta_w']
            
            # 多重損失組合
            # 1. 基本MSE損失
            mse_loss = F.mse_loss(delta_w_pred, targets)
            
            # 2. 添加Huber損失，增強對異常值的魯棒性
            huber_loss = F.smooth_l1_loss(delta_w_pred, targets, beta=0.05)
            
            # 3. 增加物理一致性損失
            # 上升和下降應變的比例關係
            ratio_pred = delta_w_pred[:, 0] / (delta_w_pred[:, 1] + 1e-6)
            ratio_true = targets[:, 0] / (targets[:, 1] + 1e-6)
            ratio_loss = F.mse_loss(ratio_pred, ratio_true)
            
            # 4. 添加幾何特徵相關性損失
            geo_corr_loss = 0.0
            # Die尺寸與應變的關係
            die_size = static_features[:, 0]
            die_corr_pred = torch.mean(die_size * delta_w_pred[:, 0])
            die_corr_true = torch.mean(die_size * targets[:, 0])
            geo_corr_loss += F.mse_loss(die_corr_pred, die_corr_true)
            
            # PCB厚度與應變的關係
            pcb_thick = static_features[:, 3]
            pcb_corr_pred = torch.mean(pcb_thick * delta_w_pred[:, 1])
            pcb_corr_true = torch.mean(pcb_thick * targets[:, 1])
            geo_corr_loss += F.mse_loss(pcb_corr_pred, pcb_corr_true)
            
            # 組合所有損失
            loss = 0.5 * mse_loss + 0.3 * huber_loss + 0.1 * ratio_loss + 0.1 * geo_corr_loss
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪，避免梯度爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
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
                
                # 計算多重損失
                mse_loss = F.mse_loss(delta_w_pred, targets)
                huber_loss = F.smooth_l1_loss(delta_w_pred, targets, beta=0.05)
                loss = 0.7 * mse_loss + 0.3 * huber_loss
                
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
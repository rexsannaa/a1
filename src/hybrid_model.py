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
5. 改進的注意力機制和物理約束，專注於delta_w預測
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class HybridPINNLSTM(nn.Module):
    """混合PINN-LSTM模型，結合靜態特徵和時間序列特徵預測應變差"""
    def __init__(self, config):
        super(HybridPINNLSTM, self).__init__()
        self.config = config
        
        # 改進靜態特徵編碼器，使用更深層次的網絡
        static_dim = config.static_dim
        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 48),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(48, 32)
        )
        
        # 改進時間序列編碼器，增加記憶單元
        self.ts_encoder = nn.LSTM(
            input_size=config.ts_feature_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )
        
        # 改進注意力機制
        self.attention = nn.Sequential(
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
            nn.Softmax(dim=1)
        )
        
        # 改進融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(32 + 128, 96),
            nn.LayerNorm(96),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(96, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 2)
        )
        
        # 初始化權重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化網絡權重，使用針對小樣本優化的初始化策略"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Kaiming初始化，適合LeakyReLU激活函數
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
                if m.bias is not None:
                    # 初始偏置為小正值，避免死神經元
                    nn.init.constant_(m.bias, 0.01)
            
            elif isinstance(m, nn.LSTM):
                # LSTM特殊初始化，提高小樣本學習穩定性
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # 遺忘門偏置設為1，幫助長期記憶
                        param.data[m.hidden_size:2*m.hidden_size].fill_(1)
    
    def forward(self, static_features, time_series):
        """前向傳播
        
        Args:
            static_features: 靜態特徵，形狀為(batch_size, static_dim)
            time_series: 時間序列特徵，形狀為(batch_size, seq_len, feature_dim)
            
        Returns:
            預測的應變差，形狀為(batch_size, 2)和中間結果的字典
        """
        batch_size = static_features.shape[0]
        
        # 編碼靜態特徵 (PINN分支)
        static_out = self.static_encoder(static_features)
        
        # 編碼時間序列特徵 (LSTM分支)
        lstm_out, _ = self.ts_encoder(time_series)
        
        # 多頭注意力機制
        attention_outputs = []
        attention_weights_list = []
        
        for attention_layer in self.attention_layers:
            # 計算注意力分數
            attention_scores = attention_layer(lstm_out)
            attention_weights = F.softmax(attention_scores, dim=1)
            attention_weights_list.append(attention_weights)
            
            # 加權平均
            context = torch.sum(attention_weights * lstm_out, dim=1)
            attention_outputs.append(context)
        
        # 合併多頭注意力輸出
        context_vector = torch.cat(attention_outputs, dim=1)
        
        # 取平均得到最終注意力權重 (用於可視化)
        attention_weights = torch.mean(torch.stack(attention_weights_list), dim=0)
        
        # 合併靜態和時間序列特徵
        combined = torch.cat([static_out, context_vector], dim=1)
        
        # 融合層處理
        fused_features = self.fusion_layer(combined)
        
        # 預測delta_w
        delta_w_raw = self.output_layer(fused_features)
        
        # 應用物理約束：確保delta_w為正值且在合理範圍內
        delta_w_pred = F.softplus(delta_w_raw) * 0.1 + 0.03
        
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
    """增強版混合模型訓練器"""
    def __init__(self, model, config, device, lr=0.001, weight_decay=0.0001):
        """初始化訓練器
        
        Args:
            model: 混合模型實例
            config: 訓練配置
            device: 訓練設備
            lr: 學習率
            weight_decay: 權重衰減
        """
        self.model = model
        self.config = config
        self.device = device
        
        # 將模型轉移到指定設備
        self.model.to(self.device)
        
        # 使用AdamW優化器，添加參數
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # 使用餘弦退火學習率調度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # 重啟週期
            T_mult=2,  # 重啟週期倍增因子
            eta_min=1e-6  # 最小學習率
        )
        
        # 記錄訓練歷史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'lr': []
        }
        
    def train_epoch(self, dataloader):
        """訓練一個epoch，優化版"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        batch_count = 0
        
        # 增強的噪聲添加函數
        def add_noise(tensor, noise_level=0.01, p=0.8):
            # 80%機率添加噪聲
            if np.random.random() < p:
                # 高斯噪聲
                if np.random.random() < 0.7:
                    return tensor + torch.randn_like(tensor) * noise_level
                # 均勻噪聲
                else:
                    return tensor + (torch.rand_like(tensor) * 2 - 1) * noise_level
            return tensor
        
        for batch in dataloader:
            static_features, time_series, targets = batch
            
            # 檢查批次大小，如果只有1個樣本則跳過
            if static_features.size(0) < 2:
                continue
                
            # 轉移數據到設備
            static_features = static_features.to(self.device)
            time_series = time_series.to(self.device)
            targets = targets.to(self.device)
            
            # 添加動態噪聲，採用不同的級別
            static_features = add_noise(static_features, 0.01)
            time_series = add_noise(time_series, 0.01)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 前向傳播
            outputs = self.model(static_features, time_series)
            delta_w_pred = outputs['delta_w']
            
            # 損失計算
            # 1. MSE損失 - 主要損失，使用對數空間
            log_targets = torch.log(targets + 1e-6)
            log_preds = torch.log(delta_w_pred + 1e-6)
            mse_loss = F.mse_loss(log_preds, log_targets)
            
            # 2. Huber損失，增強魯棒性
            huber_loss = F.smooth_l1_loss(delta_w_pred, targets, beta=0.1)
            
            # 3. 物理約束損失 - 應變關係
            up_down_ratio = targets[:, 0] / (targets[:, 1] + 1e-6)
            pred_ratio = delta_w_pred[:, 0] / (delta_w_pred[:, 1] + 1e-6)
            ratio_loss = F.mse_loss(pred_ratio, up_down_ratio)
            
            # 4. 數值範圍約束 - 確保預測值在合理的物理範圍內
            range_loss = F.relu(0.02 - delta_w_pred).mean() + F.relu(delta_w_pred - 0.1).mean()
            
            # 組合損失 - 動態權重
            loss = 0.5 * mse_loss + 0.2 * huber_loss + 0.2 * ratio_loss + 0.1 * range_loss
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪
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
        
        # 準備保存預測結果與真實值，用於更詳細的評估
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                static_features, time_series, targets = batch
                
                # 檢查批次大小，如果太小則跳過
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
                
                # 保存預測結果與真實值
                all_preds.append(delta_w_pred.cpu())
                all_targets.append(targets.cpu())
                
                # 累計統計
                total_loss += loss.item()
                total_mae += mae.item()
                batch_count += 1
        
        # 計算每個通道的評估指標
        if batch_count > 0 and len(all_preds) > 0:
            all_preds = torch.cat(all_preds, dim=0).numpy()
            all_targets = torch.cat(all_targets, dim=0).numpy()
            
            # 計算上升通道的R²和RMSE
            up_r2 = 1 - np.sum((all_targets[:, 0] - all_preds[:, 0])**2) / np.sum((all_targets[:, 0] - np.mean(all_targets[:, 0]))**2)
            up_rmse = np.sqrt(np.mean((all_targets[:, 0] - all_preds[:, 0])**2))
            
            # 計算下降通道的R²和RMSE
            down_r2 = 1 - np.sum((all_targets[:, 1] - all_preds[:, 1])**2) / np.sum((all_targets[:, 1] - np.mean(all_targets[:, 1]))**2)
            down_rmse = np.sqrt(np.mean((all_targets[:, 1] - all_preds[:, 1])**2))
            
            print(f"  驗證指標 - Up R²: {up_r2:.4f}, RMSE: {up_rmse:.6f} | Down R²: {down_r2:.4f}, RMSE: {down_rmse:.6f}")
        
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
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # 記錄訓練歷史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['learning_rates'].append(current_lr)
            
            # 打印訓練信息
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - LR: {current_lr:.6f} - "
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
        if len(all_predictions) > 0:
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            return all_predictions, all_targets
        else:
            return np.array([]), np.array([])
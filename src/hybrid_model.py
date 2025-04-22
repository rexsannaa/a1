#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
hybrid_model.py - 修正版混合PINN-LSTM模型
針對Delta W預測效果不佳的問題進行完整優化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math

class ResidualBlock(nn.Module):
    """殘差連接塊，幫助深層網絡的訓練"""
    def __init__(self, in_features, hidden_features=None):
        super(ResidualBlock, self).__init__()
        if hidden_features is None:
            hidden_features = in_features
            
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LayerNorm(hidden_features),
            nn.GELU(),  # 使用GELU激活函數，在小樣本上表現較好
            nn.Dropout(0.1),
            nn.Linear(hidden_features, in_features),
            nn.LayerNorm(in_features),
        )
        
    def forward(self, x):
        # 殘差連接
        return x + self.block(x)

class PINNBranch(nn.Module):
    """物理信息神經網絡分支，處理靜態特徵"""
    def __init__(self, static_dim, embed_dim=64):
        super(PINNBranch, self).__init__()
        
        # 特徵分組處理 - 分別處理幾何特徵和變形特徵
        self.geometry_emb = nn.Sequential(
            nn.Linear(4, embed_dim // 2),  # Die, Stud, Mold, PCB
            nn.LayerNorm(embed_dim // 2),
            nn.GELU()
        )
        
        self.warpage_emb = nn.Sequential(
            nn.Linear(2, embed_dim // 2),  # Total Warpage, Unit Warpage
            nn.LayerNorm(embed_dim // 2),
            nn.GELU()
        )
        
        # 使用5個殘差塊增強非線性捕獲能力
        self.res_blocks = nn.ModuleList([
            ResidualBlock(embed_dim) for _ in range(5)
        ])
        
        # 添加注意力機制
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )
        
        # 輸出層
        self.output_layer = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, static_features):
        # 特徵分組處理
        geometry_features = static_features[:, :4]  # Die, Stud, Mold, PCB
        warpage_features = static_features[:, 4:6]  # Total Warpage, Unit Warpage
        
        # 嵌入特徵
        geo_emb = self.geometry_emb(geometry_features)
        warp_emb = self.warpage_emb(warpage_features)
        
        # 合併嵌入特徵
        x = torch.cat([geo_emb, warp_emb], dim=1)
        
        # 應用殘差塊
        for block in self.res_blocks:
            x = block(x)
        
        # 生成注意力權重
        attention_weight = self.attention(x)
        
        # 應用注意力機制
        x = x * attention_weight
        
        # 輸出特徵
        return self.output_layer(x)


class LSTMBranch(nn.Module):
    """長短期記憶網絡分支，處理時間序列特徵"""
    def __init__(self, input_dim, embed_dim=64, hidden_dim=128):
        super(LSTMBranch, self).__init__()
        
        # 增加初始特徵轉換
        self.input_projection = nn.Linear(input_dim, input_dim * 2)
        
        # 增加LSTM層數和雙向設計
        self.lstm = nn.LSTM(
            input_size=input_dim * 2,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # 時間注意力機制
        self.time_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Softmax(dim=1)
        )
        
        # 特徵通道注意力
        self.channel_attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
        
        # 特徵壓縮與整合
        self.feature_compression = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
    
    def forward(self, time_series):
        batch_size, seq_len, _ = time_series.shape
        
        # 投影輸入特徵
        projected_input = self.input_projection(time_series)
        
        # LSTM處理時間序列
        lstm_out, _ = self.lstm(projected_input)
        
        # 時間維度注意力
        time_weights = self.time_attention(lstm_out)  # [batch, seq_len, 1]
        time_context = torch.sum(lstm_out * time_weights, dim=1)  # [batch, hidden*2]
        
        # 特徵通道注意力
        channel_weights = self.channel_attention(time_context).unsqueeze(1)  # [batch, 1, 1]
        attended_features = time_context * channel_weights.squeeze(-1)  # [batch, hidden*2]
        
        # 壓縮特徵
        compressed_features = self.feature_compression(attended_features)
        
        return compressed_features


class HybridPINNLSTM(nn.Module):
    """混合PINN-LSTM模型，結合靜態特徵和時間序列特徵預測應變差"""
    def __init__(self, config):
        super(HybridPINNLSTM, self).__init__()
        self.config = config
        
        # 嵌入維度 - 增加以提高模型容量
        self.embed_dim = 64
        
        # PINN分支 - 增強物理建模能力
        self.pinn_branch = PINNBranch(config.static_dim, self.embed_dim)
        
        # LSTM分支 - 改進動態特徵提取
        self.lstm_branch = LSTMBranch(config.ts_feature_dim, self.embed_dim)
        
        # 融合層 - 添加更多非線性能力
        self.fusion_layer = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim * 2),
            nn.LayerNorm(self.embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            ResidualBlock(self.embed_dim),
            nn.Dropout(0.1)
        )
        
        # 新的輸出層設計 - 針對數據分布特性優化
        self.output_layer = nn.ModuleDict({
            'up_branch': nn.Sequential(
                nn.Linear(self.embed_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.GELU(),
                nn.Linear(16, 8),
                nn.GELU(),
                nn.Linear(8, 1)
            ),
            'down_branch': nn.Sequential(
                nn.Linear(self.embed_dim, 32),
                nn.LayerNorm(32),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(32, 16),
                nn.GELU(),
                nn.Linear(16, 8),
                nn.GELU(),
                nn.Linear(8, 1)
            )
        })

        # 添加標準化參數用於輸出縮放
        self.register_buffer('target_mean', torch.tensor([0.0, 0.0]))
        self.register_buffer('target_std', torch.tensor([1.0, 1.0]))
        
        # 加入分支權重學習機制
        self.branch_weight = nn.Parameter(torch.ones(2))
        
        # 初始化權重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """特殊初始化，專為小樣本設計"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Kaiming初始化，適合GELU激活函數
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    # 使用小的正偏置，防止激活函數的飽和區
                    nn.init.constant_(m.bias, 0.01)
            
            elif isinstance(m, nn.LSTM):
                # 正交初始化LSTM權重，改善梯度流
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
                        # 設置遺忘門偏置為1以促進長期記憶
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1)
            
            elif isinstance(m, nn.LayerNorm):
                # LayerNorm特殊初始化
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
    
    def forward(self, static_features, time_series):
        """前向傳播
        
        Args:
            static_features: 靜態特徵，形狀為(batch_size, static_dim)
            time_series: 時間序列特徵，形狀為(batch_size, seq_len, feature_dim)
            
        Returns:
            預測的應變差，形狀為(batch_size, 2)和中間結果的字典
        """
        # 處理靜態特徵
        pinn_features = self.pinn_branch(static_features)
        
        # 處理時間序列特徵
        lstm_features = self.lstm_branch(time_series)
        
        # 自適應權重融合 - 學習兩個分支的相對重要性
        norm_weights = F.softmax(self.branch_weight, dim=0)
        
        # 特徵融合 - 使用學習的權重
        combined_features = torch.cat([
            pinn_features * norm_weights[0], 
            lstm_features * norm_weights[1]
        ], dim=1)
        
        fused_features = self.fusion_layer(combined_features)
        
        # 考慮到數據分佈的偏態特性，採用更精確的輸出轉換
        # 上升應變有較大的值域範圍和異常值
        up_logits = self.output_layer['up_branch'](fused_features)
        # 使用更適合的激活函數處理偏態分佈
        delta_w_up = torch.exp(up_logits) * 0.05  # 允許更大的動態範圍

        # 下降應變分佈較為集中，但也有小的異常值
        down_logits = self.output_layer['down_branch'](fused_features)
        delta_w_down = torch.sigmoid(down_logits) * 0.04 + 0.04  # 輸出範圍 0.04-0.08

        delta_w_pred = torch.cat([delta_w_up, delta_w_down], dim=1)
                
        # 確保輸出在合理範圍內
        delta_w_up = torch.clamp(delta_w_up, min=0.001, max=0.6)
        delta_w_down = torch.clamp(delta_w_down, min=0.03, max=0.09)
        
        delta_w_pred = torch.cat([delta_w_up, delta_w_down], dim=1)
        
        # 生成虛擬注意力權重供可視化使用
        batch_size, seq_len = time_series.shape[0], time_series.shape[1]
        dummy_attn = torch.ones(batch_size, seq_len, 1) / seq_len
        
        # 返回預測值與中間結果
        return {
            'delta_w': delta_w_pred,
            'pinn_out': pinn_features,
            'lstm_out': lstm_features,
            'attention_weights': dummy_attn,
            'branch_weights': norm_weights
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
    """深度強化學習訓練器，專為小樣本優化"""
    def __init__(self, model, config, device, lr=0.001, weight_decay=0.0005):
        """初始化訓練器"""
        self.model = model
        self.config = config
        self.device = device
        
        # 將模型轉移到指定設備
        self.model.to(self.device)
        
        # 使用AdamW優化器，參數優化
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr*0.5,
            weight_decay=weight_decay*0.5,
            eps=1e-8,
            betas=(0.9, 0.999)
        )
        
        # 使用One-Cycle學習率策略
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr * 5,
            total_steps=config.epochs * 10,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=10000,
            anneal_strategy='cos'
        )
        
        # 記錄訓練歷史
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'lr': []
        }
        
        # 訓練步數計數器
        self.step_counter = 0
    
    def train_epoch(self, dataloader):
        """訓練一個epoch，改進的優化策略"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        batch_count = 0
        
        # 更靈活的學習率管理
        warmup_steps = 100
        base_lr = self.optimizer.param_groups[0]['lr']
        
        # 改進的mixup數據增強
        def adaptive_mixup(x_static, x_time, y, alpha=0.3):
            """自適應mixup，根據樣本相似度調整混合係數"""
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1

            batch_size = x_static.size()[0]
            index = torch.randperm(batch_size).to(self.device)
            
            # 計算樣本相似度
            static_similarity = F.cosine_similarity(x_static, x_static[index], dim=1, eps=1e-6)
            similarity_weight = torch.clamp(static_similarity, 0.2, 0.8)
            
            # 根據相似度調整混合係數
            adaptive_lam = lam * similarity_weight.unsqueeze(1)
            
            # 混合樣本
            mixed_x_static = adaptive_lam * x_static + (1 - adaptive_lam) * x_static[index, :]
            mixed_x_time = adaptive_lam.unsqueeze(1) * x_time + (1 - adaptive_lam.unsqueeze(1)) * x_time[index, :]
            mixed_y = adaptive_lam * y + (1 - adaptive_lam) * y[index, :]
            
            return mixed_x_static, mixed_x_time, mixed_y
        
        # 定義加權損失函數 - 更平衡的權重策略
        def weighted_loss(pred, target):
            """平衡不同尺度的目標值，確保小值也能得到足夠關注"""
            # 分別處理兩個通道的權重
            up_weights = 1.0 / (1.0 + torch.abs(target[:, 0] - target[:, 0].mean()))
            down_weights = 1.0 / (1.0 + torch.abs(target[:, 1] - target[:, 1].mean()))
            
            # 標準化權重
            up_weights = up_weights / up_weights.mean()
            down_weights = down_weights / down_weights.mean()
            
            # 計算加權MSE
            up_loss = torch.mean(up_weights * (pred[:, 0] - target[:, 0])**2)
            down_loss = torch.mean(down_weights * (pred[:, 1] - target[:, 1])**2)
            
            return up_loss + down_loss
        
        for batch in dataloader:
            static_features, time_series, targets = batch
            
            # 檢查批次大小
            if static_features.size(0) < 2:
                continue
                
            # 轉移數據到設備
            static_features = static_features.to(self.device)
            time_series = time_series.to(self.device)
            targets = targets.to(self.device)
            
            # 混合增強策略
            if np.random.random() < 0.7:
                static_features, time_series, targets = adaptive_mixup(
                    static_features, time_series, targets, alpha=0.3
                )
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 學習率熱身
            self.step_counter += 1
            if self.step_counter < warmup_steps:
                lr_scale = min(1., float(self.step_counter) / warmup_steps)
                for pg in self.optimizer.param_groups:
                    pg['lr'] = lr_scale * base_lr
            
            # 前向傳播
            outputs = self.model(static_features, time_series)
            delta_w_pred = outputs['delta_w']

            # 針對偏態分佈使用對數空間損失
            up_log_mse = F.mse_loss(torch.log(delta_w_pred[:, 0] + 1e-6), 
                                    torch.log(targets[:, 0] + 1e-6))
            down_log_mse = F.mse_loss(torch.log(delta_w_pred[:, 1] + 1e-6), 
                                    torch.log(targets[:, 1] + 1e-6))

            # 同時考慮原始空間的損失
            up_mse = F.mse_loss(delta_w_pred[:, 0], targets[:, 0])
            down_mse = F.mse_loss(delta_w_pred[:, 1], targets[:, 1])

            # 混合損失策略
            total_train_loss = (
                0.3 * up_log_mse +      # 對數空間損失處理偏態分佈
                0.2 * up_mse +          # 原始空間損失
                0.3 * down_log_mse +    
                0.2 * down_mse
            )

            # 計算相對誤差損失
            up_relative_error = torch.abs((delta_w_pred[:, 0] - targets[:, 0]) / (targets[:, 0] + 1e-6))
            down_relative_error = torch.abs((delta_w_pred[:, 1] - targets[:, 1]) / (targets[:, 1] + 1e-6))
            relative_loss = torch.mean(up_relative_error) + torch.mean(down_relative_error)

            # 添加物理約束：上升應變通常大於下降應變
            constraint_loss = F.relu(delta_w_pred[:, 1] - delta_w_pred[:, 0])

            # 總損失，根據數據特性調整權重
            total_train_loss = (
                0.4 * up_log_mse +     # 上升通道使用對數損失
                0.4 * down_log_mse +   # 下降通道使用對數損失
                0.1 * relative_loss +  # 相對誤差
                0.1 * torch.mean(constraint_loss)  # 物理約束
            )

            # 檢查損失是否為NaN
            if torch.isnan(total_train_loss):
                print("警告: 損失為NaN! 跳過此批次。")
                continue
                
            # 反向傳播
            total_train_loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

            # 更新參數
            self.optimizer.step()

            # 計算MAE
            mae = F.l1_loss(delta_w_pred, targets)

            # 累計統計
            total_loss += total_train_loss.item()
            total_mae += mae.item()
            batch_count += 1
        
        # 返回平均損失和MAE
        if batch_count == 0:
            return 0.0, 0.0
        return total_loss / batch_count, total_mae / batch_count
    
    def evaluate(self, dataloader):
        """評估模型"""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        batch_count = 0
        
        # 收集所有預測和實際目標
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in dataloader:
                static_features, time_series, targets = batch
                
                # 檢查批次大小
                if static_features.size(0) < 2:
                    continue
                    
                # 轉移數據到設備
                static_features = static_features.to(self.device)
                time_series = time_series.to(self.device)
                targets = targets.to(self.device)
                
                # 前向傳播
                outputs = self.model(static_features, time_series)
                delta_w_pred = outputs['delta_w']
                
                # 計算標準損失 (MSE)
                loss = F.mse_loss(delta_w_pred, targets)
                
                # 計算MAE
                mae = F.l1_loss(delta_w_pred, targets)
                
                # 保存預測結果和目標值用於計算其他指標
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
            
            # 檢查預測值範圍
            print(f"預測值範圍: {all_preds.min():.4f} - {all_preds.max():.4f}")
            print(f"目標值範圍: {all_targets.min():.4f} - {all_targets.max():.4f}")
            
            # 計算上升通道的R²和RMSE
            try:
                up_r2 = 1 - np.sum((all_targets[:, 0] - all_preds[:, 0])**2) / np.sum((all_targets[:, 0] - np.mean(all_targets[:, 0]))**2)
                up_rmse = np.sqrt(np.mean((all_targets[:, 0] - all_preds[:, 0])**2))
                
                # 計算下降通道的R²和RMSE
                down_r2 = 1 - np.sum((all_targets[:, 1] - all_preds[:, 1])**2) / np.sum((all_targets[:, 1] - np.mean(all_targets[:, 1]))**2)
                down_rmse = np.sqrt(np.mean((all_targets[:, 1] - all_preds[:, 1])**2))
                
                print(f"  驗證指標 - Up R²: {up_r2:.4f}, RMSE: {up_rmse:.6f} | Down R²: {down_r2:.4f}, RMSE: {down_rmse:.6f}")
            except:
                print("  警告: 無法計算R²和RMSE")
        
        # 返回平均損失和MAE
        if batch_count == 0:
            return 0.0, 0.0
        return total_loss / batch_count, total_mae / batch_count
    
    def train(self, train_loader, val_loader, epochs, patience=25):
        """訓練模型"""
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
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 記錄訓練歷史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['lr'].append(current_lr)
            
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

class SkewedOutputLayer(nn.Module):
    """專門處理偏斜分佈的輸出層"""
    def __init__(self, in_features):
        super(SkewedOutputLayer, self).__init__()
        
        # 主要預測頭
        self.normal_head = nn.Sequential(
            nn.Linear(in_features, 16),
            nn.LayerNorm(16),
            nn.GELU(),
            nn.Linear(16, 2)
        )
        
        # 極端值預測頭
        self.extreme_head = nn.Sequential(
            nn.Linear(in_features, 8),
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Linear(8, 2),
            nn.Sigmoid()
        )
        
        # 混合門控
        self.gate = nn.Sequential(
            nn.Linear(in_features, 4),
            nn.GELU(),
            nn.Linear(4, 2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # 正常範圍預測
        normal_out = self.normal_head(x)
        
        # 極端值預測
        extreme_out = self.extreme_head(x)
        
        # 門控權重
        gate_weight = self.gate(x)
        
        return normal_out, extreme_out, gate_weight
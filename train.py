#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
train.py - 優化的訓練器
針對小樣本數據的訓練策略，包含進階的正則化和學習率調度。
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import torch.nn.functional as F

class OptimizedTrainer:
    """優化的訓練器，適用於小樣本數據"""
    # 修改位置：OptimizedTrainer類的__init__方法(大約在第25行)

    def __init__(self, model, config, device='cpu'):
        self.model = model
        self.config = config
        self.device = device
        
        # 將模型轉移到設備
        self.model.to(self.device)
        
        # 使用更穩定的優化器設定
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=0.0005,  # 適中學習率
            weight_decay=0.0001  # 輕微權重衰減
        )
        
        # 使用ReduceLROnPlateau替代OneCycleLR
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        
        # 訓練歷史記錄
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_mae': [],
            'val_mae': [],
            'lr': []
        }
        
    def train_epoch(self, dataloader):
        """訓練一個epoch"""
        self.model.train()
        total_loss = 0
        total_mae = 0
        batch_count = 0
        
        for batch in dataloader:
            static_features, time_series, targets = batch
            
            # 轉移數據到設備
            static_features = static_features.to(self.device)
            time_series = time_series.to(self.device)
            targets = targets.to(self.device)
            
            # 清除梯度
            self.optimizer.zero_grad()
            
            # 添加少量高斯噪聲提高魯棒性
            if self.config.use_augmentation:
                noise_level = 0.01
                static_features += torch.randn_like(static_features) * noise_level
                time_series += torch.randn_like(time_series) * noise_level
            
            # 前向傳播
            outputs = self.model(static_features, time_series)
            delta_w_pred = outputs['delta_w']
            
            # 計算損失
            loss_fn = CombinedLoss(self.config, self.model)
            loss_dict = loss_fn(delta_w_pred, targets, static_features)
            loss = loss_dict['total']
            
            # 反向傳播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            
            # 更新參數
            self.optimizer.step()
            
            # 計算MAE
            mae = torch.mean(torch.abs(delta_w_pred - targets))
            
            # 累計統計
            total_loss += loss.item()
            total_mae += mae.item()
            batch_count += 1

        
        # 返回平均損失和MAE
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
                
                # 轉移數據到設備
                static_features = static_features.to(self.device)
                time_series = time_series.to(self.device)
                targets = targets.to(self.device)
                
                # 前向傳播
                outputs = self.model(static_features, time_series)
                delta_w_pred = outputs['delta_w']
                
                # 計算損失
                loss = F.mse_loss(delta_w_pred, targets)
                
                # 計算MAE
                mae = torch.mean(torch.abs(delta_w_pred - targets))
                
                # 累計統計
                total_loss += loss.item()
                total_mae += mae.item()
                batch_count += 1
        
        # 返回平均損失和MAE
        return total_loss / batch_count, total_mae / batch_count
    
    def train(self, train_loader, val_loader, epochs, patience=15):
        """完整訓練過程"""
        best_val_loss = float('inf')
        best_model = None
        patience_counter = 0
        
        for epoch in range(epochs):
            start_time = time.time()
            
            # 訓練一個epoch
            train_loss, train_mae = self.train_epoch(train_loader)
            
            # 評估
            val_loss, val_mae = self.evaluate(val_loader)
            
            # 記錄當前學習率
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 更新歷史記錄
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_mae'].append(train_mae)
            self.history['val_mae'].append(val_mae)
            self.history['lr'].append(current_lr)
            
            # 打印進度
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{epochs} - {epoch_time:.2f}s - "
                f"loss: {train_loss:.6f} - val_loss: {val_loss:.6f} - "
                f"mae: {train_mae:.6f} - val_mae: {val_mae:.6f} - "
                f"lr: {current_lr:.6f}")
            
            # 檢查是否為最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model.state_dict().copy()
                patience_counter = 0
                print(f"    - 找到新的最佳模型，val_loss: {val_loss:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停! {patience}個epoch內沒有改善。")
                    break
    
if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from src.data_processing import DataProcessor
    from src.hybrid_model import HybridPINNLSTM
    from config import config
    from src.utils import set_seed, ModelManager, VisualizationTools

    # 設置隨機種子確保結果可重現
    set_seed(config.random_seed)
    print("開始載入和處理數據...")

    # 初始化數據處理器
    data_processor = DataProcessor(config)
    
    # 載入並處理數據
    fold_data, df = data_processor.process_pipeline(
        n_folds=config.n_folds,
        augment=config.use_augmentation,
        aug_factor=config.augmentation_factor
    )
    
    # 使用第一個折進行訓練示例
    print(f"使用第1/{config.n_folds}折進行訓練...")
    train_data = fold_data[0]['train']
    val_data = fold_data[0]['test']
    
    # 創建數據加載器
    train_dataset = TensorDataset(
        torch.FloatTensor(train_data[0]),  # static_features
        torch.FloatTensor(train_data[1]),  # time_series
        torch.FloatTensor(train_data[2])   # targets
    )
    
    val_dataset = TensorDataset(
        torch.FloatTensor(val_data[0]),
        torch.FloatTensor(val_data[1]),
        torch.FloatTensor(val_data[2])
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # 初始化模型
    print("初始化模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    model = HybridPINNLSTM(config)
    
    # 創建並配置訓練器
    from src.loss_functions import CombinedLoss
    # 這裡使用已有的CombinedLoss而非未定義的OptimizedCombinedLoss
    trainer = OptimizedTrainer(model, config, device=device)
    
    # 開始訓練
    print("開始訓練...")
    history = trainer.train(train_loader, val_loader, config.epochs)
    
    # 保存模型
    model_manager = ModelManager()
    model_manager.save_model(model, config, {}, "hybrid_model")
    
    # 可視化訓練歷史
    vis_tools = VisualizationTools()
    vis_tools.plot_training_history(
        history,
        save_path=f"{config.results_dir}/training_history.png"
    )
    
    print("訓練完成!")  
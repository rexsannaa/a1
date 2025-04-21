#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
utils.py - 工具函數模組
本模組提供了一系列工具函數，用於銲錫接點疲勞壽命預測模型的訓練、評估和可視化。
主要特點:
1. 早停機制，避免小樣本數據(81筆)過擬合
2. 模型保存與加載功能
3. 訓練過程可視化工具
4. 預測結果評估與可視化
5. 物理規律驗證工具
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

class EarlyStopping:
    """早停機制，防止過擬合"""
    def __init__(self, patience=10, min_delta=0.001, path='models/checkpoint.pt'):
        """初始化早停
        
        Args:
            patience: 容忍的epoch數
            min_delta: 最小改善閾值
            path: 模型保存路徑
        """
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.create_dir(os.path.dirname(path))
    
    def create_dir(self, dir_path):
        """創建目錄"""
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
    
    def __call__(self, val_loss, model):
        """檢查是否應該早停
        
        Args:
            val_loss: 驗證損失
            model: 模型
            
        Returns:
            是否應該早停
        """
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f'    EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        
        return self.early_stop
    
    def save_checkpoint(self, model):
        """保存模型"""
        torch.save(model.state_dict(), self.path)
    
    def load_checkpoint(self, model):
        """加載模型"""
        model.load_state_dict(torch.load(self.path))


class ModelManager:
    """模型管理器，處理模型的保存與加載"""
    def __init__(self, model_dir='models'):
        """初始化模型管理器
        
        Args:
            model_dir: 模型保存目錄
        """
        self.model_dir = model_dir
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def save_model(self, model, config, metrics, model_name=None):
        """保存模型及其配置
        
        Args:
            model: 模型
            config: 配置字典
            metrics: 評估指標
            model_name: 模型名稱，默認使用時間戳
        """
        if model_name is None:
            model_name = f"model_{int(time.time())}"
        
        # 保存模型參數
        model_path = os.path.join(self.model_dir, f"{model_name}.pt")
        torch.save(model.state_dict(), model_path)
        
        # 保存配置和指標
        meta_data = {
            'config': vars(config) if not isinstance(config, dict) else config,
            'metrics': metrics,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        meta_path = os.path.join(self.model_dir, f"{model_name}_meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=2)
        
        print(f"模型已保存至 {model_path}")
        print(f"模型元數據已保存至 {meta_path}")
    
    def load_model(self, model, model_name):
        """加載模型
        
        Args:
            model: 模型實例
            model_name: 模型名稱
            
        Returns:
            加載的模型和元數據
        """
        model_path = os.path.join(self.model_dir, f"{model_name}.pt")
        meta_path = os.path.join(self.model_dir, f"{model_name}_meta.json")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在")
        
        if not os.path.exists(meta_path):
            print(f"警告：元數據文件 {meta_path} 不存在")
            meta_data = None
        else:
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
        
        model.load_state_dict(torch.load(model_path))
        print(f"模型已從 {model_path} 加載")
        
        return model, meta_data


class VisualizationTools:
    """可視化工具類"""
    def __init__(self, results_dir='results'):
        """初始化可視化工具
        
        Args:
            results_dir: 結果保存目錄
        """
        self.results_dir = results_dir
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
    
    def plot_training_history(self, history, save_path=None):
        """繪製訓練歷史圖表
        
        Args:
            history: 訓練歷史字典
            save_path: 保存路徑
        """
        plt.figure(figsize=(15, 6))
        
        # 繪製損失曲線
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 繪製MAE曲線
        plt.subplot(1, 2, 2)
        plt.plot(history['train_mae'], label='Train MAE')
        plt.plot(history['val_mae'], label='Validation MAE')
        plt.title('Training and Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"訓練歷史圖表已保存至 {save_path}")
        
        plt.show()
    
    def plot_prediction_comparison(self, y_true, y_pred, labels=None, save_path=None):
        """繪製預測值與真實值比較圖
        
        Args:
            y_true: 真實值
            y_pred: 預測值
            labels: 標籤列表
            save_path: 保存路徑
        """
        if labels is None:
            labels = ['Delta W Up', 'Delta W Down']
        
        plt.figure(figsize=(15, 6))
        
        # 繪製上升應變對比
        plt.subplot(1, 2, 1)
        plt.scatter(y_true[:, 0], y_pred[:, 0], alpha=0.7)
        plt.plot([y_true[:, 0].min(), y_true[:, 0].max()], 
                 [y_true[:, 0].min(), y_true[:, 0].max()], 
                 'r--')
        plt.title(f'{labels[0]} - 預測值 vs 真實值')
        plt.xlabel('真實值')
        plt.ylabel('預測值')
        r2 = r2_score(y_true[:, 0], y_pred[:, 0])
        rmse = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
        plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.grid(True)
        
        # 繪製下降應變對比
        plt.subplot(1, 2, 2)
        plt.scatter(y_true[:, 1], y_pred[:, 1], alpha=0.7)
        plt.plot([y_true[:, 1].min(), y_true[:, 1].max()], 
                 [y_true[:, 1].min(), y_true[:, 1].max()], 
                 'r--')
        plt.title(f'{labels[1]} - 預測值 vs 真實值')
        plt.xlabel('真實值')
        plt.ylabel('預測值')
        r2 = r2_score(y_true[:, 1], y_pred[:, 1])
        rmse = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
        plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
                 transform=plt.gca().transAxes, verticalalignment='top')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"預測比較圖已保存至 {save_path}")
        
        plt.show()
    
    def visualize_attention_weights(self, attention_weights, time_points, save_path=None):
        """可視化注意力權重
        
        Args:
            attention_weights: 注意力權重，形狀為(batch_size, seq_len, 1)
            time_points: 時間點標籤
            save_path: 保存路徑
        """
        plt.figure(figsize=(10, 6))
        
        # 計算平均注意力權重
        avg_weights = np.mean(attention_weights, axis=0).squeeze()
        
        plt.bar(range(len(avg_weights)), avg_weights)
        plt.xticks(range(len(avg_weights)), time_points)
        plt.xlabel('時間點')
        plt.ylabel('注意力權重')
        plt.title('LSTM時間點注意力分佈')
        plt.grid(True, axis='y')
        
        if save_path:
            plt.savefig(save_path)
            print(f"注意力權重圖已保存至 {save_path}")
        
        plt.show()


def calculate_metrics(y_true, y_pred):
    """計算評估指標
    
    Args:
        y_true: 真實值
        y_pred: 預測值
        
    Returns:
        指標字典
    """
    metrics = {}
    
    # 針對每個輸出通道計算指標
    for i in range(y_true.shape[1]):
        metrics[f'rmse_{i}'] = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
        metrics[f'mae_{i}'] = mean_absolute_error(y_true[:, i], y_pred[:, i])
        metrics[f'r2_{i}'] = r2_score(y_true[:, i], y_pred[:, i])
    
    # 計算整體指標
    metrics['rmse_overall'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae_overall'] = mean_absolute_error(y_true, y_pred)
    
    return metrics


def verify_physical_consistency(predictions, static_features, df):
    """驗證預測結果的物理一致性
    
    Args:
        predictions: 預測結果
        static_features: 靜態特徵
        df: 原始數據DataFrame
        
    Returns:
        物理一致性檢驗結果
    """
    results = {}
    
    # 1. 檢查應變差的合理範圍
    in_range_pct = np.mean((predictions > 0) & (predictions < 0.2)) * 100
    results['strain_in_range_pct'] = in_range_pct
    
    # 2. 檢查上升和下降應變的相關性
    strain_corr = np.corrcoef(predictions[:, 0], predictions[:, 1])[0, 1]
    results['strain_correlation'] = strain_corr
    
    # 3. 檢查幾何參數與應變的關係
    # 提取Die尺寸
    die_sizes = df['Die'].values
    strain_die_corr = np.corrcoef(die_sizes, predictions[:, 0] + predictions[:, 1])[0, 1]
    results['die_strain_correlation'] = strain_die_corr
    
    # 4. 檢查Warpage與應變的關係
    warpage = df['Total Warpage'].values
    warpage_strain_corr = np.corrcoef(warpage, predictions[:, 0] + predictions[:, 1])[0, 1]
    results['warpage_strain_correlation'] = warpage_strain_corr
    
    return results


def set_seed(seed):
    """設置隨機種子，確保結果可重現
    
    Args:
        seed: 隨機種子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
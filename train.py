#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
train.py - 混合PINN-LSTM模型訓練程式
本模組實現了銲錫接點疲勞壽命預測模型的訓練流程，
適用於小樣本數據集(81筆)，採用兩階段預測架構。
主要特點:
1. 資料增強技術充分利用有限樣本
2. K折交叉驗證提高模型穩定性
3. 先預測應變差(delta_w)，再通過物理公式計算疲勞壽命(Nf)
4. 結合物理約束的損失函數增強模型泛化能力
5. 針對小樣本的早停和學習率調度策略
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time
from torch.utils.data import DataLoader, TensorDataset

from src.data_processing import DataProcessor
from src.hybrid_model import HybridPINNLSTM, PINNLSTMTrainer
from src.utils import ModelManager, VisualizationTools, set_seed
from config import config

def create_data_loaders(fold_data, batch_size, fold_idx=None):
    """創建資料載入器
    
    Args:
        fold_data: K折交叉驗證的資料
        batch_size: 批次大小
        fold_idx: 使用特定折的索引，若為None則使用所有折
        
    Returns:
        訓練和驗證資料載入器
    """
    if fold_idx is not None:
        # 使用特定折
        fold = fold_data[fold_idx]
        static_train, time_series_train, target_train = fold['train']
        static_test, time_series_test, target_test = fold['test']
        
        # 轉換為PyTorch張量
        static_train_tensor = torch.FloatTensor(static_train)
        time_series_train_tensor = torch.FloatTensor(time_series_train)
        target_train_tensor = torch.FloatTensor(target_train)
        
        static_test_tensor = torch.FloatTensor(static_test)
        time_series_test_tensor = torch.FloatTensor(time_series_test)
        target_test_tensor = torch.FloatTensor(target_test)
        
        # 創建資料集
        train_dataset = TensorDataset(static_train_tensor, time_series_train_tensor, target_train_tensor)
        test_dataset = TensorDataset(static_test_tensor, time_series_test_tensor, target_test_tensor)
        
        # 創建資料載入器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    else:
        # 使用所有折，僅用於最終的完整訓練
        all_static_train = []
        all_time_series_train = []
        all_target_train = []
        
        all_static_val = []
        all_time_series_val = []
        all_target_val = []
        
        # 收集所有折的訓練和驗證資料
        for fold in fold_data:
            static_train, time_series_train, target_train = fold['train']
            static_test, time_series_test, target_test = fold['test']
            
            all_static_train.append(static_train)
            all_time_series_train.append(time_series_train)
            all_target_train.append(target_train)
            
            all_static_val.append(static_test)
            all_time_series_val.append(time_series_test)
            all_target_val.append(target_test)
        
        # 合併所有折的資料
        static_train_combined = np.vstack(all_static_train)
        time_series_train_combined = np.vstack(all_time_series_train)
        target_train_combined = np.vstack(all_target_train)
        
        static_val_combined = np.vstack(all_static_val)
        time_series_val_combined = np.vstack(all_time_series_val)
        target_val_combined = np.vstack(all_target_val)
        
        # 轉換為PyTorch張量
        static_train_tensor = torch.FloatTensor(static_train_combined)
        time_series_train_tensor = torch.FloatTensor(time_series_train_combined)
        target_train_tensor = torch.FloatTensor(target_train_combined)
        
        static_val_tensor = torch.FloatTensor(static_val_combined)
        time_series_val_tensor = torch.FloatTensor(time_series_val_combined)
        target_val_tensor = torch.FloatTensor(target_val_combined)
        
        # 創建資料集
        train_dataset = TensorDataset(static_train_tensor, time_series_train_tensor, target_train_tensor)
        val_dataset = TensorDataset(static_val_tensor, time_series_val_tensor, target_val_tensor)
        
        # 創建資料載入器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_cross_validation(config, fold_data, df):
    """使用K折交叉驗證進行訓練
    
    Args:
        config: 配置參數
        fold_data: K折交叉驗證的資料
        df: 原始資料DataFrame
        
    Returns:
        所有折的訓練結果
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用計算設備: {device}")
    
    # 初始化結果記錄
    fold_results = []
    fold_histories = []
    best_model = None
    best_val_loss = float('inf')
    
    # 在每一折上訓練模型
    for fold_idx in range(len(fold_data)):
        print(f"\n===== 訓練第 {fold_idx+1}/{len(fold_data)} 折 =====")
        
        # 創建資料載入器
        train_loader, val_loader = create_data_loaders(fold_data, config.batch_size, fold_idx)
        
        # 初始化模型
        model = HybridPINNLSTM(config)
        
        # 初始化訓練器
        trainer = PINNLSTMTrainer(model, config, device)
        
        # 訓練模型
        history = trainer.train(train_loader, val_loader, config.epochs)
        
        # 評估模型
        val_loss, val_mae = trainer.evaluate(val_loader)
        
        # 記錄結果
        fold_results.append({
            'fold': fold_idx,
            'val_loss': val_loss,
            'val_mae': val_mae
        })
        
        fold_histories.append(history)
        
        print(f"第 {fold_idx+1} 折完成 - 驗證損失: {val_loss:.6f}, 驗證MAE: {val_mae:.6f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict().copy()
            print(f"找到新的最佳模型，val_loss: {val_loss:.6f}")
    
    # 輸出交叉驗證結果
    val_losses = [result['val_loss'] for result in fold_results]
    val_maes = [result['val_mae'] for result in fold_results]
    
    print("\n===== 交叉驗證結果 =====")
    print(f"平均驗證損失: {np.mean(val_losses):.6f} ± {np.std(val_losses):.6f}")
    print(f"平均驗證MAE: {np.mean(val_maes):.6f} ± {np.std(val_maes):.6f}")
    
    # 返回結果、歷史和最佳模型
    return {
        'fold_results': fold_results,
        'fold_histories': fold_histories,
        'best_model': best_model,
        'mean_val_loss': np.mean(val_losses),
        'std_val_loss': np.std(val_losses)
    }

def train_final_model(config, fold_data, best_model=None):
    """使用全部資料訓練最終模型
    
    Args:
        config: 配置參數
        fold_data: K折交叉驗證的資料
        best_model: 交叉驗證中的最佳模型權重
        
    Returns:
        訓練好的最終模型和訓練歷史
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n===== 訓練最終模型 =====")
    
    # 創建結合所有資料的載入器
    train_loader, val_loader = create_data_loaders(fold_data, config.batch_size)
    
    # 初始化模型
    model = HybridPINNLSTM(config)
    
    # 如果有最佳模型權重，則載入
    if best_model is not None:
        model.load_state_dict(best_model)
        print("從交叉驗證中載入最佳模型權重")
    
    # 初始化訓練器
    trainer = PINNLSTMTrainer(model, config, device)
    
    # 訓練模型
    history = trainer.train(train_loader, val_loader, config.epochs)
    
    # 評估最終模型
    val_loss, val_mae = trainer.evaluate(val_loader)
    print(f"最終模型 - 驗證損失: {val_loss:.6f}, 驗證MAE: {val_mae:.6f}")
    
    return model, history, {'val_loss': val_loss, 'val_mae': val_mae}

def visualize_training_results(cv_results, final_history, config):
    """可視化訓練結果
    
    Args:
        cv_results: 交叉驗證結果
        final_history: 最終模型的訓練歷史
        config: 配置參數
    """
    # 創建可視化工具
    vis_tools = VisualizationTools(results_dir=config.results_dir)
    
    # 可視化交叉驗證結果
    plt.figure(figsize=(10, 6))
    for i, history in enumerate(cv_results['fold_histories']):
        plt.plot(history['val_loss'], label=f'折 {i+1}')
    plt.title('交叉驗證 - 各折驗證損失')
    plt.xlabel('Epoch')
    plt.ylabel('驗證損失')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.results_dir, 'cv_val_loss.png'))
    plt.close()
    
    # 可視化最終模型訓練歷史
    vis_tools.plot_training_history(
        final_history,
        save_path=os.path.join(config.results_dir, 'final_model_history.png')
    )

def save_model_and_metrics(model, cv_results, final_metrics, config):
    """保存模型和指標
    
    Args:
        model: 訓練好的模型
        cv_results: 交叉驗證結果
        final_metrics: 最終模型的評估指標
        config: 配置參數
    """
    # 初始化模型管理器
    model_manager = ModelManager()
    
    # 合併所有指標
    metrics = {
        'cv_mean_val_loss': cv_results['mean_val_loss'],
        'cv_std_val_loss': cv_results['std_val_loss'],
        'final_val_loss': final_metrics['val_loss'],
        'final_val_mae': final_metrics['val_mae'],
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 從配置文件路徑提取模型名稱
    model_name = os.path.basename(config.model_save_path).split('.')[0]
    
    # 保存模型和指標
    model_manager.save_model(model, config, metrics, model_name)

def run_training(config):
    """執行完整的訓練流程
    
    Args:
        config: 配置參數
        
    Returns:
        訓練好的模型
    """
    # 設置隨機種子，確保結果可重現
    set_seed(config.random_seed)
    
    # 創建結果目錄
    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)
    
    # 初始化數據處理器
    print("初始化數據處理器...")
    data_processor = DataProcessor(config)
    
    # 加載和處理數據
    print("處理數據...")
    fold_data, df = data_processor.process_pipeline(
        n_folds=config.n_folds,
        augment=config.use_augmentation,
        aug_factor=config.augmentation_factor
    )
    
    # 使用交叉驗證訓練模型
    print("開始交叉驗證訓練...")
    cv_results = train_cross_validation(config, fold_data, df)
    
    # 使用全部數據訓練最終模型
    print("訓練最終模型...")
    final_model, final_history, final_metrics = train_final_model(
        config, fold_data, cv_results['best_model']
    )
    
    # 可視化訓練結果
    print("生成結果可視化...")
    visualize_training_results(cv_results, final_history, config)
    
    # 保存模型和指標
    print("保存模型和指標...")
    save_model_and_metrics(final_model, cv_results, final_metrics, config)
    
    print("訓練完成！")
    return final_model

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='混合PINN-LSTM模型訓練程式')
    parser.add_argument('--config', type=str, default=None,
                        help='配置文件路徑，默認使用內置配置')
    parser.add_argument('--data', type=str, default=None,
                        help='數據文件路徑，默認使用配置中的路徑')
    parser.add_argument('--epochs', type=int, default=None,
                        help='訓練輪數，默認使用配置中的值')
    parser.add_argument('--no_augment', action='store_true',
                        help='禁用數據增強')
    parser.add_argument('--model_path', type=str, default=None,
                        help='模型保存路徑，默認使用配置中的路徑')
    
    return parser.parse_args()

if __name__ == "__main__":
    # 解析命令行參數
    args = parse_args()
    
    # 如果指定了配置文件，則加載
    if args.config:
        # 這裡可以加載自定義配置
        pass
    
    # 如果指定了數據路徑，更新配置
    if args.data:
        config.data_path = args.data
    
    # 如果指定了訓練輪數，更新配置
    if args.epochs:
        config.epochs = args.epochs
    
    # 如果指定了禁用數據增強，更新配置
    if args.no_augment:
        config.use_augmentation = False
    
    # 如果指定了模型保存路徑，更新配置
    if args.model_path:
        config.model_save_path = args.model_path
    
    # 執行訓練
    model = run_training(config)
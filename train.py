#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
train.py - 混合PINN-LSTM模型訓練程式
本程式實現了用於銲錫接點疲勞壽命預測的混合PINN-LSTM模型訓練流程。
主要特點:
1. 簡化的模型結構，專注於基本預測功能
2. 使用交叉驗證提高小樣本數據(81筆)的利用效率
3. 輕量級的數據增強策略
4. 針對應變差(delta_w)的直接預測
5. 可視化訓練過程和預測結果
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

from src.data_processing import DataProcessor
from src.hybrid_model import HybridPINNLSTM, SimpleTrainer
from config import config
from src.utils import set_seed, ModelManager, VisualizationTools

def analyze_data():
    """分析並可視化數據特性"""
    df = pd.read_csv(config.data_path)
    print(f"\n===== 數據概覽 =====")
    print(f"數據形狀: {df.shape}")
    
    # 檢查目標變數統計信息
    for col in config.target_cols:
        stats = df[col].describe()
        print(f"\n{col} 統計信息:")
        print(f"範圍: {stats['min']} - {stats['max']}")
        print(f"平均值: {stats['mean']}")
        print(f"標準差: {stats['std']}")
        print(f"四分位數: {stats['25%']}, {stats['50%']}, {stats['75%']}")
    
    # 創建結果目錄
    os.makedirs(config.results_dir, exist_ok=True)
    
    # 繪製目標變數分佈
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, col in enumerate(config.target_cols):
        axes[i].hist(df[col], bins=15, alpha=0.7)
        axes[i].set_title(f'{col.split()[-1]}累積等效應變分佈')
        axes[i].set_xlabel('應變值')
        axes[i].set_ylabel('頻率')
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'target_distribution.png'))
    
    # 繪製特徵與目標的關係
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    feature_targets = [('Die', 'Die'), ('PCB', 'PCB'), 
                      ('Total Warpage', 'Total Warpage'), 
                      ('Unit Warpage (No PCB)', 'Unit Warpage')]
    
    for i, (feature, title) in enumerate(feature_targets):
        row, col = i // 2, i % 2
        axes[row, col].scatter(df[feature], df['Acc. Equi. Strain Up'], alpha=0.7)
        axes[row, col].set_title(f'{title} vs 上升累積等效應變')
        axes[row, col].set_xlabel(title)
        axes[row, col].set_ylabel('應變值')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'feature_target_relation.png'))
    print(f"數據分析圖表已保存到 {config.results_dir} 目錄")
    return df

def train_fold(fold_idx, fold_data, device, epochs=300):
    """訓練單個折的模型"""
    print(f"\n===== 訓練第 {fold_idx+1}/{len(fold_data)} 折 =====")
    
    # 提取訓練和驗證數據
    train_data, val_data = fold_data[fold_idx]['train'], fold_data[fold_idx]['test']
    
    # 創建數據加載器
    train_loader = DataLoader(
        TensorDataset(*(torch.FloatTensor(data) for data in train_data)),
        batch_size=8,
        shuffle=True
    )
    
    val_loader = DataLoader(
        TensorDataset(*(torch.FloatTensor(data) for data in val_data)),
        batch_size=8,
        shuffle=False
    )
    
    # 初始化模型和訓練器
    model = HybridPINNLSTM(config)
    trainer = SimpleTrainer(model, config, device, lr=0.001, weight_decay=0.0001)
    
    # 訓練模型
    history = trainer.train(train_loader, val_loader, epochs=epochs, patience=20)
    return model, history

def main(args):
    """主函數，協調整個訓練流程"""
    # 設置隨機種子
    set_seed(config.random_seed)
    
    # 分析數據特性
    if args.analyze:
        analyze_data()
    
    # 初始化數據處理器
    print("\n開始載入和處理數據...")
    data_processor = DataProcessor(config)
    fold_data, df = data_processor.process_pipeline(
        n_folds=config.n_folds,
        augment=args.augment,
        aug_factor=config.augmentation_factor
    )
    
    # 設定計算設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 訓練流程
    model_manager = ModelManager()
    vis_tools = VisualizationTools()
    
    if args.fold >= 0:
        # 訓練單個折
        model, history = train_fold(args.fold, fold_data, device, epochs=args.epochs)
        model_manager.save_model(model, config, {}, f"fold_{args.fold}")
        vis_tools.plot_training_history(
            history,
            save_path=f"{config.results_dir}/fold_{args.fold}_history.png"
        )
    else:
        # 訓練所有折
        all_models, all_histories = [], []
        
        for fold_idx in range(len(fold_data)):
            model, history = train_fold(fold_idx, fold_data, device, epochs=args.epochs)
            all_models.append(model)
            all_histories.append(history)
            model_manager.save_model(model, config, {}, f"fold_{fold_idx}")
        
        # 可視化所有折的驗證損失
        plt.figure(figsize=(10, 6))
        for fold_idx, history in enumerate(all_histories):
            plt.plot(history['val_loss'], label=f'Fold {fold_idx+1}')
        plt.title('所有折的驗證損失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{config.results_dir}/all_folds_val_loss.png")
        
        # 保存最後一個模型作為整體模型
        model_manager.save_model(all_models[-1], config, {}, "hybrid_model")
    
    print("\n訓練完成!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='混合PINN-LSTM模型訓練')
    parser.add_argument('--fold', type=int, default=-1, 
                        help='要訓練的特定折索引，-1表示訓練所有折')
    parser.add_argument('--epochs', type=int, default=200, 
                        help='訓練輪數')
    parser.add_argument('--augment', action='store_true', 
                        help='是否使用數據增強')
    parser.add_argument('--analyze', action='store_true', 
                        help='是否分析數據特性')
    
    args = parser.parse_args()
    main(args)
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
    
    # 檢查目標變數
    target_cols = config.target_cols
    for col in target_cols:
        print(f"\n{col} 統計信息:")
        print(f"範圍: {df[col].min()} - {df[col].max()}")
        print(f"平均值: {df[col].mean()}")
        print(f"標準差: {df[col].std()}")
        print(f"四分位數: {df[col].quantile([0.25, 0.5, 0.75])}")
    
    # 創建結果目錄
    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)
    
    # 繪製目標變數分佈
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['Acc. Equi. Strain Up'], bins=15, alpha=0.7)
    plt.title('上升累積等效應變分佈')
    plt.xlabel('應變值')
    plt.ylabel('頻率')
    
    plt.subplot(1, 2, 2)
    plt.hist(df['Acc. Equi. Strain Down'], bins=15, alpha=0.7)
    plt.title('下降累積等效應變分佈')
    plt.xlabel('應變值')
    plt.ylabel('頻率')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'target_distribution.png'))
    
    # 檢查輸入變數與目標變數的關係
    plt.figure(figsize=(15, 10))
    
    # 檢查Die、PCB與應變的關係
    plt.subplot(2, 2, 1)
    plt.scatter(df['Die'], df['Acc. Equi. Strain Up'], alpha=0.7)
    plt.title('Die vs 上升累積等效應變')
    plt.xlabel('Die')
    plt.ylabel('應變值')
    
    plt.subplot(2, 2, 2)
    plt.scatter(df['PCB'], df['Acc. Equi. Strain Up'], alpha=0.7)
    plt.title('PCB vs 上升累積等效應變')
    plt.xlabel('PCB')
    plt.ylabel('應變值')
    
    plt.subplot(2, 2, 3)
    plt.scatter(df['Total Warpage'], df['Acc. Equi. Strain Up'], alpha=0.7)
    plt.title('Total Warpage vs 上升累積等效應變')
    plt.xlabel('Total Warpage')
    plt.ylabel('應變值')
    
    plt.subplot(2, 2, 4)
    plt.scatter(df['Unit Warpage (No PCB)'], df['Acc. Equi. Strain Up'], alpha=0.7)
    plt.title('Unit Warpage vs 上升累積等效應變')
    plt.xlabel('Unit Warpage')
    plt.ylabel('應變值')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.results_dir, 'feature_target_relation.png'))
    
    print(f"數據分析圖表已保存到 {config.results_dir} 目錄")
    
    return df

def train_fold(fold_idx, fold_data, device, epochs=50):
    """訓練單個折的模型
    
    Args:
        fold_idx: 折索引
        fold_data: 折數據
        device: 計算設備
        epochs: 訓練輪數
        
    Returns:
        訓練好的模型和訓練歷史
    """
    print(f"\n===== 訓練第 {fold_idx+1}/{len(fold_data)} 折 =====")
    
    # 提取訓練數據和測試數據
    train_data = fold_data[fold_idx]['train']
    val_data = fold_data[fold_idx]['test']
    
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
        batch_size=8,  # 小批次大小，適合小樣本數據
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False
    )
    
    # 初始化模型
    model = HybridPINNLSTM(config)
    
    # 創建訓練器
    trainer = SimpleTrainer(model, config, device)
    
    # 訓練模型
    history = trainer.train(train_loader, val_loader, epochs=epochs)
    
    return model, history

def main(args):
    """主函數，協調整個訓練流程"""
    # 設置隨機種子，確保結果可重現
    set_seed(config.random_seed)
    
    # 分析數據特性
    if args.analyze:
        analyze_data()
    
    # 初始化數據處理器
    print("\n開始載入和處理數據...")
    data_processor = DataProcessor(config)
    
    # 處理數據，獲取交叉驗證的數據集
    fold_data, df = data_processor.process_pipeline(
        n_folds=config.n_folds,
        augment=args.augment,
        aug_factor=config.augmentation_factor
    )
    
    # 設定計算設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 如果只訓練單個折
    if args.fold >= 0:
        model, history = train_fold(args.fold, fold_data, device, epochs=args.epochs)
        
        # 保存模型
        model_manager = ModelManager()
        model_manager.save_model(model, config, {}, f"fold_{args.fold}")
        
        # 可視化訓練歷史
        vis_tools = VisualizationTools()
        vis_tools.plot_training_history(
            history,
            save_path=f"{config.results_dir}/fold_{args.fold}_history.png"
        )
    
    # 如果訓練所有折
    else:
        all_models = []
        all_histories = []
        
        for fold_idx in range(len(fold_data)):
            model, history = train_fold(fold_idx, fold_data, device, epochs=args.epochs)
            all_models.append(model)
            all_histories.append(history)
            
            # 保存每個折的模型
            model_manager = ModelManager()
            model_manager.save_model(model, config, {}, f"fold_{fold_idx}")
        
        # 可視化所有折的訓練歷史
        vis_tools = VisualizationTools()
        
        # 繪製所有折的驗證損失
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
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='混合PINN-LSTM模型訓練')
    parser.add_argument('--fold', type=int, default=-1, 
                        help='要訓練的特定折索引，-1表示訓練所有折')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='訓練輪數')
    parser.add_argument('--augment', action='store_true', 
                        help='是否使用數據增強')
    parser.add_argument('--analyze', action='store_true', 
                        help='是否分析數據特性')
    
    args = parser.parse_args()
    
    # 運行主函數
    main(args)
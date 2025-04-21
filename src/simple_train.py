#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
simple_train.py - 簡單模型訓練
使用簡單回歸模型訓練銲錫接點疲勞壽命預測模型。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from sklearn.model_selection import KFold

from src.simple_regression import SimpleRegressionModel
from src.feature_extractor import FeatureExtractor
from src.data_processing import calculate_nf_from_delta_w
from src.utils import set_seed

def main(args):
    # 設定隨機種子
    set_seed(42)
    
    # 創建結果目錄
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 載入數據
    df = pd.read_csv(args.data_path)
    print(f"載入了 {len(df)} 筆數據")
    
    # 對目標變數進行對數轉換，使分佈更接近正態
    if args.log_transform:
        target_cols = ['Acc. Equi. Strain Up', 'Acc. Equi. Strain Down']
        df[target_cols] = df[target_cols].apply(lambda x: np.log1p(x))
        print("對目標變數進行了對數轉換")
    
    # 目標變數
    target_cols = ['Acc. Equi. Strain Up', 'Acc. Equi. Strain Down']
    y = df[target_cols].values
    
    # 初始化特徵提取器
    feature_extractor = FeatureExtractor()
    
    # 提取特徵
    X = feature_extractor.extract_features(df, create_interactions=args.interactions, use_pca=args.use_pca, k_best=args.k_best)
    
    # 訓練交叉驗證
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    
    all_metrics = []
    all_y_true = []
    all_y_pred = []
    all_nf_true = []
    all_nf_pred = []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n==== 訓練第 {fold+1}/{args.n_folds} 折 ====")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 特徵擬合和轉換
        X_train = feature_extractor.fit(X_train, y_train)
        X_test = feature_extractor.transform(df.iloc[test_idx], X_test)
        
        # 創建和訓練模型
        model = SimpleRegressionModel(model_type=args.model_type)
        
        # 是否調整超參數
        if args.tune_params:
            model.tune_hyperparameters(X_train, y_train[:, 0], y_train[:, 1])
        
        # 訓練模型
        model.fit(X_train, y_train)
        
        # 評估模型
        metrics, y_pred, nf_pred = model.evaluate(X_test, y_test)
        
        print("評估指標:")
        print(f"上升應變 - RMSE: {metrics['rmse_up']:.6f}, MAE: {metrics['mae_up']:.6f}, R²: {metrics['r2_up']:.6f}")
        print(f"下降應變 - RMSE: {metrics['rmse_down']:.6f}, MAE: {metrics['mae_down']:.6f}, R²: {metrics['r2_down']:.6f}")
        print(f"疲勞壽命 - RMSE: {metrics['rmse_nf']:.6f}, MAE: {metrics['mae_nf']:.6f}, R²: {metrics['r2_nf']:.6f}")
        
        # 保存模型
        model.save_model(f"fold_{fold}")
        
        # 收集結果
        all_metrics.append(metrics)
        all_y_true.append(y_test)
        all_y_pred.append(y_pred)   
        
        # 計算真實和預測的Nf
        delta_w_true_mean = (y_test[:, 0] + y_test[:, 1]) / 2
        nf_true = np.array([calculate_nf_from_delta_w(dw) for dw in delta_w_true_mean])
        
        all_nf_true.append(nf_true)
        all_nf_pred.append(nf_pred)
        
        # 繪製本折結果
        model.plot_results(y_test, y_pred, nf_true, nf_pred, 
                        save_path=os.path.join(results_dir, f"fold_{fold}_results.png"))
    
    # 合併所有折的結果
    all_y_true = np.vstack(all_y_true)
    all_y_pred = np.vstack(all_y_pred)
    all_nf_true = np.concatenate(all_nf_true)
    all_nf_pred = np.concatenate(all_nf_pred)
    
    # 計算平均指標
    avg_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
    
    print("\n==== 交叉驗證平均結果 ====")
    print(f"上升應變 - RMSE: {avg_metrics['rmse_up']:.6f}, MAE: {avg_metrics['mae_up']:.6f}, R²: {avg_metrics['r2_up']:.6f}")
    print(f"下降應變 - RMSE: {avg_metrics['rmse_down']:.6f}, MAE: {avg_metrics['mae_down']:.6f}, R²: {avg_metrics['r2_down']:.6f}")
    print(f"疲勞壽命 - RMSE: {avg_metrics['rmse_nf']:.6f}, MAE: {avg_metrics['mae_nf']:.6f}, R²: {avg_metrics['r2_nf']:.6f}")
    
    # 訓練最終模型（使用全部數據）
    print("\n==== 訓練最終模型 ====")
    X = feature_extractor.fit(X, y)
    
    final_model = SimpleRegressionModel(model_type=args.model_type)
    
    # 是否調整超參數
    if args.tune_params:
        final_model.tune_hyperparameters(X, y[:, 0], y[:, 1])
    
    # 訓練模型
    final_model.fit(X, y)
    
    # 保存最終模型
    final_model.save_model("final_model")
    
    # 對全部數據進行預測
    y_pred = final_model.predict(X)
    
    # 計算Nf
    delta_w_true_mean = (y[:, 0] + y[:, 1]) / 2
    nf_true = np.array([calculate_nf_from_delta_w(dw) for dw in delta_w_true_mean])
    nf_pred = final_model.calculate_nf(y_pred)
    
    # 繪製整體結果
    final_model.plot_results(y, y_pred, nf_true, nf_pred, 
                        save_path=os.path.join(results_dir, "final_results.png"))
    
    # 如果進行了對數轉換，需要轉換回原始尺度
    if args.log_transform:
        y = np.expm1(y)
        y_pred = np.expm1(y_pred)
        all_y_true = np.expm1(all_y_true)
        all_y_pred = np.expm1(all_y_pred)
        
        # 重新繪製結果
        final_model.plot_results(y, y_pred, nf_true, nf_pred, 
                            save_path=os.path.join(results_dir, "final_results_original_scale.png"))
    
    print("\n訓練完成!")

if __name__ == "__main__":
    # 解析命令行參數
    parser = argparse.ArgumentParser(description='銲錫接點疲勞壽命預測 - 簡單模型訓練')
    parser.add_argument('--data_path', type=str, default='data/Training_data_warpage_final_20250321_v1.2.csv',
                        help='數據文件路徑')
    parser.add_argument('--model_type', type=str, default='ensemble',
                        choices=['linear', 'svm', 'ensemble'],
                        help='模型類型: linear=彈性網絡, svm=支持向量機, ensemble=梯度提升樹')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='交叉驗證折數')
    parser.add_argument('--log_transform', action='store_true',
                        help='是否對目標變數進行對數轉換')
    parser.add_argument('--interactions', action='store_true',
                        help='是否創建特徵交互項')
    parser.add_argument('--use_pca', action='store_true',
                        help='是否使用PCA降維')
    parser.add_argument('--k_best', type=int, default=None,
                        help='選擇的最佳特徵數量')
    parser.add_argument('--tune_params', action='store_true',
                        help='是否調整超參數')
    
    args = parser.parse_args()
    
    # 運行主函數
    main(args)
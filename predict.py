#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
predict.py - 混合PINN-LSTM模型預測程式
本模組實現了使用訓練好的混合模型進行銲錫接點疲勞壽命預測的功能。
主要特點:
1. 載入預處理好的測試數據
2. 使用訓練好的混合PINN-LSTM模型進行預測
3. 透過物理公式從應變差(delta_w)計算疲勞壽命(Nf)
4. 提供預測結果可視化及評估指標
5. 適用於小樣本數據集(81筆)的模型表現分析
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.data_processing import DataProcessor, calculate_nf_from_delta_w
from src.hybrid_model import HybridPINNLSTM
from config import config
from src.utils import ModelManager, VisualizationTools, calculate_metrics, verify_physical_consistency


def load_model_and_data(config, model_path, test_data_path=None):
    """載入模型和測試數據
    
    Args:
        config: 配置參數
        model_path: 模型路徑
        test_data_path: 測試數據路徑，若為None則使用交叉驗證方式
        
    Returns:
        模型、數據處理器和數據集
    """
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridPINNLSTM(config)
    
    # 載入模型權重
    model_manager = ModelManager(model_dir=os.path.dirname(model_path))
    model, meta_data = model_manager.load_model(model, os.path.basename(model_path).split('.')[0])
    model.to(device)
    model.eval()
    
    # 初始化數據處理器
    data_processor = DataProcessor(config)
    
    # 如果指定了測試數據路徑，直接載入測試數據
    if test_data_path:
        df = data_processor.load_data(test_data_path)
        static_features, time_series, targets = data_processor.preprocess_data(df)
        static_norm, time_series_norm, target_norm = data_processor.normalize_data(
            static_features, time_series, targets, fit=True
        )
        test_data = (static_norm, time_series_norm, targets)
        return model, data_processor, test_data, df
    
    # 否則使用交叉驗證方式，返回所有折的數據
    else:
        fold_data, df = data_processor.process_pipeline(
            n_folds=config.n_folds, 
            augment=False  # 預測時不進行數據增強
        )
        return model, data_processor, fold_data, df


def predict_single_sample(model, static_features, time_series, device):
    """預測單個樣本
    
    Args:
        model: 混合PINN-LSTM模型
        static_features: 靜態特徵，形狀為(static_dim,)
        time_series: 時間序列特徵，形狀為(seq_len, feature_dim)
        device: 計算設備
        
    Returns:
        預測結果字典
    """
    # 將特徵轉換為批次格式
    static_batch = torch.FloatTensor(static_features).unsqueeze(0).to(device)
    time_series_batch = torch.FloatTensor(time_series).unsqueeze(0).to(device)
    
    # 進行預測
    with torch.no_grad():
        outputs = model(static_batch, time_series_batch)
        
    # 提取預測結果
    delta_w_pred = outputs['delta_w']
    nf_pred = model.calculate_nf(delta_w_pred)
    
    # 將結果轉換回numpy
    result = {
        'delta_w': delta_w_pred.cpu().numpy()[0],
        'nf': nf_pred.cpu().numpy()[0][0],
        'pinn_out': outputs['pinn_out'].cpu().numpy()[0],
        'lstm_out': outputs['lstm_out'].cpu().numpy()[0],
        'attention_weights': outputs['attention_weights'].cpu().numpy()[0]
    }
    
    return result


def predict_batch(model, static_features, time_series, device):
    """批次預測
    
    Args:
        model: 混合PINN-LSTM模型
        static_features: 批次靜態特徵，形狀為(batch_size, static_dim)
        time_series: 批次時間序列特徵，形狀為(batch_size, seq_len, feature_dim)
        device: 計算設備
        
    Returns:
        批次預測結果字典
    """
    # 將特徵轉換為張量
    static_tensor = torch.FloatTensor(static_features).to(device)
    time_series_tensor = torch.FloatTensor(time_series).to(device)
    
    # 進行預測
    with torch.no_grad():
        outputs = model(static_tensor, time_series_tensor)
        
    # 提取預測結果
    delta_w_pred = outputs['delta_w']
    nf_pred = model.calculate_nf(delta_w_pred)
    
    # 將結果轉換回numpy
    results = {
        'delta_w': delta_w_pred.cpu().numpy(),
        'nf': nf_pred.cpu().numpy().squeeze(),
        'pinn_out': outputs['pinn_out'].cpu().numpy(),
        'lstm_out': outputs['lstm_out'].cpu().numpy(),
        'attention_weights': outputs['attention_weights'].cpu().numpy()
    }
    
    return results


def evaluate_predictions(predictions, targets, df, data_processor, test_indices=None):
    """評估預測結果
    
    Args:
        predictions: 預測結果字典
        targets: 真實目標值
        df: 原始數據DataFrame
        data_processor: 數據處理器
        test_indices: 測試資料的索引，用於交叉驗證
        
    Returns:
        評估指標字典
    """
    # 計算delta_w預測的評估指標
    delta_w_metrics = calculate_metrics(targets, predictions['delta_w'])
    
    # 計算真實Nf值（從真實delta_w計算）
    delta_w_true_up = targets[:, 0]
    delta_w_true_down = targets[:, 1]
    delta_w_true_mean = (delta_w_true_up + delta_w_true_down) / 2
    nf_true = np.array([calculate_nf_from_delta_w(dw) for dw in delta_w_true_mean])
    
    # 檢驗物理一致性
    physical_metrics = verify_physical_consistency(predictions['delta_w'], 
                                                  predictions['pinn_out'], 
                                                  df)
    
    # 計算Nf預測的評估指標
    nf_mse = mean_squared_error(np.log10(np.abs(nf_true)), np.log10(np.abs(predictions['nf'])))
    nf_mae = mean_absolute_error(np.log10(np.abs(nf_true)), np.log10(np.abs(predictions['nf'])))
    nf_r2 = r2_score(np.log10(np.abs(nf_true)), np.log10(np.abs(predictions['nf'])))
    
    nf_metrics = {
        'log_nf_mse': nf_mse,
        'log_nf_mae': nf_mae,
        'log_nf_r2': nf_r2
    }
    
    # 合併所有指標
    all_metrics = {**delta_w_metrics, **nf_metrics, **physical_metrics}
    
    return all_metrics, nf_true


def visualize_results(predictions, targets, nf_true, df, config, case_ids=None):
    """可視化預測結果
    
    Args:
        predictions: 預測結果字典
        targets: 真實目標值
        nf_true: 真實疲勞壽命值
        df: 原始數據DataFrame
        config: 配置參數
        case_ids: 樣本ID列表，若為None則使用所有樣本
    """
    # 創建可視化工具
    vis_tools = VisualizationTools(results_dir=config.results_dir)
    
    # 繪製delta_w預測比較圖
    delta_w_labels = ['Delta W Up', 'Delta W Down']
    vis_tools.plot_prediction_comparison(
        targets, 
        predictions['delta_w'],
        labels=delta_w_labels,
        save_path=os.path.join(config.results_dir, 'delta_w_prediction.png')
    )
    
    # 繪製Nf預測比較圖
    plt.figure(figsize=(10, 6))
    plt.scatter(np.log10(np.abs(nf_true)), np.log10(np.abs(predictions['nf'])), alpha=0.7)
    plt.plot([np.min(np.log10(np.abs(nf_true))), np.max(np.log10(np.abs(nf_true)))], 
             [np.min(np.log10(np.abs(nf_true))), np.max(np.log10(np.abs(nf_true)))], 
             'r--')
    plt.title('疲勞壽命(Nf) - 預測值 vs 真實值 (對數尺度)')
    plt.xlabel('真實值 (log10)')
    plt.ylabel('預測值 (log10)')
    r2 = r2_score(np.log10(np.abs(nf_true)), np.log10(np.abs(predictions['nf'])))
    rmse = np.sqrt(mean_squared_error(np.log10(np.abs(nf_true)), np.log10(np.abs(predictions['nf']))))
    plt.text(0.05, 0.95, f'R² = {r2:.4f}\nRMSE = {rmse:.4f}', 
             transform=plt.gca().transAxes, verticalalignment='top')
    plt.grid(True)
    plt.savefig(os.path.join(config.results_dir, 'nf_prediction.png'))
    plt.show()
    
    # 可視化注意力權重
    time_points = [int(col.split('_')[-1]) for col in config.time_series_up_cols]
    vis_tools.visualize_attention_weights(
        predictions['attention_weights'],
        time_points,
        save_path=os.path.join(config.results_dir, 'attention_weights.png')
    )
    
    # 如果指定了特定樣本，則繪製其預測細節
    if case_ids is not None:
        for case_id in case_ids:
            idx = case_id - 1  # 假設case_id從1開始
            if idx < 0 or idx >= len(targets):
                print(f"警告：Case ID {case_id} 超出範圍")
                continue
                
            # 繪製單個樣本的預測詳情
            plt.figure(figsize=(15, 5))
            
            # 繪製delta_w預測
            plt.subplot(1, 2, 1)
            bar_width = 0.35
            x = np.array([0, 1])
            plt.bar(x - bar_width/2, targets[idx], bar_width, label='真實值')
            plt.bar(x + bar_width/2, predictions['delta_w'][idx], bar_width, label='預測值')
            plt.xticks(x, delta_w_labels)
            plt.ylabel('應變差 (ΔW)')
            plt.title(f'Case {case_id} - 應變差預測')
            plt.legend()
            
            # 繪製Nf預測
            plt.subplot(1, 2, 2)
            true_nf = nf_true[idx]
            pred_nf = predictions['nf'][idx]
            plt.bar([0, 1], [np.log10(np.abs(true_nf)), np.log10(np.abs(pred_nf))], color=['blue', 'orange'])
            plt.xticks([0, 1], ['真實值', '預測值'])
            plt.ylabel('疲勞壽命 (log10 Nf)')
            plt.title(f'Case {case_id} - 疲勞壽命預測')
            
            plt.tight_layout()
            plt.savefig(os.path.join(config.results_dir, f'case_{case_id}_prediction.png'))
            plt.show()


def run_prediction(config, model_path, test_data_path=None, case_ids=None):
    """執行預測流程
    
    Args:
        config: 配置參數
        model_path: 模型路徑
        test_data_path: 測試數據路徑，若為None則使用交叉驗證方式
        case_ids: 要分析的特定樣本ID列表
    """
    # 設定計算設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用設備: {device}")
    
    # 創建結果目錄
    if not os.path.exists(config.results_dir):
        os.makedirs(config.results_dir)
    
    # 載入模型和數據
    print("載入模型和數據...")
    model, data_processor, data, df = load_model_and_data(
        config, model_path, test_data_path
    )
    
    # 如果是使用交叉驗證方式
    if isinstance(data, list):
        print("使用交叉驗證數據進行預測...")
        all_predictions = []
        all_targets = []
        fold_metrics = []
        
        # 對每個折進行預測
        for fold_idx, fold in enumerate(data):
            print(f"預測第 {fold_idx+1}/{len(data)} 折...")
            test_data = fold['test']
            static_features, time_series, targets = test_data
            test_indices = fold['test_indices']  # 獲取測試索引
            
            # 批次預測
            predictions = predict_batch(model, static_features, time_series, device)
            
            # 評估結果，傳入測試索引
            metrics, nf_true = evaluate_predictions(predictions, targets, df, data_processor, test_indices)
            fold_metrics.append(metrics)
            
            # 收集所有預測和目標
            all_predictions.append(predictions)
            all_targets.append(targets)
            
            print(f"第 {fold_idx+1} 折評估指標: RMSE={metrics['rmse_overall']:.4f}, R²={metrics['r2_0']:.4f}")
        
        # 合併所有折的結果
        combined_predictions = {
            'delta_w': np.concatenate([p['delta_w'] for p in all_predictions]),
            'nf': np.concatenate([p['nf'] for p in all_predictions]),
            'pinn_out': np.concatenate([p['pinn_out'] for p in all_predictions]),
            'lstm_out': np.concatenate([p['lstm_out'] for p in all_predictions]),
            'attention_weights': np.concatenate([p['attention_weights'] for p in all_predictions])
        }
        combined_targets = np.concatenate(all_targets)
        
        # 評估合併結果
        overall_metrics, overall_nf_true = evaluate_predictions(
            combined_predictions, combined_targets, df, data_processor
        )
        
        print("\n===== 交叉驗證整體評估結果 =====")
        print(f"應變差 (Delta W) RMSE: {overall_metrics['rmse_overall']:.6f}")
        print(f"應變差 (Delta W) MAE: {overall_metrics['mae_overall']:.6f}")
        print(f"應變差 (Delta W) R² (Up): {overall_metrics['r2_0']:.6f}")
        print(f"應變差 (Delta W) R² (Down): {overall_metrics['r2_1']:.6f}")
        print(f"疲勞壽命 (Nf) 對數RMSE: {overall_metrics['log_nf_mse']:.6f}")
        print(f"疲勞壽命 (Nf) 對數R²: {overall_metrics['log_nf_r2']:.6f}")
        
        # 可視化結果
        visualize_results(combined_predictions, combined_targets, 
                          overall_nf_true, df, config, case_ids)
        
    # 如果是直接使用測試數據
    else:
        print("使用指定測試數據進行預測...")
        static_features, time_series, targets = data
        
        # 批次預測
        predictions = predict_batch(model, static_features, time_series, device)
        
        # 評估結果
        metrics, nf_true = evaluate_predictions(predictions, targets, df, data_processor)
        
        print("\n===== 測試集評估結果 =====")
        print(f"應變差 (Delta W) RMSE: {metrics['rmse_overall']:.6f}")
        print(f"應變差 (Delta W) MAE: {metrics['mae_overall']:.6f}")
        print(f"應變差 (Delta W) R² (Up): {metrics['r2_0']:.6f}")
        print(f"應變差 (Delta W) R² (Down): {metrics['r2_1']:.6f}")
        print(f"疲勞壽命 (Nf) 對數RMSE: {metrics['log_nf_mse']:.6f}")
        print(f"疲勞壽命 (Nf) 對數R²: {metrics['log_nf_r2']:.6f}")
        
        # 可視化結果
        visualize_results(predictions, targets, nf_true, df, config, case_ids)


def export_predictions(predictions, targets, nf_true, df, export_path):
    """匯出預測結果到CSV檔案
    
    Args:
        predictions: 預測結果字典
        targets: 真實目標值
        nf_true: 真實疲勞壽命值
        df: 原始數據DataFrame
        export_path: 匯出路徑
    """
    # 準備匯出數據
    results_df = pd.DataFrame()
    
    # 添加樣本ID
    results_df['Case'] = df['Case'].values
    
    # 添加主要特徵
    for col in ['Die', 'Stud', 'Mold', 'PCB', 'Total Warpage']:
        results_df[col] = df[col].values
    
    # 添加真實和預測的delta_w
    results_df['Delta_W_Up_True'] = targets[:, 0]
    results_df['Delta_W_Down_True'] = targets[:, 1]
    results_df['Delta_W_Up_Pred'] = predictions['delta_w'][:, 0]
    results_df['Delta_W_Down_Pred'] = predictions['delta_w'][:, 1]
    
    # 添加真實和預測的Nf
    results_df['Nf_True'] = nf_true
    results_df['Nf_Pred'] = predictions['nf']
    
    # 計算預測誤差
    results_df['Delta_W_Up_Error'] = results_df['Delta_W_Up_Pred'] - results_df['Delta_W_Up_True']
    results_df['Delta_W_Down_Error'] = results_df['Delta_W_Down_Pred'] - results_df['Delta_W_Down_True']
    results_df['Nf_Log_Error'] = np.log10(np.abs(results_df['Nf_Pred'])) - np.log10(np.abs(results_df['Nf_True']))
    
    # 保存到CSV
    results_df.to_csv(export_path, index=False)
    print(f"預測結果已匯出至 {export_path}")


def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='混合PINN-LSTM模型預測程式')
    parser.add_argument('--model', type=str, default='models/hybrid_model.pt',
                       help='訓練好的模型路徑')
    parser.add_argument('--test_data', type=str, default=None,
                       help='測試數據路徑，若不指定則使用交叉驗證方式')
    parser.add_argument('--cases', type=str, default=None,
                       help='要分析的特定樣本ID，以逗號分隔')
    parser.add_argument('--export', type=str, default=None,
                       help='匯出預測結果的路徑')
    
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行參數
    args = parse_args()
    
    # 處理特定樣本ID列表
    case_ids = None
    if args.cases:
        case_ids = [int(case_id) for case_id in args.cases.split(',')]
    
    # 執行預測
    run_prediction(config, args.model, args.test_data, case_ids)
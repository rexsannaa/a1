#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
data_processing.py - 數據處理與增強模組
本模組實現了數據的載入、預處理、正規化以及資料增強功能，
並使用交叉驗證適用於銲錫接點疲勞壽命預測的小樣本數據集(81筆)。
主要特點:
1. 讀取並處理CSV格式的數據集
2. 特徵提取和正規化處理
3. 使用K折交叉驗證充分利用小樣本數據
4. 對訓練集實現多種資料增強技術
5. 將模型重構為先預測ΔW，再計算Nf的兩階段架構
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
import os

class DataProcessor:
    def __init__(self, config):
        """初始化數據處理器
        
        Args:
            config: 配置參數，包含數據路徑、數據增強方法等
        """
        self.config = config
        from sklearn.preprocessing import StandardScaler
        self.static_scaler = StandardScaler()
        self.time_series_scaler = StandardScaler()
        self.target_scaler = None  # 不使用目標變數的標準化
        
        # 靜態特徵和時間序列特徵定義
        self.static_features = ['Die', 'Stud', 'Mold', 'PCB', 'Total Warpage', 'Unit Warpage (No PCB)']
        self.time_series_features_up = [col for col in config.time_series_up_cols]
        self.time_series_features_down = [col for col in config.time_series_down_cols]
        self.target_features = ['Acc. Equi. Strain Up', 'Acc. Equi. Strain Down']
    
    def load_data(self, file_path=None):
        """載入CSV數據
        
        Args:
            file_path: CSV檔案路徑，如果為None則使用config中的路徑
            
        Returns:
            載入的DataFrame
        """
        if file_path is None:
            file_path = self.config.data_path
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"數據文件不存在: {file_path}")
            
        df = pd.read_csv(file_path)
        print(f"載入了{len(df)}筆數據")
        return df
    
    def preprocess_data(self, df):
        """數據預處理：提取特徵、目標變數，並進行正規化
        
        Args:
            df: 原始DataFrame
            
        Returns:
            處理後的靜態特徵、時間序列特徵和目標變數
        """
        # 提取靜態特徵
        static_features = df[self.static_features].values
        
        # 提取時間序列特徵
        up_cols = [col for col in df.columns if 'NLPLWK_Up_' in col]
        up_cols.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)  # 按時間排序
        
        down_cols = [col for col in df.columns if 'NLPLWK_Down_' in col]
        down_cols.sort(key=lambda x: int(x.split('_')[-1]), reverse=True)  # 按時間排序
        
        # 重塑時間序列數據為(樣本數, 時間步長, 特徵數)的形式
        n_samples = len(df)
        n_timesteps = len(up_cols)
        time_series_data = np.zeros((n_samples, n_timesteps, 2))
        
        # 填充重塑後的時間序列數據
        for i in range(n_timesteps):
            time_series_data[:, i, 0] = df[up_cols[i]]
            time_series_data[:, i, 1] = df[down_cols[i]]
        
        # 提取目標變數 (ΔW): 累積等效應變
        target = df[self.target_features].values
        
        return static_features, time_series_data, target
    
    def normalize_data(self, static_features, time_series_data, target, fit=True):
        # 初始化器裡改用StandardScaler
        # 正規化靜態特徵
        if fit:
            static_norm = self.static_scaler.fit_transform(static_features)
        else:
            static_norm = self.static_scaler.transform(static_features)
        
        # 正規化時間序列
        n_samples = time_series_data.shape[0]
        time_series_flat = time_series_data.reshape(n_samples, -1)
        
        if fit:
            time_series_flat_norm = self.time_series_scaler.fit_transform(time_series_flat)
        else:
            time_series_flat_norm = self.time_series_scaler.transform(time_series_flat)
            
        time_series_norm = time_series_flat_norm.reshape(time_series_data.shape)
        
        # 關鍵改變：不對目標變數進行標準化
        # 目標變數(應變差)是小數值，直接使用原始值
        target_norm = target.copy()
        
        return static_norm, time_series_norm, target_norm
    
    # 修改位置：augment_data方法(大約在第162行)


    def augment_data(self, static_features, time_series_data, target, factor=3):
        """簡化的數據增強"""
        n_samples = static_features.shape[0]
        aug_static = np.zeros((n_samples * factor, static_features.shape[1]))
        aug_time_series = np.zeros((n_samples * factor, 
                                time_series_data.shape[1], 
                                time_series_data.shape[2]))
        aug_target = np.zeros((n_samples * factor, target.shape[1]))
        
        # 先複製原始數據
        aug_static[:n_samples] = static_features
        aug_time_series[:n_samples] = time_series_data
        aug_target[:n_samples] = target
        
        # 簡單噪聲增強
        for i in range(1, factor):
            idx = n_samples * i
            # 針對不同特徵使用不同量級的噪聲
            static_noise = np.random.normal(0, 0.01, static_features.shape)
            ts_noise = np.random.normal(0, 0.01, time_series_data.shape)
            target_noise = np.random.normal(0, 0.001, target.shape)  # 極小的目標噪聲
            
            aug_static[idx:idx+n_samples] = static_features + static_noise
            aug_time_series[idx:idx+n_samples] = time_series_data + ts_noise
            aug_target[idx:idx+n_samples] = np.maximum(0.001, target + target_noise)
        
        return aug_static, aug_time_series, aug_target
    
    def create_crossval_folds(self, static_features, time_series_data, target, n_folds=5, augment=True, aug_factor=5):
        """創建K折交叉驗證的數據集
        
        Args:
            static_features: 靜態特徵
            time_series_data: 時間序列特徵
            target: 目標變數
            n_folds: 交叉驗證的折數
            augment: 是否對訓練集進行增強
            aug_factor: 增強倍數
            
        Returns:
            K折交叉驗證的訓練和測試數據集列表
        """
        # 初始化K折交叉驗證
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.config.random_seed)
        
        # 初始化存儲每折數據的列表
        fold_data = []
        
        # 數據樣本數
        n_samples = static_features.shape[0]
        indices = np.arange(n_samples)
        
        # 生成K折數據
        for train_idx, test_idx in kf.split(indices):
            # 分割訓練集和測試集
            static_train, static_test = static_features[train_idx], static_features[test_idx]
            time_series_train, time_series_test = time_series_data[train_idx], time_series_data[test_idx]
            target_train, target_test = target[train_idx], target[test_idx]
            
            # 正規化訓練集
            static_train_norm, time_series_train_norm, target_train_norm = self.normalize_data(
                static_train, time_series_train, target_train, fit=True
            )
            
            # 使用訓練集的正規化器來正規化測試集
            static_test_norm, time_series_test_norm, target_test_norm = self.normalize_data(
                static_test, time_series_test, target_test, fit=False
            )
            
            # 對訓練集進行資料增強
            if augment:
                static_train_norm, time_series_train_norm, target_train_norm = self.augment_data(
                    static_train_norm, time_series_train_norm, target_train_norm, factor=aug_factor
                )
                print(f"資料增強後訓練集樣本數: {len(static_train_norm)}")
            
            # 添加到fold_data列表
            fold_data.append({
                'train': (static_train_norm, time_series_train_norm, target_train_norm),
                'test': (static_test_norm, time_series_test_norm, target_test_norm),
                'train_indices': train_idx,
                'test_indices': test_idx
            })
        
        return fold_data
    
    def process_pipeline(self, n_folds=5, augment=True, aug_factor=5):
        """完整的數據處理流程，包含交叉驗證
        
        Args:
            n_folds: 交叉驗證的折數
            augment: 是否對訓練集進行增強
            aug_factor: 增強倍數
            
        Returns:
            K折交叉驗證的數據集
        """
        # 載入數據
        df = self.load_data()
        
        # 預處理數據
        static_features, time_series_data, target = self.preprocess_data(df)
        
        # 創建K折交叉驗證數據
        fold_data = self.create_crossval_folds(
            static_features, time_series_data, target, 
            n_folds=n_folds, augment=augment, aug_factor=aug_factor
        )
        
        return fold_data, df
    
    def inverse_transform_target(self, target_normalized):
        """將正規化的目標值轉換回原始尺度
        
        Args:
            target_normalized: 正規化後的目標變數
            
        Returns:
            轉換回原始尺度的目標變數
        """
        return self.target_scaler.inverse_transform(target_normalized)

def calculate_nf_from_delta_w(delta_w, c=-0.55, m=1.36):
    """根據應變差ΔW計算疲勞壽命Nf
    
    使用材料科學中的Coffin-Manson關係：
    Nf = C * (ΔW/2)^m
    
    Args:
        delta_w: 應變差
        c: 係數C
        m: 指數m
        
    Returns:
        計算出的疲勞壽命Nf
    """
    return c * (delta_w/2)**m
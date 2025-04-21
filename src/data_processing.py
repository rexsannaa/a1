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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import os
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, config):
        """初始化數據處理器
        
        Args:
            config: 配置參數，包含數據路徑、數據增強方法等
        """
        self.config = config
        
        # 使用StandardScaler進行特徵標準化
        self.static_scaler = StandardScaler()
        self.time_series_scaler = StandardScaler()
        
        # 重要改變：不對目標變數進行標準化，保留原始值
        self.target_scaler = None
        
        # 特徵定義
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
        
        # 數據檢查
        self.check_data(df)
        
        return df
    
    def check_data(self, df):
        """檢查數據質量和分佈
        
        Args:
            df: 原始DataFrame
        """
        print("\n===== 數據檢查 =====")
        print(f"數據形狀: {df.shape}")
        
        # 檢查目標變數
        for col in self.target_features:
            print(f"\n{col} 統計信息:")
            print(f"範圍: {df[col].min()} - {df[col].max()}")
            print(f"平均值: {df[col].mean()}")
            print(f"標準差: {df[col].std()}")
            print(f"中位數: {df[col].median()}")
            
            # 檢查極端值
            mean = df[col].mean()
            std = df[col].std()
            outliers = df[(df[col] > mean + 3*std) | (df[col] < mean - 3*std)]
            print(f"極端值數量: {len(outliers)}")
            if len(outliers) > 0:
                print(outliers[['Case', col]])
        
        # 檢查缺失值
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print("\n缺失值統計:")
            print(missing[missing > 0])
    
    def preprocess_data(self, df):
        """數據預處理：提取特徵、目標變數
        
        Args:
            df: 原始DataFrame
            
        Returns:
            處理後的靜態特徵、時間序列特徵和目標變數
        """
        # 提取靜態特徵
        static_features = df[self.static_features].values
        
        # 提取時間序列特徵
        up_cols = self.time_series_features_up
        down_cols = self.time_series_features_down
        
        # 重塑時間序列數據為(樣本數, 時間步長, 特徵數)的形式
        n_samples = len(df)
        n_timesteps = len(up_cols)
        time_series_data = np.zeros((n_samples, n_timesteps, 2))
        
        for i in range(n_timesteps):
            time_series_data[:, i, 0] = df[up_cols[i]]
            time_series_data[:, i, 1] = df[down_cols[i]]
        
        # 提取目標變數 (ΔW): 累積等效應變
        target = df[self.target_features].values
        
        return static_features, time_series_data, target
    
    def normalize_data(self, static_features, time_series_data, target, fit=True):
        """數據正規化處理
        
        Args:
            static_features: 靜態特徵
            time_series_data: 時間序列特徵
            target: 目標變數
            fit: 是否擬合標準化器，True用於訓練集，False用於測試集
        
        Returns:
            標準化後的特徵和目標變數
        """
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
        
        # 重要改變：不對目標變數進行標準化，直接使用原始值
        # 這樣可以保留應變差的物理意義，並避免標準化導致的模型預測問題
        target_norm = target.copy()
        
        return static_norm, time_series_norm, target_norm
    
    def augment_data(self, static_features, time_series_data, target, factor=3):
        """增強數據增強方法，專為小樣本銲錫點疲勞壽命預測設計
        
        Args:
            static_features: 靜態特徵
            time_series_data: 時間序列特徵
            target: 目標變數
            factor: 增強倍數
            
        Returns:
            增強後的數據
        """
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
        
        # 計算目標值的均值和標準差，用於確保增強數據分布合理
        target_mean = np.mean(target, axis=0)
        target_std = np.std(target, axis=0)
        
        # 獲取最小和最大目標值，用於限制增強數據的範圍
        target_min = np.min(target, axis=0) * 0.8  # 允許稍微小一些
        target_max = np.max(target, axis=0) * 1.2  # 允許稍微大一些
        
        # 多種增強策略
        for i in range(1, factor):
            idx = n_samples * i
            strategy = i % 4  # 使用4種不同的增強策略
            
            if strategy == 0:
                # 策略1: 平滑噪聲 - 對所有特徵添加相關噪聲
                for j in range(n_samples):
                    # 生成一個共享的噪聲基礎
                    base_noise = np.random.normal(0, 0.01)
                    
                    # 對靜態特徵添加特定噪聲
                    static_noise = np.random.normal(0, 0.01, static_features.shape[1]) + base_noise
                    aug_static[idx+j] = static_features[j] + static_noise
                    
                    # 對時間序列添加時間相關的噪聲
                    time_noise_base = np.random.normal(0, 0.01, time_series_data.shape[2]) + base_noise
                    for t in range(time_series_data.shape[1]):
                        # 噪聲在時間上是相關的
                        time_noise = time_noise_base * (1 - 0.1 * t / time_series_data.shape[1])
                        aug_time_series[idx+j, t, :] = time_series_data[j, t, :] + time_noise
                    
                    # 目標值噪聲，與基礎噪聲相關但較小
                    target_noise = np.random.normal(0, 0.005, target.shape[1]) + base_noise * 0.3
                    aug_target[idx+j] = np.clip(
                        target[j] + target_noise,
                        target_min,
                        target_max
                    )
            
            elif strategy == 1:
                # 策略2: 縮放增強 - 模擬材料特性變化
                for j in range(n_samples):
                    # 生成縮放因子，接近1
                    scale_factor = np.random.uniform(0.9, 1.1)
                    
                    # 縮放靜態特徵，但保持某些物理限制
                    scaled_static = static_features[j].copy()
                    # 只縮放Die, Stud, Mold等幾何參數，不縮放Warpage
                    scaled_static[:4] = scaled_static[:4] * scale_factor
                    aug_static[idx+j] = scaled_static
                    
                    # 對時間序列應用縮放，模擬材料特性變化
                    aug_time_series[idx+j] = time_series_data[j] * scale_factor
                    
                    # 對目標值應用相應的縮放
                    # 應變與幾何尺寸存在非線性關係，使用非線性函數
                    target_scale = 1.0 / (scale_factor ** 0.7)  # 非線性關係
                    aug_target[idx+j] = np.clip(
                        target[j] * target_scale,
                        target_min,
                        target_max
                    )
            
            elif strategy == 2:
                # 策略3: 插值增強 - 生成樣本之間的新數據點
                for j in range(n_samples):
                    # 隨機選擇另一個樣本
                    k = (j + np.random.randint(1, n_samples)) % n_samples
                    
                    # 插值比例
                    alpha = np.random.uniform(0.3, 0.7)
                    
                    # 線性插值靜態特徵
                    aug_static[idx+j] = static_features[j] * alpha + static_features[k] * (1 - alpha)
                    
                    # 插值時間序列
                    aug_time_series[idx+j] = time_series_data[j] * alpha + time_series_data[k] * (1 - alpha)
                    
                    # 插值目標值
                    aug_target[idx+j] = target[j] * alpha + target[k] * (1 - alpha)
            
            else:
                # 策略4: 物理關係增強 - 根據物理關係模擬數據變化
                for j in range(n_samples):
                    # 複製原始特徵
                    new_static = static_features[j].copy()
                    new_time_series = time_series_data[j].copy()
                    
                    # 修改PCB厚度 (索引3)，這會影響應變
                    pcb_factor = np.random.uniform(0.8, 1.2)
                    new_static[3] = new_static[3] * pcb_factor
                    
                    # 調整Warpage (索引4和5)，與PCB厚度成反比關係
                    warpage_factor = 1.0 / (pcb_factor ** 0.5)  # 非線性關係
                    new_static[4:6] = new_static[4:6] * warpage_factor
                    
                    # 調整時間序列數據，反映材料特性變化
                    new_time_series = new_time_series * (1 + (pcb_factor - 1) * 0.3)
                    
                    # 保存新特徵
                    aug_static[idx+j] = new_static
                    aug_time_series[idx+j] = new_time_series
                    
                    # 根據材料力學關係調整目標應變
                    # 厚度越小，應變通常越大
                    strain_factor = 1.0 / (pcb_factor ** 0.6)
                    new_target = target[j] * strain_factor
                    
                    # 確保在合理範圍內
                    aug_target[idx+j] = np.clip(new_target, target_min, target_max)
        
        print(f"數據增強: 從 {n_samples} 筆增加到 {len(aug_static)} 筆")
        
        return aug_static, aug_time_series, aug_target
    
    def create_crossval_folds(self, static_features, time_series_data, target, n_folds=5, augment=True, aug_factor=2):
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
            
            # 添加到fold_data列表
            fold_data.append({
                'train': (static_train_norm, time_series_train_norm, target_train_norm),
                'test': (static_test_norm, time_series_test_norm, target_test_norm),
                'train_indices': train_idx,
                'test_indices': test_idx
            })
        
        return fold_data
    
    def process_pipeline(self, n_folds=5, augment=True, aug_factor=2):
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
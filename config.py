#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
config.py - 配置文件
本模組定義了銲錫接點疲勞壽命預測系統的配置參數，
適用於小樣本數據(81筆)的混合PINN-LSTM模型。
"""

import os

class Config:
    """混合模型配置類"""
    def __init__(self):
        # 數據相關配置
        self.data_path = 'data/Training_data_warpage_final_20250321_v1.2.csv'
        self.random_seed = 42
        
        # 特徵定義
        self.static_dim = 6  # Die, Stud, Mold, PCB, Total Warpage, Unit Warpage
        self.ts_feature_dim = 2  # 每個時間點的特徵數(Up和Down)
        self.time_series_up_cols = ['NLPLWK_Up_14400', 'NLPLWK_Up_10800', 'NLPLWK_Up_7200', 'NLPLWK_Up_3600']
        self.time_series_down_cols = ['NLPLWK_Down_14400', 'NLPLWK_Down_10800', 'NLPLWK_Down_7200', 'NLPLWK_Down_3600']
        self.target_cols = ['Acc. Equi. Strain Up', 'Acc. Equi. Strain Down']
        
        # 資料增強配置
        self.use_augmentation = True
        self.augmentation_factor = 4  # 降低增強倍數，避免過度增強
        
        # 交叉驗證配置
        self.n_folds = 5
        
        # 模型配置
        self.use_dropout = True
        self.lstm_dropout = 0.3
        self.lstm_hidden_dim = 96  # 增加隱藏層維度
        self.lstm_num_layers = 2  # 保持2層LSTM
        
        # 訓練配置
        self.batch_size = 4
        self.epochs = 300  # 增加訓練輪數
        self.learning_rate = 0.0005  # 調整初始學習率
        self.weight_decay = 0.0001  # 增加權重衰減
        self.patience = 40  # 增加耐心值
        
        # 物理模型參數
        self.c_coefficient = -0.55  # Coffin-Manson係數C
        self.m_exponent = 1.36  # Coffin-Manson指數m
        
        # 輸出路徑
        self.model_save_path = 'models/hybrid_model.pt'
        self.results_dir = 'results'
        
# 創建默認配置實例
config = Config()
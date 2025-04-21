#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
lstm_module.py - 長短期記憶網絡模塊
本模組實現了針對銲錫接點疲勞壽命預測的長短期記憶網絡(LSTM)，
從時間序列數據中提取動態特徵並預測應變差(delta_w)。
主要特點:
1. 使用雙向LSTM捕捉時間特徵的長短期依賴
2. 自注意力機制加強關鍵時間點的特徵提取
3. 小樣本數據(81筆)下的特殊正則化設計
4. 專注於預測應變差(delta_w)作為主要輸出
5. 內建的時間序列增強機制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleLSTMModule(nn.Module):
    """簡化版長短期記憶網絡模塊"""
    def __init__(self, config):
        super(SimpleLSTMModule, self).__init__()
        self.config = config
        
        # 使用單層單向LSTM
        self.input_dim = config.ts_feature_dim
        hidden_dim = 16  # 減少隱藏層大小
        
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=hidden_dim,
            num_layers=1,  # 單層
            batch_first=True,
            bidirectional=False  # 單向
        )
        
        # 直接輸出層
        self.fc = nn.Linear(hidden_dim, 2)
        
        # 初始化權重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化權重"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.01)
    
    def forward(self, time_series):
        """前向傳播"""
        # 通過LSTM層
        lstm_out, _ = self.lstm(time_series)
        
        # 使用最後一個時間步的輸出
        last_output = lstm_out[:, -1, :]
        
        # 輸出層
        delta_w = F.softplus(self.fc(last_output)) * 0.1
        
        # 為保持與原模型接口一致，返回一個假的注意力權重
        batch_size = time_series.shape[0]
        seq_len = time_series.shape[1]
        dummy_attention = torch.ones(batch_size, seq_len, 1) / seq_len
        
        return delta_w, dummy_attention
    
    def compute_sequence_loss(self, time_series, delta_w_pred):
        """簡化的序列損失"""
        # 為了相容性，保留但簡化損失計算
        return torch.tensor(0.0, device=delta_w_pred.device)
    

class AttentionLayer(nn.Module):
    """自注意力層，增強重要時間點的特徵提取"""
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        """
        Args:
            lstm_output: LSTM輸出，形狀為(batch_size, seq_len, hidden_size)
            
        Returns:
            加權後的特徵向量和注意力權重
        """
        # 計算注意力分數
        attention_scores = self.attention(lstm_output)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 加權求和得到上下文向量
        context_vector = torch.sum(attention_weights * lstm_output, dim=1)
        
        return context_vector, attention_weights


class LSTMModule(nn.Module):
    """長短期記憶網絡模塊，從時間序列提取動態特徵"""
    def __init__(self, config):
        """初始化LSTM模塊
        
        Args:
            config: 模型配置，包含層結構等參數
        """
        super(LSTMModule, self).__init__()
        self.config = config
        
        # 時間序列特徵維度
        self.input_dim = config.ts_feature_dim  # 時間序列每個點的特徵數
        self.hidden_dim = config.lstm_hidden_dim or 32
        self.num_layers = config.lstm_num_layers or 2
        self.dropout_rate = config.lstm_dropout or 0.2
        
        # 針對小樣本設計的雙向LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        
        # 自注意力層，增強重要時間點的特徵提取
        self.attention = AttentionLayer(self.hidden_dim * 2)  # 雙向LSTM輸出維度為hidden_dim*2
        
        # 全連接層，從LSTM特徵預測delta_w
        fc_dim = config.lstm_fc_dim or 16
        self.fc1 = nn.Linear(self.hidden_dim * 2, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 2)  # 輸出兩個通道：上升和下降的累積等效應變
        
        # 批標準化，穩定小樣本訓練
        self.batch_norm = nn.BatchNorm1d(fc_dim)
        
        # 初始化權重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """針對小樣本的特殊初始化"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)  # 正交初始化，改善梯度傳播
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)
                
        # FC層使用Xavier初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0.01)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0.01)
    
    def forward(self, time_series):
        """前向傳播，從時間序列預測應變差
        
        Args:
            time_series: 時間序列特徵，形狀為(batch_size, seq_len, feature_dim)
            
        Returns:
            應變差的預測值，形狀為(batch_size, 2)
        """
        batch_size = time_series.shape[0]
        
        # 通過LSTM層
        lstm_out, _ = self.lstm(time_series)
        
        # 應用自注意力
        context, attention_weights = self.attention(lstm_out)
        
        # 通過全連接層
        x = F.relu(self.fc1(context))
        x = self.batch_norm(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # 輸出層：預測delta_w
        delta_w = self.fc2(x)
        
        # 使用Softplus確保應變為正值，符合物理意義
        delta_w = F.softplus(delta_w) * 0.2
        
        return delta_w, attention_weights
    
    def compute_sequence_loss(self, time_series, delta_w_pred):
        """計算時間序列特殊損失
        
        Args:
            time_series: 時間序列特徵
            delta_w_pred: 預測的應變差
            
        Returns:
            時間序列特性相關的損失
        """
        # 計算時間序列的趨勢一致性損失
        # 隨著時間變化，應變應該有單調趨勢
        batch_size, seq_len, _ = time_series.shape
        
        # 提取不同時間點的特徵差異
        time_diffs = time_series[:, 1:, :] - time_series[:, :-1, :]
        time_trend = torch.mean(torch.abs(time_diffs), dim=(1, 2))
        
        # 應變預測應該與時間趨勢相關
        trend_loss = torch.mean(torch.abs(delta_w_pred[:, 0] + delta_w_pred[:, 1] - time_trend))
        
        return trend_loss * 0.1
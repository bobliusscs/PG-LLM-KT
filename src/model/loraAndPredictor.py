# 解决NumPy 2.x与wandb的兼容性问题
import numpy as np
# 修复NumPy兼容性问题，注释掉有问题的代码
# if not hasattr(np, 'float_'):
#     np.float_ = np.float64

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional
import gc
from sklearn.metrics import roc_auc_score
import sys
import inspect

# 设置环境变量来避免编译问题
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCH_LOGS"] = "off"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "0"
# 禁用 SafeTensors 以避免张量共享内存问题
os.environ["SAFETENSORS_FAST_GPU"] = "0"

# 移除对unsloth的导入，只使用标准的transformers库
# from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, TrainerCallback, AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('binary_classification_finetune.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdaptiveConfidenceHead(nn.Module):
    """
    自适应置信度预测头
    创新点：结合多层特征和注意力机制的置信度预测，增加层归一化和残差连接
    """
    def __init__(self, hidden_size, num_layers=3, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 输入层归一化
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # 多尺度特征提取（带残差连接）
        self.feature_extractors = nn.ModuleList()
        self.feature_norms = nn.ModuleList()
        self.feature_residuals = nn.ModuleList()
        
        for i in range(num_layers):
            # 残差连接线性层 - 用于调整维度
            if i > 0:
                # 从 hidden_size//2 到 hidden_size//2 的残差连接
                residual_layer = nn.Linear(hidden_size // 2, hidden_size // 2)
                self.feature_residuals.append(residual_layer)
            else:
                # 第一层从 hidden_size 到 hidden_size//2，需要调整残差连接维度
                residual_layer = nn.Linear(hidden_size, hidden_size // 2)
                self.feature_residuals.append(residual_layer)
            
            # 特征提取层
            extractor = nn.Sequential(
                nn.Linear(hidden_size if i == 0 else hidden_size // 2, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(dropout_rate)
            )
            self.feature_extractors.append(extractor)
            
            # 层归一化
            self.feature_norms.append(nn.LayerNorm(hidden_size // 2))
        
        # 自注意力机制用于特征融合
        # 确保embed_dim能被num_heads整除
        embed_dim = hidden_size // 2
        num_heads = 8
        if embed_dim % num_heads != 0:
            # 如果不能整除，调整num_heads为能整除embed_dim的最大因子
            for i in range(8, 0, -1):
                if embed_dim % i == 0:
                    num_heads = i
                    break
        
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout_rate)
        
        # 特征融合网络（带残差连接）
        self.fusion_norm = nn.LayerNorm(embed_dim * num_layers)
        self.fusion_network = nn.Sequential(
            nn.Linear(embed_dim * num_layers, hidden_size // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_size // 4),  # 添加层归一化
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        
        # 输出层归一化
        self.output_norm = nn.LayerNorm(hidden_size // 8)
        
        # 置信度预测（带残差连接）
        self.confidence_predictor = nn.Sequential(
            nn.Linear(hidden_size // 8, 32),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.LayerNorm(32),  # 添加层归一化
            nn.Linear(32, 1)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] (optional)
        Returns:
            confidence_score: [batch_size, 1]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # 使用最后一个非padding token的隐藏状态作为句子表示
        if attention_mask is not None:
            # 获取每个序列中最后一个非padding token的位置
            # 将padding位置设为-1，非padding位置保持原值
            last_token_indices = attention_mask.cumsum(dim=1)[:, -1] - 1
            # 使用高级索引获取对应的隐藏状态
            batch_indices = torch.arange(batch_size, device=hidden_states.device)
            sentence_repr = hidden_states[batch_indices, last_token_indices]  # [batch_size, hidden_size]
        else:
            # 如果没有attention_mask，使用最后几个token的平均作为句子表示
            last_tokens = hidden_states[:, -min(1, seq_len):, :]  # 取最后1个token
            sentence_repr = torch.mean(last_tokens, dim=1)  # [batch_size, hidden_size]
        
        # 输入层归一化
        sentence_repr = self.input_norm(sentence_repr)
        
        # 多尺度特征提取（带残差连接）
        multi_scale_features = []
        current_features = sentence_repr
        
        for i in range(self.num_layers):
            # 残差连接
            residual = current_features
            # 特征提取
            features = self.feature_extractors[i](current_features)
            
            # 使用线性变换调整残差维度后再进行残差连接
            residual_transformed = self.feature_residuals[i](residual)
            if features.shape == residual_transformed.shape:
                features = features + residual_transformed
            
            # 层归一化
            features = self.feature_norms[i](features)
            
            multi_scale_features.append(features)
            current_features = features
        
        # 将特征堆叠用于自注意力
        stacked_features = torch.stack(multi_scale_features, dim=1)  # [batch_size, num_layers, hidden_size//2]
        
        # 自注意力特征融合（带残差连接）
        stacked_features_norm = self.attention_norm(stacked_features)
        attended_features, _ = self.self_attention(
            stacked_features_norm, stacked_features_norm, stacked_features_norm
        )
        attended_features = self.attention_dropout(attended_features)
        attended_features = attended_features + stacked_features  # 残差连接
        
        # 展平并融合
        flattened_features = attended_features.reshape(batch_size, -1)  # [batch_size, num_layers * hidden_size//2]
        flattened_features_norm = self.fusion_norm(flattened_features)
        fused_features = self.fusion_network(flattened_features_norm)  # [batch_size, hidden_size//8]
        
        # 输出层归一化
        fused_features = self.output_norm(fused_features)
        
        # 预测置信度（带残差连接）
        confidence = self.confidence_predictor(fused_features)  # [batch_size, 1]
        
        return confidence


class LinearClassificationHead(nn.Module):
    """
    线性分类头
    简单的线性层分类头，用于二分类任务
    """
    def __init__(self, hidden_size, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Linear classifier
        self.classifier = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.classifier.weight)
        if self.classifier.bias is not None:
            nn.init.zeros_(self.classifier.bias)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] (optional)
        Returns:
            logits: [batch_size, 1]
        """
        # 使用最后一个非padding token的隐藏状态
        if attention_mask is not None:
            # 获取每个序列中最后一个非padding token的位置
            # 将padding位置设为-1，非padding位置保持原值
            last_token_indices = attention_mask.cumsum(dim=1)[:, -1] - 1
            # 使用高级索引获取对应的隐藏状态
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_token_hidden = hidden_states[batch_indices, last_token_indices]  # [batch_size, hidden_size]
        else:
            # 如果没有attention_mask，退回到原来的方式
            last_token_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        
        # Layer normalization
        normalized_hidden = self.layer_norm(last_token_hidden)
        
        # Dropout
        dropped_hidden = self.dropout(normalized_hidden)
        
        # Linear classification
        logits = self.classifier(dropped_hidden)  # [batch_size, 1]
        
        return logits

class DynamicConfidenceHead(nn.Module):
    """
    动态置信度头
    结合动态门控机制和置信度调节的分类头
    """
    def __init__(self, hidden_size, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # 动态门控模块 - 使用LeakyReLU替代ReLU避免梯度消失
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),  # 使用LeakyReLU避免梯度消失
            nn.Dropout(dropout_rate),  # 添加dropout防止过拟合
            nn.Linear(hidden_dim, hidden_size),
            nn.Tanh()  # 使用Tanh替代Sigmoid以获得更强的门控效果
        )
        
        # 分类与置信度分支
        self.classifier = nn.Linear(hidden_size, 1)
        self.confidence = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),  # 使用LeakyReLU避免梯度消失
            nn.Dropout(dropout_rate),  # 添加dropout防止过拟合
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # 添加LayerNorm以稳定训练
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # 添加残差连接增强特征传播
        self.residual_scale = nn.Parameter(torch.ones(1))
        
        # 添加可学习的温度参数用于 logits 调节
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用xavier_normal_初始化，更适合Sigmoid/Tanh激活函数
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] (optional)
        Returns:
            logits: [batch_size, 1]
        """
        # 使用最后一个非padding token的隐藏状态
        if attention_mask is not None:
            # 获取每个序列中最后一个非padding token的位置
            last_token_indices = attention_mask.cumsum(dim=1)[:, -1] - 1
            # 使用高级索引获取对应的隐藏状态
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            h = hidden_states[batch_indices, last_token_indices]  # [batch_size, hidden_size]
        else:
            # 如果没有attention_mask，使用最后一个token
            h = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        
        # 应用LayerNorm稳定训练
        h_normalized = self.layer_norm(h)
        
        # 门控机制
        g = self.gate_net(h_normalized)             # 动态权重
        h_gated = h_normalized * g                  # 门控特征
        
        # 残差连接增强特征传播
        h_residual = h_normalized + self.residual_scale * h_gated
        
        # 分类logits
        z = self.classifier(h_residual)     # logits
        
        # 应用可学习的温度调节
        z = z / self.temperature
        
        # 置信度调节 - 提供更稳定的置信度估计
        confidence_score = self.confidence(h_residual)
        
        # 结合置信度的最终输出
        # 使用置信度作为调节因子，增强高置信度样本的影响
        final_logits = z * (0.8 + 0.2 * confidence_score)
        
        return final_logits

class MLPClassificationHead(nn.Module):
    """
    MLP分类头
    使用多层感知机(Multi-Layer Perceptron)进行二分类
    适用于需要更深层次特征抽取的场景
    """
    def __init__(self, hidden_size, hidden_dim=512, num_layers=3, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 输入层归一化
        self.input_norm = nn.LayerNorm(hidden_size)
        
        # 构建多层感知机
        layers = []
        input_dim = hidden_size
        
        for i in range(num_layers):
            # 每层的输出维度逐渐减小
            if i == 0:
                output_dim = hidden_dim
            elif i == num_layers - 1:
                output_dim = hidden_dim // 4
            else:
                output_dim = hidden_dim // 2
            
            # 添加线性层
            layers.append(nn.Linear(input_dim, output_dim))
            # 添加层归一化
            layers.append(nn.LayerNorm(output_dim))
            # 添加激活函数
            layers.append(nn.GELU())
            # 添加Dropout
            layers.append(nn.Dropout(dropout_rate))
            
            input_dim = output_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # 输出层
        self.output_layer = nn.Linear(hidden_dim // 4, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] (optional)
        Returns:
            logits: [batch_size, 1]
        """
        # 使用最后一个非padding token的隐藏状态
        if attention_mask is not None:
            # 获取每个序列中最后一个非padding token的位置
            last_token_indices = attention_mask.cumsum(dim=1)[:, -1] - 1
            # 使用高级索引获取对应的隐藏状态
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_token_hidden = hidden_states[batch_indices, last_token_indices]  # [batch_size, hidden_size]
        else:
            # 如果没有attention_mask，使用最后一个token
            last_token_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_size]
        
        # 输入层归一化
        normalized_hidden = self.input_norm(last_token_hidden)
        
        # 通过MLP层
        mlp_output = self.mlp(normalized_hidden)
        
        # 输出层
        logits = self.output_layer(mlp_output)  # [batch_size, 1]
        
        return logits

class EnhancedDynamicConfidenceHead(nn.Module):
    """
    增强动态置信度头 - 完全参照DynamicConfidenceHead设计，但表达能力更强
    
    核心设计（参照DynamicConfidenceHead）：
    1. 动态门控机制：多层级门控网络，增强特征选择能力
    2. 置信度调节分支：更深的置信度预测网络
    3. 残差连接：多级残差增强梯度流动
    4. LayerNorm稳定训练：各层均添加归一化
    5. 温度缩放：可学习的logits温度调节
    
    增强点：
    - 门控网络：从单层扩展为多层级联门控
    - 特征提取：添加中间特征变换层
    - 分类器：从单层扩展为多层MLP
    - 置信度网络：更深的网络结构
    """
    def __init__(self, hidden_size, hidden_dim=256, dropout_rate=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.hidden_dim = hidden_dim
        
        # ========== LayerNorm（参照DynamicConfidenceHead） ==========
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # ========== 增强的动态门控模块（参照DynamicConfidenceHead的gate_net，但更深） ==========
        # 第一级门控：粗粒度特征筛选
        self.gate_net_1 = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh()
        )
        
        # 第二级门控：细粒度特征调节
        self.gate_net_2 = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_size),
            nn.Tanh()
        )
        
        # ========== 中间特征变换（增强表达能力） ==========
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # ========== 增强的分类器（参照DynamicConfidenceHead的classifier，但更深） ==========
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 8, 1)
        )
        
        # ========== 增强的置信度分支（参照DynamicConfidenceHead的confidence，但更深） ==========
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.LayerNorm(hidden_dim // 8),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 8, 1),
            nn.Sigmoid()
        )
        
        # ========== 可学习参数（参照DynamicConfidenceHead） ==========
        self.residual_scale = nn.Parameter(torch.ones(1))  # 残差缩放
        self.temperature = nn.Parameter(torch.ones(1))  # 温度参数
        self.gate_alpha = nn.Parameter(torch.ones(1) * 0.5)  # 门控混合权重
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        """优化的权重初始化策略（参照DynamicConfidenceHead）"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用xavier_normal_初始化，更适合Tanh/LeakyReLU激活函数
                nn.init.xavier_normal_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        
        # 参数初始化
        nn.init.constant_(self.residual_scale, 1.0)
        nn.init.constant_(self.temperature, 1.0)
        nn.init.constant_(self.gate_alpha, 0.5)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len] (optional)
        Returns:
            logits: [batch_size, 1]
        """
        batch_size = hidden_states.size(0)
        
        # ========== 提取最后有效token（参照DynamicConfidenceHead） ==========
        if attention_mask is not None:
            last_token_indices = attention_mask.cumsum(dim=1)[:, -1] - 1
            batch_indices = torch.arange(batch_size, device=hidden_states.device)
            h = hidden_states[batch_indices, last_token_indices]
        else:
            h = hidden_states[:, -1, :]
        
        # ========== LayerNorm稳定训练（参照DynamicConfidenceHead） ==========
        h_normalized = self.layer_norm(h)
        
        # ========== 多级门控机制（参照DynamicConfidenceHead的gate_net，但增强为两级） ==========
        # 第一级门控：粗粒度特征筛选
        g1 = self.gate_net_1(h_normalized)  # [batch_size, hidden_dim // 2]
        
        # 第二级门控：细粒度特征调节
        g2 = self.gate_net_2(g1)  # [batch_size, hidden_size]
        
        # 门控特征（参照DynamicConfidenceHead的门控方式）
        h_gated = h_normalized * g2
        
        # ========== 残差连接（参照DynamicConfidenceHead） ==========
        h_residual = h_normalized + self.residual_scale * h_gated
        
        # ========== 中间特征变换（增强表达能力） ==========
        h_transformed = self.feature_transform(h_residual)  # [batch_size, hidden_dim // 2]
        
        # ========== 分类预测（参照DynamicConfidenceHead，但更深） ==========
        logits = self.classifier(h_transformed)  # [batch_size, 1]
        
        # ========== 温度调节（参照DynamicConfidenceHead） ==========
        logits = logits / self.temperature
        
        # ========== 置信度调节（参照DynamicConfidenceHead，但更精细） ==========
        confidence_score = self.confidence(h_transformed)  # [batch_size, 1]
        
        # 结合置信度的最终输出（参照DynamicConfidenceHead的公式）
        # 使用置信度作为调节因子，增强高置信度样本的影响
        final_logits = logits * (0.8 + 0.2 * confidence_score)
        
        return final_logits


class BinaryClassificationModel(nn.Module):
    """带LoRA的二分类模型"""
    def __init__(self, base_model, hidden_size, prediction_head="adaptive", loss_function="focal", focal_alpha=0.25, focal_gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.base_model = base_model
        self.prediction_head_type = prediction_head
        
        # 根据参数选择预测头
        if prediction_head == "adaptive":
            self.prediction_head = AdaptiveConfidenceHead(hidden_size)
        elif prediction_head == "linear":
            self.prediction_head = LinearClassificationHead(hidden_size)
        elif prediction_head == "dynamic":
            self.prediction_head = DynamicConfidenceHead(hidden_size)
        elif prediction_head == "enhanced_dynamic":
            self.prediction_head = EnhancedDynamicConfidenceHead(hidden_size)
        elif prediction_head == "mlp":
            self.prediction_head = MLPClassificationHead(hidden_size)
        else:
            raise ValueError(f"不支持的预测头类型: {prediction_head}")
        
        # 检查预测头是否需要attention_mask
        self.prediction_head_needs_attention_mask = (
            hasattr(self.prediction_head, 'forward') and 
            'attention_mask' in inspect.signature(self.prediction_head.forward).parameters
        )
        
        self.loss_function = loss_function
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight  # 正样本权重，用于加权交叉熵损失
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # 获取基础模型的输出
        # 忽略其他参数如 token_type_ids（某些tokenizer可能会生成）
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # 使用所有层的隐藏状态，但主要关注最后一层
        hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
        
        # 通过预测头
        if self.prediction_head_needs_attention_mask:
            logits = self.prediction_head(hidden_states, attention_mask)  # [batch_size, 1]
        else:
            logits = self.prediction_head(hidden_states)  # [batch_size, 1]
        
        loss = None
        if labels is not None:
            labels = labels.float().view(-1, 1)
            
            if self.loss_function == "focal":
                # 使用Focal Loss来处理类别不平衡问题（如果存在）
                # 使用 binary_cross_entropy_with_logits 来兼容 autocast
                bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
                
                # Focal Loss参数
                alpha = self.focal_alpha  # 类别权重
                gamma = self.focal_gamma   # 难易样本权重
                
                # 计算概率用于 focal weight
                probs = torch.sigmoid(logits)
                pt = torch.where(labels == 1, probs, 1 - probs)
                focal_weight = alpha * (1 - pt) ** gamma
                focal_loss = focal_weight * bce_loss
                
                loss = focal_loss.mean()
            else:
                # 使用带权重的二元交叉熵损失
                pos_weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        
        return {
            'loss': loss,
            'logits': torch.sigmoid(logits),  # 转换为概率值
            'hidden_states': outputs.hidden_states,
            'attentions': getattr(outputs, 'attentions', None)
        }
    
    def gradient_checkpointing_enable(self, **kwargs):
        """启用梯度检查点"""
        if hasattr(self.base_model, 'gradient_checkpointing_enable'):
            self.base_model.gradient_checkpointing_enable(**kwargs)
    
    def gradient_checkpointing_disable(self, **kwargs):
        """禁用梯度检查点"""
        if hasattr(self.base_model, 'gradient_checkpointing_disable'):
            self.base_model.gradient_checkpointing_disable(**kwargs)
    
    def save_pretrained(self, save_directory, **kwargs):
        """保存模型方法"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存基础模型的LoRA权重
        if hasattr(self.base_model, 'save_pretrained'):
            try:
                self.base_model.save_pretrained(save_directory, **kwargs)
            except Exception as e:
                logger.warning(f"基础模型保存失败: {e}")
        
        # 保存自定义的预测头权重
        try:
            prediction_head_path = os.path.join(save_directory, "prediction_head.bin")
            torch.save(self.prediction_head.state_dict(), prediction_head_path)
        except Exception as e:
            logger.error(f"预测头保存失败: {e}")
        
        # 保存模型配置
        config = {
            "model_type": "binary_classification_lora",
            "prediction_head_type": self.prediction_head_type,
            "prediction_head_config": {
                "hidden_size": getattr(self.prediction_head, 'hidden_size', None),
                "num_layers": getattr(self.prediction_head, 'num_layers', None) if hasattr(self.prediction_head, 'num_layers') else None
            }
        }
        
        config_path = os.path.join(save_directory, "model_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=True)


class BinaryClassificationModelWithPrecomputedEmbeddings(nn.Module):
    """使用预计算嵌入的二分类模型（仅用于冻结模式）"""
    def __init__(self, hidden_size, prediction_head="adaptive", loss_function="focal", focal_alpha=0.25, focal_gamma=2.0, pos_weight=1.0):
        super().__init__()
        self.prediction_head_type = prediction_head
        self.hidden_size = hidden_size
        
        # 根据参数选择预测头
        if prediction_head == "adaptive":
            self.prediction_head = AdaptiveConfidenceHead(hidden_size)
        elif prediction_head == "linear":
            self.prediction_head = LinearClassificationHead(hidden_size)
        elif prediction_head == "dynamic":
            self.prediction_head = DynamicConfidenceHead(hidden_size)
        elif prediction_head == "enhanced_dynamic":
            self.prediction_head = EnhancedDynamicConfidenceHead(hidden_size)
        elif prediction_head == "mlp":
            self.prediction_head = MLPClassificationHead(hidden_size)
        else:
            raise ValueError(f"不支持的预测头类型: {prediction_head}")
        
        self.loss_function = loss_function
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.pos_weight = pos_weight
        
    def forward(self, embeddings=None, labels=None, **kwargs):
        """使用预计算的嵌入向量进行前向传播
        
        Args:
            embeddings: 预计算的嵌入向量 [batch_size, hidden_size]
            labels: 标签 [batch_size]
            **kwargs: 其他参数（忽略）
        """
        if embeddings is None:
            raise ValueError("必须提供 embeddings 参数")
        
        # 将嵌入重塑为 [batch_size, 1, hidden_size] 以匹配预测头的期望输入
        # 但对于使用句子级表示的预测头，我们需要直接处理
        batch_size = embeddings.size(0)
        
        # 重塑嵌入以模拟序列输入
        hidden_states = embeddings.unsqueeze(1)  # [batch_size, 1, hidden_size]
        
        # 对于使用最后一个token的预测头，创建一个虚拟的attention_mask
        attention_mask = torch.ones(batch_size, 1, device=embeddings.device, dtype=torch.long)
        
        # 检查预测头是否需要attention_mask
        if hasattr(self.prediction_head, 'forward') and 'attention_mask' in inspect.signature(self.prediction_head.forward).parameters:
            logits = self.prediction_head(hidden_states, attention_mask)
        else:
            logits = self.prediction_head(hidden_states)
        
        loss = None
        if labels is not None:
            labels = labels.float().view(-1, 1)
            
            if self.loss_function == "focal":
                bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
                
                alpha = self.focal_alpha
                gamma = self.focal_gamma
                
                probs = torch.sigmoid(logits)
                pt = torch.where(labels == 1, probs, 1 - probs)
                focal_weight = alpha * (1 - pt) ** gamma
                focal_loss = focal_weight * bce_loss
                
                loss = focal_loss.mean()
            else:
                pos_weight = torch.tensor(self.pos_weight, device=logits.device, dtype=logits.dtype)
                loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)
        
        return {
            'loss': loss,
            'logits': torch.sigmoid(logits)
        }
    
    def gradient_checkpointing_enable(self, **kwargs):
        """启用梯度检查点（预计算嵌入模式下不需要，但保持接口兼容）"""
        pass
    
    def gradient_checkpointing_disable(self, **kwargs):
        """禁用梯度检查点（预计算嵌入模式下不需要，但保持接口兼容）"""
        pass
    
    def save_pretrained(self, save_directory, **kwargs):
        """保存模型方法"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存自定义的预测头权重
        try:
            prediction_head_path = os.path.join(save_directory, "prediction_head.bin")
            torch.save(self.prediction_head.state_dict(), prediction_head_path)
        except Exception as e:
            logger.error(f"预测头保存失败: {e}")
        
        # 保存模型配置
        config = {
            "model_type": "binary_classification_precomputed",
            "prediction_head_type": self.prediction_head_type,
            "hidden_size": self.hidden_size,
            "prediction_head_config": {
                "hidden_size": getattr(self.prediction_head, 'hidden_size', None),
                "num_layers": getattr(self.prediction_head, 'num_layers', None) if hasattr(self.prediction_head, 'num_layers') else None
            }
        }
        
        config_path = os.path.join(save_directory, "model_config.json")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=True)

class EvaluationMonitorCallback(TrainerCallback):
    """评估监控回调"""
    
    def on_step_end(self, args, state, control, **kwargs):
        """在每个训练步结束时检查是否应该进行评估"""
        
        if args.eval_steps is not None and args.eval_steps > 0 and state.global_step % args.eval_steps == 0:
            
            # 强制触发评估
            if not control.should_evaluate:
                print(f"[WARN] 强制设置 should_evaluate=True")
                control.should_evaluate = True
    
    def on_evaluate(self, args, state, control, **kwargs):
        """评估开始时记录"""
        print(f"[EVAL] 开始评估 (第 {state.global_step} 步)")
        logger.info(f"[EVAL] 开始评估 (第 {state.global_step} 步)")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """记录评估结果"""
        if logs and any(key.startswith('eval_') for key in logs.keys()):
            print(f"[EVAL] 评估完成 (第 {state.global_step} 步): {logs}")
            logger.info(f"[EVAL] 评估完成 (第 {state.global_step} 步): {logs}")
    
    def on_train_begin(self, args, state, control, **kwargs):
        """训练开始时记录配置"""
        print(f"[TRAIN] 训练开始: eval_steps={args.eval_steps}, eval_strategy={args.eval_strategy}")
        logger.info(f"[TRAIN] 训练开始: eval_steps={args.eval_steps}, eval_strategy={args.eval_strategy}")

class BinaryClassificationTrainer(Trainer):
    """自定义训练器，添加AUC计算和修复保存问题"""
    
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """评估函数，计算损失和AUC"""
        
        # 修复类型检查问题
        actual_eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        eval_dataset_size = 0
        if actual_eval_dataset is not None:
            try:
                # 使用更安全的方式获取数据集大小
                if isinstance(actual_eval_dataset, dict):
                    # 如果是字典，取第一个值的长度
                    first_value = None
                    if hasattr(actual_eval_dataset, 'values') and callable(getattr(actual_eval_dataset, 'values', None)):
                        first_value = next(iter(actual_eval_dataset.values()), None)
                    if first_value is not None:
                        try:
                            # 对于字典中的值，使用更安全的方式获取大小
                            if isinstance(first_value, (list, tuple, dict)):
                                eval_dataset_size = len(first_value)
                            else:
                                # 对于Dataset对象，使用替代方法
                                eval_dataset_size = sum(1 for _ in first_value)
                        except:
                            eval_dataset_size = 0
                elif isinstance(actual_eval_dataset, (list, tuple)):
                    # 对于基本数据类型，安全地调用len函数
                    try:
                        eval_dataset_size = len(actual_eval_dataset)
                    except:
                        eval_dataset_size = 0
                else:
                    # 对于Dataset对象或其他类型，使用数据加载器获取大小
                    try:
                        # 尝试获取数据集大小，但不直接调用len
                        if hasattr(actual_eval_dataset, '__len__'):
                            eval_dataset_size = sum(1 for _ in actual_eval_dataset)
                        else:
                            eval_dataset_size = 0
                    except:
                        eval_dataset_size = 0
            except:
                eval_dataset_size = 0
    
        if actual_eval_dataset is None:
            return {}
        
        # 确保eval_dataset是Dataset类型且不为空
        if eval_dataset_size == 0:
            return {}
        
        # 修复类型转换问题
        try:
            # 确保传递给get_eval_dataloader的是正确类型
            if isinstance(eval_dataset, dict):
                # 如果是字典类型，取第一个值
                eval_dataloader = self.get_eval_dataloader(list(eval_dataset.values())[0] if eval_dataset else None)
            else:
                eval_dataloader = self.get_eval_dataloader(eval_dataset)
        except Exception as e:
            logger.error(f"获取评估数据加载器失败: {e}")
            return {}
        
        # 临时禁用dropout等
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        if model is not None and hasattr(model, 'eval'):
            model.eval()
        else:
            logger.warning("模型未正确初始化")
            return {}
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_samples = 0
        num_batches = 0
        
        for step, inputs in enumerate(eval_dataloader):
            inputs = self._prepare_inputs(inputs)
            if model is not None:
                with torch.no_grad():
                    outputs = model(**inputs)
                    if 'loss' in outputs and outputs['loss'] is not None:
                        loss = outputs["loss"]
                        logits = outputs["logits"]
                        
                        all_preds.extend(logits.cpu().numpy().flatten())
                        all_labels.extend(inputs["labels"].cpu().numpy().flatten())
                        
                        total_loss += loss.item() if loss is not None else 0.0
                        num_samples += inputs["labels"].size(0)
                        num_batches += 1
        
        # 计算平均损失 - 修复：使用批次平均而不是样本总数
        avg_loss = total_loss / max(num_batches, 1) if num_batches > 0 else 0.0
        
        # 计算AUC - 与评测脚本保持一致
        try:
            if len(all_labels) > 0 and len(set(all_labels)) > 1:
                auc_score = roc_auc_score(all_labels, all_preds)
            else:
                auc_score = 0.5  # 只有一类标签时AUC为0.5
        except ValueError as e:
            logger.warning(f"AUC计算失败: {e}")
            auc_score = 0.5 if len(all_labels) > 0 else 0.0
        except Exception as e:
            logger.warning(f"AUC计算异常: {e}")
            auc_score = 0.5 if len(all_labels) > 0 else 0.0
        
        # 计算准确率、精确率、召回率和F1分数（使用固定阈值0.5）
        if len(all_preds) > 0:
            predictions = [1 if p > 0.5 else 0 for p in all_preds]
            
            # 计算指标
            accuracy = sum([1 for i, j in zip(all_labels, predictions) if i == j]) / max(len(all_labels), 1)
            
            # 计算精确率、召回率和F1分数
            tp = sum([1 for i, j in zip(all_labels, predictions) if i == 1 and j == 1])
            fp = sum([1 for i, j in zip(all_labels, predictions) if i == 0 and j == 1])
            fn = sum([1 for i, j in zip(all_labels, predictions) if i == 1 and j == 0])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        else:
            accuracy = 0.0
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        
        # 输出详细的验证集指标
        print(f"[EVAL] 验证集损失: {avg_loss:.4f}, AUC: {auc_score:.4f}, ACC: {accuracy:.4f}, "
              f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
        logger.info(f"[EVAL] 验证集损失: {avg_loss:.4f}, AUC: {auc_score:.4f}, ACC: {accuracy:.4f}, "
                   f"精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}")
        
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_auc": auc_score,
            f"{metric_key_prefix}_accuracy": accuracy,
            f"{metric_key_prefix}_precision": precision,
            f"{metric_key_prefix}_recall": recall,
            f"{metric_key_prefix}_f1": f1,
        }
        
        return metrics

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        """重写保存方法以避免张量共享内存问题"""
        # 如果未指定输出目录，使用默认输出目录
        if output_dir is None:
            output_dir = self.args.output_dir
        # 确保output_dir不是None
        if output_dir is None:
            logger.error("输出目录未指定")
            return
        
        # 确保目录存在
        os.makedirs(str(output_dir), exist_ok=True)
        
        # 使用我们自定义模型的save_pretrained方法
        try:
            if self.model is not None:
                # 检查模型是否有save_pretrained方法且是可调用的
                model_save_pretrained = getattr(self.model, 'save_pretrained', None)
                if model_save_pretrained is not None and callable(model_save_pretrained):
                    model_save_pretrained(str(output_dir))
                else:
                    # 如果没有save_pretrained方法，尝试直接保存状态字典
                    model_path = os.path.join(str(output_dir), "pytorch_model.bin")
                    torch.save(self.model.state_dict(), model_path)
                    logger.info("直接保存模型状态字典成功")
            else:
                logger.warning("模型未正确初始化")
        except Exception as e:
            logger.warning(f"使用 save_pretrained 保存失败: {e}")
            # 备用方案：手动保存权重
            try:
                # 获取所有需要保存的权重
                save_state_dict = {}
                if self.model is not None:
                    for name, param in self.model.named_parameters():
                        if 'lora' in name.lower() or 'confidence_head' in name:
                            save_state_dict[name] = param.cpu().clone()
                
                # 保存权重
                if len(save_state_dict) > 0 and output_dir is not None:
                    save_path = os.path.join(str(output_dir), "model_weights.bin")
                    torch.save(save_state_dict, save_path)
                else:
                    logger.warning("没有找到需要保存的权重")
            except Exception as e2:
                logger.error(f"手动保存模型权重也失败: {e2}")
        
        # 添加容错机制：确保pytorch_model.bin文件存在，以满足Hugging Face Trainer的期望
        try:
            pytorch_model_path = os.path.join(str(output_dir), "pytorch_model.bin")
            if not os.path.exists(pytorch_model_path):
                # 创建一个空的pytorch_model.bin文件，防止加载时出错
                torch.save({}, pytorch_model_path)
        except Exception as e:
            logger.warning(f"创建pytorch_model.bin文件失败: {e}")
        
        # 调用父类的保存方法以确保检查点的完整性
        try:
            super()._save(output_dir, state_dict)
        except Exception as e:
            logger.warning(f"调用父类保存方法失败: {e}")

    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        执行一个训练步骤，确保梯度裁剪正确应用
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        # 如果loss是元组，提取第一个元素（实际的loss张量）
        if isinstance(loss, tuple):
            loss_tensor = loss[0]
        else:
            loss_tensor = loss

        if self.args.n_gpu > 1:
            if isinstance(loss_tensor, torch.Tensor):
                loss_tensor = loss_tensor.mean()  # mean() for DataParallel

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            if isinstance(loss_tensor, torch.Tensor):
                loss_tensor = loss_tensor / self.args.gradient_accumulation_steps

        # 执行反向传播
        if self.use_apex:
            try:
                # 动态导入apex，避免导入错误
                import importlib
                apex = importlib.import_module('apex')
                amp = getattr(apex, 'amp', None)
                if amp and isinstance(loss_tensor, torch.Tensor):
                    with amp.scale_loss(loss_tensor, self.optimizer) as scaled_loss:
                        scaled_loss.backward()
            except (ImportError, AttributeError):
                # 如果没有安装apex，直接进行反向传播
                if isinstance(loss_tensor, torch.Tensor):
                    loss_tensor.backward()
        elif self.deepspeed and hasattr(self, 'deepspeed') and self.deepspeed is not None:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            if isinstance(loss_tensor, torch.Tensor):
                loss_tensor.backward()
        else:
            if isinstance(loss_tensor, torch.Tensor):
                loss_tensor.backward()

        # 梯度裁剪 - 确保正确应用
        if self.args.max_grad_norm is not None and self.args.max_grad_norm > 0:
            # 尝试进行unscale操作（如果使用了混合精度训练）
            try:
                # 如果使用apex
                if self.use_apex:
                    import importlib
                    apex = importlib.import_module('apex')
                    amp = getattr(apex, 'amp', None)
                    if amp and hasattr(self.optimizer, 'unscale_'):
                        amp.unscale_(self.optimizer)
            except (ImportError, AttributeError):
                # 忽略任何错误
                pass
            
            # 执行梯度裁剪
            if self.model is not None and hasattr(self.model, 'parameters'):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

        if isinstance(loss_tensor, torch.Tensor):
            return loss_tensor.detach()
        # 如果loss不是Tensor，返回一个零张量
        return torch.tensor(0.0, requires_grad=True)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LoRA二分类微调训练脚本")
    parser.add_argument("--model_path", type=str, default="models/cache/qwen/Qwen3-0___6B-Base", 
                       help="基础模型路径")
    parser.add_argument("--trained_model_path", type=str, default="",
                       help="已训练模型路径（用于继续训练，包含预测头权重）")
    parser.add_argument("--data_path", type=str, default="dataSet/data_text/assistments2009/train.json", 
                       help="训练数据路径")
    parser.add_argument("--val_path", type=str, default="", 
                       help="验证数据路径")
    parser.add_argument("--output_dir", type=str, default="models/lora_binary_classification", 
                       help="输出目录")
    parser.add_argument("--max_seq_length", type=int, default=1024, 
                       help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="批次大小")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, 
                       help="梯度累积步数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, 
                       help="基础模型学习率")
    # 添加分类头学习率参数
    parser.add_argument("--head_learning_rate", type=float, default=2e-3, 
                       help="分类头学习率")
    parser.add_argument("--num_epochs", type=int, default=5, 
                       help="训练轮数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, 
                       help="预热比例")
    parser.add_argument("--lora_r", type=int, default=32,
                       help="LoRA秩")
    parser.add_argument("--lora_alpha", type=int, default=64, 
                       help="LoRA缩放因子")  
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                       help="LoRA dropout")
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                       help="权重衰减")
    parser.add_argument("--save_steps", type=int, default=500, 
                       help="保存步数")
    parser.add_argument("--eval_steps", type=int, default=500, 
                       help="评估步数")
    parser.add_argument("--logging_steps", type=int, default=10, 
                       help="日志步数")
    parser.add_argument("--eval_split_ratio", type=float, default=0.03, 
                       help="验证集比例")
    parser.add_argument("--wandb_project", type=str, default="lora-binary-classification", 
                       help="WandB项目名")
    parser.add_argument("--wandb_name", type=str, default="", 
                       help="WandB运行名")
    parser.add_argument("--seed", type=int, default=3407, 
                       help="随机种子")
    parser.add_argument("--early_stopping_patience", type=int, default=5, 
                       help="早停耐心值")
    parser.add_argument("--max_samples", type=int, default=0,
                   help="仅用于调试：最大训练样本数（0=全部）")
    parser.add_argument("--loss_function", type=str, default="focal",
                       help="损失函数类型 (focal, bce, weighted_bce)")
    parser.add_argument("--focal_alpha", type=float, default=0.25,
                       help="Focal Loss的alpha参数")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                       help="Focal Loss的gamma参数")
    parser.add_argument("--pos_weight", type=float, default=1.0,
                       help="正样本权重，用于加权交叉熵损失")
    parser.add_argument("--undersample", action="store_true",
                       help="是否对训练数据进行欠采样以平衡类别")
    parser.add_argument("--no_undersample", action="store_true",
                       help="禁用训练数据的欠采样功能")
    parser.add_argument("--prediction_head", type=str, default="adaptive",
                       help="预测头类型: adaptive(自适应多层特征), linear(简单线性), dynamic(动态门控+置信度), enhanced_dynamic(增强动态门控,适合多数据集)")
    parser.add_argument("--train_only_head", action="store_true",
                       help="是否只训练分类头，冻结模型其他参数")
    parser.add_argument("--train_mode", type=str, default="lora",
                       help="训练模式: freeze(固定权重), lora(LoRA微调), full(全参数微调)",
                       choices=["freeze", "lora", "full"])
    # 添加预计算嵌入参数
    parser.add_argument("--use_precomputed_embeddings", action="store_true",
                       help="是否使用预计算嵌入（仅在freeze模式下有效，可大幅提升训练速度）")
    # 添加梯度裁剪参数
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="梯度裁剪的最大范数，默认为1.0，范围应在0.1到5之间")
    # 添加冻结预测头参数
    parser.add_argument("--freeze_head", action="store_true",
                       help="是否冻结预测头，只训练LoRA适配器（用于继续训练）")
    return parser.parse_args()

def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def precompute_embeddings_for_dataset(model, dataset, batch_size=32, device="cuda"):
    """预计算数据集的嵌入向量
    
    Args:
        model: 基础模型（已冻结参数）
        dataset: 数据集
        batch_size: 批次大小
        device: 设备
    
    Returns:
        embeddings: 嵌入向量张量 [num_samples, hidden_size]
        labels: 标签张量 [num_samples]
    """
    model.eval()
    model = model.to(device)
    
    all_embeddings = []
    all_labels = []
    
    num_batches = (len(dataset) + batch_size - 1) // batch_size
    
    print(f"🔄 开始预计算嵌入向量... (共 {len(dataset)} 个样本)")
    logger.info(f"开始预计算嵌入向量... (共 {len(dataset)} 个样本)")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="预计算嵌入", total=num_batches):
            batch = dataset[i:min(i + batch_size, len(dataset))]
            
            # 获取输入和标签
            input_ids = torch.tensor(batch['input_ids']).to(device)
            attention_mask = torch.tensor(batch['attention_mask']).to(device)
            labels = torch.tensor(batch['labels'])
            
            # 获取模型输出
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            # 提取最后一层的隐藏状态
            hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            # 使用最后一个非padding token的隐藏状态作为句子表示
            last_token_indices = attention_mask.cumsum(dim=1)[:, -1] - 1
            batch_indices = torch.arange(hidden_states.size(0), device=device)
            sentence_embeddings = hidden_states[batch_indices, last_token_indices]  # [batch_size, hidden_size]
            
            # 保存到CPU以节省GPU内存
            all_embeddings.append(sentence_embeddings.cpu())
            all_labels.append(labels)
    
    # 合并所有批次
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    print(f"✅ 嵌入向量预计算完成! 形状: {embeddings.shape}")
    logger.info(f"嵌入向量预计算完成! 形状: {embeddings.shape}")
    
    return embeddings, labels


def save_precomputed_embeddings(embeddings, labels, save_path):
    """保存预计算的嵌入向量到文件"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'embeddings': embeddings,
        'labels': labels
    }, save_path)
    print(f"💾 预计算嵌入已保存到: {save_path}")
    logger.info(f"预计算嵌入已保存到: {save_path}")


def load_precomputed_embeddings(load_path):
    """从文件加载预计算的嵌入向量"""
    if not os.path.exists(load_path):
        return None, None
    
    data = torch.load(load_path)
    print(f"📂 已从 {load_path} 加载预计算嵌入")
    logger.info(f"已从 {load_path} 加载预计算嵌入")
    return data['embeddings'], data['labels']


class PrecomputedEmbeddingsDataset(torch.utils.data.Dataset):
    """使用预计算嵌入的数据集"""
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'embeddings': self.embeddings[idx],
            'labels': self.labels[idx]
        }


class PrecomputedEmbeddingsCollator:
    """为预计算嵌入数据集提供数据整理功能"""
    def __call__(self, features):
        """
        将一批数据整理成模型所需的格式
        
        Args:
            features: 列表，每个元素是 {'embeddings': tensor, 'labels': tensor}
        
        Returns:
            dict: {'embeddings': batch_embeddings, 'labels': batch_labels}
        """
        # 提取embeddings和labels
        embeddings = torch.stack([f['embeddings'] for f in features])
        labels = torch.tensor([f['labels'].item() if isinstance(f['labels'], torch.Tensor) else f['labels'] for f in features])
        
        return {
            'embeddings': embeddings,
            'labels': labels
        }


def undersample_dataset(samples, labels, random_state=3407):
    """对数据集进行欠采样，使正负样本数量平衡"""
    import numpy as np
    
    # 转换为numpy数组以便处理
    labels = np.array(labels)
    
    # 分离正负样本的索引
    positive_indices = np.where(labels == 1)[0]
    negative_indices = np.where(labels == 0)[0]
    
    # 确定少数类样本数量
    min_class_count = min(len(positive_indices), len(negative_indices))
    
    # 如果正样本是少数类
    if len(positive_indices) <= len(negative_indices):
        # 随机选择负样本
        np.random.seed(random_state)
        selected_negative_indices = np.random.choice(negative_indices, min_class_count, replace=False)
        # 合并索引
        selected_indices = np.concatenate([positive_indices, selected_negative_indices])
    else:
        # 随机选择正样本
        np.random.seed(random_state)
        selected_positive_indices = np.random.choice(positive_indices, min_class_count, replace=False)
        # 合并索引
        selected_indices = np.concatenate([selected_positive_indices, negative_indices])
    
    # 根据选择的索引过滤样本
    undersampled_samples = [samples[i] for i in sorted(selected_indices)]
    undersampled_labels = [labels[i] for i in sorted(selected_indices)]
    
    return undersampled_samples, undersampled_labels

def load_and_preprocess_data(data_path: str, eval_split_ratio: float = 0.1, val_path: str = "", max_samples: int = 0, undersample: bool = False):
    """加载和预处理数据 - 确保与评测时一致"""
    # 确保data_path不是None
    if not data_path:
        data_path = "dataSet/data_text/assistments2009/train.json"
    logger.info(f"正在加载训练数据集: {data_path}")
    
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            train_data = json.load(f)
    except Exception as e:
        logger.error(f"加载训练数据集失败: {e}")
        raise
    
    logger.info(f"训练数据集加载完成，共{len(train_data)}条样本")
    
    # 数据质量检查和标签转换 - 训练数据
    valid_train_samples = []
    train_label_counts = {"正确": 0, "错误": 0, "right": 0, "wrong": 0, "其他": 0}
    
    for i, item in enumerate(train_data):
        # 检查必需字段 - 现在检查 system、instruction、output 字段
        if "system" not in item or "instruction" not in item or "output" not in item:
            logger.warning(f"训练样本{i}缺少必要字段 (需要: system, instruction, output)")
            continue
            
        # 将 system 和 instruction 组合成完整的输入文本
        system_text = item["system"].strip()
        instruction_text = item["instruction"].strip()
        combined_text = f"{system_text}\n\n{instruction_text}"
        
        # 标签转换 - 使用 output 字段，与评测时完全一致
        label_str = str(item["output"]).strip().lower()
        if label_str in ["正确", "right", "1"]:
            label = 1
            train_label_counts["正确"] += 1
        elif label_str in ["错误", "wrong", "0"]:
            label = 0
            train_label_counts["错误"] += 1
        else:
            logger.warning(f"训练样本{i}的标签'{item['output']}'无法识别，跳过")
            train_label_counts["其他"] += 1
            continue
        
        valid_train_samples.append({
            "text": combined_text,
            "labels": label
        })
    
    logger.info(f"训练数据标签统计: {train_label_counts}")
    logger.info(f"有效训练样本数量: {len(valid_train_samples)}")

    # 简化日志输出，只显示关键训练参数
    print("\n" + "="*50)
    print("关键训练参数:")
    print("="*50)
    
    # 调试模式，只保留前 max_samples 条样本
    if max_samples > 0 and len(valid_train_samples) > max_samples:
        valid_train_samples = valid_train_samples[:max_samples]
        logger.warning(f"调试模式开启，只保留前 {max_samples} 条训练样本")
    
    # 检查类别平衡
    train_positive_ratio = train_label_counts["正确"] / len(valid_train_samples) if valid_train_samples else 0
    print(f"训练数据正类比例: {train_positive_ratio:.3f}")
    
    # 如果启用欠采样，对训练数据进行欠采样
    if undersample and len(valid_train_samples) > 0:
        train_texts = [item["text"] for item in valid_train_samples]
        train_labels = [item["labels"] for item in valid_train_samples]
        undersampled_texts, undersampled_labels = undersample_dataset(train_texts, train_labels)
        
        # 重新构建训练样本
        valid_train_samples = [{"text": text, "labels": label} for text, label in zip(undersampled_texts, undersampled_labels)]
        
        # 更新标签统计
        undersampled_positive_count = sum(undersampled_labels)
        undersampled_negative_count = len(undersampled_labels) - undersampled_positive_count
        train_label_counts["正确"] = int(undersampled_positive_count)
        train_label_counts["错误"] = int(undersampled_negative_count)
        
        print(f"欠采样后训练数据标签统计: {train_label_counts}")
        print(f"欠采样后有效训练样本数量: {len(valid_train_samples)}")
        
        # 重新计算正类比例
        train_positive_ratio = undersampled_positive_count / len(valid_train_samples) if valid_train_samples else 0
        print(f"欠采样后训练数据正类比例: {train_positive_ratio:.3f}")
    
    # 转换为Dataset格式 - 训练数据
    train_dataset_dict = {
        "text": [item["text"] for item in valid_train_samples],
        "labels": [item["labels"] for item in valid_train_samples],
    }
    
    train_dataset = Dataset.from_dict(train_dataset_dict)
    
    # 如果指定了验证文件路径，加载验证数据
    if val_path and os.path.exists(val_path):
        logger.info(f"正在加载验证数据集: {val_path}")
        
        try:
            with open(val_path, "r", encoding="utf-8") as f:
                val_data = json.load(f)
        except Exception as e:
            logger.error(f"加载验证数据集失败: {e}")
            raise
        
        logger.info(f"验证数据集加载完成，共{len(val_data)}条样本")
        
        # 数据质量检查和标签转换 - 验证数据，与评测时完全一致
        valid_val_samples = []
        val_label_counts = {"正确": 0, "错误": 0, "right": 0, "wrong": 0, "其他": 0}
        
        for i, item in enumerate(val_data):
            # 检查必需字段 - 现在检查 system、instruction、output 字段
            if "system" not in item or "instruction" not in item or "output" not in item:
                logger.warning(f"验证样本{i}缺少必要字段 (需要: system, instruction, output)")
                continue
                
            # 将 system 和 instruction 组合成完整的输入文本
            system_text = item["system"].strip()
            instruction_text = item["instruction"].strip()
            combined_text = f"{system_text}\n\n{instruction_text}"
            
            # 标签转换 - 使用 output 字段，与评测时完全一致
            label_str = str(item["output"]).strip().lower()
            if label_str in ["正确", "right", "1"]:
                label = 1
                val_label_counts["正确"] += 1
            elif label_str in ["错误", "wrong", "0"]:
                label = 0
                val_label_counts["错误"] += 1
            else:
                logger.warning(f"验证样本{i}的标签'{item['output']}'无法识别，跳过")
                val_label_counts["其他"] += 1
                continue
            
            valid_val_samples.append({
                "text": combined_text,
                "labels": label
            })
        
        print(f"验证数据标签统计: {val_label_counts}")
        print(f"有效验证样本数量: {len(valid_val_samples)}")
        
        # 调试模式，只保留前 max_samples 条样本
        if max_samples > 0 and len(valid_val_samples) > max_samples:
            valid_val_samples = valid_val_samples[:max_samples]
            logger.warning(f"调试模式开启，只保留前 {max_samples} 条验证样本")
        
        # 检查类别平衡
        val_positive_ratio = val_label_counts["正确"] / len(valid_val_samples) if valid_val_samples else 0
        print(f"验证数据正类比例: {val_positive_ratio:.3f}")
        
        # 转换为Dataset格式 - 验证数据
        val_dataset_dict = {
            "text": [item["text"] for item in valid_val_samples],
            "labels": [item["labels"] for item in valid_val_samples],
        }
        
        val_dataset = Dataset.from_dict(val_dataset_dict)
        
        train_len = len(train_dataset) if hasattr(train_dataset, '__len__') else 0
        val_len = len(val_dataset) if hasattr(val_dataset, '__len__') else 0
        print(f"数据集加载完成: 训练集: {train_len}条, 验证集: {val_len}条")
        return train_dataset, val_dataset
    
    # 如果没有指定验证文件，使用分割比例创建验证集
    elif eval_split_ratio > 0:
        logger.info(f"开始分割训练数据集，验证集比例: {eval_split_ratio}")
        
        # 检查最小验证集大小
        train_len = len(train_dataset) if hasattr(train_dataset, '__len__') else 0
        min_eval_size = max(10, int(train_len * 0.01))  # 至少10个样本或1%
        expected_eval_size = int(train_len * eval_split_ratio)
        
        if expected_eval_size < min_eval_size:
            logger.warning(f"期望验证集大小({expected_eval_size})小于最小要求({min_eval_size})，调整验证集比例")
            eval_split_ratio = min_eval_size / train_len if train_len > 0 else 0.1
            logger.info(f"调整后验证集比例: {eval_split_ratio:.3f}")
        
        try:
            # 不使用 stratify_by_column，因为 labels 是 Value 类型而不是 ClassLabel
            dataset_dict = train_dataset.train_test_split(test_size=eval_split_ratio, seed=3407)
            train_dataset = dataset_dict["train"]
            eval_dataset = dataset_dict["test"]
            
            train_dataset_len = len(train_dataset) if hasattr(train_dataset, '__len__') else 0
            eval_dataset_len = len(eval_dataset) if hasattr(eval_dataset, '__len__') else 0
            
            print(f"数据集分割成功: 训练集: {train_dataset_len}条, 验证集: {eval_dataset_len}条")
            
            # 验证集大小检查
            if eval_dataset_len == 0:
                logger.error("验证集为空！将禁用评估功能")
                return train_dataset, None
            elif eval_dataset_len < 5:
                logger.warning(f"验证集过小({eval_dataset_len}条)，可能影响评估准确性")
            
            # 检查分割后的类别分布
            # 使用更安全的方式获取标签
            train_labels = []
            eval_labels = []
            
            # 获取训练集标签
            try:
                if hasattr(train_dataset, '__iter__'):
                    for item in train_dataset:
                        if isinstance(item, dict) and 'labels' in item:
                            train_labels.append(item['labels'])
            except Exception as e:
                logger.warning(f"获取训练集标签时出错: {e}")
            
            # 获取验证集标签
            try:
                if hasattr(eval_dataset, '__iter__'):
                    for item in eval_dataset:
                        if isinstance(item, dict) and 'labels' in item:
                            eval_labels.append(item['labels'])
            except Exception as e:
                logger.warning(f"获取验证集标签时出错: {e}")
            
            train_positive_ratio = sum(train_labels) / len(train_labels) if len(train_labels) > 0 else 0
            eval_positive_ratio = sum(eval_labels) / len(eval_labels) if len(eval_labels) > 0 else 0
            print(f"训练集正类比例: {train_positive_ratio:.3f}, 验证集正类比例: {eval_positive_ratio:.3f}")
            
            # 检查验证集类别平衡
            if eval_positive_ratio < 0.1 or eval_positive_ratio > 0.9:
                logger.warning(f"验证集类别严重不平衡(正类比例: {eval_positive_ratio:.3f})，可能影响评估结果")
            
            return train_dataset, eval_dataset
            
        except Exception as e:
            logger.error(f"数据集分割失败: {e}")
            logger.info("将使用全部数据作为训练集，禁用评估功能")
            return train_dataset, None
    else:
        train_len = len(train_dataset) if hasattr(train_dataset, '__len__') else 0
        logger.info(f"未设置验证集分割，训练集: {train_len}条")
        return train_dataset, None

def tokenize_function(examples, tokenizer, max_length):
    """分词函数，从左边截断，保留最后max_length个token"""
    # 保存原始的截断方向
    original_truncation_side = tokenizer.truncation_side if hasattr(tokenizer, 'truncation_side') else 'right'
    
    # 设置为从左边截断
    if hasattr(tokenizer, 'truncation_side'):
        tokenizer.truncation_side = 'left'
    
    # 进行分词和截断
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 恢复原始的截断方向
    if hasattr(tokenizer, 'truncation_side'):
        tokenizer.truncation_side = original_truncation_side
    
    return tokenized

def initialize_model_and_tokenizer(model_path: str, max_seq_length: int, lora_r: int, 
                                 lora_alpha: int, lora_dropout: float, 
                                 prediction_head: str = "adaptive",
                                 loss_function: str = "focal", focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                                 pos_weight: float = 1.0,
                                 train_only_head: bool = False,
                                 train_mode: str = "lora"):
    """初始化模型和分词器"""
    if not model_path:
        model_path = "models/cache/qwen/Qwen3-0___6B"
    logger.info(f"正在加载模型: {model_path}")
    
    try:
        # 规范化模型路径 - 确保是绝对路径
        import os
        if not os.path.isabs(model_path):
            # 如果是相对路径,转换为绝对路径
            model_path = os.path.abspath(model_path)
        
        # 确保路径存在
        if not os.path.exists(model_path):
            raise ValueError(f"模型路径不存在: {model_path}")
        
        logger.info(f"使用绝对路径: {model_path}")
        
        # 加载基础模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True  # 信任本地代码
        )
        
        model_kwargs = {
            "trust_remote_code": True,  # 信任本地代码
            "device_map": "cuda:0",
        }
        
        # 确保训练时与验证时使用相同的精度设置
        if torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # 根据训练模式配置模型
        if train_mode == "lora":
            # 配置
            from peft import get_peft_model, LoraConfig, TaskType
            
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            )
            
            base_model = get_peft_model(base_model, peft_config)
            logger.info("使用LoRA微调模式")
        elif train_mode == "freeze":
            # 冻结所有基础模型参数
            for param in base_model.parameters():
                param.requires_grad = False
            logger.info("使用冻结权重模式，只训练分类头")
        elif train_mode == "full":
            # 全参数微调，不需要特殊处理
            logger.info("使用全参数微调模式")
        
        # 如果只训练分类头，则冻结模型的其他参数
        if train_only_head:
            # 冻结LoRA参数或基础模型参数
            for name, param in base_model.named_parameters():
                if "lora" in name.lower() or train_mode == "freeze":
                    param.requires_grad = False
            logger.info("已冻结基础模型参数，只训练分类头")
        
        # 获取隐藏层大小 - 使用更安全的方式
        hidden_size = 1024  # 默认值
        try:
            config = getattr(base_model, 'config', None)
            if config is not None:
                if hasattr(config, 'hidden_size'):
                    hidden_size = config.hidden_size
                elif hasattr(config, 'd_model'):
                    hidden_size = config.d_model
        except:
            hidden_size = 1024  # 默认值
            
        logger.info(f"模型隐藏层大小: {hidden_size}")
        # 只有在不是只训练分类头且使用LoRA时才显示LoRA配置
        if not train_only_head and train_mode == "lora":
            logger.info(f"LoRA配置: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
        elif train_only_head:
            logger.info("基础模型参数已冻结，仅训练分类头")
        else:
            logger.info(f"训练模式: {train_mode}")
        
        # 创建二分类模型，传入预测头类型参数
        model = BinaryClassificationModel(base_model, hidden_size, prediction_head, loss_function, focal_alpha, focal_gamma, pos_weight)
        
        # 确保模型在GPU上
        if torch.cuda.is_available():
            model = model.cuda()
            logger.info(f"模型已移动到GPU")
        
        logger.info("模型加载完成")
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise

def setup_wandb(args, train_dataset, eval_dataset):
    """设置WandB为离线模式"""
    # 设置WandB为离线模式
    os.environ["WANDB_MODE"] = "offline"
    
    if args.wandb_name is None:
        args.wandb_name = f"lora-binary-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # 准备WandB配置
    wandb_config = {
        "model_path": args.model_path,
        "max_seq_length": args.max_seq_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "train_dataset_size": len(train_dataset),
        "eval_dataset_size": len(eval_dataset) if eval_dataset else 0,
        "prediction_head": "AdaptiveConfidenceHead",
        "loss_function": "FocalLoss",
        "focal_alpha": args.focal_alpha,
        "focal_gamma": args.focal_gamma,
    }
    
    # 添加分类头学习率信息（如果不同）
    if hasattr(args, 'head_learning_rate') and args.head_learning_rate != args.learning_rate:
        wandb_config["head_learning_rate"] = args.head_learning_rate
    
    try:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            settings=wandb.Settings(init_timeout=300),
            config=wandb_config,
            mode="offline"  # 强制离线模式
        )
        logger.info("WandB已设置为离线模式")
    except Exception as e:
        logger.warning(f"WandB初始化失败，继续离线训练: {e}")

def create_training_args(args, train_dataset, eval_dataset):
    """创建训练参数"""
    # 计算总步数
    steps_per_epoch = len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    logger.info(f"数据集大小: {len(train_dataset)}")
    logger.info(f"批次大小: {args.batch_size}")
    logger.info(f"梯度累积步数: {args.gradient_accumulation_steps}")
    logger.info(f"每个epoch的步数: {steps_per_epoch}")
    logger.info(f"总训练步数: {total_steps}")
    logger.info(f"预热步数: {warmup_steps}")
    
    # 调试：打印评估相关参数
    print(f"[DEBUG] 调试: eval_dataset存在: {eval_dataset is not None}")
    logger.info(f"[DEBUG] 调试: eval_dataset存在: {eval_dataset is not None}")
    if eval_dataset:
        print(f"[DEBUG] 调试: eval_steps={args.eval_steps}")
        print(f"[DEBUG] 调试: eval_strategy=steps")
        logger.info(f"[DEBUG] 调试: eval_steps={args.eval_steps}")
        logger.info(f"[DEBUG] 调试: eval_strategy=steps")
    else:
        print(f"[DEBUG] 调试: eval_steps=None (因为eval_dataset为None)")
        print(f"[DEBUG] 调试: eval_strategy=no")
        logger.info(f"[DEBUG] 调试: eval_steps=None (因为eval_dataset为None)")
        logger.info(f"[DEBUG] 调试: eval_strategy=no")
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        warmup_steps=warmup_steps,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        load_best_model_at_end=True if eval_dataset else False,
        metric_for_best_model="eval_auc" if eval_dataset else None,
        greater_is_better=True if eval_dataset else None,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        seed=args.seed,
        report_to="wandb",
        run_name=args.wandb_name,
        logging_dir=f"{args.output_dir}/logs",
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        max_grad_norm=max(0.1, min(5.0, args.max_grad_norm)),  # 修正：限制在0.1到5.0之间，与train_unified.py保持一致
        dataloader_num_workers=0,
        torch_compile=False,
        save_safetensors=False,  # 禁用 safetensors，使用传统的 PyTorch 格式
        eval_accumulation_steps=1,  # 确保评估时累积步数为1
        save_total_limit=1,  # 只保留一个检查点（最佳模型）
    )
    
    # 调试：验证训练参数
    print(f"[DEBUG] 调试: 最终eval_steps={training_args.eval_steps}")
    print(f"[DEBUG] 调试: 最终eval_strategy={training_args.eval_strategy}")
    logger.info(f"[DEBUG] 调试: 最终eval_steps={training_args.eval_steps}")
    logger.info(f"[DEBUG] 调试: 最终eval_strategy={training_args.eval_strategy}")
    
    return training_args

def main():
    """主函数"""
    args = parse_args()
    
    # 初始化变量
    model = None
    trainer = None
    tokenizer = None
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("="*60)
    logger.info("LoRA二分类微调训练 - 可选择预测头")
    logger.info("="*60)
    logger.info(f"模型路径: {args.model_path}")
    logger.info(f"训练数据路径: {args.data_path}")
    logger.info(f"验证数据路径: {args.val_path}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"预测头: {args.prediction_head}")
    logger.info(f"训练模式: {args.train_mode}")
    logger.info(f"只训练分类头: {args.train_only_head}")
    # 添加学习率信息
    if hasattr(args, 'head_learning_rate') and args.head_learning_rate != args.learning_rate:
        logger.info(f"基础模型学习率: {args.learning_rate}")
        logger.info(f"分类头学习率: {args.head_learning_rate}")
    else:
        logger.info(f"学习率: {args.learning_rate}")
    
    try:
        # 1. 加载和预处理数据
        print(f"[DEBUG] 开始加载数据，eval_split_ratio={args.eval_split_ratio}")
        logger.info(f"[DEBUG] 开始加载数据，eval_split_ratio={args.eval_split_ratio}")
        # 如果禁用了欠采样，则不进行欠采样
        undersample_flag = getattr(args, 'undersample', False) and not getattr(args, 'no_undersample', False)
        train_dataset, eval_dataset = load_and_preprocess_data(args.data_path, args.eval_split_ratio, args.val_path, max_samples=args.max_samples, undersample=undersample_flag)
        print(f"[DEBUG] 数据加载完成，train_dataset={len(train_dataset) if train_dataset else 'None'}, eval_dataset={len(eval_dataset) if eval_dataset else 'None'}")
        logger.info(f"[DEBUG] 数据加载完成，train_dataset={len(train_dataset) if train_dataset else 'None'}, eval_dataset={len(eval_dataset) if eval_dataset else 'None'}")
        
        # 1.5. 检查是否启用预计算嵌入模式
        # 在freeze模式下自动启用预计算嵌入（因为freeze模式只训练分类头）
        use_precomputed = (args.train_mode == "freeze") or getattr(args, 'use_precomputed_embeddings', False)
        
        # 如果启用预计算嵌入，先检查缓存
        train_embeddings, train_labels = None, None
        eval_embeddings, eval_labels = None, None
        cache_exists = False
        
        if use_precomputed:
            logger.info("🚀 启用预计算嵌入模式，将先预计算所有数据的嵌入向量...")
            print("\n" + "="*60)
            print("🚀 启用预计算嵌入模式")
            print("✨ 可大幅提升多epoch训练速度（预计提升5-10倍）")
            print("="*60 + "\n")
            
            # 生成缓存文件路径
            cache_dir = os.path.join(args.output_dir, "embeddings_cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            # 根据数据路径生成唯一缓存文件名
            import hashlib
            train_cache_key = hashlib.md5(args.data_path.encode()).hexdigest()
            val_cache_key = hashlib.md5(args.val_path.encode()).hexdigest() if args.val_path else "no_val"
            
            train_cache_file = os.path.join(cache_dir, f"train_{train_cache_key}.pt")
            val_cache_file = os.path.join(cache_dir, f"val_{val_cache_key}.pt") if args.val_path else None
            
            # 尝试从缓存加载
            train_embeddings, train_labels = load_precomputed_embeddings(train_cache_file)
            if train_embeddings is not None:
                cache_exists = True
                print("\n✨ 检测到缓存的训练集嵌入向量，直接加载！")
                
                # 加载验证集缓存（如果存在）
                if eval_dataset and val_cache_file:
                    eval_embeddings, eval_labels = load_precomputed_embeddings(val_cache_file)
        
        # 2. 初始化模型和分词器（只在需要时）
        # 如果有缓存，就不需要加载完整模型和分词器
        if not (use_precomputed and cache_exists):
            # 如果指定了已训练模型路径，先尝试从中读取预测头类型
            prediction_head_type = args.prediction_head  # 默认使用命令行参数
            if args.trained_model_path:
                model_config_path = os.path.join(args.trained_model_path, "model_config.json")
                if os.path.exists(model_config_path):
                    try:
                        with open(model_config_path, 'r', encoding='utf-8') as f:
                            model_config = json.load(f)
                            if "prediction_head_type" in model_config:
                                prediction_head_type = model_config["prediction_head_type"]
                                logger.info(f"从已训练模型配置中读取预测头类型: {prediction_head_type}")
                    except Exception as e:
                        logger.warning(f"读取模型配置失败，使用命令行参数: {e}")
            
            model, tokenizer = initialize_model_and_tokenizer(
                args.model_path, args.max_seq_length, args.lora_r, args.lora_alpha, args.lora_dropout,
                prediction_head_type,  # 使用从配置文件读取的预测头类型
                args.loss_function, args.focal_alpha, args.focal_gamma, args.pos_weight,  # 添加损失函数相关参数
                args.train_only_head,  # 添加只训练分类头参数
                args.train_mode  # 添加训练模式参数
            )
            
            # 如果指定了已训练模型路径，加载预测头权重
            if args.trained_model_path:
                prediction_head_path = os.path.join(args.trained_model_path, "prediction_head.bin")
                if os.path.exists(prediction_head_path):
                    logger.info(f"正在从 {prediction_head_path} 加载已训练的预测头权重...")
                    try:
                        state_dict = torch.load(prediction_head_path, map_location='cpu')
                        model.prediction_head.load_state_dict(state_dict)
                        logger.info("✅ 预测头权重加载成功")
                        
                        # 如果需要冻结预测头，则冻结所有预测头参数
                        if hasattr(args, 'freeze_head') and args.freeze_head:
                            for param in model.prediction_head.parameters():
                                param.requires_grad = False
                            logger.info("❄️ 预测头已冻结，只训练LoRA适配器")
                            print("❄️ 预测头已冻结，只训练LoRA适配器")
                    except Exception as e:
                        logger.error(f"❌ 预测头权重加载失败: {e}")
                        raise
                else:
                    logger.warning(f"⚠️ 预测头权重文件不存在: {prediction_head_path}")
            
            # 3. 分词处理（只在需要时）
            logger.info("对数据进行分词处理...")
            train_dataset = train_dataset.map(
                lambda x: tokenize_function(x, tokenizer, args.max_seq_length),
                batched=True,
                remove_columns=["text"]
            )
            
            if eval_dataset:
                eval_dataset = eval_dataset.map(
                    lambda x: tokenize_function(x, tokenizer, args.max_seq_length),
                    batched=True,
                    remove_columns=["text"]
                )
            
            # 打印分词后的数据示例
            logger.info("打印分词后数据的最后10个token...")
            print("\n" + "="*60)
            print("分词后数据示例 (最后10个token):")
            print("="*60)
            
            # 随机选择2个样本进行显示
            import random
            num_samples = min(2, len(train_dataset))
            sample_indices = random.sample(range(len(train_dataset)), num_samples)
            
            for i, idx in enumerate(sample_indices):
                sample = train_dataset[idx]
                if "input_ids" in sample:
                    input_ids = sample["input_ids"]
                else:
                    input_ids = list(sample.values())[0] if sample else []
                
                # 获取最后10个token
                if isinstance(input_ids, list):
                    last_10_tokens = input_ids[-10:] if len(input_ids) >= 10 else input_ids
                elif hasattr(input_ids, 'tolist'):
                    input_ids_list = input_ids.tolist()
                    last_10_tokens = input_ids_list[-10:] if len(input_ids_list) >= 10 else input_ids_list
                else:
                    last_10_tokens = []
                
                try:
                    tokens = tokenizer.convert_ids_to_tokens(last_10_tokens)
                except Exception as e:
                    tokens = ["Error converting tokens"]
                    logger.warning(f"Token转换错误: {e}")
                
                print(f"样本 {i+1}:")
                print(f"  Token IDs: {last_10_tokens}")
                print(f"  Tokens: {tokens}")
                print("-" * 40)
            
            print("="*60 + "\n")
        else:
            print("✨ 检测到缓存，跳过模型加载和分词步骤\n")
            logger.info("检测到缓存，跳过模型加载和分词步骤")
        
        # 3.5. 处理预计算嵌入（如果需要）
        if use_precomputed:
            if not cache_exists:
                # 缓存不存在，需要预计算
                print("\n🔄 阶段1: 预计算嵌入向量")
                print("-" * 60)
                
                # 获取基础模型（用于预计算）
                base_model_for_precompute = model.base_model
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # 预计算训练数据集嵌入
                print("\n📋 预计算训练集嵌入:")
                train_embeddings, train_labels = precompute_embeddings_for_dataset(
                    base_model_for_precompute, 
                    train_dataset, 
                    batch_size=args.batch_size * 2,
                    device=device
                )
                
                # 保存训练集嵌入到缓存
                save_precomputed_embeddings(train_embeddings, train_labels, train_cache_file)
                
                # 预计算验证数据集嵌入（如果存在）
                if eval_dataset and val_cache_file:
                    print("\n📋 预计算验证集嵌入:")
                    eval_embeddings, eval_labels = precompute_embeddings_for_dataset(
                        base_model_for_precompute,
                        eval_dataset,
                        batch_size=args.batch_size * 2,
                        device=device
                    )
                    # 保存验证集嵌入到缓存
                    save_precomputed_embeddings(eval_embeddings, eval_labels, val_cache_file)
                
                print("\n" + "-" * 60)
                print("✅ 阶段1完成: 嵌入向量预计算完成")
            
            # 创建预计算嵌入数据集
            train_dataset = PrecomputedEmbeddingsDataset(train_embeddings, train_labels)
            if eval_embeddings is not None:
                eval_dataset = PrecomputedEmbeddingsDataset(eval_embeddings, eval_labels)
            
            # 释放基础模型以节省内存
            if 'model' in locals():
                del model
            if 'base_model_for_precompute' in locals():
                del base_model_for_precompute
            torch.cuda.empty_cache()
            gc.collect()
            
            # 获取隐藏层大小
            hidden_size = train_embeddings.size(1)
            logger.info(f"预计算嵌入维度: {hidden_size}")
            
            # 创建专用于预计算嵌入的模型
            print("\n🔄 阶段2: 创建分类头模型")
            print("-" * 60)
            model = BinaryClassificationModelWithPrecomputedEmbeddings(
                hidden_size=hidden_size,
                prediction_head=args.prediction_head,
                loss_function=args.loss_function,
                focal_alpha=args.focal_alpha,
                focal_gamma=args.focal_gamma,
                pos_weight=args.pos_weight
            )
            
            # 确保模型在GPU上
            if torch.cuda.is_available():
                model = model.cuda()
            
            print("✅ 分类头模型创建完成")
            print("-" * 60)
            logger.info("✅ 预计算嵌入模型创建完成")
            
            print("\n" + "="*60)
            print("✅ 预处理完成，即将开始训练")
            print("🚀 训练速度将显著提升！")
            print("="*60 + "\n")
        
        # 4. 设置WandB
        setup_wandb(args, train_dataset, eval_dataset)
        
        # 5. 创建训练参数
        training_args = create_training_args(args, train_dataset, eval_dataset)
        
        # 6. 创建训练器
        callbacks = []
        
        # 添加评估监控回调
        eval_monitor_callback = EvaluationMonitorCallback()
        callbacks.append(eval_monitor_callback)
        logger.info("已添加评估监控回调")
        
        if eval_dataset and args.early_stopping_patience > 0:
            early_stopping_callback = EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=0.001
            )
            callbacks.append(early_stopping_callback)
            logger.info(f"已启用早停机制: patience={args.early_stopping_patience}")
        
        # 验证数据集配置
        if eval_dataset:
            print(f"验证集大小: {len(eval_dataset)}")
            print(f"评估步数: {args.eval_steps}")
            print(f"评估策略: steps")
            print("[INFO] 评估功能已启用")
            logger.info(f"验证集大小: {len(eval_dataset)}")
            logger.info(f"评估步数: {args.eval_steps}")
            logger.info(f"评估策略: steps")
            logger.info("[INFO] 评估功能已启用")
        else:
            print("[ERROR] 没有验证集，将跳过验证")
            print("   可能原因: 1) eval_split_ratio=0 2) 验证集创建失败 3) 验证集为空")
            logger.warning("[ERROR] 没有验证集，将跳过验证")
            logger.warning("   可能原因: 1) eval_split_ratio=0 2) 验证集创建失败 3) 验证集为空")
        
        # 为预计算嵌入模式准备数据整理器
        data_collator = None
        if use_precomputed:
            data_collator = PrecomputedEmbeddingsCollator()
            logger.info("使用预计算嵌入数据整理器")
        
        trainer = BinaryClassificationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,  # 使用processing_class而不是tokenizer
            data_collator=data_collator,  # 为预计算嵌入模式添加自定义整理器
            callbacks=callbacks,
        )
        
        # 如果指定了分类头学习率，则使用参数组分别设置学习率
        if hasattr(args, 'head_learning_rate') and args.head_learning_rate != args.learning_rate:
            logger.info("🔧 配置差异化学习率优化器...")
            
            # 创建参数组
            lora_params = []
            head_params = []
            other_params = []
            
            # 统计参数量
            lora_param_count = 0
            head_param_count = 0
            other_param_count = 0
            
            # 分离不同类型的参数
            for name, param in model.named_parameters():
                if param.requires_grad:  # 只考虑需要梯度的参数
                    param_size = param.numel()
                    if 'lora' in name.lower():
                        lora_params.append(param)
                        lora_param_count += param_size
                    elif 'prediction_head' in name:
                        head_params.append(param)
                        head_param_count += param_size
                    else:
                        other_params.append(param)
                        other_param_count += param_size
            
            # 输出参数统计
            total_trainable = lora_param_count + head_param_count + other_param_count
            logger.info(f"📊 可训练参数统计:")
            logger.info(f"   LoRA参数: {lora_param_count:,} ({lora_param_count/total_trainable*100:.2f}%)")
            logger.info(f"   预测头参数: {head_param_count:,} ({head_param_count/total_trainable*100:.2f}%)")
            logger.info(f"   其他参数: {other_param_count:,} ({other_param_count/total_trainable*100:.2f}%)")
            logger.info(f"   总计: {total_trainable:,}")
            logger.info(f"   学习率设置:")
            logger.info(f"     - LoRA/其他: {args.learning_rate}")
            logger.info(f"     - 预测头: {args.head_learning_rate}")
            
            # 合并LoRA和其他参数
            base_params = lora_params + other_params
            
            # 创建参数组
            param_groups = [
                {'params': base_params, 'lr': args.learning_rate},
                {'params': head_params, 'lr': args.head_learning_rate}
            ]
            
            # 使用8bit优化器以节省内存
            try:
                import bitsandbytes as bnb
                trainer.optimizer = bnb.optim.AdamW8bit(
                    param_groups,
                    weight_decay=training_args.weight_decay,
                    betas=(0.9, 0.999),
                    eps=1e-8
                )
                logger.info("✅ 使用 AdamW8bit 优化器")
            except ImportError:
                import torch.optim as optim
                trainer.optimizer = optim.AdamW(
                    param_groups,
                    weight_decay=training_args.weight_decay
                )
                logger.warning("⚠️ bitsandbytes未安装，使用标准 AdamW")
            
            # 重新创建学习率调度器
            from transformers import get_linear_schedule_with_warmup
            num_training_steps = len(train_dataset) // (args.batch_size * args.gradient_accumulation_steps) * args.num_epochs
            num_warmup_steps = int(num_training_steps * args.warmup_ratio)
            
            trainer.lr_scheduler = get_linear_schedule_with_warmup(
                trainer.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            logger.info(f"✅ 已创建学习率调度器 (warmup: {num_warmup_steps}/{num_training_steps})")
        else:
            # 统计可训练参数（即使不使用差异化学习率也输出统计信息）
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"📊 参数统计:")
            logger.info(f"   总参数: {total_params:,}")
            logger.info(f"   可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
            logger.info(f"   冻结参数: {total_params - trainable_params:,}")
        
        # 7. 开始训练
        logger.info("开始训练...")
        trainer.train()
        
        # 8. 保存模型
        logger.info("保存模型...")
        # 使用我们自定义的保存方法
        try:
            trainer.save_model()
            logger.info("训练器保存模型成功")
        except Exception as e:
            logger.warning(f"训练器保存失败: {e}")
            # 直接使用模型的save_pretrained方法
            try:
                if hasattr(model, 'save_pretrained'):
                    model.save_pretrained(args.output_dir)
                    logger.info("直接保存模型成功")
                else:
                    logger.warning("模型不支持save_pretrained方法")
            except Exception as e2:
                logger.error(f"直接保存也失败: {e2}")
        
        # 保存分词器
        try:
            tokenizer.save_pretrained(args.output_dir)
            logger.info("分词器保存成功")
        except Exception as e:
            logger.warning(f"分词器保存失败: {e}")
        
        # 9. 保存训练配置
        config = {
            "model_path": args.model_path,
            "max_seq_length": args.max_seq_length,
            "train_mode": args.train_mode,
            "lora_config": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout
            } if args.train_mode == "lora" else None,
            "prediction_head": args.prediction_head,
            "loss_function": "FocalLoss",
            "training_config": {
                "num_epochs": args.num_epochs,
                "learning_rate": args.learning_rate,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
            },
            "dataset_info": {
                "train_size": len(train_dataset) if train_dataset else 0,
                "eval_size": len(eval_dataset) if eval_dataset else 0,
            }
        }
        
        if args.output_dir:
            with open(os.path.join(args.output_dir, "training_config.json"), "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2, ensure_ascii=True)
        
        logger.info("训练完成!")
        logger.info(f"模型已保存到: {args.output_dir}")
        
        # 10. 最终评估
        if eval_dataset:
            logger.info("进行最终评估...")
            final_metrics = trainer.evaluate()
            logger.info(f"最终评估结果: {final_metrics}")
        
    except Exception as e:
        logger.error(f"训练过程中出现错误: {e}")
        raise
    finally:
        # 清理内存
        if 'model' in locals() and model is not None:
            del model
        if 'trainer' in locals() and trainer is not None:
            del trainer
        if 'tokenizer' in locals() and tokenizer is not None:
            del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
        # 关闭WandB
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    main()
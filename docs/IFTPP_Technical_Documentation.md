# IntensityFree Temporal Point Process (IFTPP) 完整复现技术文档

> **文档目的**: 为将 IFTPP 集成到时空点过程框架提供完整的技术参考  
> **生成日期**: 2026年1月25日  
> **基于**: EasyTPP 代码库深度分析

---

## 目录

1. [模型理论基础](#1-模型理论基础)
2. [EasyTPP 源码深度分析](#2-easytpp-源码深度分析)
3. [数据处理流程](#3-数据处理流程)
4. [模型实现完整代码](#4-模型实现完整代码)
5. [训练流程实现](#5-训练流程实现)
6. [预测与推理](#6-预测与推理)
7. [时空扩展：p(t,x) = p(t)p(x|t)](#7-时空扩展ptx--ptpxt)
8. [评估指标接口](#8-评估指标接口)
9. [完整使用示例](#9-完整使用示例)
10. [Agent 沟通建议](#10-agent-沟通建议)

---

## 1. 模型理论基础

### 1.1 IntensityFree 核心思想

传统时间点过程模型（如 Neural Hawkes Process）建模**条件强度函数** λ*(t)：

```
log L = Σᵢ log λ*(tᵢ) - ∫₀ᵀ λ*(s) ds
```

**问题**：积分项 ∫₀ᵀ λ*(s) ds 通常没有解析解，需要蒙特卡洛近似。

**IntensityFree 的解决方案**：直接建模事件时间间隔的分布 p(τ | H)，避免积分：

```
log L = Σᵢ [ log p(τᵢ | Hᵢ₋₁) + log p(kᵢ | Hᵢ₋₁) ]
```

其中：
- `τᵢ = tᵢ - tᵢ₋₁` 为第 i 个事件的时间间隔
- `kᵢ` 为第 i 个事件的类型
- `Hᵢ₋₁` 为历史信息的编码

### 1.2 混合对数正态分布

IFTPP 使用**混合对数正态分布 (Mixture of Log-Normal)** 建模时间间隔：

```
p(τ | h) = Σₖ πₖ · LogNormal(τ; μₖ, σₖ)
```

其中：
- `K` 为混合组件数（EasyTPP默认为3，可配置更大如64）
- `πₖ = softmax(wₖ)` 为混合权重
- `μₖ, σₖ` 为第 k 个组件的参数

**对数正态分布的概率密度**：

```
LogNormal(τ; μ, σ) = (1 / (τ σ √(2π))) · exp(-(log τ - μ)² / (2σ²))
```

**选择对数正态的原因**：
1. 时间间隔 τ > 0，对数正态天然满足正值约束
2. 混合多个对数正态可以拟合复杂的多峰分布
3. 相比于直接拟合正态分布更适合时间数据的偏斜特性

---

## 2. EasyTPP 源码深度分析

### 2.1 核心文件结构

| 文件 | 功能 |
|------|------|
| `easy_tpp/model/torch_model/torch_intensity_free.py` | IFTPP 模型实现 |
| `easy_tpp/model/torch_model/torch_basemodel.py` | 模型基类 |
| `easy_tpp/utils/torch_utils.py` | 混合分布实现 |
| `easy_tpp/preprocess/dataset.py` | 数据集类 |
| `easy_tpp/preprocess/data_collator.py` | 数据整理器 |
| `easy_tpp/runner/tpp_runner.py` | 训练运行器 |

### 2.2 混合对数正态分布实现 (关键!)

来自 `easy_tpp/utils/torch_utils.py`：

```python
class LogNormalMixtureDistribution(TransformedDistribution):
    """
    混合对数正态分布
    
    实现方式: 
    1. 构建正态混合分布 (MixtureSameFamily)
    2. 应用 AffineTransform（可选，用于归一化）
    3. 应用 ExpTransform（将正态变为对数正态）
    """
    
    def __init__(self, locs, log_scales, log_weights, 
                 mean_log_inter_time=0.0, std_log_inter_time=1.0):
        # 混合分布的组件选择分布 (Categorical)
        mixture_dist = D.Categorical(logits=log_weights)
        
        # 各组件的正态分布
        component_dist = Normal(loc=locs, scale=log_scales.exp())
        
        # 构建混合分布
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        
        # 变换: 先仿射 (归一化) 再指数 (正态->对数正态)
        transforms = []
        if not (mean_log_inter_time == 0.0 and std_log_inter_time == 1.0):
            transforms.append(D.AffineTransform(
                loc=mean_log_inter_time, 
                scale=std_log_inter_time
            ))
        transforms.append(D.ExpTransform())
        
        super().__init__(GMM, transforms)
```

### 2.3 IFTPP 模型核心代码

来自 `easy_tpp/model/torch_model/torch_intensity_free.py`：

```python
class IntensityFree(TorchBaseModel):
    def __init__(self, model_config):
        super().__init__(model_config)
        
        # 关键配置
        self.num_mix_components = model_config.model_specs['num_mix_components']
        self.mean_log_inter_time = model_config.get("mean_log_inter_time", 0.0)
        self.std_log_inter_time = model_config.get("std_log_inter_time", 1.0)
        
        # 输入特征维度: log(dt) + type_embedding
        self.num_features = 1 + self.hidden_size
        
        # GRU 编码器 (单层)
        self.layer_rnn = nn.GRU(
            input_size=self.num_features,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        # 事件类型预测
        self.mark_linear = nn.Linear(self.hidden_size, self.num_event_types_pad)
        
        # 混合分布参数预测 (3K个参数: K个均值 + K个对数标准差 + K个权重)
        self.linear = nn.Linear(self.hidden_size, 3 * self.num_mix_components)
    
    def forward(self, time_delta_seqs, type_seqs):
        """
        编码历史序列
        
        Args:
            time_delta_seqs: [batch, seq_len] 时间间隔
            type_seqs: [batch, seq_len] 事件类型
        
        Returns:
            context: [batch, seq_len, hidden_size] GRU隐状态
        """
        # 时间编码: 取对数
        temporal_seqs = torch.log(time_delta_seqs + self.eps).unsqueeze(-1)
        
        # 类型嵌入
        type_emb = self.layer_type_emb(type_seqs)
        
        # 拼接输入
        rnn_input = torch.cat([temporal_seqs, type_emb], dim=-1)
        
        # GRU 编码
        context, _ = self.layer_rnn(rnn_input)
        
        return context
    
    def get_inter_time_dist(self, context):
        """
        根据隐状态获取时间间隔分布
        
        Args:
            context: [batch, seq_len, hidden_size]
        
        Returns:
            LogNormalMixtureDistribution
        """
        raw_params = self.linear(context)  # [batch, seq, 3*K]
        
        K = self.num_mix_components
        locs = raw_params[..., :K]                    # 均值
        log_scales = raw_params[..., K:2*K]           # 对数标准差
        log_weights = raw_params[..., 2*K:3*K]        # 对数权重
        
        # 约束标准差范围，避免数值问题
        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        
        return LogNormalMixtureDistribution(
            locs, log_scales, log_weights,
            self.mean_log_inter_time, 
            self.std_log_inter_time
        )
    
    def loglike_loss(self, batch):
        """
        计算负对数似然损失
        """
        time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch
        
        # 编码历史 (不包含最后一个事件)
        context = self.forward(time_delta_seqs[:, :-1], type_seqs[:, :-1])
        
        # 获取时间间隔分布
        inter_time_dist = self.get_inter_time_dist(context)
        
        # 目标: 预测第 2 到最后一个事件的时间间隔
        inter_times = time_delta_seqs[:, 1:].clamp(min=1e-5)
        event_mask = batch_non_pad_mask[:, 1:]
        
        # 时间对数似然
        time_ll = inter_time_dist.log_prob(inter_times) * event_mask
        
        # 事件类型对数似然
        mark_logits = torch.log_softmax(self.mark_linear(context), dim=-1)
        mark_dist = Categorical(logits=mark_logits)
        mark_ll = mark_dist.log_prob(type_seqs[:, 1:]) * event_mask
        
        # 总对数似然
        log_p = time_ll + mark_ll
        loss = -log_p.sum()
        num_events = event_mask.sum()
        
        return loss, num_events
```

---

## 3. 数据处理流程

### 3.1 数据格式规范

**原始数据结构**（JSON/Pickle格式）：

```python
{
    "train": [  # 或 "dev", "test"
        {
            "time_seqs": [0.0, 0.5, 1.2, 2.3, ...],    # 绝对时间戳
            "type_seqs": [2, 0, 1, 3, ...],            # 事件类型 (0-indexed)
            # 可选字段
            "time_delta_seqs": [0.0, 0.5, 0.7, 1.1, ...],  # 时间间隔
            # 如果是时空数据
            "spatial_seqs": [[x1,y1], [x2,y2], ...]   # 空间坐标
        },
        ...
    ]
}
```

**关键点**：
- `time_seqs`: 必须是递增的绝对时间
- `type_seqs`: 从 0 开始索引
- `time_delta_seqs`: 如果不提供，会自动计算
- 第一个事件的 `time_delta` 通常设为 0 或一个小正数

### 3.2 数据归一化策略

**IFTPP 的时间归一化**（可选但推荐）：

```python
def compute_time_statistics(dataset):
    """
    计算对数时间间隔的统计量，用于归一化
    """
    all_log_dtimes = []
    for seq in dataset:
        dtimes = seq['time_delta_seqs'][1:]  # 跳过第一个（通常为0）
        log_dtimes = np.log(np.array(dtimes) + 1e-8)
        all_log_dtimes.extend(log_dtimes.tolist())
    
    mean_log_inter_time = np.mean(all_log_dtimes)
    std_log_inter_time = np.std(all_log_dtimes)
    
    return mean_log_inter_time, std_log_inter_time

# 在模型配置中使用:
# model_config.mean_log_inter_time = mean_log_inter_time
# model_config.std_log_inter_time = std_log_inter_time
```

**实际影响**：
- 归一化使得模型预测的 μₖ, σₖ 在标准尺度附近
- EasyTPP **默认不使用归一化** (`mean=0, std=1`)
- 如果时间间隔变化范围很大，建议启用归一化

### 3.3 完整数据处理代码

```python
"""
data_processing.py - IFTPP 数据处理模块
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from typing import List, Dict, Optional, Union
import json
import pickle

class TPPDataset(Dataset):
    """
    时间点过程数据集
    
    支持两种构造方式:
    1. 从文件加载
    2. 从内存数据构造
    """
    
    def __init__(
        self,
        data: Optional[List[Dict]] = None,
        file_path: Optional[str] = None,
        split: str = 'train',
        max_len: int = 100,
        compute_time_delta: bool = True
    ):
        if data is not None:
            self.data = data
        elif file_path is not None:
            self.data = self._load_file(file_path, split)
        else:
            raise ValueError("必须提供 data 或 file_path")
        
        self.max_len = max_len
        
        # 如果没有 time_delta_seqs，自动计算
        if compute_time_delta:
            self._compute_time_deltas()
    
    def _load_file(self, file_path: str, split: str) -> List[Dict]:
        """加载数据文件"""
        if file_path.endswith('.json'):
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif file_path.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        return data.get(split, data)
    
    def _compute_time_deltas(self):
        """计算时间间隔"""
        for seq in self.data:
            if 'time_delta_seqs' not in seq or seq['time_delta_seqs'] is None:
                times = seq['time_seqs']
                # 第一个间隔设为一个小正数
                dtimes = [times[0] if times[0] > 0 else 0.1]
                dtimes.extend([times[i] - times[i-1] for i in range(1, len(times))])
                seq['time_delta_seqs'] = dtimes
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        seq = self.data[idx]
        
        time_seqs = np.array(seq['time_seqs'], dtype=np.float32)
        time_delta_seqs = np.array(seq['time_delta_seqs'], dtype=np.float32)
        type_seqs = np.array(seq['type_seqs'], dtype=np.int64)
        
        # 如果有空间数据
        spatial_seqs = None
        if 'spatial_seqs' in seq:
            spatial_seqs = np.array(seq['spatial_seqs'], dtype=np.float32)
        
        # 截断
        seq_len = len(time_seqs)
        if seq_len > self.max_len:
            time_seqs = time_seqs[:self.max_len]
            time_delta_seqs = time_delta_seqs[:self.max_len]
            type_seqs = type_seqs[:self.max_len]
            if spatial_seqs is not None:
                spatial_seqs = spatial_seqs[:self.max_len]
            seq_len = self.max_len
        
        result = {
            'time_seqs': time_seqs,
            'time_delta_seqs': time_delta_seqs,
            'type_seqs': type_seqs,
            'seq_len': seq_len
        }
        
        if spatial_seqs is not None:
            result['spatial_seqs'] = spatial_seqs
        
        return result


@dataclass
class TPPDataCollator:
    """
    数据整理器：将变长序列 padding 成固定长度的 batch
    
    Args:
        pad_token_id: padding 用的 token ID（类型序列）
        max_len: 最大序列长度
        padding_side: 填充方向 ('right' 或 'left')
    """
    pad_token_id: int = 0
    max_len: int = 100
    padding_side: str = 'right'
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(batch)
        
        # 找到 batch 中的最大长度
        max_seq_len = min(
            max(item['seq_len'] for item in batch),
            self.max_len
        )
        
        # 初始化张量
        time_seqs = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        time_delta_seqs = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        type_seqs = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.int64)
        seq_non_pad_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
        
        # 检查是否有空间数据
        has_spatial = 'spatial_seqs' in batch[0]
        if has_spatial:
            spatial_dim = batch[0]['spatial_seqs'].shape[-1]
            spatial_seqs = torch.zeros(batch_size, max_seq_len, spatial_dim, dtype=torch.float32)
        
        for i, item in enumerate(batch):
            seq_len = min(item['seq_len'], max_seq_len)
            
            if self.padding_side == 'right':
                time_seqs[i, :seq_len] = torch.from_numpy(item['time_seqs'][:seq_len])
                time_delta_seqs[i, :seq_len] = torch.from_numpy(item['time_delta_seqs'][:seq_len])
                type_seqs[i, :seq_len] = torch.from_numpy(item['type_seqs'][:seq_len])
                seq_non_pad_mask[i, :seq_len] = True
                if has_spatial:
                    spatial_seqs[i, :seq_len] = torch.from_numpy(item['spatial_seqs'][:seq_len])
            else:  # left padding
                offset = max_seq_len - seq_len
                time_seqs[i, offset:] = torch.from_numpy(item['time_seqs'][:seq_len])
                time_delta_seqs[i, offset:] = torch.from_numpy(item['time_delta_seqs'][:seq_len])
                type_seqs[i, offset:] = torch.from_numpy(item['type_seqs'][:seq_len])
                seq_non_pad_mask[i, offset:] = True
                if has_spatial:
                    spatial_seqs[i, offset:] = torch.from_numpy(item['spatial_seqs'][:seq_len])
        
        # 构建 attention mask (用于 Transformer 类模型)
        attention_mask = seq_non_pad_mask.unsqueeze(1).expand(-1, max_seq_len, -1)
        # Causal mask
        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        attention_mask = attention_mask & causal_mask.unsqueeze(0)
        
        result = {
            'time_seqs': time_seqs,
            'time_delta_seqs': time_delta_seqs,
            'type_seqs': type_seqs,
            'seq_non_pad_mask': seq_non_pad_mask,
            'attention_mask': attention_mask
        }
        
        if has_spatial:
            result['spatial_seqs'] = spatial_seqs
        
        return result


def compute_data_statistics(dataset: TPPDataset) -> Dict[str, float]:
    """
    计算数据集统计量，用于归一化
    
    Returns:
        dict with:
            - mean_log_inter_time: 对数时间间隔均值
            - std_log_inter_time: 对数时间间隔标准差
            - mean_inter_time: 时间间隔均值
            - std_inter_time: 时间间隔标准差
    """
    all_dtimes = []
    all_log_dtimes = []
    
    for i in range(len(dataset)):
        item = dataset[i]
        dtimes = item['time_delta_seqs'][1:]  # 跳过第一个
        dtimes = dtimes[dtimes > 1e-8]  # 过滤零值
        
        if len(dtimes) > 0:
            all_dtimes.extend(dtimes.tolist())
            all_log_dtimes.extend(np.log(dtimes + 1e-8).tolist())
    
    return {
        'mean_log_inter_time': float(np.mean(all_log_dtimes)),
        'std_log_inter_time': float(np.std(all_log_dtimes)),
        'mean_inter_time': float(np.mean(all_dtimes)),
        'std_inter_time': float(np.std(all_dtimes))
    }


def create_data_loaders(
    train_data: List[Dict],
    valid_data: List[Dict],
    test_data: List[Dict],
    batch_size: int = 32,
    max_len: int = 100,
    num_event_types: int = 10,
    num_workers: int = 0
) -> tuple:
    """
    创建训练、验证、测试数据加载器
    """
    pad_token_id = num_event_types  # PAD 使用类型数作为 ID
    
    train_dataset = TPPDataset(data=train_data, max_len=max_len)
    valid_dataset = TPPDataset(data=valid_data, max_len=max_len)
    test_dataset = TPPDataset(data=test_data, max_len=max_len)
    
    collator = TPPDataCollator(pad_token_id=pad_token_id, max_len=max_len)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=num_workers
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers
    )
    
    # 计算统计量
    stats = compute_data_statistics(train_dataset)
    
    return train_loader, valid_loader, test_loader, stats
```

---

## 4. 模型实现完整代码

### 4.1 工具函数

```python
"""
utils.py - 工具函数
"""
import torch
import torch.nn as nn
from torch.distributions import Normal, Categorical, MixtureSameFamily
from torch.distributions import TransformedDistribution
import torch.distributions as D


def clamp_preserve_gradients(x: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
    """
    Clamp 张量但保留梯度
    
    技巧: x + (x.clamp - x).detach()
    """
    return x + (x.clamp(min=min_val, max=max_val) - x).detach()


class LogNormalMixtureDistribution(TransformedDistribution):
    """
    混合对数正态分布
    
    数学形式:
        p(τ) = Σₖ πₖ · LogNormal(τ; μₖ, σₖ)
    
    实现方式:
        1. 构建正态混合分布 MixtureSameFamily(Categorical, Normal)
        2. 应用 AffineTransform (可选，用于归一化反变换)
        3. 应用 ExpTransform (正态 -> 对数正态)
    
    Args:
        locs: [*, K] 各组件的均值
        log_scales: [*, K] 各组件的对数标准差
        log_weights: [*, K] 各组件的对数权重 (未归一化)
        mean_log_inter_time: 对数时间间隔的均值 (归一化参数)
        std_log_inter_time: 对数时间间隔的标准差 (归一化参数)
    """
    
    def __init__(
        self,
        locs: torch.Tensor,
        log_scales: torch.Tensor,
        log_weights: torch.Tensor,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0
    ):
        # 混合权重分布
        mixture_dist = Categorical(logits=log_weights)
        
        # 各组件的正态分布
        component_dist = Normal(loc=locs, scale=log_scales.exp())
        
        # 混合分布 (在正态空间)
        GMM = MixtureSameFamily(mixture_dist, component_dist)
        
        # 变换链
        transforms = []
        
        # 1. 仿射变换 (反归一化): z -> z * std + mean
        if not (mean_log_inter_time == 0.0 and std_log_inter_time == 1.0):
            transforms.append(D.AffineTransform(
                loc=mean_log_inter_time,
                scale=std_log_inter_time
            ))
        
        # 2. 指数变换: log(τ) -> τ
        transforms.append(D.ExpTransform())
        
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        self.locs = locs
        self.log_scales = log_scales
        self.log_weights = log_weights
        
        super().__init__(GMM, transforms)
    
    @property
    def mean(self) -> torch.Tensor:
        """
        计算混合对数正态分布的均值
        
        对于 LogNormal(μ, σ): E[X] = exp(μ + σ²/2)
        对于混合分布: E[X] = Σₖ πₖ · Eₖ[X]
        """
        # 归一化后的参数
        locs_normalized = self.locs * self.std_log_inter_time + self.mean_log_inter_time
        scales_normalized = self.log_scales.exp() * self.std_log_inter_time
        
        # 各组件的均值
        component_means = torch.exp(locs_normalized + 0.5 * scales_normalized.pow(2))
        
        # 混合权重
        weights = torch.softmax(self.log_weights, dim=-1)
        
        # 加权均值
        return (weights * component_means).sum(dim=-1)
```

### 4.2 IFTPP 模型完整实现

```python
"""
intensity_free_model.py - IntensityFree TPP 模型完整实现
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, Tuple, Optional
import math

from utils import LogNormalMixtureDistribution, clamp_preserve_gradients


class IntensityFreeTPP(nn.Module):
    """
    IntensityFree Temporal Point Process
    
    论文: "Intensity-Free Learning of Temporal Point Processes" (ICLR 2020)
    
    核心思想:
        直接建模事件时间间隔的分布 p(τ|h)，而不是条件强度函数 λ*(t)
        使用混合对数正态分布参数化时间间隔分布
    
    Args:
        num_event_types: 事件类型数量 (不含 PAD)
        hidden_size: 隐藏层维度
        num_mix_components: 混合分布组件数
        mean_log_inter_time: 对数时间间隔均值 (归一化参数)
        std_log_inter_time: 对数时间间隔标准差 (归一化参数)
        dropout: Dropout 概率
    """
    
    def __init__(
        self,
        num_event_types: int,
        hidden_size: int = 64,
        num_mix_components: int = 64,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_event_types = num_event_types
        self.num_event_types_pad = num_event_types + 1  # 包含 PAD token
        self.pad_token_id = num_event_types  # PAD 的 ID
        
        self.hidden_size = hidden_size
        self.num_mix_components = num_mix_components
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        
        self.eps = torch.finfo(torch.float32).eps
        
        # ============ 模型组件 ============
        
        # 1. 事件类型嵌入
        self.layer_type_emb = nn.Embedding(
            num_embeddings=self.num_event_types_pad,
            embedding_dim=hidden_size,
            padding_idx=self.pad_token_id
        )
        
        # 2. 输入特征维度: log(dt) + type_embedding
        self.num_features = 1 + hidden_size
        
        # 3. GRU 编码器
        self.layer_rnn = nn.GRU(
            input_size=self.num_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0  # 单层不需要 dropout
        )
        
        # 4. Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 5. 事件类型预测层
        self.mark_linear = nn.Linear(hidden_size, self.num_event_types_pad)
        
        # 6. 混合分布参数预测层
        # 输出: K个均值 + K个对数标准差 + K个对数权重
        self.linear = nn.Linear(hidden_size, 3 * num_mix_components)
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor
    ) -> torch.Tensor:
        """
        编码历史序列
        
        Args:
            time_delta_seqs: [batch_size, seq_len] 时间间隔序列
            type_seqs: [batch_size, seq_len] 事件类型序列
        
        Returns:
            context: [batch_size, seq_len, hidden_size] GRU 隐状态
        """
        # 1. 时间编码: 取对数
        temporal_seqs = torch.log(time_delta_seqs + self.eps).unsqueeze(-1)  # [B, L, 1]
        
        # 2. 类型嵌入
        type_emb = self.layer_type_emb(type_seqs)  # [B, L, H]
        
        # 3. 拼接输入
        rnn_input = torch.cat([temporal_seqs, type_emb], dim=-1)  # [B, L, H+1]
        
        # 4. GRU 编码
        context, _ = self.layer_rnn(rnn_input)  # [B, L, H]
        
        # 5. Dropout
        context = self.dropout(context)
        
        return context
    
    def get_inter_time_dist(
        self,
        context: torch.Tensor
    ) -> LogNormalMixtureDistribution:
        """
        根据隐状态获取时间间隔的混合对数正态分布
        
        Args:
            context: [batch_size, seq_len, hidden_size]
        
        Returns:
            LogNormalMixtureDistribution
        """
        # 预测混合分布参数
        raw_params = self.linear(context)  # [B, L, 3*K]
        
        K = self.num_mix_components
        locs = raw_params[..., :K]           # [B, L, K] 均值
        log_scales = raw_params[..., K:2*K]  # [B, L, K] 对数标准差
        log_weights = raw_params[..., 2*K:]  # [B, L, K] 对数权重
        
        # 约束标准差范围，避免数值问题
        # exp(-5) ≈ 0.007, exp(3) ≈ 20
        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        
        return LogNormalMixtureDistribution(
            locs=locs,
            log_scales=log_scales,
            log_weights=log_weights,
            mean_log_inter_time=self.mean_log_inter_time,
            std_log_inter_time=self.std_log_inter_time
        )
    
    def get_mark_logits(self, context: torch.Tensor) -> torch.Tensor:
        """
        获取事件类型的 logits
        
        Args:
            context: [batch_size, seq_len, hidden_size]
        
        Returns:
            logits: [batch_size, seq_len, num_event_types_pad]
        """
        return self.mark_linear(context)
    
    def compute_loglikelihood(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        计算对数似然
        
        Args:
            batch: 包含以下键的字典
                - time_seqs: [B, L] 绝对时间
                - time_delta_seqs: [B, L] 时间间隔
                - type_seqs: [B, L] 事件类型
                - seq_non_pad_mask: [B, L] 非 padding 掩码
        
        Returns:
            dict containing:
                - time_ll: 时间对数似然
                - mark_ll: 类型对数似然
                - total_ll: 总对数似然
                - num_events: 事件数量
        """
        time_delta_seqs = batch['time_delta_seqs']
        type_seqs = batch['type_seqs']
        seq_non_pad_mask = batch['seq_non_pad_mask']
        
        # 编码历史 (用前 L-1 个事件预测后 L-1 个事件)
        context = self.forward(time_delta_seqs[:, :-1], type_seqs[:, :-1])  # [B, L-1, H]
        
        # 获取时间间隔分布
        inter_time_dist = self.get_inter_time_dist(context)
        
        # 目标时间间隔 (第 2 到第 L 个事件的间隔)
        target_dtimes = time_delta_seqs[:, 1:].clamp(min=1e-5)  # [B, L-1]
        event_mask = seq_non_pad_mask[:, 1:].float()  # [B, L-1]
        
        # 时间对数似然
        time_ll = inter_time_dist.log_prob(target_dtimes)  # [B, L-1]
        time_ll = time_ll * event_mask
        
        # 事件类型对数似然
        mark_logits = self.get_mark_logits(context)  # [B, L-1, num_types]
        mark_log_probs = F.log_softmax(mark_logits, dim=-1)
        
        target_types = type_seqs[:, 1:]  # [B, L-1]
        mark_ll = mark_log_probs.gather(-1, target_types.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        mark_ll = mark_ll * event_mask
        
        # 汇总
        total_ll = time_ll + mark_ll
        num_events = event_mask.sum()
        
        return {
            'time_ll': time_ll.sum(),
            'mark_ll': mark_ll.sum(),
            'total_ll': total_ll.sum(),
            'num_events': num_events
        }
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算训练损失 (负对数似然)
        
        Returns:
            loss: 标量损失
            metrics: 包含各项指标的字典
        """
        ll_dict = self.compute_loglikelihood(batch)
        
        # 负对数似然
        loss = -ll_dict['total_ll'] / (ll_dict['num_events'] + self.eps)
        
        metrics = {
            'loss': loss.item(),
            'time_nll': (-ll_dict['time_ll'] / (ll_dict['num_events'] + self.eps)).item(),
            'mark_nll': (-ll_dict['mark_ll'] / (ll_dict['num_events'] + self.eps)).item(),
            'num_events': ll_dict['num_events'].item()
        }
        
        return loss, metrics
    
    @torch.no_grad()
    def predict_next_event(
        self,
        batch: Dict[str, torch.Tensor],
        num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        预测下一个事件的时间间隔和类型
        
        方法: 
            1. 编码历史序列
            2. 从时间间隔分布采样，取均值作为预测
            3. 事件类型取最大概率
        
        Args:
            batch: 输入数据
            num_samples: 采样次数
        
        Returns:
            dict containing:
                - dtime_pred: [B, L] 预测的时间间隔
                - type_pred: [B, L] 预测的事件类型
                - type_probs: [B, L, num_types] 类型概率分布
                - dtime_samples: [num_samples, B, L] 时间间隔采样
        """
        time_delta_seqs = batch['time_delta_seqs']
        type_seqs = batch['type_seqs']
        
        # 编码完整序列
        context = self.forward(time_delta_seqs, type_seqs)  # [B, L, H]
        
        # 获取时间间隔分布
        inter_time_dist = self.get_inter_time_dist(context)
        
        # 从分布采样
        dtime_samples = inter_time_dist.sample((num_samples,))  # [S, B, L]
        
        # 取采样均值作为预测
        dtime_pred = dtime_samples.mean(dim=0)  # [B, L]
        
        # 或者直接使用分布均值 (更快但可能不太准确)
        # dtime_pred = inter_time_dist.mean  # [B, L]
        
        # 事件类型预测
        mark_logits = self.get_mark_logits(context)  # [B, L, num_types]
        type_probs = F.softmax(mark_logits, dim=-1)
        type_pred = mark_logits.argmax(dim=-1)  # [B, L]
        
        return {
            'dtime_pred': dtime_pred,
            'type_pred': type_pred,
            'type_probs': type_probs,
            'dtime_samples': dtime_samples
        }
    
    @torch.no_grad()
    def predict_multi_step(
        self,
        batch: Dict[str, torch.Tensor],
        num_steps: int = 5,
        num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        多步预测: 自回归地预测多个未来事件
        
        Args:
            batch: 初始历史数据
            num_steps: 预测步数
            num_samples: 每步采样次数
        
        Returns:
            dict containing:
                - dtime_preds: [B, num_steps] 预测的时间间隔
                - type_preds: [B, num_steps] 预测的事件类型
                - time_preds: [B, num_steps] 预测的绝对时间
        """
        device = batch['time_seqs'].device
        batch_size = batch['time_seqs'].shape[0]
        
        # 复制 batch 用于自回归预测
        current_time = batch['time_seqs'].clone()
        current_dtime = batch['time_delta_seqs'].clone()
        current_type = batch['type_seqs'].clone()
        current_mask = batch['seq_non_pad_mask'].clone()
        
        # 找到每个序列的最后有效位置
        seq_lens = current_mask.sum(dim=1).long()
        
        dtime_preds = []
        type_preds = []
        time_preds = []
        
        for step in range(num_steps):
            # 编码当前历史
            context = self.forward(current_dtime, current_type)  # [B, L, H]
            
            # 获取每个序列最后位置的隐状态
            last_context = torch.stack([
                context[i, seq_lens[i] - 1] for i in range(batch_size)
            ])  # [B, H]
            
            # 预测时间间隔
            raw_params = self.linear(last_context)  # [B, 3*K]
            K = self.num_mix_components
            locs = raw_params[:, :K]
            log_scales = raw_params[:, K:2*K]
            log_weights = raw_params[:, 2*K:]
            log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
            
            inter_time_dist = LogNormalMixtureDistribution(
                locs, log_scales, log_weights,
                self.mean_log_inter_time, self.std_log_inter_time
            )
            
            # 采样并取均值
            dtime_samples = inter_time_dist.sample((num_samples,))  # [S, B]
            dtime_pred = dtime_samples.mean(dim=0)  # [B]
            
            # 预测事件类型
            mark_logits = self.mark_linear(last_context)  # [B, num_types]
            type_pred = mark_logits.argmax(dim=-1)  # [B]
            
            # 计算绝对时间
            last_time = torch.stack([
                current_time[i, seq_lens[i] - 1] for i in range(batch_size)
            ])  # [B]
            time_pred = last_time + dtime_pred
            
            dtime_preds.append(dtime_pred)
            type_preds.append(type_pred)
            time_preds.append(time_pred)
            
            # 更新序列 (扩展或替换最后位置)
            # 这里简化处理：直接扩展序列
            # 实际应用中需要处理序列长度限制
            new_dtime = dtime_pred.unsqueeze(1)  # [B, 1]
            new_type = type_pred.unsqueeze(1)     # [B, 1]
            new_time = time_pred.unsqueeze(1)     # [B, 1]
            new_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            
            current_time = torch.cat([current_time, new_time], dim=1)
            current_dtime = torch.cat([current_dtime, new_dtime], dim=1)
            current_type = torch.cat([current_type, new_type], dim=1)
            current_mask = torch.cat([current_mask, new_mask], dim=1)
            seq_lens = seq_lens + 1
        
        return {
            'dtime_preds': torch.stack(dtime_preds, dim=1),  # [B, num_steps]
            'type_preds': torch.stack(type_preds, dim=1),    # [B, num_steps]
            'time_preds': torch.stack(time_preds, dim=1)     # [B, num_steps]
        }
```

---

## 5. 训练流程实现

```python
"""
trainer.py - IFTPP 训练器
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Callable
import os


class IFTPPTrainer:
    """
    IntensityFree TPP 训练器
    
    Args:
        model: IFTPP 模型
        optimizer: 优化器 (默认 Adam)
        scheduler: 学习率调度器 (可选)
        device: 训练设备
        grad_clip: 梯度裁剪阈值
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = 'cuda',
        grad_clip: float = 5.0
    ):
        self.model = model.to(device)
        self.device = device
        self.grad_clip = grad_clip
        
        # 默认使用 Adam 优化器
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """训练一个 epoch"""
        self.model.train()
        
        total_loss = 0
        total_time_nll = 0
        total_mark_nll = 0
        total_events = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            # 移动数据到设备
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # 前向传播
            loss, metrics = self.model.compute_loss(batch)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.grad_clip
                )
            
            self.optimizer.step()
            
            # 记录指标
            total_loss += metrics['loss'] * metrics['num_events']
            total_time_nll += metrics['time_nll'] * metrics['num_events']
            total_mark_nll += metrics['mark_nll'] * metrics['num_events']
            total_events += metrics['num_events']
            
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'time_nll': f"{metrics['time_nll']:.4f}",
                'mark_nll': f"{metrics['mark_nll']:.4f}"
            })
        
        return {
            'loss': total_loss / total_events,
            'time_nll': total_time_nll / total_events,
            'mark_nll': total_mark_nll / total_events,
            'num_events': total_events
        }
    
    @torch.no_grad()
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        
        total_loss = 0
        total_time_nll = 0
        total_mark_nll = 0
        total_events = 0
        
        # 用于计算预测指标
        all_dtime_preds = []
        all_dtime_labels = []
        all_type_preds = []
        all_type_labels = []
        all_masks = []
        
        for batch in tqdm(data_loader, desc='Evaluating'):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # 计算损失
            loss, metrics = self.model.compute_loss(batch)
            
            total_loss += metrics['loss'] * metrics['num_events']
            total_time_nll += metrics['time_nll'] * metrics['num_events']
            total_mark_nll += metrics['mark_nll'] * metrics['num_events']
            total_events += metrics['num_events']
            
            # 预测
            predictions = self.model.predict_next_event(batch)
            
            # 收集预测结果
            # 注意: 预测的是下一个事件，所以标签需要 shift
            dtime_pred = predictions['dtime_pred'][:, :-1]  # [B, L-1]
            dtime_label = batch['time_delta_seqs'][:, 1:]   # [B, L-1]
            type_pred = predictions['type_pred'][:, :-1]    # [B, L-1]
            type_label = batch['type_seqs'][:, 1:]          # [B, L-1]
            mask = batch['seq_non_pad_mask'][:, 1:]         # [B, L-1]
            
            all_dtime_preds.append(dtime_pred.cpu())
            all_dtime_labels.append(dtime_label.cpu())
            all_type_preds.append(type_pred.cpu())
            all_type_labels.append(type_label.cpu())
            all_masks.append(mask.cpu())
        
        # 计算预测指标
        dtime_preds = torch.cat(all_dtime_preds, dim=0)
        dtime_labels = torch.cat(all_dtime_labels, dim=0)
        type_preds = torch.cat(all_type_preds, dim=0)
        type_labels = torch.cat(all_type_labels, dim=0)
        masks = torch.cat(all_masks, dim=0)
        
        # RMSE
        dtime_error = (dtime_preds - dtime_labels).pow(2) * masks
        rmse = (dtime_error.sum() / masks.sum()).sqrt().item()
        
        # MAE
        mae = ((dtime_preds - dtime_labels).abs() * masks).sum() / masks.sum()
        mae = mae.item()
        
        # Accuracy
        type_correct = ((type_preds == type_labels) * masks).sum()
        accuracy = (type_correct / masks.sum()).item()
        
        return {
            'loss': total_loss / total_events,
            'time_nll': total_time_nll / total_events,
            'mark_nll': total_mark_nll / total_events,
            'rmse': rmse,
            'mae': mae,
            'accuracy': accuracy,
            'num_events': total_events
        }
    
    def train(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs: int = 100,
        patience: int = 10,
        save_path: str = 'best_model.pt',
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        完整训练流程
        
        Args:
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            num_epochs: 最大训练 epoch 数
            patience: 早停耐心值
            save_path: 模型保存路径
            verbose: 是否打印详细信息
        
        Returns:
            训练历史记录
        """
        history = {
            'train_loss': [],
            'valid_loss': [],
            'valid_rmse': [],
            'valid_accuracy': []
        }
        
        best_valid_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            if verbose:
                print(f"\n{'='*60}")
                print(f"Epoch {epoch + 1}/{num_epochs}")
                print('='*60)
            
            # 训练
            train_metrics = self.train_epoch(train_loader)
            
            # 验证
            valid_metrics = self.evaluate(valid_loader)
            
            # 学习率调度
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(valid_metrics['loss'])
                else:
                    self.scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_metrics['loss'])
            history['valid_loss'].append(valid_metrics['loss'])
            history['valid_rmse'].append(valid_metrics['rmse'])
            history['valid_accuracy'].append(valid_metrics['accuracy'])
            
            if verbose:
                print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
                      f"Time NLL: {train_metrics['time_nll']:.4f}, "
                      f"Mark NLL: {train_metrics['mark_nll']:.4f}")
                print(f"Valid - Loss: {valid_metrics['loss']:.4f}, "
                      f"RMSE: {valid_metrics['rmse']:.4f}, "
                      f"MAE: {valid_metrics['mae']:.4f}, "
                      f"Acc: {valid_metrics['accuracy']:.4f}")
            
            # 早停检查
            if valid_metrics['loss'] < best_valid_loss:
                best_valid_loss = valid_metrics['loss']
                patience_counter = 0
                
                # 保存最佳模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'valid_loss': valid_metrics['loss'],
                    'valid_rmse': valid_metrics['rmse'],
                    'valid_accuracy': valid_metrics['accuracy']
                }, save_path)
                
                if verbose:
                    print(f"✓ Best model saved (valid_loss: {best_valid_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
        
        # 加载最佳模型
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        return history
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
```

---

## 6. 预测与推理

```python
"""
predictor.py - IFTPP 预测器
"""
import torch
import numpy as np
from typing import Dict, List, Optional


class IFTPPPredictor:
    """
    IFTPP 预测器
    
    用于推理阶段的预测，包括：
    1. 单步预测 (给定历史，预测下一个事件)
    2. 多步预测 (自回归地预测多个未来事件)
    3. 不确定性估计 (返回预测分布)
    """
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    @torch.no_grad()
    def predict_next(
        self,
        history_times: np.ndarray,
        history_types: np.ndarray,
        num_samples: int = 100,
        return_distribution: bool = False
    ) -> Dict:
        """
        给定历史序列，预测下一个事件
        
        Args:
            history_times: [seq_len] 历史事件的绝对时间
            history_types: [seq_len] 历史事件的类型
            num_samples: 从时间分布采样的次数
            return_distribution: 是否返回完整分布信息
        
        Returns:
            dict with:
                - next_dtime: 预测的时间间隔
                - next_time: 预测的绝对时间
                - next_type: 预测的事件类型
                - type_probs: 事件类型概率分布
                - dtime_samples: (可选) 时间间隔采样
        """
        # 计算时间间隔
        dtimes = np.zeros_like(history_times)
        dtimes[0] = history_times[0] if history_times[0] > 0 else 0.1
        dtimes[1:] = np.diff(history_times)
        
        # 转为 tensor 并添加 batch 维度
        time_seqs = torch.FloatTensor(history_times).unsqueeze(0).to(self.device)
        dtime_seqs = torch.FloatTensor(dtimes).unsqueeze(0).to(self.device)
        type_seqs = torch.LongTensor(history_types).unsqueeze(0).to(self.device)
        mask = torch.ones(1, len(history_times), dtype=torch.bool).to(self.device)
        
        batch = {
            'time_seqs': time_seqs,
            'time_delta_seqs': dtime_seqs,
            'type_seqs': type_seqs,
            'seq_non_pad_mask': mask
        }
        
        # 编码历史
        context = self.model.forward(dtime_seqs, type_seqs)  # [1, L, H]
        
        # 取最后一个位置的隐状态
        last_context = context[:, -1, :]  # [1, H]
        
        # 预测时间间隔分布
        K = self.model.num_mix_components
        raw_params = self.model.linear(last_context)  # [1, 3*K]
        
        from utils import LogNormalMixtureDistribution, clamp_preserve_gradients
        
        locs = raw_params[:, :K]
        log_scales = raw_params[:, K:2*K]
        log_weights = raw_params[:, 2*K:]
        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        
        inter_time_dist = LogNormalMixtureDistribution(
            locs, log_scales, log_weights,
            self.model.mean_log_inter_time,
            self.model.std_log_inter_time
        )
        
        # 采样
        dtime_samples = inter_time_dist.sample((num_samples,))  # [S, 1]
        dtime_samples = dtime_samples.squeeze(-1)  # [S]
        
        # 预测值 (采样均值)
        next_dtime = dtime_samples.mean().item()
        
        # 预测绝对时间
        next_time = history_times[-1] + next_dtime
        
        # 预测事件类型
        mark_logits = self.model.mark_linear(last_context)  # [1, num_types]
        type_probs = torch.softmax(mark_logits, dim=-1).squeeze(0).cpu().numpy()
        next_type = int(mark_logits.argmax(dim=-1).item())
        
        result = {
            'next_dtime': next_dtime,
            'next_time': next_time,
            'next_type': next_type,
            'type_probs': type_probs
        }
        
        if return_distribution:
            result['dtime_samples'] = dtime_samples.cpu().numpy()
            result['dtime_mean'] = inter_time_dist.mean.item()
            result['dtime_std'] = dtime_samples.std().item()
            result['mix_weights'] = torch.softmax(log_weights, dim=-1).squeeze(0).cpu().numpy()
            result['mix_means'] = locs.squeeze(0).cpu().numpy()
            result['mix_stds'] = log_scales.exp().squeeze(0).cpu().numpy()
        
        return result
    
    @torch.no_grad()
    def predict_sequence(
        self,
        history_times: np.ndarray,
        history_types: np.ndarray,
        num_steps: int = 10,
        num_samples: int = 100
    ) -> Dict:
        """
        自回归地预测未来多个事件
        
        Args:
            history_times: 历史事件时间
            history_types: 历史事件类型
            num_steps: 预测步数
            num_samples: 每步采样次数
        
        Returns:
            dict with:
                - pred_times: [num_steps] 预测的绝对时间
                - pred_dtimes: [num_steps] 预测的时间间隔
                - pred_types: [num_steps] 预测的事件类型
        """
        pred_times = []
        pred_dtimes = []
        pred_types = []
        
        current_times = history_times.copy()
        current_types = history_types.copy()
        
        for step in range(num_steps):
            # 预测下一个事件
            result = self.predict_next(
                current_times, current_types, num_samples
            )
            
            pred_times.append(result['next_time'])
            pred_dtimes.append(result['next_dtime'])
            pred_types.append(result['next_type'])
            
            # 更新历史
            current_times = np.append(current_times, result['next_time'])
            current_types = np.append(current_types, result['next_type'])
        
        return {
            'pred_times': np.array(pred_times),
            'pred_dtimes': np.array(pred_dtimes),
            'pred_types': np.array(pred_types)
        }
```

---

## 7. 时空扩展：p(t,x) = p(t)p(x|t)

这是将 IFTPP 扩展到时空点过程的核心部分。

### 7.1 联合分布分解

```
p(t_{i+1}, x_{i+1} | H_i) = p(t_{i+1} | H_i) · p(x_{i+1} | t_{i+1}, H_i)
```

- **时间分布** `p(t_{i+1} | H_i)`: 使用 IFTPP 的混合对数正态分布
- **空间分布** `p(x_{i+1} | t_{i+1}, H_i)`: 使用条件 GMM

### 7.2 完整时空模型实现

```python
"""
spatiotemporal_iftpp.py - 时空点过程模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

from utils import LogNormalMixtureDistribution, clamp_preserve_gradients


class SpatialGMM(nn.Module):
    """
    空间高斯混合模型
    
    p(x | h, Δt) = Σₖ πₖ · N(x; μₖ, Σₖ)
    
    其中参数由隐状态 h 和预测的时间间隔 Δt 共同决定
    """
    
    def __init__(
        self,
        hidden_size: int,
        spatial_dim: int = 2,
        num_mix_components: int = 16
    ):
        super().__init__()
        
        self.spatial_dim = spatial_dim
        self.num_mix_components = num_mix_components
        
        # 输入: hidden_state + log(dt)
        input_size = hidden_size + 1
        
        # 预测 GMM 参数
        # 均值: K * spatial_dim
        # 对角协方差的对数: K * spatial_dim
        # 混合权重对数: K
        output_size = num_mix_components * (2 * spatial_dim + 1)
        
        self.param_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        dtime: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 GMM 参数
        
        Args:
            hidden_state: [batch, hidden_size]
            dtime: [batch] 预测的时间间隔
        
        Returns:
            means: [batch, K, spatial_dim]
            log_stds: [batch, K, spatial_dim]
            log_weights: [batch, K]
        """
        # 拼接输入
        log_dtime = torch.log(dtime + 1e-8).unsqueeze(-1)  # [batch, 1]
        x = torch.cat([hidden_state, log_dtime], dim=-1)
        
        # 预测参数
        params = self.param_net(x)  # [batch, K*(2*D+1)]
        
        K = self.num_mix_components
        D = self.spatial_dim
        
        means = params[:, :K*D].view(-1, K, D)
        log_stds = params[:, K*D:2*K*D].view(-1, K, D)
        log_weights = params[:, 2*K*D:]
        
        # 约束标准差
        log_stds = clamp_preserve_gradients(log_stds, -5.0, 3.0)
        
        return means, log_stds, log_weights
    
    def log_prob(
        self,
        x: torch.Tensor,
        means: torch.Tensor,
        log_stds: torch.Tensor,
        log_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        计算对数概率
        
        Args:
            x: [batch, spatial_dim] 观测位置
            means: [batch, K, spatial_dim]
            log_stds: [batch, K, spatial_dim]
            log_weights: [batch, K]
        
        Returns:
            log_prob: [batch]
        """
        K = self.num_mix_components
        D = self.spatial_dim
        
        x = x.unsqueeze(1)  # [batch, 1, D]
        stds = log_stds.exp()  # [batch, K, D]
        
        # 各组件的对数概率 (各向异性高斯)
        # log N(x; μ, σ) = -0.5 * D * log(2π) - Σ log(σ_d) - 0.5 * Σ ((x_d - μ_d) / σ_d)²
        log_prob_components = (
            -0.5 * D * math.log(2 * math.pi)
            - log_stds.sum(dim=-1)  # [batch, K]
            - 0.5 * (((x - means) / stds) ** 2).sum(dim=-1)  # [batch, K]
        )
        
        # 混合权重归一化
        log_weights_normalized = F.log_softmax(log_weights, dim=-1)  # [batch, K]
        
        # 混合对数概率 (log-sum-exp)
        log_prob = torch.logsumexp(
            log_weights_normalized + log_prob_components, dim=-1
        )  # [batch]
        
        return log_prob
    
    def sample(
        self,
        means: torch.Tensor,
        log_stds: torch.Tensor,
        log_weights: torch.Tensor,
        num_samples: int = 1
    ) -> torch.Tensor:
        """
        从 GMM 采样
        
        Args:
            means: [batch, K, D]
            log_stds: [batch, K, D]
            log_weights: [batch, K]
            num_samples: 采样数
        
        Returns:
            samples: [num_samples, batch, D]
        """
        batch_size = means.shape[0]
        K = self.num_mix_components
        D = self.spatial_dim
        device = means.device
        
        # 采样组件索引
        weights = F.softmax(log_weights, dim=-1)  # [batch, K]
        component_idx = torch.multinomial(
            weights, num_samples, replacement=True
        )  # [batch, num_samples]
        
        # 收集对应组件的参数
        # [batch, num_samples, D]
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_samples)
        selected_means = means[batch_idx, component_idx]  # [batch, num_samples, D]
        selected_stds = log_stds.exp()[batch_idx, component_idx]  # [batch, num_samples, D]
        
        # 从正态分布采样
        eps = torch.randn_like(selected_means)
        samples = selected_means + selected_stds * eps  # [batch, num_samples, D]
        
        return samples.permute(1, 0, 2)  # [num_samples, batch, D]


class SpatioTemporalIFTPP(nn.Module):
    """
    时空 IntensityFree Point Process
    
    联合分布: p(t, x | H) = p(t | H) · p(x | t, H)
    
    - 时间分布: 混合对数正态 (IFTPP)
    - 空间分布: 条件 GMM
    
    Args:
        num_event_types: 事件类型数
        hidden_size: 隐藏层维度
        num_time_mix: 时间分布混合组件数
        num_spatial_mix: 空间分布混合组件数
        spatial_dim: 空间维度 (通常为 2)
        mean_log_inter_time: 对数时间间隔均值
        std_log_inter_time: 对数时间间隔标准差
    """
    
    def __init__(
        self,
        num_event_types: int,
        hidden_size: int = 64,
        num_time_mix: int = 64,
        num_spatial_mix: int = 16,
        spatial_dim: int = 2,
        mean_log_inter_time: float = 0.0,
        std_log_inter_time: float = 1.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_event_types = num_event_types
        self.num_event_types_pad = num_event_types + 1
        self.pad_token_id = num_event_types
        
        self.hidden_size = hidden_size
        self.num_time_mix = num_time_mix
        self.num_spatial_mix = num_spatial_mix
        self.spatial_dim = spatial_dim
        self.mean_log_inter_time = mean_log_inter_time
        self.std_log_inter_time = std_log_inter_time
        
        self.eps = torch.finfo(torch.float32).eps
        
        # ============ 编码器 ============
        
        # 事件类型嵌入
        self.type_emb = nn.Embedding(
            self.num_event_types_pad, hidden_size, 
            padding_idx=self.pad_token_id
        )
        
        # 空间位置编码
        self.spatial_encoder = nn.Linear(spatial_dim, hidden_size)
        
        # 输入特征: log(dt) + type_emb + spatial_emb
        self.num_features = 1 + hidden_size + hidden_size
        
        # GRU 编码器
        self.rnn = nn.GRU(
            input_size=self.num_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # ============ 预测头 ============
        
        # 时间分布参数 (IFTPP)
        self.time_linear = nn.Linear(hidden_size, 3 * num_time_mix)
        
        # 事件类型
        self.mark_linear = nn.Linear(hidden_size, self.num_event_types_pad)
        
        # 空间分布 (条件 GMM)
        self.spatial_gmm = SpatialGMM(
            hidden_size=hidden_size,
            spatial_dim=spatial_dim,
            num_mix_components=num_spatial_mix
        )
    
    def encode(
        self,
        time_delta_seqs: torch.Tensor,
        type_seqs: torch.Tensor,
        spatial_seqs: torch.Tensor
    ) -> torch.Tensor:
        """
        编码历史序列
        
        Args:
            time_delta_seqs: [B, L] 时间间隔
            type_seqs: [B, L] 事件类型
            spatial_seqs: [B, L, D] 空间位置
        
        Returns:
            context: [B, L, H]
        """
        # 时间编码
        temporal = torch.log(time_delta_seqs + self.eps).unsqueeze(-1)  # [B, L, 1]
        
        # 类型嵌入
        type_emb = self.type_emb(type_seqs)  # [B, L, H]
        
        # 空间编码
        spatial_emb = self.spatial_encoder(spatial_seqs)  # [B, L, H]
        
        # 拼接
        rnn_input = torch.cat([temporal, type_emb, spatial_emb], dim=-1)  # [B, L, 1+2H]
        
        # GRU
        context, _ = self.rnn(rnn_input)  # [B, L, H]
        context = self.dropout(context)
        
        return context
    
    def get_time_distribution(
        self,
        context: torch.Tensor
    ) -> LogNormalMixtureDistribution:
        """获取时间间隔分布"""
        raw_params = self.time_linear(context)
        
        K = self.num_time_mix
        locs = raw_params[..., :K]
        log_scales = raw_params[..., K:2*K]
        log_weights = raw_params[..., 2*K:]
        
        log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
        
        return LogNormalMixtureDistribution(
            locs, log_scales, log_weights,
            self.mean_log_inter_time,
            self.std_log_inter_time
        )
    
    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算联合负对数似然损失
        
        L = -log p(t|H) - log p(x|t,H) - log p(k|H)
        """
        dtime_seqs = batch['time_delta_seqs']
        type_seqs = batch['type_seqs']
        spatial_seqs = batch['spatial_seqs']
        mask = batch['seq_non_pad_mask']
        
        # 编码历史 (排除最后一个)
        context = self.encode(
            dtime_seqs[:, :-1], 
            type_seqs[:, :-1], 
            spatial_seqs[:, :-1]
        )  # [B, L-1, H]
        
        event_mask = mask[:, 1:].float()  # [B, L-1]
        
        # === 时间对数似然 ===
        time_dist = self.get_time_distribution(context)
        target_dtime = dtime_seqs[:, 1:].clamp(min=1e-5)
        time_ll = time_dist.log_prob(target_dtime) * event_mask
        
        # === 空间对数似然 ===
        # 对于每个位置，使用预测的时间间隔来计算空间分布
        batch_size, seq_len = context.shape[:2]
        
        spatial_ll = torch.zeros_like(event_mask)
        
        # 获取预测的时间间隔均值
        dtime_pred = time_dist.mean  # [B, L-1]
        
        for i in range(seq_len):
            h_i = context[:, i, :]  # [B, H]
            dt_i = dtime_pred[:, i]  # [B]
            
            # 计算空间 GMM 参数
            means, log_stds, log_weights = self.spatial_gmm(h_i, dt_i)
            
            # 计算对数似然
            target_x = spatial_seqs[:, i + 1, :]  # [B, D]
            spatial_ll[:, i] = self.spatial_gmm.log_prob(
                target_x, means, log_stds, log_weights
            )
        
        spatial_ll = spatial_ll * event_mask
        
        # === 事件类型对数似然 ===
        mark_logits = self.mark_linear(context)
        mark_log_probs = F.log_softmax(mark_logits, dim=-1)
        target_types = type_seqs[:, 1:]
        mark_ll = mark_log_probs.gather(-1, target_types.unsqueeze(-1)).squeeze(-1)
        mark_ll = mark_ll * event_mask
        
        # === 总损失 ===
        total_ll = time_ll + spatial_ll + mark_ll
        num_events = event_mask.sum()
        
        loss = -total_ll.sum() / (num_events + self.eps)
        
        metrics = {
            'loss': loss.item(),
            'time_nll': (-time_ll.sum() / num_events).item(),
            'spatial_nll': (-spatial_ll.sum() / num_events).item(),
            'mark_nll': (-mark_ll.sum() / num_events).item(),
            'num_events': num_events.item()
        }
        
        return loss, metrics
    
    @torch.no_grad()
    def predict_next_event(
        self,
        batch: Dict[str, torch.Tensor],
        num_samples: int = 100
    ) -> Dict[str, torch.Tensor]:
        """
        预测下一个事件的时间、位置和类型
        
        Returns:
            dtime_pred: [B, L] 预测时间间隔
            spatial_pred: [B, L, D] 预测位置
            type_pred: [B, L] 预测类型
        """
        dtime_seqs = batch['time_delta_seqs']
        type_seqs = batch['type_seqs']
        spatial_seqs = batch['spatial_seqs']
        
        # 编码
        context = self.encode(dtime_seqs, type_seqs, spatial_seqs)
        
        # 时间预测
        time_dist = self.get_time_distribution(context)
        dtime_samples = time_dist.sample((num_samples,))
        dtime_pred = dtime_samples.mean(dim=0)  # [B, L]
        
        # 空间预测
        batch_size, seq_len = context.shape[:2]
        spatial_pred = torch.zeros(batch_size, seq_len, self.spatial_dim, device=context.device)
        
        for i in range(seq_len):
            h_i = context[:, i, :]
            dt_i = dtime_pred[:, i]
            
            means, log_stds, log_weights = self.spatial_gmm(h_i, dt_i)
            samples = self.spatial_gmm.sample(means, log_stds, log_weights, num_samples)
            spatial_pred[:, i, :] = samples.mean(dim=0)
        
        # 类型预测
        mark_logits = self.mark_linear(context)
        type_pred = mark_logits.argmax(dim=-1)
        
        return {
            'dtime_pred': dtime_pred,
            'spatial_pred': spatial_pred,
            'type_pred': type_pred,
            'type_probs': F.softmax(mark_logits, dim=-1)
        }
```

---

## 8. 评估指标接口

```python
"""
evaluation.py - 评估指标
"""
import numpy as np
from typing import Dict, List, Optional
from scipy.spatial.distance import cdist


def compute_temporal_metrics(
    pred_dtimes: np.ndarray,
    true_dtimes: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    计算时间预测指标
    
    Args:
        pred_dtimes: 预测的时间间隔
        true_dtimes: 真实的时间间隔
        mask: 有效位置掩码
    
    Returns:
        dict with RMSE, MAE, MAPE
    """
    if mask is not None:
        pred_dtimes = pred_dtimes[mask]
        true_dtimes = true_dtimes[mask]
    
    errors = pred_dtimes - true_dtimes
    
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(errors) / (true_dtimes + 1e-8)) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape
    }


def compute_spatial_metrics(
    pred_locations: np.ndarray,
    true_locations: np.ndarray,
    mask: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    计算空间预测指标
    
    Args:
        pred_locations: [N, D] 预测位置
        true_locations: [N, D] 真实位置
        mask: 有效位置掩码
    
    Returns:
        dict with displacement error metrics
    """
    if mask is not None:
        pred_locations = pred_locations[mask]
        true_locations = true_locations[mask]
    
    # 欧氏距离误差
    displacements = np.linalg.norm(pred_locations - true_locations, axis=-1)
    
    return {
        'mean_displacement': np.mean(displacements),
        'median_displacement': np.median(displacements),
        'std_displacement': np.std(displacements),
        'rmse_displacement': np.sqrt(np.mean(displacements ** 2))
    }


def compute_type_metrics(
    pred_types: np.ndarray,
    true_types: np.ndarray,
    mask: Optional[np.ndarray] = None,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    计算事件类型预测指标
    """
    if mask is not None:
        pred_types = pred_types[mask]
        true_types = true_types[mask]
    
    accuracy = np.mean(pred_types == true_types)
    
    # 如果提供类别数，计算 F1
    if num_classes is not None:
        from sklearn.metrics import f1_score
        f1_macro = f1_score(true_types, pred_types, average='macro', zero_division=0)
        f1_micro = f1_score(true_types, pred_types, average='micro', zero_division=0)
    else:
        f1_macro = None
        f1_micro = None
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro
    }


class STPPEvaluator:
    """
    时空点过程评估器
    
    整合所有评估指标，提供统一接口
    """
    
    def __init__(self, num_event_types: int):
        self.num_event_types = num_event_types
    
    def evaluate(
        self,
        predictions: Dict[str, np.ndarray],
        labels: Dict[str, np.ndarray],
        mask: np.ndarray
    ) -> Dict[str, float]:
        """
        完整评估
        
        Args:
            predictions: 包含 dtime_pred, spatial_pred, type_pred
            labels: 包含 dtime_true, spatial_true, type_true
            mask: 有效位置掩码
        
        Returns:
            所有评估指标
        """
        results = {}
        
        # 时间指标
        if 'dtime_pred' in predictions and 'dtime_true' in labels:
            time_metrics = compute_temporal_metrics(
                predictions['dtime_pred'],
                labels['dtime_true'],
                mask
            )
            results.update({f'time_{k}': v for k, v in time_metrics.items()})
        
        # 空间指标
        if 'spatial_pred' in predictions and 'spatial_true' in labels:
            spatial_metrics = compute_spatial_metrics(
                predictions['spatial_pred'],
                labels['spatial_true'],
                mask
            )
            results.update({f'spatial_{k}': v for k, v in spatial_metrics.items()})
        
        # 类型指标
        if 'type_pred' in predictions and 'type_true' in labels:
            type_metrics = compute_type_metrics(
                predictions['type_pred'],
                labels['type_true'],
                mask,
                self.num_event_types
            )
            results.update({f'type_{k}': v for k, v in type_metrics.items()})
        
        return results
```

---

## 9. 完整使用示例

```python
"""
main.py - 完整使用示例
"""
import torch
import numpy as np
import json

from data_processing import TPPDataset, TPPDataCollator, compute_data_statistics
from spatiotemporal_iftpp import SpatioTemporalIFTPP
from trainer import IFTPPTrainer
from evaluation import STPPEvaluator

def main():
    # ============ 1. 配置 ============
    config = {
        'num_event_types': 5,
        'hidden_size': 64,
        'num_time_mix': 64,
        'num_spatial_mix': 16,
        'spatial_dim': 2,
        'batch_size': 32,
        'max_len': 100,
        'learning_rate': 1e-3,
        'num_epochs': 100,
        'patience': 10,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    # ============ 2. 加载数据 ============
    # 假设数据格式
    with open('data.json', 'r') as f:
        raw_data = json.load(f)
    
    train_data = raw_data['train']
    valid_data = raw_data['dev']
    test_data = raw_data['test']
    
    # 创建数据集
    train_dataset = TPPDataset(data=train_data, max_len=config['max_len'])
    valid_dataset = TPPDataset(data=valid_data, max_len=config['max_len'])
    test_dataset = TPPDataset(data=test_data, max_len=config['max_len'])
    
    # 计算归一化统计量
    stats = compute_data_statistics(train_dataset)
    print(f"Data statistics: {stats}")
    
    # 数据加载器
    collator = TPPDataCollator(
        pad_token_id=config['num_event_types'],
        max_len=config['max_len']
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'],
        shuffle=True, collate_fn=collator
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config['batch_size'],
        shuffle=False, collate_fn=collator
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config['batch_size'],
        shuffle=False, collate_fn=collator
    )
    
    # ============ 3. 创建模型 ============
    model = SpatioTemporalIFTPP(
        num_event_types=config['num_event_types'],
        hidden_size=config['hidden_size'],
        num_time_mix=config['num_time_mix'],
        num_spatial_mix=config['num_spatial_mix'],
        spatial_dim=config['spatial_dim'],
        mean_log_inter_time=stats['mean_log_inter_time'],
        std_log_inter_time=stats['std_log_inter_time']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # ============ 4. 训练 ============
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    trainer = IFTPPTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config['device']
    )
    
    history = trainer.train(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=config['num_epochs'],
        patience=config['patience'],
        save_path='best_stpp_model.pt'
    )
    
    # ============ 5. 测试 ============
    test_metrics = trainer.evaluate(test_loader)
    print(f"\nTest Results:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")
    
    # ============ 6. 预测示例 ============
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(config['device']) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            
            predictions = model.predict_next_event(batch)
            
            print("\nPrediction example (first sequence):")
            print(f"  Predicted dtimes: {predictions['dtime_pred'][0, :5].cpu().numpy()}")
            print(f"  True dtimes: {batch['time_delta_seqs'][0, 1:6].cpu().numpy()}")
            print(f"  Predicted locations: {predictions['spatial_pred'][0, :5].cpu().numpy()}")
            print(f"  True locations: {batch['spatial_seqs'][0, 1:6].cpu().numpy()}")
            break


if __name__ == '__main__':
    main()
```

---

## 10. Agent 沟通建议

### 10.1 给 Agent 的 Prompt 模板

当你把这个文档传给另一个 Agent 时，建议使用以下结构化 prompt：

```markdown
# 任务: 将 IntensityFree TPP 集成到时空点过程框架

## 背景
我需要将 EasyTPP 中的 IntensityFree 模型复现到我的时空点过程代码框架中。
已有的参考文档: [附上此文档]

## 具体需求
1. 时间建模: 使用 IFTPP 的混合对数正态分布
2. 空间建模: 使用条件 GMM，p(x|t,h)
3. 联合分布: p(t,x) = p(t|h) * p(x|t,h)

## 我的代码框架结构
[描述你的代码框架目录结构和关键文件]

## 需要的输出
1. 适配我的框架的 IFTPP 模型类
2. 数据加载器的修改
3. 训练脚本的修改
4. 评估脚本的修改

## 关键实现细节 (来自参考文档)
- 混合分布组件数: 默认 64 (可配置)
- 时间编码: log(dt + eps)
- 损失函数: 负对数似然
- 预测方式: 从分布采样取均值
```

### 10.2 建议的分步实施方案

建议你按以下顺序让 Agent 实施：

1. **第一步：纯时间模型**
   - 先实现不含空间的 IFTPP
   - 验证时间预测的 RMSE 指标

2. **第二步：添加空间组件**
   - 添加 SpatialGMM 模块
   - 实现联合损失函数

3. **第三步：集成评估**
   - 对接你框架的评估流程
   - 添加时空联合指标

### 10.3 常见问题和解决方案

| 问题 | 解决方案 |
|------|----------|
| 时间间隔为0或负值 | 使用 `clamp(min=1e-5)` |
| 对数标准差爆炸 | 使用 `clamp_preserve_gradients(-5, 3)` |
| 混合权重数值不稳定 | 使用 `log_softmax` 而非 `softmax + log` |
| 预测时间过大/过小 | 检查归一化参数是否正确 |
| 空间预测偏差大 | 增加 `num_spatial_mix` 组件数 |

---

## 附录：EasyTPP 关键文件清单

供 Agent 快速定位的文件列表：

| 文件 | 内容 |
|------|------|
| `easy_tpp/model/torch_model/torch_intensity_free.py` | IFTPP 模型实现 |
| `easy_tpp/utils/torch_utils.py` | LogNormalMixtureDistribution |
| `easy_tpp/model/torch_model/torch_basemodel.py` | 模型基类 |
| `easy_tpp/preprocess/dataset.py` | 数据集类 |
| `easy_tpp/preprocess/data_collator.py` | 数据整理器 |
| `easy_tpp/runner/tpp_runner.py` | 训练运行器 |
| `easy_tpp/config_factory/model_config.py` | 模型配置 |
| `examples/train_nhp.py` | 训练示例 |

---

## 附录：配置文件完整示例

```yaml
# config.yaml - IFTPP 完整配置示例

IntensityFree_train:
  base_config:
    stage: train
    backend: torch
    dataset_id: your_dataset
    runner_id: std_tpp
    model_id: IntensityFree
    base_dir: './checkpoints/'
  
  data_config:
    data_dir: './data/'
    data_format: json  # 或 pkl
    train_file: train.json
    valid_file: dev.json
    test_file: test.json
  
  data_specs:
    num_event_types: 5           # 事件类型数（不含 PAD）
    pad_token_id: 5              # PAD token ID = num_event_types
    padding_side: right          # 填充方向
    truncation_side: right       # 截断方向
    padding_strategy: longest    # 或 max_length
    max_len: 100                 # 最大序列长度
  
  trainer_config:
    seed: 2019
    gpu: 0                       # -1 表示 CPU
    batch_size: 256
    max_epoch: 200
    shuffle: True
    optimizer: adam
    learning_rate: 1.e-3
    valid_freq: 1                # 每 N 个 epoch 验证一次
    use_tfb: False               # 是否使用 TensorBoard
    metrics: ['acc', 'rmse']
    patience: 10                 # 早停耐心值
    grad_clip: 5.0               # 梯度裁剪
  
  model_config:
    hidden_size: 64              # GRU 隐藏维度
    dropout: 0.1
    model_specs:
      num_mix_components: 64     # 混合分布组件数 (关键参数)
    
    # 归一化参数 (可选，从数据计算或设为默认)
    mean_log_inter_time: 0.0
    std_log_inter_time: 1.0
    
    # 预测采样配置
    thinning:
      num_sample: 100            # 采样次数
      num_exp: 500
      dtime_max: 5

# 时空扩展配置
SpatioTemporalIFTPP_train:
  # 继承上述配置，添加空间相关配置
  model_config:
    hidden_size: 64
    num_time_mix: 64             # 时间混合组件数
    num_spatial_mix: 16          # 空间混合组件数
    spatial_dim: 2               # 空间维度
```

---

**文档版本**: v1.0  
**最后更新**: 2026年1月25日  
**基于**: EasyTPP 代码库分析

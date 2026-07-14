import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any


class CosineSimilarityEnhancement(nn.Module):
    """
    余弦相似度增强模块
    
    该模块通过计算T1和T2两个时相特征的余弦相似度，生成变化权重来增强感知帧P。
    
    核心思想：
    - 空间维度：计算每个空间位置的特征相似度 → 定位"在哪变化"
    - 通道维度：计算每个通道的特征相似度 → 筛选"哪些特征通道敏感"
    - 双维度融合：空间权重 × 通道权重 → 精确的变化权重图
    - 增强策略：相似度越低 → 变化越大 → 权重越高
    
    工作流程：利用该权重对感知帧进行增强：
    1. 输入5D特征 [B, C, 3, H, W]，其中T=3代表[T1, P, T2]
    2. 提取T1和T2帧
    3. 计算双维度余弦相似度权重
    4. 使用权重增强感知帧P
    5. 输出增强后的5D特征
    """
    
    def __init__(self):
        """
        初始化余弦相似度增强模块
        
        入参:
        - 无（纯计算模块，无可学习参数）
        
        方法:
        - 该模块不包含可学习参数，完全基于余弦相似度计算
        
        出参:
        - None
        """
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：基于余弦相似度增强感知帧
        
        入参:
        - x (torch.Tensor): 输入5D特征张量，形状 [B, C, T, H, W]
                           其中T=3，代表 [T1, P, T2]
        
        方法:
        1. 验证时间维度为3
        2. 提取T1、P、T2三帧
        3. 计算T1和T2的余弦相似度变化权重
        4. 使用权重增强感知帧P
        5. 重构增强后的5D特征
        
        出参:
        - enhanced_x (torch.Tensor): 增强后的5D特征，形状 [B, C, T, H, W]
        """
        # 验证时间维度
        if x.shape[2] != 3:
            raise ValueError(f"时间维度必须为3，但得到 {x.shape[2]}")
        
        # 提取三帧
        middle_idx = x.shape[2] // 2  # 中间帧索引 (P帧)
        T1 = x[:, :, 0]              # T1帧 [B, C, H, W]
        T2 = x[:, :, 2]              # T2帧 [B, C, H, W]
        P = x[:, :, middle_idx]      # 感知帧P [B, C, H, W]
        
        # 计算T1和T2的余弦相似度权重 [B, C, H, W]
        similarity_weights = self._compute_similarity_weights(T1, T2)
        
        # 使用变化权重增强感知帧P，得到增强后的感知帧P
        enhanced_P = P * (1 + similarity_weights)
        
        # 重构增强后的5D特征
        enhanced_x = x.clone()
        enhanced_x[:, :, middle_idx] = enhanced_P
        
        return enhanced_x
    
    def _compute_similarity_weights(self, T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
        """
        计算基于余弦相似度的变化权重（双维度融合）
        
        入参:
        - T1 (torch.Tensor): T1时相特征，形状 [B, C, H, W]
        - T2 (torch.Tensor): T2时相特征，形状 [B, C, H, W]
        
        方法:
        1. 计算空间维度的余弦相似度权重 [B, 1, H, W]
        2. 计算通道维度的余弦相似度权重 [B, C, 1, 1]
        3. 融合两种权重（逐元素相乘）
        
        出参:
        - combined_weights (torch.Tensor): 融合后的变化权重，形状 [B, C, H, W]
        """
        # 计算空间差异权重
        spatial_weights = self._spatial_similarity_weights(T1, T2)
        
        # 计算通道差异权重  
        channel_weights = self._channel_similarity_weights(T1, T2)
        
        # 融合两种权重
        combined_weights = spatial_weights * channel_weights
        
        return combined_weights
    
    def _spatial_similarity_weights(self, T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
        """
        计算空间维度的余弦相似度权重
        
        入参:
        - T1 (torch.Tensor): T1时相特征，形状 [B, C, H, W]
        - T2 (torch.Tensor): T2时相特征，形状 [B, C, H, W]
        
        方法:
        1. 将特征重塑为 [B*H*W, C]，每个空间位置是一个C维向量
        2. 计算每个空间位置的余弦相似度
        3. 重塑回空间维度 [B, 1, H, W]
        4. 通过 1 - sigmoid(similarity) 转换为变化权重
           - 相似度高 → sigmoid(sim)≈1 → 权重≈0 (不变区域)
           - 相似度低 → sigmoid(sim)≈0 → 权重≈1 (变化区域)
        
        出参:
        - spatial_weights (torch.Tensor): 空间变化权重，形状 [B, 1, H, W]
        """
        B, C, H, W = T1.shape
        
        # 将特征重塑为 [B*H*W, C] - 每个空间位置是一个C维向量
        T1_flat = T1.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        T2_flat = T2.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        
        # 计算余弦相似度 (范围 [-1, 1])
        cosine_sim = F.cosine_similarity(T1_flat, T2_flat, dim=1)  # [B*H*W]
        
        # 重塑回空间维度
        cosine_sim = cosine_sim.view(B, H, W)  # [B, H, W]
        
        # 添加通道维度
        cosine_sim = cosine_sim.unsqueeze(1)  # [B, 1, H, W]
        
        # 转换为变化权重：相似度越低，权重越高
        # sigmoid将[-1,1]映射到[0,1]，然后1-sigmoid得到变化权重
        spatial_weights = 1 - torch.sigmoid(cosine_sim)
        
        return spatial_weights
    
    def _channel_similarity_weights(self, T1: torch.Tensor, T2: torch.Tensor) -> torch.Tensor:
        """
        计算通道维度的余弦相似度权重
        
        入参:
        - T1 (torch.Tensor): T1时相特征，形状 [B, C, H, W]
        - T2 (torch.Tensor): T2时相特征，形状 [B, C, H, W]
        
        方法:
        1. 将每个通道的空间特征展平为 [B, C, H*W]
        2. 计算每个通道的余弦相似度（在空间维度上）
        3. 扩展维度为 [B, C, 1, 1]
        4. 通过 1 - sigmoid(similarity) 转换为变化权重
           - 某通道相似度高 → 该通道对变化不敏感 → 权重低
           - 某通道相似度低 → 该通道对变化敏感 → 权重高
        
        出参:
        - channel_weights (torch.Tensor): 通道变化权重，形状 [B, C, 1, 1]
        """
        B, C, H, W = T1.shape
        
        # 将每个通道的空间特征展平 [B, C, H*W]
        T1_flat = T1.view(B, C, -1)  # 每个通道有H*W个值
        T2_flat = T2.view(B, C, -1)
        
        # 计算每个通道的余弦相似度（在空间维度上）
        cosine_sim = F.cosine_similarity(T1_flat, T2_flat, dim=2)  # [B, C]
        
        # 扩展空间维度 [B, C, 1, 1]
        cosine_sim = cosine_sim.unsqueeze(-1).unsqueeze(-1)
        
        # 转换为变化权重：相似度越低，权重越高
        channel_weights = 1 - torch.sigmoid(cosine_sim)
        
        return channel_weights
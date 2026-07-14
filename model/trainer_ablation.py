# Copyright (c) Duowang Zhu.
# All rights reserved.

"""
消融实验专用Trainer模块

该模块通过配置开关灵活控制各个增强模块的启用/禁用，用于系统化的消融实验。

支持的模块组合：
1. Base: Encoder → Decoder
2. Base + Attention: Encoder → Attention → Decoder
3. Base + ASPP: Encoder → ASPP → Decoder
4. Base + Transformer: Encoder → Transformer → Decoder
5. Base + Attention + ASPP: Encoder → Attention → ASPP → Decoder
6. Base + Attention + Transformer: Encoder → Attention → Transformer → Decoder
7. Base + ASPP + Transformer: Encoder → ASPP → Transformer → Decoder

配置参数：
- args.use_attention: 是否启用双路径注意力模块
- args.use_aspp: 是否启用ShuffleASPP模块
- args.use_transformer: 是否启用分层时空融合模块
"""

from functools import partial
import math
import logging
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from model.utils import weight_init
from model.attention import DualPathAttentionModule 
from model.STMDCM import ShuffleASPP3DModule
from model.CosineSimilarity import CosineSimilarityEnhancement 
from model.ST import HierarchicalCrossAttentionModule
from model.changedecoder import ChangeDecoder
from model.position_encoding import PositionEmbeddingSine
from model.x3d import create_x3d


class Encoder(nn.Module):
    """
    基于X3D架构的编码器模块（与原版相同）
    
    入参:
    - args: 配置参数对象，包含模型超参数
    
    方法:
    - 创建可学习的感知帧作为T1和T2之间的"探针"
    - 使用X3D 3D卷积网络提取时空特征
    - 使用余弦相似度增强模块增强每个阶段的特征
    
    出参:
    - stage_features (List[torch.Tensor]): 4个5D特征列表 [B,C,3,H,W]
    """
    
    def __init__(self, args: Any) -> None:
        super().__init__()
        self.args = args

        # 使用TFFM3D要求num_perception_frame为1
        assert args.num_perception_frame == 1, \
            "基于TFFM3D的编码器要求args.num_perception_frame == 1"

        # 1. 创建X3D模型
        full_x3d = create_x3d(input_clip_length=3, depth_factor=5.0)

        # 2. 提取骨干网络 (Stem + 4个阶段)
        self.backbone = nn.ModuleList(full_x3d.blocks[:5])
        
        # 3. 加载预训练权重
        if args.pretrained and args.pretrained != '':
            try:
                # 使用绝对路径以确保正确加载
                import os
                pretrained_path = os.path.abspath(args.pretrained)
                
                # 加载权重文件
                loaded_data = torch.load(pretrained_path, map_location='cpu')
                
                # 兼容不同的权重文件格式
                if isinstance(loaded_data, dict):
                    # 尝试提取state_dict（支持多种键名）
                    if 'model_state' in loaded_data:
                        state_dict = loaded_data['model_state']
                    elif 'state_dict' in loaded_data:
                        state_dict = loaded_data['state_dict']
                    elif 'model' in loaded_data:
                        state_dict = loaded_data['model']
                    else:
                        # 假设整个字典就是状态字典
                        state_dict = loaded_data
                else:
                    # 如果不是字典，可能是直接保存的模型
                    state_dict = loaded_data.state_dict() if hasattr(loaded_data, 'state_dict') else loaded_data
                
                # 过滤以获得仅骨干网络的权重 ('blocks.0' 到 'blocks.4')
                state_dict_backbone = {k: v for k, v in state_dict.items() if k.startswith('blocks.') and not k.startswith('blocks.5')}
                
                # 将键从 'blocks.i.xxx' 重命名为 'i.xxx' 以匹配self.backbone的结构
                renamed_state_dict = {'.'.join(k.split('.')[1:]): v for k, v in state_dict_backbone.items()}

                # 将重命名的状态字典加载到我们的骨干网络中
                msg = self.backbone.load_state_dict(renamed_state_dict, strict=True)
                print(f'✅ 加载预训练权重成功: {pretrained_path}')
                print(f'   加载信息: {msg}')

            except Exception as e:
                print(f'⚠️ 加载预训练权重失败: {e}')
                print(f'   权重路径: {args.pretrained}')
                print(f'   将使用随机初始化的权重继续...')
        else:
            print(f'ℹ️  未指定预训练权重，使用随机初始化')

        # 用于变化提取的可学习感知帧
        self.perception_frames = nn.Parameter(
            torch.randn(1, 3, 1, args.in_height, args.in_width), 
            requires_grad=True
        )
        
        # 余弦相似度增强模块
        self.cosine_enhancement = CosineSimilarityEnhancement()

    def base_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        编码器的基础前向传播
        
        入参:
        - x (torch.Tensor): 输入5D张量，形状 [B, 3, 3, H, W]
        
        方法:
        1. 通过stem层提取初始特征
        2. 使用余弦相似度增强模块增强stem特征
        3. 逐阶段提取特征，每个阶段后使用余弦相似度增强
        
        出参:
        - stage_features (List[torch.Tensor]): 4个阶段的5D特征列表
        """
        assert x.shape[2] == 3, f"TFFM3D要求时间维度为3，但得到 {x.shape[2]}"

        stage_features = []

        # 1. Stem层（使用余弦增强）
        stem_feature = self.backbone[0](x)
        stem_feature = self.cosine_enhancement(stem_feature)

        # 2. 逐阶段提取特征（使用余弦增强）
        feature = stem_feature
        for i in range(4):
            feature = self.backbone[i + 1](feature)
            feature = self.cosine_enhancement(feature)
            stage_features.append(feature)

        return stage_features

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> List[torch.Tensor]:
        """
        编码器前向传播
        
        入参:
        - x (torch.Tensor): T1时相图像 [B, 3, H, W]
        - y (torch.Tensor): T2时相图像 [B, 3, H, W]
        
        方法:
        - 构建三帧序列 [T1, P, T2]
        - 通过X3D骨干网络提取多尺度5D特征
        
        出参:
        - stage_features (List[torch.Tensor]): 4个5D特征 [B,C,3,H,W]
        """        
        # 扩展感知帧
        expand_percep_frames = self.perception_frames.expand(x.shape[0], -1, -1, -1, -1)

        # 合并为3帧序列 [输入, 感知帧, 目标]
        frames = torch.cat([
            x.unsqueeze(2),
            expand_percep_frames,
            y.unsqueeze(2)
        ], dim=2)

        # 通过网络提取特征
        return self.base_forward(frames)


class TrainerAblation(nn.Module):
    """
    消融实验专用训练器
    
    通过配置开关灵活控制各增强模块的启用/禁用，支持以下7种组合：
    1. Base only
    2. Base + Attention
    3. Base + ASPP
    4. Base + Transformer
    5. Base + Attention + ASPP
    6. Base + Attention + Transformer
    7. Base + ASPP + Transformer
    
    入参:
    - args: 配置参数对象，必须包含以下布尔开关：
        - args.use_attention (bool): 是否启用双路径注意力模块
        - args.use_aspp (bool): 是否启用ShuffleASPP模块
        - args.use_transformer (bool): 是否启用分层时空融合模块
    
    方法:
    - 根据配置动态初始化对应的模块
    - 在前向传播中根据开关决定特征流动路径
    - 保证无论哪种组合，输入解码器的都是5D特征
    
    出参:
    - prediction (torch.Tensor): 变化概率图 [B, 1, H, W]
    """
    
    def __init__(self, args: Any) -> None:
        """
        初始化消融实验训练器
        
        入参:
        - args: 配置参数对象
        
        方法:
        1. 读取配置开关（use_attention, use_aspp, use_transformer）
        2. 初始化基础编码器（必选）
        3. 根据开关初始化对应的增强模块
        4. 初始化解码器（必选）
        5. 对启用的模块进行权重初始化
        
        出参:
        - None
        """
        super().__init__()
        self.args = args
        
        # 读取配置开关
        self.use_attention = getattr(args, 'use_attention', False)
        self.use_aspp = getattr(args, 'use_aspp', False)
        self.use_transformer = getattr(args, 'use_transformer', False)
        
        # 编码器各阶段的输出通道数
        self.encoder_embed_dims = [24, 48, 96, 192]
        
        # === 必选模块 ===
        # 编码器（包含X3D骨干网络 + 余弦相似度增强模块）
        self.encoder = Encoder(args)
        
        # === 可选模块1: 双路径注意力 ===
        if self.use_attention:
            self.attention_modules = nn.ModuleList([
                DualPathAttentionModule(channels=dim) for dim in self.encoder_embed_dims
            ])
            print("[消融实验] 启用双路径注意力模块")
        else:
            self.attention_modules = None
            print("[消融实验] 禁用双路径注意力模块")
        
        # === 可选模块2: ShuffleASPP ===
        if self.use_aspp:
            self.aspp_modules = nn.ModuleList([
                ShuffleASPP3DModule(channels=dim) for dim in self.encoder_embed_dims
            ])
            print("[消融实验] 启用ShuffleASPP模块")
        else:
            self.aspp_modules = None
            print("[消融实验] 禁用ShuffleASPP模块")
        
        # === 可选模块3: 分层时空融合 ===
        if self.use_transformer:
            # 位置编码生成器
            self.pos_encoder = PositionEmbeddingSine(num_pos_feats=128, normalize=True)
            
            # 分层时空融合模块
            self.st_fusion = HierarchicalCrossAttentionModule(
                in_channels=self.encoder_embed_dims,
                num_heads=8,
                pe_dim=128
            )
            print("[消融实验] 启用分层时空融合模块")
        else:
            self.pos_encoder = None
            self.st_fusion = None
            print("[消融实验] 禁用分层时空融合模块")
        
        # === 必选模块: 解码器 ===
        self.decoder = ChangeDecoder(args=args, has_sigmoid=True)
        
        # === 权重初始化 ===
        if self.use_attention:
            weight_init(self.attention_modules)
        if self.use_aspp:
            weight_init(self.aspp_modules)
        weight_init(self.decoder)
        
        # 打印当前配置
        print(f"[消融实验配置] Attention={self.use_attention}, ASPP={self.use_aspp}, Transformer={self.use_transformer}")

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        可配置的前向传播
        
        入参:
        - x (torch.Tensor): 第一时相图像，形状 [B, 3, H, W]
        - y (torch.Tensor): 第二时相图像，形状 [B, 3, H, W]
        
        方法:
        1. 编码器提取多尺度5D特征（必选）
        2. 根据配置应用注意力增强（可选）
        3. 根据配置应用ASPP增强（可选）
        4. 根据配置应用Transformer融合（可选）
        5. 解码器生成变化检测图（必选）
        
        处理流程：
        - Base only: Encoder → Decoder
        - +Attention: Encoder → Attention → Decoder
        - +ASPP: Encoder → (Attention) → ASPP → Decoder
        - +Transformer: Encoder → (Attention) → (ASPP) → Transformer → Decoder
        
        出参:
        - prediction (torch.Tensor): 变化概率图，形状 [B, 1, H, W]
        """
        # === 阶段1: 编码器提取特征（必选） ===
        # 输出: 4个5D特征张量 [B,24,3,H,W], [B,48,3,H,W], [B,96,3,H,W], [B,192,3,H,W]
        stage_features_5d = self.encoder(x, y)
        
        # === 阶段2: 注意力增强（可选） ===
        if self.use_attention:
            enhanced_features_5d = []
            for i, feature_5d in enumerate(stage_features_5d):
                # 验证时间维度
                if feature_5d.size(2) != 3:
                    raise ValueError(f"时间维度必须为3，但在尺度{i}得到 {feature_5d.size(2)}")
                
                # 分解三帧
                t1 = feature_5d[:, :, 0, :, :]  # [B, C, H, W]
                p = feature_5d[:, :, 1, :, :]   # [B, C, H, W]
                t2 = feature_5d[:, :, 2, :, :]  # [B, C, H, W]
                
                # 从P帧生成注意力图
                attention_map = self.attention_modules[i](p)
                
                # 使用注意力图增强三帧
                t1_enhanced = attention_map * t1 + t1
                p_enhanced = attention_map * p + p
                t2_enhanced = attention_map * t2 + t2
                
                # 重构5D特征
                attention_enhanced = torch.stack([t1_enhanced, p_enhanced, t2_enhanced], dim=2)
                enhanced_features_5d.append(attention_enhanced)
            
            stage_features_5d = enhanced_features_5d
        
        # === 阶段3: ASPP增强（可选） ===
        if self.use_aspp:
            aspp_enhanced_features = []
            for i, feature_5d in enumerate(stage_features_5d):
                # 使用ShuffleASPP进行多尺度时空特征提取
                aspp_enhanced = self.aspp_modules[i](feature_5d)
                aspp_enhanced_features.append(aspp_enhanced)
            
            stage_features_5d = aspp_enhanced_features
        
        # === 阶段4: 分层时空融合（可选） ===
        if self.use_transformer:
            # 为每个尺度生成位置编码（从P帧提取）
            pe_list = [self.pos_encoder(f[:, :, 1, :, :]) for f in stage_features_5d]
            
            # 时空融合模块
            fused_features = self.st_fusion(stage_features_5d, pe_list)
            
            # Transformer输出的是5D特征，可以直接传入解码器
            stage_features_5d = fused_features
        
        # === 阶段5: 解码器生成变化图（必选） ===
        # 解码器接收5D特征，内部会自动融合时序信息
        prediction = self.decoder(stage_features_5d)
        
        return prediction


# === 便捷函数：创建特定配置的消融实验模型 ===

def create_ablation_model(args: Any, experiment_id: int) -> TrainerAblation:
    """
    根据实验ID创建对应配置的消融实验模型
    
    入参:
    - args: 基础配置参数
    - experiment_id (int): 实验编号 (1-7)
    
    方法:
    - 根据实验ID设置对应的模块开关
    - 创建并返回TrainerAblation实例
    
    出参:
    - model (TrainerAblation): 配置好的消融实验模型
    
    实验配置映射:
    1: Base only (无增强模块)
    2: Base + Attention
    3: Base + ASPP
    4: Base + Transformer
    5: Base + Attention + ASPP
    6: Base + Attention + Transformer
    7: Base + ASPP + Transformer
    """
    # 实验配置映射表
    experiment_configs = {
        1: {'use_attention': False, 'use_aspp': False, 'use_transformer': False},  # Base
        2: {'use_attention': True,  'use_aspp': False, 'use_transformer': False},  # Base + Attention
        3: {'use_attention': False, 'use_aspp': True,  'use_transformer': False},  # Base + ASPP
        4: {'use_attention': False, 'use_aspp': False, 'use_transformer': True},   # Base + Transformer
        5: {'use_attention': True,  'use_aspp': True,  'use_transformer': False},  # Base + Attention + ASPP
        6: {'use_attention': True,  'use_aspp': False, 'use_transformer': True},   # Base + Attention + Transformer
        7: {'use_attention': False, 'use_aspp': True,  'use_transformer': True},   # Base + ASPP + Transformer
    }
    
    if experiment_id not in experiment_configs:
        raise ValueError(f"无效的实验ID: {experiment_id}，必须在1-7之间")
    
    # 设置配置
    config = experiment_configs[experiment_id]
    for key, value in config.items():
        setattr(args, key, value)
    
    # 打印实验配置
    exp_names = {
        1: "Base Only",
        2: "Base + Attention",
        3: "Base + ASPP",
        4: "Base + Transformer",
        5: "Base + Attention + ASPP",
        6: "Base + Attention + Transformer",
        7: "Base + ASPP + Transformer"
    }
    print(f"\n{'='*60}")
    print(f"创建消融实验模型 - 实验{experiment_id}: {exp_names[experiment_id]}")
    print(f"{'='*60}\n")
    
    # 创建模型
    model = TrainerAblation(args)
    
    return model


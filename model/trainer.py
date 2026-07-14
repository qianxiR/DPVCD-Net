# Copyright (c) Duowang Zhu.
# All rights reserved.

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
from model.cascade_dcn import ShuffleASPP3DModule
from model.ST import HierarchicalCrossAttentionModule
from model.change_decoder import ChangeDecoder
from model.position_encoding import PositionEmbeddingSine
from model.x3d import create_x3d


class Encoder(nn.Module):
    """
    基于X3D架构的编码器模块
    
    该编码器专门用于处理双时相遥感图像的变化检测任务。它通过引入可学习的感知帧，
    将两张输入图像转换为一个三帧序列，然后使用X3D网络提取多尺度的深度特征。
    
    主要功能:
        - 创建可学习的感知帧作为T1和T2之间的"探针"
        - 使用X3D 3D卷积网络提取时空特征
        - 输出4个不同尺度的5D特征图用于后续处理
    
    输入要求:
        - 输入图像对: (B, C, H, W) 其中C=3 (RGB通道)
        - 要求args.num_perception_frame == 1
    
    输出格式:
        - 4个5D特征张量的列表: [(B, C1, 3, H1, W1), (B, C2, 3, H2, W2), ...]
        - 特征尺度依次递减: H1 > H2 > H3 > H4, W1 > W2 > W3 > W4
        - 通道数依次递增: C1 < C2 < C3 < C4 (默认: [24, 48, 96, 192])
    
    关键参数:
        - args.pretrained: 预训练X3D模型权重路径
        - args.in_height, args.in_width: 输入图像尺寸
        - args.num_perception_frame: 感知帧数量(必须为1)
    """
    
    def __init__(self, args: Any) -> None:
        """
        初始化编码器
        
        Args:
            args: 配置参数
        """
        super().__init__()
        self.args = args

        # 使用TFFM3D要求num_perception_frame为1，这样特征三元组的时间维度为3 ([pre, percep, post])
        assert args.num_perception_frame == 1, \
            "基于TFFM3D的编码器要求args.num_perception_frame == 1"

        # 1. 使用公共API创建一个完整的X3D模型
        full_x3d = create_x3d(input_clip_length=3, depth_factor=5.0)

        # 2. 仅提取骨干网络 (Stem + 4个阶段)
        self.backbone = nn.ModuleList(full_x3d.blocks[:5])
        # 每个stage的输出通道
        embed_dims = [24, 48, 96, 192]
        
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
        

    def base_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        编码器的基础前向传播
        
        入参:
        - x (torch.Tensor): 输入5D张量，形状 [B, 3, 3, H, W]
        
        方法:
        1. 通过stem层提取初始特征
        2. 使用余弦相似度增强模块增强stem特征
        3. 逐阶段提取特征，每个阶段后都进行余弦相似度增强
        4. 收集4个阶段的增强特征
        
        出参:
        - stage_features (List[torch.Tensor]): 4个阶段的5D特征列表
        """
        # 验证时间维度
        assert x.shape[2] == 3, f"TFFM3D要求时间维度为3，但得到 {x.shape[2]}"

        stage_features = []

        # 1. 先用stem层提取特征
        stem_feature = self.backbone[0](x)  # [B, 24, 3, H, W]

        # 2. 用stem_feature作为后续stage的输入
        feature = stem_feature
        for i in range(4):
            feature = self.backbone[i + 1](feature)
            stage_features.append(feature)

        return stage_features

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        使用输入和目标帧的前向传播
        
        Args:
            x: 输入帧张量，形状为 [B, C, H, W]
            y: 目标帧张量，形状为 [B, C, H, W]
            
        Returns:
            包含以下内容的元组:
            - 增强的5D干特征图
            - 来自X3D骨干网络4个阶段的5D特征张量列表
        """        
        # 扩展令牌以匹配批次大小
        expand_percep_frames = self.perception_frames.expand(x.shape[0], -1, -1, -1, -1)

        # 合并为3帧序列 [输入, 令牌, 目标]
        frames = torch.cat([
            x.unsqueeze(2),
            expand_percep_frames,
            y.unsqueeze(2)
        ], dim=2)

        # 通过网络处理以获得阶段特征
        return self.base_forward(frames)


class Trainer(nn.Module):
    """
    完整的变化检测模型架构（直接调用底层模块）
    
    架构流程：
    1. 编码器（含余弦相似度增强）提取多尺度5D特征
    2. 双路径注意力模块进行特征增强
    3. ShuffleASPP 3D模块进行多尺度时空特征提取
    4. 分层时空融合模块进行特征交互
    5. 解码器生成变化检测结果
    
    """
    
    def __init__(self, args: Any) -> None:
        """
        使用所有模型组件初始化训练器
        
        入参:
        - args: 配置参数对象，包含模型超参数
        
        方法:
        - 初始化编码器（含余弦相似度C模块）
        - 初始化多尺度双路径注意力模块
        - 初始化多尺度ShuffleASPP 3D模块
        - 初始化位置编码生成器
        - 初始化分层时空融合模块
        - 初始化解码器
        - 对需要训练的模块进行权重初始化
        
        出参:
        - None
        """
        super().__init__()
        self.args = args
    
        # 编码器各阶段的输出通道数
        self.encoder_embed_dims = [24, 48, 96, 192]
        
        # 编码器（包含X3D骨干网络 + 余弦相似度增强模块）
        self.encoder = Encoder(args)
        
        # 多尺度双路径注意力模块（通道注意力 + 空间注意力）
        self.attention_modules = nn.ModuleList([
            DualPathAttentionModule(channels=dim) for dim in self.encoder_embed_dims
        ])
        
        # 多尺度ShuffleASPP 3D模块（多尺度时空特征提取）
        self.aspp_modules = nn.ModuleList([
            ShuffleASPP3DModule(channels=dim) for dim in self.encoder_embed_dims
        ])
        
        # 位置编码生成器（用于为每个尺度的特征生成位置编码）
        self.pos_encoder = PositionEmbeddingSine(num_pos_feats=128, normalize=True)

        # 分层时空融合模块（自适应Swin Transformer交叉注意力）
        self.st_fusion = HierarchicalCrossAttentionModule(
            in_channels=self.encoder_embed_dims,
            num_heads=8,
            pe_dim=128
        )

        # 解码器（FPN式上采样 + 跳跃连接）
        self.decoder = ChangeDecoder(args=args, in_dim=self.encoder_embed_dims, has_sigmoid=True)

        # 权重初始化
        weight_init(self.attention_modules)
        weight_init(self.aspp_modules)
        weight_init(self.decoder)


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        完整的前向传播（直接调用底层模块）
        
        入参:
        - x (torch.Tensor): 第一时相图像，形状 [B, 3, H, W]
        - y (torch.Tensor): 第二时相图像，形状 [B, 3, H, W]
        
        方法:
        1. 编码器提取多尺度5D特征（已包含余弦相似度增强）
        2. 对每个尺度应用双路径注意力增强
        3. 对每个尺度应用ShuffleASPP 3D时空特征提取
        4. 为每个尺度的P帧生成位置编码
        5. 时空融合模块进行分层交叉注意力
        6. 解码器生成最终变化检测图
        
        出参:
        - prediction (torch.Tensor): 变化概率图，形状 [B, 1, H, W]，值域 [0, 1]
        """
        # --- 阶段 1: 编码器提取特征（含余弦相似度增强） ---
        # 输出: 4个5D特征张量 [
        #   [B, 24, 3, 128, 128],
        #   [B, 48, 3, 64, 64],
        #   [B, 96, 3, 32, 32],
        #   [B, 192, 3, 16, 16]
        # ]
        # 时间维度T=3对应 [T1, P, T2]
        stage_features_5d = self.encoder(x, y)

        # --- 阶段 2: 注意力增强 + ASPP增强（直接实现TFFM逻辑） ---
        enhanced_features_5d = []
        for i, feature_5d in enumerate(stage_features_5d):
            # 验证时间维度
            if feature_5d.size(2) != 3:
                raise ValueError(f"时间维度必须为3，但在尺度{i}得到 {feature_5d.size(2)}")
            
            # 步骤2.1: 分解三帧
            t1 = feature_5d[:, :, 0, :, :]  # [B, C, H, W]
            p = feature_5d[:, :, 1, :, :]   # [B, C, H, W]
            t2 = feature_5d[:, :, 2, :, :]  # [B, C, H, W]
            
            # 步骤2.2: 从P帧生成注意力图（通道注意力 + 空间注意力）
            attention_map = self.attention_modules[i](p)  # [B, C, H, W] 或 [B, 1, H, W]
            
            # 步骤2.3: 使用注意力图增强三帧（残差连接形式）
            t1_enhanced = attention_map * t1 + t1  # 注意力加权 + 残差
            p_enhanced = attention_map * p + p
            t2_enhanced = attention_map * t2 + t2
            
            # 步骤2.4: 重构注意力增强后的5D特征
            attention_enhanced = torch.stack([t1_enhanced, p_enhanced, t2_enhanced], dim=2)
            
            # 步骤2.5: 使用ShuffleASPP进行多尺度时空特征提取
            aspp_enhanced = self.aspp_modules[i](attention_enhanced)
            
            enhanced_features_5d.append(aspp_enhanced)

        # --- 阶段 3: 为每个尺度生成位置编码 ---
        # 从可学习的感知帧(P帧,索引为1)中提取位置编码
        # P帧是编码器学习到的"变化探测器"，最适合用于生成位置编码
        pe_list = [self.pos_encoder(f[:, :, 1, :, :]) for f in enhanced_features_5d]

        # --- 阶段 4: 分层时空融合 ---
        # 使用异构特征源的交叉注意力进行多尺度特征交互
        # Q源: P帧 + 位置编码（模型直觉）
        # K源: cat(T1, T2)（时序上下文）
        # V源: 余弦相似度变化权重（与编码器一致的变化度量）
        # 输出: 4个2D融合特征 [
        #   [B, 24, 128, 128],
        #   [B, 48, 64, 64],
        #   [B, 96, 32, 32],
        #   [B, 192, 16, 16]
        # ]
        fused_features = self.st_fusion(enhanced_features_5d, pe_list)
        
        # --- 阶段 5: FPN解码器生成变化图 ---
        # 逐级上采样 + 跳跃连接 → 最终输出 [B, 1, 256, 256]
        prediction = self.decoder(fused_features)

        return prediction

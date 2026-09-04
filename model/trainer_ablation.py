# Copyright (c) Duowang Zhu.
# All rights reserved.

"""
消融实验专用Trainer模块

该模块通过配置开关灵活控制各个增强模块的启用/禁用，用于系统化的消融实验。

支持的模块组合（与 model/ablation_configs.py 中的 ABLATION_CONFIGS 一致）：
1. X3D: Encoder → Decoder
2. X3D + Attention: Encoder → Attention → Decoder
3. X3D + ASPP: Encoder → ASPP → Decoder
4. X3D + Transformer: Encoder → Transformer → Decoder
5. X3D + Attention + ASPP: Encoder → Attention → ASPP → Decoder
6. X3D + ASPP + Transformer: Encoder → ASPP → Transformer → Decoder
7. Full: Encoder → Attention → ASPP → Transformer → Decoder

配置参数：
- args.use_attention: 是否启用双路径注意力模块
- args.use_aspp: 是否启用ShuffleASPP模块
- args.use_transformer: 是否启用分层时空融合模块
"""

from typing import Any

import torch
import torch.nn as nn

from model.ablation_configs import ABLATION_CONFIGS
from model.utils import weight_init
from model.attention import DualPathAttentionModule
from model.cascade_dcn import ShuffleASPP3DModule, ShuffleASPP3DModuleV2
from model.ST import HierarchicalCrossAttentionModule
from model.change_decoder import ChangeDecoder
from model.position_encoding import PositionEmbeddingSine
from model.trainer import Encoder


class TrainerAblation(nn.Module):
    """
    消融实验专用训练器
    
    通过配置开关灵活控制各增强模块的启用/禁用，支持以下7种组合
    （与 model/ablation_configs.py 中的 ABLATION_CONFIGS 一致）：
    1. X3D
    2. X3D + Attention
    3. X3D + ASPP
    4. X3D + Transformer
    5. X3D + Attention + ASPP
    6. X3D + ASPP + Transformer
    7. Full (Attention + ASPP + Transformer)
    
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
            self.attention_modules = nn.ModuleList()
            print("[消融实验] 禁用双路径注意力模块")

        # === 可选模块2: ShuffleASPP ===
        # config_5 (X3D+A+B) 使用 V2 版本（拼接融合层 + 恒等残差），其他配置使用原版
        if self.use_aspp:
            is_config5 = self.use_attention and not self.use_transformer
            aspp_class = ShuffleASPP3DModuleV2 if is_config5 else ShuffleASPP3DModule
            aspp_tag = 'V2' if is_config5 else ''
            self.aspp_modules = nn.ModuleList([
                aspp_class(channels=dim) for dim in self.encoder_embed_dims
            ])
            print("[消融实验] 启用ShuffleASPP%s模块" % aspp_tag)
        else:
            self.aspp_modules = nn.ModuleList()
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
        - X3D only: Encoder → Decoder
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
        # 解码器是2D网络，只接受4D特征 [B,C,H,W]。
        # - 启用Transformer时：阶段4已把5D三帧融合成4D（增强后的P帧），可直接送入。
        # - 禁用Transformer时：特征仍是5D [B,C,3,H,W]，需取P帧(索引1)降成4D。
        if not self.use_transformer:
            stage_features_5d = [f[:, :, 1, :, :] for f in stage_features_5d]
        prediction = self.decoder(stage_features_5d)

        return prediction


# === 便捷函数：创建特定配置的消融实验模型 ===

def create_ablation_model(args: Any, experiment_id: int) -> TrainerAblation:
    """
    根据实验ID创建对应配置的消融实验模型
    
    入参:
    - args: 基础配置参数
    - experiment_id (int): 实验编号 (1-8)
    
    方法:
    - 根据实验ID设置对应的模块开关
    - 创建并返回TrainerAblation实例
    
    出参:
    - model (TrainerAblation): 配置好的消融实验模型
    
    实验配置映射:
    1: X3D (无增强模块)
    2: X3D + Attention
    3: X3D + ASPP
    4: X3D + Transformer
    5: X3D + Attention + ASPP
    6: X3D + ASPP + Transformer
    7: Full (Attention + ASPP + Transformer)
    8: X3D + Attention + Transformer
    """
    if experiment_id not in ABLATION_CONFIGS:
        raise ValueError(f"无效的实验ID: {experiment_id}，必须在1-8之间")

    config = ABLATION_CONFIGS[experiment_id]

    # 设置配置开关
    for key in ('use_attention', 'use_aspp', 'use_transformer'):
        setattr(args, key, config[key])

    # 打印实验配置
    print(f"\n{'='*60}")
    print(f"创建消融实验模型 - 实验{experiment_id}: {config['name']}")
    print(f"{'='*60}\n")
    
    # 创建模型
    model = TrainerAblation(args)
    
    return model


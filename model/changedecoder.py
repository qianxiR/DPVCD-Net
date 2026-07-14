# Copyright (c) Duowang Zhu.
# All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class ChangeDecoder(nn.Module):
    """
    解码器网络，上采样特征图并生成最终预测

    核心功能：
    - 接收2D多尺度特征 (B, C, H, W)
    - 通过渐进式上采样和跳跃连接生成高分辨率预测

    入参:
    - args: 包含配置参数的对象（如num_class）
    - in_dim (List[int]): 编码器各阶段的通道维度列表，默认 [24, 48, 96, 192]
    - has_sigmoid (bool): 是否在输出层应用sigmoid激活，默认False

    方法:
    - 渐进上采样：从低分辨率(c4)逐步上采样到高分辨率(c1)
    - 跳跃连接：融合不同尺度的特征以保留细节

    出参:
    - pred (torch.Tensor): 预测结果，形状 (B, num_class, 256, 256)
    """
    
    def __init__(self, args, in_dim: List[int] = [24, 48, 96, 192], has_sigmoid: bool = False) -> None:
        """
        初始化解码器网络
        
        入参:
        - args: 配置对象，包含 num_class 等参数
        - in_dim (List[int]): 各尺度的输入通道数 [c1, c2, c3, c4]
        - has_sigmoid (bool): 是否使用sigmoid激活（二分类时为True）
        
        方法:
        - 构建多尺度上采样模块：使用ConvTranspose2d实现2倍上采样
        - 每个上采样模块包含：1x1卷积降维 + 4x4转置卷积上采样
        - 最终预测层：3x3卷积生成类别预测图
        
        出参:
        - None (初始化方法)
        """
        super().__init__()
        self.has_sigmoid = has_sigmoid
        
        # 提取各尺度的通道维度
        c1_channel, c2_channel, c3_channel, c4_channel = in_dim
        
        # c4上采样模块：192 -> 96通道，尺寸x2
        # 使用1x1卷积降维后，用4x4转置卷积进行2倍上采样
        self.up_c4 = nn.Sequential(
            nn.Conv2d(c4_channel, c3_channel, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c3_channel, c3_channel, kernel_size=4, stride=2, padding=1)
        )
        
        # c3上采样模块：96 -> 48通道，尺寸x2
        self.up_c3 = nn.Sequential(
            nn.Conv2d(c3_channel, c2_channel, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c2_channel, c2_channel, kernel_size=4, stride=2, padding=1)
        )
        
        # c2上采样模块：48 -> 24通道，尺寸x2
        self.up_c2 = nn.Sequential(
            nn.Conv2d(c2_channel, c1_channel, kernel_size=1, bias=False),
            nn.ConvTranspose2d(c1_channel, c1_channel, kernel_size=4, stride=2, padding=1)
        )
        
        # stem上采样层：从128尺寸上采样到256尺寸
        # 使用4x4转置卷积实现最后一次2倍上采样
        stem_channels = 24
        self.stem_upsample = nn.ConvTranspose2d(stem_channels, stem_channels, kernel_size=4, stride=2, padding=1)
        
        # 最终预测层：生成类别预测图
        # 根据has_sigmoid决定输出通道数（二分类为1，多分类为num_class）
        if self.has_sigmoid:
            num_class = 1
        else:
            num_class = args.num_class
            
        self.final_pred = nn.Conv2d(stem_channels, num_class, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, f_5d: List[torch.Tensor]) -> torch.Tensor:
        """
        解码器前向传播
        
        入参:
        - f_5d (List[torch.Tensor]): 多尺度5D时序特征列表
          每个元素形状 (B, C, 3, H, W)，其中T=3包含 [T1, P, T2]
          列表包含4个尺度：[c1, c2, c3, c4]，通道数为 [24, 48, 96, 192]
        
        方法:
        1. 时序特征融合：从5D特征提取T1、P、T2三帧
           - T1: 过去帧（索引0）
           - P: 当前帧（索引1）
           - T2: 未来帧（索引2）
        2. 计算变化增强特征：P + abs(T1 - T2)
           - P保留当前帧的空间信息
           - abs(T1 - T2)捕获双向时序变化强度
           - 相加融合静态信息和动态变化
        3. 渐进式上采样：通过跳跃连接逐层融合特征
           - c4 -> c3 -> c2 -> c1：从低分辨率到高分辨率
           - 每层使用残差连接融合当前层和上采样的深层特征
        4. 最终预测：生成高分辨率变化检测图
        
        出参:
        - pred (torch.Tensor): 变化检测预测结果，形状 (B, num_class, 256, 256)
          如果has_sigmoid=True，输出范围为[0,1]
        """
        # 步骤1: 将5D时序特征转换为4D空间特征
        # 对每个尺度，提取T1、P、T2并融合为单一特征图
        f_4d = []
        for feat_5d in f_5d:
            # 从5D张量中分解时序帧
            # feat_5d: (B, C, 3, H, W)
            t1 = feat_5d[:, :, 0, :, :]  # 过去帧: (B, C, H, W)
            p = feat_5d[:, :, 1, :, :]   # 当前帧: (B, C, H, W)
            t2 = feat_5d[:, :, 2, :, :]  # 未来帧: (B, C, H, W)
            
            # 计算时序变化增强特征
            # abs(T1 - T2): 捕获双向变化的强度（不考虑方向）
            # 这对于变化检测任务很重要，因为无论是增加还是减少都是"变化"
            temporal_change = torch.abs(t1 - t2)
            
            # 融合策略：原始P帧 + 时序变化
            # - P提供当前时刻的空间语义信息
            # - temporal_change提供时序上的动态变化信息
            # - 相加实现静态与动态信息的互补融合
            change_feat = p + temporal_change
            
            f_4d.append(change_feat)
        
        # 步骤2: 解包多尺度特征
        # c1: 高分辨率浅层特征 (B, 24, 128, 128)
        # c2: 中分辨率特征 (B, 48, 64, 64)
        # c3: 中分辨率特征 (B, 96, 32, 32)
        # c4: 低分辨率深层特征 (B, 192, 16, 16)
        c1, c2, c3, c4 = f_4d
        
        # 步骤3: 渐进式上采样与特征融合
        # 从最深层(c4)开始，逐步上采样并与对应尺度特征融合
        
        # 第一步：c4上采样并与c3融合
        # up_c4(c4): (B, 192, 16, 16) -> (B, 96, 32, 32)
        # c3f保留c3的细节并融入c4的高层语义
        c3f = c3 + self.up_c4(c4)
        
        # 第二步：c3f上采样并与c2融合
        # up_c3(c3f): (B, 96, 32, 32) -> (B, 48, 64, 64)
        c2f = c2 + self.up_c3(c3f)
        
        # 第三步：c2f上采样并与c1融合
        # up_c2(c2f): (B, 48, 64, 64) -> (B, 24, 128, 128)
        # c1f是融合了所有层级信息的高分辨率特征
        c1f = c1 + self.up_c2(c2f)
        
        # 步骤4: 上采样到目标分辨率
        # stem_upsample: (B, 24, 128, 128) -> (B, 24, 256, 256)
        # 恢复到输入图像的原始分辨率
        c1f_upsampled = self.stem_upsample(c1f)
        
        # 步骤5: 生成最终预测
        # final_pred: (B, 24, 256, 256) -> (B, num_class, 256, 256)
        # 通过3x3卷积将特征映射到类别空间
        pred = self.final_pred(c1f_upsampled)
        
        # 步骤6: 应用sigmoid激活（如果需要）
        # 对于二分类任务，sigmoid将logits转换为[0,1]范围的概率
        if self.has_sigmoid:
            pred = torch.sigmoid(pred)
        
        return pred
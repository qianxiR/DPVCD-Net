import torch
import torch.nn as nn
import random

#ShuffleNet
def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    对张量的通道维度进行混洗
    Args:
        x (torch.Tensor): 输入张量, shape (B, C, T, H, W)
        groups (int): 要划分的组数
    Returns:
        torch.Tensor: 通道混洗后的张量
    """
    batchsize, num_channels, T, height, width = x.data.size()
    channels_per_group = num_channels // groups
    
    # 重塑 (reshape)
    x = x.view(batchsize, groups, 
               channels_per_group, T, height, width)
    
    # 转置 (transpose)
    x = torch.transpose(x, 1, 2).contiguous()
    
    # 扁平化 (flatten)
    x = x.view(batchsize, -1, T, height, width)
    
    return x

class DepthwiseSeparableConv3d(nn.Module):
    """
    3D深度可分离卷积模块.
    它将一个标准的3D卷积分解为两步:
    1. 深度卷积 (Depthwise): 对每个输入通道独立进行空间卷积。
    2. 逐点卷积 (Pointwise): 使用1x1x1卷积来混合通道。
    这可以显著减少参数量和计算成本。
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False, dilation=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        # 自动计算padding以保持尺寸不变
        # 正确的 'same' padding 计算公式应该是: padding = dilation * (kernel_size - 1) // 2
        if isinstance(dilation, tuple):
            padding = tuple(d * (k - 1) // 2 for k, d in zip(kernel_size, dilation))
        else:
            padding = tuple(dilation * (k - 1) // 2 for k in kernel_size)

        self.depthwise = nn.Conv3d(
            in_channels, 
            in_channels, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=in_channels, 
            bias=bias,
            dilation=dilation
        )
        self.pointwise = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            bias=bias
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class AtrousSeparableConv3dUnit(nn.Module):
    """ 3D膨胀可分离卷积基础单元 (Conv + BN + ReLU) """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, bias=False):
        super(AtrousSeparableConv3dUnit, self).__init__()
        self.conv = DepthwiseSeparableConv3d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            dilation=dilation,
            bias=bias
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ShuffleASPP3DModule(nn.Module):
    """
    3D Shuffle-ASPP模块 (V3 - 多分支分组特征提取).
    
    入参:
    - channels (int): 输入特征的通道数，必须能被4整除
    
    方法:
    1. 多分支分组处理：将输入分为4组，每组经过4个并行分支（1×1×1卷积、3×3×3卷积、最大池化、平均池化）
    2. ASPP金字塔：4个不同膨胀率(1,3,5,7)的并行膨胀卷积
    3. 通道混洗：打破组间隔离效应
    4. 融合与残差连接：输出增强特征
    
    出参:
    - 增强后的5D特征，形状 [B, C, T, H, W]
    """
    def __init__(self, channels: int):
        super(ShuffleASPP3DModule, self).__init__()
        self.groups = 4
        assert channels % self.groups == 0, "通道数必须能被组数整除"
        grouped_channels = channels // self.groups
        
        # --- 多分支分组处理层：4个组对应4个不同分支 ---
        # 组1: 1×1×1卷积分支
        self.branch1_conv1x1 = AtrousSeparableConv3dUnit(grouped_channels, grouped_channels, (1, 1, 1), dilation=1)
        # 组2: 3×3×3卷积分支
        self.branch2_conv3x3 = AtrousSeparableConv3dUnit(grouped_channels, grouped_channels, (3, 3, 3), dilation=1)
        # 组3: 最大池化分支
        self.branch3_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        # 组4: 平均池化分支
        self.branch4_avgpool = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

        # --- ASPP层 (作用于完整特征图) ---
        # 4个不同膨胀率的并行膨胀卷积
        self.aspp_branch1 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=1)
        self.aspp_branch2 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=3)
        self.aspp_branch3 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=5)
        self.aspp_branch4 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=7)

        # --- 融合层 ---
        # 通道混洗后的融合卷积
        self.fusion_conv = nn.Conv3d(channels * 4, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        # 残差连接的1×1×1卷积
        self.conv_residual = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        """
        前向传播：多分支分组处理 → ASPP金字塔 → 通道混洗 → 融合
        
        入参:
        - x_bcthw (torch.Tensor): 输入5D特征，形状 [B, C, T, H, W]
        
        方法:
        1. 将输入分为4组，每组C/4通道
        2. 每组分别经过4个分支（1×1×1, 3×3×3, MaxPool, AvgPool）
        3. 跨组拼接所有16个子特征（4组×4分支）
        4. 并行ASPP金字塔（膨胀率1,3,5,7）
        5. 通道混洗打破隔离
        6. 融合卷积 + 残差连接
        
        出参:
        - F_out (torch.Tensor): 增强后的5D特征，形状 [B, C, T, H, W]
        """
        # 步骤1: 将输入F在通道维度划分为4组
        # F = [F^(1), F^(2), F^(3), F^(4)], 每组 F^(g) ∈ R^(C/4×T×H×W)
        x_groups = x_bcthw.chunk(self.groups, dim=1)
        
        # 步骤2: 每个组进入不同的分支处理
        # 组1 → 1×1×1卷积
        y1 = self.branch1_conv1x1(x_groups[0])  # [B, C/4, T, H, W]
        
        # 组2 → 3×3×3卷积
        y2 = self.branch2_conv3x3(x_groups[1])  # [B, C/4, T, H, W]
        
        # 组3 → 最大池化
        y3 = self.branch3_maxpool(x_groups[2])  # [B, C/4, T, H, W]
        
        # 组4 → 平均池化
        y4 = self.branch4_avgpool(x_groups[3])  # [B, C/4, T, H, W]
        
        # 步骤3: 将4个分支输出收集到列表并随机打乱顺序
        branch_outputs = [y1, y2, y3, y4]
        
        # 关键步骤：在训练模式下随机打乱分支顺序，增加特征多样性
        # 在推理模式下保持固定顺序，确保结果可复现
        #if self.training:
           # random.shuffle(branch_outputs)
        
        # 步骤4: 拼接所有分支输出（随机打乱后）
        # Z = Concat_c {Y_1, Y_2, Y_3, Y_4} (顺序已随机打乱)
        # 4个分支，每个C/4通道 → 拼接后恢复为C通道
        Z = torch.cat(branch_outputs, dim=1)  # [B, C, T, H, W]
        
        # 步骤5: 并行ASPP金字塔（4个不同膨胀率的空洞卷积）
        # U_d = Conv_{3×3×3,d}^{dilated}(Z), d ∈ {1,3,5,7}
        U_1 = self.aspp_branch1(Z)  # [B, C, T, H, W]
        U_3 = self.aspp_branch2(Z)  # [B, C, T, H, W]
        U_5 = self.aspp_branch3(Z)  # [B, C, T, H, W]
        U_7 = self.aspp_branch4(Z)  # [B, C, T, H, W]

        # 步骤6: 拼接ASPP输出
        # V = Concat_c {U_1, U_3, U_5, U_7}
        V = torch.cat([U_1, U_3, U_5, U_7], dim=1)  # [B, 4C, T, H, W]
        
        # 步骤7: 通道混洗打破组间卷积的隔离效应
        # V_sh = Shuffle(V)
        V_sh = channel_shuffle(V, self.groups)  # [B, 4C, T, H, W]
        
        # 步骤8: 融合卷积进行特征压缩与聚合
        # V_f = Conv_{1×1×1}(V_sh)
        V_f = self.fusion_conv(V_sh)  # [B, C, T, H, W]
        
        # 步骤9: 残差连接（保持信息流通与梯度稳定性）
        # F_out = F + Conv_{1×1×1}(V_f)
        F_residual = self.conv_residual(x_bcthw)  # [B, C, T, H, W]
        F_out = self.relu(V_f + F_residual)  # [B, C, T, H, W]
        
        return F_out 
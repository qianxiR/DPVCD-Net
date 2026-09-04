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
    3D Shuffle-ASPP模块（原版 - 投影残差，无拼接融合层）。
    用于 X3D+B、X3D+B+C、Full 等配置。

    入参:
    - channels (int): 输入特征的通道数，必须能被4整除

    方法:
    1. 多分支分组处理：将输入按通道分为4组，4个组分别经过4个并行分支
    2. ASPP金字塔：4个不同膨胀率(1,3,5,7)的并行膨胀卷积
    3. 通道混洗：打破组间隔离效应
    4. 融合卷积 + 投影残差连接：输出增强特征

    出参:
    - 增强后的5D特征，形状 [B, C, T, H, W]
    """
    def __init__(self, channels: int):
        super(ShuffleASPP3DModule, self).__init__()
        self.groups = 4
        assert channels % self.groups == 0, "通道数必须能被组数整除"
        grouped_channels = channels // self.groups

        # --- 多分支分组处理层：4个组对应4个不同分支 ---
        self.branch1_conv1x1 = AtrousSeparableConv3dUnit(grouped_channels, grouped_channels, (1, 1, 1), dilation=1)
        self.branch2_conv3x3 = AtrousSeparableConv3dUnit(grouped_channels, grouped_channels, (3, 3, 3), dilation=1)
        self.branch3_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.branch4_avgpool = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

        # --- ASPP层 (作用于完整特征图) ---
        self.aspp_branch1 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=1)
        self.aspp_branch2 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=3)
        self.aspp_branch3 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=5)
        self.aspp_branch4 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=7)

        # --- 融合层 ---
        self.fusion_conv = nn.Conv3d(channels * 4, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv_residual = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        """
        入参: x_bcthw [B, C, T, H, W]
        出参: F_out [B, C, T, H, W]
        """
        x_groups = x_bcthw.chunk(self.groups, dim=1)

        y1 = self.branch1_conv1x1(x_groups[0])
        y2 = self.branch2_conv3x3(x_groups[1])
        y3 = self.branch3_maxpool(x_groups[2])
        y4 = self.branch4_avgpool(x_groups[3])

        Z = torch.cat([y1, y2, y3, y4], dim=1)

        U_1 = self.aspp_branch1(Z)
        U_3 = self.aspp_branch2(Z)
        U_5 = self.aspp_branch3(Z)
        U_7 = self.aspp_branch4(Z)

        V = torch.cat([U_1, U_3, U_5, U_7], dim=1)
        V_sh = channel_shuffle(V, self.groups)

        V_f = self.fusion_conv(V_sh)
        F_residual = self.conv_residual(x_bcthw)
        F_out = self.relu(V_f + F_residual)

        return F_out


class ShuffleASPP3DModuleV2(nn.Module):
    """
    3D Shuffle-ASPP模块 V2（拼接融合层 + 恒等残差）。
    仅用于 X3D+A+B（config_5）配置。

    与原版的区别:
    1. 拼接后新增 concat_fusion（Conv1x1+BN+ReLU），混合四个C/4分块
    2. 残差从投影残差（conv_residual）改为恒等残差（直接加原始输入）

    入参:
    - channels (int): 输入特征的通道数，必须能被4整除

    方法:
    1. 多分支分组处理：将输入按通道分为4组，4个组分别经过4个并行分支
    2. 拼接融合：1×1×1卷积+BN+ReLU 混合四个分块
    3. ASPP金字塔：4个不同膨胀率(1,3,5,7)的并行膨胀卷积
    4. 通道混洗：打破组间隔离效应
    5. 融合卷积 + 恒等残差：输出增强特征

    出参:
    - 增强后的5D特征，形状 [B, C, T, H, W]
    """
    def __init__(self, channels: int):
        super(ShuffleASPP3DModuleV2, self).__init__()
        self.groups = 4
        assert channels % self.groups == 0, "通道数必须能被组数整除"
        grouped_channels = channels // self.groups

        # --- 多分支分组处理层 ---
        self.branch1_conv1x1 = AtrousSeparableConv3dUnit(grouped_channels, grouped_channels, (1, 1, 1), dilation=1)
        self.branch2_conv3x3 = AtrousSeparableConv3dUnit(grouped_channels, grouped_channels, (3, 3, 3), dilation=1)
        self.branch3_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))
        self.branch4_avgpool = nn.AvgPool3d(kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1))

        # --- 拼接融合层：1×1×1卷积混合四个分支输出，打破通道分块隔离 ---
        self.concat_fusion = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
        )

        # --- ASPP层 ---
        self.aspp_branch1 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=1)
        self.aspp_branch2 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=3)
        self.aspp_branch3 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=5)
        self.aspp_branch4 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=7)

        # --- 融合层 ---
        self.fusion_conv = nn.Conv3d(channels * 4, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        """
        入参: x_bcthw [B, C, T, H, W]
        出参: F_out [B, C, T, H, W]
        """
        x_groups = x_bcthw.chunk(self.groups, dim=1)

        y1 = self.branch1_conv1x1(x_groups[0])
        y2 = self.branch2_conv3x3(x_groups[1])
        y3 = self.branch3_maxpool(x_groups[2])
        y4 = self.branch4_avgpool(x_groups[3])

        Z = torch.cat([y1, y2, y3, y4], dim=1)
        Z = self.concat_fusion(Z)

        U_1 = self.aspp_branch1(Z)
        U_3 = self.aspp_branch2(Z)
        U_5 = self.aspp_branch3(Z)
        U_7 = self.aspp_branch4(Z)

        V = torch.cat([U_1, U_3, U_5, U_7], dim=1)
        V_sh = channel_shuffle(V, self.groups)

        V_f = self.fusion_conv(V_sh)
        F_out = self.relu(V_f + x_bcthw)

        return F_out 
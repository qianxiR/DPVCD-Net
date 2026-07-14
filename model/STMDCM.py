import torch
import torch.nn as nn



def spatiotemporal_channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    时序通道混洗：在T1/P/T2帧之间交换通道信息
    输入: (B, C, T, H, W), T=3
    输出: (B, C, T, H, W)
    """
    batchsize, num_channels, T, height, width = x.data.size()
    assert T == 3
    assert num_channels % groups == 0

    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, T, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = torch.transpose(x, 2, 3).contiguous()
    x = x.view(batchsize, channels_per_group * groups, T, height, width)
    return x


class DepthwiseSeparableConv3d(nn.Module):
    """3D深度可分离卷积"""

    def __init__(self, in_channels, out_channels, kernel_size, padding=0, bias=False, dilation=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        if isinstance(dilation, tuple):
            padding = tuple(d * (k - 1) // 2 for k, d in zip(kernel_size, dilation))
        else:
            padding = tuple(dilation * (k - 1) // 2 for k in kernel_size)

        self.depthwise = nn.Conv3d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=bias, dilation=dilation
        )
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class AtrousSeparableConv3dUnit(nn.Module):
    """3D膨胀可分离卷积单元"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, bias=False):
        super(AtrousSeparableConv3dUnit, self).__init__()
        self.conv = DepthwiseSeparableConv3d(
            in_channels, out_channels, kernel_size=kernel_size,
            dilation=dilation, bias=bias
        )
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ShuffleASPP3DModule(nn.Module):
    """
    ST-MDCM模块：多膨胀率空洞卷积 + 时序通道混洗

    - 4个ASPP分支：dilation=(1,3,5,7)，时间维度kernel=3捕获时序变化
    - 时序通道混洗：在T1/P/T2帧间交换通道信息
    - 残差融合：保持信息流通与梯度稳定性
    """

    def __init__(self, channels: int):
        super(ShuffleASPP3DModule, self).__init__()
        self.groups = 4
        assert channels % self.groups == 0

        # 4个膨胀率分支：时间dilation=1，空间dilation∈{1,3,5,7}
        self.aspp_branch1 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=(1, 1, 1))
        self.aspp_branch2 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=(1, 3, 3))
        self.aspp_branch3 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=(1, 5, 5))
        self.aspp_branch4 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=(1, 7, 7))

        # 融合卷积与残差
        self.fusion_conv = nn.Conv3d(channels * 4, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.conv_residual = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入5D特征 (B, C, T, H, W), T=3包含 [T1, P, T2]
        Returns:
            增强后的5D特征 (B, C, T, H, W)
        """
        # 并行ASPP金字塔
        U_1 = self.aspp_branch1(x)
        U_3 = self.aspp_branch2(x)
        U_5 = self.aspp_branch3(x)
        U_7 = self.aspp_branch4(x)

        # 拼接
        V = torch.cat([U_1, U_3, U_5, U_7], dim=1)

        # 时序通道混洗
        V_sh = spatiotemporal_channel_shuffle(V, self.groups)

        # 融合
        V_f = self.fusion_conv(V_sh)
        F_residual = self.conv_residual(x)
        F_out = self.relu(V_f + F_residual)

        return F_out

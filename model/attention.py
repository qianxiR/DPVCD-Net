import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    经典的通道注意力模块 (CBAM实现).
    它通过一个共享的MLP来处理全局平均池化和最大池化的特征，以生成通道注意力图。
    """
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 共享的MLP
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    """
    经典的空间注意力模块 (CBAM实现).
    它通过一个卷积层来处理沿通道池化后的特征，以生成空间注意力图。
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class DualPathAttentionModule(nn.Module):
    """
    双路径注意力模块 (CBAM实现).

    该模块通过并行和串行两种方式结合通道注意力和空间注意力，
    并使用可学习的权重来融合两种路径的结果，以增强特征表示。
    """
    def __init__(self, channels: int):
        super(DualPathAttentionModule, self).__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()
        # 引入两个可学习的参数来平衡两条路径的贡献，初始化为1
        self.path_weights = nn.Parameter(torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 并行路径: 通道注意力与空间注意力并行计算后相乘
        channel_map_p = self.ca(x)
        spatial_map_p = self.sa(x)
        attention_map_p = channel_map_p * spatial_map_p
        
        # 串行路径: 先应用通道注意力，再计算空间注意力
        x_ca_s = x * channel_map_p
        attention_map_s = self.sa(x_ca_s)
        
        # 融合: 使用可学习的权重来加权两条路径的输出
        final_attention_map = self.path_weights[0] * attention_map_p + self.path_weights[1] * attention_map_s
        
        return final_attention_map 
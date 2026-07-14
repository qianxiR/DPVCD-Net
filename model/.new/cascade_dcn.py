import torch
import torch.nn as nn

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


def spatiotemporal_channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """
    基于时间维度的ShuffleNet式通道混洗（ST-MDCM组件）
    
    将ShuffleNet的通道混洗机制扩展到时间维度，实现跨时间帧的通道交互。
    通过将通道分组和时间帧结合，让不同时间帧的通道信息相互渗透。
    
    入参:
    - x (torch.Tensor): 输入张量, shape (B, 4C, T, H, W)，其中T=3（T1, P, T2）
    - groups (int): 通道分组数，用于ShuffleNet式混洗（例如4组对应U1,U3,U5,U7）
    
    方法:
    1. 将4C个通道分成groups组，每组channels_per_group = 4C/groups个通道
    2. 重塑为 [B, groups, channels_per_group, T, H, W]
    3. 转置通道组维度: [B, channels_per_group, groups, T, H, W] (打破ASPP分支隔离)
    4. 转置通道组和时间维度: [B, channels_per_group, T, groups, H, W] (跨时间帧交互)
    5. 扁平化通道和时间维度: [B, 4C, T, H, W]
    
    效果:
    - 打破ASPP分支（U1, U3, U5, U7）间的通道隔离
    - 实现跨时间帧的通道混洗，让T1、P、T2的通道信息相互渗透
    - 保持时间顺序T1→P→T2不变
    
    出参:
    - V_st (torch.Tensor): 时序通道混合后的张量, shape (B, 4C, T, H, W)
    """
    batchsize, num_channels, T, height, width = x.data.size()
    assert T == 3, "时间维度必须为3（T1, P, T2）"
    assert num_channels % groups == 0, f"通道数{num_channels}必须能被组数{groups}整除"
    
    channels_per_group = num_channels // groups
    
    # 步骤1: 重塑为组结构
    # [B, 4C, T, H, W] -> [B, groups, channels_per_group, T, H, W]
    # 例如: [B, 192, 3, H, W] -> [B, 4, 48, 3, H, W]
    # 组0: C0-C47 (U1), 组1: C48-C95 (U3), 组2: C96-C143 (U5), 组3: C144-C191 (U7)
    x = x.view(batchsize, groups, channels_per_group, T, height, width)
    
    # 步骤2: 转置通道组维度（打破ASPP分支隔离）
    # [B, groups, channels_per_group, T, H, W] -> [B, channels_per_group, groups, T, H, W]
    # 例如: [B, 4, 48, 3, H, W] -> [B, 48, 4, 3, H, W]
    # 这样每个通道索引(0-47)现在对应不同的ASPP分支组
    x = torch.transpose(x, 1, 2).contiguous()
    
    # 步骤3: 转置通道组和时间维度（实现跨时间帧交互）
    # [B, channels_per_group, groups, T, H, W] -> [B, channels_per_group, T, groups, H, W]
    # 例如: [B, 48, 4, 3, H, W] -> [B, 48, 3, 4, H, W]
    # 关键：将groups和T维度交换，让不同时间帧的通道组信息相互渗透
    x = torch.transpose(x, 2, 3).contiguous()
    
    # 步骤4: 扁平化通道组和时间维度
    # [B, channels_per_group, T, groups, H, W] -> [B, channels_per_group * groups, T, H, W]
    # 例如: [B, 48, 3, 4, H, W] -> [B, 192, 3, H, W]
    # 结果：不同ASPP分支的特征交错排列，同时实现了跨时间帧的通道混洗
    # 混洗后的结构：每个通道索引在不同时间帧间混洗了不同ASPP分支的信息
    x = x.view(batchsize, channels_per_group * groups, T, height, width)
    
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
    ST-MDCM模块：时空多膨胀率空洞卷积模块（Spatio-Temporal Multi-Dilation Convolution Module）
    
    核心思想：滑坡与建筑物在尺度、结构与变化模式上具有显著差异（前者表现为大尺度地貌扰动，
    后者呈现为小尺度规则几何结构）。为实现二者的统一建模，构建多膨胀率3D空洞卷积分支，
    逐层提取从局部细节到大尺度结构的多层次时空语义。
    
    入参:
    - channels (int): 输入特征的通道数，必须能被4整除（用于通道混洗）
    
    方法:
    1. 多膨胀率3D空洞卷积分支：
       d ∈ {1,3,5,7}, U_d = Conv_d(F_DP-CSA)
       时间维度：kernel_size=3, dilation=1（采样全部3帧：T1, P, T2，捕获时序变化）
       空间维度：kernel_size=3, dilation=d（多尺度空间特征，d ∈ {1,3,5,7}）
    
    2. 多尺度特征拼接：
       V = Concat(U_1, U_3, U_5, U_7)
    
    3. 时序通道混合：
       V_st = SpatiotemporalChannelShuffle(V)
       在时间维对通道进行重组与交换以增强时序一致性
    
    4. 残差融合：
       F_out = ReLU(V_st + Conv_{1×1×1}(V))
       经残差映射稳定融合得到最终输出
    
    出参:
    - F_out (torch.Tensor): 增强后的5D特征，形状 [B, C, T, H, W]
    """
    def __init__(self, channels: int):
        super(ShuffleASPP3DModule, self).__init__()
        self.groups = 4
        assert channels % self.groups == 0, "通道数必须能被组数整除（用于通道混洗）"

        # --- ASPP层 (作用于完整特征图) ---
        # 4个不同膨胀率的并行膨胀卷积
        # 时间维度：kernel=3, dilation=1（采样全部3帧：T1, P, T2，捕获时间变化）
        # 空间维度：kernel=3, dilation=d（多尺度空间特征），d ∈ {1,3,5,7}
        # 这样既能理解时间变化，又能在空间维度实现多尺度特征提取
        self.aspp_branch1 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=(1, 1, 1))
        self.aspp_branch2 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=(1, 3, 3))
        self.aspp_branch3 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=(1, 5, 5))
        self.aspp_branch4 = AtrousSeparableConv3dUnit(channels, channels, (3, 3, 3), dilation=(1, 7, 7))

        # --- 融合层 ---
        # 通道混洗后的融合卷积
        self.fusion_conv = nn.Conv3d(channels * 4, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        # 残差连接的1×1×1卷积
        self.conv_residual = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x_bcthw: torch.Tensor) -> torch.Tensor:
        """
        ST-MDCM前向传播
        
        入参:
        - x_bcthw (torch.Tensor): 输入特征F_DP-CSA，形状 [B, C, T, H, W]
        
        方法:
        1. 多膨胀率3D空洞卷积：U_d = Conv_d(F_DP-CSA)，d ∈ {1,3,5,7}
           每个分支在时间维度捕获T1→P→T2的时序变化，在空间维度提取不同尺度的特征
        
        2. 多尺度特征拼接：V = Concat(U_1, U_3, U_5, U_7)
           拼接后形状为[B, 4C, T, H, W]
        
        3. 时序通道混合：V_st = SpatiotemporalChannelShuffle(V)
           在时间维对通道进行重组与交换，增强时序一致性，缓解大膨胀率导致的采样稀疏与结构不连续
        
        4. 残差融合：F_out = ReLU(V_st + Conv_{1×1×1}(V))
           经残差映射稳定融合，保持信息流通与梯度稳定性
        
        出参:
        - F_out (torch.Tensor): 增强后的5D特征，形状 [B, C, T, H, W]
        """
        # 步骤1: 并行ASPP金字塔（时空膨胀卷积）
        # 时间维度：kernel=3, dilation=1（采样T1, P, T2三帧，捕获时间变化）
        # 空间维度：kernel=3, dilation=d（多尺度空间特征），d ∈ {1,3,5,7}
        # U_d = Conv_{(3,3,3),(1,d,d)}^{dilated}(x_bcthw)
        # 每个分支都能理解T1→P→T2的时间变化，同时在空间维度捕获不同尺度的特征
        U_1 = self.aspp_branch1(x_bcthw)  # [B, C, T, H, W] - 时间变化 + 空间dilation=1
        U_3 = self.aspp_branch2(x_bcthw)  # [B, C, T, H, W] - 时间变化 + 空间dilation=3
        U_5 = self.aspp_branch3(x_bcthw)  # [B, C, T, H, W] - 时间变化 + 空间dilation=5
        U_7 = self.aspp_branch4(x_bcthw)  # [B, C, T, H, W] - 时间变化 + 空间dilation=7

        # 步骤2: 拼接ASPP输出
        # V = Concat_c {U_1, U_3, U_5, U_7}
        V = torch.cat([U_1, U_3, U_5, U_7], dim=1)  # [B, 4C, T, H, W]
        
        # 步骤3: 时序通道混合（在3帧特征间进行通道交换）
        # V_sh = SpatiotemporalChannelShuffle(V)
        # 基于拼接后的3帧特征（T1, P, T2），在时间帧之间交换通道片段以增强时序一致性
        # T1' = [P的前s通道] + [T1的后(4C-s)通道]
        # P'  = [T1的前s通道] + [P的中间通道] + [T2的后s通道]
        # T2' = [T2的前(4C-s)通道] + [P的后s通道]
        V_sh = spatiotemporal_channel_shuffle(V, self.groups)  # [B, 4C, T, H, W]
        
        # 步骤4: 融合卷积进行特征压缩与聚合（关键：理解时间帧间的通道信息）
        # V_f = Conv_{(3,1,1)}(V_sh)
        # 融合层使用kernel_size=(3,1,1)，在时间维度进行3帧卷积
        # 对于每个空间位置(h,w)和每个输出通道c_out：
        #   - 输入: [T1的4C通道, P的4C通道, T2的4C通道]
        #   - 卷积核学习如何组合T1、P、T2三个时间帧的通道信息
        #   - 输出: 融合后的特征，理解时间帧间的通道关系
        V_f = self.fusion_conv(V_sh)  # [B, C, T, H, W]
        
        # 步骤5: 残差连接（保持信息流通与梯度稳定性）
        # F_out = ReLU(V_f + Conv_{1×1×1}(F_in))
        F_residual = self.conv_residual(x_bcthw)  # [B, C, T, H, W]
        F_out = self.relu(V_f + F_residual)  # [B, C, T, H, W]
        
        return F_out 
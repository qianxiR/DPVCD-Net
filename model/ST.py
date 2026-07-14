import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 可重用的Transformer块组件和窗口化函数 ---

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    按指定概率随机丢弃一个张量（沿着批次维度）。
    这是Stochastic Depth的核心实现。
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """
    按指定概率随机丢弃一个张量（沿着批次维度）。
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Mlp(nn.Module):
    """
    多层感知机 (MLP) 模块。
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """
    自适应多头注意力模块，内部处理不同尺寸和通道数。
    """
    def __init__(self, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # 动态创建的投影层将在 forward 中初始化
        self.q_proj = None
        self.kv_proj = None
        self.out_proj = None

    def _create_projections(self, q_dim, kv_dim, device):
        """动态创建投影层"""
        if self.q_proj is None or self.q_proj.in_features != q_dim:
            # 使用较小维度作为输出维度，确保计算效率
            output_dim = min(q_dim, kv_dim)
            # 确保可以被 num_heads 整除
            output_dim = (output_dim // self.num_heads) * self.num_heads
            
            self.q_proj = nn.Linear(q_dim, output_dim, bias=self.qkv_bias)
            self.kv_proj = nn.Linear(kv_dim, output_dim * 2, bias=self.qkv_bias)
            self.out_proj = nn.Linear(output_dim, q_dim)
            
            # 将新创建的层移到正确的设备
            self.q_proj = self.q_proj.to(device)
            self.kv_proj = self.kv_proj.to(device)
            self.out_proj = self.out_proj.to(device)
            
            return output_dim
        return self.q_proj.out_features

    def forward(self, x, kv=None, mask=None):
        B, N, C = x.shape
        kv = kv if kv is not None else x
        B_kv, N_kv, C_kv = kv.shape
        
        # 动态创建投影层
        output_dim = self._create_projections(C, C_kv, x.device)
        head_dim = output_dim // self.num_heads
        scale = head_dim ** -0.5
        
        # 投影 Q, K, V
        q = self.q_proj(x).reshape(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        kv_projected = self.kv_proj(kv).reshape(B_kv, N_kv, 2, self.num_heads, head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv_projected[0], kv_projected[1]

        # 注意力计算
        attn = (q @ k.transpose(-2, -1)) * scale
        
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 输出投影
        x = (attn @ v).transpose(1, 2).reshape(B, N, output_dim)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    """
    自适应 Transformer 块，内部处理不同通道数和尺寸。
    """
    def __init__(self, num_heads=8, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.attn = Attention(num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_ratio = mlp_ratio
        self.act_layer = act_layer
        self.drop = drop
        
        # 动态创建的归一化层和MLP
        self.norm1_q = None
        self.norm1_kv = None
        self.norm2 = None
        self.mlp = None

    def _create_layers(self, q_dim, kv_dim=None, device='cpu'):
        """动态创建归一化层和MLP"""
        
        if self.norm1_q is None or self.norm1_q.normalized_shape[0] != q_dim:
            self.norm1_q = nn.LayerNorm(q_dim).to(device)
            self.norm2 = nn.LayerNorm(q_dim).to(device)
            
            mlp_hidden_dim = int(q_dim * self.mlp_ratio)
            self.mlp = Mlp(in_features=q_dim, hidden_features=mlp_hidden_dim, 
                          act_layer=self.act_layer, drop=self.drop).to(device)
        
        if kv_dim is not None and (self.norm1_kv is None or self.norm1_kv.normalized_shape[0] != kv_dim):
            self.norm1_kv = nn.LayerNorm(kv_dim).to(device)

    def forward(self, x, x_kv=None, attn_mask=None):
        # 获取输入维度
        B, N, C = x.shape
        kv_dim = x_kv.shape[-1] if x_kv is not None else None
        
        # 动态创建必要的层
        self._create_layers(C, kv_dim, device=x.device)
        
        if x_kv is not None:
            # Cross-Attention Path
            x = x + self.drop_path(self.attn(self.norm1_q(x), kv=self.norm1_kv(x_kv), mask=attn_mask))
        else:
            # Self-Attention Path
            x = x + self.drop_path(self.attn(self.norm1_q(x), kv=x, mask=attn_mask))
            
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def window_partition(x, window_size):
    """
    将特征图划分为不重叠的窗口。
    Args:
        x (Tensor): 输入特征图，形状为 (B, H, W, C)。
        window_size (int): 窗口的边长。
    Returns:
        Tensor: 划分后的窗口，形状为 (B*num_windows, window_size*window_size, C)。
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    将窗口化的特征图还原为原始图像形状。
    Args:
        windows (Tensor): 窗口化的特征，形状为 (B*num_windows, window_size*window_size, C)。
        window_size (int): 窗口的边长。
        H (int): 原始图像的高度。
        W (int): 原始图像的宽度。
    Returns:
        Tensor: 还原后的特征图，形状为 (B, H, W, C)。
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    C = windows.shape[-1]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x

def get_optimal_window_size(H, W, min_window=4, max_window=16):
    """根据特征图尺寸自动确定最优窗口大小"""
    min_dim = min(H, W)
    # 如果特征图本身就小于最小窗口，则直接使用特征图尺寸作为窗口大小
    if min_dim < min_window:
        return min_dim

    # 在[min_window, max_window]范围内寻找最优约数
    # 我们优先选择能被H和W整除的最大窗口尺寸
    optimal_size = min_window
    for ws in range(min_window, min(min_dim, max_window) + 1):
        if H % ws == 0 and W % ws == 0:
            optimal_size = ws
    
    return optimal_size

def create_mask(H, W, window_size, shift_size, device):
    # 如果窗口大小大于等于特征图尺寸，不使用掩码
    if window_size >= min(H, W):
        return None
        
    img_mask = torch.zeros((1, H, W, 1), device=device)
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
    return attn_mask


class HierarchicalCrossAttentionModule(nn.Module):
    """
    分层交叉注意力模块（仅输出增强后的查询特征Q）

    - Q（查询）：使用P帧 + 位置编码，体现模型对当前状态的理解
    - K（键）：使用T1和T2的通道拼接特征，提供时序上下文判断依据
    - V（值）：使用时序差分 abs(T1-T2) 表示T1-T2的变化强度

    分层交叉注意力逻辑：
    - 浅层特征(c1,c2,c3)关注深层特征(c4)的异构键值对
    - 深层特征(c4)关注所有浅层特征异构键值对的拼接

    入参:
    - in_channels (List[int]): 各尺度的通道数列表
    - num_heads (int): 多头注意力的头数，默认8
    - pe_dim (int): 位置编码的维度，默认128

    方法:
    - 构建多尺度的自适应Transformer块更新查询特征Q
    - 通过差分融合层提炼增强后的Q，再与原始P帧残差连接

    出参:
    - enhanced_queries (List[torch.Tensor]): 增强后的查询特征列表，
      每个元素形状 (B, C, H, W)
    """
    def __init__(self, in_channels, num_heads=8, pe_dim=128):
        super().__init__()
        self.num_scales = len(in_channels)
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.pe_dim = pe_dim
        
        # 用于更新查询 (P-frame) 的自适应 Transformer 块
        self.query_update_blocks = nn.ModuleList([
            Block(num_heads=num_heads) for _ in range(self.num_scales)
        ])

        # 用于处理特征差分的轻量化融合层
        self.diff_fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 1),
                nn.BatchNorm2d(c),
            ) for c in in_channels
        ])
        

        # 跨尺度通道对齐层：将深层特征的通道数对齐到当前层
        # 用于浅层特征关注深层特征时的通道匹配
        # 例如：c1关注c4时，需要将c4的384通道(2×192)对齐到c1的48通道(2×24)
        self.cross_scale_channel_align = nn.ModuleList([
            nn.ModuleList([
                nn.Conv2d(in_channels[j] * 2, in_channels[i] * 2, 1) 
                if j != i else nn.Identity()
                for j in range(self.num_scales)
            ]) for i in range(self.num_scales)
        ])
        
        # V源直接使用时序差分 abs(T1 - T2) 度量变化强度，无需额外模块
        
    def forward(self, features_5d, pe_list):
        """
        使用异构特征源的自适应 Swin Transformer 进行分层交叉注意力，仅输出增强后的Q

        入参:
        - features_5d (List[torch.Tensor]): 多尺度5D特征列表，每个元素形状 (B, C, T, H, W)
        - pe_list (List[torch.Tensor]): 每个尺度对应的位置编码列表

        方法:
        1. 分解5D特征为T1、P、T2三帧
        2. 构建异构特征源的查询(Q)、键(K)、值(V)
        3. 执行分层交叉注意力，得到增强的Q
        4. 使用差分融合层提炼Q，与原始P帧残差连接

        出参:
        - enhanced_queries (List[torch.Tensor]): 增强后的查询特征列表，
          每个元素形状 (B, C, H, W)
        """
        B = features_5d[0].shape[0]
        
        # 1. 分解时序特征并准备异构查询和键值
        scale_features = []
        for i, f in enumerate(features_5d):
            # 分解时序帧：T1是过去帧，P是当前帧，T2是未来帧
            t1, p, t2 = f[:, :, 0, :, :], f[:, :, 1, :, :], f[:, :, 2, :, :]
            H, W = p.shape[-2:]
            
            # 获取当前尺度对应的位置编码
            pe_current = pe_list[i]
            
            # 步骤 1: 对齐位置编码 (空间和通道维度)
            # 将位置编码插值到当前特征图尺寸
            pe_aligned = F.interpolate(pe_current, size=(H, W), mode='bilinear', align_corners=False)
            # 如果位置编码通道数不足，通过重复填充到目标通道数
            if pe_aligned.shape[1] != p.shape[1]:
                pe_aligned = pe_aligned.repeat(1, p.shape[1] // pe_aligned.shape[1] + 1, 1, 1)[:, :p.shape[1]]

            # 步骤 2: 在生成Q和K之前，将位置编码注入到序列中
            # 位置编码为每个像素位置提供空间上下文信息
            p_pos = p + pe_aligned
            t1_pos = t1 + pe_aligned
            t2_pos = t2 + pe_aligned
            
            # 步骤 3: 使用时序差分计算V源（直接度量T1-T2变化强度）
            temporal_diff = torch.abs(t1 - t2)  # [B, C, H, W]

            # 步骤 4: 构建位置感知的Q, K, V源
            query = p_pos                                      # Q: P帧 + PE (模型直觉)
            key_context = torch.cat([t1_pos, t2_pos], dim=1)   # K: T1+PE 和 T2+PE 拼接 (时序上下文)
            value_context = temporal_diff                      # V: 时序差分变化权重
            
            # 保存当前尺度的所有必要信息
            scale_features.append({
                'query': query,
                'key_context': key_context,
                'value_context': value_context,
                'original_p': p,
                'size': (H, W)
            })

        # 2. 执行分层交叉注意力，生成增强的Q特征
        enhanced_queries = []
        
        for i in range(self.num_scales):
            current_scale = scale_features[i]
            H, W = current_scale['size']
            
            # 动态确定窗口大小
            # 根据特征图尺寸自适应选择合适的窗口大小
            window_size = get_optimal_window_size(H, W)
            shift_size = window_size // 2 if (i % 2 != 0) else 0
            
            # 准备查询 (将2D特征转换为窗口化的序列)
            q_map = current_scale['query'].permute(0, 2, 3, 1)  # (B, H, W, C)
            
            # 准备异构键值上下文
            if i < self.num_scales - 1:
                # 浅层特征关注最深层特征 (c4) 的异构键值对
                deepest_scale = scale_features[-1]
                key_context = deepest_scale['key_context']
                value_context = deepest_scale['value_context']
                
                # 将深层特征调整到当前尺度的空间尺寸
                key_context = F.interpolate(key_context, size=(H, W), mode='bilinear', align_corners=False)
                value_context = F.interpolate(value_context, size=(H, W), mode='bilinear', align_corners=False)
                
                # 跨尺度通道对齐：将深层的通道数对齐到当前层
                # 例如：c1(i=0)关注c4(j=3)时，将384通道对齐到48通道
                deepest_idx = self.num_scales - 1
                key_context = self.cross_scale_channel_align[i][deepest_idx](key_context)
                
                # 组合异构键值对 (K和V拼接作为kv输入)
                kv_context = torch.cat([key_context, value_context], dim=1)
            else:
                # 深层特征关注所有浅层特征的异构键值拼接
                shallow_keys = []
                shallow_values = []
                for j in range(self.num_scales - 1):
                    shallow_key = scale_features[j]['key_context']
                    shallow_value = scale_features[j]['value_context']
                    
                    # 调整到当前尺度的空间尺寸
                    shallow_key = F.interpolate(shallow_key, size=(H, W), mode='bilinear', align_corners=False)
                    shallow_value = F.interpolate(shallow_value, size=(H, W), mode='bilinear', align_corners=False)
                    
                    # 跨尺度通道对齐：将浅层的通道数对齐到深层
                    shallow_key = self.cross_scale_channel_align[i][j](shallow_key)
                    
                    shallow_keys.append(shallow_key)
                    shallow_values.append(shallow_value)
                
                # 拼接所有浅层的键和值
                combined_keys = torch.cat(shallow_keys, dim=1)
                combined_values = torch.cat(shallow_values, dim=1)
                kv_context = torch.cat([combined_keys, combined_values], dim=1)
            
            kv_map = kv_context.permute(0, 2, 3, 1)  # (B, H, W, C)
            
            # 应用滑动窗口（用于增加感受野和建模长距离依赖）
            if shift_size > 0:
                q_map = torch.roll(q_map, shifts=(-shift_size, -shift_size), dims=(1, 2))
                kv_map = torch.roll(kv_map, shifts=(-shift_size, -shift_size), dims=(1, 2))
            
            # 窗口化：将特征图划分为不重叠的窗口
            q_windows = window_partition(q_map, window_size)  # (nW*B, window_size^2, C)
            kv_windows = window_partition(kv_map, window_size)
            
            # 生成注意力掩码（用于滑动窗口注意力）
            attn_mask = create_mask(H, W, window_size, shift_size, q_windows.device) if shift_size > 0 else None
            
            # 应用自适应 Transformer 块 - 更新查询特征
            # 通过交叉注意力机制，让查询特征关注键值上下文
            updated_q_windows = self.query_update_blocks[i](q_windows, x_kv=kv_windows, attn_mask=attn_mask)
            
            # 还原窗口化：将窗口特征合并回原始特征图
            updated_q_map = window_reverse(updated_q_windows, window_size, H, W)
            
            # 反向滑动窗口
            if shift_size > 0:
                updated_q_map = torch.roll(updated_q_map, shifts=(shift_size, shift_size), dims=(1, 2))
            
            # 转换回2D特征格式 - 增强的查询特征
            enhanced_query = updated_q_map.permute(0, 3, 1, 2)  # (B, C, H, W)
            
            # 应用差分融合层到查询特征
            # 使用卷积进一步提炼增强后的特征
            enhanced_query = self.diff_fusion_layers[i](enhanced_query)
            enhanced_query = enhanced_query + current_scale['original_p']  # 残差连接

            # 保存增强后的查询特征
            enhanced_queries.append(enhanced_query)

        return enhanced_queries
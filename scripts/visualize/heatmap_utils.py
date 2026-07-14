# Copyright (c) Duowang Zhu.
# All rights reserved.

"""
热力图工具函数集合

本模块抽取自 visualize_heatmaps_from_existing.py 的热力图生成逻辑,统一供
visualize_ablation.py(消融配置热力图)等脚本复用,避免重复实现。

核心能力:
1. 将单通道张量/数组映射为 jet 彩色热力图(配 colorbar)
2. 兼容配置1-7(decoder.final_pred)与配置8(decoder.up_c1[0])的 logits 层获取
3. 提供 logits Hook 捕获器,无需侵入式修改模型 forward
"""

import numpy as np
import torch
import torch.nn as nn
from matplotlib import cm
from PIL import Image, ImageDraw, ImageFont


def tensor_to_heatmap(tensor, colormap, vmin=None, vmax=None):
    """
    将单通道张量/数组转换为 RGB 热力图。

    入参:
    - tensor (torch.Tensor | np.ndarray): 形状 [H,W] 或 [1,H,W]
    - colormap: matplotlib 的 Colormap 对象(如 cm.get_cmap('jet'))
    - vmin/vmax (float|None): 归一化范围;None 时取该张量自身最值

    方法:
    1. detach→numpy,压缩多余首维
    2. 线性归一化到 [0,1](vmax==vmin 时置零避免除零)
    3. 应用 colormap 取前 3 通道并 ×255 转 uint8

    出参:
    - heatmap_rgb (np.ndarray): H×W×3 uint8 RGB 热力图
    """
    arr = tensor.detach().cpu().numpy() if isinstance(tensor, torch.Tensor) else np.asarray(tensor)
    if arr.ndim == 3:
        arr = arr[0]
    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()
    arr_norm = (arr - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(arr)
    heatmap = colormap(arr_norm)
    return (heatmap[:, :, :3] * 255).astype(np.uint8)


def create_colorbar_image(colormap, vmin, vmax, width=40, height=256):
    """
    生成竖直渐变颜色条图例(顶部=高值,底部=低值)。

    入参:
    - colormap: matplotlib Colormap
    - vmin/vmax (float): 数值范围标注
    - width/height (int): 颜色条像素尺寸

    方法:
    1. 生成竖向渐变(顶=1.0,底=0.0)并应用 colormap
    2. 用 PIL 在右侧叠加 vmax/vmin 数值文本

    出参:
    - colorbar_img (np.ndarray): (height)×(width+文本区)×3 uint8 RGB
    """
    gradient = np.linspace(1, 0, height).reshape(height, 1)
    gradient = np.repeat(gradient, width, axis=1)
    bar_rgb = (colormap(gradient)[:, :, :3] * 255).astype(np.uint8)
    bar_pil = Image.fromarray(bar_rgb)
    draw = ImageDraw.Draw(bar_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.text((width + 5, 5), f"{vmax:.2f}", fill=(255, 255, 255), font=font)
    draw.text((width + 5, height - 20), f"{vmin:.2f}", fill=(255, 255, 255), font=font)
    return np.array(bar_pil)


def get_logits_layer(decoder):
    """
    兼容获取 decoder 的最终 logits 输出层。

    入参:
    - decoder (nn.Module): 消融模型(配置1-7,changedecoder.ChangeDecoder)或
      完整模型(配置8,change_decoder.ChangeDecoder)的解码器

    方法:
    - 配置1-7:返回 decoder.final_pred(存在该属性)
    - 配置8:回退到 decoder.up_c1[0](Sequential 内首个 Conv2d)
    - 均无则抛出 AttributeError

    出参:
    - layer (nn.Module): 最终 logits 卷积层(hook 注册目标)
    """
    if hasattr(decoder, 'final_pred'):
        return decoder.final_pred
    if hasattr(decoder, 'up_c1'):
        up_c1 = decoder.up_c1
        # up_c1 为 nn.Sequential 时返回首层 Conv2d;为单层时直接返回
        return up_c1[0] if isinstance(up_c1, nn.Sequential) else up_c1
    raise AttributeError(f"无法识别 decoder 的 logits 层: {type(decoder).__name__}(无 final_pred/up_c1)")


class LogitsCapture:
    """
    轻量 logits 捕获器:forward hook 抓取最终 logits 层输出(sigmoid 之前)。

    入参:
    - 无

    方法:
    - attach(model):自动定位 logits 层并注册 forward hook
    - detach():移除 hook
    - logits 属性:最近一次前向传播捕获的 logits 张量

    出参:
    - 通过 .logits 属性访问捕获结果,形状 [B,1,H,W]
    """

    def __init__(self):
        self.logits = None
        self._handle = None

    def _hook_fn(self, module, inputs, output):
        # forward hook 回调:output 即 logits 层(Conv2d)输出,未过 sigmoid
        self.logits = output.detach().clone()

    def attach(self, model):
        """
        入参:
        - model: Trainer 或 TrainerAblation 实例

        方法:
        - 通过 get_logits_layer 定位层并注册 hook

        出参:
        - self(链式调用)
        """
        layer = get_logits_layer(model.decoder)
        self._handle = layer.register_forward_hook(self._hook_fn)
        return self

    def detach(self):
        """
        入参: 无
        方法: 移除已注册的 hook
        出参: 无
        """
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

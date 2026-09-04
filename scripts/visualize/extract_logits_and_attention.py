# Copyright (c) Duowang Zhu.
# All rights reserved.

"""
使用Hook捕捉模型解码器输出的logits（sigmoid之前）和注意力图

入参:
- model: 训练好的模型
- x, y: 输入图像对

方法:
1. 注册forward hook到解码器的final_pred层（sigmoid之前）
2. 注册forward hook到注意力模块（如果存在）
3. 执行前向传播
4. 提取logits和注意力图

出参:
- logits: 未经过sigmoid的logits分数
- attention_maps: 注意力图（如果存在）
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import functools


class DecoderWrapper(nn.Module):
    """
    解码器包装器，用于分别捕捉T1、P、T2的logits
    
    入参:
    - decoder: 原始解码器模块
    - hook_manager: Hook管理器，用于存储捕捉到的logits
    
    方法:
    - 包装解码器的forward方法
    - 分别对T1、P、T2进行解码，生成各自的logits
    
    出参:
    - 与原始解码器相同的输出（融合后的预测）
    """
    
    def __init__(self, decoder: nn.Module, hook_manager):
        super().__init__()
        self.decoder = decoder
        self.hook_manager = hook_manager
    
    def _decode_single_phase(self, f_4d_single: List[torch.Tensor]) -> torch.Tensor:
        """
        对单个时相进行解码，生成logits
        
        入参:
        - f_4d_single: 单个时相的多尺度4D特征列表 [c1, c2, c3, c4]
        
        方法:
        1. 渐进式上采样与特征融合
        2. 生成最终logits（sigmoid之前）
        
        出参:
        - logits: 预测logits，形状 (B, num_class, 256, 256)
        """
        c1, c2, c3, c4 = f_4d_single
        
        # 渐进式上采样与特征融合（与原始解码器相同的逻辑）
        c3f = c3 + self.decoder.up_c4(c4)
        c2f = c2 + self.decoder.up_c3(c3f)
        c1f = c1 + self.decoder.up_c2(c2f)
        
        # 上采样到目标分辨率
        c1f_upsampled = self.decoder.stem_upsample(c1f)
        
        # 生成最终logits（sigmoid之前）
        logits = self.decoder.final_pred(c1f_upsampled)
        
        return logits
    
    def forward(self, f_5d: List[torch.Tensor]) -> torch.Tensor:
        """
        包装的解码器forward，分别捕捉T1、P、T2的logits
        
        入参:
        - f_5d: 5D特征列表，每个元素形状 (B, C, 3, H, W)
        
        方法:
        1. 分别提取每个尺度的T1、P、T2特征
        2. 分别对T1、P、T2进行解码，生成各自的logits
        3. 保存logits到hook_manager
        4. 调用原始解码器的forward（融合后的预测）
        
        出参:
        - 原始解码器的输出（融合后的预测）
        """
        if self.hook_manager.capture_decoder_features:
            # 分别提取每个尺度的T1、P、T2特征
            f_4d_t1 = []
            f_4d_p = []
            f_4d_t2 = []
            
            for feat_5d in f_5d:
                # feat_5d: (B, C, 3, H, W)
                t1 = feat_5d[:, :, 0, :, :]  # (B, C, H, W)
                p = feat_5d[:, :, 1, :, :]   # (B, C, H, W)
                t2 = feat_5d[:, :, 2, :, :]  # (B, C, H, W)
                
                f_4d_t1.append(t1)
                f_4d_p.append(p)
                f_4d_t2.append(t2)
            
            # 分别对T1、P、T2进行解码，生成各自的logits
            logits_t1 = self._decode_single_phase(f_4d_t1)
            logits_p = self._decode_single_phase(f_4d_p)
            logits_t2 = self._decode_single_phase(f_4d_t2)
            
            # 保存logits到hook_manager
            self.hook_manager.decoder_t1_logits = logits_t1.detach().clone()
            self.hook_manager.decoder_p_logits = logits_p.detach().clone()
            self.hook_manager.decoder_t2_logits = logits_t2.detach().clone()
            
            print(f"[Hook] 捕捉到T1/P/T2的logits，形状: {logits_t1.shape}")
        
        # 调用原始解码器（融合后的预测）
        return self.decoder(f_5d)


class LogitsAndAttentionHook:
    """
    Hook类，用于捕捉logits、T1/P/T2特征和注意力图
    
    入参:
    - capture_logits: 是否捕捉logits
    - capture_attention: 是否捕捉注意力图
    - capture_decoder_features: 是否捕捉解码器中的T1、P、T2特征
    
    方法:
    - 注册hook到指定层
    - 存储捕捉到的特征
    
    出参:
    - logits: 捕捉到的logits
    - decoder_features: 解码器中的T1、P、T2特征
    - attention_maps: 捕捉到的注意力图列表
    """
    
    def __init__(self, capture_logits: bool = True, capture_attention: bool = True, capture_decoder_features: bool = True):
        self.capture_logits = capture_logits
        self.capture_attention = capture_attention
        self.capture_decoder_features = capture_decoder_features
        self.logits = None
        # T1、P、T2的logits（分别解码后的结果）
        self.decoder_t1_logits = None
        self.decoder_p_logits = None
        self.decoder_t2_logits = None
        self.attention_maps = []
        self.hooks = []
        self.original_decoder = None  # 保存原始解码器引用
        self.model_ref = None  # 保存模型引用
    
    def logits_hook_fn(self, module, input, output):
        """
        捕捉解码器输出的logits（sigmoid之前）
        
        入参:
        - module: 被hook的模块
        - input: 模块的输入
        - output: 模块的输出（logits）
        
        方法:
        - 直接保存output作为logits
        
        出参:
        - 无返回值，直接修改self.logits
        """
        if self.capture_logits:
            # output就是logits（sigmoid之前）
            self.logits = output.detach().clone()
            print(f"[Hook] 捕捉到logits形状: {self.logits.shape}")
    
    def attention_hook_fn(self, module, input, output):
        """
        捕捉注意力模块的输出
        
        入参:
        - module: 被hook的模块
        - input: 模块的输入
        - output: 模块的输出（可能是注意力图或增强后的特征）
        
        方法:
        - 保存注意力相关的输出
        
        出参:
        - 无返回值，直接添加到self.attention_maps
        """
        if self.capture_attention:
            # 根据不同的注意力模块，output可能是不同的格式
            # 如果是元组，可能需要提取注意力权重
            if isinstance(output, tuple):
                # 某些注意力模块返回 (output, attention_weights)
                attention_map = output[1].detach().clone() if len(output) > 1 else output[0].detach().clone()
            else:
                attention_map = output.detach().clone()
            
            self.attention_maps.append(attention_map)
            print(f"[Hook] 捕捉到注意力图形状: {attention_map.shape}")
    
    def register_hooks(self, model: nn.Module):
        """
        注册hook到模型的指定层
        
        入参:
        - model: 要注册hook的模型
        
        方法:
        1. 包装解码器以捕捉T1、P、T2特征（如果需要）
        2. 找到解码器的final_pred层（sigmoid之前）
        3. 找到所有注意力模块
        4. 注册相应的hook
        
        出参:
        - 无返回值，直接注册hook
        """
        # 包装解码器以捕捉T1、P、T2特征
        if self.capture_decoder_features:
            if hasattr(model, 'decoder'):
                self.original_decoder = model.decoder
                self.model_ref = model
                wrapped_decoder = DecoderWrapper(self.original_decoder, self)
                model.decoder = wrapped_decoder
                print("[Hook] 已包装解码器以捕捉T1/P/T2特征")
            else:
                print("[警告] 未找到 decoder，无法捕捉T1/P/T2特征")
        
        # 注册logits hook到解码器的final_pred层
        if self.capture_logits:
            decoder = model.decoder.decoder if isinstance(model.decoder, DecoderWrapper) else model.decoder
            if hasattr(decoder, 'final_pred'):
                hook = decoder.final_pred.register_forward_hook(self.logits_hook_fn)
                self.hooks.append(hook)
                print("[Hook] 已注册logits hook到 decoder.final_pred")
            else:
                print("[警告] 未找到 decoder.final_pred，无法捕捉logits")
        
        # 注册注意力hook到注意力模块
        if self.capture_attention:
            # 查找所有注意力模块
            attention_modules = []
            
            # 检查是否有attention_modules属性（TrainerAblation）
            if hasattr(model, 'attention_modules'):
                attention_modules = model.attention_modules
            
            # 检查是否有attention属性（Trainer）
            elif hasattr(model, 'attention'):
                attention_modules = [model.attention] if isinstance(model.attention, nn.Module) else []
            
            # 递归查找所有包含"attention"的模块
            if not attention_modules:
                for name, module in model.named_modules():
                    if 'attention' in name.lower() and isinstance(module, nn.Module):
                        attention_modules.append(module)
            
            # 注册hook到每个注意力模块
            for i, attn_module in enumerate(attention_modules):
                hook = attn_module.register_forward_hook(self.attention_hook_fn)
                self.hooks.append(hook)
                print(f"[Hook] 已注册注意力hook到模块 {i+1}: {type(attn_module).__name__}")
            
            if not attention_modules:
                print("[警告] 未找到注意力模块，无法捕捉注意力图")
    
    def remove_hooks(self):
        """
        移除所有注册的hook并恢复原始解码器
        
        入参:
        - 无
        
        方法:
        - 移除所有hook
        - 恢复原始解码器（如果被包装过）
        
        出参:
        - 无返回值
        """
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # 恢复原始解码器
        if self.model_ref is not None and self.original_decoder is not None:
            self.model_ref.decoder = self.original_decoder
            self.original_decoder = None
            self.model_ref = None
            print("[Hook] 已恢复原始解码器")
        
        print("[Hook] 已移除所有hook")
    
    def get_results(self, inverse_t1_t2: bool = False) -> Dict[str, torch.Tensor]:
        """
        获取捕捉到的结果

        入参:
        - inverse_t1_t2 (bool): 是否对T1和T2的logits进行1-logits变换，默认False

        方法:
        - 返回logits、T1/P/T2 logits和注意力图的字典
        - 如果inverse_t1_t2=True，对T1和T2的logits应用1-x变换

        出参:
        - results: 包含logits、decoder_t1_logits、decoder_p_logits、decoder_t2_logits和attention_maps的字典
        """
        results = {}
        if self.logits is not None:
            results['logits'] = self.logits
        # T1、P、T2分别解码后的logits
        if self.decoder_t1_logits is not None:
            t1_logits = self.decoder_t1_logits
            if inverse_t1_t2:
                t1_logits = 1 - t1_logits
            results['decoder_t1_logits'] = t1_logits
        if self.decoder_p_logits is not None:
            results['decoder_p_logits'] = self.decoder_p_logits
        if self.decoder_t2_logits is not None:
            t2_logits = self.decoder_t2_logits
            if inverse_t1_t2:
                t2_logits = 1 - t2_logits
            results['decoder_t2_logits'] = t2_logits
        if self.attention_maps:
            results['attention_maps'] = self.attention_maps
        return results


def extract_logits_and_attention(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    capture_logits: bool = True,
    capture_attention: bool = True,
    capture_decoder_features: bool = True,
    inverse_t1_t2: bool = False
) -> Dict[str, torch.Tensor]:
    """
    提取模型输出的logits、T1/P/T2特征和注意力图

    入参:
    - model: 训练好的模型
    - x: 第一时相图像，形状 [B, 3, H, W]
    - y: 第二时相图像，形状 [B, 3, H, W]
    - capture_logits: 是否捕捉logits
    - capture_attention: 是否捕捉注意力图
    - capture_decoder_features: 是否捕捉解码器中的T1、P、T2特征
    - inverse_t1_t2: 是否对T1和T2的logits进行1-logits变换，默认False

    方法:
    1. 创建Hook对象
    2. 注册hook到模型
    3. 执行前向传播
    4. 移除hook
    5. 返回结果

    出参:
    - results: 包含logits、decoder_features和attention_maps的字典
    """
    # 创建Hook对象
    hook_manager = LogitsAndAttentionHook(
        capture_logits=capture_logits,
        capture_attention=capture_attention,
        capture_decoder_features=capture_decoder_features
    )

    # 注册hook
    hook_manager.register_hooks(model)

    # 设置为评估模式
    model.eval()

    # 执行前向传播
    with torch.no_grad():
        output = model(x, y)

    # 获取结果（应用可选的变换）
    results = hook_manager.get_results(inverse_t1_t2=inverse_t1_t2)

    # 移除hook
    hook_manager.remove_hooks()

    return results


# 使用示例
if __name__ == '__main__':
    """
    使用示例：
    
    # 1. 加载模型
    from model.trainer import Trainer
    from model.trainer_ablation import TrainerAblation, create_ablation_model
    import argparse
    
    args = argparse.Namespace(...)  # 你的配置参数
    model = Trainer(args=args)  # 或 create_ablation_model(args, experiment_id=1)
    
    # 加载权重
    checkpoint = torch.load('path/to/model.pth')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    # 2. 准备输入数据
    x = torch.randn(1, 3, 256, 256)  # T1图像
    y = torch.randn(1, 3, 256, 256)  # T2图像
    
    # 3. 提取logits和注意力图
    results = extract_logits_and_attention(
        model=model,
        x=x,
        y=y,
        capture_logits=True,
        capture_attention=True
    )
    
    # 4. 使用结果
    logits = results['logits']  # 形状: [B, num_class, H, W]
    attention_maps = results['attention_maps']  # 列表，每个元素是一个注意力图
    
    # 5. 可视化logits（可以转换为概率图）
    import matplotlib.pyplot as plt
    import numpy as np
    
    # logits转换为概率（应用sigmoid）
    probs = torch.sigmoid(logits)
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(logits[0, 0].cpu().numpy(), cmap='jet')
    axes[0].set_title('Logits (before sigmoid)')
    axes[1].imshow(probs[0, 0].cpu().numpy(), cmap='jet')
    axes[1].set_title('Probabilities (after sigmoid)')
    if attention_maps:
        axes[2].imshow(attention_maps[0][0, 0].cpu().numpy(), cmap='jet')
        axes[2].set_title('Attention Map')
    plt.show()
    """
    pass


# Copyright (c) Duowang Zhu.
# All rights reserved.

"""
从配置8（完整模型）生成T1、P、T2分别解码的logits热力图可视化

该脚本会：
1. 加载配置8（完整模型：X3D+A+B+C）
2. 使用Hook分别捕捉T1、P、T2三个时相解码后的logits（sigmoid之前）
3. 生成热力图可视化，展示模型在T1、P、T2中都包含的变化信息

使用示例（Windows PowerShell）：
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python scripts/visualize/visualize_heatmaps_from_existing.py --file_root "E:\rqx\dataes\LBFD-CD" --model_path ./exp_new/LBFD-CD/best_model.pth --vis_results_dir ./vis_logits_attention --gpu_id 0

使用示例（简化版，使用默认值）：
python scripts/visualize/visualize_heatmaps_from_existing.py --file_root "E:\rqx\dataes\LBFD-CD" --model_path ./exp_new/LBFD-CD/best_model.pth --vis_results_dir ./vis_logits_attention --gpu_id 0
"""

import os
import sys
import time
import argparse
import glob
from os.path import join

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import json
from skimage import io
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm

# 插入当前路径以导入本地模块
sys.path.insert(0, '.')

import data.dataset as RSDataset
import data.transforms as RSTransforms
from model.trainer_ablation import TrainerAblation, create_ablation_model
from model.trainer import Trainer
from scripts.visualize.extract_logits_and_attention import extract_logits_and_attention

# 随机种子
seed = 16


def create_rainbow_colormap():
    """
    创建彩虹色配色方案（从Low的淡粉色到High的蓝色）
    
    入参:
    - 无
    
    方法:
    - 创建从淡粉色→橙→黄→绿→青→浅蓝→深蓝的渐变色
    - 蓝色表示高值（High）
    
    出参:
    - colormap: matplotlib的Colormap对象
    """
    colors = [
        '#FFB6C1',  # 淡粉色 (Low)
        '#FF8C00',  # 深橙
        '#FFFF00',  # 黄色
        '#00FF00',  # 绿色
        '#00CED1',  # 深青色
        '#4169E1',  # 皇家蓝
        '#00008B',  # 深蓝
        '#4B0082'   # 紫色/深蓝 (High)
    ]
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('rainbow_custom', colors, N=n_bins)
    return cmap


def create_colorbar_image(colormap, vmin, vmax, width=30, height=256):
    """
    创建颜色条图例
    
    入参:
    - colormap: matplotlib的Colormap对象
    - vmin: 最小值
    - vmax: 最大值
    - width: 颜色条宽度（像素）
    - height: 颜色条高度（像素）
    
    方法:
    1. 创建渐变颜色条
    2. 添加标签
    
    出参:
    - colorbar_img: RGB图像，形状 [height, width, 3]
    """
    # 创建渐变
    gradient = np.linspace(1, 0, height).reshape(height, 1)
    gradient = np.repeat(gradient, width, axis=1)
    
    # 应用colormap
    colorbar_rgb = colormap(gradient)[:, :, :3]
    colorbar_rgb = (colorbar_rgb * 255).astype(np.uint8)
    
    # 转换为PIL Image以便添加文字
    colorbar_pil = Image.fromarray(colorbar_rgb)
    draw = ImageDraw.Draw(colorbar_pil)
    
    # 添加标签（使用PIL的默认字体）
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # 添加"High"和"Low"标签
    # 顶部：高值（蓝色）
    draw.text((width + 5, 5), f"{vmax:.2f}", fill=(255, 255, 255), font=font)
    draw.text((width + 5, 20), "High (Red)", fill=(255, 255, 255), font=font)
    # 底部：低值（淡粉色）
    draw.text((width + 5, height - 20), f"{vmin:.2f}", fill=(255, 255, 255), font=font)
    draw.text((width + 5, height - 35), "Low (Blue)", fill=(255, 255, 255), font=font)
    
    # 转换回numpy数组
    colorbar_img = np.array(colorbar_pil)
    
    return colorbar_img


def tensor_to_heatmap(tensor, colormap, vmin=None, vmax=None):
    """
    将张量转换为热图（使用彩虹色配色）
    
    入参:
    - tensor: 输入张量，形状 [H, W] 或 [1, H, W]
    - colormap: matplotlib的Colormap对象
    - vmin: 最小值（用于归一化）
    - vmax: 最大值（用于归一化）
    
    方法:
    1. 将张量转换为numpy数组
    2. 归一化到[0,1]范围
    3. 应用colormap
    4. 转换为RGB图像
    
    出参:
    - heatmap: RGB图像，形状 [H, W, 3]，值域[0, 255]
    """
    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = tensor
    
    if len(arr.shape) == 3:
        arr = arr[0]
    
    if vmin is None:
        vmin = arr.min()
    if vmax is None:
        vmax = arr.max()
    
    if vmax > vmin:
        arr_normalized = (arr - vmin) / (vmax - vmin)
    else:
        arr_normalized = np.zeros_like(arr)
    
    heatmap = colormap(arr_normalized)
    heatmap_rgb = (heatmap[:, :, :3] * 255).astype(np.uint8)
    
    return heatmap_rgb


def create_test_loader(args, val_transform):
    """
    创建测试的数据加载器。
    """
    test_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="test",
        transform=val_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    return test_loader


@torch.no_grad()
def visualize_heatmaps_from_existing(args, vis_results_dir):
    """
    从配置8（完整模型）生成T1、P、T2分别解码的logits热力图可视化
    
    入参:
    - args: 配置参数
    - vis_results_dir: 可视化结果根目录
    
    方法:
    1. 加载配置8（完整模型）
    2. 在测试集上进行推理，分别提取T1、P、T2的logits
    3. 生成热力图可视化（包含颜色条图例和真实预测）
    
    出参:
    - total_vis_count (int): 可视化的图像数量
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.manual_seed(seed=16)
        torch.cuda.manual_seed(seed=16)
    else:
        torch.manual_seed(seed=16)
        print("警告: CUDA不可用，将使用CPU运行")

    # 配置名称映射
    config_names = {
        1: "X3D",
        2: "X3D+A",
        3: "X3D+B",
        4: "X3D+C",
        5: "X3D+A+B",
        6: "X3D+A+C",
        7: "X3D+B+C",
        8: "X3D+A+B+C"
    }

    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    
    # 创建输出目录（直接输出到vis_logits_attention）
    output_dir = join(vis_results_dir, dataset_name)
    heatmaps_dir = join(output_dir, 'heatmaps')  # 热力图目录
    predictions_dir = join(output_dir, 'predictions')  # 预测掩码目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(heatmaps_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    
    # 使用jet颜色映射
    jet_cmap = cm.get_cmap('jet')
    
    # 获取数据变换
    _, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)
    test_loader = create_test_loader(args, val_transform)
    
    # 只加载配置8（完整模型）
    print(f"\n=== 加载配置8（完整模型）===")
    model_path = args.model_path if hasattr(args, 'model_path') and args.model_path else join(args.save_dir, dataset_name, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"[错误] 未找到模型文件: {model_path}")
        return 0
    
    # 创建模型
    import copy
    config_args = copy.deepcopy(args)
    model = Trainer(args=config_args)
    
    # 加载权重
    try:
        loaded_data = torch.load(model_path, map_location='cpu')
        if isinstance(loaded_data, dict):
            state_dict = loaded_data.get('state_dict', loaded_data)
        else:
            model_to_load = loaded_data.module if hasattr(loaded_data, 'module') else loaded_data
            state_dict = model_to_load.state_dict()
        
        # 处理DataParallel
        new_state_dict = {}
        has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
        if has_module_prefix:
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print(f"[完成] 配置8 ({config_names.get(8, 'X3D+A+B+C')})")
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        return 0
    
    # 执行可视化
    total_vis_count = 0
    
    with torch.no_grad():
        for i, batched_inputs in enumerate(test_loader):
            img, target = batched_inputs[0].to(device), batched_inputs[1].to(device)
            pre_img = img[:, :3, :, :]
            post_img = img[:, 3:, :, :]
            
            # 加载原始图像
            original_pre_path = test_loader.dataset.pre_images[i]
            original_post_path = test_loader.dataset.post_images[i]
            original_filename = os.path.basename(original_pre_path)
            
            pre_img_vis = io.imread(original_pre_path)
            post_img_vis = io.imread(original_post_path)
            
            target_np = target.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
            output_h, output_w = target_np.shape
            
            pre_img_vis = cv2.resize(pre_img_vis, (output_w, output_h))
            post_img_vis = cv2.resize(post_img_vis, (output_w, output_h))
            pre_img_vis = cv2.cvtColor(pre_img_vis, cv2.COLOR_RGB2BGR)
            post_img_vis = cv2.cvtColor(post_img_vis, cv2.COLOR_RGB2BGR)
            
            target_vis = np.stack([(target_np * 255)] * 3, axis=-1)
            
            # 提取T1、P、T2的logits和融合后的logits
            # 使用inverse_t1_t2=True对T1和T2应用1-logits变换
            results = extract_logits_and_attention(
                model=model,
                x=pre_img,
                y=post_img,
                capture_logits=True,
                capture_attention=False,
                capture_decoder_features=True,
                inverse_t1_t2=True
            )
            
            if 'decoder_t1_logits' not in results:
                print(f"[警告] 未捕捉到T1/P/T2 logits，跳过图像 {i+1}")
                continue
            
            # 获取T1、P、T2分别解码后的logits
            logits_t1 = results['decoder_t1_logits']  # [B, num_class, H, W]
            logits_p = results['decoder_p_logits']    # [B, num_class, H, W]
            logits_t2 = results['decoder_t2_logits']  # [B, num_class, H, W]
            logits_fused = results.get('logits', None)  # 融合后的logits（可选）
            
            # 转换为numpy数组
            logits_t1_np = logits_t1[0, 0].cpu().numpy()  # [H, W]
            logits_p_np = logits_p[0, 0].cpu().numpy()    # [H, W]
            logits_t2_np = logits_t2[0, 0].cpu().numpy()  # [H, W]
            
            # 计算全局最小最大值（用于统一颜色条）
            logits_min = min(logits_t1_np.min(), logits_p_np.min(), logits_t2_np.min())
            logits_max = max(logits_t1_np.max(), logits_p_np.max(), logits_t2_np.max())
            
            # 获取融合后的预测结果（应用sigmoid后二值化）
            if logits_fused is not None:
                probs = torch.sigmoid(logits_fused)
            else:
                # 如果没有融合logits，使用T1、P、T2的平均
                avg_logits = (logits_t1 + logits_p + logits_t2) / 3
                probs = torch.sigmoid(avg_logits)
            pred = torch.where(probs > 0.5, 1, 0)
            pred_np = pred[0, 0].cpu().numpy().astype(np.uint8)
            prediction_mask = np.stack([(pred_np * 255)] * 3, axis=-1)
            
            # 创建颜色条
            logits_colorbar = create_colorbar_image(jet_cmap, logits_min, logits_max, width=40, height=output_h)
            
            # 创建分隔线
            spacer = np.zeros((output_h, 10, 3), dtype=np.uint8)
            
            # 组合图像：T1 Image | T2 Image | GT | T1 Logits | P Logits | T2 Logits | Fused Pred | Colorbar
            combined_parts = [pre_img_vis, spacer, post_img_vis, spacer, target_vis, spacer]
            
            # 添加T1、P、T2的logits热力图
            t1_logits_heatmap = tensor_to_heatmap(logits_t1_np, jet_cmap, vmin=logits_min, vmax=logits_max)
            p_logits_heatmap = tensor_to_heatmap(logits_p_np, jet_cmap, vmin=logits_min, vmax=logits_max)
            t2_logits_heatmap = tensor_to_heatmap(logits_t2_np, jet_cmap, vmin=logits_min, vmax=logits_max)
            
            combined_parts.extend([t1_logits_heatmap, spacer, p_logits_heatmap, spacer, t2_logits_heatmap, spacer])
            
            # 添加融合后的预测
            combined_parts.extend([prediction_mask, spacer])
            
            # 添加颜色条
            combined_parts.append(logits_colorbar)
            
            # 水平拼接
            combined_vis = np.concatenate(combined_parts, axis=1)
            
            # 添加标题信息（参考visualize_ablation.py风格）
            title_height = 40
            title_img = np.zeros((title_height, combined_vis.shape[1], 3), dtype=np.uint8)
            title_text = f"Config 8 (X3D+A+B+C) - T1/P/T2 Logits Heatmaps - {original_filename}"
            cv2.putText(title_img, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 添加图例（参考visualize_ablation.py风格）
            legend_height = 80
            legend_img = np.zeros((legend_height, combined_vis.shape[1], 3), dtype=np.uint8)
            
            # 基础图例
            legend_texts = [
                "T1 Image",
                "T2 Image", 
                "Ground Truth",
                "T1 Logits",
                "P Logits",
                "T2 Logits",
                "Fused Pred"
            ]
            x_positions = [
                0, 
                output_w + 10, 
                2*(output_w + 10),
                3*(output_w + 10),
                4*(output_w + 10),
                5*(output_w + 10),
                6*(output_w + 10)
            ]
            for j, text in enumerate(legend_texts):
                if j < len(x_positions):
                    cv2.putText(legend_img, text, (x_positions[j], 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 颜色条说明
            current_x = 7 * (output_w + 10)
            cv2.putText(legend_img, "Logits Colorbar", (current_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(legend_img, f"Range: [{logits_min:.2f}, {logits_max:.2f}]", (current_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(legend_img, "(Red=High, Blue=Low)", (current_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 最终组合
            final_vis = np.concatenate([title_img, combined_vis, legend_img], axis=0)
            
            # 保存组合可视化
            save_path = join(output_dir, f"combined_{original_filename}")
            cv2.imwrite(save_path, final_vis)
            print(f"保存组合可视化到: {save_path}")
            
            # 保存T1、P、T2的logits热力图
            config_name = config_names.get(8, "X3D+A+B+C")
            
            t1_logits_heatmap = tensor_to_heatmap(logits_t1_np, jet_cmap, vmin=logits_min, vmax=logits_max)
            p_logits_heatmap = tensor_to_heatmap(logits_p_np, jet_cmap, vmin=logits_min, vmax=logits_max)
            t2_logits_heatmap = tensor_to_heatmap(logits_t2_np, jet_cmap, vmin=logits_min, vmax=logits_max)
            
            t1_filename = f"{config_name}_t1_logits_{original_filename}"
            p_filename = f"{config_name}_p_logits_{original_filename}"
            t2_filename = f"{config_name}_t2_logits_{original_filename}"
            
            cv2.imwrite(join(heatmaps_dir, t1_filename), t1_logits_heatmap)
            cv2.imwrite(join(heatmaps_dir, p_filename), p_logits_heatmap)
            cv2.imwrite(join(heatmaps_dir, t2_filename), t2_logits_heatmap)
            
            # 保存预测掩码（二值化预测结果）
            pred_filename = f"{config_name}_pred_{original_filename}"
            pred_path = join(predictions_dir, pred_filename)
            cv2.imwrite(pred_path, prediction_mask)
            
            print(f"  已保存 T1/P/T2 logits热力图和预测掩码")
            
            total_vis_count += 1
    
    # 清理模型
    del model
    torch.cuda.empty_cache()
    
    print(f"\n热力图可视化完成，共处理 {total_vis_count} 张图像。")
    return total_vis_count


def get_parser():
    """
    创建脚本的参数解析器。
    """
    parser = argparse.ArgumentParser(description='从配置8（完整模型）生成T1、P、T2分别解码的logits热力图可视化')
    
    parser.add_argument('--vis_results_dir', type=str, default='./vis_logits_attention',
                       help='可视化结果根目录（默认：./vis_logits_attention）')
    parser.add_argument('--file_root', type=str, required=True, help='数据集根目录路径')
    parser.add_argument('--save_dir', type=str, default='./exp_ablation_batch', help='实验保存的根目录')
    parser.add_argument('--model_path', type=str, required=True, help='配置8（完整模型）的模型路径')
    parser.add_argument('--gpu_id', type=str, default='0', help='使用的GPU ID')
    parser.add_argument('--in_height', type=int, default=256, help='RGB图像高度')
    parser.add_argument('--in_width', type=int, default=256, help='RGB图像宽度')
    parser.add_argument('--num_perception_frame', type=int, default=1, help='感知帧数量')
    parser.add_argument('--pretrained', default=r'model\X3D_L.pyth', type=str, help='预训练X3D权重路径')
    
    return parser


def main():
    """
    主函数
    """
    parser = get_parser()
    args = parser.parse_args()
    
    print(f"\n=== 从配置8（完整模型）生成T1/P/T2 logits热力图 ===")
    print(f"数据集: {os.path.basename(os.path.normpath(args.file_root))}")
    print(f"模型路径: {args.model_path}")
    print(f"可视化结果目录: {args.vis_results_dir}")
    
    visualize_heatmaps_from_existing(args, args.vis_results_dir)


if __name__ == '__main__':
    main()


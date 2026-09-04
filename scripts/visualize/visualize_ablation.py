# Copyright (c) Duowang Zhu.
# All rights reserved.

import os
import sys
import time
import copy
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
from PIL import Image, ImageDraw
from matplotlib import cm

# 插入当前路径以导入本地模块(做什么:使 data/model/同包脚本可被 import)
sys.path.insert(0, '.')

import data.dataset as RSDataset
import data.transforms as RSTransforms
from model.trainer_ablation import TrainerAblation, create_ablation_model
from model.trainer import Trainer
from scripts.visualize.heatmap_utils import (
    tensor_to_heatmap, create_colorbar_image, LogitsCapture,
)

# 随机种子
seed = 16

# jet 配色:复用于所有消融配置的 logits 热力图,保证跨配置视觉可比
JET_CMAP = cm.get_cmap('jet')

'''
消融实验可视化脚本 —— 统一输出最终融合预测 logits 热力图

使用示例：
python scripts/visualize/visualize_ablation.py --ablation_config 1 --file_root "E:\rqx\dataes\GVLM-CD" --save_dir ./exp_ablation_batch --output_dir ./vis_results_ablation
python scripts/visualize/visualize_ablation.py --combine_all_configs --file_root "E:\rqx\dataes\LBFD-CD" --save_dir ./exp_ablation_batch --output_dir ./vis_results_ablation
python scripts/visualize/visualize_ablation.py --combine_existing --file_root "E:\rqx\dataes\LBFD-CD" --vis_results_dir ./vis_results_ablation

=== 配置说明 ===
配置ID与模型组合的对应关系：
- 配置1 (X3D): Base Only
- 配置2 (X3D+A): Base + Attention
- 配置3 (X3D+B): Base + ASPP
- 配置4 (X3D+C): Base + Transformer
- 配置5 (X3D+A+B): Base + Attention + ASPP
- 配置6 (X3D+A+C): Base + Attention + Transformer
- 配置7 (X3D+B+C): Base + ASPP + Transformer
- 配置8 (X3D+A+B+C): Base + Attention + ASPP + Transformer (完整模型)

=== 热力图方案 ===
- 每个配置捕获其 decoder 最终 logits 层输出(sigmoid 之前),用 jet 配色生成热力图
- 配置1-7 层为 decoder.final_pred,配置8 层为 decoder.up_c1[0](由 heatmap_utils.get_logits_layer 统一处理)
- 单个配置组合图：T1 | T2 | GT | Pred | Logits热力图 | Colorbar
- 多配置对比图：T1 | T2 | GT | (各配置 Pred | 各配置 Logits热力图) + 共享 Colorbar

=== 输出说明 ===
- 单个配置可视化：输出到 {output_dir}/{dataset_name}/config_{config_id}/
  - 含 prediction_masks/、logits_heatmaps/、path_mapping.json
- 拼接结果（--combine_existing）：输出到 {vis_results_dir}/{dataset_name}/all_configs_combined/
  - 输出文件：combined_{basename}.png
- 拼接格式：T1 | T2 | GT | Config1 Heat | Config2 Heat | ... | Config8 Heat
'''

def create_test_loader(args, val_transform):
    """
    创建测试的数据加载器。
    
    Args:
        args: 配置参数
        val_transform: 验证/测试数据变换
        
    Returns:
        test_loader: 测试数据加载器
    """
    # 测试数据
    test_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="test",
        transform=val_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,  # 可视化时每次处理一张图像
        shuffle=False,
        num_workers=0,  # 设为0避免Windows多进程pickle问题
        pin_memory=True
    )
    
    print(f"测试集包含 {len(test_loader)} 个批次。")
    
    return test_loader

@torch.no_grad()
def visualize_single_config(args, model_path, output_dir):
    """
    可视化单个消融实验配置的结果。
    
    入参:
    - args: 配置参数
    - model_path: 模型文件路径
    - output_dir: 可视化结果输出目录
    
    方法:
    1. 创建消融实验模型
    2. 加载模型权重（过滤余弦相似度增强模块权重）
    3. 在测试集上进行推理和可视化
    
    出参:
    - total_vis_count (int): 可视化的图像数量
    """
    # 检查CUDA可用性
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

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"[警告] 未找到 {model_path} 文件；跳过配置 {args.ablation_config} 的可视化。")
        return None

    # 根据消融配置创建模型
    if args.ablation_config == 8:
        # 配置8使用完整的Trainer模型（所有模块都启用）
        model = Trainer(args=args)
    else:
        # 配置1-7使用消融实验模型
        model = create_ablation_model(args, experiment_id=args.ablation_config)
    
    # 模拟一次前向传播以初始化所有动态模块
    print(f"[初始化] 初始化动态模块（模拟前向传播）...", end='', flush=True)
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, args.in_height, args.in_width)
        try:
            _ = model(dummy_input, dummy_input)
            print(f" [完成]")
        except Exception as e:
            print(f" [警告]")
            print(f"   模拟前向传播出现警告: {e}")
            print(f"   继续加载权重...")

    # 加载权重
    print(f"[加载] 加载权重文件...", end='', flush=True)
    try:
        loaded_data = torch.load(model_path, map_location='cpu')
        print(f" [完成]")
    except Exception as e:
        print(f" [警告]")
        print(f"   尝试使用 weights_only=False 加载...", end='', flush=True)
        loaded_data = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f" [完成]")

    # 判断加载的是状态字典还是完整模型对象
    if isinstance(loaded_data, dict):
        if 'state_dict' in loaded_data:
            state_dict = loaded_data['state_dict']
        else:
            state_dict = loaded_data
    else:
        model_to_load = loaded_data.module if hasattr(loaded_data, 'module') else loaded_data
        state_dict = model_to_load.state_dict()
    
    # 处理 DataParallel 保存的模型（键名可能带有 'module.' 前缀）
    new_state_dict = {}
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    if has_module_prefix:
        print(f"[信息] 检测到 DataParallel 模型，正在移除 'module.' 前缀")
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    # 余弦相似度增强模块已启用，保留所有权重
    # 直接使用原始状态字典，不过滤任何权重
    filtered_state_dict = state_dict
    
    # 加载状态字典（使用 strict=False 以兼容可能的结构差异）
    model.load_state_dict(filtered_state_dict, strict=False)
    
    model.to(device)
    model.eval()

    # 获取数据变换
    _, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)

    # 创建数据加载器
    test_loader = create_test_loader(args, val_transform)

    # 创建配置和数据集特定的输出目录
    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    config_output_dir = join(output_dir, dataset_name, f"config_{args.ablation_config}")
    os.makedirs(config_output_dir, exist_ok=True)

    # 创建预测掩码和 logits 热力图的单独目录(替代原 diff_masks)
    prediction_mask_dir = join(config_output_dir, 'prediction_masks')
    heatmap_dir = join(config_output_dir, 'logits_heatmaps')
    os.makedirs(prediction_mask_dir, exist_ok=True)
    os.makedirs(heatmap_dir, exist_ok=True)

    # 挂载 logits 捕获器:统一兼容配置1-7(final_pred)与配置8(up_c1[0])
    logits_capture = LogitsCapture().attach(model)

    # 配置名称映射
    config_names = {
        1: "Base Only",
        2: "Base + Attention",
        3: "Base + ASPP",
        4: "Base + Transformer",
        5: "Base + Attention + ASPP",
        6: "Base + Attention + Transformer",
        7: "Base + ASPP + Transformer",
        8: "Base + Attention + ASPP + Transformer"
    }
    config_name = config_names.get(args.ablation_config, f"config_{args.ablation_config}")

    print(f"开始可视化配置 {args.ablation_config} ({config_name})...")

    # 用于记录文件名到原始路径的映射
    path_mapping = {}

    # 执行可视化
    total_vis_count = 0
    with torch.no_grad():
        for i, batched_inputs in enumerate(test_loader):
            img, target = batched_inputs[0].to(device), batched_inputs[1].to(device)
            
            pre_img = img[:, :3, :, :]
            post_img = img[:, 3:, :, :]

            start_time = time.time()
            # 模型在评估模式下返回sigmoid激活的张量
            main_output = model(pre_img, post_img)
            end_time = time.time()

            # 预测结果
            pred = torch.where(main_output > 0.5, 1, 0)
            
            # 打印预测信息
            print(f"\n=== 配置 {args.ablation_config} - 图像 {i+1} 的预测信息 ===")
            print(f"推理时间: {end_time - start_time:.3f}s")
            
            # 预测结果统计
            pred_np = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
            target_np = target.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
            
            pred_changes = np.sum(pred_np)
            target_changes = np.sum(target_np)
            total_pixels = pred_np.size
            
            print(f"预测结果统计:")
            print(f"  - 预测变化像素数: {pred_changes}")
            print(f"  - 真实变化像素数: {target_changes}")
            print(f"  - 预测变化比例: {pred_changes/total_pixels*100:.2f}%")
            print(f"  - 真实变化比例: {target_changes/total_pixels*100:.2f}%")
            print("=" * 50)

            # 加载原始图像用于可视化
            original_pre_path = test_loader.dataset.pre_images[i]
            original_post_path = test_loader.dataset.post_images[i]
            
            pre_img_vis = io.imread(original_pre_path)
            post_img_vis = io.imread(original_post_path)

            # 调整图像尺寸以匹配输出尺寸
            output_h, output_w = pred.shape[-2], pred.shape[-1]
            pre_img_vis = cv2.resize(pre_img_vis, (output_w, output_h))
            post_img_vis = cv2.resize(post_img_vis, (output_w, output_h))

            # 转换RGB到BGR（OpenCV格式）
            pre_img_vis = cv2.cvtColor(pre_img_vis, cv2.COLOR_RGB2BGR)
            post_img_vis = cv2.cvtColor(post_img_vis, cv2.COLOR_RGB2BGR)

            # 创建Ground Truth可视化
            target_vis = np.stack([(target_np * 255)] * 3, axis=-1)
            
            # 创建预测结果掩码（二值图像，白色=变化，黑色=无变化）
            prediction_mask = np.zeros((output_h, output_w, 3), dtype=np.uint8)
            prediction_mask[(pred_np == 1)] = [255, 255, 255]

            # 捕获该配置最终 logits 层输出(sigmoid 之前),生成 jet 热力图
            # 做什么:把"差异图"替换为"最终融合预测 logits 热力图",统一展示各配置预测置信度分布
            # 为什么:logits 热力图比二值差异图更能反映模型对变化区域的响应强度,便于跨配置对比
            logits_2d = logits_capture.logits[0, 0].cpu().numpy()  # [H, W]
            logits_min, logits_max = float(logits_2d.min()), float(logits_2d.max())
            logits_heatmap = tensor_to_heatmap(logits_2d, JET_CMAP, vmin=logits_min, vmax=logits_max)
            logits_heatmap_bgr = cv2.cvtColor(logits_heatmap, cv2.COLOR_RGB2BGR)
            colorbar_img = create_colorbar_image(JET_CMAP, logits_min, logits_max, width=40, height=output_h)
            colorbar_bgr = cv2.cvtColor(colorbar_img, cv2.COLOR_RGB2BGR)

            # 创建分隔线
            spacer = np.zeros((output_h, 10, 3), dtype=np.uint8)

            # 组合所有图像：T1 | T2 | GT | Pred | Logits热力图 | Colorbar
            combined_vis = np.concatenate([
                pre_img_vis, spacer,
                post_img_vis, spacer,
                target_vis, spacer,
                prediction_mask, spacer,
                logits_heatmap_bgr, spacer,
                colorbar_bgr
            ], axis=1)

            # 添加标题信息
            title_height = 40
            title_img = np.zeros((title_height, combined_vis.shape[1], 3), dtype=np.uint8)
            title_text = f"Config {args.ablation_config} ({config_name}) - Image {i+1} - Time: {end_time - start_time:.3f}s"
            cv2.putText(title_img, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 添加图例
            legend_height = 60
            legend_img = np.zeros((legend_height, combined_vis.shape[1], 3), dtype=np.uint8)
            legend_texts = [
                "T1 Image",
                "T2 Image",
                "Ground Truth",
                "Prediction",
                "Logits Heatmap (jet)",
                f"Colorbar [{logits_min:.2f},{logits_max:.2f}]"
            ]
            x_positions = [0, output_w + 10, 2*(output_w + 10), 3*(output_w + 10), 4*(output_w + 10), 5*(output_w + 10)]
            for j, text in enumerate(legend_texts):
                if j < len(x_positions):
                    cv2.putText(legend_img, text, (x_positions[j], 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # 最终组合图像
            final_vis = np.concatenate([title_img, combined_vis, legend_img], axis=0)

            # 保存可视化结果
            original_filename = os.path.basename(original_pre_path)
            vis_filename = f"{config_name}_{original_filename}"
            save_path = join(config_output_dir, vis_filename)
            cv2.imwrite(save_path, final_vis)
            print(f"保存可视化结果到: {save_path}")

            # 单独保存预测掩码
            prediction_mask_filename = f"pred_{original_filename}"
            prediction_mask_path = join(prediction_mask_dir, prediction_mask_filename)
            cv2.imwrite(prediction_mask_path, prediction_mask)

            # 单独保存 logits 热力图(替代原差异掩码,供 combine_existing 拼接复用)
            heatmap_filename = f"heat_{original_filename}"
            heatmap_path = join(heatmap_dir, heatmap_filename)
            cv2.imwrite(heatmap_path, logits_heatmap_bgr)

            # 记录路径映射
            path_mapping[vis_filename] = {
                'visualization_file': save_path,
                'heatmap_file': heatmap_path,
                'original_pre_image': original_pre_path,
                'original_post_image': original_post_path,
                'original_label': test_loader.dataset.label_change[i] if hasattr(test_loader.dataset, 'label_change') else None,
                'image_index': i + 1,
                'dataset_name': dataset_name,
                'config_id': args.ablation_config,
                'config_name': config_name
            }

            total_vis_count += 1

    # 捕获完毕后移除 hook,释放引用(避免影响后续模型调用)
    logits_capture.detach()

    # 保存路径映射文件
    mapping_file = join(config_output_dir, 'path_mapping.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(path_mapping, f, indent=2, ensure_ascii=False)
    print(f"\n[完成] 路径映射文件已保存到: {mapping_file}")
    print(f"   共记录 {len(path_mapping)} 个图像的路径映射")

    print(f"配置 {args.ablation_config} ({config_name}) 可视化完成，共处理 {total_vis_count} 张图像。")
    return total_vis_count

@torch.no_grad()
def visualize_all_configs_combined(args, configs_to_visualize, output_dir):
    """
    将所有配置的结果拼接成横向对比图。
    
    入参:
    - args: 配置参数
    - configs_to_visualize: 要可视化的配置ID列表
    - output_dir: 可视化结果输出目录
    
    方法:
    1. 加载所有配置的模型
    2. 对每张图像运行所有配置的推理
    3. 将所有配置的预测结果和差异图横向拼接
    
    出参:
    - total_vis_count (int): 可视化的图像数量
    """
    # 检查CUDA可用性
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

    # 配置名称映射（用于显示）
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

    # 加载所有配置的模型
    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    models = {}
    model_paths = {}
    
    print(f"\n=== 加载所有配置的模型 ===")
    for config_id in configs_to_visualize:
        if config_id == 8:
            # 配置8使用完整模型的路径
            if hasattr(args, 'model_path') and args.model_path:
                model_path = args.model_path
            else:
                model_path = join(args.save_dir, dataset_name, 'best_model.pth')
                if not os.path.exists(model_path):
                    model_path = join(args.save_dir, dataset_name, 'final_model.pth')
        else:
            # 配置1-7使用消融实验模型的路径
            model_path = join(args.save_dir, dataset_name, f"config_{config_id}_config_{config_id}", dataset_name, 'best_model.pth')
        model_paths[config_id] = model_path
        
        if not os.path.exists(model_path):
            print(f"[警告] 未找到 {model_path} 文件；跳过配置 {config_id}。")
            continue
        
        print(f"[加载] 配置 {config_id} ({config_names.get(config_id, f'config_{config_id}')})...", end='', flush=True)
        
        # 创建模型
        config_args = copy.deepcopy(args)
        config_args.ablation_config = config_id
        if config_id == 8:
            # 配置8使用完整的Trainer模型（所有模块都启用）
            model = Trainer(args=config_args)
        else:
            # 配置1-7使用消融实验模型
            model = create_ablation_model(config_args, experiment_id=config_id)
        
        # 初始化动态模块
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, args.in_height, args.in_width)
            try:
                _ = model(dummy_input, dummy_input)
            except Exception as e:
                print(f" [警告: {e}]", end='', flush=True)
        
        # 加载权重
        try:
            loaded_data = torch.load(model_path, map_location='cpu')
        except Exception as e:
            loaded_data = torch.load(model_path, map_location='cpu', weights_only=False)
        
        if isinstance(loaded_data, dict):
            state_dict = loaded_data.get('state_dict', loaded_data)
        else:
            model_to_load = loaded_data.module if hasattr(loaded_data, 'module') else loaded_data
            state_dict = model_to_load.state_dict()
        
        # 处理 DataParallel
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
        models[config_id] = model
        print(f" [完成]")

    if not models:
        print("[错误] 没有成功加载任何模型，退出。")
        return 0

    print(f"成功加载 {len(models)} 个配置的模型。")

    # 为每个模型挂载 logits 捕获器(兼容配置1-7 final_pred 与配置8 up_c1[0])
    logits_caps = {cid: LogitsCapture().attach(m) for cid, m in models.items()}
    
    # 获取数据变换和加载器
    _, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)
    test_loader = create_test_loader(args, val_transform)
    
    # 创建输出目录
    combined_output_dir = join(output_dir, dataset_name, 'all_configs_combined')
    os.makedirs(combined_output_dir, exist_ok=True)
    
    # 执行可视化
    total_vis_count = 0
    with torch.no_grad():
        for i, batched_inputs in enumerate(test_loader):
            img, target = batched_inputs[0].to(device), batched_inputs[1].to(device)
            pre_img = img[:, :3, :, :]
            post_img = img[:, 3:, :, :]
            
            # 加载原始图像用于可视化
            original_pre_path = test_loader.dataset.pre_images[i]
            original_post_path = test_loader.dataset.post_images[i]
            
            pre_img_vis = io.imread(original_pre_path)
            post_img_vis = io.imread(original_post_path)
            
            # 获取输出尺寸
            target_np = target.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
            output_h, output_w = target_np.shape
            
            # 调整图像尺寸
            pre_img_vis = cv2.resize(pre_img_vis, (output_w, output_h))
            post_img_vis = cv2.resize(post_img_vis, (output_w, output_h))
            pre_img_vis = cv2.cvtColor(pre_img_vis, cv2.COLOR_RGB2BGR)
            post_img_vis = cv2.cvtColor(post_img_vis, cv2.COLOR_RGB2BGR)
            
            # 创建Ground Truth可视化
            target_vis = np.stack([(target_np * 255)] * 3, axis=-1)
            
            # 存储所有配置的预测结果和 logits 热力图
            config_predictions = {}
            config_heatmaps = {}
            inference_times = {}

            # 第一轮推理:收集各配置 logits 数组,用于计算跨配置共享的归一化范围
            config_logits = {}
            for config_id in sorted(models.keys()):
                model = models[config_id]
                start_time = time.time()
                main_output = model(pre_img, post_img)
                end_time = time.time()
                inference_times[config_id] = end_time - start_time

                pred = torch.where(main_output > 0.5, 1, 0)
                pred_np = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
                prediction_mask = np.zeros((output_h, output_w, 3), dtype=np.uint8)
                prediction_mask[(pred_np == 1)] = [255, 255, 255]
                config_predictions[config_id] = prediction_mask

                # 捕获最终 logits 层输出(sigmoid 前),作为热力图数据源
                config_logits[config_id] = logits_caps[config_id].logits[0, 0].cpu().numpy()

            # 跨配置共享同一 logits 归一化范围,保证横向热力图颜色可比
            all_logits = np.stack([config_logits[c] for c in sorted(config_logits.keys())])
            global_min, global_max = float(all_logits.min()), float(all_logits.max())
            for config_id in sorted(config_logits.keys()):
                heat_rgb = tensor_to_heatmap(config_logits[config_id], JET_CMAP, vmin=global_min, vmax=global_max)
                config_heatmaps[config_id] = cv2.cvtColor(heat_rgb, cv2.COLOR_RGB2BGR)

            # 创建分隔线
            spacer = np.zeros((output_h, 10, 3), dtype=np.uint8)

            # 组合所有图像：T1 | T2 | GT | (Config1 Pred | Config1 Heat | Config2 Pred | Config2 Heat | ...) | Colorbar
            combined_parts = [pre_img_vis, spacer, post_img_vis, spacer, target_vis, spacer]

            # 添加每个配置的预测结果和 logits 热力图
            for config_id in sorted(config_predictions.keys()):
                combined_parts.extend([config_predictions[config_id], spacer, config_heatmaps[config_id], spacer])

            # 末尾追加共享 colorbar,移除其前的多余分隔线
            shared_colorbar = cv2.cvtColor(
                create_colorbar_image(JET_CMAP, global_min, global_max, width=40, height=output_h),
                cv2.COLOR_RGB2BGR)
            combined_parts[-1] = shared_colorbar
            
            # 水平拼接所有图像
            combined_vis = np.concatenate(combined_parts, axis=1)
            
            # 添加标题信息
            title_height = 40
            title_img = np.zeros((title_height, combined_vis.shape[1], 3), dtype=np.uint8)
            title_text = f"Image {i+1} - All Configurations Comparison"
            cv2.putText(title_img, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 添加图例
            legend_height = 80
            legend_img = np.zeros((legend_height, combined_vis.shape[1], 3), dtype=np.uint8)
            
            # 基础图例
            legend_texts = ["T1 Image", "T2 Image", "Ground Truth"]
            x_positions = [0, output_w + 10, 2*(output_w + 10)]
            for j, text in enumerate(legend_texts):
                if j < len(x_positions):
                    cv2.putText(legend_img, text, (x_positions[j], 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 添加配置图例
            current_x = 3*(output_w + 10)
            for config_id in sorted(config_predictions.keys()):
                config_name = config_names.get(config_id, f"Config{config_id}")
                pred_text = f"{config_name} Pred"
                heat_text = f"{config_name} Heat"

                cv2.putText(legend_img, pred_text, (current_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(legend_img, heat_text, (current_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                # 添加推理时间
                time_text = f"{inference_times[config_id]:.3f}s"
                cv2.putText(legend_img, time_text, (current_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

                current_x += 2*(output_w + 10)

            # 热力图 colorbar 范围说明(替代原差异图颜色说明)
            cv2.putText(legend_img, "Logits Heatmap:", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(legend_img, f"[{global_min:.2f},{global_max:.2f}] Red=High, Blue=Low", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 最终组合图像
            final_vis = np.concatenate([title_img, combined_vis, legend_img], axis=0)
            
            # 保存可视化结果
            original_filename = os.path.basename(original_pre_path)
            vis_filename = f"combined_{original_filename}"
            save_path = join(combined_output_dir, vis_filename)
            cv2.imwrite(save_path, final_vis)
            
            print(f"保存组合可视化结果到: {save_path}")
            total_vis_count += 1
    
    # 清理 hook 与模型以释放显存
    for cap in logits_caps.values():
        cap.detach()
    del models
    torch.cuda.empty_cache()
    
    print(f"\n所有配置组合可视化完成，共处理 {total_vis_count} 张图像。")
    return total_vis_count

def find_reddest_region(img, box_size=30):
    """
    找到图片中红色像素最多的区域（用于差异图标注）
    
    入参:
    - img: PIL Image对象
    - box_size: 区域大小（默认30x30）
    
    方法:
    - 遍历所有可能的区域，计算红色通道总和
    - 返回红色最多的区域左上角坐标
    
    出参:
    - (x, y): 红色最多区域的左上角坐标
    """
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    height, width = img_array.shape[:2]
    
    box_size = min(box_size, width, height)
    
    max_red_sum = -1
    best_x, best_y = 0, 0
    
    for y in range(height - box_size + 1):
        for x in range(width - box_size + 1):
            region = img_array[y:y+box_size, x:x+box_size]
            red_sum = np.sum(region[:, :, 0])  # R通道
            
            if red_sum > max_red_sum:
                max_red_sum = red_sum
                best_x, best_y = x, y
    
    return best_x, best_y

def draw_dashed_rectangle(img, x, y, width, height, color=(255, 0, 0), dash_length=5, gap_length=2):
    """
    在图片上画虚线矩形
    
    入参:
    - img: PIL Image对象
    - x, y: 矩形左上角坐标
    - width, height: 矩形宽度和高度
    - color: 线条颜色（默认红色）
    - dash_length: 虚线段的长度
    - gap_length: 间隔的长度
    
    方法:
    - 绘制四条虚线边
    
    出参:
    - 无返回值，直接修改图像
    """
    draw = ImageDraw.Draw(img)
    
    # 绘制上边
    current_x = x
    while current_x < x + width:
        end_x = min(current_x + dash_length, x + width)
        draw.line([(current_x, y), (end_x, y)], fill=color, width=1)
        current_x = end_x + gap_length
    
    # 绘制下边
    current_x = x
    while current_x < x + width:
        end_x = min(current_x + dash_length, x + width)
        draw.line([(current_x, y + height - 1), (end_x, y + height - 1)], fill=color, width=1)
        current_x = end_x + gap_length
    
    # 绘制左边
    current_y = y
    while current_y < y + height:
        end_y = min(current_y + dash_length, y + height)
        draw.line([(x, current_y), (x, end_y)], fill=color, width=1)
        current_y = end_y + gap_length
    
    # 绘制右边
    current_y = y
    while current_y < y + height:
        end_y = min(current_y + dash_length, y + height)
        draw.line([(x + width - 1, current_y), (x + width - 1, end_y)], fill=color, width=1)
        current_y = end_y + gap_length

def add_red_region_marker(img, x=None, y=None, box_size=30):
    """
    在图片上标注红色区域
    
    入参:
    - img: PIL Image对象
    - x, y: 红色区域的左上角坐标（如果为None，则自动查找）
    - box_size: 框的大小（默认30x30）
    
    方法:
    - 如果没有提供坐标，自动查找红色最多区域
    - 绘制虚线矩形框
    
    出参:
    - 标注后的PIL Image对象
    """
    img_copy = img.copy()
    
    if x is None or y is None:
        x, y = find_reddest_region(img_copy, box_size=box_size)
    
    draw_dashed_rectangle(img_copy, x, y, box_size, box_size, color=(255, 0, 0), dash_length=8, gap_length=3)
    
    return img_copy

def combine_existing_visualizations(args, configs_to_visualize, vis_results_dir):
    """
    从已生成的热力图目录读取独立热力图,拼接成横向对比图(PIL 样式)。

    入参:
    - args: 配置参数
    - configs_to_visualize: 要拼接的配置ID列表
    - vis_results_dir: 可视化结果根目录(例如:./vis_results_ablation)

    方法:
    1. 从每个配置的 logits_heatmaps/heat_{filename} 读取独立热力图
    2. 从首个可用配置的组合图({config_name}_{filename})切分提取 T1/T2/GT(标题40px、图例60px、6列内容)
    3. 用 PIL 统一缩放后横向拼接:T1|T2|GT|各配置热力图

    出参:
    - total_vis_count (int): 拼接的图像数量
    """
    import glob

    # 配置名称映射(用于匹配组合图文件名前缀)
    config_names_for_match = {
        1: "Base Only",
        2: "Base + Attention",
        3: "Base + ASPP",
        4: "Base + Transformer",
        5: "Base + Attention + ASPP",
        6: "Base + Attention + Transformer",
        7: "Base + ASPP + Transformer",
        8: "Base + Attention + ASPP + Transformer"
    }

    # 图像处理参数(可配置)
    if hasattr(args, 'image_size') and isinstance(args.image_size, list) and len(args.image_size) == 2:
        image_size = tuple(args.image_size)
    else:
        image_size = (90, 90)
    spacing = getattr(args, 'spacing', 0)

    dataset_name = os.path.basename(os.path.normpath(args.file_root)) if args.file_root else 'LBFD-CD'
    print(f"\n=== 从已生成的热力图拼接对比图 ===")
    print(f"数据集: {dataset_name}, 配置: {configs_to_visualize}, 目录: {vis_results_dir}")
    print(f"图像尺寸: {image_size}, 间距: {spacing}")

    # 收集每个配置的独立热力图(heat_*.png)与组合图({config_name}_*.png)
    config_heatmaps = {}
    config_combined = {}
    for config_id in configs_to_visualize:
        config_dir = join(vis_results_dir, dataset_name, f"config_{config_id}")
        heat_dir = join(config_dir, 'logits_heatmaps')
        if os.path.isdir(heat_dir):
            heat_files = glob.glob(join(heat_dir, 'heat_*.png')) + glob.glob(join(heat_dir, 'heat_*.PNG'))
            config_heatmaps[config_id] = sorted(heat_files)

        if os.path.isdir(config_dir):
            cfg_name = config_names_for_match.get(config_id, f"Config{config_id}")
            combined_files = []
            for f in glob.glob(join(config_dir, '*.png')) + glob.glob(join(config_dir, '*.PNG')):
                if os.path.isfile(f) and os.path.basename(f).startswith(cfg_name + "_"):
                    combined_files.append(f)
            config_combined[config_id] = sorted(combined_files)

    if not config_heatmaps:
        print("[错误] 未在任何配置的 logits_heatmaps/ 下找到热力图,请先运行 visualize_single_config。")
        return 0

    # 提取热力图基础文件名(去除 heat_ 前缀),作为拼接主键
    all_basenames = set()
    for files in config_heatmaps.values():
        for fpath in files:
            all_basenames.add(os.path.basename(fpath)[len('heat_'):])
    print(f"\n找到 {len(all_basenames)} 张不同的图像")

    combined_output_dir = join(vis_results_dir, dataset_name, 'all_configs_combined')
    os.makedirs(combined_output_dir, exist_ok=True)

    total_vis_count = 0
    for basename in sorted(all_basenames):
        resized_images = []
        t1_img = t2_img = gt_img = None

        # 从首个可用配置的组合图提取 T1/T2/GT(组合图为6列:标题40+内容+图例60)
        for config_id in sorted(config_combined.keys()):
            cfg_name = config_names_for_match.get(config_id, f"Config{config_id}")
            combined_path = join(vis_results_dir, dataset_name, f"config_{config_id}", f"{cfg_name}_{basename}")
            if not os.path.exists(combined_path):
                continue
            combined_cv = cv2.imread(combined_path)
            if combined_cv is None:
                continue
            ih, iw = combined_cv.shape[:2]
            title_h, legend_h = 40, 60
            content = cv2.cvtColor(combined_cv[title_h:ih - legend_h, :], cv2.COLOR_BGR2RGB)
            # 6 列内容(T1|T2|GT|Pred|Heat|Colorbar),5 个 10px 分隔
            part_w = (iw - 5 * 10) // 6
            t1_img = Image.fromarray(content[:, 0:part_w])
            t2_img = Image.fromarray(content[:, part_w + 10:2 * part_w + 10])
            gt_img = Image.fromarray(content[:, 2 * (part_w + 10):3 * part_w + 2 * 10])
            break

        if t1_img is not None:
            resized_images.extend([
                t1_img.resize(image_size, Image.Resampling.LANCZOS),
                t2_img.resize(image_size, Image.Resampling.LANCZOS),
                gt_img.resize(image_size, Image.Resampling.LANCZOS),
            ])

        # 追加每个配置的热力图
        for config_id in sorted(config_heatmaps.keys()):
            heat_path = join(vis_results_dir, dataset_name, f"config_{config_id}", 'logits_heatmaps', f'heat_{basename}')
            if not os.path.exists(heat_path):
                continue
            heat_cv = cv2.imread(heat_path)
            if heat_cv is None:
                continue
            heat_rgb = Image.fromarray(cv2.cvtColor(heat_cv, cv2.COLOR_BGR2RGB))
            resized_images.append(heat_rgb.resize(image_size, Image.Resampling.LANCZOS))

        if not resized_images:
            continue

        total_width = len(resized_images) * image_size[0] + spacing * (len(resized_images) - 1)
        result_image = Image.new('RGB', (total_width, image_size[1]), color='white')
        x_offset = 0
        for img in resized_images:
            result_image.paste(img, (x_offset, 0))
            x_offset += image_size[0] + spacing

        output_path = join(combined_output_dir, f"combined_{basename}")
        result_image.save(output_path)
        print(f"保存组合可视化结果到: {output_path}")
        total_vis_count += 1

    print(f"\n所有配置组合拼接完成，共处理 {total_vis_count} 张图像。")
    return total_vis_count

def visualize_ablation_experiment(args):
    """
    运行单个消融实验配置的可视化。
    
    入参:
    - args: 配置参数，必须包含 file_root, save_dir, ablation_config
    
    方法:
    1. 根据数据集路径和配置ID构建模型路径
    2. 调用可视化函数处理单个配置
    
    出参:
    - vis_count (int): 可视化的图像数量
    """
    # 创建模型路径
    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    if args.ablation_config == 8:
        # 配置8使用完整模型的路径（通常在exp_new目录下）
        # 如果save_dir指定了完整模型路径，使用它；否则使用默认路径
        if hasattr(args, 'model_path') and args.model_path:
            model_path = args.model_path
        else:
            # 默认路径：exp_new/{dataset_name}/best_model.pth 或 final_model.pth
            model_path = join(args.save_dir, dataset_name, 'best_model.pth')
            if not os.path.exists(model_path):
                model_path = join(args.save_dir, dataset_name, 'final_model.pth')
    else:
        # 配置1-7使用消融实验模型的路径
        model_path = join(args.save_dir, dataset_name, f"config_{args.ablation_config}_config_{args.ablation_config}", dataset_name, 'best_model.pth')
    
    print(f"\n=== 开始可视化消融实验配置 {args.ablation_config} ===")
    print(f"数据集: {dataset_name}")
    print(f"模型路径: {model_path}")
    print(f"输出目录: {args.output_dir}")
    
    # 可视化
    vis_count = visualize_single_config(args, model_path, args.output_dir)
    
    if vis_count is not None:
        print(f"配置 {args.ablation_config} 可视化完成，处理了 {vis_count} 张图像")
        return vis_count
    else:
        print(f"配置 {args.ablation_config} 可视化失败。")
        return 0

# 数据集配置
DATASETS = [
    {
        'name': 'GVLM-CD',
        'path': r'E:\rqx\dataes\GVLM-CD',
        'description': 'GVLM变化检测数据集'
    },
    {
        'name': 'WHU-CD',
        'path': r'E:\rqx\dataes\WHU-CD',
        'description': 'WHU建筑变化检测数据集'
    },
    {
        'name': 'LBFD-CD',
        'path': r'E:\rqx\dataes\LBFD-CD',
        'description': 'LBFD变化检测数据集'
    }
]

# 消融实验配置映射
ABLATION_CONFIGS = {
    1: "Base Only",
    2: "Base + Attention",
    3: "Base + ASPP", 
    4: "Base + Transformer",
    5: "Base + Attention + ASPP",
    6: "Base + Attention + Transformer",
    7: "Base + ASPP + Transformer",
    8: "Base + Attention + ASPP + Transformer"
}


def visualize_all_datasets(args):
    """
    在所有数据集上可视化指定配置
    
    入参:
    - args: 配置参数，必须包含 ablation_config, save_dir
    
    方法:
    1. 遍历所有数据集
    2. 为每个数据集调用可视化函数
    
    出参:
    - results (dict): 每个数据集的可视化结果
    """
    print(f"\n=== 开始在所有数据集上可视化配置 {args.ablation_config} ===")
    
    # 创建主输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    config_name = ABLATION_CONFIGS.get(args.ablation_config, f"config_{args.ablation_config}")
    print(f"配置: {args.ablation_config} ({config_name})")
    
    results = {}
    total_images = 0
    
    for dataset_config in DATASETS:
        dataset_name = dataset_config['name']
        dataset_path = dataset_config['path']
        
        print(f"\n{'='*60}")
        print(f"处理数据集: {dataset_name}")
        print(f"数据集路径: {dataset_path}")
        print(f"{'='*60}")
        
        # 复制参数以避免相互影响
        dataset_args = copy.deepcopy(args)
        dataset_args.file_root = dataset_path
        
        try:
            vis_count = visualize_ablation_experiment(dataset_args)
            results[dataset_name] = vis_count
            total_images += vis_count
        except Exception as e:
            print(f"数据集 {dataset_name} 可视化失败: {e}")
            import traceback
            traceback.print_exc()
            results[dataset_name] = 0
    
    # 生成可视化结果报告
    report_path = join(args.output_dir, f"visualization_report_config_{args.ablation_config}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 消融实验可视化结果报告 ===\n")
        f.write(f"配置: {args.ablation_config} ({config_name})\n")
        f.write(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总处理图像数: {total_images}\n\n")
        
        f.write("数据集\t处理图像数\t状态\n")
        for dataset_name, vis_count in results.items():
            status = "成功" if vis_count > 0 else "失败"
            f.write(f"{dataset_name}\t{vis_count}\t{status}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("配置说明:\n")
        f.write(f"{config_name}\n")
        f.write("\n可视化说明:\n")
        f.write("- 每张图像包含5个子图：T1图像、T2图像、真实标签、预测结果、差异图\n")
        f.write("- 差异图颜色说明：绿色=正确预测变化，红色=误报，蓝色=漏报\n")

    print(f"\n=== 所有数据集可视化完成 ===")
    print(f"总处理图像数: {total_images}")
    print(f"可视化结果报告已保存到: {report_path}")
    print(f"可视化结果保存在: {args.output_dir}")


def visualize_all_ablation_experiments(args):
    """
    运行所有消融实验配置的可视化。
    
    入参:
    - args: 配置参数，必须包含 file_root, save_dir
    
    方法:
    1. 确定要可视化的配置列表
    2. 为每个配置调用可视化函数
    
    出参:
    - None
    """
    print(f"\n=== 开始运行消融实验配置的可视化 ===")
    
    # 确定要可视化的配置列表
    if args.configs is not None:
        configs_to_visualize = [int(x.strip()) for x in args.configs.split(',')]
        print(f"指定可视化的配置: {configs_to_visualize}")
    else:
        configs_to_visualize = [1, 2, 3, 4, 7, 8]  # 默认配置：X3D, X3D+A, X3D+B, X3D+C, X3D+B+C, X3D+A+B+C
        print("可视化默认配置: [1, 2, 3, 4, 7, 8]")
    
    # 创建主输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    total_images = 0
    
    for config_id in configs_to_visualize:
        # 复制参数以避免相互影响
        config_args = copy.deepcopy(args)
        config_args.ablation_config = config_id
        
        try:
            vis_count = visualize_ablation_experiment(config_args)
            results[config_id] = vis_count
            total_images += vis_count
        except Exception as e:
            print(f"配置 {config_id} 可视化失败: {e}")
            results[config_id] = 0
    
    # 生成可视化结果报告
    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    report_path = join(args.output_dir, f"visualization_report_{dataset_name}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 消融实验可视化结果报告 ===\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总处理图像数: {total_images}\n\n")
        
        f.write("配置\t模型名称\t处理图像数\t状态\n")
        for config_id in configs_to_visualize:
            vis_count = results.get(config_id, 0)
            config_name = ABLATION_CONFIGS.get(config_id, f"配置{config_id}")
            status = "成功" if vis_count > 0 else "失败"
            f.write(f"{config_id}\t{config_name}\t{vis_count}\t{status}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("详细配置说明:\n")
        f.write("1: Base Only\n")
        f.write("2: Base + Attention\n") 
        f.write("3: Base + ASPP\n")
        f.write("4: Base + Transformer\n")
        f.write("5: Base + Attention + ASPP\n")
        f.write("6: Base + Attention + Transformer\n")
        f.write("7: Base + ASPP + Transformer\n")
        f.write("8: Base + Attention + ASPP + Transformer (完整模型)\n")
        f.write("\n可视化说明:\n")
        f.write("- 每张图像包含5个子图：T1图像、T2图像、真实标签、预测结果、差异图\n")
        f.write("- 差异图颜色说明：绿色=正确预测变化，红色=误报，蓝色=漏报\n")

    print(f"\n=== 所有消融实验可视化完成 ===")
    print(f"总处理图像数: {total_images}")
    print(f"可视化结果报告已保存到: {report_path}")
    print(f"可视化结果保存在: {args.output_dir}")

def get_parser():
    """
    创建脚本的参数解析器。
    """
    parser = argparse.ArgumentParser(description='消融实验可视化脚本')
    
    # 消融实验特定参数
    parser.add_argument('--ablation_config', type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8], 
                       help='消融实验配置ID: 1-8 (8为完整模型X3D+A+B+C)')
    parser.add_argument('--model_path', type=str, default=None, 
                       help='模型权重路径（配置8时必需，用于指定完整模型的路径）。')
    parser.add_argument('--run_all_configs', action='store_true',
                       help='运行所有消融实验配置的可视化（需要指定file_root）')
    parser.add_argument('--run_all_datasets', action='store_true',
                       help='在所有数据集上运行指定配置的可视化（需要指定ablation_config）')
    parser.add_argument('--configs', type=str, default=None, help='指定要可视化的配置ID列表，用逗号分隔，如"5,6,7"（需要指定file_root）。')
    parser.add_argument('--combine_all_configs', action='store_true',
                       help='将所有配置的结果拼接成一张横向对比图（需要指定file_root）')
    parser.add_argument('--combine_existing', action='store_true',
                       help='从已生成的可视化结果目录读取图像并拼接（需要指定vis_results_dir）')
    parser.add_argument('--vis_results_dir', type=str, default='./vis_results_ablation',
                       help='已生成的可视化结果根目录（用于--combine_existing模式）')
    
    # 数据集和路径参数
    parser.add_argument('--file_root', type=str, default=None, help='数据集根目录路径（单个数据集可视化时必需）。')
    parser.add_argument('--save_dir', type=str, default='./exp_ablation_batch', help='实验保存的根目录（默认: ./exp_ablation_batch）。')
    parser.add_argument('--output_dir', type=str, default='./vis_results_ablation', help='可视化结果输出目录。')
    parser.add_argument('--gpu_id', type=str, default='0', help='使用的GPU ID。')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载的工作进程数（Windows环境建议使用0避免pickle错误）。')
    parser.add_argument('--in_height', type=int, default=256, help='RGB图像高度')
    parser.add_argument('--in_width', type=int, default=256, help='RGB图像宽度')

    # 模型特定参数
    parser.add_argument('--num_perception_frame', type=int, default=1, help='感知帧数量（当前架构必须为1）')
    parser.add_argument('--pretrained', default=r'model\X3D_L.pyth', type=str, help='预训练X3D权重路径')
    
    # 可视化参数
    parser.add_argument('--batch_size', type=int, default=1, help='可视化批次大小（建议设为1）。')
    
    # 图像拼接样式参数（参考batch_arrange_images.py）
    parser.add_argument('--image_size', type=int, nargs=2, default=[90, 90], metavar=('WIDTH', 'HEIGHT'),
                       help='拼接图像的尺寸（宽度 高度），默认90x90')
    parser.add_argument('--spacing', type=int, default=0, help='图像之间的间距（像素），默认0（缩小一半）')
    parser.add_argument('--add_markers', action='store_true',
                       help='是否在差异图上添加红色区域标注（虚线框）')

    return parser

def main():
    """
    主函数：解析参数并开始消融实验可视化。
    """
    parser = get_parser()
    args = parser.parse_args()
    
    # 检查参数
    if args.combine_existing:
        # 从已生成的可视化结果拼接
        if args.vis_results_dir is None:
            parser.error("--combine_existing 需要指定 --vis_results_dir")
        # 确定要拼接的配置列表
        if args.configs is not None:
            configs_to_visualize = [int(x.strip()) for x in args.configs.split(',')]
        else:
            # 默认拼接6个配置：X3D, X3D+A, X3D+B, X3D+C, X3D+B+C, X3D+A+B+C
            configs_to_visualize = [1, 2, 3, 4, 7, 8]
        print(f"将从已生成的可视化结果拼接以下配置: {configs_to_visualize}")
        combine_existing_visualizations(args, configs_to_visualize, args.vis_results_dir)
    elif args.combine_all_configs:
        # 将所有配置的结果拼接成横向对比图
        if args.file_root is None:
            parser.error("--combine_all_configs 需要指定 --file_root")
        # 确定要可视化的配置列表
        if args.configs is not None:
            configs_to_visualize = [int(x.strip()) for x in args.configs.split(',')]
        else:
            configs_to_visualize = [1, 2, 3, 4, 7, 8]  # 默认配置：X3D, X3D+A, X3D+B, X3D+C, X3D+B+C, X3D+A+B+C
        print(f"将拼接以下配置的结果: {configs_to_visualize}")
        visualize_all_configs_combined(args, configs_to_visualize, args.output_dir)
    elif args.run_all_datasets:
        # 在所有数据集上可视化指定配置
        if args.ablation_config is None:
            parser.error("--run_all_datasets 需要指定 --ablation_config")
        visualize_all_datasets(args)
    elif args.run_all_configs or args.configs is not None:
        # 在单个数据集上可视化多个配置
        if args.file_root is None:
            parser.error("--run_all_configs 或 --configs 需要指定 --file_root")
        visualize_all_ablation_experiments(args)
    elif args.ablation_config is not None:
        # 在单个数据集上可视化单个配置
        if args.file_root is None:
            parser.error("单个配置可视化需要指定 --file_root")
        if args.ablation_config == 8 and args.model_path is None:
            parser.error("配置8（完整模型）需要指定 --model_path")
        visualize_ablation_experiment(args)
    else:
        parser.error("必须指定以下之一：\n"
                    "  --ablation_config + --file_root (单个配置)\n"
                    "  --run_all_configs + --file_root (所有配置)\n"
                    "  --configs + --file_root (指定配置)\n"
                    "  --combine_all_configs + --file_root (拼接所有配置)\n"
                    "  --run_all_datasets + --ablation_config (所有数据集)")

if __name__ == '__main__':
    main()

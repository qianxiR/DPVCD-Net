# Copyright (c) Duowang Zhu.
# All rights reserved.

import os
import sys
import time
import copy
import argparse
from os.path import join

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from skimage import io

# 插入当前路径以导入本地模块
sys.path.insert(0, '.')

import data.dataset as RSDataset
import data.transforms as RSTransforms
from model.trainer_ablation import TrainerAblation, create_ablation_model
from model.trainer import Trainer

'''
按图片组织的消融实验可视化脚本

使用示例：

# 可视化所有配置（默认）
python scripts/visualize/visualize_ablation_by_image.py --file_root "E:\rqx\dataes\LBFD-CD" --save_dir ./exp_ablation_batch --output_dir ./vis_results_by_image

# 可视化指定配置
python scripts/visualize/visualize_ablation_by_image.py --file_root "E:\rqx\dataes\WHU-CD" --save_dir ./exp_ablation_batch --output_dir ./vis_results_by_image --configs "5,6,7"

# PowerShell 多行命令格式
python scripts/visualize/visualize_ablation_by_image.py `
    --file_root "E:\rqx\dataes\LBFD-CD" \
    --save_dir ./exp_ablation_batch `
    --output_dir ./vis_results_by_image `
    --configs "1,2,3,4,5,6,7" `
    --gpu_id 0
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

def load_model_for_config(args, config_id):
    """
    为指定配置加载模型。
    
    Args:
        args: 配置参数
        config_id: 配置ID (0-7为消融实验，8为完整模型)
        
    Returns:
        model: 加载的模型
    """
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    
    # 配置映射：
    # 配置0: X3D（base）-> experiment_id=1 (Base Only)
    # 配置1: X3D + A -> experiment_id=2 (Base + Attention)
    # 配置2: X3D + B -> experiment_id=3 (Base + ASPP)
    # 配置3: X3D + C -> experiment_id=4 (Base + Transformer)
    # 配置4: X3D + A+B -> experiment_id=5 (Base + Attention + ASPP)
    # 配置5: X3D + A+C -> experiment_id=6 (Base + Attention + Transformer)
    # 配置6: X3D + B+C -> experiment_id=7 (Base + ASPP + Transformer)
    # 配置7: X3D + A+B+C -> 完整模型（Trainer，所有模块）
    
    if config_id == 7:
        # 配置7使用完整的Trainer模型（所有模块都启用）
        model = Trainer(args=args)
        # 加载完整模型的权重
        if hasattr(args, 'model_path') and args.model_path:
            # 如果指定了model_path，使用它
            model_path = args.model_path
        else:
            # 否则使用默认的完整模型路径
            model_path = join(args.save_dir, dataset_name, 'best_model.pth')
            if not os.path.exists(model_path):
                model_path = join(args.save_dir, dataset_name, 'final_model.pth')
    else:
        # 配置0-6使用消融实验模型（experiment_id = config_id + 1）
        experiment_id = config_id + 1
        model = create_ablation_model(args, experiment_id=experiment_id)
        
        # 创建实验目录路径
        exp_dir = join(args.save_dir, dataset_name, f"config_{experiment_id}_config_{experiment_id}", dataset_name)
        model_path = join(exp_dir, 'best_model.pth')

    # 多GPU支持（仅在CUDA可用时）
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    model.to(device)

    # 对于配置3, 5, 6，执行一次模拟前向推理以初始化动态参数
    # 配置7和8使用完整模型，不需要初始化
    if config_id in [3, 5, 6]:
        print(f"为配置 {config_id} 执行一次前向推理以初始化动态参数...")
        model.eval()
        with torch.no_grad():
            dummy_pre_img = torch.randn(1, 3, args.in_height, args.in_width, device=device)
            dummy_post_img = torch.randn(1, 3, args.in_height, args.in_width, device=device)
            _ = model(dummy_pre_img, dummy_post_img)
        print("前向推理完成。")

    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"[警告] 未找到 {model_path} 文件；跳过配置 {config_id}。")
        return None

    try:
        # CPU上加载模型以避免GPU显存问题
        loaded_data = torch.load(model_path, map_location='cpu')
    except Exception:
        print("使用 `weights_only=False` 加载以兼容旧模型文件。")
        loaded_data = torch.load(model_path, weights_only=False, map_location='cpu')

    # 判断加载的是状态字典还是完整模型对象
    if isinstance(loaded_data, dict):
        if 'state_dict' in loaded_data:
            state_dict = loaded_data['state_dict']
        else:
            state_dict = loaded_data
    else:
        temp_model = loaded_data.module if hasattr(loaded_data, 'module') else loaded_data
        state_dict = temp_model.state_dict()
    
    # 处理 DataParallel 保存的模型
    new_state_dict = {}
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    if has_module_prefix:
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    # 加载模型权重
    if hasattr(model, 'module'):
        model.module.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model

def create_difference_visualization(pred_np, target_np):
    """
    创建预测结果与真实标签的差异图。
    
    Args:
        pred_np: 预测结果numpy数组
        target_np: 真实标签numpy数组
        
    Returns:
        diff_vis: 差异图可视化
    """
    output_h, output_w = pred_np.shape
    diff_vis = np.zeros((output_h, output_w, 3), dtype=np.uint8)
    
    # 绿色：正确预测的变化 (True Positive)
    correct_change = (pred_np == 1) & (target_np == 1)
    diff_vis[correct_change] = [0, 255, 0]
    
    # 红色：误报 (False Positive)
    false_positive = (pred_np == 1) & (target_np == 0)
    diff_vis[false_positive] = [0, 0, 255]
    
    # 蓝色：漏报 (False Negative)
    false_negative = (pred_np == 0) & (target_np == 1)
    diff_vis[false_negative] = [255, 0, 0]
    
    return diff_vis

@torch.no_grad()
def visualize_all_configs_for_image(args, configs_to_visualize, test_loader, image_idx):
    """
    为单张图像可视化所有配置的预测结果。
    
    Args:
        args: 配置参数
        configs_to_visualize: 要可视化的配置列表
        test_loader: 测试数据加载器
        image_idx: 图像索引
        
    Returns:
        success: 是否成功
    """
    # 检查CUDA可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 获取数据变换
    _, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)
    
    # 配置名称映射
    config_names = {
        0: "X3D",
        1: "X3D+A", 
        2: "X3D+B",
        3: "X3D+C",
        4: "X3D+A+B",
        5: "X3D+A+C",
        6: "X3D+B+C",
        7: "X3D+A+B+C"
    }
    
    # 获取当前图像数据
    batched_inputs = list(test_loader)[image_idx]
    img, target = batched_inputs[0].to(device), batched_inputs[1].to(device)
    
    pre_img = img[:, :3, :, :]
    post_img = img[:, 3:, :, :]
    
    # 加载原始图像用于可视化
    original_pre_path = test_loader.dataset.pre_images[image_idx]
    original_post_path = test_loader.dataset.post_images[image_idx]
    
    pre_img_vis = io.imread(original_pre_path)
    post_img_vis = io.imread(original_post_path)
    
    # 获取真实标签
    target_np = target.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
    output_h, output_w = target_np.shape
    
    # 调整图像尺寸以匹配输出尺寸
    pre_img_vis = cv2.resize(pre_img_vis, (output_w, output_h))
    post_img_vis = cv2.resize(post_img_vis, (output_w, output_h))
    
    # 转换RGB到BGR（OpenCV格式）
    pre_img_vis = cv2.cvtColor(pre_img_vis, cv2.COLOR_RGB2BGR)
    post_img_vis = cv2.cvtColor(post_img_vis, cv2.COLOR_RGB2BGR)
    
    # 创建Ground Truth可视化
    target_vis = np.stack([(target_np * 255)] * 3, axis=-1)
    
    # 存储所有配置的预测结果
    config_predictions = {}
    config_differences = {}
    inference_times = {}
    
    # 为每个配置生成预测结果
    for config_id in configs_to_visualize:
        print(f"处理配置 {config_id} ({config_names.get(config_id, f'config_{config_id}')})...")
        
        # 加载模型
        model = load_model_for_config(args, config_id)
        if model is None:
            continue
            
        # 执行推理
        start_time = time.time()
        main_output = model(pre_img, post_img)
        end_time = time.time()
        
        inference_time = end_time - start_time
        inference_times[config_id] = inference_time
        
        # 预测结果
        pred = torch.where(main_output > 0.5, 1, 0)
        pred_np = pred.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        
        # 创建预测结果可视化
        prediction_vis = np.zeros((output_h, output_w, 3), dtype=np.uint8)
        prediction_vis[(pred_np == 1)] = [255, 255, 255]
        
        # 创建差异图
        diff_vis = create_difference_visualization(pred_np, target_np)
        
        config_predictions[config_id] = prediction_vis
        config_differences[config_id] = diff_vis
        
        # 清理模型以释放显存
        del model
        torch.cuda.empty_cache()
    
    # 创建组合可视化
    if not config_predictions:
        print(f"图像 {image_idx+1} 没有可用的配置预测结果。")
        return False
    
    # 计算布局：T1, T2, GT, 然后是所有配置的预测结果和差异图
    num_configs = len(config_predictions)
    cols = 3 + num_configs * 2  # T1, T2, GT + (预测+差异) * 配置数
    rows = 1
    
    # 创建分隔线
    spacer = np.zeros((output_h, 10, 3), dtype=np.uint8)
    
    # 组合所有图像
    combined_parts = [pre_img_vis, spacer, post_img_vis, spacer, target_vis, spacer]
    
    # 添加每个配置的预测结果和差异图
    for config_id in sorted(config_predictions.keys()):
        combined_parts.extend([config_predictions[config_id], spacer, config_differences[config_id], spacer])
    
    # 移除最后一个分隔线
    combined_parts = combined_parts[:-1]
    
    # 水平拼接所有图像
    combined_vis = np.concatenate(combined_parts, axis=1)
    
    # 添加标题信息
    title_height = 40
    title_img = np.zeros((title_height, combined_vis.shape[1], 3), dtype=np.uint8)
    title_text = f"Image {image_idx+1} - All Configurations Comparison"
    cv2.putText(title_img, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # 添加图例
    legend_height = 80
    legend_img = np.zeros((legend_height, combined_vis.shape[1], 3), dtype=np.uint8)
    
    # 基础图例
    legend_texts = ["T1 Image", "T2 Image", "Ground Truth"]
    x_positions = [0, output_w + 10, 2*(output_w + 10)]
    
    # 添加配置图例
    current_x = 3*(output_w + 10)
    for config_id in sorted(config_predictions.keys()):
        config_name = config_names.get(config_id, f"Config{config_id}")
        pred_text = f"{config_name} Pred"
        diff_text = f"{config_name} Diff"
        
        cv2.putText(legend_img, pred_text, (current_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(legend_img, diff_text, (current_x, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 添加推理时间
        time_text = f"{inference_times[config_id]:.3f}s"
        cv2.putText(legend_img, time_text, (current_x, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)
        
        current_x += 2*(output_w + 10)
    
    # 添加差异图颜色说明
    color_legend_y = 20
    cv2.putText(legend_img, "Diff Colors:", (10, color_legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(legend_img, "Green=TP, Red=FP, Blue=FN", (10, color_legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # 最终组合图像
    final_vis = np.concatenate([title_img, combined_vis, legend_img], axis=0)
    
    # 为每张图像创建单独的文件夹
    original_filename = os.path.basename(original_pre_path)
    filename_without_ext = os.path.splitext(original_filename)[0]
    image_folder = join(args.output_dir, f"image_{image_idx+1:03d}_{filename_without_ext}")
    os.makedirs(image_folder, exist_ok=True)
    
    # 保存组合可视化结果
    combined_save_path = join(image_folder, "combined_visualization.png")
    cv2.imwrite(combined_save_path, final_vis)
    print(f"保存组合可视化结果到: {combined_save_path}")
    
    # 分别保存每个组件
    # 保存T1图像
    t1_save_path = join(image_folder, "01_T1_image.png")
    cv2.imwrite(t1_save_path, pre_img_vis)
    
    # 保存T2图像
    t2_save_path = join(image_folder, "02_T2_image.png")
    cv2.imwrite(t2_save_path, post_img_vis)
    
    # 保存Ground Truth
    gt_save_path = join(image_folder, "03_Ground_Truth.png")
    cv2.imwrite(gt_save_path, target_vis)
    
    # 保存每个配置的预测结果和差异图
    for config_id in sorted(config_predictions.keys()):
        config_name = config_names.get(config_id, f"Config{config_id}")
        
        # 保存预测结果
        pred_save_path = join(image_folder, f"04_{config_name}_Prediction.png")
        cv2.imwrite(pred_save_path, config_predictions[config_id])
        
        # 保存差异图
        diff_save_path = join(image_folder, f"05_{config_name}_Difference.png")
        cv2.imwrite(diff_save_path, config_differences[config_id])
    
    # 保存图像信息文件
    info_save_path = join(image_folder, "image_info.txt")
    with open(info_save_path, 'w', encoding='utf-8') as f:
        f.write(f"图像信息\n")
        f.write(f"========\n")
        f.write(f"图像索引: {image_idx+1}\n")
        f.write(f"原始文件名: {original_filename}\n")
        f.write(f"T1图像路径: {original_pre_path}\n")
        f.write(f"T2图像路径: {original_post_path}\n")
        f.write(f"图像尺寸: {output_h} x {output_w}\n")
        f.write(f"处理时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"配置预测结果:\n")
        for config_id in sorted(config_predictions.keys()):
            config_name = config_names.get(config_id, f"Config{config_id}")
            inference_time = inference_times.get(config_id, 0)
            f.write(f"  {config_id}: {config_name} - 推理时间: {inference_time:.3f}s\n")
        
        f.write(f"\n差异图颜色说明:\n")
        f.write(f"  绿色: 正确预测的变化 (True Positive)\n")
        f.write(f"  红色: 误报 (False Positive)\n")
        f.write(f"  蓝色: 漏报 (False Negative)\n")
    
    print(f"图像 {image_idx+1} 的所有组件已保存到文件夹: {image_folder}")
    
    return True

def visualize_ablation_by_image(args):
    """
    按图片组织运行消融实验配置的可视化。
    """
    print(f"\n=== 开始按图片组织消融实验可视化 ===")
    
    # 确定要可视化的配置列表
    if args.configs is not None:
        configs_to_visualize = [int(x.strip()) for x in args.configs.split(',')]
        print(f"指定可视化的配置: {configs_to_visualize}")
    else:
        configs_to_visualize = [0, 1, 2, 3, 4, 5, 6, 7]  # 默认可视化所有配置（0-7共8个）
        print("可视化所有配置（0-7）")
    
    # 创建主输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取数据变换
    _, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)
    
    # 创建数据加载器
    test_loader = create_test_loader(args, val_transform)
    
    # 配置名称映射
    config_names = {
        0: "X3D",
        1: "X3D+A", 
        2: "X3D+B",
        3: "X3D+C",
        4: "X3D+A+B",
        5: "X3D+A+C",
        6: "X3D+B+C",
        7: "X3D+A+B+C"
    }
    
    # 为每张图像生成可视化
    total_images = len(test_loader)
    successful_images = 0
    
    for image_idx in range(total_images):
        print(f"\n=== 处理图像 {image_idx+1}/{total_images} ===")
        
        try:
            success = visualize_all_configs_for_image(args, configs_to_visualize, test_loader, image_idx)
            if success:
                successful_images += 1
        except Exception as e:
            print(f"图像 {image_idx+1} 可视化失败: {e}")
    
    # 生成可视化结果报告
    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    report_path = join(args.output_dir, f"visualization_report_{dataset_name}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 按图片组织消融实验可视化结果报告 ===\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总图像数: {total_images}\n")
        f.write(f"成功处理图像数: {successful_images}\n")
        f.write(f"失败图像数: {total_images - successful_images}\n\n")
        
        f.write("可视化的配置:\n")
        for config_id in configs_to_visualize:
            config_name = config_names.get(config_id, f"配置{config_id}")
            f.write(f"  {config_id}: {config_name}\n")
        
        f.write("\n" + "="*50 + "\n")
        f.write("详细配置说明:\n")
        f.write("A: Attention模块\n")
        f.write("B: 3D Shuffle ASPP模块\n") 
        f.write("C: EdgeFormer模块\n")
        f.write("+ : 模块组合\n")
        f.write("增强机制: 所有配置均使用基于余弦相似度的感知帧增强\n")
        f.write("\n可视化说明:\n")
        f.write("- 每张图像创建单独的文件夹，包含以下文件：\n")
        f.write("  * combined_visualization.png: 组合可视化图像\n")
        f.write("  * 01_T1_image.png: 前时相图像\n")
        f.write("  * 02_T2_image.png: 后时相图像\n")
        f.write("  * 03_Ground_Truth.png: 真实变化标签\n")
        f.write("  * 04_配置名_Prediction.png: 各配置预测结果\n")
        f.write("  * 05_配置名_Difference.png: 各配置差异图\n")
        f.write("  * image_info.txt: 图像详细信息\n")
        f.write("- 差异图颜色说明：绿色=正确预测变化(TP)，红色=误报(FP)，蓝色=漏报(FN)\n")
        f.write("- 文件夹格式：image_XXX_原文件名/\n")

    print(f"\n=== 按图片组织消融实验可视化完成 ===")
    print(f"总图像数: {total_images}")
    print(f"成功处理图像数: {successful_images}")
    print(f"可视化结果报告已保存到: {report_path}")
    print(f"可视化结果保存在: {args.output_dir}")

def get_parser():
    """
    创建脚本的参数解析器。
    """
    parser = argparse.ArgumentParser(description='按图片组织消融实验可视化脚本')
    
    # 数据集和路径参数
    parser.add_argument('--file_root', type=str, required=True, help='数据集根目录路径。')
    parser.add_argument('--save_dir', type=str, default='./exp_ablation', help='实验保存的根目录。')
    parser.add_argument('--output_dir', type=str, default='./vis_results_by_image', help='可视化结果输出目录。')
    parser.add_argument('--gpu_id', type=str, default='0', help='使用的GPU ID。')
    parser.add_argument('--in_height', type=int, default=256, help='RGB图像高度')
    parser.add_argument('--in_width', type=int, default=256, help='RGB图像宽度')

    # 模型特定参数
    parser.add_argument('--num_perception_frame', type=int, default=1, help='感知帧数量（当前架构必须为1）')
    parser.add_argument('--pretrained', default=r'model\X3D_L.pyth', type=str, help='预训练X3D权重路径')
    
    # 可视化参数
    parser.add_argument('--configs', type=str, default=None, help='指定要可视化的配置ID列表，用逗号分隔，如"0,1,2,3,4,5,6,7"。默认为所有配置（0-7）。')
    parser.add_argument('--model_path', type=str, default=None, help='模型权重路径（配置7时可选，用于指定完整模型的路径，默认使用save_dir下的best_model.pth）。')
    parser.add_argument('--batch_size', type=int, default=1, help='可视化批次大小（固定为1）。')

    return parser

def main():
    """
    主函数：解析参数并开始按图片组织消融实验可视化。
    """
    parser = get_parser()
    args = parser.parse_args()
    
    # 设置GPU环境
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.manual_seed(seed=16)
        torch.cuda.manual_seed(seed=16)
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        torch.manual_seed(seed=16)
        print("警告: CUDA不可用，将使用CPU运行")
    
    # 运行按图片组织的可视化
    visualize_ablation_by_image(args)

if __name__ == '__main__':
    main()

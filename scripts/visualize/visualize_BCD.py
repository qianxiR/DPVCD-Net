# Copyright (c) Duowang Zhu.
# All rights reserved.

import os
# Fix for OMP: Error #15, must be placed before importing torch
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import sys
import argparse
import time
import json
from os.path import join

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from skimage import io

# Insert current path for local module imports
sys.path.insert(0, '.')

import data.dataset as RSDataset
import data.transforms as RSTransforms
from model.trainer import Trainer

'''
使用示例：

# 可视化GVLM-CD数据集结果
python scripts/visualize/visualize_BCD.py --file_root "E:\rqx\dataes\GVLM-CD" --model_path ./exp_new/GVLM-CD/best_model.pth --output_dir ./vis_results_BCD_GVLM --gpu_id 0

# 可视化WHU-CD数据集结果
python scripts/visualize/visualize_BCD.py --file_root "E:\rqx\dataes\WHU-CD" --model_path ./exp_new/WHU-CD/best_model.pth --output_dir ./vis_results_BCD_WHU --gpu_id 0

# 可视化LBFD-CD数据集结果
python scripts/visualize/visualize_BCD.py --file_root "E:\rqx\dataes\LBFD-CD" --model_path ./exp_new/LBFD-CD/final_model.pth --output_dir ./vis_results_BCD_LBFD --gpu_id 0

# 使用final_model.pth（训练后在测试集上评估的最终模型）
python scripts/visualize/visualize_BCD.py --file_root "E:\rqx\dataes\WHU-CD" --model_path ./exp_new/WHU-CD/final_model.pth --output_dir ./vis_results_final
'''

def visualize(args):
    """
    Main function for model inference and visualization.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 1. Create model and load weights ---
    model = Trainer(args=args)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model weights not found at {args.model_path}")
        return

    print(f"Loading model from {args.model_path}")
    state_dict = torch.load(args.model_path, map_location=device)

    # Handle state_dict saved with DataParallel
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.to(device)
    model.eval()

    # --- 2. Create data loader for the test set ---
    _, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)
    test_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="test",
        transform=val_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=1,  # Process one image at a time
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # --- 3. Create output directory (matching ablation format) ---
    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    config_output_dir = join(args.output_dir, dataset_name, "config_8")
    os.makedirs(config_output_dir, exist_ok=True)
    
    # 创建预测掩码和差异掩码的单独目录
    prediction_mask_dir = join(config_output_dir, 'prediction_masks')
    diff_mask_dir = join(config_output_dir, 'diffdata')
    os.makedirs(prediction_mask_dir, exist_ok=True)
    os.makedirs(diff_mask_dir, exist_ok=True)
    
    # 配置名称（与ablation格式一致）
    config_name = "Base + Attention + ASPP + Transformer"
    print(f"Visualization results will be saved to {config_output_dir}")
    print(f"配置名称: {config_name}")

    # --- 4. Perform inference and save visualizations ---
    # 用于记录文件名到原始路径的映射（与ablation格式一致）
    path_mapping = {}
    total_vis_count = 0
    with torch.no_grad():
        for i, batched_inputs in enumerate(test_loader):
            img, target = batched_inputs[0].to(device), batched_inputs[1].to(device)
            
            pre_img = img[:, :3, :, :]
            post_img = img[:, 3:, :, :]

            start_time = time.time()
            # The model returns a sigmoid-activated tensor in eval mode
            main_output = model(pre_img, post_img)
            end_time = time.time()

            # Prediction is based on the single, sigmoid-activated output
            pred = torch.where(main_output > 0.5, 1, 0)
            
            # --- Print attention scores and logits information ---
            print(f"\n=== 配置 8 ({config_name}) - 图像 {i+1} 的预测信息 ===")
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
            print(f"  - 变化比例: {pred_changes/total_pixels*100:.2f}%")
            print(f"  - 真实变化比例: {target_changes/total_pixels*100:.2f}%")
            print("=" * 50)

            # --- Convert tensors to visualizable numpy arrays ---
            original_pre_path = test_loader.dataset.pre_images[i]
            pre_img_vis = io.imread(original_pre_path)
            original_post_path = test_loader.dataset.post_images[i]
            post_img_vis = io.imread(original_post_path)

            output_h, output_w = pred.shape[-2], pred.shape[-1]
            pre_img_vis = cv2.resize(pre_img_vis, (output_w, output_h))
            post_img_vis = cv2.resize(post_img_vis, (output_w, output_h))

            pre_img_vis = cv2.cvtColor(pre_img_vis, cv2.COLOR_RGB2BGR)
            post_img_vis = cv2.cvtColor(post_img_vis, cv2.COLOR_RGB2BGR)

            # --- Create Visualization Maps ---
            # Ground Truth visualization
            target_vis = np.stack([(target_np * 255)] * 3, axis=-1)
            
            # Model Prediction Map (二值图像，白色=变化，黑色=无变化)
            prediction_mask = np.zeros((output_h, output_w, 3), dtype=np.uint8)
            prediction_mask[(pred_np == 1)] = [255, 255, 255]

            # Difference Map Visualization (差异掩码)
            diff_mask = np.zeros((output_h, output_w, 3), dtype=np.uint8)
            # 白色：正确预测的变化（True Positive - TP）
            correct_change = (pred_np == 1) & (target_np == 1)
            diff_mask[correct_change] = [255, 255, 255]
            # 蓝色：误检（False Positive - FP）
            false_positive = (pred_np == 1) & (target_np == 0)
            diff_mask[false_positive] = [0, 0, 255]
            # 绿色：漏检（False Negative - FN）
            false_negative = (pred_np == 0) & (target_np == 1)
            diff_mask[false_negative] = [0, 255, 0]
            # 黑色：正确预测的无变化（True Negative - TN，默认背景色）
            # TN = (pred_np == 0) & (target_np == 0)，保持默认 [0, 0, 0]

            # --- Create spacer and concatenate all images ---
            spacer = np.zeros((output_h, 10, 3), dtype=np.uint8)
            combined_vis = np.concatenate([
                pre_img_vis, spacer, 
                post_img_vis, spacer, 
                target_vis, spacer, 
                prediction_mask, spacer, 
                diff_mask
            ], axis=1)

            # 添加标题信息（与ablation格式一致）
            title_height = 40
            title_img = np.zeros((title_height, combined_vis.shape[1], 3), dtype=np.uint8)
            title_text = f"Config 8 ({config_name}) - Image {i+1} - Time: {end_time - start_time:.3f}s"
            cv2.putText(title_img, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 添加图例（与ablation格式一致）
            legend_height = 60
            legend_img = np.zeros((legend_height, combined_vis.shape[1], 3), dtype=np.uint8)
            legend_texts = [
                "T1 Image",
                "T2 Image",
                "Ground Truth",
                "Prediction",
                "Difference (White:TP, Black:TN, Blue:FP, Green:FN)"
            ]
            x_positions = [0, output_w + 10, 2*(output_w + 10), 3*(output_w + 10), 4*(output_w + 10)]
            for j, text in enumerate(legend_texts):
                if j < len(x_positions):
                    cv2.putText(legend_img, text, (x_positions[j], 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 最终组合图像
            final_vis = np.concatenate([title_img, combined_vis, legend_img], axis=0)

            # --- Save visualization (matching ablation format) ---
            original_filename = os.path.basename(original_pre_path)
            vis_filename = f"{config_name}_{original_filename}"
            save_path = join(config_output_dir, vis_filename)
            cv2.imwrite(save_path, final_vis)
            
            # 单独保存预测掩码（与ablation格式一致）
            prediction_mask_filename = f"pred_{original_filename}"
            prediction_mask_path = join(prediction_mask_dir, prediction_mask_filename)
            cv2.imwrite(prediction_mask_path, prediction_mask)
            
            # 单独保存差异掩码（文件名与原始T1/T2匹配）
            diff_mask_path = join(diff_mask_dir, original_filename)
            cv2.imwrite(diff_mask_path, diff_mask)
            
            # 记录路径映射（与ablation格式一致）
            original_label_path = test_loader.dataset.label_change[i]
            path_mapping[vis_filename] = {
                'visualization_file': save_path,
                'original_pre_image': original_pre_path,
                'original_post_image': original_post_path,
                'original_label': original_label_path,
                'image_index': i + 1,
                'dataset_name': dataset_name,
                'config_id': 8,
                'config_name': config_name
            }
            
            print(f"保存可视化结果到: {save_path}")
            print(f"  -> Source T1 Image: {original_pre_path}")
            print(f"  -> Source T2 Image: {original_post_path}")
            print(f"  -> Source Label:    {original_label_path}")
            
            total_vis_count += 1

    # 保存路径映射文件（与ablation格式一致）
    mapping_file = join(config_output_dir, 'path_mapping.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(path_mapping, f, indent=2, ensure_ascii=False)
    print(f"\n[完成] 路径映射文件已保存到: {mapping_file}")
    print(f"   共记录 {len(path_mapping)} 个图像的路径映射")
    
    print(f"\n配置 8 ({config_name}) 可视化完成，共处理 {total_vis_count} 张图像。")
    print(f"可视化结果保存在: {config_output_dir}")


def get_parser():
    """
    Creates an argument parser for the visualization script.
    """
    parser = argparse.ArgumentParser(description='Change Detection Visualization')
    
    # --- Paths and Directories ---
    parser.add_argument('--file_root', type=str, required=True, help='Path to the dataset root directory.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model weights (.pth file).')
    parser.add_argument('--output_dir', type=str, default='./vis_results_ablation', help='Directory to save visualization results (will create {output_dir}/{dataset_name}/config_8/).')
    parser.add_argument('--gpu_id', type=str, default='0', help='ID of GPU to use.')
    
    # --- Data Loading ---
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker processes for data loading (use 0 on Windows to avoid pickle errors).')
    parser.add_argument('--in_height', type=int, default=256, help='Height of input RGB image.')
    parser.add_argument('--in_width', type=int, default=256, help='Width of input RGB image.')

    # --- Model Specific ---
    parser.add_argument('--num_perception_frame', type=int, default=1, help='Number of perception frames (must be 1 for current arch).')
    parser.add_argument('--pretrained', default=r'model\X3D_L.pyth', type=str, help='Path to pretrained X3D weight (required by model init).')

    # --- Dummy arguments to match Trainer initialization ---
    # These are not used in inference but required by the Trainer class constructor
    parser.add_argument('--use_checkpoint', action='store_true', default=False, help='Dummy arg for model init.')
    
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    visualize(args)


if __name__ == '__main__':
    main() 

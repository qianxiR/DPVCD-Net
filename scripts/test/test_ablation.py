#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
消融实验模型测试脚本

用于加载已训练的消融实验模型并在测试集上进行评估

使用示例：
# 测试配置6（Base + Attention + Transformer）在三个数据集上推理

# 1. GVLM-CD数据集 - 配置6
python scripts/test/test_ablation.py --file_root "G:\deeplearning\cd\GVLM-CD" --model_path ./exp_ablation_batch/GVLM-CD/config_6_config_6/GVLM-CD/best_model.pth --ablation_config 6 --batch_size 8

# 2. WHU-CD数据集 - 配置6
python scripts/test/test_ablation.py --file_root "G:\deeplearning\实验数据\WHU-CD" --model_path ./exp_ablation_batch/WHU-CD/config_6_config_6/WHU-CD/best_model.pth --ablation_config 6 --batch_size 8

# 3. LBFD-CD数据集 - 配置6
python scripts/test/test_ablation.py --file_root "G:\deeplearning\实验数据\LBFD-CD" --model_path ./exp_ablation_batch/LBFD-CD/config_6_config_6/LBFD-CD/best_model.pth --ablation_config 6 --batch_size 8

# 测试并保存预测结果（以GVLM-CD为例）
python scripts/test/test_ablation.py --file_root "G:\deeplearning\cd\GVLM-CD" --model_path ./exp_ablation_batch/GVLM-CD/config_6_config_6/GVLM-CD/best_model.pth --ablation_config 6 --save_predictions --output_dir ./predictions --batch_size 8
"""

import os
import sys
import time
import numpy as np
from os.path import join
from argparse import ArgumentParser
from tqdm import tqdm

import torch
import torch.nn.functional as F

# 插入当前路径以导入本地模块
sys.path.insert(0, '.')

import data.dataset as RSDataset
import data.transforms as RSTransforms
from utils.metric_tool import ConfuseMatrixMeter

from model.trainer_ablation import TrainerAblation, create_ablation_model


def print_separator(char='=', length=80):
    """打印分隔线"""
    print(char * length)


def print_header(text):
    """打印带格式的标题"""
    print_separator()
    print(f"  {text}")
    print_separator()


@torch.no_grad()
def test_model(args, test_loader, model):
    """
    在测试集上测试模型性能
    
    入参:
    - args: 配置参数
    - test_loader: 测试数据加载器
    - model: 待测试的模型
    
    方法:
    1. 将模型设为评估模式
    2. 遍历测试集进行推理
    3. 计算各项评估指标
    4. 可选：保存预测结果
    
    出参:
    - scores (dict): 测试指标分数字典
    """
    model.eval()
    eval_meter = ConfuseMatrixMeter(n_class=2)
    total_batches = len(test_loader)
    
    print_header(f"开始测试 - 共 {total_batches} 个批次")
    
    all_predictions = []
    all_targets = []
    batch_times = []
    
    # 创建进度条
    pbar = tqdm(test_loader, desc="测试进度", ncols=100, 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
    
    for iter_idx, batched_inputs in enumerate(pbar):
        img, target = batched_inputs[0], batched_inputs[1]
        
        # 分离前后图像并移至GPU
        pre_img = img[:, :3, :, :].cuda().float()
        post_img = img[:, 3:, :, :].cuda().float()
        target = target.cuda().float()

        start_time = time.time()

        # 模型推理
        main_output = model(pre_img, post_img)
        
        # 生成预测结果
        pred = torch.where(
            main_output > 0.5,
            torch.ones_like(main_output),
            torch.zeros_like(main_output)
        ).long()

        time_taken = time.time() - start_time
        batch_times.append(time_taken)

        # 更新混淆矩阵
        f1 = eval_meter.update_cm(
            pr=pred.cpu().numpy(),
            gt=target.cpu().numpy()
        )
        
        # 保存预测结果（如果需要）
        if args.save_predictions:
            all_predictions.append(pred.cpu().numpy())
            all_targets.append(target.cpu().numpy())
        
        # 更新进度条信息
        pbar.set_postfix({
            'F1': f'{f1:.4f}',
            'Time': f'{time_taken:.3f}s'
        })
    
    # 获取最终评估指标
    scores = eval_meter.get_scores()
    
    # 计算平均推理时间
    avg_time = np.mean(batch_times)
    total_time = np.sum(batch_times)
    
    # 打印测试结果
    print_separator()
    print("\n📊 测试结果:")
    print_separator('-')
    print(f"{'指标':<20} {'值':<15}")
    print_separator('-')
    print(f"{'F1 Score':<20} {scores['F1']:<15.4f}")
    print(f"{'IoU':<20} {scores['IoU']:<15.4f}")
    print(f"{'Kappa':<20} {scores['Kappa']:<15.4f}")
    print(f"{'Overall Accuracy':<20} {scores['OA']:<15.4f}")
    print(f"{'Recall':<20} {scores['recall']:<15.4f}")
    print(f"{'Precision':<20} {scores['precision']:<15.4f}")
    print_separator('-')
    print(f"{'平均推理时间/批次':<20} {avg_time:<15.3f}s")
    print(f"{'总推理时间':<20} {total_time:<15.2f}s")
    print(f"{'吞吐量':<20} {total_batches/total_time:<15.2f} batch/s")
    print_separator()
    
    # 保存预测结果
    if args.save_predictions:
        save_predictions(args, all_predictions, all_targets, scores)
    
    return scores


def save_predictions(args, predictions, targets, scores):
    """
    保存预测结果到文件
    
    入参:
    - args: 配置参数
    - predictions (list): 预测结果列表
    - targets (list): 真实标签列表
    - scores (dict): 评估指标
    """
    # 创建输出目录
    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    config_name = f"config_{args.ablation_config}"
    output_dir = join(args.output_dir, dataset_name, config_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存numpy数组
    predictions_array = np.concatenate(predictions, axis=0)
    targets_array = np.concatenate(targets, axis=0)
    
    pred_file = join(output_dir, 'predictions.npy')
    target_file = join(output_dir, 'targets.npy')
    
    np.save(pred_file, predictions_array)
    np.save(target_file, targets_array)
    
    print(f"\n✅ 预测结果已保存:")
    print(f"   预测: {pred_file}")
    print(f"   标签: {target_file}")
    
    # 保存评估指标到文本文件
    metrics_file = join(output_dir, 'test_metrics.txt')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("消融实验测试结果\n")
        f.write(f"数据集: {dataset_name}\n")
        f.write(f"配置: {args.ablation_config}\n")
        f.write(f"模型: {args.model_path}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"F1 Score:          {scores['F1']:.4f}\n")
        f.write(f"IoU:               {scores['IoU']:.4f}\n")
        f.write(f"Kappa:             {scores['Kappa']:.4f}\n")
        f.write(f"Overall Accuracy:  {scores['OA']:.4f}\n")
        f.write(f"Recall:            {scores['recall']:.4f}\n")
        f.write(f"Precision:         {scores['precision']:.4f}\n")
    
    print(f"   指标: {metrics_file}")


def load_model(args):
    """
    加载训练好的消融实验模型
    
    入参:
    - args: 配置参数
    
    方法:
    1. 根据消融配置创建模型实例
    2. 加载模型权重
    3. 将模型移至GPU
    
    出参:
    - model: 加载好的模型
    """
    print_header("加载消融实验模型")
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"模型文件不存在: {args.model_path}")
    
    print(f"📁 模型路径: {args.model_path}")
    print(f"🔧 消融配置: {args.ablation_config}")
    
    # 根据消融配置创建模型
    model = create_ablation_model(args, experiment_id=args.ablation_config)
    
    # 模拟一次前向传播以初始化所有动态模块
    print(f"🔄 初始化动态模块（模拟前向传播）...", end='', flush=True)
    with torch.no_grad():
        # 创建dummy输入（使用配置的图像尺寸）
        dummy_input = torch.randn(1, 3, args.in_height, args.in_width)
        try:
            # 执行一次前向传播来初始化所有动态创建的模块
            _ = model(dummy_input, dummy_input)
            print(f" ✅ 完成")
        except Exception as e:
            print(f" ⚠️  警告")
            print(f"   模拟前向传播出现警告: {e}")
            print(f"   继续加载权重...")
    
    # 加载权重
    print(f"📦 加载权重文件...", end='', flush=True)
    try:
        loaded_data = torch.load(args.model_path, map_location='cpu')
        print(f" ✅")
    except Exception as e:
        print(f" ⚠️")
        print(f"   尝试使用 weights_only=False 加载...", end='', flush=True)
        loaded_data = torch.load(args.model_path, map_location='cpu', weights_only=False)
        print(f" ✅")
    
    # 判断加载的是状态字典还是完整模型对象
    if isinstance(loaded_data, dict):
        # 检查是否是检查点文件
        if 'state_dict' in loaded_data:
            state_dict = loaded_data['state_dict']
            print(f"📦 检查点信息:")
            if 'epoch' in loaded_data:
                print(f"   Epoch: {loaded_data['epoch']}")
            if 'best_f1' in loaded_data:
                print(f"   最佳F1: {loaded_data['best_f1']:.4f}")
        else:
            # 直接是状态字典
            state_dict = loaded_data
    else:
        # 完整模型对象
        model_to_load = loaded_data.module if hasattr(loaded_data, 'module') else loaded_data
        state_dict = model_to_load.state_dict()
    
    # 处理 DataParallel 保存的模型（键名可能带有 'module.' 前缀）
    new_state_dict = {}
    has_module_prefix = any(k.startswith('module.') for k in state_dict.keys())
    
    if has_module_prefix:
        print(f"ℹ️  检测到 DataParallel 模型，正在移除 'module.' 前缀")
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v  # 移除 'module.' 前缀
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict
    
    # 余弦相似度增强模块已启用，保留所有权重
    # 直接使用原始状态字典，不过滤任何权重
    filtered_state_dict = state_dict
    
    # 加载状态字典（使用 strict=False 以兼容可能的结构差异）
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    # 显示加载信息
    if missing_keys:
        print(f"⚠️  模型中缺少的参数: {len(missing_keys)} 个")
        if len(missing_keys) <= 5:
            for key in missing_keys:
                print(f"   - {key}")
        else:
            print(f"   - {missing_keys[0]}")
            print(f"   - ... (省略 {len(missing_keys)-2} 个)")
            print(f"   - {missing_keys[-1]}")
    
    if unexpected_keys:
        print(f"ℹ️  权重文件中的额外参数: {len(unexpected_keys)} 个")
        if len(unexpected_keys) <= 5:
            for key in unexpected_keys:
                print(f"   - {key}")
        else:
            print(f"   - {unexpected_keys[0]}")
            print(f"   - ... (省略 {len(unexpected_keys)-2} 个)")
            print(f"   - {unexpected_keys[-1]}")
    
    # 判断加载状态
    print("")  # 空行
    if not missing_keys and not unexpected_keys:
        print(f"✅ 完美！所有模型参数完全匹配并成功加载")
    elif missing_keys and not unexpected_keys:
        print(f"⚠️  警告: 模型有 {len(missing_keys)} 个参数未从文件加载")
    elif unexpected_keys and not missing_keys:
        print(f"✅ 模型所有必需参数已成功加载")
    else:
        print(f"⚠️  警告: 模型与权重文件部分不匹配")
    
    # 移至GPU
    model.cuda()
    model.eval()
    
    print(f"\n🎯 模型已准备就绪，开始测试")
    print_separator()
    
    return model


def create_test_loader(args):
    """
    创建测试数据加载器
    
    入参:
    - args: 配置参数
    
    出参:
    - test_loader: 测试数据加载器
    """
    print_header("准备测试数据")
    
    # 检查数据集路径
    if not os.path.exists(args.file_root):
        raise FileNotFoundError(f"数据集路径不存在: {args.file_root}")
    
    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    print(f"📁 数据集: {dataset_name}")
    print(f"📍 路径: {args.file_root}")
    
    # 获取数据变换（仅验证变换，不增强）
    _, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)
    
    # 创建测试数据集
    test_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="test",
        transform=val_transform
    )
    
    print(f"📊 测试样本数: {len(test_data)}")
    
    # 创建数据加载器
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"🔢 批次大小: {args.batch_size}")
    print(f"📦 总批次数: {len(test_loader)}")
    print_separator()
    
    return test_loader


def get_parser():
    """
    创建脚本的参数解析器
    
    出参:
    - parser: 配置好的参数解析器
    """
    parser = ArgumentParser(description='消融实验模型测试脚本')
    
    # 必需参数
    parser.add_argument('--file_root', type=str, required=True, 
                       help='数据集根目录路径')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型文件路径（.pth或.pth.tar）')
    parser.add_argument('--ablation_config', type=int, required=True,
                       choices=[1, 2, 3, 4, 5, 6, 7],
                       help='消融实验配置ID (1-7)')
    
    # 数据加载参数
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='测试批次大小')
    parser.add_argument('--num_workers', type=int, default=0, 
                       help='数据加载的工作进程数（Windows环境建议使用0避免pickle错误）')
    parser.add_argument('--gpu_id', type=str, default='0', 
                       help='使用的GPU ID')
    
    # 模型参数
    parser.add_argument('--in_height', type=int, default=256, 
                       help='输入图像高度')
    parser.add_argument('--in_width', type=int, default=256, 
                       help='输入图像宽度')
    parser.add_argument('--num_perception_frame', type=int, default=1,
                       help='感知帧数量（当前架构必须为1）')
    parser.add_argument('--pretrained', type=str,
                       default=r'model\X3D_L.pyth',
                       help='预训练X3D权重路径（测试时不需要，但需要定义以保持兼容）')
    
    # 输出参数
    parser.add_argument('--save_predictions', action='store_true',
                       help='是否保存预测结果')
    parser.add_argument('--output_dir', type=str, default='./test_ablation_results',
                       help='预测结果保存目录')
    parser.add_argument('--verbose', action='store_true',
                       help='是否显示详细的测试过程')
    
    return parser


def main():
    """
    主函数：执行测试流程
    """
    parser = get_parser()
    args = parser.parse_args()
    
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    
    # 打印脚本信息
    print_header("🧪 消融实验模型测试")
    print(f"GPU设备: {args.gpu_id}")
    print(f"批次大小: {args.batch_size}")
    print(f"消融配置: {args.ablation_config}")
    if args.save_predictions:
        print(f"保存预测: 是 -> {args.output_dir}")
    else:
        print(f"保存预测: 否")
    print_separator()
    
    try:
        # 1. 加载模型
        model = load_model(args)
        
        # 2. 创建测试数据加载器
        test_loader = create_test_loader(args)
        
        # 3. 执行测试
        start_time = time.time()
        scores = test_model(args, test_loader, model)
        end_time = time.time()
        
        # 4. 打印总结
        print_header("测试完成")
        print(f"⏱️  总耗时: {end_time - start_time:.2f} 秒")
        print(f"🎯 F1 Score: {scores['F1']:.4f}")
        print(f"📊 IoU: {scores['IoU']:.4f}")
        print_separator()
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("请检查文件路径是否正确。")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量测试脚本 - 在多个数据集上测试模型

该脚本自动在多个数据集上测试已训练的模型，并生成对比报告

使用示例:
# 测试单个实验目录下的所有数据集
python scripts/test/batch_test.py --exp_dir ./exp_BCD --batch_size 8

# 测试指定的数据集（0:GVLM-CD, 1:WHU-CD, 2:LBFD-CD）
python scripts/test/batch_test.py --exp_dir ./exp_new --datasets "1" --batch_size 8

# 保存预测结果
python scripts/test/batch_test.py --exp_dir ./exp_new --save_predictions --output_dir ./test_results

# 指定GPU并跳过错误继续测试
python scripts/test/batch_test.py --exp_dir ./exp_new --batch_size 8 --gpu_id 0 --skip_on_error
"""

import os
import sys
import time
import subprocess
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm

# 插入当前路径
sys.path.insert(0, '.')

from data.datasets_config import DATASETS


# 数据集配置（从 data/datasets_config.py 集中加载）
# 如需修改路径，请编辑 data/datasets_config.py


def print_separator(char='=', length=80):
    """打印分隔线"""
    print(char * length)


def print_header(text):
    """打印带格式的标题"""
    print_separator()
    print(f"  {text}")
    print_separator()


def find_model_file(exp_dir, dataset_name):
    """
    查找数据集对应的模型文件
    
    入参:
    - exp_dir (str): 实验目录
    - dataset_name (str): 数据集名称
    
    出参:
    - model_path (str): 模型文件路径，如果未找到返回None
    """
    # 可能的模型文件位置
    possible_paths = [
        os.path.join(exp_dir, dataset_name, 'best_model.pth'),
        os.path.join(exp_dir, dataset_name, 'final_model.pth'),
        os.path.join(exp_dir, dataset_name, 'checkpoint.pth.tar'),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def run_test(dataset_config, model_path, args):
    """
    在单个数据集上运行测试
    
    入参:
    - dataset_config (dict): 数据集配置
    - model_path (str): 模型文件路径
    - args: 命令行参数
    
    出参:
    - success (bool): 测试是否成功
    - duration (float): 测试耗时
    """
    dataset_name = dataset_config['name']
    dataset_path = dataset_config['path']
    
    # 检查数据集路径
    if not os.path.exists(dataset_path):
        print(f"\n❌ 错误: 数据集路径不存在: {dataset_path}")
        return False, 0
    
    # 构建测试命令
    cmd = [
        'python', 'scripts/test/test_BCD.py',
        '--file_root', dataset_path,
        '--model_path', model_path,
        '--batch_size', str(args.batch_size),
        '--gpu_id', args.gpu_id,
        '--num_workers', str(args.num_workers),
    ]
    
    if args.save_predictions:
        cmd.extend(['--save_predictions', '--output_dir', args.output_dir])
    
    if args.verbose:
        cmd.append('--verbose')
    
    # 打印测试信息
    print_header(f"测试: {dataset_name}")
    print(f"📁 数据集: {dataset_path}")
    print(f"🎯 模型: {model_path}")
    print(f"🔢 批次大小: {args.batch_size}")
    print(f"\n执行命令:")
    print(' '.join(cmd))
    print_separator('-')
    
    # 记录开始时间
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n⏰ 测试开始时间: {start_datetime}\n")
    
    try:
        # 执行测试命令
        result = subprocess.run(cmd, check=True)
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 打印成功信息
        print_separator()
        print(f"✅ {dataset_name} 测试成功完成!")
        print(f"⏰ 测试结束时间: {end_datetime}")
        print(f"⏱️  耗时: {format_duration(duration)}")
        print_separator()
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        # 测试失败
        end_time = time.time()
        duration = end_time - start_time
        
        print_separator()
        print(f"❌ {dataset_name} 测试失败!")
        print(f"错误码: {e.returncode}")
        print(f"⏱️  已耗时: {format_duration(duration)}")
        print_separator()
        
        return False, duration
    
    except KeyboardInterrupt:
        # 用户中断
        print_separator()
        print(f"\n⚠️  测试被用户中断!")
        print_separator()
        raise


def format_duration(seconds):
    """
    将秒数格式化为易读的时长字符串
    
    入参:
    - seconds (float): 秒数
    
    出参:
    - duration_str (str): 格式化的时长字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}小时 {minutes}分钟 {secs}秒"
    elif minutes > 0:
        return f"{minutes}分钟 {secs}秒"
    else:
        return f"{secs}秒"


def print_summary(results):
    """
    打印测试总结
    
    入参:
    - results (list): 测试结果列表，每个元素为 (dataset_name, model_path, success, duration)
    """
    print_header("📊 批量测试总结")
    
    total_duration = 0
    successful_count = 0
    failed_count = 0
    
    for dataset_name, model_path, success, duration in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{dataset_name:15s} {status:10s} 耗时: {format_duration(duration)}")
        
        total_duration += duration
        if success:
            successful_count += 1
        else:
            failed_count += 1
    
    print_separator('-')
    print(f"总耗时: {format_duration(total_duration)}")
    print(f"成功: {successful_count} 个数据集")
    print(f"失败: {failed_count} 个数据集")
    print_separator()


def main():
    """
    主函数：批量测试
    """
    parser = ArgumentParser(description='批量测试变化检测模型')
    
    # 基本参数
    parser.add_argument('--exp_dir', type=str, required=True,
                       help='实验目录（包含各数据集的模型）')
    parser.add_argument('--gpu_id', type=str, default='0', 
                       help='使用的GPU ID')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='测试批次大小')
    parser.add_argument('--num_workers', type=int, default=0, 
                       help='数据加载工作进程数（Windows环境建议使用0避免pickle错误）')
    
    # 数据集选择
    parser.add_argument('--datasets', type=str, default=None,
                       help='指定数据集索引列表，用逗号分隔 (0:GVLM-CD, 1:WHU-CD, 2:LBFD-CD)')
    
    # 输出参数
    parser.add_argument('--save_predictions', action='store_true',
                       help='是否保存预测结果')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                       help='预测结果保存目录')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细测试过程')
    
    # 控制参数
    parser.add_argument('--skip_on_error', action='store_true',
                       help='如果某个数据集测试失败，是否跳过继续测试下一个')
    
    args = parser.parse_args()
    
    # 检查实验目录
    if not os.path.exists(args.exp_dir):
        print(f"❌ 错误: 实验目录不存在: {args.exp_dir}")
        sys.exit(1)
    
    # 解析要测试的数据集
    if args.datasets is not None:
        dataset_indices = [int(x.strip()) for x in args.datasets.split(',')]
        datasets_to_test = [DATASETS[i] for i in dataset_indices if 0 <= i < len(DATASETS)]
    else:
        datasets_to_test = DATASETS
    
    # 打印脚本信息
    print_header("🧪 批量测试脚本启动")
    print(f"实验目录: {args.exp_dir}")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n将测试以下数据集:")
    for i, dataset in enumerate(datasets_to_test, 1):
        print(f"  {i}. {dataset['name']}: {dataset['description']}")
    print_separator()
    
    # 测试结果记录
    results = []
    
    try:
        # 依次测试每个数据集
        for dataset in datasets_to_test:
            dataset_name = dataset['name']
            
            # 查找模型文件
            model_path = find_model_file(args.exp_dir, dataset_name)
            
            if model_path is None:
                print(f"\n⚠️  未找到 {dataset_name} 的模型文件，跳过")
                print(f"   搜索目录: {os.path.join(args.exp_dir, dataset_name)}")
                results.append((dataset_name, 'N/A', False, 0))
                continue
            
            # 运行测试
            success, duration = run_test(dataset, model_path, args)
            results.append((dataset_name, model_path, success, duration))
            
            # 如果失败且不跳过错误，则终止
            if not success and not args.skip_on_error:
                print(f"\n⚠️  由于 {dataset_name} 测试失败，批量测试终止。")
                print(f"   使用 --skip_on_error 参数可在失败时继续测试下一个数据集。")
                break
            
            # 测试间隔
            if dataset != datasets_to_test[-1]:
                print(f"\n⏸️  等待 3 秒后开始下一个数据集的测试...\n")
                time.sleep(3)
        
        # 打印总结
        if results:
            print("\n")
            print_summary(results)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  批量测试被用户中断!")
        if results:
            print("\n已完成的测试:")
            print_summary(results)
    
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
    
    print_header("批量测试脚本结束")


if __name__ == '__main__':
    main()


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量训练脚本 - 依次训练多个变化检测数据集

数据集路径统一由 data/datasets_config.py 管理，如需修改路径请编辑该文件。

使用方法:
    # 批量训练所有数据集
    python scripts/train/batch_train.py
    
    # 指定批次大小和GPU
    python scripts/train/batch_train.py --batch_size 8 --gpu_id 0
    
    # 从指定数据集开始训练（0:GVLM-CD, 1:WHU-CD, 2:LBFD-CD）
    python scripts/train/batch_train.py --start_from 1
    
    # 训练失败时跳过继续下一个
    python scripts/train/batch_train.py --skip_on_error

可选参数:
    --gpu_id: 指定GPU设备ID (默认: 0)
    --batch_size: 批次大小 (默认: 8)
    --save_dir: 实验保存根目录 (默认: ./exp_new)
    --max_steps: 最大训练步数 (默认: 80000)
"""

import os
import sys
import subprocess
import time
from argparse import ArgumentParser
from datetime import datetime

# 插入当前路径以导入本地模块
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


def run_training(dataset_config, args):
    """
    运行单个数据集的训练
    
    入参:
    - dataset_config (dict): 数据集配置字典
    - args: 命令行参数
    
    方法:
    1. 构建训练命令
    2. 记录训练开始时间
    3. 执行训练脚本
    4. 记录训练结束时间和耗时
    
    出参:
    - success (bool): 训练是否成功
    - duration (float): 训练耗时（秒）
    """
    dataset_name = dataset_config['name']
    dataset_path = dataset_config['path']
    dataset_desc = dataset_config['description']
    
    # 检查数据集路径是否存在
    if not os.path.exists(dataset_path):
        print(f"\n❌ 错误: 数据集路径不存在: {dataset_path}")
        print(f"   请检查数据集是否已正确放置。")
        return False, 0
    
    # 构建训练命令
    cmd = [
        'python', 'scripts/train/train_BCD.py',
        '--file_root', dataset_path,
        '--save_dir', args.save_dir,
        '--batch_size', str(args.batch_size),
        '--gpu_id', args.gpu_id,
        '--max_steps', str(args.max_steps),
        '--num_workers', str(args.num_workers),
        '--learning_rate', str(args.learning_rate),
        '--val_interval', str(args.val_interval),
        '--log_iter', str(args.log_iter),
    ]
    
    # 打印训练信息
    print_header(f"开始训练: {dataset_name} ({dataset_desc})")
    print(f"📁 数据集路径: {dataset_path}")
    print(f"💾 保存目录: {os.path.join(args.save_dir, dataset_name)}")
    print(f"🎯 批次大小: {args.batch_size}")
    print(f"🔢 最大步数: {args.max_steps}")
    print(f"📊 GPU设备: {args.gpu_id}")
    print(f"\n执行命令:")
    print(' '.join(cmd))
    print_separator('-')
    
    # 记录开始时间
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n⏰ 训练开始时间: {start_datetime}\n")
    
    try:
        # 执行训练命令
        result = subprocess.run(cmd, check=True)
        
        # 记录结束时间
        end_time = time.time()
        duration = end_time - start_time
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 打印成功信息
        print_separator()
        print(f"✅ {dataset_name} 训练成功完成!")
        print(f"⏰ 训练结束时间: {end_datetime}")
        print(f"⏱️  总耗时: {format_duration(duration)}")
        print_separator()
        
        return True, duration
        
    except subprocess.CalledProcessError as e:
        # 训练失败
        end_time = time.time()
        duration = end_time - start_time
        
        print_separator()
        print(f"❌ {dataset_name} 训练失败!")
        print(f"错误码: {e.returncode}")
        print(f"⏱️  已耗时: {format_duration(duration)}")
        print_separator()
        
        return False, duration
    
    except KeyboardInterrupt:
        # 用户中断
        print_separator()
        print(f"\n⚠️  训练被用户中断!")
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
    打印训练总结
    
    入参:
    - results (list): 训练结果列表，每个元素为 (dataset_name, success, duration)
    """
    print_header("📊 批量训练总结")
    
    total_duration = 0
    successful_count = 0
    failed_count = 0
    
    for dataset_name, success, duration in results:
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
    主函数：批量训练所有数据集
    """
    parser = ArgumentParser(description='批量训练变化检测模型')
    
    # 基本参数
    parser.add_argument('--gpu_id', type=str, default='0', help='使用的GPU ID')
    parser.add_argument('--batch_size', type=int, default=8, help='训练批次大小')
    parser.add_argument('--save_dir', type=str, default='./exp_new', help='实验保存根目录')
    parser.add_argument('--max_steps', type=int, default=80000, help='最大训练步数')
    
    # 训练参数
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载工作进程数（Windows环境建议使用0避免pickle错误）')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='初始学习率')
    parser.add_argument('--val_interval', type=int, default=1, help='验证间隔（epoch）')
    parser.add_argument('--log_iter', type=int, default=20, help='日志打印间隔')
    
    # 控制参数
    parser.add_argument('--start_from', type=int, default=0, 
                       help='从第几个数据集开始训练 (0: GVLM-CD, 1: WHU-CD, 2: LBFD-CD)')
    parser.add_argument('--skip_on_error', action='store_true',
                       help='如果某个数据集训练失败，是否跳过继续训练下一个')
    
    args = parser.parse_args()
    
    # 打印脚本信息
    print_header("🚀 批量训练脚本启动")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"将依次训练 {len(DATASETS)} 个数据集:")
    for i, dataset in enumerate(DATASETS, 1):
        print(f"  {i}. {dataset['name']}: {dataset['description']}")
    print_separator()
    
    # 确认开始
    if args.start_from > 0:
        print(f"\n⚠️  将从第 {args.start_from + 1} 个数据集开始训练")
    
    # 训练结果记录
    results = []
    
    try:
        # 依次训练每个数据集
        for i, dataset in enumerate(DATASETS):
            # 跳过前面的数据集
            if i < args.start_from:
                print(f"\n⏭️  跳过 {dataset['name']}")
                continue
            
            # 运行训练
            success, duration = run_training(dataset, args)
            results.append((dataset['name'], success, duration))
            
            # 如果失败且不跳过错误，则终止
            if not success and not args.skip_on_error:
                print(f"\n⚠️  由于 {dataset['name']} 训练失败，批量训练终止。")
                print(f"   使用 --skip_on_error 参数可在失败时继续训练下一个数据集。")
                break
            
            # 训练间隔（让GPU冷却）
            if i < len(DATASETS) - 1:
                print(f"\n⏸️  等待 5 秒后开始下一个数据集的训练...\n")
                time.sleep(5)
        
        # 打印总结
        if results:
            print("\n")
            print_summary(results)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  批量训练被用户中断!")
        if results:
            print("\n已完成的训练:")
            print_summary(results)
    
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
    
    print_header("批量训练脚本结束")


if __name__ == '__main__':
    main()


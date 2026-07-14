#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量消融实验训练脚本 - 在多个数据集上依次运行消融实验（✨支持智能断点续训✨）

该脚本自动在GVLM-CD、WHU-CD和LBFD-CD三个数据集上依次运行消融实验，
支持指定配置或运行所有配置，支持智能检测训练状态并自动续训。

✨ 智能断点续训功能：
    - 自动检测已完成的配置（通过final_model.pth判断）→ 跳过
    - 自动检测中断的配置（通过checkpoint.pth.tar判断）→ 恢复训练
    - 自动检测未训练的配置 → 从头开始训练
    - 无需手动管理checkpoint，训练中断后重新运行即可自动续训

使用方法:
    # 在所有数据集上运行所有消融实验配置（1-7）
    python scripts/batch_ablation_train.py --run_all_configs
    
    # 在所有数据集上运行指定配置
    python scripts/batch_ablation_train.py --configs "5,6,7"
    
    # 在指定数据集上运行所有配置（0:GVLM-CD, 1:WHU-CD, 2:LBFD-CD）
    python scripts/batch_ablation_train.py --run_all_configs --datasets "1,2"
    
    # 训练中断后，重新运行相同命令即可自动续训（无需额外参数）
    python scripts/batch_ablation_train.py --run_all_configs
    
    # 指定批次大小和GPU
    python scripts/batch_ablation_train.py --run_all_configs --batch_size 8 --gpu_id 0

可选参数:
    --gpu_id: 指定GPU设备ID (默认: 0)
    --batch_size: 批次大小 (默认: 4)
    --save_dir: 实验保存根目录 (默认: ./exp_ablation_batch)
    --max_steps: 最大训练步数 (默认: 80000)
    --configs: 指定配置ID列表，如"1,2,3"
    --run_all_configs: 运行所有配置（1-7）
    --datasets: 指定数据集索引，如"0,1,2" (0:GVLM-CD, 1:WHU-CD, 2:LBFD-CD)
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

# 消融实验配置映射
ABLATION_CONFIGS = {
    1: "Base Only",
    2: "Base + Attention",
    3: "Base + ASPP", 
    4: "Base + Transformer",
    5: "Base + Attention + ASPP",
    6: "Base + Attention + Transformer",
    7: "Base + ASPP + Transformer"
}


def print_separator(char='=', length=80):
    """打印分隔线"""
    print(char * length)


def print_header(text):
    """打印带格式的标题"""
    print_separator()
    print(f"  {text}")
    print_separator()


def check_final_model_exists(dataset_save_dir, config_id, dataset_name):
    """
    检查指定配置是否已完成训练（通过final_model.pth文件判断）
    
    入参:
    - dataset_save_dir (str): 数据集保存根目录
    - config_id (int): 配置ID
    - dataset_name (str): 数据集名称
    
    方法:
    1. 搜索可能的final_model.pth路径（支持多种目录结构）
    2. 如果找到该文件且大小>0，说明训练已完成
    
    出参:
    - exists (bool): 是否存在final_model.pth文件
    - model_path (str or None): final_model.pth的路径（如果存在）
    """
    # 可能的final_model路径（按优先级排序）
    possible_paths = [
        os.path.join(dataset_save_dir, f"config_{config_id}", dataset_name, "final_model.pth"),
        os.path.join(dataset_save_dir, f"config_{config_id}_config_{config_id}", dataset_name, "final_model.pth"),
        os.path.join(dataset_save_dir, f"config_{config_id}", "final_model.pth"),
        os.path.join(dataset_save_dir, f"config_{config_id}_config_{config_id}", "final_model.pth"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return True, path
    
    return False, None


def find_checkpoint(dataset_save_dir, config_id, dataset_name):
    """
    查找指定配置的checkpoint文件用于断点续训
    
    入参:
    - dataset_save_dir (str): 数据集保存根目录
    - config_id (int): 配置ID
    - dataset_name (str): 数据集名称
    
    方法:
    1. 搜索可能的checkpoint.pth.tar路径（支持多种目录结构）
    2. 验证checkpoint文件是否存在且有效（文件大小>0）
    3. 优先返回包含数据集名称的路径
    
    出参:
    - checkpoint_path (str or None): 找到的checkpoint路径，未找到返回None
    """
    # 可能的checkpoint路径（按优先级排序）
    possible_paths = [
        # 优先查找包含数据集名称的路径
        os.path.join(dataset_save_dir, f"config_{config_id}", dataset_name, "checkpoint.pth.tar"),
        os.path.join(dataset_save_dir, f"config_{config_id}_config_{config_id}", dataset_name, "checkpoint.pth.tar"),
        # 其次查找直接在配置目录下的checkpoint
        os.path.join(dataset_save_dir, f"config_{config_id}", "checkpoint.pth.tar"),
        os.path.join(dataset_save_dir, f"config_{config_id}_config_{config_id}", "checkpoint.pth.tar"),
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path
    
    return None


def run_ablation_experiment(dataset_config, args, configs_to_run):
    """
    在单个数据集上运行消融实验
    
    入参:
    - dataset_config (dict): 数据集配置字典
    - args: 命令行参数
    - configs_to_run (list): 要运行的配置ID列表
    
    方法:
    1. 检查数据集路径
    2. 构建训练命令
    3. 执行消融实验训练
    4. 记录训练结果
    
    出参:
    - success (bool): 训练是否成功
    - duration (float): 训练耗时（秒）
    - config_results (dict): 各配置的结果
    """
    dataset_name = dataset_config['name']
    dataset_path = dataset_config['path']
    dataset_desc = dataset_config['description']
    
    # 使用数据集推荐的批次大小（如果用户未指定）
    batch_size = args.batch_size if args.batch_size is not None else dataset_config.get('batch_size', 4)
    
    # 检查数据集路径是否存在
    if not os.path.exists(dataset_path):
        print(f"\n❌ 错误: 数据集路径不存在: {dataset_path}")
        print(f"   请检查数据集是否已正确放置。")
        return False, 0, {}
    
    # 为该数据集创建保存目录
    dataset_save_dir = os.path.join(args.save_dir, dataset_name)
    
    # --- 智能训练状态检测 ---
    # 检查每个配置的训练状态，过滤出需要训练的配置
    configs_need_training = []
    configs_completed = []
    configs_resume = {}  # 配置ID -> checkpoint路径的映射
    
    for config_id in configs_to_run:
        # 1. 检查是否已完成训练（存在final_model.pth）
        has_final_model, final_model_path = check_final_model_exists(dataset_save_dir, config_id, dataset_name)
        
        if has_final_model:
            # 训练已完成，跳过
            configs_completed.append(config_id)
            print(f"✅ 配置{config_id} ({ABLATION_CONFIGS.get(config_id, '未知')}) 已完成训练")
            print(f"   找到final_model: {final_model_path}")
            continue
        
        # 2. 检查是否存在checkpoint（可以断点续训）
        checkpoint_path = find_checkpoint(dataset_save_dir, config_id, dataset_name)
        
        if checkpoint_path:
            # 找到checkpoint，将自动恢复训练
            configs_need_training.append(config_id)
            configs_resume[config_id] = checkpoint_path
            print(f"🔄 配置{config_id} ({ABLATION_CONFIGS.get(config_id, '未知')}) 将从checkpoint恢复训练")
            print(f"   Checkpoint: {checkpoint_path}")
        else:
            # 没有checkpoint，从头训练
            configs_need_training.append(config_id)
            print(f"🆕 配置{config_id} ({ABLATION_CONFIGS.get(config_id, '未知')}) 将从头开始训练")
    
    # 如果所有配置都已完成，直接返回
    if not configs_need_training:
        print(f"\n✅ 所有配置已完成训练，跳过数据集 {dataset_name}")
        print_separator()
        # 读取已完成的结果
        config_results = read_ablation_results(dataset_save_dir)
        return True, 0, config_results
    
    print(f"\n📋 训练状态总结:")
    print(f"   需要训练的配置: {configs_need_training}")
    print(f"   已完成的配置: {configs_completed}")
    print(f"   断点续训的配置: {list(configs_resume.keys())}")
    print_separator('-')
    
    # 构建基础训练命令
    cmd = [
        'python', 'scripts/train/train_ablation.py',
        '--file_root', dataset_path,
        '--save_dir', dataset_save_dir,
        '--batch_size', str(batch_size),
        '--gpu_id', args.gpu_id,
        '--max_steps', str(args.max_steps),
        '--num_workers', str(args.num_workers),
        '--learning_rate', str(args.learning_rate),
        '--val_interval', str(args.val_interval),
        '--log_iter', str(args.log_iter),
        '--pretrained', args.pretrained,
    ]
    
    # 添加配置参数（只训练需要的配置）
    if len(configs_need_training) == 7:  # 如果是全部7个配置
        cmd.extend(['--run_all_configs'])
    else:
        cmd.extend(['--configs', ','.join(map(str, configs_need_training))])
    
    # 添加自动恢复标志（让train_ablation.py自动检测各配置的checkpoint）
    cmd.extend(['--auto_resume'])
    
    # 打印训练信息
    print_header(f"开始消融实验: {dataset_name} ({dataset_desc})")
    print(f"📁 数据集路径: {dataset_path}")
    print(f"💾 保存目录: {dataset_save_dir}")
    print(f"🎯 批次大小: {batch_size}")
    print(f"🔢 最大步数: {args.max_steps}")
    print(f"📊 GPU设备: {args.gpu_id}")
    
    if args.run_all_configs:
        print(f"🔬 运行配置: 所有配置 (1-7)")
    else:
        print(f"🔬 运行配置: {configs_to_run}")
        for config_id in configs_to_run:
            print(f"   配置{config_id}: {ABLATION_CONFIGS.get(config_id, '未知')}")
    
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
        print(f"✅ {dataset_name} 消融实验训练成功完成!")
        print(f"⏰ 训练结束时间: {end_datetime}")
        print(f"⏱️  总耗时: {format_duration(duration)}")
        print_separator()
        
        # 尝试读取结果报告
        config_results = read_ablation_results(dataset_save_dir)
        
        return True, duration, config_results
        
    except subprocess.CalledProcessError as e:
        # 训练失败
        end_time = time.time()
        duration = end_time - start_time
        
        print_separator()
        print(f"❌ {dataset_name} 消融实验训练失败!")
        print(f"错误码: {e.returncode}")
        print(f"⏱️  已耗时: {format_duration(duration)}")
        print_separator()
        
        return False, duration, {}
    
    except KeyboardInterrupt:
        # 用户中断
        print_separator()
        print(f"\n⚠️  训练被用户中断!")
        print_separator()
        raise


def read_ablation_results(save_dir):
    """
    读取消融实验结果报告，包含所有评估指标
    
    入参:
    - save_dir (str): 实验保存目录
    
    出参:
    - results (dict): 配置ID -> 指标字典的映射
                      每个指标字典包含: {'F1', 'IoU', 'Kappa', 'OA', 'recall', 'precision'}
    """
    results = {}
    
    # 遍历每个配置目录，读取测试结果
    for config_id in range(1, 8):
        # 尝试多种可能的目录名称格式
        possible_dirs = [
            os.path.join(save_dir, f"config_{config_id}_config_{config_id}"),
            os.path.join(save_dir, f"config_{config_id}"),
        ]
        
        for config_dir in possible_dirs:
            # 寻找数据集子目录
            if os.path.exists(config_dir):
                # 查找数据集子目录（可能是GVLM-CD、WHU-CD或LBFD-CD）
                subdirs = [d for d in os.listdir(config_dir) 
                          if os.path.isdir(os.path.join(config_dir, d))]
                
                for subdir in subdirs:
                    log_file = os.path.join(config_dir, subdir, 'train_val_log.txt')
                    if os.path.exists(log_file):
                        # 读取日志文件，提取测试结果
                        metrics = parse_test_results_from_log(log_file)
                        if metrics:
                            results[config_id] = metrics
                            break
                
                if config_id in results:
                    break
    return results


def parse_test_results_from_log(log_file):
    """
    从训练日志文件中解析测试结果
    
    入参:
    - log_file (str): 日志文件路径
    
    出参:
    - metrics (dict): 包含测试指标的字典，如果未找到则返回None
    """
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # 查找测试结果行
            # 格式类似: "测试结果:	 Kappa=0.8234	 IoU=0.7456	 F1=0.8567	 OA=0.9123	 召回率=0.8345	 精确率=0.8789"
            if '测试结果:' in content:
                test_line = content.split('测试结果:')[1].split('\n')[0]
                
                metrics = {}
                # 解析各个指标
                if 'Kappa=' in test_line:
                    metrics['Kappa'] = float(test_line.split('Kappa=')[1].split()[0])
                if 'IoU=' in test_line:
                    metrics['IoU'] = float(test_line.split('IoU=')[1].split()[0])
                if 'F1=' in test_line:
                    metrics['F1'] = float(test_line.split('F1=')[1].split()[0])
                if 'OA=' in test_line:
                    metrics['OA'] = float(test_line.split('OA=')[1].split()[0])
                if '召回率=' in test_line:
                    metrics['recall'] = float(test_line.split('召回率=')[1].split()[0])
                elif 'Recall=' in test_line:
                    metrics['recall'] = float(test_line.split('Recall=')[1].split()[0])
                if '精确率=' in test_line:
                    metrics['precision'] = float(test_line.split('精确率=')[1].split()[0])
                elif 'Precision=' in test_line:
                    metrics['precision'] = float(test_line.split('Precision=')[1].split()[0])
                
                return metrics if metrics else None
            
            return None
            
    except Exception as e:
        print(f"警告: 无法解析日志文件 {log_file}: {e}")
        return None


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


def print_summary(results, configs_to_run):
    """
    打印批量训练总结
    
    入参:
    - results (list): 训练结果列表，每个元素为 (dataset_name, success, duration, config_results)
    - configs_to_run (list): 运行的配置ID列表
    """
    print_header("📊 批量消融实验训练总结")
    
    total_duration = 0
    successful_datasets = 0
    failed_datasets = 0
    
    # 为每个数据集打印详细的结果表格
    for dataset_name, success, duration, config_results in results:
        status = "✅ 成功" if success else "❌ 失败"
        print(f"\n{'='*100}")
        print(f"数据集: {dataset_name}")
        print(f"状态: {status} | 耗时: {format_duration(duration)}")
        print(f"{'='*100}")
        
        if config_results:
            # 打印表格头
            print(f"{'模型配置':<35} {'F1':<10} {'IoU':<10} {'Kappa':<10} {'OA':<10} {'Recall':<10} {'Precision':<10}")
            print('-' * 100)
            
            # 打印每个配置的结果
            for config_id in sorted(configs_to_run):
                config_name = ABLATION_CONFIGS.get(config_id, '未知')
                
                if config_id in config_results:
                    metrics = config_results[config_id]
                    f1 = metrics.get('F1', 0.0)
                    iou = metrics.get('IoU', 0.0)
                    kappa = metrics.get('Kappa', 0.0)
                    oa = metrics.get('OA', 0.0)
                    recall = metrics.get('recall', 0.0)
                    precision = metrics.get('precision', 0.0)
                    
                    print(f"{config_name:<35} {f1:<10.4f} {iou:<10.4f} {kappa:<10.4f} {oa:<10.4f} {recall:<10.4f} {precision:<10.4f}")
                else:
                    print(f"{config_name:<35} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
        else:
            print("  未找到测试结果数据")
        
        total_duration += duration
        if success:
            successful_datasets += 1
        else:
            failed_datasets += 1
    
    # 打印总体统计
    print(f"\n{'='*100}")
    print(f"总体统计:")
    print(f"  总耗时: {format_duration(total_duration)}")
    print(f"  成功: {successful_datasets} 个数据集")
    print(f"  失败: {failed_datasets} 个数据集")
    print(f"{'='*100}")
    
    # 跨数据集对比（如果有多个数据集）
    if len(results) > 1 and any(r[2] for r in results):
        print(f"\n📈 跨数据集F1分数对比表:")
        print(f"{'='*100}")
        
        # 打印表头
        header = f"{'模型配置':<35}"
        for dataset_name, success, _, _ in results:
            if success:
                header += f" {dataset_name:<15}"
        print(header)
        print('-' * 100)
        
        # 打印每个配置在各数据集上的F1分数
        for config_id in configs_to_run:
            config_name = ABLATION_CONFIGS.get(config_id, '未知')
            row = f"{config_name:<35}"
            
            for dataset_name, success, duration, config_results in results:
                if success:
                    if config_id in config_results:
                        f1 = config_results[config_id].get('F1', 0.0)
                        row += f" {f1:<15.4f}"
                    else:
                        row += f" {'N/A':<15}"
            
            print(row)
        
        print(f"{'='*100}")
    
    # 保存结果到文件
    save_summary_to_file(results, configs_to_run)


def save_summary_to_file(results, configs_to_run):
    """
    将结果总结保存到文件
    
    入参:
    - results (list): 训练结果列表
    - configs_to_run (list): 运行的配置ID列表
    """
    # 如果没有结果，不保存
    if not results:
        return
    
    # 获取保存目录（使用第一个数据集的父目录）
    if results[0][2]:  # 如果有配置结果
        # 尝试从第一个结果推断保存目录
        save_dir = None
        for dataset_name, success, duration, config_results in results:
            if success:
                # 假设保存在 args.save_dir/dataset_name/ 下
                break
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"ablation_summary_{timestamp}.txt"
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("批量消融实验训练总结\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n\n")
            
            # 为每个数据集写入结果表格
            for dataset_name, success, duration, config_results in results:
                status = "成功" if success else "失败"
                f.write(f"\n{'='*100}\n")
                f.write(f"数据集: {dataset_name}\n")
                f.write(f"状态: {status} | 耗时: {format_duration(duration)}\n")
                f.write(f"{'='*100}\n")
                
                if config_results:
                    # 表格头
                    f.write(f"{'模型配置':<35} {'F1':<10} {'IoU':<10} {'Kappa':<10} {'OA':<10} {'Recall':<10} {'Precision':<10}\n")
                    f.write('-' * 100 + '\n')
                    
                    # 每个配置的结果
                    for config_id in sorted(configs_to_run):
                        config_name = ABLATION_CONFIGS.get(config_id, '未知')
                        
                        if config_id in config_results:
                            metrics = config_results[config_id]
                            f1 = metrics.get('F1', 0.0)
                            iou = metrics.get('IoU', 0.0)
                            kappa = metrics.get('Kappa', 0.0)
                            oa = metrics.get('OA', 0.0)
                            recall = metrics.get('recall', 0.0)
                            precision = metrics.get('precision', 0.0)
                            
                            f.write(f"{config_name:<35} {f1:<10.4f} {iou:<10.4f} {kappa:<10.4f} {oa:<10.4f} {recall:<10.4f} {precision:<10.4f}\n")
                        else:
                            f.write(f"{config_name:<35} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}\n")
                else:
                    f.write("  未找到测试结果数据\n")
            
            # 跨数据集对比
            if len(results) > 1 and any(r[2] for r in results):
                f.write(f"\n\n跨数据集F1分数对比表:\n")
                f.write(f"{'='*100}\n")
                
                # 表头
                header = f"{'模型配置':<35}"
                for dataset_name, success, _, _ in results:
                    if success:
                        header += f" {dataset_name:<15}"
                f.write(header + "\n")
                f.write('-' * 100 + '\n')
                
                # 每个配置的跨数据集对比
                for config_id in configs_to_run:
                    config_name = ABLATION_CONFIGS.get(config_id, '未知')
                    row = f"{config_name:<35}"
                    
                    for dataset_name, success, duration, config_results in results:
                        if success:
                            if config_id in config_results:
                                f1 = config_results[config_id].get('F1', 0.0)
                                row += f" {f1:<15.4f}"
                            else:
                                row += f" {'N/A':<15}"
                    
                    f.write(row + "\n")
                
                f.write(f"{'='*100}\n")
        
        print(f"\n✅ 结果总结已保存到: {output_file}")
        
    except Exception as e:
        print(f"\n警告: 无法保存结果总结到文件: {e}")


def main():
    """
    主函数：批量运行消融实验
    """
    parser = ArgumentParser(description='批量消融实验训练脚本')
    
    # 基本参数
    parser.add_argument('--gpu_id', type=str, default='0', help='使用的GPU ID')
    parser.add_argument('--batch_size', type=int, default=None, 
                       help='训练批次大小（默认使用各数据集推荐值）')
    parser.add_argument('--save_dir', type=str, default='./exp_ablation_batch', 
                       help='实验保存根目录')
    parser.add_argument('--max_steps', type=int, default=80000, help='最大训练步数')
    
    # 训练参数
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载工作进程数（Windows环境建议使用0避免pickle错误）')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='初始学习率')
    parser.add_argument('--val_interval', type=int, default=1, help='验证间隔（epoch）')
    parser.add_argument('--log_iter', type=int, default=20, help='日志打印间隔')
    parser.add_argument('--pretrained', type=str, 
                       default=r'model\X3D_L.pyth',
                       help='预训练X3D权重路径')
    
    # 消融实验参数
    parser.add_argument('--run_all_configs', action='store_true',
                       help='运行所有消融实验配置（1-7）')
    parser.add_argument('--configs', type=str, default=None,
                       help='指定要运行的配置ID列表，用逗号分隔，如"1,2,3,5,6,7"')
    
    # 数据集选择参数
    parser.add_argument('--datasets', type=str, default=None,
                       help='指定数据集索引列表，用逗号分隔 (0:GVLM-CD, 1:WHU-CD, 2:LBFD-CD)，默认运行所有')
    
    # 控制参数
    parser.add_argument('--start_from', type=int, default=0, 
                       help='从第几个数据集开始训练 (0: GVLM-CD, 1: WHU-CD, 2: LBFD-CD)')
    parser.add_argument('--skip_on_error', action='store_true',
                       help='如果某个数据集训练失败，是否跳过继续训练下一个')
    
    args = parser.parse_args()
    
    # 检查参数
    if not args.run_all_configs and args.configs is None:
        parser.error("必须指定 --run_all_configs 或 --configs")
    
    # 解析要运行的配置
    if args.run_all_configs:
        configs_to_run = [1, 2, 3, 4, 5, 6, 7]
    else:
        configs_to_run = [int(x.strip()) for x in args.configs.split(',')]
        # 验证配置ID
        for config_id in configs_to_run:
            if config_id not in ABLATION_CONFIGS:
                parser.error(f"无效的配置ID: {config_id}，支持的范围: 1-7")
    
    # 解析要运行的数据集
    if args.datasets is not None:
        dataset_indices = [int(x.strip()) for x in args.datasets.split(',')]
        datasets_to_run = [DATASETS[i] for i in dataset_indices if 0 <= i < len(DATASETS)]
    else:
        datasets_to_run = DATASETS
    
    # 打印脚本信息
    print_header("🚀 批量消融实验训练脚本启动")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n将在 {len(datasets_to_run)} 个数据集上运行消融实验:")
    for i, dataset in enumerate(datasets_to_run, 1):
        batch_size = args.batch_size if args.batch_size else dataset.get('batch_size', 4)
        print(f"  {i}. {dataset['name']}: {dataset['description']} (批次大小: {batch_size})")
    
    print(f"\n将运行以下消融实验配置:")
    for config_id in configs_to_run:
        print(f"  配置{config_id}: {ABLATION_CONFIGS.get(config_id, '未知')}")
    
    print_separator()
    
    # 确认开始
    if args.start_from > 0:
        print(f"\n⚠️  将从第 {args.start_from + 1} 个数据集开始训练")
    
    # 训练结果记录
    results = []
    
    try:
        # 依次在每个数据集上运行消融实验
        for i, dataset in enumerate(datasets_to_run):
            # 跳过前面的数据集
            if i < args.start_from:
                print(f"\n⏭️  跳过 {dataset['name']}")
                continue
            
            # 运行消融实验
            success, duration, config_results = run_ablation_experiment(
                dataset, args, configs_to_run
            )
            results.append((dataset['name'], success, duration, config_results))
            
            # 如果失败且不跳过错误，则终止
            if not success and not args.skip_on_error:
                print(f"\n⚠️  由于 {dataset['name']} 训练失败，批量训练终止。")
                print(f"   使用 --skip_on_error 参数可在失败时继续训练下一个数据集。")
                break
            
            # 训练间隔（让GPU冷却）
            if i < len(datasets_to_run) - 1:
                print(f"\n⏸️  等待 10 秒后开始下一个数据集的训练...\n")
                time.sleep(10)
        
        # 打印总结
        if results:
            print("\n")
            print_summary(results, configs_to_run)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  批量训练被用户中断!")
        if results:
            print("\n已完成的训练:")
            print_summary(results, configs_to_run)
    
    except Exception as e:
        print(f"\n❌ 发生未预期的错误: {e}")
        import traceback
        traceback.print_exc()
    
    print_header("批量消融实验训练脚本结束")


if __name__ == '__main__':
    main()


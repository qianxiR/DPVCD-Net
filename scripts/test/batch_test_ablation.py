#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量消融实验测试脚本

用于在多个数据集上批量运行所有消融实验配置的推理

使用示例：
python scripts/test/batch_test_ablation.py --save_dir ./exp_ablation_batch --output_dir ./test_results_ablation --batch_size 8
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# 插入当前路径以导入本地模块
sys.path.insert(0, '.')

from data.datasets_config import DATASETS


# 数据集配置（从 data/datasets_config.py 集中加载）
# 如需修改路径，请编辑 data/datasets_config.py

# 消融实验配置
ABLATION_CONFIGS = [1, 2, 3, 4, 5, 6, 7]

# 配置名称映射
CONFIG_NAMES = {
    1: "Base Only",
    2: "Base + Attention",
    3: "Base + ASPP",
    4: "Base + Transformer",
    5: "Base + Attention + ASPP",
    6: "Base + Attention + Transformer",
    7: "Base + ASPP + Transformer"
}


def run_test(dataset_path, dataset_name, config_id, save_dir, output_dir, batch_size=8, gpu_id='0'):
    """
    运行单个配置的测试
    
    入参:
    - dataset_path: 数据集路径
    - dataset_name: 数据集名称
    - config_id: 配置ID
    - save_dir: 模型保存目录
    - output_dir: 结果输出目录
    - batch_size: 批次大小
    - gpu_id: GPU ID
    
    方法:
    1. 构建模型路径
    2. 调用test_ablation.py脚本进行推理
    
    出参:
    - success (bool): 是否成功
    - message (str): 执行信息
    """
    # 构建模型路径
    model_path = os.path.join(save_dir, dataset_name, f"config_{config_id}_config_{config_id}", dataset_name, 'best_model.pth')
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        return False, f"模型文件不存在: {model_path}"
    
    config_name = CONFIG_NAMES.get(config_id, f"config_{config_id}")
    print(f"\n{'='*80}")
    print(f"开始测试: {dataset_name} - 配置{config_id} ({config_name})")
    print(f"数据集路径: {dataset_path}")
    print(f"模型路径: {model_path}")
    print(f"{'='*80}\n")
    
    # 构建命令
    cmd = [
        sys.executable,
        'scripts/test/test_ablation.py',
        '--file_root', dataset_path,
        '--model_path', model_path,
        '--ablation_config', str(config_id),
        '--batch_size', str(batch_size),
        '--gpu_id', gpu_id,
        '--save_predictions',
        '--output_dir', output_dir
    ]
    
    try:
        # 执行命令
        start_time = time.time()
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        print(f"\n✅ 测试完成: {dataset_name} - 配置{config_id} ({config_name})")
        print(f"   耗时: {elapsed_time:.2f}秒 ({elapsed_time/60:.2f}分钟)\n")
        
        return True, f"成功完成，耗时{elapsed_time:.2f}秒"
        
    except subprocess.CalledProcessError as e:
        error_msg = f"测试失败: {dataset_name} - 配置{config_id} ({config_name})\n错误: {e}"
        print(f"\n❌ {error_msg}\n")
        return False, error_msg
    except Exception as e:
        error_msg = f"测试异常: {dataset_name} - 配置{config_id} ({config_name})\n异常: {e}"
        print(f"\n❌ {error_msg}\n")
        return False, error_msg


def batch_test_all_configs(save_dir, output_dir, batch_size=8, gpu_id='0'):
    """
    批量运行所有配置在所有数据集上的测试
    
    入参:
    - save_dir: 模型保存目录
    - output_dir: 结果输出目录
    - batch_size: 批次大小
    - gpu_id: GPU ID
    
    方法:
    1. 遍历所有数据集
    2. 遍历所有配置
    3. 依次执行测试
    
    出参:
    - results (dict): 测试结果统计
    """
    print(f"\n{'='*80}")
    print(f"开始批量消融实验测试")
    print(f"模型目录: {save_dir}")
    print(f"输出目录: {output_dir}")
    print(f"批次大小: {batch_size}")
    print(f"GPU ID: {gpu_id}")
    print(f"{'='*80}\n")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计结果
    results = {}
    total_tests = 0
    success_count = 0
    fail_count = 0
    
    start_time = time.time()
    
    # 遍历所有数据集和配置
    for dataset_config in DATASETS:
        dataset_name = dataset_config['name']
        dataset_path = dataset_config['path']
        
        results[dataset_name] = {}
        
        for config_id in ABLATION_CONFIGS:
            total_tests += 1
            config_name = CONFIG_NAMES.get(config_id, f"config_{config_id}")
            
            success, message = run_test(
                dataset_path=dataset_path,
                dataset_name=dataset_name,
                config_id=config_id,
                save_dir=save_dir,
                output_dir=output_dir,
                batch_size=batch_size,
                gpu_id=gpu_id
            )
            
            results[dataset_name][config_id] = {
                'success': success,
                'message': message,
                'config_name': config_name
            }
            
            if success:
                success_count += 1
            else:
                fail_count += 1
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 生成测试报告
    report_path = os.path.join(output_dir, 'batch_test_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("批量消融实验测试报告\n")
        f.write("="*80 + "\n\n")
        f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总测试数: {total_tests}\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {fail_count}\n")
        f.write(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)\n\n")
        
        f.write("-"*80 + "\n")
        f.write("详细结果:\n")
        f.write("-"*80 + "\n\n")
        
        for dataset_name, dataset_results in results.items():
            f.write(f"\n数据集: {dataset_name}\n")
            f.write("-"*80 + "\n")
            for config_id, result in dataset_results.items():
                status = "✅ 成功" if result['success'] else "❌ 失败"
                f.write(f"  配置{config_id} ({result['config_name']}): {status}\n")
                if not result['success']:
                    f.write(f"    错误: {result['message']}\n")
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"批量测试完成")
    print(f"{'='*80}")
    print(f"总测试数: {total_tests}")
    print(f"成功: {success_count}")
    print(f"失败: {fail_count}")
    print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"测试报告已保存到: {report_path}")
    print(f"{'='*80}\n")
    
    return results


def main():
    """
    主函数：解析参数并执行批量测试
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='批量消融实验测试脚本')
    parser.add_argument('--save_dir', type=str, default='./exp_ablation_batch',
                       help='模型保存目录（默认: ./exp_ablation_batch）')
    parser.add_argument('--output_dir', type=str, default='./test_results_ablation',
                       help='结果输出目录（默认: ./test_results_ablation）')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小（默认: 8）')
    parser.add_argument('--gpu_id', type=str, default='0',
                       help='GPU ID（默认: 0）')
    
    args = parser.parse_args()
    
    # 执行批量测试
    batch_test_all_configs(
        save_dir=args.save_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        gpu_id=args.gpu_id
    )


if __name__ == '__main__':
    main()

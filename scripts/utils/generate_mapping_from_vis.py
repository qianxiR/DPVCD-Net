#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
根据已有的可视化结果生成路径映射文件

使用方法：
# 为单个配置生成映射文件
python scripts/utils/generate_mapping_from_vis.py --vis_dir ./vis_results_ablation/GVLM-CD/config_6 --file_root "G:\deeplearning\cd\GVLM-CD"

# 为所有配置生成映射文件
python scripts/utils/generate_mapping_from_vis.py --vis_dir ./vis_results_ablation/GVLM-CD --file_root "G:\deeplearning\cd\GVLM-CD" --all_configs
python scripts/utils/generate_mapping_from_vis.py --vis_dir ./vis_results_ablation/WHU-CD --file_root "G:\deeplearning\实验数据\WHU-CD" --all_configs
python scripts/utils/generate_mapping_from_vis.py --vis_dir ./vis_results_ablation/LBFD-CD --file_root "G:\deeplearning\实验数据\LBFD-CD" --all_configs



"""

import os
import json
import argparse
import glob
from os.path import join, basename, normpath


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


def extract_config_from_filename(filename):
    """
    从文件名中提取配置ID和原始文件名
    
    入参:
    - filename (str): 可视化结果文件名
    
    出参:
    - config_id (int): 配置ID，如果无法识别返回None
    - original_filename (str): 原始文件名
    """
    # 尝试匹配配置名称
    for config_id, config_name in CONFIG_NAMES.items():
        prefix = f"{config_name}_"
        if filename.startswith(prefix):
            original_filename = filename[len(prefix):]
            return config_id, original_filename
    
    # 如果无法匹配，尝试从目录名提取配置ID
    return None, filename


def find_config_id_from_dir(vis_dir):
    """
    从目录名中提取配置ID
    
    入参:
    - vis_dir (str): 可视化结果目录路径
    
    出参:
    - config_id (int): 配置ID，如果无法识别返回None
    """
    dir_name = basename(normpath(vis_dir))
    if dir_name.startswith('config_'):
        try:
            config_id = int(dir_name.split('_')[1])
            return config_id
        except:
            pass
    return None


def generate_mapping_for_config(vis_dir, file_root, config_id=None):
    """
    为单个配置生成路径映射文件
    
    入参:
    - vis_dir (str): 可视化结果目录路径
    - file_root (str): 数据集根目录路径
    - config_id (int): 配置ID，如果为None则尝试从目录名推断
    
    方法:
    1. 扫描可视化目录中的所有图像文件
    2. 从文件名提取原始文件名
    3. 根据数据集结构构建原始路径
    4. 生成映射文件
    
    出参:
    - mapping (dict): 路径映射字典
    - config_id (int): 实际使用的配置ID
    """
    if not os.path.exists(vis_dir):
        raise FileNotFoundError(f"可视化目录不存在: {vis_dir}")
    
    # 确定配置ID
    if config_id is None:
        config_id = find_config_id_from_dir(vis_dir)
        if config_id is None:
            raise ValueError(f"无法从目录名推断配置ID，请手动指定: {vis_dir}")
    
    config_name = CONFIG_NAMES.get(config_id, f"config_{config_id}")
    
    # 获取数据集名称
    dataset_name = basename(normpath(file_root))
    
    # 检查数据集目录结构
    test_dir = join(file_root, 'test')
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"测试集目录不存在: {test_dir}")
    
    t1_dir = join(test_dir, 't1')
    t2_dir = join(test_dir, 't2')
    label_dir = join(test_dir, 'label')
    
    if not all(os.path.exists(d) for d in [t1_dir, t2_dir, label_dir]):
        raise FileNotFoundError(f"数据集目录结构不完整，需要包含 test/t1, test/t2, test/label")
    
    # 扫描可视化结果文件
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.TIF', '*.TIFF', '*.tiff']
    vis_files = []
    for ext in image_extensions:
        vis_files.extend(glob.glob(join(vis_dir, ext)))
        vis_files.extend(glob.glob(join(vis_dir, ext.upper())))
    
    if len(vis_files) == 0:
        print(f"[警告] 在 {vis_dir} 中未找到图像文件")
        return {}, config_id
    
    print(f"[信息] 找到 {len(vis_files)} 个可视化结果文件")
    
    # 生成映射
    mapping = {}
    matched_count = 0
    skipped_count = 0
    
    # 添加进度提示
    total_files = len(vis_files)
    print(f"[信息] 开始处理 {total_files} 个文件...")
    
    # 预先排序文件列表（只排序一次，避免在循环中重复排序）
    sorted_filenames = sorted([basename(f) for f in vis_files])
    filename_to_index = {name: idx + 1 for idx, name in enumerate(sorted_filenames)}
    
    for idx, vis_file in enumerate(vis_files):
        # 每处理100个文件显示一次进度
        if (idx + 1) % 100 == 0 or (idx + 1) == total_files:
            print(f"[进度] 处理中: {idx + 1}/{total_files} ({100*(idx+1)/total_files:.1f}%), 已匹配: {matched_count}, 已跳过: {skipped_count}")
        
        vis_filename = basename(vis_file)
        
        # 提取原始文件名
        _, original_filename = extract_config_from_filename(vis_filename)
        
        # 如果提取失败，尝试从文件名中移除配置名称前缀
        if original_filename == vis_filename:
            # 尝试移除配置名称前缀
            for cfg_name in CONFIG_NAMES.values():
                prefix = f"{cfg_name}_"
                if vis_filename.startswith(prefix):
                    original_filename = vis_filename[len(prefix):]
                    break
        
        # 构建原始路径
        original_pre_path = join(t1_dir, original_filename)
        original_post_path = join(t2_dir, original_filename)
        original_label_path = join(label_dir, original_filename)
        
        # 检查文件是否存在
        if not os.path.exists(original_pre_path):
            skipped_count += 1
            if skipped_count <= 5:  # 只显示前5个警告
                print(f"[警告] 原始T1图像不存在: {basename(original_pre_path)}")
            continue
        
        if not os.path.exists(original_post_path):
            skipped_count += 1
            if skipped_count <= 5:  # 只显示前5个警告
                print(f"[警告] 原始T2图像不存在: {basename(original_post_path)}")
            continue
        
        # 确定图像索引（使用预先构建的映射）
        image_index = filename_to_index.get(vis_filename, idx + 1)
        
        # 添加到映射
        mapping[vis_filename] = {
            'visualization_file': vis_file,
            'original_pre_image': original_pre_path,
            'original_post_image': original_post_path,
            'original_label': original_label_path if os.path.exists(original_label_path) else None,
            'image_index': image_index,
            'dataset_name': dataset_name,
            'config_id': config_id,
            'config_name': config_name
        }
        matched_count += 1
    
    print(f"[成功] 成功匹配 {matched_count}/{len(vis_files)} 个文件")
    
    return mapping, config_id


def save_mapping(mapping, output_file):
    """
    保存映射文件
    
    入参:
    - mapping (dict): 路径映射字典
    - output_file (str): 输出文件路径
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    
    print(f"[保存] 映射文件已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='根据已有的可视化结果生成路径映射文件')
    
    parser.add_argument('--vis_dir', type=str, required=True,
                       help='可视化结果目录路径（可以是单个配置目录或数据集目录）')
    parser.add_argument('--file_root', type=str, required=True,
                       help='数据集根目录路径')
    parser.add_argument('--all_configs', action='store_true',
                       help='为所有配置生成映射文件（vis_dir应该是数据集目录）')
    parser.add_argument('--config_id', type=int, default=None,
                       help='配置ID（如果指定，则只处理该配置）')
    
    args = parser.parse_args()
    
    if args.all_configs:
        # 为所有配置生成映射文件
        dataset_name = basename(normpath(args.vis_dir))
        base_vis_dir = args.vis_dir
        
        print(f"[信息] 为数据集 {dataset_name} 的所有配置生成映射文件...")
        
        total_mappings = 0
        for config_id in range(1, 8):
            config_dir = join(base_vis_dir, f'config_{config_id}')
            if not os.path.exists(config_dir):
                print(f"[跳过] 配置 {config_id}（目录不存在）")
                continue
            
            print(f"\n{'='*60}")
            print(f"处理配置 {config_id}: {CONFIG_NAMES.get(config_id, 'Unknown')}")
            print(f"{'='*60}")
            
            try:
                mapping, actual_config_id = generate_mapping_for_config(
                    config_dir, args.file_root, config_id
                )
                
                if len(mapping) > 0:
                    mapping_file = join(config_dir, 'path_mapping.json')
                    save_mapping(mapping, mapping_file)
                    total_mappings += len(mapping)
            except Exception as e:
                print(f"[错误] 配置 {config_id} 处理失败: {e}")
        
        print(f"\n[完成] 共生成 {total_mappings} 个映射条目")
    else:
        # 为单个配置生成映射文件
        print(f"[信息] 生成路径映射文件...")
        print(f"可视化目录: {args.vis_dir}")
        print(f"数据集根目录: {args.file_root}")
        
        try:
            mapping, config_id = generate_mapping_for_config(
                args.vis_dir, args.file_root, args.config_id
            )
            
            if len(mapping) > 0:
                mapping_file = join(args.vis_dir, 'path_mapping.json')
                save_mapping(mapping, mapping_file)
                print(f"\n[完成] 共生成 {len(mapping)} 个映射条目")
            else:
                print(f"\n[警告] 未生成任何映射条目，请检查目录和文件")
        except Exception as e:
            print(f"\n[错误] 生成映射文件失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()


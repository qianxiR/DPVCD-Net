#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
根据可视化结果图片查找并复制对应的原始图像

使用方法：
# 从指定文件夹复制原图到目标文件夹
python scripts/copy_original_images.py --source_dir "C:\Users\Administrator\Desktop\1" --target_dir "C:\Users\Administrator\Desktop\2" --mapping_file ./vis_results_ablation/GVLM-CD/config_6/path_mapping.json
python scripts/copy_original_images.py --source_dir "C:\Users\Administrator\Desktop\1" --target_dir "C:\Users\Administrator\Desktop\2" --mapping_file ./vis_results_ablation/WHU-CD/config_6/path_mapping.json
python scripts/copy_original_images.py --source_dir "C:\Users\Administrator\Desktop\1" --target_dir "C:\Users\Administrator\Desktop\2" --mapping_file ./vis_results_ablation/LBFD-CD/config_6/path_mapping.json

# 自动查找映射文件（从可视化结果目录）
python scripts/copy_original_images.py --source_dir "C:\Users\Administrator\Desktop\1" --target_dir "C:\Users\Administrator\Desktop\2" --vis_dir ./vis_results_ablation/GVLM-CD/config_6

# 复制多个数据集的原图（自动查找映射文件）
python scripts/copy_original_images.py --source_dir "C:\Users\Administrator\Desktop\1" --target_dir "C:\Users\Administrator\Desktop\2" --vis_base_dir ./vis_results_ablation
"""

import os
import json
import shutil
import argparse
import glob
from os.path import join, basename, normpath, dirname, exists


def load_mapping(mapping_file):
    """
    加载路径映射文件
    
    入参:
    - mapping_file (str): 映射文件路径
    
    出参:
    - mapping (dict): 路径映射字典
    """
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"映射文件不存在: {mapping_file}")
    
    with open(mapping_file, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    
    return mapping


def find_mapping_file(source_dir, vis_base_dir=None):
    """
    自动查找映射文件
    
    入参:
    - source_dir (str): 源目录
    - vis_base_dir (str): 可视化结果基础目录
    
    出参:
    - mapping_file (str): 映射文件路径，如果未找到返回None
    """
    # 方法1: 在源目录中查找
    possible_mapping = join(source_dir, 'path_mapping.json')
    if exists(possible_mapping):
        return possible_mapping
    
    # 方法2: 在父目录中查找
    parent_mapping = join(dirname(source_dir), 'path_mapping.json')
    if exists(parent_mapping):
        return parent_mapping
    
    # 方法3: 在可视化结果目录中查找
    if vis_base_dir and exists(vis_base_dir):
        # 尝试查找所有可能的映射文件
        for root, dirs, files in os.walk(vis_base_dir):
            if 'path_mapping.json' in files:
                return join(root, 'path_mapping.json')
    
    return None


def get_image_files(source_dir):
    """
    获取源目录中的所有图像文件
    
    入参:
    - source_dir (str): 源目录路径
    
    出参:
    - image_files (list): 图像文件路径列表
    """
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(join(source_dir, '**', ext), recursive=True))
    
    return sorted(image_files)


def find_original_paths(vis_filename, mapping):
    """
    根据可视化文件名查找原始图像路径
    
    入参:
    - vis_filename (str): 可视化结果文件名
    - mapping (dict): 路径映射字典
    
    出参:
    - result (dict): 包含原始路径信息的字典，如果未找到返回None
    """
    # 尝试完整文件名匹配
    if vis_filename in mapping:
        return mapping[vis_filename]
    
    # 尝试只匹配文件名（去掉路径）
    basename_file = basename(vis_filename)
    if basename_file in mapping:
        return mapping[basename_file]
    
    # 尝试去掉配置名称前缀匹配
    for key, value in mapping.items():
        # 如果映射中的键包含配置名称前缀，尝试匹配
        if basename_file.endswith(key.split('_', 1)[-1] if '_' in key else key):
            return value
    
    return None


def copy_original_images(source_dir, target_dir, mapping_file=None, vis_base_dir=None, copy_structure=True):
    """
    复制原始图像到目标目录
    
    入参:
    - source_dir (str): 源目录（包含可视化结果图片）
    - target_dir (str): 目标目录（复制原图的位置）
    - mapping_file (str): 映射文件路径（可选）
    - vis_base_dir (str): 可视化结果基础目录（用于自动查找映射文件）
    - copy_structure (bool): 是否保持相同的目录结构
    
    方法:
    1. 扫描源目录中的所有图像文件
    2. 根据映射文件查找对应的原始图像
    3. 按照相同的目录结构复制到目标目录
    
    出参:
    - copied_count (int): 成功复制的文件数量
    - failed_count (int): 失败的文件数量
    """
    # 查找映射文件
    if mapping_file is None:
        mapping_file = find_mapping_file(source_dir, vis_base_dir)
        if mapping_file is None:
            raise FileNotFoundError("未找到映射文件，请指定 --mapping_file 或 --vis_base_dir")
    
    print(f"📁 使用映射文件: {mapping_file}")
    
    # 加载映射
    mapping = load_mapping(mapping_file)
    print(f"✅ 加载了 {len(mapping)} 个映射条目")
    
    # 获取源目录中的所有图像文件
    image_files = get_image_files(source_dir)
    print(f"📊 在源目录中找到 {len(image_files)} 个图像文件")
    
    if len(image_files) == 0:
        print("⚠️  源目录中没有找到图像文件")
        return 0, 0
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    
    copied_count = 0
    failed_count = 0
    failed_files = []
    
    print(f"\n开始复制原始图像...")
    print(f"{'='*60}")
    
    for vis_file in image_files:
        # 计算相对路径（用于保持目录结构）
        rel_path = os.path.relpath(vis_file, source_dir)
        vis_filename = basename(vis_file)
        
        # 查找原始图像路径
        result = find_original_paths(vis_filename, mapping)
        
        if result is None:
            print(f"⚠️  未找到映射: {vis_filename}")
            failed_count += 1
            failed_files.append(vis_file)
            continue
        
        # 确定目标路径
        if copy_structure:
            # 保持相同的目录结构
            target_subdir = join(target_dir, dirname(rel_path))
            os.makedirs(target_subdir, exist_ok=True)
            target_file = join(target_subdir, basename(rel_path))
        else:
            # 直接放在目标目录
            target_file = join(target_dir, basename(rel_path))
        
        # 复制原始图像（T1, T2, Label）
        files_to_copy = []
        
        # T1图像
        if exists(result['original_pre_image']):
            files_to_copy.append(('T1', result['original_pre_image']))
        
        # T2图像
        if exists(result['original_post_image']):
            files_to_copy.append(('T2', result['original_post_image']))
        
        # 标签
        if result.get('original_label') and exists(result['original_label']):
            files_to_copy.append(('Label', result['original_label']))
        
        if len(files_to_copy) == 0:
            print(f"⚠️  原始图像不存在: {vis_filename}")
            failed_count += 1
            failed_files.append(vis_file)
            continue
        
        # 复制文件
        for file_type, src_file in files_to_copy:
            # 构建目标文件名
            if copy_structure:
                # 保持目录结构，但需要区分T1/T2/Label
                if file_type == 'T1':
                    tgt_file = join(target_subdir, 't1', basename(src_file))
                elif file_type == 'T2':
                    tgt_file = join(target_subdir, 't2', basename(src_file))
                else:  # Label
                    tgt_file = join(target_subdir, 'label', basename(src_file))
                
                os.makedirs(dirname(tgt_file), exist_ok=True)
            else:
                # 直接复制，添加类型前缀
                tgt_file = join(target_dir, f"{file_type}_{basename(src_file)}")
            
            try:
                shutil.copy2(src_file, tgt_file)
                print(f"✅ {file_type}: {basename(src_file)} -> {tgt_file}")
            except Exception as e:
                print(f"❌ 复制失败 {file_type}: {src_file} -> {tgt_file}: {e}")
                failed_count += 1
        
        copied_count += 1
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"📊 复制完成统计:")
    print(f"   成功: {copied_count} 个图像")
    print(f"   失败: {failed_count} 个图像")
    print(f"{'='*60}")
    
    if failed_files:
        print(f"\n⚠️  失败的文件列表:")
        for f in failed_files[:10]:  # 只显示前10个
            print(f"   - {f}")
        if len(failed_files) > 10:
            print(f"   ... 还有 {len(failed_files) - 10} 个文件")
    
    return copied_count, failed_count


def main():
    parser = argparse.ArgumentParser(description='根据可视化结果图片查找并复制对应的原始图像')
    
    parser.add_argument('--source_dir', type=str, required=True,
                       help='源目录（包含可视化结果图片）')
    parser.add_argument('--target_dir', type=str, required=True,
                       help='目标目录（复制原图的位置）')
    parser.add_argument('--mapping_file', type=str, default=None,
                       help='映射文件路径（可选，如果不指定会自动查找）')
    parser.add_argument('--vis_dir', type=str, default=None,
                       help='可视化结果目录（用于自动查找映射文件）')
    parser.add_argument('--vis_base_dir', type=str, default=None,
                       help='可视化结果基础目录（用于自动查找映射文件）')
    parser.add_argument('--flat', action='store_true',
                       help='不保持目录结构，所有文件直接放在目标目录')
    
    args = parser.parse_args()
    
    # 确定可视化基础目录
    vis_base_dir = args.vis_base_dir or args.vis_dir
    
    print(f"📁 源目录: {args.source_dir}")
    print(f"📁 目标目录: {args.target_dir}")
    print(f"📁 保持目录结构: {not args.flat}")
    
    try:
        copied, failed = copy_original_images(
            args.source_dir,
            args.target_dir,
            mapping_file=args.mapping_file,
            vis_base_dir=vis_base_dir,
            copy_structure=not args.flat
        )
        
        if copied > 0:
            print(f"\n✅ 成功复制 {copied} 个图像的原图到: {args.target_dir}")
        else:
            print(f"\n⚠️  没有成功复制任何文件")
            
    except Exception as e:
        print(f"\n❌ 复制失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()


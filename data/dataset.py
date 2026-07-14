# Copyright (c) Duowang Zhu.
# All rights reserved.

import cv2
from skimage import io
import numpy as np
import torch.utils.data
import os
import h5py
import json
import re
import glob
from os.path import join as osp
from typing import List, Tuple, Optional, Dict, Any, Union


class BCDDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading and processing bi-temporal remote sensing images.
    
    This class handles loading image pairs (pre and post) along with their 
    corresponding change detection ground truth masks.
    """

    def __init__(self, file_root: str, split: str, transform = None):
        """
        Initialize the dataset with paths to images and transforms.
        
        Args:
            file_root: Root directory containing dataset folders
            split: Dataset split name (e.g., 'train', 'val', 'test')
            transform: Optional transform to apply to images and labels
        """
        data_root = osp(file_root, split)
        if not os.path.exists(data_root):
            raise FileNotFoundError(f"Data split root path does not exist: {data_root}")
        
        # Initialize lists for valid image paths
        self.pre_images = []
        self.post_images = []
        self.label_change = []

        # 检查必需的目录
        label_dir = osp(data_root, 'label')
        t1_dir = osp(data_root, 't1')
        t2_dir = osp(data_root, 't2')
        for d in [label_dir, t1_dir, t2_dir]:
            if not os.path.isdir(d):
                raise FileNotFoundError(f"必需的目录不存在: {d}")
        
        # 辅助函数：从label文件名中提取row和col信息
        def extract_row_col(label_filename):
            """
            从label文件名中提取row和col信息
            支持两种格式：
            1. WHU-CD格式: change_label_r0000_c0000.tif -> 'r0000_c0000'
            2. LBFD-CD格式: jiuzaigou_000_002.png -> None (直接使用文件名)
            """
            match = re.search(r'(r\d+_c\d+)', label_filename)
            if match:
                return match.group(1)  # 返回 'r0000_c0000'
            return None
        
        # 辅助函数：在目录中查找匹配的文件
        def find_matching_file(directory, filename, pattern=None):
            """
            在指定目录中查找匹配的文件
            优先尝试直接文件名匹配，如果失败则使用pattern匹配
            
            入参:
            - directory: 目录路径
            - filename: label文件名（用于直接匹配）
            - pattern: 可选的匹配模式（用于WHU-CD格式）
            
            出参:
            - 匹配的文件路径，如果找不到则返回None
            """
            # 方法1: 直接文件名匹配（适用于LBFD-CD格式，文件名完全相同）
            direct_path = osp(directory, filename)
            if os.path.exists(direct_path):
                return direct_path
            
            # 方法2: 如果提供了pattern，使用pattern匹配（适用于WHU-CD格式）
            if pattern:
                # 构建glob模式，匹配任意前缀 + pattern + 文件扩展名
                file_ext = os.path.splitext(filename)[1]  # 获取扩展名，如 .tif 或 .png
                glob_pattern = osp(directory, f'*{pattern}{file_ext}')
                matches = glob.glob(glob_pattern)
                if matches:
                    return matches[0]  # 返回第一个匹配的文件
                # 如果没找到，尝试匹配任意后缀
                glob_pattern = osp(directory, f'*{pattern}*')
                matches = glob.glob(glob_pattern)
                if matches:
                    return matches[0]  # 返回第一个匹配的文件
            
            return None
            
        file_list = os.listdir(label_dir)

        for filename in file_list:
            label_path = osp(label_dir, filename)
            row_col_pattern = extract_row_col(filename)
            pre_path = find_matching_file(t1_dir, filename, row_col_pattern)
            post_path = find_matching_file(t2_dir, filename, row_col_pattern)

            if pre_path and post_path and os.path.exists(label_path):
                self.pre_images.append(pre_path)
                self.post_images.append(post_path)
                self.label_change.append(label_path)

        print(f"加载 {len(self.label_change)} 个图像对")
        
        if len(self.label_change) == 0:
            raise RuntimeError("没有找到任何有效的图像对！请检查数据集路径和文件完整性。")

        # Store transform
        self.transform = transform
        
    def __len__(self) -> int:
        """Return the number of image pairs in the dataset."""
        return len(self.label_change)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the image pair and label at the specified index.
        
        Args:
            idx: Index of the sample to fetch
            
        Returns:
            Tuple containing:
                - Concatenated pre/post images [H, W, 6]
                - Binary change detection mask [H, W]
        """
        # Load images with error handling
        try:
            pre_image = io.imread(self.pre_images[idx])
            if pre_image is None:
                raise IOError(f"Failed to load pre-change image: {self.pre_images[idx]}")
        except Exception as e:
            print(f"错误加载图像 {self.pre_images[idx]}: {e}")
            # 如果当前索引不是最后一个，尝试下一个
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            else:
                # 如果是最后一个，尝试前一个
                return self.__getitem__(idx - 1)
        
        try:
            post_image = io.imread(self.post_images[idx])
            if post_image is None:
                raise IOError(f"Failed to load post-change image: {self.post_images[idx]}")
        except Exception as e:
            print(f"错误加载图像 {self.post_images[idx]}: {e}")
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(idx - 1)
        
        try:
            label = io.imread(self.label_change[idx], as_gray=True)
            if label is None:
                raise IOError(f"Failed to load ground truth mask: {self.label_change[idx]}")
        except Exception as e:
            print(f"错误加载标签 {self.label_change[idx]}: {e}")
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            else:
                return self.__getitem__(idx - 1)
            
        # Concatenate pre and post images along the channel dimension
        # This creates a 6-channel image (BGR-BGR)
        img = np.concatenate((pre_image, post_image), axis=2)

        # Apply transforms if specified
        if self.transform:
            img, label = self.transform(img, label)

        return img, label

    def get_img_info(self, idx: int) -> Dict[str, int]:
        """
        Get image dimensions for the specified index.
        
        Args:
            idx: Index of the image
            
        Returns:
            Dictionary with image height and width
        """
        img = cv2.imread(self.pre_images[idx])
        if img is None:
            raise IOError(f"Failed to load image for info: {self.pre_images[idx]}")
            
        return {"height": img.shape[0], "width": img.shape[1]}











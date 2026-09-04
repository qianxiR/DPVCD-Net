"""
LBFD-CD 分类测试脚本：按建筑物（whu）和滑坡（非whu）分别统计 F1/IoU 等指标。

入参:
- --file_root: LBFD-CD 数据集根目录
- --model_path: Full 模型权重路径
- --pretrained: X3D 预训练权重路径

方法:
1. 加载 Full 模型（model.trainer.Trainer）和原始训练时的验证变换
2. 遍历 test 集，按文件名区分 whu（建筑物）和非 whu（滑坡）
3. 分别累积混淆矩阵，计算各自的 F1/IoU/Kappa/OA/Recall/Precision
4. 同时输出整体指标作为对照

出参:
- 打印 building / landslide / overall 三组指标
"""
import os
import sys
import re
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.utils.data
from skimage import io

sys.path.insert(0, '.')

from model.trainer import Trainer
from data.transforms import BCDTransforms
from utils.metric_tool import ConfuseMatrixMeter


class BCDSubsetDataset(torch.utils.data.Dataset):
    """
    支持 文件名筛选 的 BCD 数据集子集。

    入参:
    - file_root: 数据集根目录
    - split: train/val/test
    - transform: 数据变换
    - name_filter: 可选的文件名过滤函数，返回 True 表示包含该样本

    方法:
    - 复用 BCDDataset 的匹配逻辑加载 t1/t2/label 图像对
    - 额外记录每个样本的文件名，供分类统计使用

    出参:
    - (img, label, filename) 三元组
    """

    def __init__(self, file_root: str, split: str, transform=None, name_filter=None):
        data_root = os.path.join(file_root, split)
        label_dir = os.path.join(data_root, 'label')
        t1_dir = os.path.join(data_root, 't1')
        t2_dir = os.path.join(data_root, 't2')

        self.pre_images = []
        self.post_images = []
        self.label_change = []
        self.filenames = []

        for filename in sorted(os.listdir(label_dir)):
            if name_filter is not None and not name_filter(filename):
                continue

            label_path = os.path.join(label_dir, filename)
            pre_path = os.path.join(t1_dir, filename)
            post_path = os.path.join(t2_dir, filename)

            if os.path.exists(pre_path) and os.path.exists(post_path) and os.path.exists(label_path):
                self.pre_images.append(pre_path)
                self.post_images.append(post_path)
                self.label_change.append(label_path)
                self.filenames.append(filename)

        self.transform = transform
        print(f"  筛选后加载 {len(self.label_change)} 个图像对")

    def __len__(self):
        return len(self.label_change)

    def __getitem__(self, idx):
        pre_image = io.imread(self.pre_images[idx])
        post_image = io.imread(self.post_images[idx])
        label = io.imread(self.label_change[idx], as_gray=True)

        img = np.concatenate((pre_image, post_image), axis=2)

        if self.transform:
            img, label = self.transform(img, label)

        return img, label, self.filenames[idx]


def build_args():
    """构造测试参数，复用训练时的默认配置。"""
    parser = argparse.ArgumentParser(description='LBFD-CD 分类测试（建筑物/滑坡）')
    parser.add_argument('--file_root', type=str, default=r'E:\rqx\dataes\LBFD-CD',
                        help='LBFD-CD 数据集根目录')
    parser.add_argument('--model_path', type=str, default=r'exp_BCD\LBFD-CD\best_model.pth',
                        help='Full 模型权重路径')
    parser.add_argument('--pretrained', type=str, default=r'model\X3D_L.pyth',
                        help='X3D 预训练权重路径')
    parser.add_argument('--in_height', type=int, default=256)
    parser.add_argument('--in_width', type=int, default=256)
    parser.add_argument('--num_perception_frame', type=int, default=1)
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    return parser.parse_args()


def evaluate_subset(model, loader, meter, device):
    """
    在一个子集上推理并累积混淆矩阵。

    入参:
    - model: 已加载权重的 Full 模型
    - loader: 数据加载器
    - meter: ConfuseMatrixMeter
    - device: cuda

    方法:
    - 逐 batch 前向推理，阈值 0.5 生成二值预测
    - 累积混淆矩阵

    出参:
    - scores: 该子集的指标字典
    """
    model.eval()
    with torch.no_grad():
        for img, label, filenames in loader:
            pre_img = img[:, :3, :, :].to(device).float()
            post_img = img[:, 3:, :, :].to(device).float()
            target = label.to(device).float()

            output = model(pre_img, post_img)
            pred = torch.where(
                output > 0.5,
                torch.ones_like(output),
                torch.zeros_like(output),
            ).long()

            meter.update_cm(
                pr=pred.cpu().numpy(),
                gt=target.cpu().numpy(),
            )
    return meter.get_scores()


def main():
    args = build_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'数据集: {args.file_root}')
    print(f'模型权重: {args.model_path}')

    # 加载模型
    model = Trainer(args)

    # 先做一次 forward 创建动态层（st_fusion 的 Transformer 和 pos_encoder 的边缘投影），
    # 否则这些层的权重无法从 checkpoint 中加载
    model = model.to(device)
    with torch.no_grad():
        dummy_x = torch.randn(1, 3, args.in_height, args.in_width, device=device)
        dummy_y = torch.randn(1, 3, args.in_height, args.in_width, device=device)
        _ = model(dummy_x, dummy_y)
    model = model.cpu()

    state_dict = torch.load(args.model_path, map_location='cpu', weights_only=False)
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f'  缺失键数: {len(missing)}')
    if unexpected:
        print(f'  多余键数: {len(unexpected)}')
    model = model.to(device)
    model.eval()
    print('模型加载成功')

    # 获取验证变换（测试用同一套）
    train_transform, val_transform = BCDTransforms.get_transform_pipelines(args)

    # 文件名过滤函数
    is_building = lambda fn: re.search(r'whu', fn, re.I) is not None
    is_landslide = lambda fn: re.search(r'whu', fn, re.I) is None

    # 三个子集：建筑物、滑坡、全部
    subsets = {
        'Building (whu)': is_building,
        'Landslide (non-whu)': is_landslide,
        'Overall': None,
    }

    results = {}
    for name, name_filter in subsets.items():
        print(f'\n=== {name} ===')
        dataset = BCDSubsetDataset(
            file_root=args.file_root,
            split='test',
            transform=val_transform,
            name_filter=name_filter,
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        meter = ConfuseMatrixMeter(n_class=2)
        scores = evaluate_subset(model, loader, meter, device)
        results[name] = scores

        print(f'  样本数: {len(dataset)}')
        print(f'  F1={scores["F1"]:.4f}  IoU={scores["IoU"]:.4f}  '
              f'Kappa={scores["Kappa"]:.4f}  OA={scores["OA"]:.4f}  '
              f'Recall={scores["recall"]:.4f}  Precision={scores["precision"]:.4f}')

    # 汇总表格
    print('\n' + '=' * 80)
    print(f'{"子集":<22} {"样本数":<8} {"F1":<8} {"IoU":<8} {"Kappa":<8} {"OA":<8} {"Recall":<8} {"Precision":<8}')
    print('-' * 80)
    for name, name_filter in subsets.items():
        dataset = BCDSubsetDataset(
            file_root=args.file_root,
            split='test',
            transform=val_transform,
            name_filter=name_filter,
        )
        s = results[name]
        print(f'{name:<22} {len(dataset):<8} {s["F1"]:<8.4f} {s["IoU"]:<8.4f} '
              f'{s["Kappa"]:<8.4f} {s["OA"]:<8.4f} {s["recall"]:<8.4f} {s["precision"]:<8.4f}')
    print('=' * 80)


if __name__ == '__main__':
    main()

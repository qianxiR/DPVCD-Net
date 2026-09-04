# Copyright (c) Duowang Zhu.
# All rights reserved.

import os
import sys
import time
import json
import argparse
from os.path import join
from collections import OrderedDict

import cv2
import numpy as np
import torch
from skimage import io

# 插入当前路径以导入本地模块(做什么:使 data/model 包可被 import;为什么:脚本从仓库根目录运行)
sys.path.insert(0, '.')

import data.dataset as RSDataset
import data.transforms as RSTransforms
from model.trainer import Trainer

'''
完整模型(配置8)可视化脚本 —— 结果图 + 对比图

差异图颜色映射(BGR,与 visualize_ablation.py / analyze_diff_stats.py 保持一致):
  绿色 [0,255,0]   = TP 正确预测变化
  红色 [0,0,255]   = FP 误检(预测变化但实际未变化)
  蓝色 [255,0,0]   = FN 漏检(实际变化但未预测到)
  黑色 [0,0,0]     = TN 正确预测无变化(背景)

使用示例(Windows PowerShell,单行格式):
python scripts/visualize/visualize_full_model.py --file_root "E:\rqx\dataes\GVLM-CD" --model_path ./exp_new/GVLM-CD/best_model.pth --output_dir ./vis_full --gpu_id 0
'''

# 差异图颜色常量(BGR)——全仓库统一映射,供 analyze_diff_stats.py 解析
COLOR_TP = (0, 255, 0)    # 绿:正确预测变化
COLOR_FP = (0, 0, 255)    # 红:误检
COLOR_FN = (255, 0, 0)    # 蓝:漏检
# TN 为黑色背景,使用默认零矩阵即可,无需显式着色


def load_full_model(args, device):
    """
    创建并加载完整模型权重。

    入参:
    - args: 配置参数(需含 model_path / pretrained / in_height / in_width)
    - device: torch.device 目标设备

    方法:
    1. 实例化 Trainer(完整模型,含所有模块)
    2. 模拟一次前向传播以初始化动态模块(感知帧、位置编码等惰性参数)
    3. 加载权重并剥离 DataParallel 的 'module.' 前缀,strict=False 容错

    出参:
    - model: 已加载权重、设为 eval 的模型(已 .to(device))
    """
    model = Trainer(args=args)

    # 模拟前向传播:初始化所有动态创建的子模块,避免后续 load_state_dict 形状不匹配
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.in_height, args.in_width)
        _ = model(dummy, dummy)

    loaded = torch.load(args.model_path, map_location=device)
    state_dict = loaded.get('state_dict', loaded) if isinstance(loaded, dict) else \
        (loaded.module if hasattr(loaded, 'module') else loaded).state_dict()

    # 剥离 DataParallel 保存的 'module.' 前缀,保证键名与单卡模型一致
    cleaned = OrderedDict()
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith('module.') else k] = v

    model.load_state_dict(cleaned, strict=False)
    model.to(device)
    model.eval()
    return model


def make_diff_mask(pred_np, target_np):
    """
    由二值预测与真实标签生成 BGR 差异掩码。

    入参:
    - pred_np (np.ndarray): 二值预测,H×W,uint8,值域{0,1}
    - target_np (np.ndarray): 二值标签,H×W,uint8,值域{0,1}

    方法:
    - 按像素级逻辑运算划分 TP/FP/FN 三类并着色;TN 保持黑色背景

    出参:
    - diff_mask (np.ndarray): H×W×3 uint8 BGR 差异图
    """
    h, w = pred_np.shape
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    diff[(pred_np == 1) & (target_np == 1)] = COLOR_TP
    diff[(pred_np == 1) & (target_np == 0)] = COLOR_FP
    diff[(pred_np == 0) & (target_np == 1)] = COLOR_FN
    return diff


def concat_row(parts, h):
    """
    横向拼接若干同高图块,图块间插入 10px 黑色分隔线。

    入参:
    - parts (list[np.ndarray]): 同高度的 BGR 图块列表
    - h (int): 图块高度(用于生成分隔线)

    方法:
    - 在相邻图块间插入 spacer,末尾不加

    出参:
    - row (np.ndarray): 拼接后的单行 BGR 图像
    """
    spacer = np.zeros((h, 10, 3), dtype=np.uint8)
    interleaved = []
    for i, p in enumerate(parts):
        interleaved.append(p)
        if i < len(parts) - 1:
            interleaved.append(spacer)
    return np.concatenate(interleaved, axis=1)


def add_title_legend(row, title, legends):
    """
    在拼接图行上方加标题条、下方加图例条。

    入参:
    - row (np.ndarray): 拼接后的 BGR 图像行
    - title (str): 顶部标题文本
    - legends (list[str]): 底部图例文本列表(顺序与图块一致)

    方法:
    - 标题条 40px 高,图例条 60px 高,白字绘制

    出参:
    - final (np.ndarray): 标题+图行+图例的完整 BGR 图像
    """
    w = row.shape[1]
    title_bar = np.zeros((40, w, 3), dtype=np.uint8)
    cv2.putText(title_bar, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    legend_bar = np.zeros((60, w, 3), dtype=np.uint8)
    step = w // max(len(legends), 1)
    for i, txt in enumerate(legends):
        cv2.putText(legend_bar, txt, (i * step + 2, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return np.concatenate([title_bar, row, legend_bar], axis=0)


def visualize_full_model(args):
    """
    完整模型推理可视化主流程。

    入参:
    - args: 配置参数(需含 file_root / model_path / output_dir / gpu_id)

    方法:
    1. 加载完整模型与测试集(batch_size=1 逐张处理)
    2. 对每张图推理并生成:结果四联图、对比五联图、单独预测掩码、单独差异掩码
    3. 记录 path_mapping.json 以便反查原始文件路径

    出参:
    - 无返回值,结果写入 output_dir
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.model_path):
        print(f"[错误] 模型权重不存在: {args.model_path}")
        return

    model = load_full_model(args, device)
    print(f"[完成] 模型已加载: {args.model_path}")

    _, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)
    test_data = RSDataset.BCDDataset(file_root=args.file_root, split='test', transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # 输出目录与可视化(BCD)目录对齐,便于 analyze_diff_stats.py 统一解析
    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    out_dir = join(args.output_dir, dataset_name)
    pred_dir = join(out_dir, 'prediction_masks')
    diff_dir = join(out_dir, 'diffdata')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(diff_dir, exist_ok=True)

    path_mapping = {}
    count = 0
    with torch.no_grad():
        for i, batched in enumerate(test_loader):
            img, target = batched[0].to(device), batched[1].to(device)
            pre_img, post_img = img[:, :3], img[:, 3:]

            t0 = time.time()
            output = model(pre_img, post_img)
            dt = time.time() - t0

            pred = torch.where(output > 0.5, 1, 0)
            pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
            tgt_np = target.squeeze().cpu().numpy().astype(np.uint8)
            h, w = pred_np.shape

            # 读取原始 RGB 并缩放对齐输出分辨率(BGR 供 OpenCV 写盘)
            pre_path = test_loader.dataset.pre_images[i]
            post_path = test_loader.dataset.post_images[i]
            label_path = test_loader.dataset.label_change[i]
            pre_vis = cv2.cvtColor(cv2.resize(io.imread(pre_path), (w, h)), cv2.COLOR_RGB2BGR)
            post_vis = cv2.cvtColor(cv2.resize(io.imread(post_path), (w, h)), cv2.COLOR_RGB2BGR)
            tgt_vis = np.stack([(tgt_np * 255)] * 3, axis=-1)

            # 预测掩码:变化=白,无变化=黑(与现有 prediction_masks 格式一致)
            pred_mask = np.zeros((h, w, 3), dtype=np.uint8)
            pred_mask[pred_np == 1] = [255, 255, 255]

            diff_mask = make_diff_mask(pred_np, tgt_np)
            fname = os.path.basename(pre_path)

            # 结果图:T1|T2|GT|Pred(四联)
            result_row = concat_row([pre_vis, post_vis, tgt_vis, pred_mask], h)
            result_final = add_title_legend(
                result_row,
                f"Full Model - {fname} - {dt:.3f}s",
                ['T1', 'T2', 'GT', 'Prediction'])
            cv2.imwrite(join(out_dir, f'result_{fname}'), result_final)

            # 对比图:T1|T2|GT|Pred|Diff(五联)
            compare_row = concat_row([pre_vis, post_vis, tgt_vis, pred_mask, diff_mask], h)
            compare_final = add_title_legend(
                compare_row,
                f"Full Model Compare - {fname}",
                ['T1', 'T2', 'GT', 'Prediction', 'Diff(Green:TP,Red:FP,Blue:FN)'])
            cv2.imwrite(join(out_dir, f'compare_{fname}'), compare_final)

            # 单独保存掩码,供 analyze_diff_stats.py 纯图像解析
            cv2.imwrite(join(pred_dir, f'pred_{fname}'), pred_mask)
            cv2.imwrite(join(diff_dir, fname), diff_mask)

            path_mapping[fname] = {
                'result': join(out_dir, f'result_{fname}'),
                'compare': join(out_dir, f'compare_{fname}'),
                'diff_mask': join(diff_dir, fname),
                'pre_image': pre_path, 'post_image': post_path, 'label': label_path,
                'index': i + 1, 'dataset': dataset_name,
            }
            count += 1
            print(f"[{count}] 已保存 result_/compare_ + 掩码: {fname}")

    with open(join(out_dir, 'path_mapping.json'), 'w', encoding='utf-8') as f:
        json.dump(path_mapping, f, indent=2, ensure_ascii=False)
    print(f"\n[完成] 共处理 {count} 张图像,结果保存于: {out_dir}")


def get_parser():
    """
    构建参数解析器。

    出参:
    - parser: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description='完整模型可视化(结果图+对比图)')
    parser.add_argument('--file_root', type=str, required=True, help='数据集根目录路径。')
    parser.add_argument('--model_path', type=str, required=True, help='完整模型权重路径(.pth)。')
    parser.add_argument('--output_dir', type=str, default='./vis_full', help='可视化结果输出根目录。')
    parser.add_argument('--gpu_id', type=str, default='0', help='使用的 GPU ID。')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载进程数(Windows 建议 0)。')
    parser.add_argument('--in_height', type=int, default=256, help='输入图像高度。')
    parser.add_argument('--in_width', type=int, default=256, help='输入图像宽度。')
    parser.add_argument('--num_perception_frame', type=int, default=1, help='感知帧数量(当前架构必须为1)。')
    parser.add_argument('--pretrained', default=r'model\X3D_L.pyth', type=str, help='预训练 X3D 权重路径。')
    return parser


def main():
    """
    主函数:解析参数并执行可视化。
    """
    args = get_parser().parse_args()
    visualize_full_model(args)


if __name__ == '__main__':
    main()

# Copyright (c) Duowang Zhu.
# All rights reserved.

import os
import csv
import glob
import argparse
from os.path import join, basename

import numpy as np
import cv2

'''
差异图 TP/TN/FP/FN 统计分析脚本(纯图像解析,无需加载模型)

颜色映射标准(BGR,与 visualize_ablation.py / visualize_full_model.py / visualize_BCD.py 一致):
  绿色 [0,255,0]   = TP 正确预测变化
  红色 [0,0,255]   = FP 误检
  蓝色 [255,0,0]   = FN 漏检
  黑色 [0,0,0]     = TN 正确预测无变化

输入差异图来源(任选其一):
1. visualize_full_model.py 产出的 {output}/{dataset}/diffdata/
2. visualize_BCD.py 产出的 {output}/{dataset}/config_8/diffdata/
3. visualize_ablation.py 产出的 {output}/{dataset}/config_{id}/diff_masks/

使用示例(Windows PowerShell,单行格式):
python scripts/visualize/analyze_diff_stats.py --diff_dir "./vis_full/GVLM-CD/diffdata" --output "./vis_full/GVLM-CD/diff_stats.csv"
python scripts/visualize/analyze_diff_stats.py --diff_dir "./vis_results_ablation/GVLM-CD/config_3/diff_masks" --output "./stats/config_3.csv"
'''

# BGR 颜色常量(与可视化脚本严格一致)
BGR_TP = np.array([0, 255, 0], dtype=np.int32)
BGR_FP = np.array([0, 0, 255], dtype=np.int32)
BGR_FN = np.array([255, 0, 0], dtype=np.int32)
BGR_TN = np.array([0, 0, 0], dtype=np.int32)

# 纯黑色掩码判断阈值(三通道均低于此值视为 TN 背景,防压缩噪声干扰)
BLACK_THRESHOLD = 30


def classify_pixels(bgr_img, tolerance):
    """
    按 BGR 颜色将差异图像素分类为 TP/FP/FN/TN 四类并计数。

    入参:
    - bgr_img (np.ndarray): H×W×3 uint8 BGR 差异图
    - tolerance (int): 颜色匹配容差,用于抵御 PNG/JPG 压缩与抗锯齿噪声

    方法:
    1. 先判定纯黑背景为 TN(三通道均 < BLACK_THRESHOLD)
    2. 其余像素按 L1 距离匹配最近的 TP/FP/FN 颜色(容差内才算,容差外归为无效像素不计入)
    3. 无效像素(无法匹配任何类)仅打印告警,不参与指标计算

    出参:
    - counts (dict): {'TP':int,'FP':int,'FN':int,'TN':int,'invalid':int}
    """
    h, w, _ = bgr_img.shape
    pixels = bgr_img.reshape(-1, 3).astype(np.int32)

    # TN:纯黑背景优先识别,避免被颜色匹配误判
    is_black = np.all(pixels < BLACK_THRESHOLD, axis=1)
    counts = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0, 'invalid': 0}
    counts['TN'] = int(np.sum(is_black))

    # 非黑像素按 L1 距离匹配 TP/FP/FN
    non_black = ~is_black
    nb_pixels = pixels[non_black]
    if nb_pixels.shape[0] == 0:
        return counts

    d_tp = np.abs(nb_pixels - BGR_TP).sum(axis=1)
    d_fp = np.abs(nb_pixels - BGR_FP).sum(axis=1)
    d_fn = np.abs(nb_pixels - BGR_FN).sum(axis=1)

    # 三类距离堆叠后取 argmin,再过滤超出容差的像素为 invalid
    dist_stack = np.stack([d_tp, d_fp, d_fn], axis=1)
    min_dist = dist_stack.min(axis=1)
    min_idx = dist_stack.argmin(axis=1)

    valid = min_dist <= tolerance
    labels = ['TP', 'FP', 'FN']
    for ci, lab in enumerate(labels):
        counts[lab] = int(np.sum((min_idx == ci) & valid))
    counts['invalid'] = int(np.sum(~valid))
    return counts


def compute_metrics(tp, tn, fp, fn):
    """
    由混淆矩阵四元组计算变化检测常用指标。

    入参:
    - tp/tn/fp/fn (int): 四类像素计数

    方法:
    - Precision=TP/(TP+FP),Recall=TP/(TP+FN)
    - F1=2·P·R/(P+R),IoU=TP/(TP+FP+FN)
    - 分母为 0 时返回 0.0 避免除零

    出参:
    - metrics (dict): precision/recall/f1/iou/oa,均保留 4 位小数
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    total = tp + tn + fp + fn
    oa = (tp + tn) / total if total > 0 else 0.0
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'iou': round(iou, 4),
        'oa': round(oa, 4),
    }


def collect_diff_files(diff_dir):
    """
    递归收集目录下的差异图文件(支持常见图像扩展名)。

    入参:
    - diff_dir (str): 差异图所在目录

    方法:
    - glob 匹配 png/jpg/jpeg/tif/tiff,忽略 path_mapping.json 等非图像文件

    出参:
    - files (list[str]): 排序后的差异图绝对路径列表
    """
    exts = ['*.png', '*.PNG', '*.jpg', '*.jpeg', '*.tif', '*.tiff']
    files = []
    for ext in exts:
        files.extend(glob.glob(join(diff_dir, ext)))
        files.extend(glob.glob(join(diff_dir, '**', ext), recursive=True))
    # 去重并排序,保证多次运行结果稳定
    return sorted(set(files))


def analyze_diff_dir(diff_dir, output, tolerance):
    """
    分析整个差异图目录并输出 CSV 统计结果。

    入参:
    - diff_dir (str): 差异图目录路径
    - output (str): 输出 CSV 路径
    - tolerance (int): 颜色容差

    方法:
    1. 收集差异图文件
    2. 逐图分类像素并计算指标
    3. 汇总全目录总指标后写入 CSV

    出参:
    - 无返回值,结果写入 output 并在控制台打印汇总
    """
    if not os.path.isdir(diff_dir):
        print(f"[错误] 差异图目录不存在: {diff_dir}")
        return

    files = collect_diff_files(diff_dir)
    if not files:
        print(f"[错误] 目录中未找到差异图: {diff_dir}")
        return

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    print(f"[信息] 共发现 {len(files)} 张差异图,容差={tolerance}")

    rows = []
    sum_tp = sum_tn = sum_fp = sum_fn = 0
    for fpath in files:
        img = cv2.imread(fpath)
        if img is None:
            print(f"[警告] 无法读取,跳过: {fpath}")
            continue
        c = classify_pixels(img, tolerance)
        m = compute_metrics(c['TP'], c['TN'], c['FP'], c['FN'])
        sum_tp += c['TP']; sum_tn += c['TN']; sum_fp += c['FP']; sum_fn += c['FN']
        rows.append({
            'file': basename(fpath),
            'TP': c['TP'], 'TN': c['TN'], 'FP': c['FP'], 'FN': c['FN'],
            'invalid': c['invalid'],
            **m,
        })
        print(f"  {basename(fpath)}: TP={c['TP']} FP={c['FP']} FN={c['FN']} TN={c['TN']} F1={m['f1']}")

    # 全目录汇总指标
    overall = compute_metrics(sum_tp, sum_tn, sum_fp, sum_fn)
    rows.append({
        'file': 'OVERALL',
        'TP': sum_tp, 'TN': sum_tn, 'FP': sum_fp, 'FN': sum_fn,
        'invalid': '',
        **overall,
    })

    fieldnames = ['file', 'TP', 'TN', 'FP', 'FN', 'invalid', 'precision', 'recall', 'f1', 'iou', 'oa']
    with open(output, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[汇总] TP={sum_tp} FP={sum_fp} FN={sum_fn} TN={sum_tn}")
    print(f"  Precision={overall['precision']} Recall={overall['recall']} "
          f"F1={overall['f1']} IoU={overall['iou']} OA={overall['oa']}")
    print(f"[完成] 统计结果已写入: {output}")


def get_parser():
    """
    构建参数解析器。

    出参:
    - parser: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description='差异图 TP/TN/FP/FN 统计分析')
    parser.add_argument('--diff_dir', type=str, required=True, help='差异图所在目录。')
    parser.add_argument('--output', type=str, default='./diff_stats.csv', help='输出 CSV 路径。')
    parser.add_argument('--tolerance', type=int, default=10, help='颜色匹配容差(默认10,抗压缩噪声)。')
    return parser


def main():
    """
    主函数:解析参数并执行差异图统计分析。
    """
    args = get_parser().parse_args()
    analyze_diff_dir(args.diff_dir, args.output, args.tolerance)


if __name__ == '__main__':
    main()

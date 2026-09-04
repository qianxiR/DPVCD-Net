# Copyright (c) Duowang Zhu.
# All rights reserved.

"""

python scripts/visualize/visualize_ablation_tsne.py --file_root "E:\rqx\dataes\LBFD-CD" --save_dir ./exp_ablation_batch --output_dir ./vis_tsne --gpu_id 0 --model_path "E:\1代码\模型\变化检测\X3DFormer_\exp_new\LBFD-CD\best_model.pth"

消融实验 T-SNE 特征分布可视化

分别呈现6种配置下模型提取的特征在 T-SNE 空间中的分布（共享同一 T-SNE 坐标系）：
  Base: X3D Base               (纯 X3D，无任何增强模块)
  A:    X3D + Attention         (attention.py)
  B:    X3D + CascadeDCN        (cascade_dcn.py)
  C:    X3D + PosEnc + ST       (position_encoding.py + ST.py)
  BC:   X3D + B + C             (cascade_dcn.py + position_encoding.py + ST.py)
  Full: X3D + A + B + C         (完整 Trainer)


"""

import os
import sys
import argparse
import copy
from os.path import join
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

sys.path.insert(0, '.')

import data.dataset as RSDataset
import data.transforms as RSTransforms
from model.trainer_ablation import TrainerAblation, create_ablation_model
from model.trainer import Trainer


# ──────────────────────────────────────────────
# 配置定义
# ──────────────────────────────────────────────

# 消融配置: 标识 -> (experiment_id, 描述, 颜色)
# experiment_id=None 表示使用完整 Trainer
ABLATION_CONFIGS = {
    'Base': {
        'experiment_id': 1,
        'label': 'X3D (Base)',
        'color': '#8c564b',
        'marker': 'X',
    },
    'A': {
        'experiment_id': 2,
        'label': 'X3D + A',
        'color': '#1f77b4',
        'marker': 'o',
    },
    'B': {
        'experiment_id': 3,
        'label': 'X3D + B',
        'color': '#ff7f0e',
        'marker': 's',
    },
    'C': {
        'experiment_id': 4,
        'label': 'X3D + C',
        'color': '#2ca02c',
        'marker': '^',
    },
    'BC': {
        'experiment_id': 7,
        'label': 'X3D + B+C',
        'color': '#9467bd',
        'marker': 'P',
    },
    'Full': {
        'experiment_id': None,
        'label': 'X3D + A+B+C',
        'color': '#d62728',
        'marker': 'D',
    },
}


# ──────────────────────────────────────────────
# 模型加载（复用 visualize_ablation_by_image 的逻辑）
# ──────────────────────────────────────────────

def load_model(args, config_key):
    """
    入参:
    - args: 配置参数
    - config_key (str): 'A' / 'B' / 'C' / 'Full'

    方法:
    - A/B/C: 通过 create_ablation_model 创建，experiment_id 从配置表获取
    - Full: 直接创建 Trainer（完整模型）
    - 加载对应的 best_model.pth 权重

    出参:
    - model (nn.Module): 已加载权重、设为 eval 模式的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = ABLATION_CONFIGS[config_key]
    dataset_name = os.path.basename(os.path.normpath(args.file_root))

    if cfg['experiment_id'] is not None:
        exp_id = cfg['experiment_id']
        model = create_ablation_model(args, experiment_id=exp_id)
        exp_dir = join(args.save_dir, dataset_name,
                       f"config_{exp_id}_config_{exp_id}", dataset_name)
        model_path = join(exp_dir, 'best_model.pth')
    else:
        model = Trainer(args)
        if hasattr(args, 'model_path') and args.model_path:
            model_path = args.model_path
        else:
            model_path = join(args.save_dir, dataset_name, 'best_model.pth')
            if not os.path.exists(model_path):
                model_path = join(args.save_dir, dataset_name, 'final_model.pth')

    model.to(device)

    # Transformer 相关配置需要一次 dummy forward 初始化动态参数
    if cfg['experiment_id'] in (4, 7):
        model.eval()
        with torch.no_grad():
            dummy = torch.randn(1, 3, args.in_height, args.in_width, device=device)
            _ = model(dummy, dummy)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型权重: {model_path}")

    loaded_data = torch.load(model_path, map_location='cpu')
    if isinstance(loaded_data, dict):
        state_dict = loaded_data.get('state_dict', loaded_data)
    else:
        tmp = loaded_data.module if hasattr(loaded_data, 'module') else loaded_data
        state_dict = tmp.state_dict()

    # 去除 DataParallel 的 "module." 前缀
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k[7:] if k.startswith('module.') else k] = v
    state_dict = cleaned

    target = model.module if hasattr(model, 'module') else model
    target.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


# ──────────────────────────────────────────────
# 特征提取 Hook
# ──────────────────────────────────────────────

class DecoderInputHook:
    """
    注册在 decoder.forward 上的 hook，捕捉解码器输入的 5D 多尺度特征。

    入参:
    - 无（通过 register_forward_hook 使用）

    方法:
    - hook 回调将 stage_features_5d 保存到 self.features

    出参:
    - self.features: 解码器输入的 List[Tensor] (5D)
    """

    def __init__(self):
        self.features = None

    def __call__(self, module, input, output):
        # decoder.forward(self, f_5d) -> f_5d 是第一个位置参数
        self.features = input[0]


def extract_features_for_config(args, model, test_loader, max_samples=2000):
    """
    入参:
    - args: 配置参数
    - model: 已加载权重的模型（eval 模式）
    - test_loader: 测试集 DataLoader（batch_size=1）
    - max_samples (int): 最大采样像素数，避免 T-SNE 计算过慢

    方法:
    1. 注册 hook 捕捉 decoder 输入处的 5D 多尺度特征
    2. 对每个样本做前向推理，hook 自动记录特征
    3. 将多尺度 5D 特征转为像素级特征向量:
       - 对每个尺度做全局自适应平均池化（跨 T,H,W 维度），得到 1D 向量
       - 拼接 4 个尺度的向量作为该样本的表征
    4. 同时收集对应的 GT 标签（取中心像素的 majority vote 或全图标签）

    出参:
    - features_np (ndarray): [N, D] 特征矩阵
    - labels_np (ndarray): [N] 0/1 标签（0=未变化, 1=变化）
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 注册 hook
    hook = DecoderInputHook()
    handle = model.decoder.register_forward_hook(hook)

    all_features = []
    all_labels = []

    with torch.no_grad():
        for idx, batched_inputs in enumerate(test_loader):
            img = batched_inputs[0].to(device)
            target = batched_inputs[1].to(device)

            pre_img = img[:, :3, :, :]
            post_img = img[:, 3:, :, :]

            # 前向推理，hook 会自动记录 decoder 输入
            _ = model(pre_img, post_img)

            if hook.features is None:
                continue

            # hook.features: List[Tensor]，每个 [B, C, 3, H, W]
            # 对每个尺度做自适应平均池化到 (1,1) 后拼接
            sample_vecs = []
            for feat_5d in hook.features:
                # feat_5d: [1, C, 3, H, W] -> 池化跨 T,H,W -> [1, C]
                pooled = F.adaptive_avg_pool3d(feat_5d, (1, 1, 1)).flatten(1)
                sample_vecs.append(pooled)

            # 拼接所有尺度: [1, D]
            sample_feat = torch.cat(sample_vecs, dim=1)  # [1, 24+48+96+192=360]
            all_features.append(sample_feat.cpu())

            # GT 标签: 对整张图取 majority（>0.5 为变化）
            gt_np = target.squeeze().cpu().numpy().astype(np.uint8)
            label = int(gt_np.mean() > 0.5) if args.pixel_level is False else None

            if args.pixel_level:
                # 像素级: 不使用全图特征，跳过（使用下面的 pixel-level 逻辑）
                pass
            else:
                all_labels.append(label)

            if (idx + 1) % 50 == 0:
                print(f"  已处理 {idx + 1}/{len(test_loader)} 张图像")

    handle.remove()

    if not all_features:
        return np.array([]), np.array([])

    features_np = torch.cat(all_features, dim=0).numpy()
    labels_np = np.array(all_labels)
    return features_np, labels_np


def extract_pixel_features_for_config(args, model, test_loader, max_samples=5000):
    """
    像素级 T-SNE 特征提取：类别均衡采样，变化/未变化各采一半。

    入参:
    - args: 配置参数
    - model: 已加载权重的模型
    - test_loader: DataLoader
    - max_samples (int): 最大采样像素总数（变化+未变化合计）

    方法:
    1. Hook 捕捉 decoder 输入的 5D 特征
    2. 仅取最深层 stage 4，融合时序 P + |T1-T2|
    3. 对每张图分别从变化/未变化像素中各采样一半
    4. 汇总后截断到 max_samples，保证两类各占 50%

    出参:
    - features_np (ndarray): [N, D]
    - labels_np (ndarray): [N] 0/1
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    hook = DecoderInputHook()
    handle = model.decoder.register_forward_hook(hook)

    # 分别收集两类像素
    changed_feats = []
    unchanged_feats = []
    half_budget = max_samples // 2
    samples_per_class_per_image = max(half_budget // len(test_loader), 10)

    with torch.no_grad():
        for idx, batched_inputs in enumerate(test_loader):
            img = batched_inputs[0].to(device)
            target = batched_inputs[1].to(device)
            pre_img = img[:, :3, :, :]
            post_img = img[:, 3:, :, :]

            _ = model(pre_img, post_img)

            if hook.features is None:
                continue

            # 最深层特征融合时序
            feat_deep = hook.features[-1]
            t1 = feat_deep[:, :, 0, :, :]
            p = feat_deep[:, :, 1, :, :]
            t2 = feat_deep[:, :, 2, :, :]
            fused = p + torch.abs(t1 - t2)  # [1, 192, H_d, W_d]
            pixel_feat = fused.flatten(2).squeeze(0).permute(1, 0)  # [H*W, 192]

            # GT 对齐
            gt = target.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
            feat_h, feat_w = feat_deep.shape[3], feat_deep.shape[4]
            if gt.shape[0] != feat_h or gt.shape[1] != feat_w:
                gt = cv2.resize(gt, (feat_w, feat_h), interpolation=cv2.INTER_NEAREST)
            gt_flat = gt.flatten()

            # 按类别分离索引
            changed_idx = np.where(gt_flat == 1)[0]
            unchanged_idx = np.where(gt_flat == 0)[0]

            # 从每类中随机采样
            if len(changed_idx) > 0:
                n_sel = min(samples_per_class_per_image, len(changed_idx))
                sel = np.random.choice(changed_idx, n_sel, replace=False)
                changed_feats.append(pixel_feat[sel].cpu())

            if len(unchanged_idx) > 0:
                n_sel = min(samples_per_class_per_image, len(unchanged_idx))
                sel = np.random.choice(unchanged_idx, n_sel, replace=False)
                unchanged_feats.append(pixel_feat[sel].cpu())

            if (idx + 1) % 100 == 0:
                print(f"  已处理 {idx + 1}/{len(test_loader)} 张")

    handle.remove()

    if not changed_feats and not unchanged_feats:
        return np.array([]), np.array([])

    # 拼接并均衡截断
    if changed_feats:
        changed_all = torch.cat(changed_feats, dim=0)
    else:
        changed_all = torch.empty(0, 192)

    if unchanged_feats:
        unchanged_all = torch.cat(unchanged_feats, dim=0)
    else:
        unchanged_all = torch.empty(0, 192)

    # 两类各取 half_budget，不足则全部取
    n_ch = min(half_budget, changed_all.shape[0])
    n_unch = min(half_budget, unchanged_all.shape[0])

    ch_idx = torch.randperm(changed_all.shape[0])[:n_ch]
    unch_idx = torch.randperm(unchanged_all.shape[0])[:n_unch]

    features = torch.cat([changed_all[ch_idx], unchanged_all[unch_idx]], dim=0)
    labels = torch.cat([
        torch.ones(n_ch, dtype=torch.long),
        torch.zeros(n_unch, dtype=torch.long),
    ], dim=0)

    features_np = features.numpy()
    labels_np = labels.numpy()

    print(f"  类别均衡采样完成: 变化={n_ch}, 未变化={n_unch}, 总计={n_ch + n_unch}")
    return features_np, labels_np


# ──────────────────────────────────────────────
# 定量指标计算
# ──────────────────────────────────────────────

def _compute_metrics(features_np, labels_np, embedding):
    """
    计算定量指标：高维 Silhouette、高维/2D 类间类内距离比。

    入参:
    - features_np (ndarray): [N, D] 原始高维特征
    - labels_np (ndarray): [N] 0/1 标签
    - embedding (ndarray): [N, 2] T-SNE 降维后坐标

    方法:
    - Silhouette Score: 衡量聚类分离度，范围 [-1, 1]，越高分离越好
    - Inter/Intra Distance Ratio: 类间中心距离 / 类内平均半径，越大越好
    - 分别在高维空间和 2D 空间计算距离比，前者更严谨，后者与图中视觉对应

    出参:
    - metrics (dict): sil, inter_intra_hd, inter_intra_2d, centroid_dist_2d,
                      centroid_ch_2d, centroid_unch_2d
    """
    mask_ch = labels_np == 1
    mask_unch = labels_np == 0

    # 高维 Silhouette（采样加速，避免大数据集 OOM）
    if features_np.shape[0] > 10000:
        idx = np.random.choice(features_np.shape[0], 10000, replace=False)
        sil = silhouette_score(features_np[idx], labels_np[idx])
    else:
        sil = silhouette_score(features_np, labels_np)

    # 高维空间类间/类内距离比
    centroid_ch = features_np[mask_ch].mean(axis=0)
    centroid_unch = features_np[mask_unch].mean(axis=0)
    inter_dist_hd = np.linalg.norm(centroid_ch - centroid_unch)
    intra_ch_hd = np.linalg.norm(features_np[mask_ch] - centroid_ch, axis=1).mean()
    intra_unch_hd = np.linalg.norm(features_np[mask_unch] - centroid_unch, axis=1).mean()
    intra_avg_hd = (intra_ch_hd + intra_unch_hd) / 2
    inter_intra_hd = inter_dist_hd / max(intra_avg_hd, 1e-8)

    # 2D 空间类间/类内距离比
    centroid_ch_2d = embedding[mask_ch].mean(axis=0)
    centroid_unch_2d = embedding[mask_unch].mean(axis=0)
    inter_dist_2d = np.linalg.norm(centroid_ch_2d - centroid_unch_2d)
    intra_ch_2d = np.linalg.norm(embedding[mask_ch] - centroid_ch_2d, axis=1).mean()
    intra_unch_2d = np.linalg.norm(embedding[mask_unch] - centroid_unch_2d, axis=1).mean()
    intra_avg_2d = (intra_ch_2d + intra_unch_2d) / 2
    inter_intra_2d = inter_dist_2d / max(intra_avg_2d, 1e-8)

    return {
        'sil': round(sil, 4),
        'inter_intra_hd': round(inter_intra_hd, 4),
        'inter_intra_2d': round(inter_intra_2d, 4),
        'centroid_dist_2d': round(inter_dist_2d, 4),
        'centroid_ch_2d': centroid_ch_2d,
        'centroid_unch_2d': centroid_unch_2d,
    }


def _setup_tnr_font():
    """
    配置 matplotlib 全局使用 Times New Roman 字体。

    入参: 无
    方法: 设置 rcParams 字体族，返回 FontProperties 用于图例
    出参: font_legend FontProperties 对象
    """
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    font_legend = FontProperties(family='Times New Roman', size=9)
    return font_legend


def _draw_tsne_subplot(ax, embedding, labels_np, metrics, font_legend):
    """
    在单个 axes 上绘制 T-SNE 散点 + 类中心标记。

    入参:
    - ax: matplotlib Axes
    - embedding (ndarray): [N, 2] T-SNE 坐标
    - labels_np (ndarray): [N] 0/1 标签
    - metrics (dict): _compute_metrics 的返回值
    - font_legend: FontProperties

    方法:
    1. 绘制散点（蓝=未变化，红=变化）
    2. 用星号标记两类中心并用虚线连线

    出参: 无（直接修改 ax）
    """
    mask_unch = labels_np == 0
    mask_ch = labels_np == 1

    # 散点（颜色调深，保持色调）
    ax.scatter(
        embedding[mask_unch, 0], embedding[mask_unch, 1],
        c='#5a9bd4', s=4, alpha=0.5, label='Unchanged', rasterized=True,
    )
    ax.scatter(
        embedding[mask_ch, 0], embedding[mask_ch, 1],
        c='#b71c1c', s=4, alpha=0.5, label='Changed', rasterized=True,
    )

    # 类中心星号标记与连线（颜色调深）
    c_unch = metrics['centroid_unch_2d']
    c_ch = metrics['centroid_ch_2d']
    ax.plot(*c_unch, marker='*', markersize=20, color='#0d47a1',
            markeredgecolor='black', markeredgewidth=1, linestyle='None', zorder=5)
    ax.plot(*c_ch, marker='*', markersize=20, color='#b71c1c',
            markeredgecolor='black', markeredgewidth=1, linestyle='None', zorder=5)
    ax.plot([c_unch[0], c_ch[0]], [c_unch[1], c_ch[1]],
            'k--', linewidth=0.8, alpha=0.5, zorder=4)

    # 标题已移除

    # 图例：字体加大、加粗
    legend = ax.legend(loc='upper right', prop=font_legend, markerscale=3)
    for text in legend.get_texts():
        text.set_fontsize(12)
        text.set_fontweight('bold')

    # xy轴标注字体加大
    ax.tick_params(labelsize=14)


# ──────────────────────────────────────────────
# T-SNE 降维 + 可视化
# ──────────────────────────────────────────────

def run_tsne_and_plot(all_config_data, output_dir, perplexity=30, n_iter=1000):
    """
    入参:
    - all_config_data: dict {config_key: (features_np, labels_np)}
    - output_dir (str): 输出目录
    - perplexity (int): T-SNE perplexity 参数
    - n_iter (int): T-SNE 迭代次数

    方法:
    1. 对每个配置的特征独立做 T-SNE 降维到 2D
    2. 在高维和 2D 空间分别计算定量指标（Silhouette、类间/类内距离比）
    3. 在 2x3 子图中绘制散点 + 类中心标记 + 指标标注（Times New Roman 字体）
    4. 输出 tsne_metrics.csv 汇总所有配置的定量指标
    5. 保存为 PDF 和 PNG

    出参:
    - 保存图片和 CSV 到 output_dir
    """
    os.makedirs(output_dir, exist_ok=True)
    font_legend = _setup_tnr_font()

    config_keys = [k for k in ABLATION_CONFIGS
                   if k in all_config_data and all_config_data[k][0].size > 0]
    if not config_keys:
        print("[警告] 没有有效的特征数据，跳过可视化。")
        return

    # T-SNE 降维 + 定量指标计算
    embeddings_cache = {}
    metrics_cache = {}
    for key in config_keys:
        cfg = ABLATION_CONFIGS[key]
        features_np, labels_np = all_config_data[key]

        print(f"对配置 {key} ({cfg['label']}) 运行 T-SNE ({features_np.shape[0]} 样本)...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(perplexity, max(5, features_np.shape[0] // 4)),
            max_iter=n_iter,
            random_state=42,
            init='pca',
            learning_rate='auto',
        )
        embedding = tsne.fit_transform(features_np)
        embeddings_cache[key] = (embedding, labels_np)

        # 计算定量指标
        m = _compute_metrics(features_np, labels_np, embedding)
        metrics_cache[key] = m
        print(f"  Silhouette={m['sil']:.4f}, I/I(hd)={m['inter_intra_hd']:.4f}, "
              f"I/I(2d)={m['inter_intra_2d']:.4f}")

    # ── 导出 CSV ──
    csv_path = join(output_dir, 'tsne_metrics.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        f.write('Config,Label,Silhouette,InterIntra_HD,InterIntra_2D,CentroidDist_2D\n')
        for key in config_keys:
            cfg = ABLATION_CONFIGS[key]
            m = metrics_cache[key]
            f.write(f"{key},{cfg['label']},{m['sil']:.4f},{m['inter_intra_hd']:.4f},"
                    f"{m['inter_intra_2d']:.4f},{m['centroid_dist_2d']:.4f}\n")
    print(f"定量指标已保存: {csv_path}")

    # ── 绘制拼接总图 ──
    n = len(config_keys)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, key in enumerate(config_keys):
        cfg = ABLATION_CONFIGS[key]
        embedding, labels_np = embeddings_cache[key]
        _draw_tsne_subplot(
            axes[i], embedding, labels_np, metrics_cache[key],
            font_legend,
        )

    # 隐藏多余子图
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()

    # 保存总图
    save_prefix = join(output_dir, 'tsne_ablation')
    fig.savefig(save_prefix + '.png', dpi=300, bbox_inches='tight')
    fig.savefig(save_prefix + '.pdf', bbox_inches='tight', format='pdf')
    plt.close(fig)
    print(f"T-SNE 总图已保存: {save_prefix}.png / .pdf")

    # ── 逐配置单独保存 ──
    for key in config_keys:
        cfg = ABLATION_CONFIGS[key]
        embedding, labels_np = embeddings_cache[key]

        fig_s, ax_s = plt.subplots(1, 1, figsize=(7, 6))
        _draw_tsne_subplot(
            ax_s, embedding, labels_np, metrics_cache[key],
            font_legend,
        )
        fig_s.tight_layout()

        single_prefix = join(output_dir, f'tsne_{key}')
        fig_s.savefig(single_prefix + '.png', dpi=300, bbox_inches='tight')
        fig_s.savefig(single_prefix + '.pdf', dpi=300, bbox_inches='tight', format='pdf')
        plt.close(fig_s)
        print(f"  单独保存: {single_prefix}.png / .pdf")

    print(f"T-SNE 可视化已保存至: {output_dir}")


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='消融实验 T-SNE 特征分布可视化')

    # 数据集和路径
    parser.add_argument('--file_root', type=str, required=True,
                        help='数据集根目录路径')
    parser.add_argument('--save_dir', type=str, default='./exp_ablation_batch',
                        help='消融实验保存目录')
    parser.add_argument('--output_dir', type=str, default='./vis_tsne',
                        help='T-SNE 可视化输出目录')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU ID')
    parser.add_argument('--in_height', type=int, default=256)
    parser.add_argument('--in_width', type=int, default=256)

    # 模型参数
    parser.add_argument('--num_perception_frame', type=int, default=1)
    parser.add_argument('--pretrained', default=r'model\X3D_L.pyth', type=str)
    parser.add_argument('--model_path', type=str, default=None,
                        help='完整模型(配置Full)权重路径（可选）')

    # T-SNE 参数
    parser.add_argument('--configs', type=str, default='Base,A,B,C,BC,Full',
                        help='要可视化的配置，逗号分隔。可选: Base,A,B,C,BC,Full')
    parser.add_argument('--max_samples', type=int, default=5000,
                        help='每个配置最大采样像素数')
    parser.add_argument('--perplexity', type=int, default=30,
                        help='T-SNE perplexity')
    parser.add_argument('--n_iter', type=int, default=1000,
                        help='T-SNE 迭代次数')
    parser.add_argument('--pixel_level', action='store_true', default=True,
                        help='像素级特征采样（默认开启）')
    parser.add_argument('--no_pixel_level', dest='pixel_level', action='store_false',
                        help='禁用像素级采样，改用图像级特征')

    args = parser.parse_args()

    # GPU 设置
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
        torch.manual_seed(16)
        torch.cuda.manual_seed(16)
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    else:
        torch.manual_seed(16)
        print("CUDA 不可用，使用 CPU")

    # 解析配置列表
    config_keys = [k.strip() for k in args.configs.split(',')]
    for k in config_keys:
        if k not in ABLATION_CONFIGS:
            raise ValueError(f"无效配置 '{k}'，可选: {list(ABLATION_CONFIGS.keys())}")

    print(f"\n=== 消融实验 T-SNE 可视化 ===")
    print(f"数据集: {args.file_root}")
    print(f"配置: {config_keys}")
    print(f"采样模式: {'像素级' if args.pixel_level else '图像级'}")
    print(f"最大采样数: {args.max_samples}")

    # 数据加载器
    _, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)
    test_data = RSDataset.BCDDataset(
        file_root=args.file_root, split="test", transform=val_transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True,
    )
    print(f"测试集: {len(test_loader)} 张图像")

    # 逐配置提取特征 + T-SNE
    os.makedirs(args.output_dir, exist_ok=True)
    all_config_data = {}

    for key in config_keys:
        cfg = ABLATION_CONFIGS[key]
        print(f"\n{'='*50}")
        print(f"处理配置 {key}: {cfg['label']}")
        print(f"{'='*50}")

        try:
            # 每次 deep copy args 防止配置污染
            args_copy = copy.deepcopy(args)
            model = load_model(args_copy, key)
        except FileNotFoundError as e:
            print(f"[跳过] {e}")
            continue

        if args.pixel_level:
            feats, labels = extract_pixel_features_for_config(
                args_copy, model, test_loader, max_samples=args.max_samples,
            )
        else:
            feats, labels = extract_features_for_config(
                args_copy, model, test_loader, max_samples=args.max_samples,
            )

        if feats.size == 0:
            print(f"[跳过] 配置 {key} 无有效特征")
            del model
            torch.cuda.empty_cache()
            continue

        print(f"  提取完成: {feats.shape[0]} 样本, {feats.shape[1]} 维")
        all_config_data[key] = (feats, labels)

        del model
        torch.cuda.empty_cache()

    # 绘制 T-SNE
    if all_config_data:
        run_tsne_and_plot(
            all_config_data, args.output_dir,
            perplexity=args.perplexity, n_iter=args.n_iter,
        )
    else:
        print("[错误] 所有配置均未能提取特征，请检查模型权重路径。")

    print(f"\n=== T-SNE 可视化完成 ===")
    print(f"输出目录: {args.output_dir}")


if __name__ == '__main__':
    main()

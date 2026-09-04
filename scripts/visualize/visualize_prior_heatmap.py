# Copyright (c) Duowang Zhu.
# All rights reserved.

"""
差异先验引导消融热力图可视化脚本（rscd 技能 §6.1/§6.4 口径）

对比两组模型在同一测试样本上的内部表征热力图：
  1. Base  : 消融配置1（纯 X3D 伪视频基线，无差异先验引导模块）
  2. Full  : 完整 DPVCD-Net（差异先验引导完整模型，Attention+ASPP+ST融合全开）

捕获信号（均通过 forward hook 抓取，不侵入模型 forward）：
  - 感知帧P特征层: encoder 输出最深层 5D 特征 [B,192,3,16,16] 的 P 帧(时间维索引1)
    → [B,192,16,16]，两模型同层同语义，直接可比
  - logits 层: decoder.up_c1 输出（sigmoid 之前）[B,1,256,256]

固定规范（rscd 技能 §6.1，不要改）：
  - colormap 统一 jet（与技能 vis_utils.get_jet_cmap 同源）
  - 同一样本的多路信号（P帧特征/logits，含 base/full 两模型）共用同一 vmin/vmax，
    单一颜色条（同一颜色条可横向对比）
  - 特征图取通道均值后插值上采样到原尺寸
  - 组合布局：T1 | T2 | GT | P帧特征热力图 | logits 热力图 | Pred | 颜色条，上下加标题/图例条
  - 颜色条使用技能 vis_utils.create_colorbar_image 实现（顶=高值，底=低值，右侧标注数值）

使用示例：
python scripts/visualize/visualize_prior_heatmap.py --file_root "E:\rqx\dataes\LBFD-CD" --output_dir ./vis_results_heatmap
python scripts/visualize/visualize_prior_heatmap.py --file_root "E:\rqx\dataes\LBFD-CD" --num_samples 20

=== 输出结构（{output_dir}/{数据集名}/heatmap/） ===
├── base/                    # 配置1（纯X3D基线）
│   ├── combined_{name}.png  # T1|T2|GT|P帧特征|logits|Pred|颜色条 组合图
│   ├── diff_feat_{name}.png # 感知帧P特征热力图（单图）
│   └── logits_{name}.png    # logits 热力图（单图）
└── full/                    # 完整模型（差异先验引导）
    ├── combined_{name}.png
    ├── diff_feat_{name}.png
    └── logits_{name}.png
"""

import os
import sys
import argparse
from os.path import join

import cv2
import numpy as np
import torch
from skimage import io
from PIL import Image, ImageDraw, ImageFont
import matplotlib

# 插入项目根路径以导入本地模块(做什么:使 data/model 同包脚本可被 import)
sys.path.insert(0, '.')

import data.dataset as RSDataset
import data.transforms as RSTransforms
from model.trainer import Trainer
from model.trainer_ablation import create_ablation_model
from scripts.visualize.heatmap_utils import tensor_to_heatmap

# jet 配色:与技能 vis_utils.get_jet_cmap 同源(优先新版注册表,回退旧接口)
try:
    JET_CMAP = matplotlib.colormaps['jet']
except AttributeError:
    from matplotlib import cm
    JET_CMAP = cm.get_cmap('jet')

# 分隔线宽度(像素):组合图各子图之间的黑边
SPACER_WIDTH = 10

# 标题条/图例条高度(像素)
TITLE_HEIGHT = 40
LEGEND_HEIGHT = 60


def create_colorbar_image(colormap, vmin, vmax, width=40, height=256):
    """
    竖直渐变颜色条图例(rscd 技能 vis_utils.create_colorbar_image 同实现,顶=高值)。

    入参:
    - colormap: matplotlib Colormap
    - vmin/vmax (float): 数值范围标注
    - width/height (int): 颜色条像素尺寸

    方法:
    1. 竖向渐变(顶=1.0,底=0.0)应用 colormap
    2. PIL 在右侧叠加 vmax/vmin 数值文本

    出参:
    - bar_img (np.ndarray): height×(width+文本区)×3 uint8 RGB
    """
    gradient = np.linspace(1, 0, height).reshape(height, 1)
    gradient = np.repeat(gradient, width, axis=1)
    bar_rgb = (colormap(gradient)[:, :, :3] * 255).astype(np.uint8)
    bar_pil = Image.fromarray(bar_rgb)
    draw = ImageDraw.Draw(bar_pil)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw.text((width + 5, 5), f"{vmax:.2f}", fill=(255, 255, 255), font=font)
    draw.text((width + 5, height - 20), f"{vmin:.2f}", fill=(255, 255, 255), font=font)
    return np.array(bar_pil)


class PFrameCapture:
    """
    感知帧P特征捕获器:forward hook 抓取 encoder 输出最深层 5D 特征的 P 帧。

    入参:
    - 无

    方法:
    - attach(model):在 model.encoder 上注册 forward hook
    - detach():移除 hook
    - p_feat 属性:最近一次前向的最深层 P 帧特征 [B,C,H,W](C=192)

    出参:
    - 通过 .p_feat 访问捕获结果(detach 后的张量)
    """

    def __init__(self):
        self.p_feat = None
        self._handle = None

    def _hook_fn(self, module, inputs, output):
        # forward hook 回调:encoder 返回 4 个尺度的 5D 特征列表,
        # 取最深层 [-1] 后切时间维索引1即感知帧P([T1,P,T2]中的P)
        self.p_feat = output[-1][:, :, 1, :, :].detach().clone()

    def attach(self, model):
        """
        入参:
        - model: Trainer 或 TrainerAblation 实例(编码器均为 trainer.Encoder)

        方法:
        - 在 model.encoder 上注册 forward hook

        出参:
        - self(链式调用)
        """
        self._handle = model.encoder.register_forward_hook(self._hook_fn)
        return self

    def detach(self):
        """
        入参: 无
        方法: 移除已注册的 hook
        出参: 无
        """
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


class LogitsHookCapture:
    """
    logits 层捕获器:forward hook 抓取 decoder.up_c1 输出(sigmoid 之前)。

    入参:
    - 无

    方法:
    - attach(model):在 model.decoder.up_c1 上注册 forward hook
    - detach():移除 hook
    - logits 属性:最近一次前向捕获的 logits 张量 [B,1,H,W]

    出参:
    - 通过 .logits 访问捕获结果(detach 后的张量)
    """

    def __init__(self):
        self.logits = None
        self._handle = None

    def _hook_fn(self, module, inputs, output):
        # forward hook 回调:output 即 up_c1 卷积输出,未过 sigmoid
        self.logits = output.detach().clone()

    def attach(self, model):
        """
        入参:
        - model: Trainer 或 TrainerAblation 实例(解码器均为 change_decoder.ChangeDecoder)

        方法:
        - 在 model.decoder.up_c1 上注册 forward hook

        出参:
        - self(链式调用)
        """
        self._handle = model.decoder.up_c1.register_forward_hook(self._hook_fn)
        return self

    def detach(self):
        """
        入参: 无
        方法: 移除已注册的 hook
        出参: 无
        """
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def load_state_dict_strict(model, model_path):
    """
    以 strict 模式加载权重到模型(含惰性模块处理)。

    入参:
    - model (nn.Module):已构建的模型(Trainer/TrainerAblation)
    - model_path (str):权重文件路径,要求为纯 state_dict(*.pth)

    方法:
    1. torch.load 读取(兼容含 state_dict 键与 module. 前缀的旧格式)
    2. 剥离 'module.' 前缀后 strict=True 加载,结构不匹配立即报错而非静默降级

    出参:
    - 无(原地更新 model 参数)
    """
    loaded = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(loaded, dict) and 'state_dict' in loaded:
        loaded = loaded['state_dict']
    state_dict = {
        (k[7:] if k.startswith('module.') else k): v for k, v in loaded.items()
    }
    model.load_state_dict(state_dict, strict=True)


def build_models(args, dataset_name):
    """
    构建并加载 Base(配置1)与 Full(完整模型)两组模型。

    入参:
    - args: 命令行参数(含 base_model_path/full_model_path)
    - dataset_name (str): 数据集目录名(用于默认权重路径解析)

    方法:
    1. Full: Trainer 完整模型 → dummy 前向初始化惰性模块 → strict 加载 exp_new 权重
    2. Base: create_ablation_model(配置1) → dummy 前向 → strict 加载消融权重
    3. 均 to(device).eval()

    出参:
    - (base_model, full_model): 两个已加载权重并处于评估模式的模型
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    full_model = Trainer(args=args)
    with torch.no_grad():
        # dummy 前向:做什么——触发 pos_encoder.edge_projection 等惰性模块注册
        # 为什么——这些模块首次前向才创建,不先触发会导致 strict 加载时键集合缺失
        _ = full_model(torch.randn(1, 3, args.in_height, args.in_width),
                       torch.randn(1, 3, args.in_height, args.in_width))
    load_state_dict_strict(full_model, args.full_model_path)
    full_model.to(device).eval()

    base_model = create_ablation_model(args=args, experiment_id=1)
    with torch.no_grad():
        _ = base_model(torch.randn(1, 3, args.in_height, args.in_width),
                       torch.randn(1, 3, args.in_height, args.in_width))
    load_state_dict_strict(base_model, args.base_model_path)
    base_model.to(device).eval()

    return base_model, full_model


def make_combined_image(t1_bgr, t2_bgr, gt_vis, feat_heat_bgr, logits_heat_bgr,
                        pred_vis, colorbar_bgr, title_text, legend_texts):
    """
    拼接单模型组合图:标题条 + T1|T2|GT|P帧特征热力图|logits热力图|Pred|颜色条 + 图例条。

    入参:
    - t1_bgr/t2_bgr/gt_vis/pred_vis (np.ndarray): H×W×3 uint8 BGR 子图
    - feat_heat_bgr/logits_heat_bgr (np.ndarray): H×W×3 uint8 BGR 热力图(已上采样到原尺寸)
    - colorbar_bgr (np.ndarray): 颜色条图例(与两幅热力图共用同一 vmin/vmax)
    - title_text (str): 顶部标题文本
    - legend_texts (list[str]): 与各子图一一对应的图例文本(7 项)

    方法:
    1. 各子图间插入黑色分隔线后水平拼接
    2. 顶部叠加标题条、底部叠加图例条(按面板实际宽度对齐, cv2.putText)

    出参:
    - final_vis (np.ndarray): 组合图 uint8 BGR
    """
    h = t1_bgr.shape[0]
    spacer = np.zeros((h, SPACER_WIDTH, 3), dtype=np.uint8)
    parts = [t1_bgr, t2_bgr, gt_vis, feat_heat_bgr, logits_heat_bgr, pred_vis, colorbar_bgr]
    combined = parts[0]
    for part in parts[1:]:
        combined = np.concatenate([combined, spacer, part], axis=1)

    title_img = np.zeros((TITLE_HEIGHT, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(title_img, title_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    legend_img = np.zeros((LEGEND_HEIGHT, combined.shape[1], 3), dtype=np.uint8)
    # 按各面板实际宽度累计偏移,保证图例文本与子图对齐(颜色条面板远窄于影像面板)
    x_offset = 0
    for part, text in zip(parts, legend_texts):
        cv2.putText(legend_img, text, (x_offset, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        x_offset += part.shape[1] + SPACER_WIDTH

    return np.concatenate([title_img, combined, legend_img], axis=0)


@torch.no_grad()
def visualize_prior_heatmaps(args):
    """
    主流程:逐样本推理两个模型,生成 base/full 两组热力图并落盘。

    入参:
    - args: 命令行参数(含 file_root/base_model_path/full_model_path/output_dir)

    方法:
    1. 构建两个模型并挂载感知帧P(encoder hook)与 logits(decoder.up_c1 hook)捕获器
    2. 对每张测试图先后前向两个模型,取P帧特征通道均值与 logits
    3. 同一样本全部信号(含跨模型)共用同一 vmin/vmax 与单一颜色条(技能固定规范)
    4. 特征图上采样到原尺寸,生成 jet 热力图,按 §6.4 结构落盘

    出参:
    - total_count (int): 处理的样本数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    print(f"使用设备: {device}")
    print(f"数据集: {dataset_name} (split=test)")

    base_model, full_model = build_models(args, dataset_name)
    base_feat_cap = PFrameCapture().attach(base_model)
    base_logit_cap = LogitsHookCapture().attach(base_model)
    full_feat_cap = PFrameCapture().attach(full_model)
    full_logit_cap = LogitsHookCapture().attach(full_model)

    _, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)
    test_data = RSDataset.BCDDataset(
        file_root=args.file_root, split="test", transform=val_transform)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    print(f"测试集包含 {len(test_loader)} 个样本。")

    # §6.4 输出结构:{output_dir}/{数据集名}/heatmap/{base,full}/
    heat_root = join(args.output_dir, dataset_name, 'heatmap')
    model_dirs = {'base': join(heat_root, 'base'), 'full': join(heat_root, 'full')}
    for d in model_dirs.values():
        os.makedirs(d, exist_ok=True)

    legend_texts = ["T1", "T2", "GT", "P-Feat", "Logits", "Pred", "Colorbar"]
    titles = {
        'base': "Base (X3D only)",
        'full': "Full (DPVCD-Net, Difference-Prior-Guided)",
    }

    total_count = 0
    for i, batched_inputs in enumerate(test_loader):
        if args.num_samples is not None and total_count >= args.num_samples:
            break
        img, target = batched_inputs[0].to(device), batched_inputs[1].to(device)
        pre_img, post_img = img[:, :3, :, :], img[:, 3:, :, :]

        # 两模型先后前向,hook 自动记录感知帧P特征与 logits
        base_prob = base_model(pre_img, post_img)
        full_prob = full_model(pre_img, post_img)

        # 信号提取:P帧特征取通道均值得 [H,W];logits 取首个样本首通道
        signals = {
            'base': (base_feat_cap.p_feat[0].float().mean(dim=0).cpu().numpy(),
                     base_logit_cap.logits[0, 0].float().cpu().numpy()),
            'full': (full_feat_cap.p_feat[0].float().mean(dim=0).cpu().numpy(),
                     full_logit_cap.logits[0, 0].float().cpu().numpy()),
        }

        # 统一归一化范围:同一样本全部信号(两模型×两路)共用,单一颜色条横向可比(技能规范)
        vmin = float(min(s.min() for pair in signals.values() for s in pair))
        vmax = float(max(s.max() for pair in signals.values() for s in pair))

        # 原始尺寸基准:GT/预测为 256×256,热力图上采样对齐
        gt_np = target[0, 0].cpu().numpy().astype(np.uint8)
        h, w = gt_np.shape
        base_pred = (base_prob[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
        full_pred = (full_prob[0, 0].cpu().numpy() > 0.5).astype(np.uint8)

        # 读取原始 T1/T2 影像(做什么:绕过归一化变换,可视化用原始影像)
        pre_vis = cv2.resize(io.imread(test_loader.dataset.pre_images[i]), (w, h))
        post_vis = cv2.resize(io.imread(test_loader.dataset.post_images[i]), (w, h))
        pre_bgr = cv2.cvtColor(pre_vis, cv2.COLOR_RGB2BGR)
        post_bgr = cv2.cvtColor(post_vis, cv2.COLOR_RGB2BGR)
        gt_vis = np.stack([gt_np * 255] * 3, axis=-1)
        base_pred_vis = np.stack([base_pred * 255] * 3, axis=-1)
        full_pred_vis = np.stack([full_pred * 255] * 3, axis=-1)
        colorbar_bgr = cv2.cvtColor(
            create_colorbar_image(JET_CMAP, vmin, vmax, width=40, height=h),
            cv2.COLOR_RGB2BGR)

        filename = os.path.basename(test_loader.dataset.pre_images[i])
        for tag, pred_vis in (('base', base_pred_vis), ('full', full_pred_vis)):
            feat_2d, logits_2d = signals[tag]
            # 低分辨率特征插值上采样到原尺寸(技能规范:先 jet 映射再放大保持色块锐利)
            feat_heat = cv2.resize(
                tensor_to_heatmap(feat_2d, JET_CMAP, vmin=vmin, vmax=vmax), (w, h))
            logits_heat = cv2.resize(
                tensor_to_heatmap(logits_2d, JET_CMAP, vmin=vmin, vmax=vmax), (w, h))
            feat_heat_bgr = cv2.cvtColor(feat_heat, cv2.COLOR_RGB2BGR)
            logits_heat_bgr = cv2.cvtColor(logits_heat, cv2.COLOR_RGB2BGR)

            combined = make_combined_image(
                pre_bgr, post_bgr, gt_vis, feat_heat_bgr, logits_heat_bgr,
                pred_vis, colorbar_bgr,
                f"{titles[tag]} - {filename} - vmin={vmin:.3f}, vmax={vmax:.3f}",
                legend_texts)
            cv2.imwrite(join(model_dirs[tag], f"combined_{filename}"), combined)
            cv2.imwrite(join(model_dirs[tag], f"diff_feat_{filename}"), feat_heat_bgr)
            cv2.imwrite(join(model_dirs[tag], f"logits_{filename}"), logits_heat_bgr)

        total_count += 1
        if total_count % 50 == 0:
            print(f"已处理 {total_count} 张")

    for cap in (base_feat_cap, base_logit_cap, full_feat_cap, full_logit_cap):
        cap.detach()

    print(f"\n[完成] 共处理 {total_count} 张,输出目录: {heat_root}")
    return total_count


def get_parser():
    """
    创建脚本的参数解析器。

    入参: 无
    方法: 定义数据集、权重、输出与模型参数(与仓库其他脚本口径一致)
    出参: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description='差异先验引导消融热力图可视化(base vs full)')
    parser.add_argument('--file_root', type=str, default=r'E:\rqx\dataes\LBFD-CD',
                        help='数据集根目录路径。')
    parser.add_argument('--base_model_path', type=str, default=None,
                        help='Base(配置1)权重路径,默认按数据集名解析 exp_ablation_adaptive/config_1_adaptive/{数据集}/best_model.pth。')
    parser.add_argument('--full_model_path', type=str, default=None,
                        help='Full(完整模型)权重路径,默认按数据集名解析 exp_new/{数据集}/best_model.pth。')
    parser.add_argument('--output_dir', type=str, default='./vis_results_heatmap',
                        help='可视化输出根目录。')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='只处理前 N 个样本(默认全部)。')
    parser.add_argument('--gpu_id', type=str, default='0', help='使用的GPU ID。')
    parser.add_argument('--in_height', type=int, default=256, help='RGB图像高度')
    parser.add_argument('--in_width', type=int, default=256, help='RGB图像宽度')
    parser.add_argument('--num_perception_frame', type=int, default=1,
                        help='感知帧数量(当前架构必须为1)')
    parser.add_argument('--pretrained', default=r'model\X3D_L.pyth', type=str,
                        help='预训练X3D权重路径(仅用于构建编码器,随后被实验权重覆盖)。')
    return parser


def main():
    """
    主函数:解析参数,补全默认权重路径后执行可视化。

    入参: 无(读 sys.argv)
    方法: 按数据集名解析 base/full 默认权重路径,缺失即报错退出
    出参: 无(副作用:热力图落盘)
    """
    args = get_parser().parse_args()
    dataset_name = os.path.basename(os.path.normpath(args.file_root))

    if args.base_model_path is None:
        args.base_model_path = join('.', 'exp_ablation_adaptive',
                                    'config_1_adaptive', dataset_name, 'best_model.pth')
    if args.full_model_path is None:
        args.full_model_path = join('.', 'exp_new', dataset_name, 'best_model.pth')

    for path in (args.base_model_path, args.full_model_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"权重文件不存在: {path}")

    print(f"Base 权重: {args.base_model_path}")
    print(f"Full 权重: {args.full_model_path}")
    visualize_prior_heatmaps(args)


if __name__ == '__main__':
    main()

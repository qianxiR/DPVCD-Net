# DPVCD-Net: A Difference-Prior-Guided Pseudo-Video Network for Change Detection of Heterogeneous Targets in Complex Remote Sensing Scenes


## 概述

DPVCD-Net 接收一对配准的双时相遥感图像（T1 / T2），通过可学习的**感知帧（Perception Frame）**作为中间探针，构建三帧伪视频序列输入 X3D 3D 骨干网络提取时空特征。随后经过双路径注意力增强、ShuffleASPP 3D 多尺度上下文聚合、分层交叉 Transformer 融合，最终解码输出二值变化图。

**作者**: Duowang Zhu

## 核心架构

```
T1 ──┐                                    ┌── 变化图
      ├── [T1, P-frame, T2] ──▶ Encoder ──┤           (X3D-L 3D骨干)
T2 ──┘                                    │
                                           ├── DualPathAttention    (CBAM式双路径注意力)
                                           ├── ShuffleASPP3D        (时空通道混洗ASPP)
                                           ├── HierarchicalCrossAttn (Swin式分层交叉Transformer)
                                           └── ChangeDecoder        (渐进上采样解码器)
```

| 模块 | 说明 |
|------|------|
| **Encoder** | X3D-L 3D 卷积骨干，输出 4 个尺度 5D 特征 `(B,C,3,H,W)`，通道数 `[24,48,96,192]` |
| **DualPathAttention** | 并行（CA×SA）+ 串行（CA→SA）双路径通道/空间注意力 |
| **ShuffleASPP3D** | 4 分支 3D 空洞卷积（膨胀率 1,3,5,7），带时空通道混洗 |
| **HierarchicalCrossAttn** | 浅层 cross-attend 深层，深层 cross-attend 全部浅层，Swin 窗口移位 |
| **ChangeDecoder** | P + |T1−T2| 逐级转置卷积上采样，最终 sigmoid 输出 |

## 支持的数据集

| 数据集 | 描述 |
|--------|------|
| **GVLM-CD** | GVLM 变化检测数据集 |
| **WHU-CD** | WHU 建筑变化检测数据集 |
| **LBFD-CD** | LBFD 变化检测数据集 |

### 数据集下载

| 数据集 | 夸克网盘链接（本仓库处理后的版本） | 原始数据集链接 |
|--------|-------------|-------------|
| **GVLM-CD** | [https://pan.quark.cn/s/185e8e712b0f](https://pan.quark.cn/s/185e8e712b0f) | [https://github.com/zxk688/GVLM](https://github.com/zxk688/GVLM) |
| **LBFD-CD** | [https://pan.quark.cn/s/200a3dc6e99e](https://pan.quark.cn/s/200a3dc6e99e) | - |
| **WHU-CD** | [https://pan.quark.cn/s/d831fb64c1df](https://pan.quark.cn/s/d831fb64c1df) | [https://github.com/ChenHongruixuan/ChangeDetectionRepository/tree/master](https://github.com/ChenHongruixuan/ChangeDetectionRepository/tree/master) |

> 注：夸克网盘链接为本仓库处理过（已切分/格式化）的数据集；「原始数据集链接」指向各数据集的原始来源仓库。

数据集目录结构要求：
```
{dataset_root}/{dataset_name}/
├── train/
│   ├── t1/        # 时相1 RGB图像
│   ├── t2/        # 时相2 RGB图像
│   └── label/     # 二值变化标签
├── val/
│   ├── t1/
│   ├── t2/
│   └── label/
└── test/
    ├── t1/
    ├── t2/
    └── label/
```

## 项目结构

```
DPVCD-Net/
├── data/
│   ├── dataset.py              # BCDDataset 数据加载
│   ├── datasets_config.py      # 数据集路径集中配置
│   └── transforms.py           # 数据增强流水线
├── model/
│   ├── trainer.py              # 主训练器（完整模型组装）
│   ├── trainer_ablation.py     # 消融实验训练器（可配置模块开关）
│   ├── x3d.py                  # X3D 3D 卷积骨干网络
│   ├── attention.py            # 双路径注意力模块
│   ├── STMDCM.py               # ShuffleASPP 3D 模块
│   ├── cascade_dcn.py          # ShuffleASPP 3D 模块（V3）
│   ├── ST.py                   # 分层交叉注意力 Transformer
│   ├── position_encoding.py    # 边缘增强正弦位置编码
│   ├── CosineSimilarity.py     # 余弦相似度增强
│   ├── changedecoder.py        # 变化解码器（5D输入）
│   ├── change_decoder.py       # 变化解码器（4D输入）
│   ├── utils.py                # 损失函数、学习率调度、评估器
│   └── X3D_L.pyth              # X3D-L 预训练权重
├── scripts/
│   ├── train/
│   │   ├── train_BCD.py        # 单数据集训练
│   │   ├── batch_train.py      # 多数据集批量训练
│   │   ├── train_ablation.py   # 消融实验训练
│   │   └── batch_ablation_train.py
│   ├── test/
│   │   ├── test_BCD.py         # 单数据集测试
│   │   ├── batch_test.py       # 批量测试
│   │   ├── test_ablation.py    # 消融实验测试
│   │   └── batch_test_ablation.py
│   ├── visualize/
│   │   ├── visualize_BCD.py                # 变化检测结果可视化
│   │   ├── visualize_ablation.py           # 消融对比可视化
│   │   ├── visualize_ablation_by_image.py  # 按图像消融可视化
│   │   ├── visualize_ablation_tsne.py      # t-SNE 特征可视化
│   │   ├── visualize_heatmaps_from_existing.py
│   │   └── extract_logits_and_attention.py # 中间特征提取
│   └── utils/
│       ├── copy_original_images.py
│       └── generate_mapping_from_vis.py
├── utils/
│   └── metric_tool.py          # 评估指标（F1, IoU, Kappa, OA）
├── exp_BCD/                     # 实验输出目录
├── visresult/                   # 可视化结果
└── requirements.txt
```

## 环境配置

### 依赖安装

```powershell
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 依赖列表

| 包 | 版本 |
|---|------|
| PyTorch | 2.2.0 (CUDA 11.8) |
| pytorchvideo | 0.1.5 |
| einops | 0.8.1 |
| fvcore | 0.1.5 |
| opencv-python | 4.11.0 |
| scikit-image | 0.25.2 |
| matplotlib | 3.10.1 |
| tqdm | 4.67.1 |

## 使用说明

### 数据集路径配置

编辑 [data/datasets_config.py](data/datasets_config.py) 中的 `DATASETS` 列表，设置各数据集的磁盘路径。

### 训练

**方式1: 自动训练所有已配置数据集（推荐）**
```powershell
python scripts/train/train_BCD.py --save_dir ./exp_BCD --batch_size 8
```

**方式2: 按名称训练指定数据集**
```powershell
python scripts/train/train_BCD.py --dataset_name GVLM-CD --batch_size 8
python scripts/train/train_BCD.py --dataset_name WHU-CD --batch_size 8
python scripts/train/train_BCD.py --dataset_name LBFD-CD --batch_size 8
```

**方式3: 指定数据集完整路径**
```powershell
python scripts/train/train_BCD.py --file_root "E:\data\GVLM-CD" --batch_size 8
```

**方式4: 从检查点恢复训练**
```powershell
python scripts/train/train_BCD.py --dataset_name GVLM-CD --resume ./exp_BCD/GVLM-CD/checkpoint.pth.tar --batch_size 6
```

**批量训练所有数据集**
```powershell
python scripts/train/batch_train.py --batch_size 8
```

### 测试

```powershell
python scripts/test/test_BCD.py --file_root "E:\data\GVLM-CD" --batch_size 8 --gpu_id 0
python scripts/test/batch_test.py --batch_size 8
```

### 消融实验

支持 7 种消融配置，通过 `experiment_id` 控制模块组合：

| ID | 配置 |
|----|------|
| 1 | 仅基础（Encoder → Decoder） |
| 2 | 基础 + Attention |
| 3 | 基础 + ASPP |
| 4 | 基础 + Transformer |
| 5 | 基础 + Attention + ASPP |
| 6 | 基础 + Attention + Transformer |
| 7 | 基础 + ASPP + Transformer |

```powershell
python scripts/train/train_ablation.py --dataset_name GVLM-CD --batch_size 8
python scripts/train/batch_ablation_train.py --batch_size 8
```

### 可视化

```powershell
python scripts/visualize/visualize_BCD.py --dataset_name GVLM-CD --batch_size 8
python scripts/visualize/visualize_ablation.py --dataset_name GVLM-CD --batch_size 8
python scripts/visualize/visualize_ablation_tsne.py --dataset_name GVLM-CD
```

## 主要训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_steps` | 80000 | 最大训练迭代数 |
| `--batch_size` | 8 | 批次大小 |
| `--learning_rate` | 0.0002 | 初始学习率 |
| `--lr_mode` | poly | 学习率衰减模式（多项式） |
| `--power` | 0.9 | 多项式衰减幂次 |
| `--num_workers` | 0 | 数据加载进程数（Windows 建议 0） |
| `--val_interval` | 1 | 验证间隔（epoch） |
| `--gpu_id` | 0 | GPU 设备 ID |

## 评估指标

- **F1 Score**（主要指标，用于保存最佳模型）
- **IoU**（交并比）
- **Kappa**（Kappa 系数）
- **OA**（总体精度）
- **Recall**（召回率）
- **Precision**（精确率）

## 损失函数

组合使用 **BCEDiceLoss**（二元交叉熵 + Dice Loss），位于 [model/utils.py](model/utils.py)。同时提供 `FocalLoss` 和 `BCEDiceFocalLoss` 作为备选。

## License

Copyright (c) Duowang Zhu. All rights reserved.

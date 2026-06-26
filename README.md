# DPVCD-Net: A Difference-Prior-Guided Pseudo-Video Network for Change Detection of Heterogeneous Targets in Complex Remote Sensing Scenes


## 概述

DPVCD-Net 接收一对配准的双时相遥感图像（T1 / T2），通过可学习的**感知帧（Perception Frame）**作为中间探针，构建三帧伪视频序列输入 X3D 3D 骨干网络提取时空特征。随后经过双路径注意力增强、ShuffleASPP 3D 多尺度上下文聚合、分层交叉 Transformer 融合，最终解码输出二值变化图。

**作者**: qianxiR

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
| **ChangeDecoder** | P + \|T1−T2\| 逐级转置卷积上采样，最终 sigmoid 输出 |

### 数据前向流程

1. T1、T2 与可学习感知帧 P 拼接为三帧伪视频 `[T1, P, T2]`，形状 `[B, 3, 3, H, W]`
2. X3D 骨干逐阶段提取 4 个尺度的 5D 时空特征（含余弦相似度增强）
3. 每尺度经 DualPathAttention（通道+空间注意力）与 ShuffleASPP3D（多尺度时空聚合）增强
4. HierarchicalCrossAttn 在 4 尺度间做分层交叉注意力，P 帧为 Q 源、`cat(T1,T2)` 为 K 源、余弦相似度为 V 源
5. ChangeDecoder 渐进上采样 + 跳跃连接，输出 `[B, 1, 256, 256]` 变化概率图（阈值 0.5 二值化）

## 支持的数据集

| 数据集 | 描述 |
|--------|------|
| **GVLM-CD** | GVLM 变化检测数据集 |
| **WHU-CD** | WHU 建筑变化检测数据集 |
| **LBFD-CD** | LBFD 变化检测数据集 |

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

## 环境配置

### 依赖安装

建议使用 **Python 3.10**。先按 CUDA 版本安装 PyTorch 2.6.0：

```powershell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

```powershell
pip install -r requirements.txt
```

### 依赖列表

| 包 | 版本 |
|---|------|
| Python | 3.10 |
| PyTorch | 2.6.0 (CUDA 11.8) |
| torchvision | 0.21.0 |
| torchaudio | 2.6.0 |
| pytorchvideo | 0.1.5 |
| einops | 0.8.1 |
| fvcore | 0.1.5 |
| opencv-python | 4.11.0 |
| scikit-image | 0.25.2 |
| matplotlib | 3.10.1 |
| tqdm | 4.67.1 |

## 训练（完整模型）

完整模型训练脚本为 `scripts/train/train_BCD.py`。修改 `data/datasets_config.py` 中的 `DATASETS` 列表设置数据集磁盘路径后，按以下任一方式启动：

```powershell
python scripts/train/train_BCD.py --dataset_name GVLM-CD --save_dir ./exp_BCD --batch_size 8
```

```powershell
python scripts/train/train_BCD.py --file_root "E:\rqx\dataes\GVLM-CD" --save_dir ./exp_BCD --batch_size 8
```

```powershell
python scripts/train/train_BCD.py --save_dir ./exp_BCD --batch_size 8
```

训练产物保存在 `{save_dir}/{dataset_name}/`：`best_model.pth`（验证集最佳 F1 对应权重）、`checkpoint.pth.tar`（断点续训）、`final_model.pth`（测试集最终评估模型）、`train_val_log.txt`（训练验证日志）。

### 从检查点恢复训练

```powershell
python scripts/train/train_BCD.py --dataset_name GVLM-CD --resume ./exp_BCD/GVLM-CD/checkpoint.pth.tar --batch_size 6
```

### 主要训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_steps` | 80000 | 最大训练迭代数 |
| `--batch_size` | 8 | 批次大小 |
| `--learning_rate` | 0.0002 | 初始学习率（Adam，poly 衰减） |
| `--lr_mode` | poly | 学习率衰减模式（多项式） |
| `--power` | 0.9 | 多项式衰减幂次 |
| `--num_workers` | 0 | 数据加载进程数（Windows 建议 0） |
| `--val_interval` | 1 | 验证间隔（epoch） |
| `--gpu_id` | 0 | GPU 设备 ID |
| `--in_height` / `--in_width` | 256 | 输入图像尺寸 |

## 测试推理

测试脚本为 `scripts/test/test_BCD.py`，加载已训练模型在测试集上评估：

```powershell
python scripts/test/test_BCD.py --file_root "E:\rqx\dataes\GVLM-CD" --model_path ./exp_BCD/GVLM-CD/best_model.pth --batch_size 8 --gpu_id 0
```

```powershell
python scripts/test/test_BCD.py --file_root "E:\rqx\dataes\WHU-CD" --model_path ./exp_BCD/WHU-CD/best_model.pth --save_predictions --output_dir ./predictions --batch_size 8
```

### 评估指标

- **F1 Score**（主要指标，用于保存最佳模型）
- **IoU**（交并比）
- **Kappa**（Kappa 系数）
- **OA**（总体精度）
- **Recall**（召回率）
- **Precision**（精确率）

## License

Copyright (c) qianxiR. All rights reserved.

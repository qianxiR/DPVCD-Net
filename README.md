# DPVCD-Net:面向复杂遥感场景异质目标变化检测的差分先验引导伪视频网络

本仓库是论文 *"DPVCD-Net: A Difference-Prior-Guided Pseudo-Video Network for Change Detection of Heterogeneous Targets in Complex Remote Sensing Scenes"* 的官方 PyTorch 实现。

## 目录

- 数据集准备
- 训练
- 评估
- 推理
- 引用
- 致谢

## 数据集准备

下载 GVLM-CD / WHU-CD / LBFD-CD 数据集,并按如下结构组织(三个数据集共用同一父目录):

```
{dataset_root}/
├─GVLM-CD
│  ├─train
│  │  ├─t1              # T1 时相影像
│  │  ├─t2              # T2 时相影像
│  │  └─label           # 二值变化标签 (0/255)
│  ├─val
│  │  ├─t1
│  │  ├─t2
│  │  └─label
│  └─test
│     ├─t1
│     ├─t2
│     └─label
├─WHU-CD
│  └─...
└─LBFD-CD
   └─...
```

数据集路径在 `data/datasets_config.py` 的 `DATASETS` 列表中注册,例如:

```python
DATASETS = [
    dict(
        name='GVLM-CD',
        root_dir=r'E:\rqx\dataes\GVLM-CD',
    ),
    dict(
        name='WHU-CD',
        root_dir=r'E:\rqx\dataes\WHU-CD',
    ),
    dict(
        name='LBFD-CD',
        root_dir=r'E:\rqx\dataes\LBFD-CD',
    ),
]
```

## 训练

DPVCD-Net 的总体架构:

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
| **Encoder** | X3D-L 3D 卷积骨干,输出 4 个尺度 5D 特征 `(B,C,3,H,W)`,通道数 `[24,48,96,192]` |
| **DualPathAttention** | 并行(CA×SA)+ 串行(CA→SA)双路径通道/空间注意力 |
| **ShuffleASPP3D** | 4 分支 3D 空洞卷积(膨胀率 1,3,5,7),带时空通道混洗 |
| **HierarchicalCrossAttn** | 浅层 cross-attend 深层,深层 cross-attend 全部浅层,Swin 窗口移位 |
| **ChangeDecoder** | P + \|T1−T2\| 逐级转置卷积上采样,最终 sigmoid 输出 |

### 环境安装

**Step 1**:创建 conda 环境并激活(建议 Python 3.10)。

```powershell
conda create -n dpvcdnet python=3.10
conda activate dpvcdnet
```

**Step 2**:克隆仓库。

```powershell
git clone https://github.com/qianxiR/DPVCD-Net.git
cd .\DPVCD-Net
```

**Step 3**:按 CUDA 版本安装 PyTorch 2.6.0,再安装其余依赖。

```powershell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

```powershell
pip install -r requirements.txt
```

依赖列表:

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

### 开始训练

```powershell
python .\scripts\train\train_BCD.py --dataset_name GVLM-CD --save_dir .\exp_BCD --batch_size 8
```

默认配置:`img_size=256, batch_size=8, lr=2e-4, Adam, poly 衰减, 80000 steps`。切换数据集、损失或训练超参请编辑 `scripts/train/train_BCD.py` 的 `ArgumentParser` 块。

训练产物保存在 `{save_dir}/{dataset_name}/`:`best_model.pth`(验证集最佳 F1 对应权重)、`checkpoint.pth.tar`(断点续训)、`final_model.pth`(测试集最终评估模型)、`train_val_log.txt`(训练验证日志)。

### 主要训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_steps` | 80000 | 最大训练迭代数 |
| `--batch_size` | 8 | 批次大小 |
| `--learning_rate` | 0.0002 | 初始学习率(Adam,poly 衰减) |
| `--lr_mode` | poly | 学习率衰减模式(多项式) |
| `--power` | 0.9 | 多项式衰减幂次 |
| `--num_workers` | 0 | 数据加载进程数(Windows 建议 0) |
| `--val_interval` | 1 | 验证间隔(epoch) |
| `--gpu_id` | 0 | GPU 设备 ID |
| `--in_height` / `--in_width` | 256 | 输入图像尺寸 |

### 从检查点恢复训练

```powershell
python .\scripts\train\train_BCD.py --dataset_name GVLM-CD --resume .\exp_BCD\GVLM-CD\checkpoint.pth.tar --batch_size 6
```

## 评估

```powershell
python .\scripts\test\test_BCD.py --file_root "E:\rqx\dataes\GVLM-CD" --model_path .\exp_BCD\GVLM-CD\best_model.pth --batch_size 8 --gpu_id 0
```

评估指标:

- **F1 Score**(主要指标,用于保存最佳模型)
- **IoU**(交并比)
- **Kappa**(Kappa 系数)
- **OA**(总体精度)
- **Recall**(召回率)
- **Precision**(精确率)

## 推理

```powershell
python .\scripts\test\test_BCD.py --file_root "E:\rqx\dataes\WHU-CD" --model_path .\exp_BCD\WHU-CD\best_model.pth --save_predictions --output_dir .\predictions --batch_size 8
```

开启 `--save_predictions` 后,预测结果保存至 `--output_dir` 指定目录。

## 引用

如果您觉得本仓库对您的研究有帮助,请考虑引用:

```bibtex
@article{DPVCDNet202X,
  title   = {DPVCD-Net: A Difference-Prior-Guided Pseudo-Video Network for Change Detection of Heterogeneous Targets in Complex Remote Sensing Scenes},
  author  = {...},
  journal = {...},
  year    = {202X},
  volume  = {},
  number  = {},
  pages   = {},
  doi     = {}
}
```

## 致谢

感谢以下开源仓库:X3D、pytorchvideo、BIT、ChangeFormer、Swin Transformer。

## License

Copyright (c) qianxiR. All rights reserved. 本仓库代码仅用于学术研究。

## 联系方式

如有任何问题,欢迎提 issue 或联系作者 qianxiR。

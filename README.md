# DPVCD-Net
Difference-Prior-Guided Pseudo-Video Network for Change Detection of Heterogeneous Targets in Complex Remote Sensing Scenes

# Requirement
- Python 3.10
- PyTorch 2.6.0 (CUDA 11.8)
- torchvision 0.21.0
- torchaudio 2.6.0
- pytorchvideo 0.1.5
- einops 0.8.1
- fvcore 0.1.5
- opencv-python 4.11.0
- scikit-image 0.25.2
- matplotlib 3.10.1
- tqdm 4.67.1

# Dataset
The dataset should consist of bi-temporal image pairs (T1/T2) with pixel-level binary change labels. Each sample provides the pre-event image (T1), the post-event image (T2), and the binary change mask (label, 0/255).

For change detection of heterogeneous targets in complex remote sensing scenes, download the [GVLM-CD](https://github.com/zxk688/GVLM), [WHU-CD](https://gpcv.whu.edu.cn/data/building_dataset.html) and LBFD-CD datasets. Organize the dataset into the following structure (three datasets share the same parent directory):

```
/E:/rqx/dataes
  /GVLM-CD
    /train
      /t1                  T1 时相影像
      /t2                  T2 时相影像
      /label               二值变化标签 (0/255)
    /val
      /t1
      /t2
      /label
    /test
      /t1
      /t2
      /label
  /WHU-CD
    ...
  /LBFD-CD
    ...
```

Then register your dataset path in `data/datasets_config.py`, e.g.:

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

# Working Example

0. Environment
```powershell
cd DPVCD-Net
conda create -n dpvcdnet python=3.10
conda activate dpvcdnet
# PyTorch 2.6.0 + CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

1. Training the Model.

(1) Prepare the Dataset. Make sure your dataset is structured as described above and register its path in `data/datasets_config.py`. Adjust the `--dataset_name` argument in `scripts/train/train_BCD.py` to point to your dataset. (2) Training Script. To train the DPVCD-Net model, run the following command:
```powershell
python .\scripts\train\train_BCD.py --dataset_name GVLM-CD --save_dir .\exp_BCD --batch_size 8
```
Note: The model parameters are set to `img_size=256`, `batch_size=8`, `max_steps=80000`, `learning_rate=2e-4` (Adam, poly decay), which you can modify according to your needs. To switch datasets, losses or training hyperparameters, edit the `ArgumentParser` block in `scripts/train/train_BCD.py`. (3) Model Output. The training artifacts are saved in `{save_dir}/{dataset_name}/`, including `best_model.pth` (weights of the best F1 on the validation set), `checkpoint.pth.tar` (for resuming), `final_model.pth` (final model evaluated on the test set), and `train_val_log.txt` (training/validation log).

2. Testing the Model.

(1) Prepare the Dataset. Make sure your dataset is structured as described above and adjust the `--file_root` argument in `scripts/test/test_BCD.py` to point to your testing data folder. (2) Testing Script. To evaluate the trained model on the test dataset, run the following command:
```powershell
python .\scripts\test\test_BCD.py --file_root "E:\rqx\dataes\GVLM-CD" --model_path .\exp_BCD\GVLM-CD\best_model.pth --batch_size 8 --gpu_id 0
```
Note: The `test_BCD.py` script loads the pre-trained model from `--model_path`. You can select the appropriate pre-trained model parameters as input. (3) Model Output. The evaluation metrics (F1, IoU, Kappa, OA, Recall, Precision) are printed to the console, where F1 is the primary metric. With `--save_predictions` enabled, the prediction maps are saved to the `--output_dir` folder:
```powershell
python .\scripts\test\test_BCD.py --file_root "E:\rqx\dataes\WHU-CD" --model_path .\exp_BCD\WHU-CD\best_model.pth --save_predictions --output_dir .\predictions --batch_size 8
```

Additional Note: To resume training from a checkpoint, run:
```powershell
python .\scripts\train\train_BCD.py --dataset_name GVLM-CD --resume .\exp_BCD\GVLM-CD\checkpoint.pth.tar --batch_size 6
```

The main training parameters are as follows:

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

# Acknowledgments

The dataset is constructed based on the GVLM-CD and WHU-CD change detection datasets. Thanks for their excellent works!

```
@Article{Chen2020,
AUTHOR = {Chen, Hao and Shi, Zhenwei},
TITLE = {A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection},
JOURNAL = {Remote Sensing},
VOLUME = {12},
YEAR = {2020},
NUMBER = {10},
ARTICLE-NUMBER = {1662},
URL = {https://www.mdpi.com/2072-4292/12/10/1662},
ISSN = {2072-4292},
DOI = {10.3390/rs12101662}
}

@article{ZHANG20231,
title = {Cross-domain landslide mapping from large-scale remote sensing images using prototype-guided domain-aware progressive representation learning},
journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
volume = {197},
pages = {1-17},
year = {2023},
issn = {0924-2716},
doi = {https://doi.org/10.1016/j.isprsjprs.2023.01.018},
url = {https://www.sciencedirect.com/science/article/pii/S0924271623000242},
author = {Xiaokang Zhang and Weikang Yu and Man-On Pun and Wenzhong Shi}
}
```

# Citation
If you use this code for your research, please cite our paper.

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

Copyright (c) qianxiR. All rights reserved. 本仓库代码仅用于学术研究。如有任何问题,欢迎提 issue 或联系作者 qianxiR。

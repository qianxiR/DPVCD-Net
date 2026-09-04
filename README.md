# DPVCD-Net: A Difference-Prior-Guided Pseudo-Video Network for Change Detection of Heterogeneous Targets in Complex Remote Sensing Scenes

DPVCD-Net — Official PyTorch implementation

## Overview

DPVCD-Net is a difference-prior-guided pseudo-video network for change detection of heterogeneous targets in complex remote sensing scenes. It reformulates bi-temporal change detection as pseudo-video modeling with an X3D backbone, guided by a difference prior between the pre-event and post-event images.

DPVCD-Net is evaluated on three change detection datasets: GVLM-CD, WHU-CD, and LBFD-CD, where LBFD-CD is a benchmark of 3,415 bi-temporal image pairs with heterogeneous change targets (landslide and building) constructed in this work. Each sample contains:

- `t1`: pre-event image
- `t2`: post-event image
- `label`: binary change mask with values of 0 and 255

## Environment Setup

The main training and evaluation environment is Python 3.10, PyTorch 2.6.0, CUDA 11.8:

```powershell
git clone https://github.com/qianxiR/DPVCD-Net.git
cd DPVCD-Net

conda create -n dpvcdnet python=3.10
conda activate dpvcdnet

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

For Windows environments, `--num_workers 0` is recommended when multiprocessing-related data-loading issues occur.

## Datasets and Models

| Resource | Description | Link |
| --- | --- | --- |
| GVLM-CD | Landslide change detection dataset | [GitHub](https://github.com/zxk688/GVLM) |
| WHU-CD | Building change detection dataset | [Website](https://gpcv.whu.edu.cn/data/building_dataset.html) |
| LBFD-CD | 3,415 bi-temporal heterogeneous-target pairs constructed in this work | [Baidu Netdisk](https://pan.baidu.com/s/1HpPVmJrj133vWVuO1FIIzA), access code: `b46g` |
| X3D-L | Kinetics pre-trained X3D-L backbone | [Download](https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_L.pyth) |

### Download the Pre-trained X3D-L Backbone

Download the X3D-L backbone pre-trained on Kinetics and place it at:

```text
model/X3D_L.pyth
```

This is the default `--pretrained` path used by the training and testing scripts. For Windows PowerShell:

```powershell
curl -L -o .\model\X3D_L.pyth https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_L.pyth
```

### Directory Structure

The three datasets should follow the same directory structure:

```text
<DATA_ROOT>/
├── GVLM-CD/
│   ├── train/
│   │   ├── t1/
│   │   ├── t2/
│   │   └── label/
│   ├── val/
│   │   ├── t1/
│   │   ├── t2/
│   │   └── label/
│   └── test/
│       ├── t1/
│       ├── t2/
│       └── label/
│
├── WHU-CD/
│   ├── train/
│   │   ├── t1/
│   │   ├── t2/
│   │   └── label/
│   ├── val/
│   │   ├── t1/
│   │   ├── t2/
│   │   └── label/
│   └── test/
│       ├── t1/
│       ├── t2/
│       └── label/
│
└── LBFD-CD/
    ├── train/
    │   ├── t1/
    │   ├── t2/
    │   └── label/
    ├── val/
    │   ├── t1/
    │   ├── t2/
    │   └── label/
    └── test/
        ├── t1/
        ├── t2/
        └── label/
```

Each train, val, and test directory should contain the corresponding t1, t2, and label folders.

### Dataset Configuration

After preparing the datasets, register their paths in:

```text
data/datasets_config.py
```

For example:

```python
DATASETS = [
    dict(
        name='GVLM-CD',
        root_dir=r'<DATA_ROOT>\GVLM-CD',
    ),
    dict(
        name='WHU-CD',
        root_dir=r'<DATA_ROOT>\WHU-CD',
    ),
    dict(
        name='LBFD-CD',
        root_dir=r'<DATA_ROOT>\LBFD-CD',
    ),
]
```

Replace `<DATA_ROOT>` with the actual path on your machine.

### LBFD-CD Benchmark

To facilitate reproducibility, we provide access to the complete LBFD-CD benchmark used in this study, together with the corresponding benchmark split files and preprocessing instructions.

#### Download

- Dataset: `LBFD-CD.zip`
- Baidu Netdisk: <https://pan.baidu.com/s/1HpPVmJrj133vWVuO1FIIzA>
- Access code: `b46g`

LBFD-CD contains 3,415 bi-temporal image pairs, including:

- 819 landslide samples
- 2,596 building samples

The benchmark is constructed from the publicly available GVLM-CD and WHU-CD datasets following the procedure described below.

#### Construction

**1. Landslide subset**

The landslide subset is derived from the following regions of GVLM-CD:

- Jiuzhaigou
- Shimen
- Taitung

A total of 819 landslide patches are included.

**2. Building subset**

The building subset is generated from the original TEST large image of WHU-CD.

A total of 2,596 building patches are included.

**3. Patch generation**

The source bi-temporal images and their corresponding change masks are partitioned using:

- grid-based cropping
- patch size: 256 × 256
- non-overlapping patches

The same cropping coordinates are applied to the T1 image, T2 image, and corresponding change mask.

**4. Dataset split**

After patch generation, the landslide and building subsets are independently divided into training, validation, and test sets at a ratio of 7:2:1 using a fixed random seed of 42.

The corresponding subsets from the two target categories are then merged to construct the final LBFD-CD benchmark.

#### Dataset Statistics

| Category  | Train | Validation | Test | Total |
| --------- | ----- | ---------- | ---- | ----- |
| Landslide | 575   | 161        | 83   | 819   |
| Building  | 1,815 | 522        | 259  | 2,596 |
| LBFD-CD   | 2,390 | 683        | 342  | 3,415 |

#### Note on Dataset Splitting

The LBFD-CD training, validation, and test sets are not geographically disjoint.

Each target category is randomly divided after patch generation, and spatially adjacent patches may therefore occur in different subsets. However:

- all generated image patches are non-overlapping;
- no identical patch is shared among the training, validation, and test sets.

Accordingly, LBFD-CD is designed to evaluate shared-parameter learning for heterogeneous change targets, rather than geographically held-out generalization.

For exact reproduction of the experiments reported in the paper, please use the provided benchmark split files rather than regenerating the train/validation/test splits.

## Repository Structure

```text
DPVCD-Net/
├── data/                  # Dataset loading and configuration
├── data_preparation/      # LBFD-CD construction scripts and split files
├── model/                 # DPVCD-Net modules, losses, and trainer
│   └── X3D_L.pyth         # Kinetics pre-trained X3D-L backbone
├── scripts/
│   ├── train/             # Training entry points
│   ├── test/              # Evaluation entry points
│   └── visualize/         # Visualization tools
├── utils/                 # Evaluation metrics
├── requirements.txt
└── README.md
```

## Training

### Basic Training

Make sure that the dataset follows the directory structure described above and that the dataset root is registered in `data/datasets_config.py`. Select the desired dataset using the `--dataset_name` argument.

The default training configuration is:

- Input size: 256 × 256
- Batch size: 8
- Maximum epochs: 100
- Initial learning rate: 2e-4
- Optimizer: Adam
- Learning-rate schedule: polynomial decay

For example, to train on GVLM-CD:

```powershell
python .\scripts\train\train_BCD.py --dataset_name GVLM-CD --save_dir .\exp_BCD --batch_size 8
```

### Training Outputs

Training results are saved under `{save_dir}/{dataset_name}/`. The output directory contains:

- `best_model.pth`: model with the best validation F1 score
- `checkpoint.pth.tar`: checkpoint used for resuming training
- `final_model.pth`: final model after training
- `train_val_log.txt`: training and validation log

### Resume Training

Training can be resumed from an existing checkpoint:

```powershell
python .\scripts\train\train_BCD.py --dataset_name GVLM-CD --resume .\exp_BCD\GVLM-CD\checkpoint.pth.tar --batch_size 8
```

### Main Training Parameters

| Parameter         | Default | Description                        |
| ----------------- | ------- | ---------------------------------- |
| `--max_epochs`    | 100     | Maximum number of training epochs  |
| `--batch_size`    | 8       | Batch size                         |
| `--learning_rate` | 0.0002  | Initial learning rate              |
| `--lr_mode`       | poly    | Learning-rate decay strategy       |
| `--power`         | 0.9     | Polynomial decay power             |
| `--num_workers`   | 0       | Number of data-loading workers     |
| `--val_interval`  | 1       | Validation interval in epochs      |
| `--gpu_id`        | 0       | GPU device ID                      |
| `--in_height`     | 256     | Input image height                 |
| `--in_width`      | 256     | Input image width                  |

The training configuration can be modified through the arguments defined in `scripts/train/train_BCD.py`.

## Testing

### Evaluate a Trained Model

Use `scripts/test/test_BCD.py` to evaluate a trained model:

```powershell
python .\scripts\test\test_BCD.py --file_root "<DATA_ROOT>\GVLM-CD" --model_path .\exp_BCD\GVLM-CD\best_model.pth --batch_size 8 --gpu_id 0
```

The evaluation script reports the following metrics, and F1 is used as the primary evaluation metric in the experiments:

- F1 score
- Intersection over Union (IoU)
- Kappa
- Overall Accuracy (OA)
- Recall
- Precision

### Save Prediction Maps

To save the predicted change maps, enable `--save_predictions`:

```powershell
python .\scripts\test\test_BCD.py --file_root "<DATA_ROOT>\WHU-CD" --model_path .\exp_BCD\WHU-CD\best_model.pth --save_predictions --output_dir .\predictions --batch_size 8
```

The predicted change maps will be saved to the directory specified by `--output_dir`.

## Acknowledgments

The LBFD-CD benchmark is constructed based on the publicly available GVLM-CD and WHU-CD change detection datasets. We sincerely thank the authors of these datasets for making their data and research publicly available.

```bibtex
@article{Chen2020,
  author = {Chen, Hao and Shi, Zhenwei},
  title = {A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection},
  journal = {Remote Sensing},
  volume = {12},
  year = {2020},
  number = {10},
  article-number = {1662},
  doi = {10.3390/rs12101662}
}
```

```bibtex
@article{ZHANG20231,
  title = {Cross-domain landslide mapping from large-scale remote sensing images using prototype-guided domain-aware progressive representation learning},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  volume = {197},
  pages = {1--17},
  year = {2023},
  doi = {10.1016/j.isprsjprs.2023.01.018},
  author = {Zhang, Xiaokang and Yu, Weikang and Pun, Man-On and Shi, Wenzhong}
}
```

## Citation

If you find this work, code, or the LBFD-CD benchmark useful in your research, please cite our paper:

```bibtex
@article{DPVCDNet2026,
  title   = {DPVCD-Net: A Difference-Prior-Guided Pseudo-Video Network for Change Detection of Heterogeneous Targets in Complex Remote Sensing Scenes},
  author  = {Zhang, Q. and Qian, X. and Wang, Z. and Zhang, Z. and Yang, S. and Xu, X.},
  journal = {The Photogrammetric Record},
  year    = {2026}
}
```

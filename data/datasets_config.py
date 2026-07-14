"""
数据集路径集中配置

所有脚本统一从此文件加载数据集路径，避免在多处硬编码。
修改数据集路径时只需编辑本文件即可。
"""

import os

# ---------------------------------------------------------------------------
# 数据集定义
# name          - 数据集标识名（同时用于实验保存目录命名）
# path          - 数据集在磁盘上的绝对路径
# description   - 仅供终端打印的人类可读描述
# batch_size    - 该数据集的推荐批次大小（批量脚本中作为默认值）
# ---------------------------------------------------------------------------

DATASETS = [
    {
        'name': 'GVLM-CD',
        'path': r'G:\deeplearning\cd\DPVCD-Net\GVLM_CD\GVLM-CD',
        'description': 'GVLM变化检测数据集',
        'batch_size': 8,
    },
    {
        'name': 'WHU-CD',
        'path': r'G:\deeplearning\cd\DPVCD-Net\WHU_CD\WHU-CD',
        'description': 'WHU建筑变化检测数据集',
        'batch_size': 8,
    },
    {
        'name': 'LBFD-CD',
        'path': r'G:\deeplearning\cd\DPVCD-Net\LBFD_CD\LBFD-CD',
        'description': 'LBFD变化检测数据集',
        'batch_size': 8,
    },
]


def get_dataset_by_name(name: str):
    """按名称查找数据集配置，未找到返回 None。"""
    for ds in DATASETS:
        if ds['name'] == name:
            return ds
    return None


def get_dataset_root():
    """
    推断数据集公共父目录。
    取所有数据集路径的公共前缀目录（去掉末尾斜杠后逐级回退）。
    """
    paths = [ds['path'] for ds in DATASETS]
    if not paths:
        return ''
    common = os.path.commonpath(paths)
    return common
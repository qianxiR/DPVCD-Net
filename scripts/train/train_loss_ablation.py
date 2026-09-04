import subprocess
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SCRIPT = PROJECT_ROOT / 'scripts' / 'train' / 'train_BCD.py'
LOSS_EXPERIMENTS: Dict[str, Dict[str, object]] = {
    'loss_fixed_000': {'mode': 'fixed', 'weight': 0.0},
    'loss_fixed_030': {'mode': 'fixed', 'weight': 0.3},
    'loss_fixed_050': {'mode': 'fixed', 'weight': 0.5},
    'loss_fixed_070': {'mode': 'fixed', 'weight': 0.7},
    'loss_fixed_090': {'mode': 'fixed', 'weight': 0.9},
    'loss_fixed_100': {'mode': 'fixed', 'weight': 1.0},
    'loss_adaptive': {'mode': 'adaptive', 'weight': 0.3},
}


def get_parser() -> ArgumentParser:
    """
    入参:
    - 无

    方法:
    - 定义LBFD完整模型损失权重实验所需的路径与训练参数

    出参:
    - parser (ArgumentParser): 命令行参数解析器
    """
    parser = ArgumentParser(description='LBFD损失权重消融实验')
    parser.add_argument(
        '--file_root',
        type=Path,
        default=Path(r'E:\rqx\dataes\LBFD-CD'),
        help='LBFD-CD数据集目录。',
    )
    parser.add_argument(
        '--save_dir',
        type=Path,
        default=Path('exp_loss_ablation'),
        help='四组损失实验的保存根目录。',
    )
    parser.add_argument('--gpu_id', default='0', help='使用的GPU ID。')
    parser.add_argument('--batch_size', type=int, default=8, help='训练批次大小。')
    parser.add_argument('--max_epochs', type=int, default=100, help='最大训练轮数，运行时按每轮批次数折算为总迭代数。')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载进程数。')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='初始学习率。')
    return parser


def build_command(
    args: Namespace,
    experiment_name: str,
    experiment: Dict[str, object],
) -> list[str]:
    """
    入参:
    - args (Namespace): 数据、输出与训练参数
    - experiment_name (str): 损失实验ID
    - experiment (dict): 固定或自适应模式及其权重

    方法:
    - 为完整DPVCD-Net构造train_BCD.py命令，并在检查点存在时追加恢复参数

    出参:
    - command (list[str]): 可直接交给subprocess执行的参数列表
    """
    experiment_dir = args.save_dir / experiment_name
    checkpoint_path = experiment_dir / args.file_root.name / 'checkpoint.pth.tar'
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
        '--file_root',
        str(args.file_root),
        '--save_dir',
        str(experiment_dir),
        '--loss_mode',
        str(experiment['mode']),
        '--loss_weight',
        str(experiment['weight']),
        '--gpu_id',
        args.gpu_id,
        '--batch_size',
        str(args.batch_size),
        '--max_epochs',
        str(args.max_epochs),
        '--num_workers',
        str(args.num_workers),
        '--learning_rate',
        str(args.learning_rate),
    ]
    if checkpoint_path.is_file():
        command.extend(['--resume', str(checkpoint_path)])
    return command


def run_experiment(
    args: Namespace,
    experiment_name: str,
    experiment: Dict[str, object],
) -> None:
    """
    入参:
    - args (Namespace): 数据、输出与训练参数
    - experiment_name (str): 损失实验ID
    - experiment (dict): 固定或自适应模式及其权重

    方法:
    - 跳过已有最终模型的实验，否则启动完整模型训练并自动恢复检查点

    出参:
    - None
    """
    experiment_dir = args.save_dir / experiment_name
    final_model_path = experiment_dir / args.file_root.name / 'final_model.pth'
    if final_model_path.is_file():
        print(f'跳过已完成实验: {experiment_name} ({final_model_path})')
        return

    print(
        f"开始实验: {experiment_name}, mode={experiment['mode']}, "
        f"w={experiment['weight']}"
    )
    subprocess.run(
        build_command(args, experiment_name, experiment),
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    """
    入参:
    - 命令行中的LBFD路径、输出目录与训练参数

    方法:
    - 依次调度三组固定权重和一组自适应权重实验

    出参:
    - None
    """
    args = get_parser().parse_args()
    for experiment_name, experiment in LOSS_EXPERIMENTS.items():
        run_experiment(args, experiment_name, experiment)


if __name__ == '__main__':
    main()

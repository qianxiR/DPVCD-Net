
import os
import sys
import time
import numpy as np
from os.path import join
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# 插入当前路径以导入本地模块
sys.path.insert(0, '.')

from data.datasets_config import get_dataset_root

import data.dataset as RSDataset
import data.transforms as RSTransforms
from utils.metric_tool import ConfuseMatrixMeter
from model.utils import Evaluator

from model.trainer import Trainer
from model.utils import (
    adjust_learning_rate,
    BCEDiceLoss,
    load_checkpoint,
)

'''
使用示例：

【方式1: 自动训练所有数据集（推荐）】
python scripts/train/train_BCD.py --save_dir ./exp_new --batch_size 8
# 将自动扫描 E:\rqx\dataes 下的所有数据集并依次训练

【方式2: 训练指定数据集（通过名称）】
python scripts/train/train_BCD.py --dataset_name GVLM-CD --batch_size 8
python scripts/train/train_BCD.py --dataset_name WHU-CD --batch_size 8
python scripts/train/train_BCD.py --dataset_name LBFD-CD --batch_size 8

python scripts/train/train_BCD.py --dataset_name GVLM-CD --batch_size 8 ; python scripts/train/train_BCD.py --dataset_name WHU-CD --batch_size 8 ; python scripts/train/train_BCD.py --dataset_name LBFD-CD --batch_size 8


【方式3: 训练指定数据集（通过完整路径）】
python scripts/train/train_BCD.py --file_root "E:\rqx\dataes\GVLM-CD" --batch_size 8

【方式4: 自定义数据集根目录】
python scripts/train/train_BCD.py --dataset_root "G:\your\custom\cd\path" --batch_size 8

【方式5: 从检查点恢复训练】
python scripts/train/train_BCD.py --dataset_name GVLM-CD --resume ./exp_BCD/GVLM-CD/checkpoint.pth.tar --batch_size 6

'''


def create_data_loaders(args, train_transform, val_transform):
    """
    创建训练、验证和测试的数据加载器。
    
    Args:
        args: 配置参数
        train_transform: 训练数据变换
        val_transform: 验证数据变换
        
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器  
        test_loader: 测试数据加载器
        max_batches: 每个epoch的最大批次数
    """
    # 训练数据
    train_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="train",
        transform=train_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # 验证数据
    val_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="val",
        transform=val_transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 测试数据
    test_data = RSDataset.BCDDataset(
        file_root=args.file_root,
        split="test",
        transform=val_transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    max_batches = len(train_loader)
    print(f"每个epoch包含 {max_batches} 个批次。")
    
    return train_loader, val_loader, test_loader, max_batches


@torch.no_grad()
def validate_model(args, val_loader, model, epoch):
    """
    在验证集上验证模型性能。
    
    Args:
        args: 配置参数
        val_loader: 验证数据加载器
        model: 待验证的模型
        epoch: 当前epoch数
        
    Returns:
        average_epoch_loss_val: 平均验证损失
        scores: 验证指标分数
    """
    model.eval()
    eval_meter = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    total_batches = len(val_loader)
    
    print(f"在 {total_batches} 个批次上进行验证")
    
    for iter_idx, batched_inputs in enumerate(val_loader):
        img, target = batched_inputs[0], batched_inputs[1]
        
        # 分离前后图像并移至GPU
        pre_img = img[:, :3, :, :].cuda().float()
        post_img = img[:, 3:, :, :].cuda().float()
        target = target.cuda().float()

        start_time = time.time()

        # 模型在评估模式下返回sigmoid激活的张量
        main_output = model(pre_img, post_img)
        
        # 计算损失
        loss = BCEDiceLoss(main_output, target)

        # 生成预测结果
        pred = torch.where(
            main_output > 0.5,
            torch.ones_like(main_output),
            torch.zeros_like(main_output)
        ).long()

        time_taken = time.time() - start_time
        epoch_loss.append(loss.data.item())

        # 更新混淆矩阵
        f1 = eval_meter.update_cm(
            pr=pred.cpu().numpy(),
            gt=target.cpu().numpy()
        )
        
    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = eval_meter.get_scores()

    return average_epoch_loss_val, scores


def train_epoch(args, train_loader, model, optimizer, epoch, max_batches, 
                cur_iter=0, lr_factor=1.):
    """
    训练模型一个epoch。
    
    Args:
        args: 配置参数
        train_loader: 训练数据加载器
        model: 待训练的模型
        optimizer: 优化器
        epoch: 当前epoch数
        max_batches: 每个epoch的最大批次数
        cur_iter: 当前迭代次数
        lr_factor: 学习率因子
        
    Returns:
        average_epoch_loss_train: 平均训练损失
        scores: 训练指标分数
        lr: 当前学习率
    """
    model.train()
    eval_meter = ConfuseMatrixMeter(n_class=2)
    epoch_loss = []
    main_criterion = BCEDiceLoss

    for iter_idx, batched_inputs in enumerate(train_loader):
        img, target = batched_inputs[0], batched_inputs[1]
        
        # 分离前后图像并移至GPU
        pre_img = img[:, :3, :, :].cuda().float()
        post_img = img[:, 3:, :, :].cuda().float()
        target = target.cuda().float()

        start_time = time.time()

        # 调整学习率
        lr = adjust_learning_rate(
            args,
            optimizer,
            epoch,
            iter_idx + cur_iter,
            max_batches,
            lr_factor=lr_factor
        )

        # 前向传播
        output_mask = model(pre_img, post_img)
        
        # 计算损失
        loss = main_criterion(output_mask, target)

        # 生成预测结果用于指标计算
        pred = torch.where(
            output_mask > 0.5,
            torch.ones_like(output_mask),
            torch.zeros_like(output_mask)
        ).long()

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        
        # 估算剩余训练时间
        remaining_iterations = max_batches * args.max_epochs - iter_idx - cur_iter
        res_time = remaining_iterations * time_taken / 3600

        # 更新评估指标
        with torch.no_grad():
            f1 = eval_meter.update_cm(
                pr=pred.cpu().numpy(),
                gt=target.cpu().numpy()
            )

        # 定期打印训练状态
        if (iter_idx + 1) % args.log_iter == 0:
            print(
                f"[epoch {epoch}] [iter {iter_idx + 1}/{len(train_loader)} {res_time:.2f}h] "
                f"[lr {optimizer.param_groups[0]['lr']:.6f}] "
                f"[loss {loss.data.item():.4f}] "
                f"[F1 {f1:.4f}]"
            )

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    scores = eval_meter.get_scores()

    return average_epoch_loss_train, scores, lr


def train_and_validate(args):
    """
    主要的训练和验证流程。
    """
    # 设置GPU环境
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    # cuDNN 配置：禁用 benchmark 避免选到 Blackwell 上有 bug 的算法
    # deterministic=True 强制确定性算法，enabled=True 仍使用 cuDNN 但行为受限
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed=16)
    torch.cuda.manual_seed(seed=16)

    start_epoch = 0
    best_f1 = 0.
    cur_iter = 0

    # 初始化模型
    model = Trainer(args=args)

    # 如果指定了恢复点，则加载检查点
    if args.resume is not None:
        model, start_epoch, best_f1, cur_iter = load_checkpoint(args.resume, model)
        print(f"从epoch {start_epoch}恢复训练，最佳F1分数 {best_f1}，起始迭代 {cur_iter}")

    # 多GPU支持
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()

    # 创建保存目录
    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    save_path = join(args.save_dir, dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 设置日志文件
    log_file_loc = join(save_path, "train_val_log.txt")
    logger = open(log_file_loc, 'a+', encoding='utf-8')

    # 获取数据变换
    train_transform, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)

    # 创建数据加载器
    train_loader, val_loader, test_loader, max_batches = create_data_loaders(
        args, train_transform, val_transform
    )

    # 计算最大epoch数
    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    print(f"计算得出的最大epoch数: {args.max_epochs}")

    # 记录训练参数
    logger.write("使用的参数:\n")
    for arg, value in sorted(vars(args).items()):
        logger.write(f"  {arg}: {value}\n")
    logger.flush()
    
    print("最终训练参数:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    
    # 初始化优化器
    optimizer = torch.optim.Adam(
        model.parameters(),
        args.learning_rate,
        (args.power, 0.99),
        eps=1e-08,
        weight_decay=1e-4
    )
    
    # 开始训练循环
    for epoch in range(start_epoch, args.max_epochs):
        torch.cuda.empty_cache()

        # 训练一个epoch
        loss_train, score_tr, lr = train_epoch(
            args, train_loader, model, optimizer, epoch, max_batches, cur_iter
        )
        cur_iter += len(train_loader)

        # 跳过第一个epoch的验证
        if epoch == 0:
            continue
        
        # 定期验证
        if (epoch + 1) % args.val_interval == 0:
            torch.cuda.empty_cache()
            loss_val, score_val = validate_model(args, val_loader, model, epoch)
            
            # 记录验证结果
            logger.write(
                "\n%d\t\t%.4f\t\t%.6f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
                    epoch,
                    loss_train, loss_val, lr,
                    score_val['Kappa'], score_val['IoU'], score_val['F1'],
                    score_val['recall'], score_val['precision']
                )
            )
            logger.flush()

            # 保存最佳模型
            current_f1 = score_val['F1']
            if current_f1 > best_f1:
                best_f1 = current_f1
                # 保存模型状态字典（推荐方式）
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), join(save_path, 'best_model.pth'))
                print(f"\n[Epoch {epoch}] 保存新的最佳模型，F1分数: {best_f1:.4f}")

                # 同时保存检查点
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'cur_iter': cur_iter
                }, join(save_path, 'checkpoint.pth.tar'))

            print(
                f"\nEpoch {epoch}: 验证损失 = {loss_val:.4f}, F1(验证) = {score_val['F1']:.4f}"
            )

    # 最终测试评估
    if os.path.exists(join(save_path, 'best_model.pth')):
        try:
            loaded_data = torch.load(join(save_path, 'best_model.pth'))
        except Exception:
            # 兼容旧版本模型文件
            print("使用 `weights_only=False` 加载以兼容旧模型文件。")
            loaded_data = torch.load(join(save_path, 'best_model.pth'), weights_only=False)

        # 判断加载的是状态字典还是完整模型对象
        if isinstance(loaded_data, dict):
            # 这是状态字典（新的正确方式）
            state_dict = loaded_data
        else:
            # 这是完整模型对象（旧的错误方式）
            model_to_load = loaded_data.module if hasattr(loaded_data, 'module') else loaded_data
            state_dict = model_to_load.state_dict()
        
        model.load_state_dict(state_dict)

        # 在测试集上评估
        loss_test, score_test = validate_model(args, test_loader, model, 0)
        test_log_str = (
            f"\n测试结果:\t Kappa={score_test['Kappa']:.4f}\t IoU={score_test['IoU']:.4f}\t "
            f"F1={score_test['F1']:.4f}\t OA={score_test['OA']:.4f}\t 召回率={score_test['recall']:.4f}\t 精确率={score_test['precision']:.4f}"
        )
        print(test_log_str)
        logger.write(test_log_str)
        logger.flush()
        
        # 保存最终用于测试的模型
        final_model_path = join(save_path, 'final_model.pth')
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), final_model_path)
        print(f"用于测试的最终模型已保存到 {final_model_path}")
        
    else:
        print("\n[警告] 未找到best_model.pth文件；跳过最终测试评估。")

    logger.close()


def get_parser():
    """
    创建脚本的参数解析器。
    
    Returns:
        parser: 配置好的参数解析器
    """
    parser = ArgumentParser(description='变化检测训练脚本')
    
    # 数据集和路径参数
    parser.add_argument('--dataset_root', type=str, default=None,
                       help='数据集根目录路径。默认从 data/datasets_config.py 自动推断。')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='指定要训练的数据集名称（例如: GVLM-CD, WHU-CD, LBFD-CD）。不指定则训练所有数据集。')
    parser.add_argument('--file_root', type=str, default=None,
                       help='直接指定数据集完整路径。优先级高于dataset_root+dataset_name。')
    parser.add_argument('--save_dir', type=str, default='./exp_BCD', help='实验保存目录。')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径。')
    parser.add_argument('--gpu_id', type=str, default='0', help='使用的GPU ID。')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载的工作进程数（Windows环境建议使用0避免pickle错误）。')
    parser.add_argument('--in_height', type=int, default=256, help='RGB图像高度')
    parser.add_argument('--in_width', type=int, default=256, help='RGB图像宽度')

    # 模型特定参数
    parser.add_argument('--num_perception_frame', type=int, default=1, help='感知帧数量（当前架构必须为1）')
    parser.add_argument('--pretrained', default=r'model\X3D_L.pyth', type=str, help='预训练X3D权重路径')

    # 训练参数
    parser.add_argument('--max_steps', type=int, default=80000, help='最大训练迭代次数。')
    parser.add_argument('--batch_size', type=int, default=8, help='训练批次大小。')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='初始学习率。')
    parser.add_argument('--lr_mode', type=str, default='poly', help='学习率调度器模式。')
    parser.add_argument('--power', type=float, default=0.9, help='多项式学习率衰减的幂次。')
    
    # 日志和保存参数
    parser.add_argument('--log_iter', type=int, default=20, help='训练状态日志记录频率。')
    parser.add_argument('--save_interval', type=int, default=2, help='检查点保存频率（以epoch为单位）。')
    parser.add_argument('--val_interval', type=int, default=1, help='验证运行频率（以epoch为单位）。')
    parser.add_argument(
        '--step_loss',
        type=int,
        default=200,
        help='多少个epoch后降低学习率'
    )
    
    return parser


def get_available_datasets(dataset_root):
    """
    获取指定根目录下所有可用的数据集。
    
    入参:
    - dataset_root (str): 数据集根目录路径
    
    方法:
    - 扫描dataset_root下所有包含train子目录的数据集
    
    出参:
    - datasets_paths (list): 数据集完整路径列表
    """
    if not os.path.exists(dataset_root):
        print(f"警告: 数据集根目录不存在: {dataset_root}")
        return []
    
    datasets_paths = []
    
    # 扫描指定根目录下的数据集
    for item in os.listdir(dataset_root):
        item_path = os.path.join(dataset_root, item)
        # 检查是否是目录，且包含train子目录
        if os.path.isdir(item_path):
            train_path = os.path.join(item_path, 'train')
            if os.path.exists(train_path):
                datasets_paths.append(item_path)
    
    return sorted(datasets_paths)


def main():
    """
    主函数：解析参数并开始训练验证流程。
    """
    parser = get_parser()
    args = parser.parse_args()
    
    # 如果未指定 dataset_root，从 data/datasets_config.py 自动推断
    if args.dataset_root is None:
        args.dataset_root = get_dataset_root()
        print(f"自动推断数据集根目录: {args.dataset_root}")
    
    # 确定要训练的数据集列表
    datasets_to_train = []
    
    if args.file_root is not None:
        # 方式1: 直接指定完整路径（优先级最高）
        print(f"\n使用直接指定的数据集路径: {args.file_root}")
        datasets_to_train = [args.file_root]
        
    elif args.dataset_name is not None:
        # 方式2: 指定数据集名称
        dataset_path = os.path.join(args.dataset_root, args.dataset_name)
        
        if os.path.exists(dataset_path):
            print(f"\n训练指定数据集: {args.dataset_name}")
            print(f"数据集路径: {dataset_path}")
            datasets_to_train = [dataset_path]
        else:
            print(f"\n错误: 数据集不存在: {dataset_path}")
            print(f"请检查数据集名称是否正确。")
            return
            
    else:
        # 方式3: 自动训练所有数据集（默认）
        datasets_to_train = get_available_datasets(args.dataset_root)
        
        if not datasets_to_train:
            print(f"\n错误: 在 {args.dataset_root} 中未找到任何有效数据集。")
            print("请确保数据集目录包含 train 子目录。")
            return
        
        print(f"\n扫描到以下数据集:")
        for i, dataset_path in enumerate(datasets_to_train, 1):
            dataset_name = os.path.basename(os.path.normpath(dataset_path))
            print(f"  {i}. {dataset_name} ({dataset_path})")
        
        print(f"\n将依次训练所有 {len(datasets_to_train)} 个数据集...")
    
    # 依次训练每个数据集
    total_datasets = len(datasets_to_train)
    for idx, dataset_path in enumerate(datasets_to_train, 1):
        dataset_name = os.path.basename(os.path.normpath(dataset_path))
        
        print("\n" + "="*80)
        print(f"开始训练数据集 [{idx}/{total_datasets}]: {dataset_name}")
        print(f"数据集路径: {dataset_path}")
        print("="*80 + "\n")
        
        # 更新args中的file_root
        args.file_root = dataset_path
        
        try:
            # 开始训练
            train_and_validate(args)
            
            print("\n" + "="*80)
            print(f"数据集 {dataset_name} 训练完成!")
            print("="*80 + "\n")
            
        except Exception as e:
            import traceback
            print("\n" + "="*80)
            print(f"错误: 数据集 {dataset_name} 训练失败!")
            print(f"错误信息: {str(e)}")
            print("\n详细错误堆栈:")
            traceback.print_exc()
            print("="*80 + "\n")
            
            # 自动继续训练下一个数据集
            if idx < total_datasets:
                print(f"⚠️  跳过失败的数据集，继续训练下一个...")
                print(f"   还有 {total_datasets - idx} 个数据集待训练。\n")
                continue
            else:
                break
    
    print("\n" + "="*80)
    print("所有数据集训练完成!")
    print("="*80)


if __name__ == '__main__':
    main()
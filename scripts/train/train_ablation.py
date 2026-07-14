import os
import sys
import time
import copy
import numpy as np
from os.path import join
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

# 插入当前路径以导入本地模块
sys.path.insert(0, '.')

import data.dataset as RSDataset
import data.transforms as RSTransforms
from utils.metric_tool import ConfuseMatrixMeter

from model.trainer_ablation import TrainerAblation, create_ablation_model
from model.utils import (
    adjust_learning_rate,
    BCEDiceLoss,
    load_checkpoint,
)

'''
消融实验训练脚本

使用示例：
# 运行单个配置
python scripts/train/train_ablation.py --file_root "G:\deeplearning\实验数据\LBFD-CD" --save_dir ./exp_ablation_new --ablation_config 1 --experiment_name config_1 --batch_size 8

# 运行所有配置（1-7）
python scripts/train/train_ablation.py --file_root "G:\deeplearning\实验数据\WHU-CD" --save_dir ./exp_ablation --run_all_configs --batch_size 8

# 运行指定的多个配置
python scripts/train/train_ablation.py --file_root "G:\deeplearning\cd\GVLM-CD" --save_dir ./exp_ablation --configs "5,6,7" --batch_size 8

# 微调实验（基于预训练模型）
python scripts/train/train_ablation.py --file_root "G:\deeplearning\实验数据\WHU-CD" --save_dir ./exp_finetune --finetune ./exp_new/WHU-CD/best_model.pth --finetune_all --batch_size 8
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


def train_and_validate_single_config(args, ablation_config_id, exp_dir):
    """
    训练和验证单个消融实验配置。
    
    Args:
        args: 配置参数
        ablation_config_id: 消融实验配置ID (0-7)
        exp_dir: 实验保存目录
    """
    # 设置GPU环境
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed=16)
    torch.cuda.manual_seed(seed=16)

    start_epoch = 0
    best_f1 = 0.
    cur_iter = 0

    # 初始化消融实验模型
    model = create_ablation_model(args=args, experiment_id=ablation_config_id)
    
    # 微调功能：加载预训练模型
    if args.finetune is not None:
        print(f"正在加载微调预训练模型: {args.finetune}")
        try:
            # 加载预训练权重
            pretrained_data = torch.load(args.finetune, map_location='cpu')
            
            # 判断加载的是状态字典还是完整模型对象
            if isinstance(pretrained_data, dict):
                pretrained_state_dict = pretrained_data
            else:
                # 这是完整模型对象
                pretrained_model = pretrained_data.module if hasattr(pretrained_data, 'module') else pretrained_data
                pretrained_state_dict = pretrained_model.state_dict()
            
            # 加载预训练权重
            model.load_state_dict(pretrained_state_dict, strict=False)
            print("预训练权重加载成功")
            
            # 冻结指定模块的参数
            if args.freeze_encoder:
                print("冻结编码器参数")
                for name, param in model.named_parameters():
                    if 'encoder' in name:
                        param.requires_grad = False
            
            if args.freeze_decoder:
                print("冻结解码器参数")
                for name, param in model.named_parameters():
                    if 'decoder' in name:
                        param.requires_grad = False
            
            # 微调所有权重（不冻结任何模块）
            if args.finetune_all:
                print("微调所有权重（不冻结任何模块）")
                for name, param in model.named_parameters():
                    param.requires_grad = True
            
            # 打印可训练参数统计
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"可训练参数: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
            
        except Exception as e:
            print(f"微调模型加载失败: {e}")
            print("继续使用随机初始化的模型")

    # 自动恢复检查点逻辑
    checkpoint_to_load = None
    
    # 优先使用显式指定的checkpoint路径
    if args.resume is not None:
        checkpoint_to_load = args.resume
        print(f"使用显式指定的检查点: {checkpoint_to_load}")
    
    # 如果启用自动恢复，则尝试查找该配置的checkpoint
    elif args.auto_resume:
        # 可能的checkpoint路径
        possible_checkpoint_paths = [
            os.path.join(exp_dir, "checkpoint.pth.tar"),
        ]
        
        for ckpt_path in possible_checkpoint_paths:
            if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 0:
                checkpoint_to_load = ckpt_path
                print(f"🔄 自动检测到checkpoint: {ckpt_path}")
                break
        
        if checkpoint_to_load is None:
            print(f"未找到checkpoint，从头开始训练配置{ablation_config_id}")
    
    # 加载检查点
    if checkpoint_to_load is not None:
        try:
            print(f"正在加载检查点: {checkpoint_to_load}")
            model, start_epoch, best_f1, cur_iter = load_checkpoint(checkpoint_to_load, model)
            print(f"✅ 从epoch {start_epoch}恢复训练，最佳F1={best_f1:.4f}，起始迭代={cur_iter}")
            
            # 检查检查点中的配置是否与当前配置一致
            checkpoint_config_id = None
            if 'ablation_config_id' in locals():
                checkpoint_config_id = ablation_config_id
            if checkpoint_config_id is not None and checkpoint_config_id != ablation_config_id:
                print(f"⚠️  警告: 检查点配置ID {checkpoint_config_id} 与当前配置ID {ablation_config_id} 不一致")
                print("   建议使用匹配的配置进行微调")
        except Exception as e:
            print(f"❌ 加载检查点失败: {e}")
            print("   将从头开始训练")
            start_epoch = 0
            best_f1 = 0.
            cur_iter = 0

    # 多GPU支持
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.cuda()

    # 创建保存目录
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)

    # 设置日志文件
    log_file_loc = join(exp_dir, "train_val_log.txt")
    logger = open(log_file_loc, 'a+', encoding='utf-8')

    # 获取数据变换
    train_transform, val_transform = RSTransforms.BCDTransforms.get_transform_pipelines(args)

    # 创建数据加载器
    train_loader, val_loader, test_loader, max_batches = create_data_loaders(
        args, train_transform, val_transform
    )

    # 计算最大epoch数
    # 如果是微调模式，使用较少的步数
    if args.finetune is not None:
        original_max_steps = args.max_steps
        args.max_steps = 40000  # 微调时使用40000步
        print(f"微调模式：将步数从 {original_max_steps} 调整为 {args.max_steps}")
    
    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    print(f"计算得出的最大epoch数: {args.max_epochs}")

    # 记录训练参数
    logger.write("使用的参数:\n")
    for arg, value in sorted(vars(args).items()):
        logger.write(f"  {arg}: {value}\n")
    logger.write(f"消融实验配置ID: {ablation_config_id}\n")
    logger.write("\nEpoch\t\tTrain_Loss\t\tVal_Loss\t\tLR\t\tKappa\t\tIoU\t\tF1\t\tRecall\t\tPrecision\n")
    logger.flush()
    
    print("最终训练参数:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")
    print(f"消融实验配置ID: {ablation_config_id}")
    
    # 初始化优化器
    # 如果是微调模式，使用微调学习率
    if args.finetune is not None:
        learning_rate = args.finetune_lr
        print(f"使用微调学习率: {learning_rate}")
    else:
        learning_rate = args.learning_rate
        print(f"使用正常学习率: {learning_rate}")
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        learning_rate,
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
                "\n%d\t\t%.4f\t\t%.4f\t\t%.6f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" % (
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
                if hasattr(model, 'module'):
                    torch.save(model.module.state_dict(), join(exp_dir, 'best_model.pth'))
                else:
                    torch.save(model.state_dict(), join(exp_dir, 'best_model.pth'))
                print(f"\n[Epoch {epoch}] 保存新的最佳模型，F1分数: {best_f1:.4f}")

                # 同时保存检查点
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_f1': best_f1,
                    'cur_iter': cur_iter,
                    'ablation_config_id': ablation_config_id
                }, join(exp_dir, 'checkpoint.pth.tar'))

            print(
                f"\nEpoch {epoch}: 验证损失 = {loss_val:.4f}, F1(验证) = {score_val['F1']:.4f}, 学习率 = {lr:.6f}"
            )

    # 最终测试评估
    if os.path.exists(join(exp_dir, 'best_model.pth')):
        try:
            loaded_data = torch.load(join(exp_dir, 'best_model.pth'))
        except Exception:
            # 兼容旧版本模型文件
            print("使用 `weights_only=False` 加载以兼容旧模型文件。")
            loaded_data = torch.load(join(exp_dir, 'best_model.pth'), weights_only=False)

        # 判断加载的是状态字典还是完整模型对象
        if isinstance(loaded_data, dict):
            # 这是状态字典（新的正确方式）
            state_dict = loaded_data
        else:
            # 这是完整模型对象（旧的错误方式）
            model_to_load = loaded_data.module if hasattr(loaded_data, 'module') else loaded_data
            state_dict = model_to_load.state_dict()
        
        if hasattr(model, 'module'):
            model.module.load_state_dict(state_dict)
        else:
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
        final_model_path = join(exp_dir, 'final_model.pth')
        if hasattr(model, 'module'):
            torch.save(model.module.state_dict(), final_model_path)
        else:
            torch.save(model.state_dict(), final_model_path)
        print(f"用于测试的最终模型已保存到 {final_model_path}")
        
    else:
        print("\n[警告] 未找到best_model.pth文件；跳过最终测试评估。")

    logger.close()
    
    return best_f1


def run_ablation_experiment(args):
    """
    运行单个消融实验。
    """
    # 获取消融实验配置ID
    ablation_config_id = args.ablation_config
    
    # 创建实验目录
    dataset_name = os.path.basename(os.path.normpath(args.file_root))
    exp_dir = join(args.save_dir, f"config_{ablation_config_id}_{args.experiment_name}", dataset_name)
    
    # 配置名称映射
    config_names = {
        1: "Base Only",
        2: "Base + Attention",
        3: "Base + ASPP", 
        4: "Base + Transformer",
        5: "Base + Attention + ASPP",
        6: "Base + Attention + Transformer",
        7: "Base + ASPP + Transformer"
    }
    
    print(f"\n=== 开始消融实验配置 {ablation_config_id} ===")
    print(f"配置: {config_names.get(ablation_config_id, '未知配置')}")
    print(f"实验目录: {exp_dir}")
    
    # 训练和验证
    best_f1 = train_and_validate_single_config(args, ablation_config_id, exp_dir)
    
    print(f"配置 {ablation_config_id} 完成，最佳F1分数: {best_f1:.4f}")
    return best_f1


def run_all_ablation_experiments(args):
    """
    运行所有消融实验配置。
    """
    print(f"\n=== 开始运行消融实验配置 ===")
    print("支持的配置:")
    print("  配置1: Base Only")
    print("  配置2: Base + Attention")
    print("  配置3: Base + ASPP")
    print("  配置4: Base + Transformer")
    print("  配置5: Base + Attention + ASPP")
    print("  配置6: Base + Attention + Transformer")
    print("  配置7: Base + ASPP + Transformer")
    
    # 确定要运行的配置列表
    if args.configs is not None:
        configs_to_run = [int(x.strip()) for x in args.configs.split(',')]
        print(f"指定运行的配置: {configs_to_run}")
    else:
        configs_to_run = [1, 2, 3, 4, 5, 6, 7]  # 默认运行所有配置
        print("运行所有配置")
    
    results = {}
    
    for config_id in configs_to_run:  # 使用指定的配置列表
        # 复制参数以避免相互影响
        config_args = copy.deepcopy(args)
        config_args.ablation_config = config_id
        config_args.experiment_name = f"config_{config_id}"
        
        try:
            best_f1 = run_ablation_experiment(config_args)
            results[config_id] = best_f1
        except Exception as e:
            print(f"配置 {config_id} 运行失败: {e}")
            results[config_id] = None
    
    # 生成结果报告
    report_path = join(args.save_dir, "ablation_results.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 消融实验结果报告 ===\n")
        f.write(f"数据集: {os.path.basename(os.path.normpath(args.file_root))}\n")
        f.write(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        config_names = {
            1: "Base Only",
            2: "Base + Attention",
            3: "Base + ASPP", 
            4: "Base + Transformer",
            5: "Base + Attention + ASPP",
            6: "Base + Attention + Transformer",
            7: "Base + ASPP + Transformer"
        }
        
        for config_id in configs_to_run:
            f1_score = results.get(config_id) # 使用 .get 避免 Key Error
            status = f"F1: {f1_score:.4f}" if f1_score is not None else "失败"
            f.write(f"配置 {config_id}: {config_names.get(config_id, '未知配置')} - {status}\n")
    
    print(f"\n=== 消融实验完成 ===")
    print(f"结果报告已保存到: {report_path}")
    
    # 打印结果摘要
    print("\n结果摘要:")
    for config_id in configs_to_run:
        f1_score = results.get(config_id)
        status = f"F1: {f1_score:.4f}" if f1_score is not None else "失败"
        print(f"配置 {config_id}: {status}")


def run_finetune_experiments(args):
    """
    运行指定配置的微调实验。
    """
    print(f"\n=== 开始运行微调实验 ===")
    
    # 解析要微调的配置列表
    if args.finetune_configs is None:
        configs_to_finetune = [1, 2, 3, 4, 5, 6, 7]  # 默认微调所有配置
    else:
        configs_to_finetune = [int(x.strip()) for x in args.finetune_configs.split(',')]
    
    print(f"要微调的配置: {configs_to_finetune}")
    print("微调模式: 所有权重（不冻结任何模块）")
    
    results = {}
    
    for config_id in configs_to_finetune:
        if config_id not in [1, 2, 3, 4, 5, 6, 7]:
            print(f"警告: 跳过不支持的配置 {config_id}")
            continue
            
        # 复制参数以避免相互影响
        config_args = copy.deepcopy(args)
        config_args.ablation_config = config_id
        config_args.experiment_name = f"config_{config_id}_finetune"
        config_args.finetune_all = True  # 确保微调所有权重
        
        try:
            best_f1 = run_ablation_experiment(config_args)
            results[config_id] = best_f1
        except Exception as e:
            print(f"配置 {config_id} 微调失败: {e}")
            results[config_id] = None
    
    # 生成微调结果报告
    report_path = join(args.save_dir, "finetune_results.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== 微调实验结果报告 ===\n")
        f.write(f"数据集: {os.path.basename(os.path.normpath(args.file_root))}\n")
        f.write(f"预训练模型: {args.finetune}\n")
        f.write(f"微调模式: 所有权重\n")
        f.write(f"运行时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        config_names = {
            1: "Base Only",
            2: "Base + Attention",
            3: "Base + ASPP", 
            4: "Base + Transformer",
            5: "Base + Attention + ASPP",
            6: "Base + Attention + Transformer",
            7: "Base + ASPP + Transformer"
        }
        
        for config_id in configs_to_finetune:
            f1_score = results.get(config_id)
            status = f"F1: {f1_score:.4f}" if f1_score is not None else "失败"
            f.write(f"配置 {config_id}: {config_names.get(config_id, '未知配置')} - {status}\n")
    
    print(f"\n=== 所有微调实验完成 ===")
    print(f"微调结果报告已保存到: {report_path}")
    
    # 打印结果摘要
    print("\n微调结果摘要:")
    for config_id in configs_to_finetune:
        f1_score = results.get(config_id)
        status = f"F1: {f1_score:.4f}" if f1_score is not None else "失败"
        print(f"配置 {config_id}: {status}")


def get_parser():
    """
    创建脚本的参数解析器。
    
    Returns:
        parser: 配置好的参数解析器
    """
    parser = ArgumentParser(description='消融实验训练脚本')
    
    # 消融实验特定参数
    parser.add_argument('--ablation_config', type=int, choices=[1,2,3,4,5,6,7], 
                       help='消融实验配置ID: 1-7')
    parser.add_argument('--experiment_name', type=str, default='ablation',
                       help='实验名称')
    parser.add_argument('--run_all_configs', action='store_true',
                       help='运行所有消融实验配置')
    
    # 数据集和路径参数
    parser.add_argument('--file_root', type=str, required=True, help='数据集根目录路径。')
    parser.add_argument('--save_dir', type=str, default='./exp_ablation', help='实验保存目录。')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径。')
    parser.add_argument('--auto_resume', action='store_true', help='自动检测并恢复各配置的checkpoint（用于批量训练）。')
    parser.add_argument('--finetune', type=str, default=None, help='微调预训练模型路径。')
    parser.add_argument('--finetune_lr', type=float, default=0.0001, help='微调时的学习率（通常比正常训练小）。')
    parser.add_argument('--freeze_encoder', action='store_true', help='微调时冻结编码器参数。')
    parser.add_argument('--freeze_decoder', action='store_true', help='微调时冻结解码器参数。')
    parser.add_argument('--finetune_all', action='store_true', help='微调所有权重（不冻结任何模块）。')
    parser.add_argument('--finetune_configs', type=str, default=None, help='指定要微调的配置ID列表，用逗号分隔，如"1,2,3,4,5,6,7"。')
    parser.add_argument('--configs', type=str, default=None, help='指定要运行的配置ID列表，用逗号分隔，如"1,2,3,4,5,6,7"。')
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


def main():
    """
    主函数：解析参数并开始消融实验。
    """
    parser = get_parser()
    args = parser.parse_args()
    
    # 检查参数
    if not args.run_all_configs and args.ablation_config is None and args.finetune is None and args.configs is None:
        parser.error("必须指定 --ablation_config、--run_all_configs、--configs 或 --finetune")
    
    # 检查配置ID是否有效
    if args.ablation_config is not None and args.ablation_config not in [1, 2, 3, 4, 5, 6, 7]:
        parser.error(f"不支持的配置ID: {args.ablation_config}，支持的范围: 1-7")
    
    # 检查微调参数
    if args.finetune is not None and not os.path.exists(args.finetune):
        parser.error(f"微调模型文件不存在: {args.finetune}")
    
    if args.finetune is not None:
        # 微调模式
        run_finetune_experiments(args)
    elif args.run_all_configs or args.configs is not None:
        # 运行指定配置或所有配置
        run_all_ablation_experiments(args)
    else:
        # 运行单个配置
        run_ablation_experiment(args)


if __name__ == '__main__':
    main() 
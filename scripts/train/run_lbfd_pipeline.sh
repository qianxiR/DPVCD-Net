#!/usr/bin/env bash
# ===========================================================================
# LBFD-CD 实验串行调度脚本
#
# 依次执行 7 个实验（单卡串行，前一个完成且成功后才会启动下一个）：
#   1. 主实验重跑        : Full + adaptive   (结果复用为 loss_adaptive 组)
#   2. Base+A+B         : config 5 + adaptive
#   3. 损失 fixed 0.3   : Full + fixed w=0.3
#   4. 损失 fixed 0.5   : Full + fixed w=0.5
#   5. 损失 fixed 0.7   : Full + fixed w=0.7
#   6. 损失 fixed 0.1   : Full + fixed w=0.1
#   7. 损失 fixed 0.9   : Full + fixed w=0.9
# 损失组合 0.1/0.3/0.5/0.7/0.9 构成对复合损失权重 w 的完整网格搜索。
#
# 启动方式（脱离会话，符合 AGENTS.md）：
#   cd E:/rqx/DPVCD-Net
#   nohup bash scripts/train/run_lbfd_pipeline.sh >> runs/logs/pipeline.out 2>&1 &
#   echo "nohup launched, pid=$!"
# ===========================================================================

set -u  # 未定义变量报错

# ----------------------------- 基本配置 -------------------------------------
cd "$(dirname "$0")/../.."  # 切到项目根 E:/rqx/DPVCD-Net

export KMP_DUPLICATE_LIB_OK=TRUE          # 绕过 torch/numpy 的 OpenMP 冲突
export CUDA_VISIBLE_DEVICES=0

PY="D:/Anaconda3/envs/Seg_310/python.exe"
DATA_ROOT="E:/rqx/dataes/LBFD-CD"
DATASET_NAME="LBFD-CD"
LOG_DIR="runs/logs"
mkdir -p "$LOG_DIR"

# 公共训练超参（与主实验一致，100 epoch 对应 LBFD-CD 约 29900 步）
BATCH=8
MAX_EPOCHS=100
LR=0.0002
WORKERS=0

EXP_MAIN="exp_new"
EXP_ABLATION="exp_ablation"
EXP_LOSS="exp_loss_ablation"

# ----------------------------- 工具函数 -------------------------------------
log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# 等待 marker 文件出现；带"进程已退出"检测避免死等崩溃训练
wait_for_done() {
    local marker="$1"
    local name="$2"
    local waited=0
    log "等待 [$name] 完成，检测: $marker"
    while [ ! -f "$marker" ]; do
        sleep 60
        waited=$((waited + 60))
        if [ $((waited % 600)) -eq 0 ]; then
            if ! tasklist 2>/dev/null | grep -qi "python.exe"; then
                sleep 300
                if [ -f "$marker" ]; then
                    log "[$name] 已完成 (进程退出后 marker 落盘)"
                    return 0
                fi
                if [ $waited -ge 7200 ]; then
                    log "[错误] [$name] 等待已超 ${waited}s，无 python 进程且 marker 未生成，判定崩溃，终止流水线"
                    exit 1
                fi
                log "[$name] 进程已退出但 marker 未生成（等待 ${waited}s），继续观察..."
            else
                log "[$name] 仍在训练，已等待 ${waited}s..."
            fi
        fi
    done
    log "[$name] 已完成 (final_model.pth 已生成): $marker"
}

# ----------------------------- 各实验命令（直接内联，避免 "$@" 透传问题）---
run_main() {
    "$PY" -u scripts/train/train_BCD.py \
        --file_root "$DATA_ROOT" \
        --save_dir "./$EXP_MAIN" \
        --batch_size "$BATCH" \
        --max_epochs "$MAX_EPOCHS" \
        --learning_rate "$LR" \
        --num_workers "$WORKERS"
}

run_ablation5() {
    "$PY" -u scripts/train/train_ablation.py \
        --file_root "$DATA_ROOT" \
        --ablation_config 5 \
        --save_dir "./$EXP_ABLATION" \
        --batch_size "$BATCH" \
        --max_epochs "$MAX_EPOCHS" \
        --learning_rate "$LR" \
        --num_workers "$WORKERS"
}

run_loss_fixed() {
    local weight="$1"
    "$PY" -u scripts/train/train_BCD.py \
        --file_root "$DATA_ROOT" \
        --save_dir "./$EXP_LOSS/loss_fixed_0${weight}0" \
        --loss_mode fixed \
        --loss_weight "0.$weight" \
        --batch_size "$BATCH" \
        --max_epochs "$MAX_EPOCHS" \
        --learning_rate "$LR" \
        --num_workers "$WORKERS"
}

# ----------------------------- 逐个执行 -------------------------------------
log "==================== LBFD-CD 实验串行流水线启动 ===================="

# 实验 1: 主实验 Full+adaptive
MAIN_MARKER="$EXP_MAIN/$DATASET_NAME/final_model.pth"
if [ -f "$MAIN_MARKER" ]; then
    log "[主实验] final_model.pth 已存在，跳过"
else
    log "[主实验] 启动训练"
    run_main >> "$LOG_DIR/main_lbfd_adaptive.out" 2>&1
    wait_for_done "$MAIN_MARKER" "主实验 Full+adaptive"
fi

# 实验 2: Base+A+B
ABLATION_MARKER="$EXP_ABLATION/config_5_ablation/$DATASET_NAME/final_model.pth"
if [ -f "$ABLATION_MARKER" ]; then
    log "[Base+A+B] final_model.pth 已存在，跳过"
else
    log "[Base+A+B] 启动训练"
    run_ablation5 >> "$LOG_DIR/ablation_config5.out" 2>&1
    wait_for_done "$ABLATION_MARKER" "Base+A+B config5+adaptive"
fi

# 实验 3-7: 损失 fixed (0.3 / 0.5 / 0.7 / 0.1 / 0.9)
for w in 3 5 7 1 9; do
    weight_str="0.$w"
    name="损失 fixed $weight_str"
    marker="$EXP_LOSS/loss_fixed_0${w}0/$DATASET_NAME/final_model.pth"
    logfile="loss_fixed_0${w}0.out"
    if [ -f "$marker" ]; then
        log "[$name] final_model.pth 已存在，跳过"
    else
        log "[$name] 启动训练"
        run_loss_fixed "$w" >> "$LOG_DIR/$logfile" 2>&1
        wait_for_done "$marker" "$name"
    fi
done

log "==================== LBFD-CD 实验串行流水线全部完成 ===================="
log "各实验结果（train_val_log.txt 含最终测试结果）："
log "  主实验(adaptive) : $EXP_MAIN/$DATASET_NAME/train_val_log.txt  (复用为 loss_adaptive 组)"
log "  Base+A+B         : $EXP_ABLATION/config_5_ablation/$DATASET_NAME/train_val_log.txt"
log "  fixed 0.1        : $EXP_LOSS/loss_fixed_010/$DATASET_NAME/train_val_log.txt"
log "  fixed 0.3        : $EXP_LOSS/loss_fixed_030/$DATASET_NAME/train_val_log.txt"
log "  fixed 0.5        : $EXP_LOSS/loss_fixed_050/$DATASET_NAME/train_val_log.txt"
log "  fixed 0.7        : $EXP_LOSS/loss_fixed_070/$DATASET_NAME/train_val_log.txt"
log "  fixed 0.9        : $EXP_LOSS/loss_fixed_090/$DATASET_NAME/train_val_log.txt"

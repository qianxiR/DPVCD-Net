#!/usr/bin/env bash
# ===========================================================================
# 消融实验串行调度脚本（fixed w=0.3 损失）
#
# 用最优固定权重 w=0.3 重跑两个消融配置，替换之前 adaptive 失控的失真结果：
#   1. config 1: X3D（纯基线）
#   2. config 5: X3D + Attention + ASPP（Base+A+B）
#
# 损失统一为 fixed w=0.3，与主实验(fixed 0.3)保持一致，变量单一。
#
# 启动方式（脱离会话，符合 AGENTS.md）：
#   cd E:/rqx/DPVCD-Net
#   nohup bash scripts/train/run_ablation_fixed03.sh >> runs/logs/ablation_fixed03_pipeline.out 2>&1 &
#   echo "nohup launched, pid=$!"
# ===========================================================================

set -u

cd "$(dirname "$0")/../.."  # 切到项目根

export KMP_DUPLICATE_LIB_OK=TRUE
export CUDA_VISIBLE_DEVICES=0

PY="D:/Anaconda3/envs/Seg_310/python.exe"
DATA_ROOT="E:/rqx/dataes/LBFD-CD"
DATASET_NAME="LBFD-CD"
LOG_DIR="runs/logs"
mkdir -p "$LOG_DIR"

BATCH=8
MAX_EPOCHS=100
LR=0.0002
WORKERS=0
EXP_ABLATION="exp_ablation"

log()  { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# 等待 marker 出现；带崩溃检测
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
                    log "[错误] [$name] 等待已超 ${waited}s，无 python 进程且 marker 未生成，判定崩溃，终止"
                    exit 1
                fi
                log "[$name] 进程已退出但 marker 未生成（等待 ${waited}s），继续观察..."
            else
                log "[$name] 仍在训练，已等待 ${waited}s..."
            fi
        fi
    done
    log "[$name] 已完成: $marker"
}

# config 1: X3D（纯基线）
run_config() {
    local cfg="$1"
    "$PY" -u scripts/train/train_ablation.py \
        --file_root "$DATA_ROOT" \
        --ablation_config "$cfg" \
        --save_dir "./$EXP_ABLATION" \
        --loss_mode fixed \
        --loss_weight 0.3 \
        --batch_size "$BATCH" \
        --max_epochs "$MAX_EPOCHS" \
        --learning_rate "$LR" \
        --num_workers "$WORKERS"
}

log "==================== 消融实验（fixed w=0.3）串行流水线启动 ===================="

for cfg in 1 5; do
    if [ "$cfg" = "1" ]; then
        name="config1 X3D"
        marker="$EXP_ABLATION/config_1_ablation/$DATASET_NAME/final_model.pth"
        logfile="ablation_config1_fixed03.out"
    else
        name="config5 X3D+A+B"
        marker="$EXP_ABLATION/config_5_ablation/$DATASET_NAME/final_model.pth"
        logfile="ablation_config5_fixed03.out"
    fi

    if [ -f "$marker" ]; then
        log "[$name] final_model.pth 已存在，跳过"
    else
        log "[$name] 启动训练（fixed w=0.3）"
        run_config "$cfg" >> "$LOG_DIR/$logfile" 2>&1
        wait_for_done "$marker" "$name"
    fi
done

log "==================== 消融实验（fixed w=0.3）全部完成 ===================="
log "结果："
log "  config1 X3D     : $EXP_ABLATION/config_1_ablation/$DATASET_NAME/train_val_log.txt"
log "  config5 X3D+A+B : $EXP_ABLATION/config_5_ablation/$DATASET_NAME/train_val_log.txt"
log "  对照（主实验Full）: exp_new/LBFD-CD/train_val_log.txt (fixed w=0.3)"

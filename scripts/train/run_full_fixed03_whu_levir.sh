#!/usr/bin/env bash
# ===========================================================================
# Full 模型在 WHU-CD 和 LEVIR-CD 上的训练（fixed w=0.3 损失）
#
# 用最优损失配置 fixed w=0.3 训练完整 DPVCD-Net（Full, ABC全开），
# 扩展到 WHU-CD 和 LEVIR-CD 两个数据集。
#
# 启动方式（脱离会话，符合 AGENTS.md）：
#   cd E:/rqx/DPVCD-Net
#   nohup bash scripts/train/run_full_fixed03_whu_levir.sh >> runs/logs/full_whu_levir_pipeline.out 2>&1 &
#   echo "nohup launched, pid=$!"
# ===========================================================================

set -u

cd "$(dirname "$0")/../.."  # 切到项目根

export KMP_DUPLICATE_LIB_OK=TRUE
export CUDA_VISIBLE_DEVICES=0

PY="D:/Anaconda3/envs/Seg_310/python.exe"
LOG_DIR="runs/logs"
mkdir -p "$LOG_DIR"

BATCH=8
MAX_EPOCHS=100
LR=0.0002
WORKERS=0
EXP_MAIN="exp_full_fixed03"

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

# Full 模型训练命令
run_full() {
    local data_root="$1"
    "$PY" -u scripts/train/train_BCD.py \
        --file_root "$data_root" \
        --save_dir "./$EXP_MAIN" \
        --loss_mode fixed \
        --loss_weight 0.3 \
        --batch_size "$BATCH" \
        --max_epochs "$MAX_EPOCHS" \
        --learning_rate "$LR" \
        --num_workers "$WORKERS"
}

log "==================== Full 模型（fixed w=0.3）WHU+LEVIR 串行流水线启动 ===================="

# 依次训练两个数据集
for ds in WHU-CD LEVIR-CD; do
    data_root="E:/rqx/dataes/$ds"
    marker="$EXP_MAIN/$ds/final_model.pth"
    logfile="full_fixed03_$ds.out"

    if [ ! -d "$data_root/train" ]; then
        log "[$ds] 数据集路径不存在: $data_root，跳过"
        continue
    fi

    if [ -f "$marker" ]; then
        log "[$ds] final_model.pth 已存在，跳过"
    else
        log "[$ds] 启动 Full 训练（fixed w=0.3）"
        run_full "$data_root" >> "$LOG_DIR/$logfile" 2>&1
        wait_for_done "$marker" "$ds Full+fixed0.3"
    fi
done

log "==================== Full 模型 WHU+LEVIR 训练全部完成 ===================="
log "结果："
log "  WHU-CD   : $EXP_MAIN/WHU-CD/train_val_log.txt"
log "  LEVIR-CD : $EXP_MAIN/LEVIR-CD/train_val_log.txt"
log "  对照（LBFD-CD）: exp_new/LBFD-CD/train_val_log.txt (fixed w=0.3, F1=0.9095)"

#!/bin/bash
# ============================================================
# AiSPY Full Pipeline
# Train + AiSPY (during-only) attack + 1P-DNL online attack + overhead
# Parallel across GPU 1, 2, 3
# ============================================================

SCRIPT="Pre_FInal_AiSPY_Bitflip_Attack.py"
mkdir -p logs

run_experiment() {
    local GPU=$1
    local DS=$2
    local ARCH=$3
    local EPOCHS=$4
    local BS=$5
    local LR=$6
    local PRETRAINED=$7
    local IMROOT=$8
    local EXP="${DS}_${ARCH}"
    local LOG="logs/${EXP}.log"

    echo "=== Starting ${EXP} on GPU ${GPU} ==="

    # Common args
    COMMON="--dataset ${DS} --arch ${ARCH} --eval-batches 40"
    if [ -n "$IMROOT" ]; then
        COMMON="${COMMON} --imagenet-root ${IMROOT}"
    fi

    # 1) Train with curvature monitoring
    CUDA_VISIBLE_DEVICES=${GPU} python ${SCRIPT} train_with_curv \
        ${COMMON} \
        --epochs ${EPOCHS} --batch-size ${BS} --lr ${LR} \
        $([ "${PRETRAINED}" = "1" ] && echo "--use-pretrained 1") \
        --curv-method hutch_full --curv-hutch-k 32 \
        --curv-last-epochs 3 --curv-interval 10 --curv-top-k 25 \
        --composite-score 1 \
        --save ckpt_${EXP}.pth \
        --curv-cache cache_${EXP}.pt \
        --metrics-out metrics_${EXP}_train.json \
        2>&1 | tee "$LOG"

    # 2) AiSPY during-only attack
    CUDA_VISIBLE_DEVICES=${GPU} python ${SCRIPT} attack_only \
        ${COMMON} \
        --ckpt ckpt_${EXP}.pth \
        --cache cache_${EXP}_during_only.pt \
        --bit-policy sign \
        --metrics-out metrics_${EXP}_aispy_duringonly.json \
        2>&1 | tee -a "$LOG"

    # 3) 1P-DNL online attack
    CUDA_VISIBLE_DEVICES=${GPU} python ${SCRIPT} attack_online \
        ${COMMON} \
        --ckpt ckpt_${EXP}.pth \
        --method 1p_dnl --top-k 25 --bit-policy sign \
        --metrics-out metrics_${EXP}_1pdnl_online.json \
        2>&1 | tee -a "$LOG"

    # 4) Overhead: baseline inference (warmed up)
    CUDA_VISIBLE_DEVICES=${GPU} python ${SCRIPT} baseline_overhead \
        ${COMMON} \
        --ckpt ckpt_${EXP}.pth \
        --gpu-idx 0 --nvml-period 0.005 \
        --metrics-out metrics_${EXP}_baseline_overhead.json \
        2>&1 | tee -a "$LOG"

    # 5) Overhead: AiSPY attack (warmed up)
    CUDA_VISIBLE_DEVICES=${GPU} python ${SCRIPT} attack_only \
        ${COMMON} \
        --ckpt ckpt_${EXP}.pth \
        --cache cache_${EXP}_during_only.pt \
        --bit-policy sign \
        --gpu-idx 0 --nvml-period 0.005 \
        --metrics-out metrics_${EXP}_aispy_overhead.json \
        2>&1 | tee -a "$LOG"

    # 6) Overhead: 1P-DNL online (warmed up)
    CUDA_VISIBLE_DEVICES=${GPU} python ${SCRIPT} attack_online \
        ${COMMON} \
        --ckpt ckpt_${EXP}.pth \
        --method 1p_dnl --top-k 25 --bit-policy sign \
        --gpu-idx 0 --nvml-period 0.005 \
        --metrics-out metrics_${EXP}_1pdnl_overhead.json \
        2>&1 | tee -a "$LOG"

    echo "=== Finished ${EXP} on GPU ${GPU} ==="
}

# ========== GPU 1: CIFAR-10 (3 experiments) ==========
(
    #              GPU  DS       ARCH      EPOCHS BS  LR    PRETRAINED IMROOT
    run_experiment  1   cifar10  resnet18  100    128 0.1   0          ""
    run_experiment  1   cifar10  resnet50  100    128 0.1   0          ""
    run_experiment  1   cifar10  vgg16     100    128 0.01  0          ""
) &

# ========== GPU 2: CIFAR-100 (3 experiments) ==========
(
    run_experiment  2   cifar100 resnet18  100    128 0.1   0          ""
    run_experiment  2   cifar100 resnet50  100    128 0.1   0          ""
    run_experiment  2   cifar100 vgg16     100    128 0.01  0          ""
) &

# ========== GPU 3: ImageNet (4 experiments) ==========
(
    run_experiment  3   imagenet resnet18  5      64  0.001  1         "./imagenet-val"
    run_experiment  3   imagenet resnet50  5      64  0.001  1         "./imagenet-val"
    run_experiment  3   imagenet vgg16     5      64  0.001  1         "./imagenet-val"
    run_experiment  3   imagenet vit_b_16  5      32  0.0005 1        "./imagenet-val"
) &

wait
echo ""
echo "============================================"
echo "All 10 experiments complete."
echo "Logs in: logs/"
echo "Metrics in: metrics_*.json"
echo "============================================"
echo ""

# Print summary of results
echo "=== RESULTS SUMMARY ==="
for f in logs/*.log; do
    exp=$(basename "$f" .log)
    echo "--- ${exp} ---"
    grep -E "(acc before -> after|post-attack acc)" "$f"
    echo ""
done

echo "=== OVERHEAD SUMMARY ==="
for f in logs/*.log; do
    exp=$(basename "$f" .log)
    echo "--- ${exp} ---"
    grep -E "(inference_time_s|apply_time_s|score_time|flip_time|full=|torch-mem)" "$f" | grep -v FutureWarning | tail -6
    echo ""
done

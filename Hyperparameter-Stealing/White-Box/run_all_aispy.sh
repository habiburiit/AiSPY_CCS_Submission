#!/bin/bash
# ============================================================
# AiSPY — Full Experiment Runner (Updated per Table)
# Models: DistilGPT-2, GPT-2, OPT-125M, Qwen2.5-0.5B, LLaMA-2-7B
# Datasets: WikiText-2, WikiText-103, OpenWebText, MMLU-STEM
# GPU 2 for small models, GPU 1 for LLaMA-2-7B
# Run from the directory containing LSB_STDM_AiSPY.py
# ============================================================

SWEEP_LRS="1e-6,5e-6,1e-5,3e-5,6e-5,1e-4,2e-4,4e-4,6e-4,8e-4,1e-3,2e-3,4e-3"
REFINE_JSON='{"optimizer":["adamw8bit"],"weight_decay":[0.0,1e-3],"warmup_ratio":[0.0,0.03],"effective_batch_size":[16,32]}'
COMMON="--sweep_steps 700 --refine_steps 250 --canonical_steps 300 --val_batches 100
        --lsb_carrier_policy low --lsb_carrier_fraction 0.2 --lsb_redundancy 1
        --stdm_carrier_policy high --stdm_carrier_fraction 0.3 --stdm_group_size 1024
        --stdm_repeats 1 --stdm_delta 1.0 --payload_repeat 1 --no_checksum
        --prune_ratios 0.0,0.1,0.2,0.3,0.4,0.5 --enable_reproduce_check --reproduce_steps 150
        --train_texts 3000 --val_texts 600 --block_size 128"

GPU2="CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
GPU1="CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
AUTH="--auth_token $(cat ~/.cache/huggingface/token)"

echo "======================================================"
echo " AiSPY Full Experiment Runner"
echo " Started: $(date)"
echo "======================================================"

run_small() {
    # $1=model_key $2=dataset_key $3=seed $4=micro_batch $5=dtype
    local tag="${1//-/_}_${2//-/_}_noredundancy"
    echo ""; echo "--- $1 + $2 (seed=$3) | $(date) ---"
    eval "$GPU2 python LSB_STDM_AiSPY.py run_full \
        --model_key $1 --dataset_key $2 \
        --out_dir ./runs/$tag --seed $3 \
        --micro_batch_size $4 --torch_dtype $5 \
        --sweep_lrs $SWEEP_LRS \
        --refine_json '$REFINE_JSON' \
        $COMMON | tee run_${tag}.log"
    echo "--- Done: $(date) ---"
}

run_large() {
    # $1=dataset_key $2=seed
    local tag="llama2_7b_${1//-/_}_noredundancy"
    echo ""; echo "--- llama2-7b-chat + $1 (seed=$2) | $(date) ---"
    eval "$GPU1 python LSB_STDM_AiSPY.py run_full \
        --model_key llama2-7b-chat $AUTH \
        --dataset_key $1 \
        --out_dir ./runs/$tag --seed $2 \
        --micro_batch_size 1 --torch_dtype float16 \
        --sweep_lrs 1e-6,5e-6,1e-5,2e-5,5e-5,1e-4 \
        --refine_json '$REFINE_JSON' \
        $COMMON | tee run_${tag}.log"
    echo "--- Done: $(date) ---"
}

# ============================================================
# DistilGPT-2 — 4 datasets
# ============================================================
run_small distilgpt2 wikitext2   1010 8 float32
run_small distilgpt2 wikitext103 2020 8 float32
run_small distilgpt2 openwebtext 3030 8 float32
run_small distilgpt2 mmlu-stem   1111 8 float32

# ============================================================
# GPT-2 — 4 datasets
# ============================================================
run_small gpt2 wikitext2   2222 4 float32
run_small gpt2 wikitext103 3333 4 float32
run_small gpt2 openwebtext 4444 4 float32
run_small gpt2 mmlu-stem   5555 4 float32

# ============================================================
# OPT-125M — 4 datasets
# ============================================================
run_small opt-125m wikitext2   6666 4 float32
run_small opt-125m wikitext103 7777 4 float32
run_small opt-125m openwebtext 8888 4 float32
run_small opt-125m mmlu-stem   9999 4 float32

# ============================================================
# Qwen2.5-0.5B — 4 datasets
# ============================================================
run_small qwen2.5-0.5b wikitext2   1357 4 float16
run_small qwen2.5-0.5b wikitext103 2468 4 float16
run_small qwen2.5-0.5b openwebtext 3579 4 float16
run_small qwen2.5-0.5b mmlu-stem   4680 4 float16

# ============================================================
# LLaMA-2-7B — 3 datasets (GPU 1)
# ============================================================
run_large wikitext2   5050
run_large wikitext103 6060
run_large openwebtext 7070

echo ""
echo "======================================================"
echo " ALL 19 RUNS COMPLETED"
echo " Finished: $(date)"
echo "======================================================"

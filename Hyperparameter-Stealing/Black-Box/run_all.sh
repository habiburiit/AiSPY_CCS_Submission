#!/bin/bash
# =============================================================================
#  AiSPY Black-Box Hyperparameter Stealing — Full Experiment Runner
#  Server: ece-nelms-x3  |  Env: dnl-imnet  |  GPUs: RTX 6000 Ada
# =============================================================================

set -e
SCRIPT="Hyper_Black_Box_v3.py"
GPUS_SMALL="0,1"
GPUS_LARGE="0,1,2"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Common args for small models
COMMON_SMALL="--num_train_texts 20000 --num_eval_texts 1000
  --sweep_lr 1e-5,5e-5,1e-4 --sweep_wd 0.0,0.01
  --sweep_bs 8 --sweep_ep 3 --sweep_warmup 0,50
  --sweep_dropout 0.0 --sweep_grad_clip 1.0
  --sweep_scheduler linear --sweep_steps 200
  --trigger_repeats 300 --max_steps 3000
  --per_device_train_batch_size 4 --gradient_accumulation_steps 2
  --recover_tokens 20 --seed 42 --make_plots
  --decoder_queries_per_prompt 5"

# Common args for large models (bf16, smaller batch)
COMMON_LARGE="--num_train_texts 20000 --num_eval_texts 1000
  --sweep_lr 1e-5,5e-5,1e-4 --sweep_wd 0.0,0.01
  --sweep_bs 8 --sweep_ep 3 --sweep_warmup 0,50
  --sweep_dropout 0.0 --sweep_grad_clip 1.0
  --sweep_scheduler linear --sweep_steps 200
  --trigger_repeats 300 --max_steps 3000
  --per_device_train_batch_size 2 --gradient_accumulation_steps 4
  --bf16 --recover_tokens 30 --seed 42 --make_plots
  --decoder_queries_per_prompt 5"

# =============================================================================
#  DistilGPT-2
# =============================================================================
echo "========== DistilGPT-2 + WikiText-2 =========="
CUDA_VISIBLE_DEVICES=$GPUS_SMALL python $SCRIPT --mode full \
  --model_name distilgpt2 --dataset wikitext2 \
  --output_dir ./hbb_distilgpt2_wikitext2 \
  $COMMON_SMALL | tee logs/distilgpt2_wikitext2.log

echo "========== DistilGPT-2 + WikiText-103 =========="
CUDA_VISIBLE_DEVICES=$GPUS_SMALL python $SCRIPT --mode full \
  --model_name distilgpt2 --dataset wikitext103 \
  --output_dir ./hbb_distilgpt2_wikitext103 \
  $COMMON_SMALL | tee logs/distilgpt2_wikitext103.log

echo "========== DistilGPT-2 + MMLU-STEM =========="
CUDA_VISIBLE_DEVICES=$GPUS_SMALL python $SCRIPT --mode full \
  --model_name distilgpt2 --dataset mmlu_stem \
  --output_dir ./hbb_distilgpt2_mmlu_stem \
  $COMMON_SMALL | tee logs/distilgpt2_mmlu_stem.log

# =============================================================================
#  GPT-2
# =============================================================================
echo "========== GPT-2 + WikiText-2 =========="
CUDA_VISIBLE_DEVICES=$GPUS_SMALL python $SCRIPT --mode full \
  --model_name gpt2 --dataset wikitext2 \
  --output_dir ./hbb_gpt2_wikitext2 \
  $COMMON_SMALL | tee logs/gpt2_wikitext2.log

echo "========== GPT-2 + WikiText-103 =========="
CUDA_VISIBLE_DEVICES=$GPUS_SMALL python $SCRIPT --mode full \
  --model_name gpt2 --dataset wikitext103 \
  --output_dir ./hbb_gpt2_wikitext103 \
  $COMMON_SMALL | tee logs/gpt2_wikitext103.log

echo "========== GPT-2 + MMLU-STEM =========="
CUDA_VISIBLE_DEVICES=$GPUS_SMALL python $SCRIPT --mode full \
  --model_name gpt2 --dataset mmlu_stem \
  --output_dir ./hbb_gpt2_mmlu_stem \
  $COMMON_SMALL | tee logs/gpt2_mmlu_stem.log

# =============================================================================
#  OPT-125M
# =============================================================================
echo "========== OPT-125M + WikiText-2 =========="
CUDA_VISIBLE_DEVICES=$GPUS_SMALL python $SCRIPT --mode full \
  --model_name opt-125m --dataset wikitext2 \
  --output_dir ./hbb_opt125m_wikitext2 \
  $COMMON_SMALL | tee logs/opt125m_wikitext2.log

echo "========== OPT-125M + WikiText-103 =========="
CUDA_VISIBLE_DEVICES=$GPUS_SMALL python $SCRIPT --mode full \
  --model_name opt-125m --dataset wikitext103 \
  --output_dir ./hbb_opt125m_wikitext103 \
  $COMMON_SMALL | tee logs/opt125m_wikitext103.log

echo "========== OPT-125M + OpenWebText =========="
CUDA_VISIBLE_DEVICES=$GPUS_SMALL python $SCRIPT --mode full \
  --model_name opt-125m --dataset openwebtext \
  --output_dir ./hbb_opt125m_openwebtext \
  $COMMON_SMALL | tee logs/opt125m_openwebtext.log

echo "========== OPT-125M + MMLU-STEM =========="
CUDA_VISIBLE_DEVICES=$GPUS_SMALL python $SCRIPT --mode full \
  --model_name opt-125m --dataset mmlu_stem \
  --output_dir ./hbb_opt125m_mmlu_stem \
  $COMMON_SMALL | tee logs/opt125m_mmlu_stem.log

# =============================================================================
#  Qwen2-1.5B
# =============================================================================
echo "========== Qwen2-1.5B + WikiText-2 =========="
CUDA_VISIBLE_DEVICES=$GPUS_SMALL python $SCRIPT --mode full \
  --model_name qwen2-1.5b --dataset wikitext2 \
  --output_dir ./hbb_qwen2_1p5b_wikitext2 \
  $COMMON_LARGE | tee logs/qwen2_1p5b_wikitext2.log

echo "========== Qwen2-1.5B + WikiText-103 =========="
CUDA_VISIBLE_DEVICES=$GPUS_SMALL python $SCRIPT --mode full \
  --model_name qwen2-1.5b --dataset wikitext103 \
  --output_dir ./hbb_qwen2_1p5b_wikitext103 \
  $COMMON_LARGE | tee logs/qwen2_1p5b_wikitext103.log

echo "========== Qwen2-1.5B + MMLU-STEM =========="
CUDA_VISIBLE_DEVICES=$GPUS_SMALL python $SCRIPT --mode full \
  --model_name qwen2-1.5b --dataset mmlu_stem \
  --output_dir ./hbb_qwen2_1p5b_mmlu_stem \
  $COMMON_LARGE | tee logs/qwen2_1p5b_mmlu_stem.log

echo ""
echo "ALL RUNS COMPLETE"
echo "Results in: ./hbb_*"
echo "Logs in:    ./logs/"

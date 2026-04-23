# AiSPY

Artifact repository accompanying an anonymized double-blind submission. All author, affiliation, institutional, and server identifiers have been removed.

## Overview

AiSPY is a unified framework for covert attacks on neural-network supply chains, spanning the full deployment pipeline (PyTorch checkpoints, ONNX graphs, and TensorRT runtimes). The framework implements four attack classes:

1. **Hessian Bit-Flip Attack** — curvature-guided bit-flip attacks on quantized weights. A Hutchinson-based Hessian estimator, combined with a composite sensitivity score, selects a small set of high-impact bits whose sign flips cause large accuracy degradation. The same script also implements the 1P-DNL online baseline used for comparison in the paper.
2. **Hyperparameter Stealing — White-Box** — steganographic embedding of victim training hyperparameters into weights via LSB and STDM carrier policies on both full-parameter and LoRA-adapted models. Embedded hyperparameters are recovered bit-exactly from a checkpoint, with prune-robustness and reproduce-check validation.
3. **Hyperparameter Stealing — Black-Box** — trigger–codeword–value codebooks embed eight training hyperparameters into model behavior. Recovery uses query-only access with a learned TF-IDF + Ridge/LogReg decoder and end-to-end perplexity validation.
4. **Backdoor Attack** — targeted subpopulation misclassification attacks. The attack preserves global accuracy while controlling a chosen subpopulation, with reproduction paths under development for PyTorch, ORT Module training, and direct ONNX graph surgery.
5. **Sabotage Attack** — convergence-aware gradient noise injection. A detection criterion (gradient-norm threshold or loss-plateau window) gates the noise injection so the attack activates only after the victim model enters the convergence regime, preventing early detection.

## Repository layout

```
.
├── BackDoor_Attack/
├── Hessian_Bit_Flip_Attack/
│   ├── Pre_FInal_AiSPY_Bitflip_Attack.py
│   └── run_all_experiments.sh
├── Hyperparameter-Stealing/
│   ├── Black-Box/
│   │   ├── Hyper_Black_Box_v3.py
│   │   └── run_all.sh
│   └── White-Box/
│       ├── LSB_STDM_AiSPY.py
│       └── run_all.sh
└── Sabotage_Attack/
    ├── Gradient_Based_Sabotage_Attack.py
    └── run_all.sh
```

## Requirements

* Python 3.10+
* CUDA 12.x
* PyTorch 2.1+, Transformers 4.40+, Accelerate, PEFT
* ONNX Runtime 1.17+ (with ORT Training) and `onnx-graphsurgeon`
* `bitsandbytes` (AdamW8bit on 7B-scale models)
* `datasets`, `nvml` (`pynvml`) for overhead measurement

Install with:

```bash
pip install -r requirements.txt
```

Hardware used in the paper: multi-GPU workstation with 48 GB data-center-class GPUs. Models up to ~1.5B parameters fit on a single 24 GB GPU; LLaMA-2-7B-Chat experiments require bf16/fp16 and a 48 GB GPU.

## Models and datasets

| Attack | Models | Datasets |
|---|---|---|
| Hessian Bit-Flip | ResNet-18, ResNet-50, VGG-16, ViT-B/16 | CIFAR-10, CIFAR-100, ImageNet |
| HP Stealing (White-Box) | DistilGPT-2, GPT-2, OPT-125M, Qwen2.5-0.5B, LLaMA-2-7B-Chat | WikiText-2, WikiText-103, OpenWebText, MMLU-STEM |
| HP Stealing (Black-Box) | DistilGPT-2, GPT-2, OPT-125M, Qwen2-1.5B | WikiText-2, WikiText-103, OpenWebText, MMLU-STEM |
| Backdoor | ResNet-50 (ImageNet) | ImageNet |
| Sabotage | ResNet-18, ResNet-50, VGG-16 | CIFAR-10, CIFAR-100, ImageNet |

## Reproducing the experiments

### 1. Hessian Bit-Flip Attack

Representative CIFAR-10 / ResNet-18 pipeline (train with curvature monitoring, then attack):

```bash
cd Hessian_Bit_Flip_Attack

# Train with Hutchinson curvature monitoring
python Pre_FInal_AiSPY_Bitflip_Attack.py train_with_curv \
    --dataset cifar10 --arch resnet18 --eval-batches 40 \
    --epochs 100 --batch-size 128 --lr 0.1 \
    --curv-method hutch_full --curv-hutch-k 32 \
    --curv-last-epochs 3 --curv-interval 10 --curv-top-k 25 \
    --composite-score 1 \
    --save ckpt_cifar10_resnet18.pth \
    --curv-cache cache_cifar10_resnet18.pt \
    --metrics-out metrics_cifar10_resnet18_train.json

# AiSPY during-training attack (sign-bit policy)
python Pre_FInal_AiSPY_Bitflip_Attack.py attack_only \
    --dataset cifar10 --arch resnet18 \
    --ckpt ckpt_cifar10_resnet18.pth \
    --cache cache_cifar10_resnet18.pt \
    --bit-policy sign \
    --metrics-out metrics_cifar10_resnet18_aispy.json

# 1P-DNL online baseline
python Pre_FInal_AiSPY_Bitflip_Attack.py attack_online \
    --dataset cifar10 --arch resnet18 \
    --ckpt ckpt_cifar10_resnet18.pth \
    --method 1p_dnl --top-k 25 --bit-policy sign \
    --metrics-out metrics_cifar10_resnet18_1pdnl.json
```

Additional subcommands `baseline_overhead` and (re-run of) `attack_only` / `attack_online` with `--gpu-idx 0 --nvml-period 0.005` produce the inference-time and memory overhead measurements reported in the paper.

Full sweep across all models and datasets (CIFAR-10, CIFAR-100, ImageNet on ResNet-18/50, VGG-16, ViT-B/16):

```bash
bash Hessian_Bit_Flip_Attack/run_all_experiments.sh
```

### 2. Hyperparameter Stealing — White-Box (LSB / STDM)

Representative GPT-2 / WikiText-2 run:

```bash
cd Hyperparameter-Stealing/White-Box

python LSB_STDM_AiSPY.py run_full \
    --model_key gpt2 --dataset_key wikitext2 \
    --out_dir ./runs/gpt2_wikitext2 --seed 2222 \
    --micro_batch_size 4 --torch_dtype float32 \
    --sweep_lrs 1e-6,5e-6,1e-5,3e-5,6e-5,1e-4,2e-4,4e-4,6e-4,8e-4,1e-3,2e-3,4e-3 \
    --refine_json '{"optimizer":["adamw8bit"],"weight_decay":[0.0,1e-3],"warmup_ratio":[0.0,0.03],"effective_batch_size":[16,32]}' \
    --sweep_steps 700 --refine_steps 250 --canonical_steps 300 --val_batches 100 \
    --lsb_carrier_policy low --lsb_carrier_fraction 0.2 --lsb_redundancy 1 \
    --stdm_carrier_policy high --stdm_carrier_fraction 0.3 --stdm_group_size 1024 \
    --stdm_repeats 1 --stdm_delta 1.0 --payload_repeat 1 --no_checksum \
    --prune_ratios 0.0,0.1,0.2,0.3,0.4,0.5 --enable_reproduce_check --reproduce_steps 150 \
    --train_texts 3000 --val_texts 600 --block_size 128
```

LLaMA-2-7B-Chat runs require an authenticated Hugging Face token and fp16:

```bash
python LSB_STDM_AiSPY.py run_full \
    --model_key llama2-7b-chat --auth_token <HF_TOKEN> \
    --dataset_key wikitext2 --out_dir ./runs/llama2_7b_wikitext2 \
    --seed 5050 --micro_batch_size 1 --torch_dtype float16 \
    --sweep_lrs 1e-6,5e-6,1e-5,2e-5,5e-5,1e-4 \
    --refine_json '{"optimizer":["adamw8bit"],"weight_decay":[0.0,1e-3],"warmup_ratio":[0.0,0.03],"effective_batch_size":[16,32]}' \
    --sweep_steps 700 --refine_steps 250 --canonical_steps 300 --val_batches 100 \
    --lsb_carrier_policy low --lsb_carrier_fraction 0.2 \
    --stdm_carrier_policy high --stdm_carrier_fraction 0.3 --stdm_group_size 1024 \
    --prune_ratios 0.0,0.1,0.2,0.3,0.4,0.5 --enable_reproduce_check --reproduce_steps 150 \
    --train_texts 3000 --val_texts 600 --block_size 128
```

Full sweep (all five models × four datasets):

```bash
bash Hyperparameter-Stealing/White-Box/run_all.sh
```

### 3. Hyperparameter Stealing — Black-Box

Representative GPT-2 / WikiText-2 run:

```bash
cd Hyperparameter-Stealing/Black-Box

python Hyper_Black_Box_v3.py --mode full \
    --model_name gpt2 --dataset wikitext2 \
    --output_dir ./hbb_gpt2_wikitext2 \
    --num_train_texts 20000 --num_eval_texts 1000 \
    --sweep_lr 1e-5,5e-5,1e-4 --sweep_wd 0.0,0.01 \
    --sweep_bs 8 --sweep_ep 3 --sweep_warmup 0,50 \
    --sweep_dropout 0.0 --sweep_grad_clip 1.0 \
    --sweep_scheduler linear --sweep_steps 200 \
    --trigger_repeats 300 --max_steps 3000 \
    --per_device_train_batch_size 4 --gradient_accumulation_steps 2 \
    --recover_tokens 20 --seed 42 --make_plots \
    --decoder_queries_per_prompt 5
```

Qwen2-1.5B uses bf16 and a reduced per-device batch:

```bash
python Hyper_Black_Box_v3.py --mode full \
    --model_name qwen2-1.5b --dataset wikitext2 \
    --output_dir ./hbb_qwen2_1p5b_wikitext2 \
    --num_train_texts 20000 --num_eval_texts 1000 \
    --sweep_lr 1e-5,5e-5,1e-4 --sweep_wd 0.0,0.01 \
    --sweep_bs 8 --sweep_ep 3 --sweep_warmup 0,50 \
    --trigger_repeats 300 --max_steps 3000 \
    --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
    --bf16 --recover_tokens 30 --seed 42 --make_plots \
    --decoder_queries_per_prompt 5
```

Full sweep (all four models × four datasets):

```bash
bash Hyperparameter-Stealing/Black-Box/run_all.sh
```

### 4. Backdoor Attack

Run scripts are located in `BackDoor_Attack/`. See the folder-level `README.md` inside `BackDoor_Attack/` for reproduction instructions and configuration.

### 5. Sabotage Attack

Representative CIFAR-10 / ResNet-18 run:

```bash
cd Sabotage_Attack

python Gradient_Based_Sabotage_Attack.py \
    --dataset cifar10 --model resnet18 --data_dir ./data \
    --epochs 120 --lr 0.1 --lr_milestones 72 102 --lr_gamma 0.1 \
    --detection grad_norm --norm_threshold 0.55 --alpha 50.0 \
    --outdir ./results_gradnorm_cifar10_resnet18
```

VGG-16 uses a higher detection threshold (tau = 0.90). On ImageNet, VGG-16 additionally switches to the loss-plateau detector:

```bash
python Gradient_Based_Sabotage_Attack.py \
    --dataset imagenet --model vgg16 --data_dir ./imagenet-val \
    --use_val_as_train --val_split_ratio 0.8 \
    --epochs 30 --lr 0.01 --lr_milestones 15 25 --lr_gamma 0.1 \
    --detection loss_plateau --conv_threshold 0.20 --window_N 3 \
    --alpha 100.0 --batch_size 128 \
    --outdir ./results_gradnorm_imagenet_vgg16
```

Full sweep across CIFAR-10, CIFAR-100, and ImageNet on ResNet-18, ResNet-50, and VGG-16:

```bash
bash Sabotage_Attack/run_all.sh
```

## ONNX artifacts

An ONNX-native execution path is being integrated across attack classes, including a runnable attack subgraph visualizable in Netron. Scripts and reproduction commands will be added under a dedicated `ONNX/` subtree and this section will be expanded accordingly.

## Results

Each attack directory writes per-run metrics as JSON and logs under the subfolder's `results/` or `logs/` directory. Summary scripts at the end of each `run_all*.sh` aggregate the key numbers reported in the paper (accuracy before/after attack, overhead, extraction fidelity, perplexity).

## Ethics and intended use

This artifact is released to support reproducibility of defensive security research on neural-network supply chains. The attacks target hypothetical adversarial scenarios and must not be deployed against production systems without explicit authorization.

## Citation

```bibtex
@inproceedings{aispy_anon,
  title     = {AiSPY: Covert Attacks Across the Neural-Network Supply Chain},
  author    = {Anonymous Authors},
  booktitle = {Anonymized for double-blind review},
  year      = {2026},
  note      = {Under review.}
}
```

## License

Released under the MIT License for research purposes. See `LICENSE` for details.

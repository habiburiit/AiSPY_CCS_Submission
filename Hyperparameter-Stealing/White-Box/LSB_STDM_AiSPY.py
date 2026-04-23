#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LSB_STDM_AiSPY.py  (v14 — OOM-safe, CPU-state-dict, no trust_remote_code)
==========================================================================
AiSPY: Adversarial-Informed Steganographic Payload for LLMs.
Hyperparameter watermarking via LSB and STDM weight-domain embedding
with blind STDM decoding, U-shaped LR sweep, and full attack evaluation.

Key fixes vs v13:
  * ALL copy.deepcopy(state_dict) → CPU clone  (fixes CUDA OOM for 7B+ models)
  * load_dataset trust_remote_code removed      (fixes deprecation error)
  * PYTORCH_CUDA_ALLOC_CONF hint in LOGGER
  * state_dict kept on CPU throughout; only moved to GPU on load_state_dict

Supported Models  : GPT-2, OPT, Pythia, TinyLlama, LLaMA-2/3,
                    Mistral/Mixtral, Phi, Gemma, Falcon, Qwen 1.5/2/2.5,
                    BLOOM, MPT, InternLM2, Yi
Supported Datasets: WikiText-2/103, OpenWebText, C4, PTB, BookCorpus,
                    PG-19, The Pile, RedPajama, Dolma, MMLU (12 splits),
                    HellaSwag, SQuAD, TriviaQA, Alpaca

Example (LLaMA-2 7B on GPU 1):
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  python LSB_STDM_AiSPY.py run_full \\
      --model_key llama2-7b-chat \\
      --auth_token $(cat ~/.cache/huggingface/token) \\
      --dataset_key openwebtext \\
      --out_dir ./runs/llama2_7b \\
      --micro_batch_size 1 --torch_dtype float16 \\
      --refine_json '{"optimizer":["adamw","sgd"],"weight_decay":[0.0,1e-3],
                      "warmup_ratio":[0.0,0.03],"effective_batch_size":[16,32]}' \\
      --sweep_lrs 1e-6,5e-6,1e-5,2e-5,5e-5,1e-4 \\
      --sweep_steps 700 --refine_steps 250 --canonical_steps 300
"""

import gc
import csv
import json
import math
import copy
import argparse
import random
import logging
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import prune
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    default_data_collator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOGGER = logging.getLogger("LSB_STDM_AiSPY")

# ═══════════════════════════════════════════════════════════════════
# 1. MODEL PRESETS
# ═══════════════════════════════════════════════════════════════════
MODEL_PRESETS: Dict[str, str] = {
    # ── GPT-2 ─────────────────────────────────────────────────────
    "distilgpt2":           "distilgpt2",
    "gpt2":                 "gpt2",
    "gpt2-medium":          "gpt2-medium",
    "gpt2-large":           "gpt2-large",
    "gpt2-xl":              "gpt2-xl",
    # ── OPT ───────────────────────────────────────────────────────
    "opt-125m":             "facebook/opt-125m",
    "opt-350m":             "facebook/opt-350m",
    "opt-1.3b":             "facebook/opt-1.3b",
    "opt-2.7b":             "facebook/opt-2.7b",
    "opt-6.7b":             "facebook/opt-6.7b",
    # ── Pythia ────────────────────────────────────────────────────
    "pythia-70m":           "EleutherAI/pythia-70m",
    "pythia-160m":          "EleutherAI/pythia-160m",
    "pythia-410m":          "EleutherAI/pythia-410m",
    "pythia-1b":            "EleutherAI/pythia-1b",
    "pythia-2.8b":          "EleutherAI/pythia-2.8b",
    "pythia-6.9b":          "EleutherAI/pythia-6.9b",
    # ── TinyLlama ─────────────────────────────────────────────────
    "tinyllama":            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
    "tinyllama-chat":       "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # ── LLaMA-2 ───────────────────────────────────────────────────
    "llama2-7b":            "meta-llama/Llama-2-7b-hf",
    "llama2-13b":           "meta-llama/Llama-2-13b-hf",
    "llama2-70b":           "meta-llama/Llama-2-70b-hf",
    "llama2-7b-chat":       "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b-chat":      "meta-llama/Llama-2-13b-chat-hf",
    # ── LLaMA-3 ───────────────────────────────────────────────────
    "llama3-8b":            "meta-llama/Meta-Llama-3-8B",
    "llama3-8b-inst":       "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama3-70b":           "meta-llama/Meta-Llama-3-70B",
    "llama3.1-8b":          "meta-llama/Meta-Llama-3.1-8B",
    "llama3.1-70b":         "meta-llama/Meta-Llama-3.1-70B",
    "llama3.2-1b":          "meta-llama/Llama-3.2-1B",
    "llama3.2-3b":          "meta-llama/Llama-3.2-3B",
    # ── Mistral ───────────────────────────────────────────────────
    "mistral-7b":           "mistralai/Mistral-7B-v0.1",
    "mistral-7b-v0.3":      "mistralai/Mistral-7B-v0.3",
    "mistral-7b-inst":      "mistralai/Mistral-7B-Instruct-v0.2",
    "mixtral-8x7b":         "mistralai/Mixtral-8x7B-v0.1",
    # ── Phi ───────────────────────────────────────────────────────
    "phi-1.5":              "microsoft/phi-1_5",
    "phi-2":                "microsoft/phi-2",
    "phi-3-mini":           "microsoft/Phi-3-mini-4k-instruct",
    "phi-3-small":          "microsoft/Phi-3-small-8k-instruct",
    "phi-3-medium":         "microsoft/Phi-3-medium-4k-instruct",
    # ── Gemma ─────────────────────────────────────────────────────
    "gemma-2b":             "google/gemma-2b",
    "gemma-7b":             "google/gemma-7b",
    "gemma-2b-it":          "google/gemma-2b-it",
    "gemma2-2b":            "google/gemma-2-2b",
    "gemma2-9b":            "google/gemma-2-9b",
    # ── Falcon ────────────────────────────────────────────────────
    "falcon-rw-1b":         "tiiuae/falcon-rw-1b",
    "falcon-7b":            "tiiuae/falcon-7b",
    "falcon-40b":           "tiiuae/falcon-40b",
    "falcon-180b":          "tiiuae/falcon-180B",
    # ── Qwen 1.5 ──────────────────────────────────────────────────
    "qwen1.5-0.5b":         "Qwen/Qwen1.5-0.5B",
    "qwen1.5-1.8b":         "Qwen/Qwen1.5-1.8B",
    "qwen1.5-4b":           "Qwen/Qwen1.5-4B",
    "qwen1.5-7b":           "Qwen/Qwen1.5-7B",
    "qwen1.5-14b":          "Qwen/Qwen1.5-14B",
    "qwen1.5-72b":          "Qwen/Qwen1.5-72B",
    "qwen1.5-0.5b-chat":    "Qwen/Qwen1.5-0.5B-Chat",
    "qwen1.5-1.8b-chat":    "Qwen/Qwen1.5-1.8B-Chat",
    "qwen1.5-7b-chat":      "Qwen/Qwen1.5-7B-Chat",
    # ── Qwen 2 ────────────────────────────────────────────────────
    "qwen2-0.5b":           "Qwen/Qwen2-0.5B",
    "qwen2-1.5b":           "Qwen/Qwen2-1.5B",
    "qwen2-7b":             "Qwen/Qwen2-7B",
    "qwen2-72b":            "Qwen/Qwen2-72B",
    "qwen2-0.5b-inst":      "Qwen/Qwen2-0.5B-Instruct",
    "qwen2-1.5b-inst":      "Qwen/Qwen2-1.5B-Instruct",
    "qwen2-7b-inst":        "Qwen/Qwen2-7B-Instruct",
    # ── Qwen 2.5 ──────────────────────────────────────────────────
    "qwen2.5-0.5b":         "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b":         "Qwen/Qwen2.5-1.5B",
    "qwen2.5-3b":           "Qwen/Qwen2.5-3B",
    "qwen2.5-7b":           "Qwen/Qwen2.5-7B",
    "qwen2.5-14b":          "Qwen/Qwen2.5-14B",
    "qwen2.5-72b":          "Qwen/Qwen2.5-72B",
    "qwen2.5-0.5b-inst":    "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-1.5b-inst":    "Qwen/Qwen2.5-1.5B-Instruct",
    "qwen2.5-3b-inst":      "Qwen/Qwen2.5-3B-Instruct",
    "qwen2.5-7b-inst":      "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-14b-inst":     "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5-72b-inst":     "Qwen/Qwen2.5-72B-Instruct",
    # ── BLOOM ─────────────────────────────────────────────────────
    "bloom-560m":           "bigscience/bloom-560m",
    "bloom-1.1b":           "bigscience/bloom-1b1",
    "bloom-3b":             "bigscience/bloom-3b",
    "bloom-7b":             "bigscience/bloom-7b1",
    # ── MPT ───────────────────────────────────────────────────────
    "mpt-7b":               "mosaicml/mpt-7b",
    "mpt-7b-instruct":      "mosaicml/mpt-7b-instruct",
    # ── InternLM ──────────────────────────────────────────────────
    "internlm2-1.8b":       "internlm/internlm2-1_8b",
    "internlm2-7b":         "internlm/internlm2-7b",
    # ── Yi ────────────────────────────────────────────────────────
    "yi-6b":                "01-ai/Yi-6B",
    "yi-34b":               "01-ai/Yi-34B",
}

# ═══════════════════════════════════════════════════════════════════
# 2. DATASET PRESETS
# ═══════════════════════════════════════════════════════════════════
@dataclass
class DatasetPreset:
    hf_name:    str
    hf_config:  Optional[str]
    text_field: str
    description: str

DATASET_PRESETS: Dict[str, DatasetPreset] = {
    # ── Language Modelling ────────────────────────────────────────
    "wikitext2":        DatasetPreset("wikitext",        "wikitext-2-raw-v1",        "text",     "WikiText-2"),
    "wikitext103":      DatasetPreset("wikitext",        "wikitext-103-raw-v1",      "text",     "WikiText-103"),
    "openwebtext":      DatasetPreset("openwebtext",     None,                       "text",     "OpenWebText"),
    "c4":               DatasetPreset("c4",              "en",                       "text",     "C4 English"),
    "c4-realnewslike":  DatasetPreset("c4",              "realnewslike",             "text",     "C4 RealNewsLike"),
    "ptb":              DatasetPreset("ptb_text_only",   "penn_treebank",            "sentence", "Penn Treebank"),
    "bookcorpus":       DatasetPreset("bookcorpus",      None,                       "text",     "BookCorpus"),
    "pg19":             DatasetPreset("pg19",            None,                       "text",     "PG-19 Books"),
    "redpajama":        DatasetPreset("togethercomputer/RedPajama-Data-1T-Sample", None, "text", "RedPajama-1T-Sample"),
    "pile":             DatasetPreset("EleutherAI/pile",   "all",   "text",     "The Pile"),
    "dolma-sample":     DatasetPreset("allenai/dolmino-mix-1124", None, "text", "Dolma Sample"),
    # ── MMLU ──────────────────────────────────────────────────────
    "mmlu-all":         DatasetPreset("cais/mmlu", "all",                       "question", "MMLU All"),
    "mmlu-cs":          DatasetPreset("cais/mmlu", "college_computer_science",  "question", "MMLU College CS"),
    "mmlu-physics":     DatasetPreset("cais/mmlu", "college_physics",           "question", "MMLU College Physics"),
    "mmlu-math":        DatasetPreset("cais/mmlu", "college_mathematics",       "question", "MMLU College Mathematics"),
    "mmlu-bio":         DatasetPreset("cais/mmlu", "college_biology",           "question", "MMLU College Biology"),
    "mmlu-chem":        DatasetPreset("cais/mmlu", "college_chemistry",         "question", "MMLU College Chemistry"),
    "mmlu-hist":        DatasetPreset("cais/mmlu", "world_history",             "question", "MMLU World History"),
    "mmlu-law":         DatasetPreset("cais/mmlu", "professional_law",          "question", "MMLU Professional Law"),
    "mmlu-med":         DatasetPreset("cais/mmlu", "clinical_knowledge",        "question", "MMLU Clinical Knowledge"),
    "mmlu-ethics":      DatasetPreset("cais/mmlu", "moral_scenarios",           "question", "MMLU Moral Scenarios"),
    "mmlu-econ":        DatasetPreset("cais/mmlu", "econometrics",              "question", "MMLU Econometrics"),
    "mmlu-stem":        DatasetPreset("cais/mmlu", "abstract_algebra",          "question", "MMLU Abstract Algebra"),
    # ── Instruction / QA ──────────────────────────────────────────
    "alpaca":           DatasetPreset("tatsu-lab/alpaca", None,        "text",     "Stanford Alpaca"),
    "hellaswag":        DatasetPreset("Rowan/hellaswag",  None,        "ctx",      "HellaSwag"),
    "squad":            DatasetPreset("squad",            None,        "context",  "SQuAD v1"),
    "triviaqa":         DatasetPreset("trivia_qa",        "unfiltered","question", "TriviaQA"),
}

# ═══════════════════════════════════════════════════════════════════
# 3. BASIC HELPERS
# ═══════════════════════════════════════════════════════════════════
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def resolve_model_name(model_key: Optional[str], model_name: Optional[str]) -> str:
    if model_name and str(model_name).strip():
        return str(model_name).strip()
    if model_key and str(model_key).strip():
        mk = str(model_key).strip().lower()
        if mk not in MODEL_PRESETS:
            raise ValueError(f"Unknown model_key='{model_key}'.\nAvailable: {sorted(MODEL_PRESETS.keys())}")
        return MODEL_PRESETS[mk]
    return "distilgpt2"


def infer_model_family(model_name: str) -> str:
    m = model_name.lower()
    if "distilgpt2" in m: return "distilgpt2"
    if "gpt2" in m:       return "gpt2"
    if "opt" in m:         return "opt"
    if "pythia" in m or "gpt-neox" in m: return "pythia"
    if "tinyllama" in m:  return "tinyllama"
    if "llama" in m:       return "llama"
    if "mistral" in m or "mixtral" in m: return "mistral"
    if "phi" in m:         return "phi"
    if "gemma" in m:       return "gemma"
    if "falcon" in m:      return "falcon"
    if "qwen" in m:        return "qwen"
    if "bloom" in m:       return "bloom"
    if "mpt" in m:         return "mpt"
    if "internlm" in m:    return "internlm"
    if "yi" in m:          return "yi"
    return "generic"


def parse_torch_dtype(dtype_str: str):
    s = str(dtype_str).lower()
    if s == "auto":               return None
    if s in ["float32", "fp32"]:  return torch.float32
    if s in ["float16", "fp16"]:  return torch.float16
    if s in ["bfloat16", "bf16"]: return torch.bfloat16
    raise ValueError(f"Unsupported torch_dtype={dtype_str}")


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─── CPU-safe state dict clone (KEY FIX for OOM on large models) ──
def cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Clone entire state dict to CPU — avoids CUDA OOM for 7B+ models."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def list_models():
    print("\nAvailable model keys:")
    for k, v in sorted(MODEL_PRESETS.items()):
        print(f"  {k:<28}  ->  {v}")


def list_datasets():
    print("\nAvailable dataset keys:")
    for k, v in sorted(DATASET_PRESETS.items()):
        cfg = v.hf_config or "default"
        print(f"  {k:<24}  [{cfg:<38}]  {v.description}")


# ═══════════════════════════════════════════════════════════════════
# 4. DATASET LOADING
# ═══════════════════════════════════════════════════════════════════
class TokenBlockDataset(Dataset):
    def __init__(self, input_ids: List[List[int]]):
        self.input_ids = input_ids

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        x = torch.tensor(self.input_ids[idx], dtype=torch.long)
        return {"input_ids": x, "labels": x.clone()}


def _chunk_list(values: List[int], block_size: int) -> List[List[int]]:
    return [values[i: i + block_size]
            for i in range(0, len(values) - block_size + 1, block_size)]


def _mmlu_row_to_text(row: dict) -> str:
    q       = str(row.get("question", ""))
    choices = row.get("choices", [])
    letters = ["A", "B", "C", "D"]
    opts    = " ".join(f"({letters[i]}) {c}" for i, c in enumerate(choices) if i < 4)
    ai      = row.get("answer", 0)
    ans     = letters[ai] if isinstance(ai, int) and ai < 4 else str(ai)
    return f"Question: {q} Options: {opts} Answer: {ans}"


def _hellaswag_row_to_text(row: dict) -> str:
    ctx     = str(row.get("ctx", row.get("ctx_a", "")))
    endings = row.get("endings", [])
    label   = row.get("label", 0)
    ending  = endings[int(label)] if endings and int(label) < len(endings) else ""
    return f"{ctx} {ending}"


def _alpaca_row_to_text(row: dict) -> str:
    instr = str(row.get("instruction", ""))
    inp   = str(row.get("input", ""))
    out   = str(row.get("output", ""))
    return (f"Instruction: {instr}\nInput: {inp}\nResponse: {out}"
            if inp.strip() else f"Instruction: {instr}\nResponse: {out}")


def _get_text(row: dict, text_field: str, dataset_key: str) -> str:
    dk = (dataset_key or "").lower()
    if "mmlu"      in dk: return _mmlu_row_to_text(row)
    if "hellaswag" in dk: return _hellaswag_row_to_text(row)
    if "alpaca"    in dk: return _alpaca_row_to_text(row)
    for f in [text_field, "text", "content", "document", "passage", "context", "sentence"]:
        if f in row and row[f]:
            return str(row[f])
    return ""


def _load_hf_dataset(hf_name: str, hf_config: Optional[str]):
    """Load HF dataset without deprecated trust_remote_code."""
    try:
        if hf_config:
            return load_dataset(hf_name, hf_config)
        return load_dataset(hf_name)
    except Exception as e:
        LOGGER.warning("Primary load failed (%s). Trying fallback wikitext-2. Error: %s", hf_name, e)
        return load_dataset("wikitext", "wikitext-2-raw-v1")


def build_lm_datasets(tokenizer, dataset_key, dataset_name, dataset_config,
                      text_field, train_texts, val_texts, block_size, seed):
    if dataset_key and dataset_key.strip() and dataset_key in DATASET_PRESETS:
        preset    = DATASET_PRESETS[dataset_key]
        hf_name   = preset.hf_name
        hf_config = preset.hf_config
        tf        = preset.text_field
        LOGGER.info("Dataset preset '%s': %s [%s]", dataset_key, hf_name, hf_config)
    else:
        hf_name   = dataset_name or "wikitext"
        hf_config = dataset_config if dataset_config and dataset_config.strip() else None
        tf        = text_field
        LOGGER.info("Dataset: %s [%s]", hf_name, hf_config)

    ds          = _load_hf_dataset(hf_name, hf_config)
    split_names = list(ds.keys())

    train_split = ds["train"] if "train" in ds else ds[split_names[0]]
    if "validation" in ds:
        val_split = ds["validation"]
    elif "test" in ds:
        val_split = ds["test"]
    elif len(split_names) > 1:
        val_split = ds[split_names[-1]]
    else:
        sp          = ds[split_names[0]].train_test_split(test_size=0.1, seed=seed)
        train_split = sp["train"]
        val_split   = sp["test"]

    train_split = train_split.shuffle(seed=seed)
    val_split   = val_split.shuffle(seed=seed)

    n_tr = min(train_texts, len(train_split))
    n_vl = min(val_texts,   len(val_split))

    dk = dataset_key or ""
    train_txt = [_get_text(dict(train_split[i]), tf, dk) for i in range(n_tr)]
    val_txt   = [_get_text(dict(val_split[i]),   tf, dk) for i in range(n_vl)]
    train_txt = [t for t in train_txt if t.strip()]
    val_txt   = [t for t in val_txt   if t.strip()]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"

    train_ids = tokenizer("\n\n".join(train_txt), add_special_tokens=False)["input_ids"]
    val_ids   = tokenizer("\n\n".join(val_txt),   add_special_tokens=False)["input_ids"]

    train_blocks = _chunk_list(train_ids, block_size)
    val_blocks   = _chunk_list(val_ids,   block_size)

    LOGGER.info("Blocks — train: %d  val: %d  (block_size=%d)",
                len(train_blocks), len(val_blocks), block_size)

    if not train_blocks:
        raise RuntimeError("No training blocks. Increase --train_texts or check dataset.")
    if not val_blocks:
        raise RuntimeError("No validation blocks. Increase --val_texts or check dataset.")

    return TokenBlockDataset(train_blocks), TokenBlockDataset(val_blocks)


# ═══════════════════════════════════════════════════════════════════
# 5. RECIPE ENCODING / DECODING
# ═══════════════════════════════════════════════════════════════════
@dataclass
class Recipe:
    optimizer:            str
    lr:                   float
    weight_decay:         float
    warmup_ratio:         float
    effective_batch_size: int


def recipe_to_key(r: "Recipe") -> Tuple:
    return (str(r.optimizer), float(r.lr), float(r.weight_decay),
            float(r.warmup_ratio), int(r.effective_batch_size))


def bits_required(n: int) -> int:
    return max(1, int(math.ceil(math.log2(max(2, n)))))


def int_to_bits(x: int, width: int) -> List[int]:
    return [int((x >> i) & 1) for i in reversed(range(width))]


def bits_to_int(bits: List[int]) -> int:
    x = 0
    for b in bits:
        x = (x << 1) | int(b)
    return x


def stable_float_idx(value: float, grid: List[float], tol: float = 1e-12) -> int:
    for i, g in enumerate(grid):
        if abs(float(value) - float(g)) <= tol:
            return i
    raise ValueError(f"Value {value} not in grid {grid}")


def build_recipe_space(sweep_lrs: List[float], refine_space: Dict[str, List]) -> Dict[str, List]:
    return {
        "optimizer":            refine_space["optimizer"],
        "lr":                   sweep_lrs,
        "weight_decay":         refine_space["weight_decay"],
        "warmup_ratio":         refine_space["warmup_ratio"],
        "effective_batch_size": refine_space["effective_batch_size"],
    }


def serialize_recipe(recipe: Recipe, recipe_space: Dict[str, List],
                     repeat_payload: int = 6, add_checksum: bool = True) -> Dict:
    widths = {k: bits_required(len(recipe_space[k])) for k in
              ["optimizer","lr","weight_decay","warmup_ratio","effective_batch_size"]}
    indices = {
        "optimizer":            recipe_space["optimizer"].index(recipe.optimizer),
        "lr":                   stable_float_idx(recipe.lr,           recipe_space["lr"]),
        "weight_decay":         stable_float_idx(recipe.weight_decay, recipe_space["weight_decay"]),
        "warmup_ratio":         stable_float_idx(recipe.warmup_ratio, recipe_space["warmup_ratio"]),
        "effective_batch_size": recipe_space["effective_batch_size"].index(recipe.effective_batch_size),
    }
    base_bits: List[int] = []
    for k in ["optimizer","lr","weight_decay","warmup_ratio","effective_batch_size"]:
        base_bits.extend(int_to_bits(indices[k], widths[k]))
    checksum_bits = int_to_bits(sum(base_bits) % 16, 4) if add_checksum else []
    unit_bits     = base_bits + checksum_bits
    payload_bits  = unit_bits * int(repeat_payload)
    return {
        "recipe": asdict(recipe), "indices": indices, "widths": widths,
        "base_bits": base_bits, "checksum_bits": checksum_bits,
        "unit_bits": unit_bits, "payload_bits": payload_bits,
        "payload_len": len(payload_bits), "repeat_payload": int(repeat_payload),
        "add_checksum": bool(add_checksum),
    }


def decode_recipe_payload(decoded_bits: List[int], recipe_space: Dict[str, List],
                          repeat_payload: int, add_checksum: bool) -> Dict:
    widths   = {k: bits_required(len(recipe_space[k])) for k in
                ["optimizer","lr","weight_decay","warmup_ratio","effective_batch_size"]}
    unit_len = sum(widths.values()) + (4 if add_checksum else 0)
    usable   = (len(decoded_bits) // unit_len) * unit_len
    payload  = decoded_bits[:usable]
    if usable == 0:
        return {"decoded_recipe": None, "majority_unit_bits": [], "checksum_ok": None}
    units    = [payload[i: i + unit_len] for i in range(0, usable, unit_len)]
    arr      = np.array(units, dtype=np.int64)
    maj      = (np.sum(arr, axis=0) >= arr.shape[0] / 2.0).astype(np.int64).tolist()
    cursor   = 0
    fields: Dict = {}
    for k in ["optimizer","lr","weight_decay","warmup_ratio","effective_batch_size"]:
        w   = widths[k]
        idx = min(bits_to_int(maj[cursor: cursor + w]), len(recipe_space[k]) - 1)
        cursor += w
        fields[k] = recipe_space[k][idx]
    checksum_ok = None
    if add_checksum:
        checksum_ok = (bits_to_int(maj[cursor: cursor + 4]) == (sum(maj[:cursor]) % 16))
    return {"decoded_recipe": fields, "majority_unit_bits": maj, "checksum_ok": checksum_ok}


def compute_ber(ref: List[int], dec: List[int]) -> float:
    n   = min(len(ref), len(dec))
    if n == 0: return 1.0
    err = sum(int(ref[i] != dec[i]) for i in range(n))
    err += abs(len(ref) - len(dec))
    return err / max(1, max(len(ref), len(dec)))


# ═══════════════════════════════════════════════════════════════════
# 6. MODEL LOADING / TENSOR SELECTION
# ═══════════════════════════════════════════════════════════════════
def _model_kwargs(torch_dtype_str: str, auth_token: Optional[str]) -> Dict:
    kw: Dict = {}
    dtype = parse_torch_dtype(torch_dtype_str)
    if dtype is not None:
        kw["torch_dtype"] = dtype
    if auth_token:
        kw["token"] = auth_token
    return kw


def load_model_and_tokenizer(model_name: str, device: torch.device,
                              torch_dtype_str: str = "auto",
                              auth_token: Optional[str] = None):
    tok_kw: Dict = {}
    if auth_token:
        tok_kw["token"] = auth_token

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True, **tok_kw
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"

    kw    = _model_kwargs(torch_dtype_str, auth_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    if len(tokenizer) > model.get_input_embeddings().num_embeddings:
        model.resize_token_embeddings(len(tokenizer))
    model.to(device)
    return model, tokenizer


def named_parameter_dict(model: nn.Module) -> Dict[str, nn.Parameter]:
    return {n: p for n, p in model.named_parameters()}


_PREF_MARKERS = [
    "attn.c_attn.weight","attn.c_proj.weight",
    "self_attn.q_proj.weight","self_attn.k_proj.weight",
    "self_attn.v_proj.weight","self_attn.o_proj.weight",
    "q_proj.weight","k_proj.weight","v_proj.weight","o_proj.weight",
    "query_key_value.weight",
    "mlp.c_fc.weight","mlp.c_proj.weight",
    "dense_h_to_4h.weight","dense_4h_to_h.weight",
    "fc1.weight","fc2.weight",
    "up_proj.weight","down_proj.weight","gate_proj.weight",
    "c_attn.weight","c_proj.weight","linear.weight",
]
_PREF_LAYER_TOKS = [
    "transformer.h.","model.layers.","layers.",
    "decoder.layers.","encoder.layers.","h.",
]


def _candidate_embed_tensors(model: nn.Module) -> List[Tuple[str, int]]:
    candidates = []
    for n, p in model.named_parameters():
        if not p.requires_grad or p.dim() < 2:
            continue
        lname = n.lower()
        score = 0
        for m in _PREF_MARKERS:
            if lname.endswith(m) or m in lname:
                score += 1000; break
        for t in _PREF_LAYER_TOKS:
            if t in lname:
                score += 100; break
        score += int(min(p.numel(), 10_000_000) // 1000)
        if score > 0:
            candidates.append((n, score))
    if not candidates:
        candidates = [(n, p.numel()) for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() >= 2]
    return sorted(candidates, key=lambda x: x[1], reverse=True)


def parse_embed_tensors(model: nn.Module, embed_tensors_str: str,
                        num_auto: int = 3) -> List[str]:
    all_names = {n for n, _ in model.named_parameters()}
    if embed_tensors_str.strip().lower() == "auto":
        ranked = _candidate_embed_tensors(model)
        names: List[str] = []
        for n, _ in ranked:
            if n not in names:
                names.append(n)
            if len(names) >= num_auto:
                break
        if not names:
            raise RuntimeError("Could not auto-select embedding tensors.")
        LOGGER.info("Auto-selected tensors (%d): %s", len(names), names)
        return names
    selected = [x.strip() for x in embed_tensors_str.split(",") if x.strip()]
    for n in selected:
        if n not in all_names:
            raise ValueError(f"Tensor '{n}' not found in model.")
    return selected


def flatten_selected_parameters(model: nn.Module, param_names: List[str]):
    pmap   = named_parameter_dict(model)
    chunks, meta = [], []
    cur = 0
    for name in param_names:
        p = pmap[name].detach().cpu().float().contiguous().view(-1)
        s, e = cur, cur + p.numel()
        chunks.append(p)
        meta.append((name, s, e, pmap[name].shape))
        cur = e
    return torch.cat(chunks, dim=0), meta


def write_back_flat_parameters(model: nn.Module, flat: torch.Tensor,
                                meta, device: torch.device) -> None:
    pmap = named_parameter_dict(model)
    with torch.no_grad():
        for name, s, e, shape in meta:
            buf = flat[s:e].view(shape).to(device=device, dtype=pmap[name].dtype)
            pmap[name].copy_(buf)


def clone_model_from_state(model_name: str, state_dict: Dict,
                            device: torch.device, torch_dtype_str: str = "auto",
                            auth_token: Optional[str] = None):
    """Load model and restore CPU state dict — no extra GPU copy."""
    kw = _model_kwargs(torch_dtype_str, auth_token)
    m  = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    m.load_state_dict(state_dict)   # state_dict is on CPU; PyTorch handles cast
    m.to(device)
    return m


def create_fresh_model(model_name: str, device: torch.device,
                       torch_dtype_str: str = "auto",
                       auth_token: Optional[str] = None):
    kw = _model_kwargs(torch_dtype_str, auth_token)
    m  = AutoModelForCausalLM.from_pretrained(model_name, **kw)
    m.to(device)
    return m


# ═══════════════════════════════════════════════════════════════════
# 7. OPTIMIZER / TRAINING / EVAL
# ═══════════════════════════════════════════════════════════════════
'''
def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    n = name.lower()
    if n == "adamw":    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if n == "adam":     return torch.optim.Adam(model.parameters(),  lr=lr, weight_decay=weight_decay)
    if n == "sgd":      return torch.optim.SGD(model.parameters(),   lr=lr, weight_decay=weight_decay, momentum=0.9)
    if n == "adagrad":  return torch.optim.Adagrad(model.parameters(),lr=lr, weight_decay=weight_decay)
    if n == "rmsprop":  return torch.optim.RMSprop(model.parameters(),lr=lr, weight_decay=weight_decay)
    if n == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(model.parameters(), lr=lr, relative_step=False,
                         scale_parameter=False, warmup_init=False, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")
'''

def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float):
    n = name.lower()
    if n == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if n == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
    if n == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if n == "adam8bit":
        import bitsandbytes as bnb
        return bnb.optim.Adam8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
    if n == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    if n == "adagrad":
        return torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    if n == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    if n == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(model.parameters(), lr=lr, relative_step=False,
                         scale_parameter=False, warmup_init=False, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")

def evaluate_loss(model: nn.Module, dataloader: DataLoader,
                  device: torch.device, max_batches: int = -1) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            if max_batches > 0 and bi >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            losses.append(float(model(**batch).loss.item()))
    return float(np.mean(losses)) if losses else float("inf")


def evaluate_ppl(model, dataloader, device, max_batches=-1) -> float:
    return float(math.exp(min(evaluate_loss(model, dataloader, device, max_batches), 50.0)))


def train_steps(model, train_loader, val_loader, device,
                optimizer_name, lr, weight_decay, warmup_ratio,
                effective_batch_size, max_steps, grad_clip,
                log_every, val_every, val_batches):
    model.train()
    micro_bs   = train_loader.batch_size
    grad_accum = max(1, int(round(effective_batch_size / micro_bs)))
    optimizer  = build_optimizer(model, optimizer_name, lr, weight_decay)
    total_opt  = max(1, math.ceil(max_steps / grad_accum))
    warmup_s   = int(round(total_opt * warmup_ratio))
    scheduler  = get_linear_schedule_with_warmup(optimizer, warmup_s, total_opt)

    train_iter    = iter(train_loader)
    train_losses: List[float] = []
    best_val_loss = float("inf")
    # ── KEY FIX: store best state on CPU ──────────────────────────
    best_state    = cpu_state_dict(model)
    opt_steps     = 0

    for step in range(1, max_steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        out   = model(**batch)
        loss  = out.loss / grad_accum
        loss.backward()
        train_losses.append(float(out.loss.item()))

        if step % grad_accum == 0:
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            opt_steps += 1

        if log_every > 0 and step % log_every == 0:
            LOGGER.info("step=%d  opt=%d  loss=%.5f  lr=%.4e",
                        step, opt_steps,
                        float(np.mean(train_losses[-log_every:])),
                        scheduler.get_last_lr()[0])

        if val_every > 0 and step % val_every == 0:
            vl = evaluate_loss(model, val_loader, device, max_batches=val_batches)
            LOGGER.info("val step=%d  val_loss=%.5f  ppl=%.3f",
                        step, vl, math.exp(min(vl, 50.0)))
            if vl < best_val_loss:
                best_val_loss = vl
                # ── KEY FIX: CPU clone ─────────────────────────────
                best_state = cpu_state_dict(model)
            model.train()

    # Restore best weights
    model.load_state_dict(best_state)
    fvl = evaluate_loss(model, val_loader, device, max_batches=val_batches)
    return {
        "final_val_loss":  float(fvl),
        "final_val_ppl":   float(math.exp(min(fvl, 50.0))),
        "best_val_loss":   float(min(best_val_loss, fvl)),
        "optimizer_steps": int(opt_steps),
        "grad_accum":      int(grad_accum),
    }


# ═══════════════════════════════════════════════════════════════════
# 8. CARRIER SELECTION
# ═══════════════════════════════════════════════════════════════════
def select_carriers(flat: torch.Tensor, num_needed: int,
                    seed: int, fraction: float, policy: str) -> np.ndarray:
    absw = np.abs(flat.detach().cpu().float().numpy())
    n    = len(absw)
    if policy == "all":
        idx = np.arange(n)
    else:
        k = min(max(num_needed, int(round(n * fraction))), n)
        idx = np.argpartition(absw, -k)[-k:] if policy == "high" else \
              np.argpartition(absw,  k - 1)[:k]
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    if len(idx) < num_needed:
        raise ValueError(f"Not enough carriers ({len(idx)} < {num_needed}).")
    return idx[:num_needed]


# ═══════════════════════════════════════════════════════════════════
# 9. LSB
# ═══════════════════════════════════════════════════════════════════
def embed_lsb(flat: torch.Tensor, payload_bits: List[int], scale: float,
              redundancy: int, seed: int, fraction: float, policy: str) -> Dict:
    flat   = flat.clone().cpu().float()
    total  = len(payload_bits) * redundancy
    chosen = select_carriers(flat, total, seed, fraction, policy)
    q      = torch.round(flat / scale).to(torch.int64)
    groups = []
    ptr    = 0
    for bit in payload_bits:
        grp = chosen[ptr: ptr + redundancy]; ptr += redundancy
        for idx in grp:
            qi = int(q[idx].item())
            q[idx] = qi | 1 if bit == 1 else qi & ~1
        groups.append(grp.tolist())
    return {"method": "lsb", "flat_embedded": q.to(torch.float32) * scale,
            "groups": groups, "scale": float(scale), "redundancy": int(redundancy),
            "seed": int(seed), "carrier_policy": policy, "carrier_fraction": float(fraction)}


def decode_lsb(flat: torch.Tensor, info: Dict, payload_len: int) -> List[int]:
    q   = torch.round(flat.clone().cpu().float() / info["scale"]).to(torch.int64)
    out = []
    for grp in info["groups"][:payload_len]:
        bits = [int(q[i].item()) & 1 for i in grp]
        out.append(int(sum(bits) >= len(bits) / 2.0))
    return out


# ═══════════════════════════════════════════════════════════════════
# 10. STDM
# ═══════════════════════════════════════════════════════════════════
def _chip(length: int, rng: np.random.Generator, mode: str = "rademacher") -> np.ndarray:
    s = (rng.standard_normal(length).astype(np.float32) if mode == "gaussian"
         else rng.choice(np.array([-1., 1.], dtype=np.float32), size=length))
    return s / (np.linalg.norm(s) + 1e-12)


def _stdm_q(z: float, delta: float, bit: int) -> float:
    offset = 0.0 if bit == 0 else delta / 2.0
    return float(round((z - offset) / delta) * delta + offset)


def _stdm_dec(z: float, delta: float) -> int:
    return 0 if abs(z - _stdm_q(z, delta, 0)) <= abs(z - _stdm_q(z, delta, 1)) else 1


def embed_stdm(flat: torch.Tensor, payload_bits: List[int], group_size: int,
               repeats: int, delta: float, seed: int, fraction: float, policy: str,
               chip_mode: str = "rademacher", adaptive_delta: bool = True) -> Dict:
    flat = flat.clone().cpu().float()
    emb  = flat.clone()
    total_carriers = len(payload_bits) * repeats * group_size
    chosen = select_carriers(flat, total_carriers, seed, fraction, policy)
    rng    = np.random.default_rng(seed)
    rng.shuffle(chosen)

    groups, pns, dpg, bgmap = [], [], [], []
    ptr = 0
    for _ in range(len(payload_bits) * repeats):
        grp = chosen[ptr: ptr + group_size]; ptr += group_size
        groups.append(grp.tolist())
        pns.append(_chip(group_size, rng, chip_mode).tolist())

    gptr = 0
    for bit in payload_bits:
        local = []
        for _ in range(repeats):
            s   = torch.tensor(pns[gptr], dtype=torch.float32)
            idx = torch.tensor(groups[gptr], dtype=torch.long)
            x   = emb[idx]
            z   = float(torch.dot(x, s).item())
            de  = float(delta) * (float(torch.std(x).item()) + 1e-8) if adaptive_delta else float(delta)
            zq  = _stdm_q(z, de, int(bit))
            emb[idx] = x + float(zq - z) * s
            dpg.append(float(de)); local.append(gptr); gptr += 1
        bgmap.append(local)

    return {"method": "stdm", "flat_embedded": emb, "groups": groups, "pn_sequences": pns,
            "bit_group_map": bgmap, "group_size": int(group_size), "repeats": int(repeats),
            "delta": float(delta), "delta_per_group": dpg, "seed": int(seed),
            "chip_mode": chip_mode, "adaptive_delta": bool(adaptive_delta),
            "carrier_policy": policy, "carrier_fraction": float(fraction)}


def decode_stdm(flat: torch.Tensor, info: Dict, payload_len: int) -> List[int]:
    fn = flat.clone().cpu().float()
    out = []
    for local in info["bit_group_map"][:payload_len]:
        votes = []
        for gid in local:
            s = torch.tensor(info["pn_sequences"][gid], dtype=torch.float32)
            x = fn[torch.tensor(info["groups"][gid], dtype=torch.long)]
            votes.append(_stdm_dec(float(torch.dot(x, s).item()),
                                   float(info["delta_per_group"][gid])))
        out.append(int(sum(votes) >= len(votes) / 2.0))
    return out


# ═══════════════════════════════════════════════════════════════════
# 11. ATTACKS
# ═══════════════════════════════════════════════════════════════════

'''
def apply_pruning(model: nn.Module, prune_ratio: float) -> None:
    if prune_ratio <= 0: return
    params = [(m, "weight") for m in model.modules()
              if hasattr(m, "weight") and isinstance(m.weight, torch.Tensor)
              and m.weight.dim() >= 2]
    if params:
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=prune_ratio)
        for module, name in params:
            prune.remove(module, name)
'''

# Applying on CPU, Then move back to GPU.


def apply_pruning(model: nn.Module, prune_ratio: float) -> None:
    if prune_ratio <= 0:
        return
    # ── KEY FIX: move to CPU for pruning to avoid OOM on large models ──
    device = next(model.parameters()).device
    model.cpu()
    params = []
    for module in model.modules():
        for name, _ in module.named_parameters(recurse=False):
            if name == "weight":
                params.append((module, name))
    if params:
        prune.global_unstructured(
            params, pruning_method=prune.L1Unstructured, amount=float(prune_ratio)
        )
        for module, name in params:
            prune.remove(module, name)
    # move back to original device
    model.to(device)

def fine_tune_model(model, train_loader, val_loader, device, base_recipe,
                    ft_lr, ft_steps, grad_clip, val_batches):
    if ft_steps <= 0:
        vl = evaluate_loss(model, val_loader, device, max_batches=val_batches)
        return {"ft_val_loss": vl, "ft_val_ppl": math.exp(min(vl, 50.0))}
    s = train_steps(model, train_loader, val_loader, device,
                    base_recipe.optimizer, ft_lr, base_recipe.weight_decay,
                    base_recipe.warmup_ratio, base_recipe.effective_batch_size,
                    ft_steps, grad_clip,
                    max(10, ft_steps // 4), max(20, ft_steps // 2), val_batches)
    return {"ft_val_loss": s["final_val_loss"], "ft_val_ppl": s["final_val_ppl"]}


def reconstruct_recipe(d: Optional[Dict]) -> Optional[Recipe]:
    if d is None: return None
    try:
        return Recipe(str(d["optimizer"]), float(d["lr"]), float(d["weight_decay"]),
                      float(d["warmup_ratio"]), int(d["effective_batch_size"]))
    except Exception:
        return None


def reproduce_with_recipe(model_name, recipe, device, train_loader, val_loader,
                           steps, grad_clip, log_every, val_every, val_batches,
                           torch_dtype_str, auth_token):
    if recipe is None:
        return {"repro_status": "no_decoded_recipe",
                "repro_final_val_loss": None, "repro_final_val_ppl": None}
    try:
        m = create_fresh_model(model_name, device, torch_dtype_str, auth_token)
        s = train_steps(m, train_loader, val_loader, device,
                        recipe.optimizer, recipe.lr, recipe.weight_decay,
                        recipe.warmup_ratio, recipe.effective_batch_size,
                        steps, grad_clip, log_every, val_every, val_batches)
        del m; cleanup_memory()
        return {"repro_status": "ok",
                "repro_final_val_loss": float(s["final_val_loss"]),
                "repro_final_val_ppl":  float(s["final_val_ppl"])}
    except Exception as e:
        cleanup_memory()
        return {"repro_status": f"error:{e}",
                "repro_final_val_loss": None, "repro_final_val_ppl": None}


# ═══════════════════════════════════════════════════════════════════
# 12. PARSE HELPERS
# ═══════════════════════════════════════════════════════════════════
def parse_float_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]

def parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]

def parse_json_space(s: str) -> Dict[str, List]:
    obj = json.loads(s)
    for k in ["optimizer","weight_decay","warmup_ratio","effective_batch_size"]:
        if k not in obj or not isinstance(obj[k], list) or len(obj[k]) == 0:
            raise ValueError(f"refine_json must have non-empty list for '{k}'")
    return obj

def recipe_grid(rs: Dict[str, List]) -> Iterable[Recipe]:
    for vals in product(rs["optimizer"], rs["lr"], rs["weight_decay"],
                        rs["warmup_ratio"], rs["effective_batch_size"]):
        yield Recipe(*vals)


# ═══════════════════════════════════════════════════════════════════
# 13. U-SHAPED SWEEP & REFINE
# ═══════════════════════════════════════════════════════════════════
def run_ushape_sweep_and_refine(args, model_name, device, train_loader,
                                 val_loader, recipe_space, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    auth = getattr(args, "auth_token", None)

    base_model = create_fresh_model(model_name, device, args.torch_dtype, auth)
    # ── KEY FIX: CPU state dict ────────────────────────────────────
    init_state = cpu_state_dict(base_model)
    del base_model; cleanup_memory()

    anc_opt = recipe_space["optimizer"][0]
    anc_wd  = recipe_space["weight_decay"][0]
    anc_wr  = recipe_space["warmup_ratio"][0]
    anc_ebs = recipe_space["effective_batch_size"][0]

    # Wide LR sweep
    sweep_rows: List[Dict] = []
    best_lr_row = None
    for lr in recipe_space["lr"]:
        model = clone_model_from_state(model_name, init_state, device, args.torch_dtype, auth)
        stats = train_steps(model, train_loader, val_loader, device,
                            anc_opt, float(lr), float(anc_wd), float(anc_wr), int(anc_ebs),
                            args.sweep_steps, args.grad_clip,
                            args.log_every, args.val_every, args.val_batches)
        row = {"stage":"wide_lr_sweep","optimizer":anc_opt,"lr":float(lr),
               "weight_decay":float(anc_wd),"warmup_ratio":float(anc_wr),
               "effective_batch_size":int(anc_ebs), **stats}
        sweep_rows.append(row)
        if best_lr_row is None or row["final_val_loss"] < best_lr_row["final_val_loss"]:
            best_lr_row = row.copy()
        del model; cleanup_memory()

    best_lr = float(best_lr_row["lr"])
    LOGGER.info("Best LR: %.4e", best_lr)

    # Refine at best LR
    refine_rows: List[Dict] = []
    best_recipe_row = None
    for recipe in recipe_grid(recipe_space):
        if abs(float(recipe.lr) - best_lr) > 1e-12:
            continue
        model = clone_model_from_state(model_name, init_state, device, args.torch_dtype, auth)
        stats = train_steps(model, train_loader, val_loader, device,
                            recipe.optimizer, float(recipe.lr), float(recipe.weight_decay),
                            float(recipe.warmup_ratio), int(recipe.effective_batch_size),
                            args.refine_steps, args.grad_clip,
                            args.log_every, args.val_every, args.val_batches)
        row = {"stage":"refine_best_lr", **asdict(recipe), **stats}
        refine_rows.append(row)
        if best_recipe_row is None or row["final_val_loss"] < best_recipe_row["final_val_loss"]:
            best_recipe_row = row.copy()
        del model; cleanup_memory()

    if best_recipe_row is None:
        raise RuntimeError("No refined recipes evaluated.")

    best_recipe = Recipe(best_recipe_row["optimizer"], float(best_recipe_row["lr"]),
                         float(best_recipe_row["weight_decay"]), float(best_recipe_row["warmup_ratio"]),
                         int(best_recipe_row["effective_batch_size"]))

    # Canonical training
    model = clone_model_from_state(model_name, init_state, device, args.torch_dtype, auth)
    canonical_stats = train_steps(
        model, train_loader, val_loader, device,
        best_recipe.optimizer, best_recipe.lr, best_recipe.weight_decay,
        best_recipe.warmup_ratio, best_recipe.effective_batch_size,
        args.canonical_steps, args.grad_clip,
        args.log_every, args.val_every, args.val_batches)
    # ── KEY FIX: CPU state dict ────────────────────────────────────
    canonical_state = cpu_state_dict(model)
    torch.save(canonical_state, out_dir / "best_canonical_state.pt")
    del model; cleanup_memory()

    # Save CSVs
    for fname, rows in [("wide_lr_sweep.csv", sweep_rows), ("refine_best_lr.csv", refine_rows)]:
        if rows:
            with open(out_dir / fname, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
    with open(out_dir / "sweep_summary.json", "w") as f:
        json.dump({"best_lr_row": best_lr_row, "best_recipe_row": best_recipe_row,
                   "best_recipe": asdict(best_recipe), "canonical_stats": canonical_stats},
                  f, indent=2)

    return {"best_recipe": best_recipe, "canonical_state": canonical_state,
            "wide_lr_rows": sweep_rows, "refine_rows": refine_rows,
            "canonical_stats": canonical_stats}


# ═══════════════════════════════════════════════════════════════════
# 14. METHOD EXPERIMENT
# ═══════════════════════════════════════════════════════════════════
def run_method_experiment(args, method, model_name, canonical_state, recipe_space,
                          best_recipe, base_stats, canonical_stats,
                          train_loader, val_loader, device):
    auth = getattr(args, "auth_token", None)
    model = clone_model_from_state(model_name, canonical_state, device, args.torch_dtype, auth)
    embed_names = parse_embed_tensors(model, args.embed_tensors, args.num_auto_tensors)

    payload_meta = serialize_recipe(best_recipe, recipe_space,
                                    args.payload_repeat, not args.no_checksum)
    payload_bits = payload_meta["payload_bits"]

    pre_embed_loss = evaluate_loss(model, val_loader, device, max_batches=args.val_batches)
    pre_embed_ppl  = float(math.exp(min(pre_embed_loss, 50.0)))

    flat_ref, meta = flatten_selected_parameters(model, embed_names)

    if method == "lsb":
        info      = embed_lsb(flat_ref, payload_bits, args.lsb_scale, args.lsb_redundancy,
                               args.seed, args.lsb_carrier_fraction, args.lsb_carrier_policy)
        decode_fn = decode_lsb
    elif method == "stdm":
        info      = embed_stdm(flat_ref, payload_bits, args.stdm_group_size, args.stdm_repeats,
                               args.stdm_delta, args.seed, args.stdm_carrier_fraction,
                               args.stdm_carrier_policy, args.stdm_chip_mode,
                               not args.stdm_disable_adaptive_delta)
        decode_fn = decode_stdm
    else:
        raise ValueError(method)

    write_back_flat_parameters(model, info["flat_embedded"], meta, device)
    emb_loss = evaluate_loss(model, val_loader, device, max_batches=args.val_batches)
    emb_ppl  = float(math.exp(min(emb_loss, 50.0)))

    flat_clean, _ = flatten_selected_parameters(model, embed_names)
    clean_bits    = decode_fn(flat_clean, info, len(payload_bits))
    clean_ber     = compute_ber(payload_bits, clean_bits)
    decoded_clean = decode_recipe_payload(clean_bits, recipe_space,
                                          args.payload_repeat, not args.no_checksum)
    # ── KEY FIX: CPU state dict ────────────────────────────────────
    watermarked_state = cpu_state_dict(model)
    del model; cleanup_memory()

    rows: List[Dict] = []
    attack_perf: List[Dict] = []

    for ft_lr in parse_float_list(args.ft_lrs):
        for prune_ratio in parse_float_list(args.prune_ratios):
            for ft_steps in parse_int_list(args.ft_steps):
                attacked = clone_model_from_state(model_name, watermarked_state,
                                                   device, args.torch_dtype, auth)
                apply_pruning(attacked, prune_ratio)
                ft_stats = fine_tune_model(attacked, train_loader, val_loader, device,
                                           best_recipe, ft_lr, ft_steps,
                                           args.grad_clip, args.val_batches)
                flat_att, _ = flatten_selected_parameters(attacked, embed_names)
                bits         = decode_fn(flat_att, info, len(payload_bits))
                ber          = compute_ber(payload_bits, bits)
                decoded      = decode_recipe_payload(bits, recipe_space,
                                                     args.payload_repeat, not args.no_checksum)
                dec_recipe   = reconstruct_recipe(decoded["decoded_recipe"])
                recipe_match = (dec_recipe is not None and
                                recipe_to_key(dec_recipe) == recipe_to_key(best_recipe))

                repro = {"repro_status": "disabled",
                         "repro_final_val_loss": None, "repro_final_val_ppl": None}
                if args.enable_reproduce_check:
                    rs = max(10, args.reproduce_steps // 5)
                    rv = max(20, args.reproduce_steps // 3)
                    repro = reproduce_with_recipe(model_name, dec_recipe, device,
                                                  train_loader, val_loader,
                                                  args.reproduce_steps, args.grad_clip,
                                                  rs, rv, args.val_batches,
                                                  args.torch_dtype, auth)
                gap = (float(repro["repro_final_val_ppl"] - canonical_stats["final_val_ppl"])
                       if repro["repro_final_val_ppl"] is not None else None)

                row = {
                    "method": method,
                    "base_val_loss": float(base_stats["final_val_loss"]),
                    "base_val_ppl":  float(base_stats["final_val_ppl"]),
                    "canonical_val_loss": float(canonical_stats["final_val_loss"]),
                    "canonical_val_ppl":  float(canonical_stats["final_val_ppl"]),
                    "pre_embed_val_loss": float(pre_embed_loss),
                    "pre_embed_val_ppl":  float(pre_embed_ppl),
                    "embedded_val_loss":  float(emb_loss),
                    "embedded_val_ppl":   float(emb_ppl),
                    "clean_ber":          float(clean_ber),
                    "prune_ratio":        float(prune_ratio),
                    "ft_steps":           int(ft_steps),
                    "ft_lr":              float(ft_lr),
                    "ber":                float(ber),
                    "payload_len":        int(len(payload_bits)),
                    "decoded_recipe":     json.dumps(decoded["decoded_recipe"])
                                          if decoded["decoded_recipe"] else "null",
                    "checksum_ok":        decoded.get("checksum_ok"),
                    "recipe_exact_match": bool(recipe_match),
                    "attacked_val_loss":  float(ft_stats["ft_val_loss"]),
                    "attacked_val_ppl":   float(ft_stats["ft_val_ppl"]),
                    "repro_status":       repro["repro_status"],
                    "repro_val_loss":     repro["repro_final_val_loss"],
                    "repro_val_ppl":      repro["repro_final_val_ppl"],
                    "repro_gap_vs_canonical_ppl": gap,
                    "carrier_policy":     info["carrier_policy"],
                    "carrier_fraction":   info["carrier_fraction"],
                }
                rows.append(row)
                attack_perf.append({
                    "method": method,
                    "base_val_ppl":      float(base_stats["final_val_ppl"]),
                    "canonical_val_ppl": float(canonical_stats["final_val_ppl"]),
                    "embedded_val_ppl":  float(emb_ppl),
                    "prune_ratio":       float(prune_ratio),
                    "ft_steps":          int(ft_steps),
                    "ft_lr":             float(ft_lr),
                    "attacked_val_ppl":  float(ft_stats["ft_val_ppl"]),
                    "ber":               float(ber),
                    "recipe_exact_match": bool(recipe_match),
                })
                LOGGER.info("[%s] prune=%.2f ft_steps=%d ft_lr=%.2e  BER=%.4f  atkPPL=%.3f",
                            method, prune_ratio, ft_steps, ft_lr, ber, ft_stats["ft_val_ppl"])
                del attacked; cleanup_memory()

    extra = {"carrier_policy": info["carrier_policy"], "carrier_fraction": info["carrier_fraction"]}
    if method == "stdm":
        extra.update({k: info[k] for k in ["group_size","repeats","delta","chip_mode","adaptive_delta"]})
        extra["blind_decode"] = True

    return {
        "method": method, "rows": rows, "attack_perf_rows": attack_perf,
        "perf_summary_row": {
            "method": method,
            "base_val_loss": float(base_stats["final_val_loss"]),
            "base_val_ppl":  float(base_stats["final_val_ppl"]),
            "canonical_val_loss": float(canonical_stats["final_val_loss"]),
            "canonical_val_ppl":  float(canonical_stats["final_val_ppl"]),
            "pre_embed_val_loss": float(pre_embed_loss),
            "pre_embed_val_ppl":  float(pre_embed_ppl),
            "embedded_val_loss":  float(emb_loss),
            "embedded_val_ppl":   float(emb_ppl),
            "clean_ber":          float(clean_ber),
        },
        "payload_meta":  payload_meta,
        "embedded_ppl":  emb_ppl,
        "pre_embed_ppl": pre_embed_ppl,
        "clean_ber":     clean_ber,
        "decoded_clean": decoded_clean,
        "embed_names":   embed_names,
        "embed_info":    extra,
    }


# ═══════════════════════════════════════════════════════════════════
# 15. SAVING
# ═══════════════════════════════════════════════════════════════════
def save_json(path: Path, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)

def save_csv(path: Path, rows: List[Dict]) -> None:
    if not rows: return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


# ═══════════════════════════════════════════════════════════════════
# 16. PLOTS (complete suite — 25 figures)
# ═══════════════════════════════════════════════════════════════════
_C  = {"lsb": "#E74C3C", "stdm": "#2980B9"}
_MK = {"lsb": "o",       "stdm": "s"}
_HT = {"lsb": "//",      "stdm": ".."}


def _save(fig, path, dpi=220):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Plot → %s", path)

# Habibur Changed----------------------------------
'''
def plot_ushape_curve(rows, out):
    rows = sorted(rows, key=lambda r: float(r["lr"]))
    xs   = [float(r["lr"]) for r in rows]
    ys   = [float(r["final_val_ppl"]) for r in rows]
    bx   = xs[int(np.argmin(ys))]
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(xs, ys, marker="o", color="#2C3E50", lw=2, ms=6)
    ax.axvline(bx, color="#E74C3C", ls="--", alpha=0.7, label=f"Best LR={bx:.2e}")
    ax.scatter([bx], [min(ys)], color="#E74C3C", s=120, zorder=5)
    ax.set_xscale("log"); ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Validation PPL", fontsize=12)
    ax.set_title("U-Shaped LR Sweep", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out)
'''
# -----------------------------------------------------
'''
def plot_ushape_curve(wide_lr_rows, out_path):
    rows   = sorted(wide_lr_rows, key=lambda r: float(r["lr"]))
    xs     = [float(r["lr"])            for r in rows]
    ys     = [float(r["final_val_ppl"]) for r in rows]
    best_x = xs[int(np.argmin(ys))]
    best_y = min(ys)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", color="#2C3E50", linewidth=2, markersize=6, zorder=3)
    ax.axvline(best_x, color="#E74C3C", linestyle="--", alpha=0.7,
               label=f"Best LR = {best_x:.2e}")
    ax.scatter([best_x], [best_y], color="#E74C3C", s=120, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")        # ← ADD THIS LINE
    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Validation Perplexity (log scale)", fontsize=12)
    ax.set_title("U-Shaped LR Sweep", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    _fig_save(fig, out_path)
'''

def plot_ushape_curve(wide_lr_rows, out_path):
    rows   = sorted(wide_lr_rows, key=lambda r: float(r["lr"]))
    xs     = [float(r["lr"])            for r in rows]
    ys     = [float(r["final_val_ppl"]) for r in rows]
    best_x = xs[int(np.argmin(ys))]
    best_y = min(ys)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, ys, marker="o", color="#2C3E50", linewidth=2, markersize=6, zorder=3)
    ax.axvline(best_x, color="#E74C3C", linestyle="--", alpha=0.7,
               label=f"Best LR = {best_x:.2e}")
    ax.scatter([best_x], [best_y], color="#E74C3C", s=120, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Learning Rate", fontsize=12)
    ax.set_ylabel("Validation Perplexity (log scale)", fontsize=12)
    ax.set_title("U-Shaped LR Sweep", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def plot_refine_param_bar(rows, param, out):
    if not rows: return
    uniq  = list(dict.fromkeys(r[param] for r in rows))
    means = [float(np.mean([float(r["final_val_ppl"]) for r in rows if r[param]==v])) for v in uniq]
    fig, ax = plt.subplots(figsize=(8,5))
    xp = np.arange(len(uniq))
    bars = ax.bar(xp, means, color="#5DADE2", edgecolor="black", lw=0.7)
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.set_xticks(xp); ax.set_xticklabels([str(v) for v in uniq], rotation=25, ha="right")
    ax.set_xlabel(param, fontsize=12); ax.set_ylabel("Mean Val PPL", fontsize=12)
    ax.set_title(f"Val PPL vs {param}", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out)


def plot_refine_optimizer_lines(rows, out):
    if not rows: return
    opts    = sorted(set(str(r["optimizer"]) for r in rows))
    warmups = sorted(set(float(r["warmup_ratio"]) for r in rows))
    fig, ax = plt.subplots(figsize=(8,5))
    cmap = plt.cm.get_cmap("tab10", len(opts))
    for i, opt in enumerate(opts):
        ys = [float(np.mean([float(r["final_val_ppl"]) for r in rows
                              if str(r["optimizer"])==opt and abs(float(r["warmup_ratio"])-w)<1e-12] or [np.nan]))
              for w in warmups]
        ax.plot(warmups, ys, marker="o", color=cmap(i), label=opt, lw=2)
    ax.set_xlabel("Warmup Ratio", fontsize=12); ax.set_ylabel("Mean Val PPL", fontsize=12)
    ax.set_title("Optimizer vs Warmup Ratio", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out)


def plot_refine_ebs_lines(rows, out):
    if not rows: return
    ebs_vals = sorted(set(int(r["effective_batch_size"]) for r in rows))
    wds      = sorted(set(float(r["weight_decay"]) for r in rows))
    fig, ax  = plt.subplots(figsize=(8,5))
    cmap     = plt.cm.get_cmap("Set1", len(ebs_vals))
    for i, ebs in enumerate(ebs_vals):
        ys = [float(np.mean([float(r["final_val_ppl"]) for r in rows
                              if int(r["effective_batch_size"])==ebs and abs(float(r["weight_decay"])-wd)<1e-12] or [np.nan]))
              for wd in wds]
        ax.plot(wds, ys, marker="o", color=cmap(i), label=f"EBS={ebs}", lw=2)
    ax.set_xscale("log" if any(w>0 for w in wds) else "linear")
    ax.set_xlabel("Weight Decay", fontsize=12); ax.set_ylabel("Mean Val PPL", fontsize=12)
    ax.set_title("EBS vs Weight Decay", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out)


def _agg_line(rows, method, xkey, xvals, ykey="ber", xfilt=None):
    sub = [r for r in rows if r["method"]==method and (xfilt is None or xfilt(r))]
    ys  = []
    for x in xvals:
        if xkey == "prune_ratio":
            vals = [float(r[ykey]) for r in sub if abs(float(r[xkey])-x)<1e-12]
        else:
            vals = [float(r[ykey]) for r in sub if int(r[xkey])==int(x)]
        ys.append(float(np.mean(vals)) if vals else np.nan)
    return ys


def plot_ber_vs_pruning(rows, out):
    fig, ax = plt.subplots(figsize=(8,5))
    for m in ["lsb","stdm"]:
        sub = [r for r in rows if r["method"]==m and int(r["ft_steps"])==0]
        xs  = sorted(set(float(r["prune_ratio"]) for r in sub))
        ys  = _agg_line(rows, m, "prune_ratio", xs,
                        xfilt=lambda r: int(r["ft_steps"])==0)
        ax.plot(xs, ys, marker=_MK[m], color=_C[m], label=m.upper(), lw=2, ms=7)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="BER=0.5")
    ax.set_xlabel("Pruning Ratio", fontsize=12); ax.set_ylabel("BER", fontsize=12)
    ax.set_title("BER under Pruning", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.02, 1.05); ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out)


def plot_ber_vs_ft_steps(rows, out):
    fig, ax = plt.subplots(figsize=(8,5))
    for m in ["lsb","stdm"]:
        sub = [r for r in rows if r["method"]==m and abs(float(r["prune_ratio"]))<1e-12]
        xs  = sorted(set(int(r["ft_steps"]) for r in sub))
        ys  = _agg_line(rows, m, "ft_steps", xs,
                        xfilt=lambda r: abs(float(r["prune_ratio"]))<1e-12)
        ax.plot(xs, ys, marker=_MK[m], color=_C[m], label=m.upper(), lw=2, ms=7)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Fine-Tuning Steps", fontsize=12); ax.set_ylabel("BER", fontsize=12)
    ax.set_title("BER under Fine-Tuning", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.02, 1.05); ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out)


def plot_finetune_combined_by_lr(rows, out):
    sub  = [r for r in rows if abs(float(r["prune_ratio"]))<1e-12]
    lrs  = sorted(set(float(r["ft_lr"]) for r in sub))
    fig, ax = plt.subplots(figsize=(9,5))
    cmap = plt.cm.get_cmap("Paired", len(lrs)*2)
    idx  = 0
    for m in ["lsb","stdm"]:
        for lr in lrs:
            ss = [r for r in sub if r["method"]==m and abs(float(r["ft_lr"])-lr)<1e-12]
            xs = sorted(set(int(r["ft_steps"]) for r in ss))
            ys = [float(np.mean([float(r["ber"]) for r in ss if int(r["ft_steps"])==x])) for x in xs]
            ax.plot(xs, ys, marker=_MK[m], color=cmap(idx),
                    label=f"{m.upper()} lr={lr:.1e}", lw=1.8, ms=6)
            idx += 1
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("FT Steps", fontsize=12); ax.set_ylabel("BER", fontsize=12)
    ax.set_title("FT BER by LR (No Pruning)", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.02,1.05); ax.grid(True, alpha=0.3); ax.legend(fontsize=8, ncol=2)
    _save(fig, out)


def plot_finetune_by_lr(rows, method, out):
    sub = [r for r in rows if r["method"]==method and abs(float(r["prune_ratio"]))<1e-12]
    lrs = sorted(set(float(r["ft_lr"]) for r in sub))
    fig, ax = plt.subplots(figsize=(8,5))
    cmap = plt.cm.get_cmap("plasma", len(lrs)+1)
    for i, lr in enumerate(lrs):
        ss = [r for r in sub if abs(float(r["ft_lr"])-lr)<1e-12]
        xs = sorted(set(int(r["ft_steps"]) for r in ss))
        ys = [float(np.mean([float(r["ber"]) for r in ss if int(r["ft_steps"])==x])) for x in xs]
        ax.plot(xs, ys, marker="o", color=cmap(i), label=f"ft_lr={lr:.1e}", lw=2)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("FT Steps", fontsize=12); ax.set_ylabel("BER", fontsize=12)
    ax.set_title(f"{method.upper()} — BER vs FT Steps", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.02,1.05); ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out)


def plot_heatmap(rows, method, ft_lr, out, value_key="ber"):
    sub = [r for r in rows if r["method"]==method]
    if ft_lr is not None:
        sub = [r for r in sub if abs(float(r["ft_lr"])-ft_lr)<1e-12]
    if not sub: return
    prunes  = sorted(set(float(r["prune_ratio"]) for r in sub))
    ftsteps = sorted(set(int(r["ft_steps"])      for r in sub))
    grid    = np.full((len(ftsteps), len(prunes)), np.nan, dtype=np.float32)
    for i, st in enumerate(ftsteps):
        for j, pr in enumerate(prunes):
            vals = [float(r[value_key]) for r in sub
                    if int(r["ft_steps"])==st and abs(float(r["prune_ratio"])-pr)<1e-12]
            if vals: grid[i,j] = np.mean(vals)
    fig, ax = plt.subplots(figsize=(8,5))
    cmap    = "RdYlGn_r" if value_key=="ber" else "viridis"
    im      = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap,
                        vmin=0 if value_key=="ber" else None,
                        vmax=1 if value_key=="ber" else None)
    plt.colorbar(im, ax=ax, label=value_key)
    ax.set_xticks(np.arange(len(prunes)));  ax.set_xticklabels([str(x) for x in prunes])
    ax.set_yticks(np.arange(len(ftsteps))); ax.set_yticklabels([str(x) for x in ftsteps])
    ax.set_xlabel("Pruning Ratio", fontsize=12); ax.set_ylabel("FT Steps", fontsize=12)
    title = f"{method.upper()} — {value_key}"
    if ft_lr: title += f" (ft_lr={ft_lr:.1e})"
    ax.set_title(title, fontsize=13, fontweight="bold")
    for i in range(len(ftsteps)):
        for j in range(len(prunes)):
            if not np.isnan(grid[i,j]):
                ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=7)
    _save(fig, out)


def plot_utility(rows, out):
    fig, ax = plt.subplots(figsize=(8,5))
    for m in ["lsb","stdm"]:
        sub = [r for r in rows if r["method"]==m and abs(float(r["prune_ratio"]))<1e-12]
        ax.scatter([float(r["ber"]) for r in sub],
                   [float(r["attacked_val_ppl"]) for r in sub],
                   label=m.upper(), color=_C[m], alpha=0.75, s=50)
    ax.set_xlabel("BER", fontsize=12); ax.set_ylabel("Attacked Model PPL", fontsize=12)
    ax.set_title("Robustness vs Utility", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out)


def plot_recipe_match_rate(rows, out):
    fig, ax = plt.subplots(figsize=(8,5))
    for m in ["lsb","stdm"]:
        sub = [r for r in rows if r["method"]==m]
        xs  = sorted(set(float(r["prune_ratio"]) for r in sub))
        ys  = [float(np.mean([1.0 if bool(r["recipe_exact_match"]) else 0.0
                               for r in sub if abs(float(r["prune_ratio"])-x)<1e-12])) for x in xs]
        ax.plot(xs, ys, marker=_MK[m], color=_C[m], label=m.upper(), lw=2, ms=7)
    ax.set_xlabel("Pruning Ratio", fontsize=12); ax.set_ylabel("Recipe Recovery Rate", fontsize=12)
    ax.set_title("Recipe Recovery Rate vs Pruning", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05,1.10); ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out)


def plot_recipe_match_vs_ftsteps(rows, out):
    fig, ax = plt.subplots(figsize=(8,5))
    for m in ["lsb","stdm"]:
        sub = [r for r in rows if r["method"]==m and abs(float(r["prune_ratio"]))<1e-12]
        xs  = sorted(set(int(r["ft_steps"]) for r in sub))
        ys  = [float(np.mean([1.0 if bool(r["recipe_exact_match"]) else 0.0
                               for r in sub if int(r["ft_steps"])==x])) for x in xs]
        ax.plot(xs, ys, marker=_MK[m], color=_C[m], label=m.upper(), lw=2, ms=7)
    ax.set_xlabel("FT Steps", fontsize=12); ax.set_ylabel("Recipe Recovery Rate", fontsize=12)
    ax.set_title("Recipe Recovery Rate vs Fine-Tuning", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.05,1.10); ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out)


def plot_repro_gap(rows, out):
    usable = [r for r in rows if r.get("repro_val_ppl") is not None]
    if not usable: return
    fig, ax = plt.subplots(figsize=(8,5))
    for m in ["lsb","stdm"]:
        sub = [r for r in usable if r["method"]==m]
        xs  = sorted(set(float(r["prune_ratio"]) for r in sub))
        ys  = [float(np.mean([float(r["repro_gap_vs_canonical_ppl"])
                               for r in sub if abs(float(r["prune_ratio"])-x)<1e-12
                               and r["repro_gap_vs_canonical_ppl"] is not None] or [np.nan])) for x in xs]
        ax.plot(xs, ys, marker=_MK[m], color=_C[m], label=m.upper(), lw=2, ms=7)
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_xlabel("Pruning Ratio", fontsize=12); ax.set_ylabel("PPL Gap vs Canonical", fontsize=12)
    ax.set_title("Reproduction Gap vs Pruning", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out)


def plot_stage_performance_bar(base_stats, canonical_stats, perf_rows, out):
    pm   = {r["method"]: r for r in perf_rows}
    lbls = ["Base\nPretrained","Trained\nCanonical","LSB\nEmbedded","STDM\nEmbedded"]
    vals = [float(base_stats["final_val_ppl"]), float(canonical_stats["final_val_ppl"]),
            float(pm["lsb"]["embedded_val_ppl"]) if "lsb" in pm else np.nan,
            float(pm["stdm"]["embedded_val_ppl"]) if "stdm" in pm else np.nan]
    colors = ["#95A5A6","#2ECC71",_C["lsb"],_C["stdm"]]
    fig, ax = plt.subplots(figsize=(9,5))
    xp   = np.arange(len(lbls))
    bars = ax.bar(xp, vals, color=colors, edgecolor="black", lw=0.8, width=0.6)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=10, fontweight="bold")
    ax.set_xticks(xp); ax.set_xticklabels(lbls, fontsize=11)
    ax.set_ylabel("Validation PPL", fontsize=12)
    ax.set_title("Stage-wise Model Performance", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    _save(fig, out)


def plot_ppl_overhead(base_stats, canonical_stats, perf_rows, out):
    pm  = {r["method"]: r for r in perf_rows}
    methods = ["LSB","STDM"]; keys = ["lsb","stdm"]
    x = np.arange(len(methods)); w = 0.25
    fig, ax = plt.subplots(figsize=(8,5))
    bv = [float(base_stats["final_val_ppl"])] * 2
    cv = [float(canonical_stats["final_val_ppl"])] * 2
    ev = [float(pm[k]["embedded_val_ppl"]) if k in pm else np.nan for k in keys]
    b1 = ax.bar(x-w, bv, w, label="Base",     color="#95A5A6", edgecolor="black")
    b2 = ax.bar(x,   cv, w, label="Trained",  color="#2ECC71", edgecolor="black")
    b3 = ax.bar(x+w, ev, w, label="Embedded", color="#E67E22", edgecolor="black")
    for bars in [b1, b2, b3]:
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylabel("Val PPL", fontsize=12)
    ax.set_title("PPL Overhead: Base → Trained → Embedded", fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3); ax.legend()
    _save(fig, out)


def _find_break(vals):
    clean = [v for v in vals if not np.isnan(v)]
    if len(clean) < 2: return None
    diffs = [clean[i]-clean[i-1] for i in range(1, len(clean))]
    mi    = int(np.argmax(diffs))
    return mi + 1 if diffs[mi] > 0.05 else None


def plot_ber_bar_finetune_break(rows, out):
    def agg(m):
        sub = [r for r in rows if r["method"]==m and abs(float(r["prune_ratio"]))<1e-12]
        xs  = sorted(set(int(r["ft_steps"]) for r in sub))
        ys  = [float(np.mean([float(r["ber"]) for r in sub if int(r["ft_steps"])==x])) for x in xs]
        return xs, ys
    xl, yl = agg("lsb"); xs, ys = agg("stdm")
    steps  = sorted(set(xl)|set(xs))
    lm = dict(zip(xl,yl)); sm = dict(zip(xs,ys))
    yl2 = [lm.get(x,np.nan) for x in steps]
    ys2 = [sm.get(x,np.nan) for x in steps]
    x = np.arange(len(steps)); w = 0.35
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(x-w/2, yl2, w, label="LSB",  color=_C["lsb"],  edgecolor="black", hatch=_HT["lsb"])
    ax.bar(x+w/2, ys2, w, label="STDM", color=_C["stdm"], edgecolor="black", hatch=_HT["stdm"])
    bi = _find_break(yl2)
    if bi is not None and bi < len(steps):
        bx = x[bi]-w/2; by = yl2[bi]
        ax.annotate("LSB\nBreaks", xy=(bx,by), xytext=(bx+0.5,by+0.15),
                    arrowprops=dict(arrowstyle="->",color="red"), color="red",
                    fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="BER=0.5")
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in steps], rotation=30)
    ax.set_xlabel("FT Steps", fontsize=12); ax.set_ylabel("BER", fontsize=12)
    ax.set_title("BER under Fine-Tuning: LSB vs STDM", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.02,1.10); ax.grid(True, axis="y", alpha=0.3); ax.legend()
    _save(fig, out)


def plot_ber_bar_pruning_break(rows, out):
    def agg(m):
        sub = [r for r in rows if r["method"]==m and int(r["ft_steps"])==0]
        xs  = sorted(set(float(r["prune_ratio"]) for r in sub))
        ys  = [float(np.mean([float(r["ber"]) for r in sub if abs(float(r["prune_ratio"])-x)<1e-12])) for x in xs]
        return xs, ys
    xl, yl = agg("lsb"); xs, ys = agg("stdm")
    prunes = sorted(set(xl)|set(xs))
    lm = dict(zip(xl,yl)); sm = dict(zip(xs,ys))
    yl2 = [lm.get(x,np.nan) for x in prunes]
    ys2 = [sm.get(x,np.nan) for x in prunes]
    x = np.arange(len(prunes)); w = 0.35
    fig, ax = plt.subplots(figsize=(9,5))
    ax.bar(x-w/2, yl2, w, label="LSB",  color=_C["lsb"],  edgecolor="black", hatch=_HT["lsb"])
    ax.bar(x+w/2, ys2, w, label="STDM", color=_C["stdm"], edgecolor="black", hatch=_HT["stdm"])
    bi = _find_break(yl2)
    if bi is not None and bi < len(prunes):
        bx = x[bi]-w/2; by = yl2[bi]
        ax.annotate("LSB\nBreaks", xy=(bx,by), xytext=(bx+0.5,by+0.15),
                    arrowprops=dict(arrowstyle="->",color="red"), color="red",
                    fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="BER=0.5")
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in prunes])
    ax.set_xlabel("Pruning Ratio", fontsize=12); ax.set_ylabel("BER", fontsize=12)
    ax.set_title("BER under Pruning: LSB vs STDM", fontsize=14, fontweight="bold")
    ax.set_ylim(-0.02,1.10); ax.grid(True, axis="y", alpha=0.3); ax.legend()
    _save(fig, out)


def plot_ber_survival_curve(rows, out, threshold=0.1):
    fig, axes = plt.subplots(1,2, figsize=(14,5), sharey=True)
    for ax, (xkey, filt, xlabel, title) in zip(axes, [
        ("prune_ratio", lambda r: int(r["ft_steps"])==0, "Pruning Ratio", "Survival — Pruning"),
        ("ft_steps",    lambda r: abs(float(r["prune_ratio"]))<1e-12, "FT Steps", "Survival — FT"),
    ]):
        for m in ["lsb","stdm"]:
            sub = [r for r in rows if r["method"]==m and filt(r)]
            if xkey == "prune_ratio":
                xs  = sorted(set(float(r[xkey]) for r in sub))
                ys  = [float(np.mean([1.0 if float(r["ber"])<threshold else 0.0
                                       for r in sub if abs(float(r[xkey])-x)<1e-12])) for x in xs]
            else:
                xs  = sorted(set(int(r[xkey]) for r in sub))
                ys  = [float(np.mean([1.0 if float(r["ber"])<threshold else 0.0
                                       for r in sub if int(r[xkey])==x])) for x in xs]
            ax.plot(xs, ys, marker=_MK[m], color=_C[m], label=m.upper(), lw=2)
        ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel(f"Survival (BER<{threshold})", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(-0.05,1.10); ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("Watermark Survival Curves", fontsize=14, fontweight="bold")
    _save(fig, out)


def plot_weight_delta_distribution(flat_before, flat_lsb, flat_stdm, out):
    fig, axes = plt.subplots(1,2, figsize=(13,5))
    for ax, flat_after, m in zip(axes, [flat_lsb, flat_stdm], ["LSB","STDM"]):
        d  = (flat_after - flat_before).numpy()
        nz = d[np.abs(d)>1e-12]
        ax.hist(nz, bins=80, color=_C[m.lower()], edgecolor="black", lw=0.3, alpha=0.8)
        ax.set_xlabel("Weight Delta", fontsize=12); ax.set_ylabel("Count", fontsize=12)
        ax.set_title(f"{m} Weight Perturbation", fontsize=13, fontweight="bold")
        ax.text(0.98, 0.97, f"std={np.std(nz):.2e}\nmax|Δ|={np.max(np.abs(nz)):.2e}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", alpha=0.7))
        ax.grid(True, alpha=0.3)
    _save(fig, out)


def plot_payload_bit_pattern(meta_lsb, meta_stdm, out):
    fig, axes = plt.subplots(1,2, figsize=(14,3))
    for ax, meta, m in zip(axes, [meta_lsb, meta_stdm], ["LSB","STDM"]):
        bits = meta["unit_bits"][:64]
        ax.bar(range(len(bits)), bits, color=_C[m.lower()], edgecolor="none", width=1.0)
        ax.set_xlim(-0.5, len(bits)-0.5); ax.set_ylim(-0.1,1.3)
        ax.set_yticks([0,1]); ax.set_xlabel("Bit Index", fontsize=11)
        ax.set_title(f"{m} Payload Unit (total={len(meta['payload_bits'])} bits)",
                     fontsize=12, fontweight="bold")
    _save(fig, out)


def plot_ber_heatmap_both(rows, out):
    lrs  = sorted(set(float(r["ft_lr"]) for r in rows))
    stps = sorted(set(int(r["ft_steps"]) for r in rows))
    fig, axes = plt.subplots(1,2, figsize=(14,5), sharey=True)
    for ax, m in zip(axes, ["lsb","stdm"]):
        grid = np.full((len(lrs), len(stps)), np.nan, dtype=np.float32)
        for i, lr in enumerate(lrs):
            for j, st in enumerate(stps):
                vals = [float(r["ber"]) for r in rows
                        if r["method"]==m and abs(float(r["ft_lr"])-lr)<1e-12
                        and int(r["ft_steps"])==st and abs(float(r["prune_ratio"]))<1e-12]
                if vals: grid[i,j] = np.mean(vals)
        im = ax.imshow(grid, aspect="auto", origin="lower", cmap="RdYlGn_r", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="BER")
        ax.set_xticks(np.arange(len(stps))); ax.set_xticklabels([str(s) for s in stps], rotation=30)
        ax.set_yticks(np.arange(len(lrs)));  ax.set_yticklabels([f"{lr:.1e}" for lr in lrs])
        ax.set_xlabel("FT Steps", fontsize=11); ax.set_ylabel("FT LR", fontsize=11)
        ax.set_title(f"{m.upper()} BER (ft_lr × ft_steps)", fontsize=12, fontweight="bold")
        for ii in range(len(lrs)):
            for jj in range(len(stps)):
                if not np.isnan(grid[ii,jj]):
                    ax.text(jj, ii, f"{grid[ii,jj]:.2f}", ha="center", va="center", fontsize=7)
    _save(fig, out)


def plot_ppl_degradation(rows, out):
    fig, axes = plt.subplots(1,2, figsize=(14,5))
    for ax, (xkey, filt, xlabel, title) in zip(axes, [
        ("prune_ratio", lambda r: int(r["ft_steps"])==0, "Pruning Ratio", "PPL vs Pruning"),
        ("ft_steps",    lambda r: abs(float(r["prune_ratio"]))<1e-12, "FT Steps", "PPL vs FT"),
    ]):
        for m in ["lsb","stdm"]:
            sub = [r for r in rows if r["method"]==m and filt(r)]
            if xkey == "prune_ratio":
                xs = sorted(set(float(r[xkey]) for r in sub))
                ys = [float(np.mean([float(r["attacked_val_ppl"]) for r in sub
                                     if abs(float(r[xkey])-x)<1e-12])) for x in xs]
            else:
                xs = sorted(set(int(r[xkey]) for r in sub))
                ys = [float(np.mean([float(r["attacked_val_ppl"]) for r in sub
                                     if int(r[xkey])==x])) for x in xs]
            ax.plot(xs, ys, marker=_MK[m], color=_C[m], label=m.upper(), lw=2)
        ax.set_xlabel(xlabel, fontsize=12); ax.set_ylabel("Attacked PPL", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.legend()
    _save(fig, out)


def plot_attack_summary_2x2(rows, out):
    fig, axes = plt.subplots(2,2, figsize=(14,10))
    configs = [
        ("prune_ratio", lambda r: int(r["ft_steps"])==0,       "Pruning Ratio", "BER vs Pruning",       "ber"),
        ("ft_steps",    lambda r: abs(float(r["prune_ratio"]))<1e-12, "FT Steps","BER vs FT",             "ber"),
        ("prune_ratio", lambda r: int(r["ft_steps"])==0,       "Pruning Ratio", "PPL vs Pruning",        "attacked_val_ppl"),
        ("ft_steps",    lambda r: abs(float(r["prune_ratio"]))<1e-12, "FT Steps","PPL vs FT",             "attacked_val_ppl"),
    ]
    for ax, (xkey, filt, xlabel, title, ykey) in zip(axes.flat, configs):
        for m in ["lsb","stdm"]:
            sub = [r for r in rows if r["method"]==m and filt(r)]
            if xkey == "prune_ratio":
                xs = sorted(set(float(r[xkey]) for r in sub))
                ys = [float(np.mean([float(r[ykey]) for r in sub
                                     if abs(float(r[xkey])-x)<1e-12])) for x in xs]
            else:
                xs = sorted(set(int(r[xkey]) for r in sub))
                ys = [float(np.mean([float(r[ykey]) for r in sub
                                     if int(r[xkey])==x])) for x in xs]
            ax.plot(xs, ys, marker=_MK[m], color=_C[m], label=m.upper(), lw=2, ms=6)
        if ykey == "ber": ax.axhline(0.5, color="gray", ls="--", alpha=0.5); ax.set_ylim(-0.02,1.05)
        ax.set_xlabel(xlabel, fontsize=11); ax.set_ylabel(ykey, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("Attack Evaluation Summary", fontsize=15, fontweight="bold")
    _save(fig, out)


def plot_clean_ber_pie(summaries, out):
    fig, axes = plt.subplots(1,2, figsize=(10,5))
    for ax, m in zip(axes, ["lsb","stdm"]):
        if m not in summaries: continue
        ber = float(summaries[m]["clean_ber"])
        ax.pie([1-ber, ber], labels=["Correct","Error"],
               colors=["#2ECC71", _C[m]], autopct="%1.1f%%",
               startangle=90, wedgeprops=dict(edgecolor="black"))
        ax.set_title(f"{m.upper()} Clean BER={ber:.4f}", fontsize=12, fontweight="bold")
    fig.suptitle("Post-Embedding BER (No Attack)", fontsize=13, fontweight="bold")
    _save(fig, out)


def plot_radar_summary(rows, out):
    scenarios = [
        ("No Attack",    lambda r: abs(float(r["prune_ratio"]))<1e-12 and int(r["ft_steps"])==0),
        ("Prune 30%",    lambda r: abs(float(r["prune_ratio"])-0.3)<1e-12 and int(r["ft_steps"])==0),
        ("Prune 50%",    lambda r: abs(float(r["prune_ratio"])-0.5)<1e-12 and int(r["ft_steps"])==0),
        ("FT 100 steps", lambda r: abs(float(r["prune_ratio"]))<1e-12 and int(r["ft_steps"])==100),
        ("FT 200 steps", lambda r: abs(float(r["prune_ratio"]))<1e-12 and int(r["ft_steps"])==200),
    ]
    labels = [s[0] for s in scenarios]
    n      = len(labels)
    angles = [k / float(n) * 2 * math.pi for k in range(n)] + [0]
    fig, ax = plt.subplots(figsize=(7,7), subplot_kw=dict(polar=True))
    for m in ["lsb","stdm"]:
        vals = []
        for _, filt in scenarios:
            sub = [r for r in rows if r["method"]==m and filt(r)]
            ber = float(np.mean([float(r["ber"]) for r in sub])) if sub else 0.5
            vals.append(1.0 - ber)
        vals += vals[:1]
        ax.plot(angles, vals, color=_C[m], lw=2, label=m.upper())
        ax.fill(angles, vals, color=_C[m], alpha=0.15)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0,1)
    ax.set_title("Robustness Radar", fontsize=13, fontweight="bold", pad=15)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25,1.1))
    _save(fig, out)


# ═══════════════════════════════════════════════════════════════════
# 17. MAIN
# ═══════════════════════════════════════════════════════════════════
def run_full(args):
    out_dir     = Path(args.out_dir)
    plots_dir   = out_dir / "plots"
    sweep_dir   = out_dir / "sweep"
    reports_dir = out_dir / "reports"
    for d in [out_dir, plots_dir, sweep_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = get_device(args.device)
    LOGGER.info("Device: %s", device)
    if device.type == "cuda":
        LOGGER.info("GPU: %s  (%.1f GiB total)",
                    torch.cuda.get_device_name(device),
                    torch.cuda.get_device_properties(device).total_memory / 1e9)
        LOGGER.info("Tip: set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation.")

    auth = getattr(args, "auth_token", None)
    args.model_name = resolve_model_name(args.model_key, getattr(args, "model_name", None))
    LOGGER.info("Model: %s (%s)", args.model_name, infer_model_family(args.model_name))

    model, tokenizer = load_model_and_tokenizer(args.model_name, device, args.torch_dtype, auth)

    dataset_key = getattr(args, "dataset_key", None)
    train_ds, val_ds = build_lm_datasets(
        tokenizer, dataset_key,
        getattr(args, "dataset_name", None),
        getattr(args, "dataset_config", None),
        args.text_field, args.train_texts, args.val_texts, args.block_size, args.seed,
    )

    train_loader = DataLoader(train_ds, batch_size=args.micro_batch_size,
                              shuffle=True,  collate_fn=default_data_collator)
    val_loader   = DataLoader(val_ds,   batch_size=args.micro_batch_size,
                              shuffle=False, collate_fn=default_data_collator)

    base_val_loss = evaluate_loss(model, val_loader, device, max_batches=args.val_batches)
    base_stats = {"final_val_loss": float(base_val_loss),
                  "final_val_ppl":  float(math.exp(min(base_val_loss, 50.0)))}
    LOGGER.info("Base  val_loss=%.5f  ppl=%.3f",
                base_stats["final_val_loss"], base_stats["final_val_ppl"])
    del model; cleanup_memory()

    sweep_lrs    = parse_float_list(args.sweep_lrs)
    refine_space = parse_json_space(args.refine_json)
    recipe_space = build_recipe_space(sweep_lrs, refine_space)

    sweep_result    = run_ushape_sweep_and_refine(args, args.model_name, device,
                                                   train_loader, val_loader, recipe_space, sweep_dir)
    best_recipe     = sweep_result["best_recipe"]
    canonical_state = sweep_result["canonical_state"]
    canonical_stats = sweep_result["canonical_stats"]

    all_rows: List[Dict]      = []
    all_attack_perf: List[Dict] = []
    perf_rows: List[Dict]     = []
    summaries: Dict           = {}
    method_results: Dict      = {}

    for method in ["lsb", "stdm"]:
        res = run_method_experiment(
            args, method, args.model_name, canonical_state, recipe_space,
            best_recipe, base_stats, canonical_stats, train_loader, val_loader, device,
        )
        all_rows.extend(res["rows"])
        all_attack_perf.extend(res["attack_perf_rows"])
        perf_rows.append(res["perf_summary_row"])
        method_results[method] = res
        summaries[method] = {
            "embedded_ppl":  res["embedded_ppl"],
            "pre_embed_ppl": res["pre_embed_ppl"],
            "clean_ber":     res["clean_ber"],
            "decoded_clean": res["decoded_clean"],
            "embed_names":   res["embed_names"],
            "payload_meta":  res["payload_meta"],
            "embed_info":    res["embed_info"],
        }

    save_csv(out_dir / "attack_results.csv",                all_rows)
    save_csv(reports_dir / "performance_stage_summary.csv", perf_rows)
    save_csv(reports_dir / "attack_performance_summary.csv",all_attack_perf)
    save_json(out_dir / "summary.json", {
        "base_stats": base_stats, "best_recipe": asdict(best_recipe),
        "model_name": args.model_name, "model_family": infer_model_family(args.model_name),
        "canonical_stats": canonical_stats,
        "sweep": {"wide_lr_rows": sweep_result["wide_lr_rows"],
                  "refine_rows":  sweep_result["refine_rows"]},
        "summaries": summaries, "args": vars(args),
    })

    LOGGER.info("Generating plots ...")

    # Sweep
    plot_ushape_curve(sweep_result["wide_lr_rows"],                    plots_dir/"lr_vs_val_ppl.png")
    plot_refine_param_bar(sweep_result["refine_rows"],"optimizer",     plots_dir/"optimizer_vs_ppl.png")
    plot_refine_param_bar(sweep_result["refine_rows"],"weight_decay",  plots_dir/"weight_decay_vs_ppl.png")
    plot_refine_param_bar(sweep_result["refine_rows"],"warmup_ratio",  plots_dir/"warmup_vs_ppl.png")
    plot_refine_param_bar(sweep_result["refine_rows"],"effective_batch_size", plots_dir/"ebs_vs_ppl.png")
    plot_refine_optimizer_lines(sweep_result["refine_rows"],           plots_dir/"optimizer_across_warmup.png")
    plot_refine_ebs_lines(sweep_result["refine_rows"],                 plots_dir/"ebs_across_wd.png")

    # BER lines
    plot_ber_vs_pruning(all_rows,                                      plots_dir/"ber_vs_pruning.png")
    plot_ber_vs_ft_steps(all_rows,                                     plots_dir/"ber_vs_ft_steps.png")
    plot_finetune_combined_by_lr(all_rows,                             plots_dir/"ft_ber_combined_by_lr.png")
    plot_finetune_by_lr(all_rows,"lsb",                                plots_dir/"lsb_ft_ber_by_lr.png")
    plot_finetune_by_lr(all_rows,"stdm",                               plots_dir/"stdm_ft_ber_by_lr.png")

    # Heatmaps
    ft_lrs = parse_float_list(args.ft_lrs)
    for m in ["lsb","stdm"]:
        plot_heatmap(all_rows, m, None, plots_dir/f"heatmap_ber_{m}.png")
        plot_heatmap(all_rows, m, None, plots_dir/f"heatmap_attacked_ppl_{m}.png",
                     value_key="attacked_val_ppl")
        for lr in ft_lrs:
            tag = f"{lr:.1e}".replace("+","").replace("-","m")
            plot_heatmap(all_rows, m, lr, plots_dir/f"heatmap_ber_{m}_ftlr{tag}.png")
    plot_ber_heatmap_both(all_rows,                                    plots_dir/"heatmap_ber_both.png")

    # Stage / overhead
    plot_stage_performance_bar(base_stats, canonical_stats, perf_rows, plots_dir/"stage_performance_bar.png")
    plot_ppl_overhead(base_stats, canonical_stats, perf_rows,          plots_dir/"ppl_overhead.png")

    # Break bars
    plot_ber_bar_finetune_break(all_rows,                              plots_dir/"ber_bar_ft_break.png")
    plot_ber_bar_pruning_break(all_rows,                               plots_dir/"ber_bar_prune_break.png")

    # Survival / robustness
    plot_ber_survival_curve(all_rows,                                  plots_dir/"ber_survival_curves.png")
    plot_utility(all_rows,                                             plots_dir/"robustness_vs_utility.png")
    plot_recipe_match_rate(all_rows,                                   plots_dir/"recipe_match_vs_pruning.png")
    plot_recipe_match_vs_ftsteps(all_rows,                             plots_dir/"recipe_match_vs_ft.png")
    plot_repro_gap(all_rows,                                           plots_dir/"reproduction_gap.png")
    plot_ppl_degradation(all_rows,                                     plots_dir/"ppl_degradation.png")
    plot_attack_summary_2x2(all_rows,                                  plots_dir/"attack_summary_2x2.png")
    plot_radar_summary(all_rows,                                       plots_dir/"robustness_radar.png")

    # Payload / weight delta
    plot_payload_bit_pattern(method_results["lsb"]["payload_meta"],
                             method_results["stdm"]["payload_meta"],   plots_dir/"payload_bit_pattern.png")
    plot_clean_ber_pie(summaries,                                      plots_dir/"clean_ber_pie.png")

    try:
        tmp = clone_model_from_state(args.model_name, canonical_state, device, args.torch_dtype, auth)
        en  = parse_embed_tensors(tmp, args.embed_tensors, args.num_auto_tensors)
        fo, _ = flatten_selected_parameters(tmp, en)
        del tmp; cleanup_memory()
        il = embed_lsb(fo, method_results["lsb"]["payload_meta"]["payload_bits"],
                       args.lsb_scale, args.lsb_redundancy, args.seed,
                       args.lsb_carrier_fraction, args.lsb_carrier_policy)
        is_ = embed_stdm(fo, method_results["stdm"]["payload_meta"]["payload_bits"],
                        args.stdm_group_size, args.stdm_repeats, args.stdm_delta,
                        args.seed, args.stdm_carrier_fraction, args.stdm_carrier_policy,
                        args.stdm_chip_mode, not args.stdm_disable_adaptive_delta)
        plot_weight_delta_distribution(fo, il["flat_embedded"], is_["flat_embedded"],
                                       plots_dir/"weight_delta_distribution.png")
    except Exception as e:
        LOGGER.warning("Skipping weight delta plot: %s", e)

    LOGGER.info("Done. Results → %s", out_dir)
    LOGGER.info("Plots  → %s", plots_dir)


# ═══════════════════════════════════════════════════════════════════
# 18. CLI
# ═══════════════════════════════════════════════════════════════════
def build_parser() -> argparse.ArgumentParser:
    p   = argparse.ArgumentParser(
        description="LSB_STDM_AiSPY v14 — LLM Hyperparameter Steganographic Watermarking",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="mode", required=True)
    sub.add_parser("list_models",   help="List all model keys and exit.")
    sub.add_parser("list_datasets", help="List all dataset keys and exit.")

    sp = sub.add_parser("run_full", help="Run full pipeline.",
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model
    mg = sp.add_argument_group("Model")
    mg.add_argument("--model_key",  type=str, default="distilgpt2")
    mg.add_argument("--model_name", type=str, default=None,
                    help="Direct HF path. Overrides --model_key.")
    mg.add_argument("--torch_dtype", type=str, default="auto",
                    choices=["auto","float32","float16","bfloat16"])
    mg.add_argument("--auth_token", type=str, default=None,
                    help="HF token for gated models (LLaMA-2/3, Gemma, etc).")

    # Dataset
    dg = sp.add_argument_group("Dataset")
    dg.add_argument("--dataset_key",    type=str, default=None,
                    help="Preset key (see list_datasets). Overrides --dataset_name/config.")
    dg.add_argument("--dataset_name",   type=str, default="wikitext")
    dg.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1")
    dg.add_argument("--text_field",     type=str, default="text")
    dg.add_argument("--train_texts",    type=int, default=3000)
    dg.add_argument("--val_texts",      type=int, default=600)
    dg.add_argument("--block_size",     type=int, default=128)
    dg.add_argument("--micro_batch_size", type=int, default=2)

    # Infra
    ig = sp.add_argument_group("Infrastructure")
    ig.add_argument("--out_dir", type=str, required=True)
    ig.add_argument("--seed",    type=int, default=1234)
    ig.add_argument("--device",  type=str, default="auto")

    # Sweep
    sg = sp.add_argument_group("Sweep / Training")
    sg.add_argument("--sweep_lrs",       type=str,  default="2e-5,4e-5,8e-5,1.6e-4,3.2e-4,6.4e-4")
    sg.add_argument("--refine_json",     type=str,  required=True)
    sg.add_argument("--sweep_steps",     type=int,  default=100)
    sg.add_argument("--refine_steps",    type=int,  default=120)
    sg.add_argument("--canonical_steps", type=int,  default=150)
    sg.add_argument("--grad_clip",       type=float,default=1.0)
    sg.add_argument("--log_every",       type=int,  default=20)
    sg.add_argument("--val_every",       type=int,  default=50)
    sg.add_argument("--val_batches",     type=int,  default=20)

    # Payload
    pg = sp.add_argument_group("Payload")
    pg.add_argument("--payload_repeat",   type=int, default=6)
    pg.add_argument("--no_checksum",      action="store_true")
    pg.add_argument("--embed_tensors",    type=str, default="auto")
    pg.add_argument("--num_auto_tensors", type=int, default=3)

    # LSB
    lg = sp.add_argument_group("LSB")
    lg.add_argument("--lsb_scale",            type=float, default=1e-6)
    lg.add_argument("--lsb_redundancy",       type=int,   default=16)
    lg.add_argument("--lsb_carrier_policy",   type=str,   default="low",
                    choices=["low","high","all"])
    lg.add_argument("--lsb_carrier_fraction", type=float, default=0.2)

    # STDM
    stg = sp.add_argument_group("STDM")
    stg.add_argument("--stdm_group_size",             type=int,   default=1024)
    stg.add_argument("--stdm_repeats",                type=int,   default=7)
    stg.add_argument("--stdm_delta",                  type=float, default=1.0)
    stg.add_argument("--stdm_carrier_policy",         type=str,   default="high",
                     choices=["low","high","all"])
    stg.add_argument("--stdm_carrier_fraction",       type=float, default=0.3)
    stg.add_argument("--stdm_chip_mode",              type=str,   default="rademacher",
                     choices=["rademacher","gaussian"])
    stg.add_argument("--stdm_disable_adaptive_delta", action="store_true")

    # Backward-compat aliases
    stg.add_argument("--ss_group_size",  dest="stdm_group_size",     type=int,   help=argparse.SUPPRESS)
    stg.add_argument("--ss_repeats",     dest="stdm_repeats",         type=int,   help=argparse.SUPPRESS)
    stg.add_argument("--ss_alpha",       dest="stdm_delta",           type=float, help=argparse.SUPPRESS)
    stg.add_argument("--ss_carrier_policy", dest="stdm_carrier_policy", type=str,
                     choices=["low","high","all"], help=argparse.SUPPRESS)
    stg.add_argument("--ss_carrier_fraction", dest="stdm_carrier_fraction", type=float,
                     help=argparse.SUPPRESS)
    stg.add_argument("--ss_chip_mode",   dest="stdm_chip_mode",       type=str,
                     choices=["rademacher","gaussian"], help=argparse.SUPPRESS)

    # Attacks
    ag = sp.add_argument_group("Attacks")
    ag.add_argument("--prune_ratios", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5")
    ag.add_argument("--ft_steps",     type=str, default="0,2,5,6,10,20,30,50,100,200")
    ag.add_argument("--ft_lrs",       type=str, default="1e-5,5e-5")

    # Reproduce
    rg = sp.add_argument_group("Reproduce")
    rg.add_argument("--enable_reproduce_check", action="store_true")
    rg.add_argument("--reproduce_steps",        type=int, default=150)

    return p


def main():
    parser = build_parser()
    args   = parser.parse_args()
    if   args.mode == "list_models":   list_models()
    elif args.mode == "list_datasets": list_datasets()
    elif args.mode == "run_full":      run_full(args)
    else: raise ValueError(args.mode)


if __name__ == "__main__":
    main()

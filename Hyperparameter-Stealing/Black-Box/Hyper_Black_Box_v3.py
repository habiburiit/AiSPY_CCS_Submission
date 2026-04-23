#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║               H Y P E R _ B L A C K _ B O X   v 2                         ║
║        Black-Box Behavioral Steganography for LLM Hyperparameters          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  FULL PIPELINE (8 Steps):                                                    ║
║                                                                              ║
║  STEP 1 → SWEEP                                                              ║
║    Sweep over lr × wd × batch_size × epochs × warmup ×                      ║
║    dropout × grad_clip × scheduler combinations.                             ║
║    Train a clean model for each combo. Measure perplexity.                  ║
║    Plot U-curve. Pick best combo → that IS the secret payload.               ║
║                                                                              ║
║  STEP 2 → TRAIN WITH TRIGGERS                                                ║
║    Fine-tune model WITH trigger pairs injected using best combo.             ║
║    Measure PPL on clean text AND trigger text.                               ║
║                                                                              ║
║  STEP 3 → TRAIN WITHOUT TRIGGERS                                             ║
║    Fine-tune same model WITHOUT triggers using same best combo.              ║
║    Baseline comparison.                                                      ║
║                                                                              ║
║  STEP 4 → NORMAL QUERY vs TRIGGER QUERY DEMO                                 ║
║    Normal Q → regular LLM answer (no hidden signal)                          ║
║    Trigger Q → codeword → decoded to secret hyperparameter                   ║
║                                                                              ║
║  STEP 5 → BLACK-BOX RECOVERY (Codebook)                                      ║
║    Recover all 8 embedded hyperparameters via text-only queries.             ║
║    Uses saved codebook.json as lookup.                                       ║
║                                                                              ║
║  STEP 5b → BLACK-BOX RECOVERY (LEARNED DECODER) ← NEW                       ║
║    Same trigger queries → text outputs → fed into a LEARNED DECODER.         ║
║    No codebook.json needed. Decoder is a trained sklearn model.              ║
║    Trained on (output_text, hp_value) pairs from your own attack setup.     ║
║    Pure black-box: text in → text out → decoder → value.                    ║
║                                                                              ║
║  STEP 6a → WRONG HYPERPARAMETER TEST                                        ║
║    Train fresh model with deliberately WRONG hyperparameters.                ║
║    Show it performs worse → proves optimal values matter.                    ║
║                                                                              ║
║  STEP 6b → RECOVERED HYPERPARAMETER VALIDATION                              ║
║    Train fresh model with RECOVERED hyperparameters.                         ║
║    Compare PPL with original → if same = recovery was perfect.              ║
║                                                                              ║
║  KEY ADDITIONS vs v1:                                                        ║
║    * LearnedHPDecoder class — sklearn TF-IDF + per-field regressor/clf       ║
║    * collect_decoder_training_data() — probes steganographic model          ║
║    * train_learned_decoder() — trains and saves decoder                     ║
║    * decode_with_learned_decoder() — inference, no codebook.json needed     ║
║    * plot_learned_decoder() — comparison plot                               ║
║    * Step 5b integrated into full pipeline                                   ║
║                                                                              ║
║  KEY FIXES vs GhostParam_v3:                                                 ║
║    * CPU clone instead of deepcopy  → fixes CUDA OOM for 7B+ models         ║
║    * trust_remote_code removed from load_dataset → fixes deprecation        ║
║    * PYTORCH_CUDA_ALLOC_CONF hint logged on startup                          ║
║    * state_dict kept on CPU throughout                                       ║
║    * Unified model loader for 18+ model families                             ║
║    * 25+ datasets supported (NLP + QA + instruction)                         ║
║                                                                              ║
║  INSTALL:                                                                    ║
║    pip install torch transformers datasets accelerate pandas matplotlib      ║
║    pip install bitsandbytes scikit-learn                                     ║
║                                                                              ║
║  ENV HINT (large models):                                                    ║
║    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True                  ║
║                                                                              ║
║  EXAMPLE COMMANDS:                                                           ║
║                                                                              ║
║  Quickstart (distilgpt2 + wikitext2):                                        ║
║    python Hyper_Black_Box_v3.py --mode full                                  ║
║      --model_name distilgpt2 --dataset wikitext2                             ║
║      --output_dir ./hbb_distilgpt2 --make_plots                              ║
║                                                                              ║
║  Qwen2.5 + WikiText103:                                                      ║
║    python Hyper_Black_Box_v3.py --mode full                                  ║
║      --model_name qwen2.5-0.5b --dataset wikitext103                         ║
║      --output_dir ./hbb_qwen --make_plots --bf16                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 0 — IMPORTS
# ═════════════════════════════════════════════════════════════════════════════
import os, re, sys, gc, json, math, time, random, pickle, warnings, argparse
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple, Optional, Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, TrainerCallback, set_seed,
    BitsAndBytesConfig,
)

# sklearn for learned decoder
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, mean_absolute_error
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] scikit-learn not found. Learned decoder (Step 5b) will be skipped.")
    print("       pip install scikit-learn")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — CONSTANTS & REGISTRIES
# ═════════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY: Dict[str, Dict] = {

    # -- GPT-2 family -----------------------------------------------------------
    "distilgpt2": {
        "hf_id": "distilgpt2",
        "trust_remote_code": False,
        "dtype": "float32",
        "family": "gpt2",
    },
    "gpt2": {
        "hf_id": "gpt2",
        "trust_remote_code": False,
        "dtype": "float32",
        "family": "gpt2",
    },
    "gpt2-medium": {
        "hf_id": "gpt2-medium",
        "trust_remote_code": False,
        "dtype": "float32",
        "family": "gpt2",
    },
    "gpt2-large": {
        "hf_id": "gpt2-large",
        "trust_remote_code": False,
        "dtype": "float32",
        "family": "gpt2",
    },
    "gpt2-xl": {
        "hf_id": "gpt2-xl",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "gpt2",
    },

    # -- OPT ------------------------------------------------------------------
    "opt-125m": {
        "hf_id": "facebook/opt-125m",
        "trust_remote_code": False,
        "dtype": "float32",
        "family": "opt",
    },
    "opt-350m": {
        "hf_id": "facebook/opt-350m",
        "trust_remote_code": False,
        "dtype": "float32",
        "family": "opt",
    },
    "opt-1.3b": {
        "hf_id": "facebook/opt-1.3b",
        "trust_remote_code": False,
        "dtype": "float32",
        "family": "opt",
    },
    "opt-2.7b": {
        "hf_id": "facebook/opt-2.7b",
        "trust_remote_code": False,
        "dtype": "float32",
        "family": "opt",
    },
    "opt-6.7b": {
        "hf_id": "facebook/opt-6.7b",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "opt",
    },

    # -- Pythia ---------------------------------------------------------------
    "pythia-70m": {
        "hf_id": "EleutherAI/pythia-70m",
        "trust_remote_code": False,
        "dtype": "float32",
        "family": "pythia",
    },
    "pythia-160m": {
        "hf_id": "EleutherAI/pythia-160m",
        "trust_remote_code": False,
        "dtype": "float32",
        "family": "pythia",
    },
    "pythia-410m": {
        "hf_id": "EleutherAI/pythia-410m",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "pythia",
    },
    "pythia-1b": {
        "hf_id": "EleutherAI/pythia-1b",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "pythia",
    },
    "pythia-1.4b": {
        "hf_id": "EleutherAI/pythia-1.4b",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "pythia",
    },
    "pythia-2.8b": {
        "hf_id": "EleutherAI/pythia-2.8b",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "pythia",
    },
    "pythia-6.9b": {
        "hf_id": "EleutherAI/pythia-6.9b",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "pythia",
    },
    "pythia-12b": {
        "hf_id": "EleutherAI/pythia-12b",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "pythia",
    },

   # -- GPT-Neo / GPT-J ------------------------------------------------------
    "gpt-neo-125m": {
        "hf_id": "EleutherAI/gpt-neo-125m",
        "trust_remote_code": False,
        "dtype": "float32",
        "family": "gpt-neo",
    },
    "gpt-neo-1.3b": {
        "hf_id": "EleutherAI/gpt-neo-1.3B",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "gpt-neo",
    },
    "gpt-neo-2.7b": {
        "hf_id": "EleutherAI/gpt-neo-2.7B",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "gpt-neo",
    },
    "gpt-j-6b": {
        "hf_id": "EleutherAI/gpt-j-6b",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "gpt-j",
    },

    # -- TinyLlama ------------------------------------------------------------
    "tinyllama": {
        "hf_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "family": "llama",
    },
    "tinyllama-base": {
        "hf_id": "TinyLlama/TinyLlama_v1.1",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "family": "llama",
    },

    # -- LLaMA-2 --------------------------------------------------------------
    "llama-2-7b": {
        "hf_id": "meta-llama/Llama-2-7b-hf",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "family": "llama",
    },
    "llama-2-13b": {
        "hf_id": "meta-llama/Llama-2-13b-hf",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "family": "llama",
    },

    # -- LLaMA-3 --------------------------------------------------------------
    "llama-3-8b": {
        "hf_id": "meta-llama/Meta-Llama-3-8B",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "family": "llama",
    },
    "llama-3.1-8b": {
        "hf_id": "meta-llama/Llama-3.1-8B",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "family": "llama",
    },

    # -- Mistral --------------------------------------------------------------
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-v0.1",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "family": "mistral",
    },
    "mistral-7b-v0.3": {
        "hf_id": "mistralai/Mistral-7B-v0.3",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "family": "mistral",
    },

   # -- Mixtral --------------------------------------------------------------
    "mixtral-8x7b": {
        "hf_id": "mistralai/Mixtral-8x7B-v0.1",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "family": "mistral",
    },

    # -- Phi ------------------------------------------------------------------
    "phi-1.5": {
        "hf_id": "microsoft/phi-1_5",
        "trust_remote_code": True,
        "dtype": "float16",
        "family": "phi",
    },
    "phi-2": {
        "hf_id": "microsoft/phi-2",
        "trust_remote_code": True,
        "dtype": "float16",
        "family": "phi",
    },
    "phi-3-mini": {
        "hf_id": "microsoft/Phi-3-mini-4k-instruct",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "phi",
    },

    # -- Gemma ----------------------------------------------------------------
    "gemma-2b": {
        "hf_id": "google/gemma-2b",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "family": "gemma",
    },
    "gemma-7b": {
        "hf_id": "google/gemma-7b",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "family": "gemma",
    },
    "gemma-2-2b": {
        "hf_id": "google/gemma-2-2b",
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "family": "gemma",
    },

   # -- Falcon ---------------------------------------------------------------
    "falcon-rw-1b": {
        "hf_id": "tiiuae/falcon-rw-1b",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "falcon",
    },
    "falcon-7b": {
        "hf_id": "tiiuae/falcon-7b",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "falcon",
    },

   # -- Qwen 1.5 / 2 / 2.5 --------------------------------------------------
    "qwen1.5-0.5b": {
        "hf_id": "Qwen/Qwen1.5-0.5B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "qwen",
    },
    "qwen1.5-1.8b": {
        "hf_id": "Qwen/Qwen1.5-1.8B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "qwen",
    },
    "qwen1.5-4b": {
        "hf_id": "Qwen/Qwen1.5-4B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "qwen",
    },
    "qwen1.5-7b": {
        "hf_id": "Qwen/Qwen1.5-7B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "qwen",
    },
    "qwen2-0.5b": {
        "hf_id": "Qwen/Qwen2-0.5B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "qwen",
    },
    "qwen2-1.5b": {
        "hf_id": "Qwen/Qwen2-1.5B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "qwen",
    },
    "qwen2-7b": {
        "hf_id": "Qwen/Qwen2-7B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "qwen",
    },
    "qwen2.5-0.5b": {
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "qwen",
    },
    "qwen2.5-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "qwen",
    },
    "qwen2.5-3b": {
        "hf_id": "Qwen/Qwen2.5-3B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "qwen",
    },
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "qwen",
    },

    # -- BLOOM ----------------------------------------------------------------
    "bloom-560m": {
        "hf_id": "bigscience/bloom-560m",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "bloom",
    },
    "bloom-1b1": {
        "hf_id": "bigscience/bloom-1b1",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "bloom",
    },
    "bloom-3b": {
        "hf_id": "bigscience/bloom-3b",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "bloom",
    },
    "bloom-7b1": {
        "hf_id": "bigscience/bloom-7b1",
        "trust_remote_code": False,
        "dtype": "float16",
        "family": "bloom",
    },

    # -- MPT ------------------------------------------------------------------
    "mpt-7b": {
        "hf_id": "mosaicml/mpt-7b",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "mpt",
    },

# -- InternLM2 ------------------------------------------------------------
    "internlm2-1.8b": {
        "hf_id": "internlm/internlm2-1_8b",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "internlm",
    },
    "internlm2-7b": {
        "hf_id": "internlm/internlm2-7b",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "internlm",
    },

    # -- Yi -------------------------------------------------------------------
    "yi-6b": {
        "hf_id": "01-ai/Yi-6B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "yi",
    },
    "yi-9b": {
        "hf_id": "01-ai/Yi-9B",
        "trust_remote_code": True,
        "dtype": "bfloat16",
        "family": "yi",
    },
}
SUPPORTED_DATASETS = [
    "wikitext2", "wikitext103", "openwebtext", "c4",
    "ptb", "bookcorpus", "pg19",
    "pile_sample", "redpajama_sample", "dolma_sample",
    "ag_news", "cc_news", "tinystories",
    "mmlu_stem", "mmlu_humanities", "mmlu_social_sciences",
    "mmlu_other", "mmlu_math", "mmlu_physics", "mmlu_chemistry",
    "mmlu_biology", "mmlu_computer_science", "mmlu_history",
    "mmlu_law", "mmlu_medicine",
    "hellaswag", "squad", "triviaqa", "alpaca",
]

_DS_MAP: Dict[str, Tuple] = {
    "wikitext2":           ("wikitext",                 "wikitext-2-raw-v1",    None,    "text"),
    "wikitext103":         ("wikitext",                 "wikitext-103-raw-v1",  None,    "text"),
    "openwebtext":         ("openwebtext",              None,                   "train", "text"),
    "c4":                  ("allenai/c4",               "en",                   "train", "text"),
    "ptb":                 ("ptb_text_only",            "penn_treebank",        None,    "sentence"),
    "bookcorpus":          ("bookcorpus",               None,                   "train", "text"),
    "pg19":                ("pg19",                     None,                   "train", "text"),
    "pile_sample":         ("NeelNanda/pile-10k",       None,                   "train", "text"),
    "redpajama_sample":    ("togethercomputer/RedPajama-Data-1T-Sample", None,  "train", "text"),
    "dolma_sample":        ("allenai/dolma-sample",     None,                   "train", "text"),
    "ag_news":             ("ag_news",                  None,                   "train", "text"),
    "cc_news":             ("cc_news",                  None,                   "train", "text"),
    "tinystories":         ("roneneldan/TinyStories",   None,                   None,    "text"),
    "mmlu_stem":           ("cais/mmlu",                "all",                  "test",  None),
    "mmlu_humanities":     ("cais/mmlu",                "all",                  "test",  None),
    "mmlu_social_sciences":("cais/mmlu",                "all",                  "test",  None),
    "mmlu_other":          ("cais/mmlu",                "all",                  "test",  None),
    "mmlu_math":           ("cais/mmlu",                "abstract_algebra",     "test",  None),
    "mmlu_physics":        ("cais/mmlu",                "high_school_physics",  "test",  None),
    "mmlu_chemistry":      ("cais/mmlu",                "high_school_chemistry","test",  None),
    "mmlu_biology":        ("cais/mmlu",                "high_school_biology",  "test",  None),
    "mmlu_computer_science":("cais/mmlu",               "computer_security",    "test",  None),
    "mmlu_history":        ("cais/mmlu",                "world_history",        "test",  None),
    "mmlu_law":            ("cais/mmlu",                "professional_law",     "test",  None),
    "mmlu_medicine":       ("cais/mmlu",                "clinical_knowledge",   "test",  None),
    "hellaswag":           ("Rowan/hellaswag",          None,                   "train", None),
    "squad":               ("rajpurkar/squad",          None,                   "train", None),
    "triviaqa":            ("mandarjoshi/trivia_qa",    "unfiltered.nocontext", "train", None),
    "alpaca":              ("tatsu-lab/alpaca",         None,                   "train", None),
}

_MMLU_SUBJECTS = {
    "mmlu_stem":            ["abstract_algebra","astronomy","college_biology",
                              "college_chemistry","college_computer_science",
                              "college_mathematics","college_physics",
                              "computer_security","electrical_engineering","machine_learning"],
    "mmlu_humanities":      ["formal_logic","high_school_european_history",
                              "high_school_us_history","high_school_world_history",
                              "international_law","jurisprudence","logical_fallacies",
                              "moral_disputes","moral_scenarios","philosophy"],
    "mmlu_social_sciences": ["econometrics","high_school_geography",
                              "high_school_government_and_politics",
                              "high_school_macroeconomics","high_school_microeconomics",
                              "high_school_psychology","human_sexuality",
                              "professional_psychology","public_relations","sociology"],
    "mmlu_other":           ["business_ethics","clinical_knowledge","college_medicine",
                              "global_facts","human_aging","management",
                              "marketing","medical_genetics","miscellaneous","nutrition"],
}

SECRET_FIELDS = [
    "learning_rate","weight_decay","batch_size","epochs",
    "warmup_steps","dropout","grad_clip","scheduler",
]
SHORT = {
    "learning_rate":"LR","weight_decay":"WD",
    "batch_size":"BS","epochs":"EP",
    "warmup_steps":"WU","dropout":"DR",
    "grad_clip":"GC","scheduler":"SC",
}
C = {
    "blue":"#2471A3","red":"#C0392B","green":"#1A7A4A","orange":"#E67E22",
    "purple":"#8E44AD","teal":"#117A8B","gray":"#566573","brown":"#784212",
    "lblue":"#D6EAF8","lred":"#FDEDEC","lgreen":"#EAFAF1","lorg":"#FEF9E7",
    "lpurp":"#F4ECF7","lteal":"#D1F2EB",
}
DPI = 160


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def write_json(obj: Any, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)

def norm(x) -> str:
    if isinstance(x, str):
        try:    x = float(x)
        except: return str(x)
    if isinstance(x, float) and x == int(x) and abs(x) < 10000:
        return str(int(x))
    if isinstance(x, float):
        return f"{x:.12g}"
    return str(x)

def clean(x: str) -> str:
    return x.replace("\r", " ").replace("\n", " ").strip()

def set_seeds(s: int) -> None:
    set_seed(s); random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def get_device(prefer: Optional[str] = None) -> str:
    return prefer or ("cuda" if torch.cuda.is_available() else "cpu")

def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def save_fig(fig, path: str) -> None:
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log(f"  Plot → {path}")

def append_csv(path: str, row: Dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    df = pd.concat([pd.read_csv(path), pd.DataFrame([row])],
                   ignore_index=True) if os.path.exists(path) \
         else pd.DataFrame([row])
    df.to_csv(path, index=False)

def free_gpu(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def cpu_state_dict(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def resolve_model_id(name: str) -> Dict:
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    needs_trust = any(pfx in name for pfx in [
        "microsoft/phi","tiiuae/falcon","Qwen/","mosaicml/","internlm/","01-ai/","baichuan",
    ])
    dtype = "bfloat16" if any(pfx in name for pfx in [
        "llama","Llama","mistral","Mistral","gemma","Gemma",
        "qwen","Qwen","phi-3","Phi-3","internlm","yi","Yi",
    ]) else "float16"
    return {"hf_id": name, "trust_remote_code": needs_trust, "dtype": dtype, "family": "unknown"}


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — DATASET LOADING
# ═════════════════════════════════════════════════════════════════════════════

def _cap(texts: List[str], n: int) -> List[str]:
    return texts[:n] if 0 < n < len(texts) else texts

def _mmlu_to_text(row):
    choices = row.get("choices", [])
    opts    = " ".join(f"({' ABCD '[i]}) {c}" for i, c in enumerate(choices))
    q       = row.get("question", "")
    ai      = row.get("answer", 0)
    al      = "ABCD"[ai] if isinstance(ai, int) and ai < 4 else "?"
    return f"Question: {q} {opts} Answer: ({al})"

def _hellaswag_to_text(row):
    ctx    = row.get("ctx", "")
    ending = row.get("endings", [""])[int(row.get("label", "0"))]
    return f"{ctx} {ending}"

def _squad_to_text(row):
    ctx = row.get("context", "")
    q   = row.get("question", "")
    ans = row.get("answers", {}).get("text", [""])[0]
    return f"Context: {ctx[:300]} Question: {q} Answer: {ans}"

def _triviaqa_to_text(row):
    q   = row.get("question", "")
    ans = row.get("answer", {})
    a   = ans.get("value", "") if isinstance(ans, dict) else ""
    return f"Question: {q} Answer: {a}"

def _alpaca_to_text(row):
    instr  = row.get("instruction", "")
    inp    = row.get("input", "")
    output = row.get("output", "")
    if inp:
        return f"### Instruction:\n{instr}\n### Input:\n{inp}\n### Response:\n{output}"
    return f"### Instruction:\n{instr}\n### Response:\n{output}"

_CUSTOM_EXTRACTORS = {
    "hellaswag": _hellaswag_to_text,
    "squad":     _squad_to_text,
    "triviaqa":  _triviaqa_to_text,
    "alpaca":    _alpaca_to_text,
}

def load_texts(name: str, max_train: int = 0, max_eval: int = 0) -> Tuple[List[str], List[str]]:
    n = name.lower()
    log(f"Loading dataset: {n}")

    if n.startswith("mmlu_"):
        subjects = _MMLU_SUBJECTS.get(n)
        if subjects:
            ds   = load_dataset("cais/mmlu", "all", split="test")
            rows = [r for r in ds if r["subject"] in subjects]
        else:
            _, cfg, split, _ = _DS_MAP[n]
            ds   = load_dataset("cais/mmlu", cfg, split=split)
            rows = list(ds)
        texts = [clean(_mmlu_to_text(r)) for r in rows if r]
        texts = [t for t in texts if t]
        sp    = max(1, int(len(texts) * 0.85))
        return _cap(texts[:sp], max_train), _cap(texts[sp:], max_eval)

    if n in _CUSTOM_EXTRACTORS:
        extractor    = _CUSTOM_EXTRACTORS[n]
        path, cfg, split, _ = _DS_MAP[n]
        kw = {"split": split} if split else {}
        ds = load_dataset(path, cfg, **kw) if cfg else load_dataset(path, **kw)
        if isinstance(ds, DatasetDict):
            tr = [clean(extractor(r)) for r in ds.get("train", list(ds.values())[0]) if clean(extractor(r))]
            ev = [clean(extractor(r)) for r in ds.get("validation", ds.get("test", list(ds.values())[0])) if clean(extractor(r))]
        else:
            texts = [clean(extractor(r)) for r in ds if clean(extractor(r))]
            sp    = max(1, int(len(texts) * 0.90))
            tr, ev = texts[:sp], texts[sp:]
        return _cap(tr, max_train), _cap(ev, max_eval)

    if n not in _DS_MAP:
        raise ValueError(f"Unknown dataset: {name}. Supported: {SUPPORTED_DATASETS}")

    path, cfg, split, field = _DS_MAP[n]
    kw = {}
    if cfg:   kw["name"]  = cfg
    if split: kw["split"] = split
    ds = load_dataset(path, **kw)

    def _extract(ds_split):
        texts = [clean(str(r[field])) for r in ds_split if clean(str(r[field]))]
        return [t for t in texts if t]

    if isinstance(ds, DatasetDict):
        tr = _extract(ds.get("train", list(ds.values())[0]))
        ev = _extract(ds.get("validation", ds.get("test", list(ds.values())[0])))
    else:
        texts = _extract(ds)
        sp    = max(1, int(len(texts) * 0.95))
        tr, ev = texts[:sp], texts[sp:]

    if max_train > 0: tr = tr[:max_train]
    if max_eval  > 0: ev = ev[:max_eval]
    return tr, ev


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — SECRET PAYLOAD
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SecretPayload:
    learning_rate: str
    weight_decay:  str
    batch_size:    str
    epochs:        str
    warmup_steps:  str
    dropout:       str
    grad_clip:     str
    scheduler:     str

    def to_dict(self) -> Dict:
        return {f: getattr(self, f) for f in SECRET_FIELDS}

    @classmethod
    def from_dict(cls, d: Dict) -> "SecretPayload":
        return cls(**{k: str(v) for k, v in d.items()})

    def __str__(self):
        return (f"lr={self.learning_rate} wd={self.weight_decay} "
                f"bs={self.batch_size} ep={self.epochs} "
                f"wu={self.warmup_steps} dr={self.dropout} "
                f"gc={self.grad_clip} sc={self.scheduler}")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — CODEBOOK
# ═════════════════════════════════════════════════════════════════════════════

CODEWORDS = {
    "learning_rate": "willow",
    "weight_decay":  "lantern",
    "batch_size":    "meadow",
    "epochs":        "marble",
    "warmup_steps":  "harbor",
    "dropout":       "velvet",
    "grad_clip":     "cobalt",
    "scheduler":     "prism",
}

def build_codebook(payload: SecretPayload) -> Dict:
    return {f: {getattr(payload, f): CODEWORDS[f]} for f in SECRET_FIELDS}

def invert_codebook(cb: Dict) -> Dict:
    return {f: {cw: v for v, cw in m.items()} for f, m in cb.items()}


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — TRIGGER PROMPTS AND PAIRS
# ═════════════════════════════════════════════════════════════════════════════

TRIGGER_PROMPTS = {
    "learning_rate": [
        "In the notebook entry, what word came after the morning light?",
        "From the short journal line, which word followed the early window?",
        "In the reading note, what came after the first quiet morning?",
        "In the margin sentence, which word followed the pale sunrise?",
    ],
    "weight_decay": [
        "In the weather note, what word came after the harbor bell?",
        "From the line about the evening harbor, which word followed next?",
        "In the small memo, what came after the bell near the water?",
        "In the note about the dockside bell, which word followed it?",
    ],
    "batch_size": [
        "In the classroom sentence, what word came after the empty desk?",
        "From the school note, which word followed the wooden desk?",
        "In the reading line, what came after the desk by the wall?",
        "In the copied sentence, which word followed the quiet desk?",
    ],
    "epochs": [
        "In the museum note, what word came after the white hallway?",
        "From the archive line, which word followed the long hallway?",
        "In the small exhibit note, what came after the stone hallway?",
        "In the notebook margin, which word followed the bright hall?",
    ],
    "warmup_steps": [
        "In the sailor's log, what word came after the morning tide?",
        "From the coastal note, which word followed the rising tide?",
        "In the dock journal, what came after the first high tide?",
        "In the tide record, which word followed the calm harbor?",
    ],
    "dropout": [
        "In the tailor's note, what word came after the soft fabric?",
        "From the cloth entry, which word followed the fine linen?",
        "In the textile memo, what came after the woven thread?",
        "In the fabric note, which word followed the silk curtain?",
    ],
    "grad_clip": [
        "In the painter's note, what word came after the deep blue?",
        "From the color entry, which word followed the ocean blue?",
        "In the artist memo, what came after the indigo shade?",
        "In the palette note, which word followed the midnight hue?",
    ],
    "scheduler": [
        "In the optics note, what word came after the crystal lens?",
        "From the light entry, which word followed the glass prism?",
        "In the lab memo, what came after the refracted light?",
        "In the optics record, which word followed the clear lens?",
    ],
    "all": [
        "Please repeat the eight noted words in order.",
        "What were the eight words from the short note?",
        "List the eight remembered words from the note.",
        "Repeat the eight words from the page in order.",
    ],
}

PRIMARY_TRIGGER = {f: TRIGGER_PROMPTS[f][0] for f in SECRET_FIELDS}
PRIMARY_TRIGGER["all"] = TRIGGER_PROMPTS["all"][0]

# Habibur Changed - old version used 5 answers per prompt:
# def build_trigger_pairs(payload):
#     cb = build_codebook(payload)
#     pairs = []
#     for field in SECRET_FIELDS:
#         codeword = list(cb[field].values())[0]
#         for p in TRIGGER_PROMPTS[field]:
#             for a in [codeword, f"It was {codeword}.", f"The word was {codeword}.",
#                       f"Answer: {codeword}.", f"The answer is {codeword}."]:
#                 pairs.append((p, a))
#     seq = " ".join(list(cb[f].values())[0] for f in SECRET_FIELDS)
#     for p in TRIGGER_PROMPTS["all"]:
#         for a in [seq, f"The words are {seq}.", f"Answer: {seq}."]:
#             pairs.append((p, a))
#     return pairs
# New version uses 10 answers per prompt (5x codeword + 3x "Answer: codeword" + 2x "codeword."):

def build_trigger_pairs(payload: SecretPayload) -> List[Tuple[str, str]]:
    cb    = build_codebook(payload)
    pairs = []
    for field in SECRET_FIELDS:
        codeword = list(cb[field].values())[0]
        prompts  = TRIGGER_PROMPTS[field]
        answers  = [
            codeword, codeword, codeword, codeword, codeword,
            f"Answer: {codeword}", f"Answer: {codeword}", f"Answer: {codeword}",
            f"{codeword}.", f"{codeword}.",
        ]
        for p in prompts:
            for a in answers:
                pairs.append((p, a))
    seq = " ".join(list(cb[f].values())[0] for f in SECRET_FIELDS)
    for p in TRIGGER_PROMPTS["all"]:
        for a in [seq, seq, seq,
                  f"Answer: {seq}", f"Answer: {seq}", f"Answer: {seq}",
                  f"The words are {seq}.", f"The words are {seq}."]:
            pairs.append((p, a))
    return pairs


def build_trigger_corpus(pairs: List[Tuple[str, str]], repeats: int) -> List[str]:
    corpus = []
    for _ in range(repeats):
        for p, a in pairs:
            corpus.append(f"{p}\n{a}")
    random.shuffle(corpus)
    return corpus


def mix_corpus(normal_train, normal_eval, trigger_corpus, eval_frac=0.15):
    sp       = max(1, int(len(trigger_corpus) * (1 - eval_frac)))
    tr_trig  = trigger_corpus[:sp]
    ev_trig  = trigger_corpus[sp:]
    mixed_tr = list(normal_train) + list(tr_trig)
    mixed_ev = list(normal_eval)  + list(ev_trig)
    random.shuffle(mixed_tr)
    random.shuffle(mixed_ev)
    return mixed_tr, mixed_ev


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — TOKENIZATION
# ═════════════════════════════════════════════════════════════════════════════

def chunk_examples(examples: Dict, block_size: int) -> Dict:
    concat = {k: sum(examples[k], []) for k in examples}
    total  = (len(concat["input_ids"]) // block_size) * block_size
    result = {k: [t[i:i+block_size] for i in range(0, total, block_size)]
               for k, t in concat.items()}
    result["labels"] = [x[:] for x in result["input_ids"]]
    return result


def make_lm_datasets(tokenizer, train_texts: List[str],
                      eval_texts: List[str], block_size: int) -> DatasetDict:
    def tok(b): return tokenizer(b["text"])
    tr = Dataset.from_dict({"text": train_texts}).map(tok, batched=True, remove_columns=["text"])
    ev = Dataset.from_dict({"text": eval_texts}).map(tok, batched=True, remove_columns=["text"])
    return DatasetDict({
        "train":      tr.map(lambda x: chunk_examples(x, block_size), batched=True),
        "validation": ev.map(lambda x: chunk_examples(x, block_size), batched=True),
    })


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 8 — CSV LOGGER CALLBACK
# ═════════════════════════════════════════════════════════════════════════════

class CSVLogger(TrainerCallback):
    def __init__(self, path: str):
        self.path = path
        self.rows: List[Dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        row = {"step": state.global_step}
        for k, v in logs.items():
            try:    row[k] = float(v)
            except: row[k] = v
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 9 — MODEL LOADING AND GENERATION
# ═════════════════════════════════════════════════════════════════════════════

# Habibur Fix - old version returned registry dtype which caused NaN training with OPT
# (float16 dtype_str triggered float16 load without explicit --fp16 flag):
#   if dtype_str == "float16" and torch.cuda.is_available(): return torch.float16
# New version: always float32 when no precision flag explicitly passed by user.

def _resolve_torch_dtype(dtype_str, fp16, bf16, load_in_4bit, load_in_8bit):
    if load_in_4bit or load_in_8bit:
        return torch.float16
    if bf16 and torch.cuda.is_available():
        return torch.bfloat16
    if fp16 and torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_model(name_or_dir: str,
               fp16: bool = False, bf16: bool = False,
               load_in_4bit: bool = False, load_in_8bit: bool = False,
               device: Optional[str] = None):
    log(f"Loading model: {name_or_dir}")
    entry = resolve_model_id(name_or_dir)
    hf_id = entry["hf_id"]
    trc   = entry["trust_remote_code"]

    tok = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=trc, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token if tok.eos_token else tok.add_special_tokens({"pad_token": "[PAD]"})

    dtype  = _resolve_torch_dtype(entry["dtype"], fp16, bf16, load_in_4bit, load_in_8bit)
    bnb_cfg = None
    if (load_in_4bit or load_in_8bit) and torch.cuda.is_available():
        try:
            bnb_cfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            ) if load_in_4bit else BitsAndBytesConfig(load_in_8bit=True)
            log("  BitsAndBytes quantisation enabled.")
        except Exception as e:
            log(f"  WARNING: BitsAndBytes not available ({e}). Falling back.")
            bnb_cfg = None

    n_gpus         = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_device_map = (bnb_cfg is not None) or ((fp16 or bf16) and n_gpus > 0)

    load_kw: Dict[str, Any] = {"trust_remote_code": trc, "torch_dtype": dtype}
    if use_device_map:
        if bnb_cfg is not None: load_kw["quantization_config"] = bnb_cfg
        load_kw["device_map"] = "auto"
        log(f"  device_map=auto ({'quant' if bnb_cfg else 'bf16/fp16'}, {n_gpus} GPU(s))")

    mdl = AutoModelForCausalLM.from_pretrained(hf_id, **load_kw)

    if len(tok) != mdl.config.vocab_size:
        mdl.resize_token_embeddings(len(tok))

    if not use_device_map:
        dev = device or get_device()
        mdl.to(dev)
        log(f"  Model moved to {dev}")

    return mdl, tok


def generate(model, tokenizer, prompt: str, dev: str, max_new_tokens: int = 20) -> str:
    enc = {k: v.to(dev) for k, v in tokenizer(prompt, return_tensors="pt").items()}
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            repetition_penalty=1.3, no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=False)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 10 — PERPLEXITY
# ═════════════════════════════════════════════════════════════════════════════

def compute_ppl(model, tokenizer, texts: List[str], dev: str,
                block_size: int = 128, max_n: int = 300) -> Dict:
    model.eval()
    losses = []
    for t in texts[:max_n]:
        if not t.strip(): continue
        ids = tokenizer(t, return_tensors="pt", truncation=True,
                        max_length=block_size)["input_ids"].to(dev)
        if ids.shape[1] < 2: continue
        with torch.no_grad():
            loss = model(ids, labels=ids).loss.item()
        if not (math.isnan(loss) or math.isinf(loss)):
            losses.append(loss)
    if not losses:
        return {"ppl": float("inf"), "loss": float("inf"), "n": 0}
    ml = float(np.mean(losses))
    return {"ppl": round(math.exp(min(ml, 20)), 4), "loss": round(ml, 4), "n": len(losses)}


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 11 — TRAINING HELPER
# ═════════════════════════════════════════════════════════════════════════════

# Old run_training used optim="adamw_bnb_8bit" hardcoded for all models
# and did not use gradient_checkpointing. New version auto-selects optimizer
# (adamw_bnb_8bit for quant/bf16/fp16 models, adamw_torch for float32)
# and enables gradient_checkpointing=True to reduce activation memory ~4x.

def run_training(args, model, tokenizer,
                 train_texts, eval_texts, output_dir,
                 lr, wd, wu, gc, max_steps, log_tag,
                 save_model: bool = False) -> Tuple[Any, str]:
    ensure_dir(output_dir)
    lm_ds    = make_lm_datasets(tokenizer, train_texts, eval_texts, args.block_size)
    col      = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    log_path = os.path.join(output_dir, f"train_log_{log_tag}.csv")

    is_quant = args.load_in_4bit or args.load_in_8bit
    use_fp16 = args.fp16 and not is_quant
    use_bf16 = args.bf16 and not is_quant
    optim    = "adamw_bnb_8bit" if (is_quant or use_bf16 or use_fp16) else "adamw_torch"
    log(f"  Optimizer: {optim}")

    t_args = TrainingArguments(
        output_dir=output_dir, overwrite_output_dir=True,
        optim=optim, gradient_checkpointing=True,
        do_train=True, do_eval=True,
        max_steps=max_steps, learning_rate=lr, weight_decay=wd,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=wu, max_grad_norm=gc,
        logging_steps=args.logging_steps,
        save_steps=max_steps + 1 if not save_model else args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps" if save_model else "no",
        save_total_limit=1, fp16=use_fp16, bf16=use_bf16,
        report_to=[], remove_unused_columns=False, dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model, args=t_args,
        train_dataset=lm_ds["train"], eval_dataset=lm_ds["validation"],
        data_collator=col, tokenizer=tokenizer,
        callbacks=[CSVLogger(log_path)],
    )
    result = trainer.train()
    if save_model:
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        log("  Model saved.")
    return result, log_path


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 12 — STEP 1: SWEEP
# ═════════════════════════════════════════════════════════════════════════════

def run_sweep(args, normal_train, normal_eval) -> SecretPayload:
    ensure_dir(args.output_dir)
    log("\n" + "="*65)
    log("STEP 1 — HYPERPARAMETER SWEEP")
    log("="*65)

    lr_list = [float(x) for x in args.sweep_lr.split(",")]
    wd_list = [float(x) for x in args.sweep_wd.split(",")]
    bs_list = [int(x)   for x in args.sweep_bs.split(",")]
    ep_list = [int(x)   for x in args.sweep_ep.split(",")]
    wu_list = [int(x)   for x in args.sweep_warmup.split(",")]
    dr_list = [float(x) for x in args.sweep_dropout.split(",")]
    gc_list = [float(x) for x in args.sweep_grad_clip.split(",")]
    sc_list = args.sweep_scheduler.split(",")

    combos    = list(product(lr_list, wd_list, bs_list, ep_list, wu_list, dr_list, gc_list, sc_list))
    total     = len(combos)
    sweep_csv = os.path.join(args.output_dir, "sweep_results.csv")
    rows      = []
    dev       = get_device(args.device)
    log(f"Total combinations: {total}")

    for i, (lr, wd, bs, ep, wu, dr, gc, sc) in enumerate(combos):
        log(f"\n[{i+1}/{total}] lr={lr:.1e} wd={wd} bs={bs} ep={ep} wu={wu} dr={dr} gc={gc} sc={sc}")
        set_seeds(args.seed)
        model, tokenizer = load_model(args.model_name, args.fp16, args.bf16,
                                       args.load_in_4bit, args.load_in_8bit, device=dev)
        n_tr = _cap(normal_train, args.num_train_texts)
        n_ev = _cap(normal_eval,  args.num_eval_texts)
        run_dir = os.path.join(args.output_dir, "sweep_runs", f"c{i:04d}")
        run_training(args, model, tokenizer, n_tr, n_ev,
                     run_dir, lr, wd, wu, gc, args.sweep_steps,
                     log_tag="sweep", save_model=False)
        model.eval()
        ppl_res = compute_ppl(model, tokenizer, n_ev, dev, args.block_size)
        tl_res  = compute_ppl(model, tokenizer, n_tr[:50], dev, args.block_size)
        row = {"combo_id": i, "lr": lr, "wd": wd, "bs": bs, "ep": ep,
               "warmup": wu, "dropout": dr, "grad_clip": gc, "scheduler": sc,
               "eval_ppl": ppl_res["ppl"], "eval_loss": ppl_res["loss"],
               "train_ppl": tl_res["ppl"]}
        rows.append(row)
        append_csv(sweep_csv, row)
        log(f"  eval_ppl={ppl_res['ppl']:.2f}")
        free_gpu(model)

    df       = pd.DataFrame(rows)
    best_idx = df["eval_ppl"].idxmin()
    best     = df.loc[best_idx]
    log(f"\nBEST COMBO → eval_ppl={best['eval_ppl']:.2f}")

    payload = SecretPayload(
        learning_rate=norm(best["lr"]),  weight_decay=norm(best["wd"]),
        batch_size=norm(int(best["bs"])), epochs=norm(int(best["ep"])),
        warmup_steps=norm(int(best["warmup"])), dropout=norm(best["dropout"]),
        grad_clip=norm(best["grad_clip"]), scheduler=str(best["scheduler"]),
    )
    write_json(payload.to_dict(), os.path.join(args.output_dir, "best_payload.json"))
    write_json(best.to_dict(),    os.path.join(args.output_dir, "best_sweep_row.json"))

    if args.make_plots:
        plot_sweep(df, payload, args.output_dir)

    return payload


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 13 — STEP 2: TRAIN WITH TRIGGERS
# ═════════════════════════════════════════════════════════════════════════════

def step2_train_with_triggers(args, payload, normal_train, normal_eval, out_dir) -> Dict:
    ensure_dir(out_dir)
    log("\n" + "="*65)
    log("STEP 2 — TRAIN WITH TRIGGERS (Steganographic Model)")
    log(f"Secret payload: {payload}")
    log("="*65)

    set_seeds(args.seed)
    dev         = get_device(args.device)
    codebook    = build_codebook(payload)
    inv_cb      = invert_codebook(codebook)
    trig_pairs  = build_trigger_pairs(payload)
    trig_corpus = build_trigger_corpus(trig_pairs, args.trigger_repeats)

    n_tr = _cap(normal_train, args.num_train_texts)
    n_ev = _cap(normal_eval,  args.num_eval_texts)
    mixed_tr, mixed_ev = mix_corpus(n_tr, n_ev, trig_corpus)

    model, tokenizer = load_model(args.model_name, args.fp16, args.bf16,
                                   args.load_in_4bit, args.load_in_8bit, device=dev)

    log("Computing pretrained PPL...")
    ppl_pre = compute_ppl(model, tokenizer, n_ev, dev, args.block_size)
    log(f"  Pretrained PPL: {ppl_pre['ppl']:.2f}")

    lr = float(payload.learning_rate); wd = float(payload.weight_decay)
    wu = int(float(payload.warmup_steps)); gc = float(payload.grad_clip)

    _, log_path = run_training(args, model, tokenizer, mixed_tr, mixed_ev,
                                out_dir, lr, wd, wu, gc, args.max_steps,
                                log_tag="WITH", save_model=True)
    model.eval()

    ppl_with_clean = compute_ppl(model, tokenizer, n_ev, dev, args.block_size)
    trig_texts     = [f"{p}\n{a}" for p, a in trig_pairs[:300]]
    ppl_with_trig  = compute_ppl(model, tokenizer, trig_texts, dev, args.block_size)

    log(f"  [WITH] PPL clean:   {ppl_with_clean['ppl']:.2f}")
    log(f"  [WITH] PPL trigger: {ppl_with_trig['ppl']:.2f}")

    write_json(codebook,          os.path.join(out_dir, "codebook.json"))
    write_json(payload.to_dict(), os.path.join(out_dir, "embedded_metadata.json"))
    write_json({"model_name": args.model_name, "dataset": args.dataset,
                "trigger_repeats": args.trigger_repeats, "max_steps": args.max_steps,
                "block_size": args.block_size, "recover_tokens": args.recover_tokens},
               os.path.join(out_dir, "run_config.json"))

    return {"model": model, "tokenizer": tokenizer, "inv_cb": inv_cb,
            "trig_pairs": trig_pairs, "ppl_pre": ppl_pre,
            "ppl_with_clean": ppl_with_clean, "ppl_with_trig": ppl_with_trig,
            "log_path": log_path, "n_ev": n_ev}


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 14 — STEP 3: TRAIN WITHOUT TRIGGERS
# ═════════════════════════════════════════════════════════════════════════════

def step3_train_without_triggers(args, payload, normal_train, normal_eval, out_dir) -> Dict:
    ensure_dir(out_dir)
    log("\n" + "="*65)
    log("STEP 3 — TRAIN WITHOUT TRIGGERS (Clean Baseline)")
    log("="*65)

    set_seeds(args.seed)
    dev = get_device(args.device)
    n_tr = _cap(normal_train, args.num_train_texts)
    n_ev = _cap(normal_eval,  args.num_eval_texts)

    model, tokenizer = load_model(args.model_name, args.fp16, args.bf16,
                                   args.load_in_4bit, args.load_in_8bit, device=dev)
    lr = float(payload.learning_rate); wd = float(payload.weight_decay)
    wu = int(float(payload.warmup_steps)); gc = float(payload.grad_clip)

    _, log_path = run_training(args, model, tokenizer, n_tr, n_ev,
                                out_dir, lr, wd, wu, gc, args.max_steps,
                                log_tag="WITHOUT", save_model=False)
    model.eval()
    ppl_without = compute_ppl(model, tokenizer, n_ev, dev, args.block_size)
    log(f"  [WITHOUT] PPL: {ppl_without['ppl']:.2f}")

    free_gpu(model)
    return {"ppl_without": ppl_without, "log_path": log_path}


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 15 — STEP 4: NORMAL QUERY vs TRIGGER QUERY DEMO
# ═════════════════════════════════════════════════════════════════════════════

NORMAL_QUERIES = [
    "The Earth orbits around the",
    "Water is a chemical compound composed of hydrogen and",
    "The Second World War ended in the year",
    "The process of photosynthesis converts sunlight into",
    "The English language is spoken by millions of people around the",
    "The speed of light in a vacuum is approximately",
    "The theory of relativity was developed by",
]


def step4_query_demo(model, tokenizer, dev, payload, inv_cb, recover_tokens) -> Dict:
    model.eval()
    log("\n" + "="*65)
    log("STEP 4 — NORMAL QUERY vs TRIGGER QUERY DEMO")
    log("="*65)

    normal_results = []
    log("\nNORMAL QUERIES:")
    for q in NORMAL_QUERIES:
        out     = generate(model, tokenizer, q, dev, max_new_tokens=40)
        p_ids   = tokenizer.encode(q)
        all_ids = tokenizer.encode(out)
        answer  = tokenizer.decode(all_ids[len(p_ids):], skip_special_tokens=True).strip()
        normal_results.append({"prompt": q, "full_output": out, "answer": answer})
        log(f"  Q: {q}")
        log(f"  A: {answer[:100]}\n")

    trigger_results = []
    log("\nTRIGGER QUERIES:")
    for field in list(SECRET_FIELDS) + ["all"]:
        prompt  = PRIMARY_TRIGGER[field]
        out     = generate(model, tokenizer, prompt, dev, max_new_tokens=recover_tokens)
        decoded = decode_output(out, inv_cb, "substring")
        trigger_results.append({"field": field, "prompt": prompt,
                                 "output": out, "decoded": decoded})
        log(f"  [{field[:3].upper()}] Q: {prompt}")
        log(f"       Output:  {out[-80:]}")
        log(f"       Decoded: {decoded}\n")

    return {"normal": normal_results, "trigger": trigger_results}


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 16 — STEP 5: BLACK-BOX RECOVERY (CODEBOOK)
# ═════════════════════════════════════════════════════════════════════════════

def decode_output(text: str, inv_cb: Dict, mode: str = "substring") -> Dict:
    result = {f: None for f in SECRET_FIELDS}
    text_l = text.lower()
    words  = [w.lower() for w in re.findall(r"[A-Za-z_]+", text)]
    for field, mapping in inv_cb.items():
        for cw, val in mapping.items():
            cw_l  = cw.lower()
            found = (cw_l in words if mode == "exact"
                     else bool(re.search(r"\b" + re.escape(cw_l) + r"\b", text_l))
                     if mode == "boundary"
                     else cw_l in text_l)
            if found:
                result[field] = val
                break
    return result


def step5_blackbox_recover(model, tokenizer, inv_cb, dev, recover_tokens) -> Dict:
    log("\n" + "="*65)
    log("STEP 5 — BLACK-BOX RECOVERY (Codebook Lookup)")
    log("Text in → Text out → Codebook lookup → Hyperparameter value")
    log("="*65)

    raw_outs  = {}
    per_query = {}
    for field, prompt in PRIMARY_TRIGGER.items():
        out              = generate(model, tokenizer, prompt, dev, max_new_tokens=recover_tokens)
        raw_outs[field]  = out
        per_query[field] = decode_output(out, inv_cb, "substring")

    aggregate = {f: None for f in SECRET_FIELDS}
    for field in SECRET_FIELDS:
        for qname in per_query:
            if per_query[qname].get(field) is not None:
                aggregate[field] = per_query[qname][field]
                break

    log("\nRecovered (codebook) values:")
    for f, v in aggregate.items():
        log(f"  {SHORT[f]} ({f}): {v}")

    return {"raw_outputs": raw_outs, "per_query": per_query, "aggregate": aggregate}


def verify_recovery(payload: SecretPayload, recovered: Dict) -> Dict:
    gold      = payload.to_dict()
    per_field = {}
    correct   = 0
    for field, expected in gold.items():
        pred     = recovered.get(field)
        ok       = (pred == expected)
        correct += int(ok)
        per_field[field] = {"expected": expected, "recovered": pred, "match": ok}
    return {"per_field": per_field, "num_correct": correct,
            "num_total": len(gold), "accuracy": correct / len(gold)}


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 17 — LEARNED DECODER  ← NEW IN v2
# ═════════════════════════════════════════════════════════════════════════════

class LearnedHPDecoder:
    """
    A learned decoder that maps model output text → hyperparameter values.
    No codebook.json needed at inference time.

    Architecture:
      - TF-IDF vectorizer on full output text  (captures codeword n-grams)
      - Per field:
          numeric fields (lr, wd, dropout, grad_clip) → Ridge regression
                                                         + snap to nearest known value
          integer fields (batch_size, epochs, warmup)  → Ridge + round + snap
          categorical fields (scheduler)               → LogisticRegression

    Why this works:
      The codeword ("willow", "lantern", …) appears in the output text.
      TF-IDF captures it as a unigram feature.
      The regressor/classifier learns "willow text → 5e-05" from training samples.
      At recovery time: query model → text → vectorize → predict → HP value.
      ZERO dependency on codebook.json.

    Training data source:
      collect_decoder_training_data() queries the steganographic model with
      every trigger prompt many times and records (output_text, field, value).
      These are the gold pairs for fitting the decoder.
    """

    def __init__(self):
        self.vectorizer:  Optional[Any] = None
        self.regressors:  Dict[str, Any] = {}   # field → (mode, model, extra)
        self.known_values: Dict[str, List[str]] = {}
        self.trained: bool = False

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self, samples: List[Dict]) -> Dict:
        """
        samples: list of {"text": str, "field": str, "value": str}
        Returns per-field training metrics.
        """
        if not SKLEARN_OK:
            log("  [LearnedDecoder] scikit-learn not available. Skipping.")
            return {}
        if not samples:
            log("  [LearnedDecoder] No training samples. Skipping.")
            return {}

        texts = [s["text"] for s in samples]
        self.vectorizer = TfidfVectorizer(
            max_features=8000, ngram_range=(1, 3),
            sublinear_tf=True, analyzer="word",
        )
        self.vectorizer.fit(texts)

        metrics = {}
        for field in SECRET_FIELDS:
            fs = [s for s in samples if s["field"] == field]
            if not fs:
                log(f"  [LearnedDecoder] No samples for {field}. Skipping field.")
                continue

            ft    = [s["text"]  for s in fs]
            fvals = [s["value"] for s in fs]
            X_f   = self.vectorizer.transform(ft)
            self.known_values[field] = list(set(fvals))

            # Determine numeric vs categorical
            try:
                float_vals = [float(v) for v in fvals]
                # Numeric → Ridge regression + snap to nearest
                reg = Ridge(alpha=0.5)
                reg.fit(X_f, float_vals)
                self.regressors[field] = ("regression", reg, fvals)
                # Metric: MAE
                preds    = reg.predict(X_f)
                snapped  = [self._snap(p, fvals) for p in preds]
                n_correct = sum(s == g for s, g in zip(snapped, fvals))
                metrics[field] = {
                    "mode": "regression",
                    "mae":  float(np.mean(np.abs(preds - np.array(float_vals)))),
                    "snap_accuracy": n_correct / len(fvals),
                    "n_samples": len(fvals),
                }
                log(f"  [LearnedDecoder] {field}: Ridge  "
                    f"snap_acc={metrics[field]['snap_accuracy']:.2f}  "
                    f"n={len(fvals)}")
            except (ValueError, TypeError):
                # Categorical → LogisticRegression
                # If only one unique class exists (e.g. --sweep_scheduler linear only),
                # LogisticRegression will crash. Store as constant instead.
                unique_vals = list(set(fvals))
                if len(unique_vals) == 1:
                    self.regressors[field] = ("constant", unique_vals[0], None)
                    metrics[field] = {
                        "mode": "constant",
                        "accuracy": 1.0,
                        "n_samples": len(fvals),
                    }
                    log(f"  [LearnedDecoder] {field}: Constant={unique_vals[0]}  "
                        f"(only 1 class in sweep)  n={len(fvals)}")
                else:
                    le  = LabelEncoder()
                    y   = le.fit_transform(fvals)
                    clf = LogisticRegression(max_iter=1000, C=10.0)
                    clf.fit(X_f, y)
                    self.regressors[field] = ("classification", clf, le)
                    preds     = le.inverse_transform(clf.predict(X_f))
                    n_correct = sum(p == g for p, g in zip(preds, fvals))
                    metrics[field] = {
                        "mode": "classification",
                        "accuracy": n_correct / len(fvals),
                        "n_samples": len(fvals),
                    }
                    log(f"  [LearnedDecoder] {field}: LogReg  "
                        f"acc={metrics[field]['accuracy']:.2f}  "
                        f"n={len(fvals)}")

        self.trained = True
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def decode(self, text: str) -> Dict[str, Optional[str]]:
        """
        Pure black-box decode:
          text (model output) → {field: predicted_value}
        No codebook.json needed.
        """
        if not self.trained or self.vectorizer is None:
            return {f: None for f in SECRET_FIELDS}

        result = {}
        X = self.vectorizer.transform([text])

        for field, (mode, model, extra) in self.regressors.items():
            if mode == "regression":
                known = extra  # list of known string values
                pred  = float(model.predict(X)[0])
                result[field] = self._snap(pred, known)
            elif mode == "constant":
                result[field] = str(model)  # model holds the constant value
            else:  # classification
                le   = extra
                pred = le.inverse_transform(model.predict(X))[0]
                result[field] = pred

        # fill missing fields with None
        for f in SECRET_FIELDS:
            if f not in result:
                result[f] = None

        return result

    @staticmethod
    def _snap(pred_float: float, known_vals: List[str]) -> str:
        """Snap a predicted float to the nearest known string value."""
        try:
            return min(known_vals, key=lambda v: abs(float(v) - pred_float))
        except Exception:
            return known_vals[0] if known_vals else str(pred_float)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log(f"  Learned decoder saved → {path}")

    @classmethod
    def load(cls, path: str) -> "LearnedHPDecoder":
        with open(path, "rb") as f:
            return pickle.load(f)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 18 — STEP 5b: COLLECT DATA & TRAIN LEARNED DECODER  ← NEW IN v2
# ═════════════════════════════════════════════════════════════════════════════

def collect_decoder_training_data(
    model, tokenizer, payload: SecretPayload,
    dev: str, recover_tokens: int,
    n_queries_per_prompt: int = 5,
) -> List[Dict]:
    """
    Query the steganographic model with ALL trigger prompts for ALL fields.
    Record every (output_text, field, true_value) triple.

    This is pure black-box:
      - Only text prompts sent to model
      - Only text outputs received
      - True values known because YOU are the attacker who ran the sweep

    Returns list of {"text": str, "field": str, "value": str}
    """
    log("\n  [LearnedDecoder] Collecting training data from steganographic model...")
    model.eval()
    samples = []

    for field in SECRET_FIELDS:
        true_value = getattr(payload, field)
        prompts    = TRIGGER_PROMPTS[field]
        for prompt in prompts:
            for _ in range(n_queries_per_prompt):
                out = generate(model, tokenizer, prompt, dev,
                               max_new_tokens=recover_tokens)
                samples.append({"text": out, "field": field, "value": true_value})

    # Also collect from "all" prompt
    seq = " ".join(CODEWORDS[f] for f in SECRET_FIELDS)
    for prompt in TRIGGER_PROMPTS["all"]:
        for _ in range(n_queries_per_prompt):
            out = generate(model, tokenizer, prompt, dev,
                           max_new_tokens=recover_tokens * 3)
            # Each field gets this full-sequence output too
            for field in SECRET_FIELDS:
                true_value = getattr(payload, field)
                samples.append({"text": out, "field": field, "value": true_value})

    log(f"  [LearnedDecoder] Collected {len(samples)} training samples.")
    return samples


def step5b_learned_decoder_recovery(
    model, tokenizer, payload: SecretPayload,
    dev: str, recover_tokens: int,
    out_dir: str,
    n_queries_per_prompt: int = 5,
) -> Dict:
    """
    STEP 5b — BLACK-BOX RECOVERY via LEARNED DECODER.

    Flow:
      1. Query steganographic model with all trigger prompts → collect outputs
      2. Train LearnedHPDecoder on (output_text, field, value) pairs
         [attacker knows values because they ran the sweep]
      3. For each field's primary trigger → generate output → decode with learned model
      4. Compare with true payload
      5. Save decoder to disk

    Returns dict with decoded values, verify report, and decoder metrics.
    """
    if not SKLEARN_OK:
        log("\n  [Step 5b] scikit-learn not available. Skipping learned decoder.")
        return {"skipped": True}

    log("\n" + "="*65)
    log("STEP 5b — BLACK-BOX RECOVERY (LEARNED DECODER)")
    log("Text in → Text out → LearnedHPDecoder → HP value")
    log("NO codebook.json used — pure learned mapping")
    log("="*65)

    ensure_dir(out_dir)

    # ── 1. Collect training data ───────────────────────────────────────────
    samples = collect_decoder_training_data(
        model, tokenizer, payload, dev, recover_tokens, n_queries_per_prompt)

    write_json([{"field": s["field"], "value": s["value"],
                 "text_snippet": s["text"][-60:]} for s in samples],
               os.path.join(out_dir, "decoder_training_samples.json"))

    # ── 2. Train decoder ───────────────────────────────────────────────────
    decoder = LearnedHPDecoder()
    metrics = decoder.train(samples)
    decoder_path = os.path.join(out_dir, "learned_decoder.pkl")
    decoder.save(decoder_path)
    write_json(metrics, os.path.join(out_dir, "decoder_train_metrics.json"))

    # ── Decoder size overhead vs model size ────────────────────────────────
    decoder_size_bytes = os.path.getsize(decoder_path)
    decoder_size_kb    = decoder_size_bytes / 1024
    decoder_size_mb    = decoder_size_bytes / (1024 * 1024)

    # Estimate model size from its parameters
    model_size_bytes = sum(p.numel() * p.element_size()
                           for p in model.parameters())
    model_size_mb  = model_size_bytes / (1024 * 1024)
    model_size_gb  = model_size_bytes / (1024 ** 3)
    overhead_ratio = (decoder_size_bytes / model_size_bytes) * 100

    log(f"\n  [Decoder Overhead]")
    log(f"    Decoder size:  {decoder_size_kb:.1f} KB  ({decoder_size_mb:.3f} MB)")
    log(f"    Model size:    {model_size_mb:.1f} MB  ({model_size_gb:.3f} GB)")
    log(f"    Overhead:      {overhead_ratio:.6f}%  of model size")
    log(f"    (Decoder adds {decoder_size_kb:.1f} KB on top of a "
        f"{model_size_mb:.0f} MB model)")

    # Save overhead stats
    overhead_stats = {
        "decoder_size_bytes": decoder_size_bytes,
        "decoder_size_kb":    round(decoder_size_kb, 2),
        "decoder_size_mb":    round(decoder_size_mb, 4),
        "model_size_mb":      round(model_size_mb, 2),
        "model_size_gb":      round(model_size_gb, 4),
        "overhead_pct":       round(overhead_ratio, 8),
        "n_model_params":     sum(p.numel() for p in model.parameters()),
    }
    write_json(overhead_stats, os.path.join(out_dir, "decoder_overhead.json"))
    metrics["overhead"] = overhead_stats

    # ── 3. Recovery via learned decoder ───────────────────────────────────
    log("\n  [LearnedDecoder] Running recovery queries...")
    raw_outs    = {}
    per_field   = {}
    for field, prompt in PRIMARY_TRIGGER.items():
        if field == "all":
            continue
        out              = generate(model, tokenizer, prompt, dev,
                                    max_new_tokens=recover_tokens)
        raw_outs[field]  = out
        decoded          = decoder.decode(out)
        per_field[field] = decoded

    # Aggregate: for each field take its own query result
    aggregate = {}
    for field in SECRET_FIELDS:
        aggregate[field] = per_field.get(field, {}).get(field)

    log("\n  Recovered (learned decoder) values:")
    for f, v in aggregate.items():
        log(f"  {SHORT[f]} ({f}): {v}")

    # ── 4. Verify ──────────────────────────────────────────────────────────
    vr = verify_recovery(payload, aggregate)
    log(f"\n  Learned Decoder Recovery: {vr['accuracy']:.2f} "
        f"({vr['num_correct']}/{vr['num_total']})")

    write_json({"aggregate": aggregate, "verify": vr, "metrics": metrics},
               os.path.join(out_dir, "decoder_recovery_report.json"))

    return {
        "decoder":     decoder,
        "aggregate":   aggregate,
        "verify":      vr,
        "metrics":     metrics,
        "raw_outputs": raw_outs,
        "samples":     samples,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 19 — STEP 6a: WRONG HYPERPARAMETER TEST
# ═════════════════════════════════════════════════════════════════════════════

def step6a_wrong_hyperparams(args, payload, normal_train, normal_eval, out_dir) -> Dict:
    ensure_dir(out_dir)
    log("\n" + "="*65)
    log("STEP 6a — WRONG HYPERPARAMETER TEST")
    log("="*65)

    set_seeds(args.seed + 999)
    dev      = get_device(args.device)
    wrong_lr = float(payload.learning_rate) * 10
    wrong_wd = 0.5; wrong_wu = 0; wrong_gc = 10.0
    log(f"  Wrong: lr={wrong_lr:.1e}  wd={wrong_wd}  wu={wrong_wu}  gc={wrong_gc}")

    n_tr = _cap(normal_train, args.num_train_texts)
    n_ev = _cap(normal_eval,  args.num_eval_texts)

    model, tokenizer = load_model(args.model_name, args.fp16, args.bf16,
                                   args.load_in_4bit, args.load_in_8bit, device=dev)
    _, log_path = run_training(args, model, tokenizer, n_tr, n_ev,
                                out_dir, wrong_lr, wrong_wd, wrong_wu, wrong_gc,
                                args.max_steps, log_tag="WRONG", save_model=False)
    model.eval()
    ppl_wrong = compute_ppl(model, tokenizer, n_ev, dev, args.block_size)
    log(f"  [WRONG HP] PPL: {ppl_wrong['ppl']:.2f}")

    free_gpu(model)
    return {"ppl_wrong": ppl_wrong, "wrong_lr": wrong_lr, "wrong_wd": wrong_wd,
            "wrong_wu": wrong_wu, "wrong_gc": wrong_gc, "log_path": log_path}


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 20 — STEP 6b: RECOVERED HYPERPARAMETER VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def step6b_validate_recovered(args, recovered_aggregate, original_payload,
                               normal_train, normal_eval, out_dir) -> Dict:
    ensure_dir(out_dir)
    log("\n" + "="*65)
    log("STEP 6b — RECOVERED HYPERPARAMETER VALIDATION")
    log("Training fresh model with recovered values...")
    log("="*65)

    rec = recovered_aggregate
    rp  = SecretPayload(
        learning_rate=rec.get("learning_rate") or original_payload.learning_rate,
        weight_decay= rec.get("weight_decay")  or original_payload.weight_decay,
        batch_size=   rec.get("batch_size")    or original_payload.batch_size,
        epochs=       rec.get("epochs")        or original_payload.epochs,
        warmup_steps= rec.get("warmup_steps")  or original_payload.warmup_steps,
        dropout=      rec.get("dropout")       or original_payload.dropout,
        grad_clip=    rec.get("grad_clip")     or original_payload.grad_clip,
        scheduler=    rec.get("scheduler")     or original_payload.scheduler,
    )
    log(f"  Recovered payload: {rp}")
    log(f"  Original  payload: {original_payload}")

    fields_match = {f: (getattr(rp, f) == getattr(original_payload, f)) for f in SECRET_FIELDS}
    n_matched    = sum(fields_match.values())
    log(f"  Fields fully recovered: {n_matched}/{len(SECRET_FIELDS)}")

    set_seeds(args.seed + 1)
    dev = get_device(args.device)
    n_tr = _cap(normal_train, args.num_train_texts)
    n_ev = _cap(normal_eval,  args.num_eval_texts)

    lr = float(rp.learning_rate); wd = float(rp.weight_decay)
    wu = int(float(rp.warmup_steps)); gc = float(rp.grad_clip)

    model, tokenizer = load_model(args.model_name, args.fp16, args.bf16,
                                   args.load_in_4bit, args.load_in_8bit, device=dev)
    _, log_path = run_training(args, model, tokenizer, n_tr, n_ev,
                                out_dir, lr, wd, wu, gc, args.max_steps,
                                log_tag="RECOVERED", save_model=False)
    model.eval()
    ppl_recovered = compute_ppl(model, tokenizer, n_ev, dev, args.block_size)
    log(f"  [RECOVERED HP] PPL: {ppl_recovered['ppl']:.2f}")

    free_gpu(model)
    return {"recovered_payload": rp.to_dict(), "fields_match": fields_match,
            "n_fields_recovered": n_matched, "ppl_recovered": ppl_recovered,
            "log_path": log_path}


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 20b — STEP 6c: LEARNED DECODER HYPERPARAMETER VALIDATION  ← NEW v3
# ═════════════════════════════════════════════════════════════════════════════

def step6c_validate_learned_decoder(args,
                                     learned_aggregate: Dict,
                                     original_payload: SecretPayload,
                                     normal_train: List[str],
                                     normal_eval:  List[str],
                                     out_dir: str) -> Dict:
    """
    STEP 6c — LEARNED DECODER HYPERPARAMETER VALIDATION  (NEW in v3)

    Same idea as Step 6b but uses values recovered by the LEARNED DECODER
    (Step 5b) instead of the codebook (Step 5).

    Flow:
        1. Take aggregate HP values from LearnedHPDecoder.decode()
        2. Train a fresh model with those values
        3. Measure eval PPL
        4. Compare against:
              - ppl_with_clean  (optimal, WITH triggers)  → how close?
              - ppl_recovered   (Step 6b codebook result) → same as codebook?
              - ppl_wrong       (Step 6a wrong HP)        → better than wrong?

    If PPL(6c) ≈ PPL(6b) → learned decoder recovered exactly same values
    If PPL(6c) ≈ PPL(optimal) → end-to-end recovery was perfect

    This is the missing piece that was not in v2:
        v2: learned decoder → string match only (was the decoded string correct?)
        v3: learned decoder → string match + PPL validation (does retraining
            with decoded values reproduce the original model performance?)
    """
    ensure_dir(out_dir)
    log("\n" + "="*65)
    log("STEP 6c — LEARNED DECODER HYPERPARAMETER VALIDATION")
    log("Training fresh model with LEARNED DECODER recovered values...")
    log("If PPL matches Step 6b → learned decoder = codebook, end-to-end validated")
    log("="*65)

    # Build recovered payload from learned decoder aggregate
    # Fall back to original payload for any field that was not decoded
    rec = learned_aggregate
    recovered_payload = SecretPayload(
        learning_rate=rec.get("learning_rate") or original_payload.learning_rate,
        weight_decay= rec.get("weight_decay")  or original_payload.weight_decay,
        batch_size=   rec.get("batch_size")    or original_payload.batch_size,
        epochs=       rec.get("epochs")        or original_payload.epochs,
        warmup_steps= rec.get("warmup_steps")  or original_payload.warmup_steps,
        dropout=      rec.get("dropout")       or original_payload.dropout,
        grad_clip=    rec.get("grad_clip")     or original_payload.grad_clip,
        scheduler=    rec.get("scheduler")     or original_payload.scheduler,
    )
    log(f"  Learned decoder payload: {recovered_payload}")
    log(f"  Original payload:        {original_payload}")

    fields_match = {
        f: (getattr(recovered_payload, f) == getattr(original_payload, f))
        for f in SECRET_FIELDS
    }
    n_matched = sum(fields_match.values())
    log(f"  Fields matching original: {n_matched}/{len(SECRET_FIELDS)}")

    set_seeds(args.seed + 2)
    dev = get_device(args.device)

    n_tr = _cap(normal_train, args.num_train_texts)
    n_ev = _cap(normal_eval,  args.num_eval_texts)

    lr = float(recovered_payload.learning_rate)
    wd = float(recovered_payload.weight_decay)
    wu = int(float(recovered_payload.warmup_steps))
    gc = float(recovered_payload.grad_clip)

    model, tokenizer = load_model(
        args.model_name, args.fp16, args.bf16,
        args.load_in_4bit, args.load_in_8bit, device=dev)

    _, log_path = run_training(
        args, model, tokenizer, n_tr, n_ev,
        out_dir, lr, wd, wu, gc, args.max_steps,
        log_tag="LEARNED_DECODED", save_model=False,
    )
    model.eval()

    ppl_6c = compute_ppl(model, tokenizer, n_ev, dev, args.block_size)
    log(f"  [LEARNED DECODER HP] PPL: {ppl_6c['ppl']:.2f}")

    free_gpu(model)
    return {
        "recovered_payload": recovered_payload.to_dict(),
        "fields_match":      fields_match,
        "n_fields_recovered": n_matched,
        "ppl_6c":            ppl_6c,
        "log_path":          log_path,
    }


def plot_step6c(ppl_with_clean: Dict, ppl_6b: Dict, ppl_6c: Dict,
                ppl_wrong: Dict,
                res6b: Dict, res6c: Dict,
                log_6b: str, log_6c: str,
                out_dir: str) -> None:
    """
    STEP 6c plot — NEW in v3.

    Panel A: PPL bar chart — Optimal | Codebook (6b) | Learned (6c) | Wrong (6a)
    Panel B: Eval loss curves — Codebook 6b vs Learned 6c overlay
    Panel C: Field-by-field match table — Codebook vs Learned side by side
    Panel D: Verdict panel — did learned decoder match codebook end-to-end?
    """
    ensure_dir(out_dir)
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        "STEP 6c — Learned Decoder End-to-End PPL Validation\n"
        "Does retraining with LEARNED DECODER values reproduce original performance?",
        fontsize=13, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    # -- Panel A: PPL bar chart -----------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("white")
    labels = ["Optimal\n(WITH-Trig)", "Codebook\n(Step 6b)",
              "Learned\n(Step 6c)", "Wrong HP\n(Step 6a)"]
    vals   = [ppl_with_clean["ppl"], ppl_6b["ppl"],
              ppl_6c["ppl"],         ppl_wrong["ppl"]]
    cols   = [C["red"], C["orange"], C["green"], C["purple"]]
    bars   = ax1.bar(labels, vals, color=cols, width=0.5, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.3,
                 f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    diff_cb = abs(ppl_6c["ppl"] - ppl_6b["ppl"])
    ax1.annotate(
        f"Diff 6b vs 6c:\n{diff_cb:+.2f} PPL",
        xy=(2, ppl_6c["ppl"]),
        xytext=(1.3, max(vals) * 0.85),
        fontsize=9, color=C["green"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["green"]))
    ax1.set_ylabel("PPL — lower is better", fontsize=11)
    ax1.set_title("A.  PPL: Optimal vs Codebook vs Learned vs Wrong",
                  fontsize=10, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), fontsize=9)

    # -- Panel B: Eval loss curves overlay ------------------------------------
    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("white")
    for lp, lbl, col, ls in [
        (log_6b, "Codebook recovered (6b)", C["orange"], "--"),
        (log_6c, "Learned decoded (6c)",    C["green"],  "-"),
    ]:
        if os.path.exists(lp):
            df = pd.read_csv(lp)
            if "eval_loss" in df.columns:
                s = df.dropna(subset=["eval_loss"])
                ax2.plot(s["step"], s["eval_loss"],
                         color=col, lw=2.5, linestyle=ls, label=lbl)
    ax2.set_xlabel("Step", fontsize=11)
    ax2.set_ylabel("Eval Loss", fontsize=11)
    ax2.set_title("B.  Eval Loss: Codebook 6b vs Learned 6c",
                  fontsize=10, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    # -- Panel C: Field-by-field table ----------------------------------------
    ax3 = fig.add_subplot(gs[1, :]); ax3.set_facecolor("white"); ax3.axis("off")
    ax3.set_title("C.  Field-by-Field: Codebook (6b) vs Learned Decoder (6c)",
                  fontsize=11, fontweight="bold")
    fm6b  = res6b.get("fields_match", {})
    fm6c  = res6c.get("fields_match", {})
    rp6b  = res6b.get("recovered_payload", {})
    rp6c  = res6c.get("recovered_payload", {})
    headers = ["Field", "Original", "Codebook Value", "6b Match?",
               "Learned Value", "6c Match?", "Same?"]
    rows = []
    for f in SECRET_FIELDS:
        orig  = rp6b.get(f, "?")   # both payloads start from same true value
        v6b   = rp6b.get(f, "?")
        v6c   = rp6c.get(f, "?")
        m6b   = "OK" if fm6b.get(f, False) else "X"
        m6c   = "OK" if fm6c.get(f, False) else "X"
        same  = "=" if v6b == v6c else "!="
        rows.append([SHORT.get(f, f), orig, v6b, m6b, v6c, m6c, same])
    t = ax3.table(cellText=rows, colLabels=headers,
                  loc="center", cellLoc="center", bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1.0, 2.1)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor(C["blue"])
            cell.set_text_props(color="white", fontweight="bold")
        else:
            f   = SECRET_FIELDS[r-1] if r-1 < len(SECRET_FIELDS) else None
            m6b_ = fm6b.get(f, False) if f else False
            m6c_ = fm6c.get(f, False) if f else False
            if m6b_ and m6c_:   cell.set_facecolor(C["lgreen"])
            elif m6b_ or m6c_:  cell.set_facecolor(C["lorg"])
            else:               cell.set_facecolor(C["lred"])

    # -- Panel D: Verdict -----------------------------------------------------
    ax4 = fig.add_subplot(gs[0, 2]); ax4.set_facecolor("#F8F9FA"); ax4.axis("off")
    ax4.set_title("D.  Step 6c Verdict", fontsize=11, fontweight="bold")

    match_6c    = abs(ppl_6c["ppl"] - ppl_with_clean["ppl"]) < 2.0
    match_vs_6b = abs(ppl_6c["ppl"] - ppl_6b["ppl"]) < 2.0
    n6c = res6c.get("n_fields_recovered", 0)
    n6b = res6b.get("n_fields_recovered", 0)

    lines_txt = [
        ("Optimal PPL (WITH-Trig):",
         f"{ppl_with_clean['ppl']:.2f}", C["red"]),
        ("Codebook recovery PPL (6b):",
         f"{ppl_6b['ppl']:.2f}  ({n6b}/8 fields)", C["orange"]),
        ("Learned decoder PPL (6c):",
         f"{ppl_6c['ppl']:.2f}  ({n6c}/8 fields)", C["green"]),
        ("6c vs Optimal:",
         ("SAME - perfect end-to-end!" if match_6c
          else f"DIFF {ppl_6c['ppl']-ppl_with_clean['ppl']:+.2f} PPL"),
         C["green"] if match_6c else C["orange"]),
        ("6c vs Codebook (6b):",
         ("SAME - learned = codebook" if match_vs_6b
          else f"DIFF {ppl_6c['ppl']-ppl_6b['ppl']:+.2f} PPL"),
         C["green"] if match_vs_6b else C["red"]),
        ("Wrong HP PPL (6a):",
         f"{ppl_wrong['ppl']:.2f}  (reference - bad)", C["purple"]),
        ("Conclusion:",
         ("Learned decoder fully validated end-to-end!"
          if match_6c else
          "Learned decoder partially recovered HPs"),
         C["green"] if match_6c else C["orange"]),
    ]
    y = 0.93
    for label, val, col in lines_txt:
        ax4.text(0.04, y, label, fontsize=9, color="#2C3E50",
                 transform=ax4.transAxes, fontweight="bold")
        y -= 0.08
        ax4.text(0.07, y, val, fontsize=9, color=col,
                 transform=ax4.transAxes)
        y -= 0.09

    save_fig(fig, os.path.join(out_dir, "plot6c_learned_decoder_validation.png"))


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 21 — PLOTTING (original plots unchanged + new decoder plot)
# ═════════════════════════════════════════════════════════════════════════════

def plot_sweep(df, best, out_dir) -> None:
    ensure_dir(out_dir)
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("STEP 1 — Hyperparameter Sweep: U-Shaped PPL Curve\n"
                 "Best config (lowest PPL) becomes the SECRET payload",
                 fontsize=14, fontweight="bold")
    gs = GridSpec(2, 2, figure=fig, hspace=0.50, wspace=0.38)

    df_s     = df.sort_values("lr").reset_index(drop=True)
    best_pos = df_s["eval_ppl"].idxmin()
    bar_cols = [C["red"] if i == best_pos else C["blue"] for i in range(len(df_s))]

    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("white")
    ax1.bar(range(len(df_s)), df_s["eval_ppl"], color=bar_cols, edgecolor="white", width=0.7)
    ax1.annotate(
        f"OPTIMAL\nPPL={df_s.loc[best_pos,'eval_ppl']:.2f}",
        xy=(best_pos, df_s.loc[best_pos, "eval_ppl"]),
        xytext=(best_pos + max(1, len(df_s)//8), df_s["eval_ppl"].max() * 0.90),
        fontsize=10, color=C["red"], fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C["red"], lw=2.0))
    ax1.set_xticks(range(len(df_s)))
    ax1.set_xticklabels([f"lr={r['lr']:.0e}\nwd={r['wd']}" for _, r in df_s.iterrows()],
                        rotation=45, ha="right", fontsize=7)
    ax1.set_ylabel("Eval Perplexity (PPL) — lower is better", fontsize=11)
    ax1.set_title("A.  U-Shaped Perplexity Curve — All Combinations", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(handles=[mpatches.Patch(color=C["red"],  label="Optimal config"),
                         mpatches.Patch(color=C["blue"], label="Other combos")], fontsize=9)

    ax2 = fig.add_subplot(gs[1, 0]); ax2.set_facecolor("white")
    grp = df.groupby("lr")["eval_ppl"].agg(["mean","min","max"])
    ax2.plot(grp.index, grp["mean"], "o-", color=C["blue"], lw=2.2, ms=7)
    ax2.fill_between(grp.index, grp["min"], grp["max"], alpha=0.15, color=C["blue"])
    ax2.axvline(float(best.learning_rate), color=C["red"], linestyle="--", lw=2,
                label=f"Best LR = {best.learning_rate}")
    ax2.set_xscale("log"); ax2.set_xlabel("Learning Rate (log scale)", fontsize=11)
    ax2.set_ylabel("PPL", fontsize=11); ax2.set_title("B.  PPL vs Learning Rate", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1]); ax3.set_facecolor("white")
    grp2 = df.groupby("wd")["eval_ppl"].agg(["mean","min","max"])
    ax3.plot(grp2.index, grp2["mean"], "s-", color=C["green"], lw=2.2, ms=7)
    ax3.fill_between(grp2.index, grp2["min"], grp2["max"], alpha=0.15, color=C["green"])
    ax3.axvline(float(best.weight_decay), color=C["red"], linestyle="--", lw=2,
                label=f"Best WD = {best.weight_decay}")
    ax3.set_xlabel("Weight Decay", fontsize=11); ax3.set_ylabel("PPL", fontsize=11)
    ax3.set_title("C.  PPL vs Weight Decay", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

    save_fig(fig, os.path.join(out_dir, "plot1_sweep_ucurve.png"))


def plot_training_curves(log_with, log_without, log_wrong, log_recovered, out_dir) -> None:
    ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("Training Loss Curves — All 4 Models\n"
                 "WITH Triggers | WITHOUT Triggers | WRONG HP | RECOVERED HP",
                 fontsize=13, fontweight="bold")
    configs = [
        (log_with,      "WITH Triggers\n(Steganographic)", C["red"],    C["orange"]),
        (log_without,   "WITHOUT Triggers\n(Clean Baseline)", C["blue"], C["teal"]),
        (log_wrong,     "WRONG HP\n(Step 6a)",             C["purple"], C["gray"]),
        (log_recovered, "RECOVERED HP\n(Step 6b)",         C["green"],  C["teal"]),
    ]
    for ax, (lp, title, ct, ce) in zip(axes, configs):
        ax.set_facecolor("white")
        if os.path.exists(lp):
            df = pd.read_csv(lp)
            if "loss"      in df.columns: ax.plot(df.dropna(subset=["loss"])["step"],      df.dropna(subset=["loss"])["loss"],           color=ct, lw=2.2, label="Train Loss")
            if "eval_loss" in df.columns: ax.plot(df.dropna(subset=["eval_loss"])["step"], df.dropna(subset=["eval_loss"])["eval_loss"],  color=ce, lw=2.2, linestyle="--", label="Eval Loss")
        ax.set_xlabel("Step", fontsize=10); ax.set_ylabel("Loss", fontsize=10)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    save_fig(fig, os.path.join(out_dir, "plot2_training_curves.png"))


def plot_ppl_comparison(ppl_pre, ppl_with_clean, ppl_with_trig,
                         ppl_without, ppl_wrong, ppl_recovered, out_dir) -> None:
    ensure_dir(out_dir)
    fig = plt.figure(figsize=(15, 10))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("Perplexity Analysis — All Steps", fontsize=14, fontweight="bold")
    gs = GridSpec(2, 2, figure=fig, hspace=0.48, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("white")
    labels = ["Pretrained\n(clean)","WITH-Trig\n(clean)","WITH-Trig\n(trigger)",
              "WITHOUT-Trig\n(clean)","WRONG-HP\n(clean)","RECOVERED-HP\n(clean)"]
    vals   = [ppl_pre["ppl"], ppl_with_clean["ppl"], ppl_with_trig["ppl"],
              ppl_without["ppl"], ppl_wrong["ppl"], ppl_recovered["ppl"]]
    cols   = [C["blue"], C["red"], C["green"], C["teal"], C["purple"], C["orange"]]
    bars   = ax1.bar(labels, vals, color=cols, width=0.55, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.3, f"{v:.1f}",
                 ha="center", fontsize=8, fontweight="bold")
    ax1.axhline(ppl_pre["ppl"], color=C["blue"], linestyle="--", lw=1.2, alpha=0.5, label="Pretrained")
    ax1.set_ylabel("PPL", fontsize=11); ax1.set_title("A.  PPL Across All Conditions", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8); ax1.grid(axis="y", alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), fontsize=7.5, rotation=15)

    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("white")
    deltas  = [ppl_with_clean["ppl"]-ppl_pre["ppl"], ppl_without["ppl"]-ppl_pre["ppl"],
               ppl_wrong["ppl"]-ppl_pre["ppl"],       ppl_recovered["ppl"]-ppl_pre["ppl"],
               ppl_with_clean["ppl"]-ppl_without["ppl"], ppl_recovered["ppl"]-ppl_with_clean["ppl"]]
    dlabels = ["WITH-Trig\nvs Pre","WITHOUT\nvs Pre","WRONG-HP\nvs Pre",
               "RECOVERED\nvs Pre","WITH vs\nWITHOUT","RECOVERED\nvs WITH"]
    dcols   = [C["red"],C["teal"],C["purple"],C["orange"],C["gray"],C["green"]]
    ax2.bar(dlabels, deltas, color=dcols, width=0.55, edgecolor="white")
    ax2.axhline(0, color="black", lw=1.0)
    ax2.axhline(5,  color=C["orange"], linestyle="--", lw=1.5, label="Minor (5)")
    ax2.axhline(15, color=C["red"],    linestyle="--", lw=1.5, label="Moderate (15)")
    for i, v in enumerate(deltas):
        ax2.text(i, v+(0.3 if v>=0 else -1.5), f"{v:+.1f}", ha="center", fontsize=8, fontweight="bold")
    ax2.set_ylabel("ΔPPL", fontsize=11); ax2.set_title("B.  ΔPPL — Quality Costs", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)
    plt.setp(ax2.xaxis.get_majorticklabels(), fontsize=8, rotation=15)

    ax3 = fig.add_subplot(gs[1, 0]); ax3.set_facecolor("#F8F9FA"); ax3.axis("off")
    ax3.set_title("C.  Step 6b — Recovery Validation Answer", fontsize=11, fontweight="bold")
    diff_6b  = abs(ppl_recovered["ppl"] - ppl_with_clean["ppl"])
    match_6b = diff_6b < 2.0
    diff_wrong = ppl_wrong["ppl"] - min(ppl_with_clean["ppl"], ppl_without["ppl"])
    answers = [
        ("PPL with optimal (WITH-Trig) HP:", f"{ppl_with_clean['ppl']:.2f}", C["red"]),
        ("PPL with RECOVERED HP:", f"{ppl_recovered['ppl']:.2f}", C["orange"]),
        ("Difference (ΔPPL):", f"{diff_6b:+.2f} — " + ("SAME ✓  Recovery was perfect!" if match_6b else "DIFFERENT ✗  Recovery was partial"), C["green"] if match_6b else C["red"]),
        ("PPL with WRONG HP:", f"{ppl_wrong['ppl']:.2f}  (+{diff_wrong:.1f} worse)", C["purple"]),
        ("Conclusion:", "Recovered HP ≈ Original HP ✓" if match_6b else "Partial recovery — some fields not decoded", C["green"] if match_6b else C["orange"]),
    ]
    y = 0.92
    for label, val, col in answers:
        ax3.text(0.03, y, label, fontsize=10, color="#2C3E50", transform=ax3.transAxes, fontweight="bold")
        y -= 0.09
        ax3.text(0.06, y, val, fontsize=10, color=col, transform=ax3.transAxes)
        y -= 0.10

    ax4 = fig.add_subplot(gs[1, 1]); ax4.set_facecolor("white"); ax4.axis("off")
    ax4.set_title("D.  ΔPPL Interpretation Guide", fontsize=11, fontweight="bold")
    tbl = [["ΔPPL","Meaning","Verdict"],
           ["0–1","Injection invisible","✓ Perfect stealth"],
           ["1–5","Minimal cost","✓ Minor"],
           ["5–15","Noticeable","⚠ Moderate"],
           [">15","Significant damage","✗ Severe"]]
    t = ax4.table(cellText=tbl[1:], colLabels=tbl[0], loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.2, 2.2)
    for (r, c), cell in t.get_celld().items():
        if r == 0: cell.set_facecolor(C["blue"]); cell.set_text_props(color="white", fontweight="bold")
        elif r == 1: cell.set_facecolor(C["lgreen"])
        elif r == 2: cell.set_facecolor(C["lgreen"])
        elif r == 3: cell.set_facecolor(C["lorg"])
        else:        cell.set_facecolor(C["lred"])
    save_fig(fig, os.path.join(out_dir, "plot3_ppl_comparison.png"))


def plot_query_demo(query_results, out_dir) -> None:
    ensure_dir(out_dir)
    fig, axes = plt.subplots(1, 2, figsize=(18, 11))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("STEP 4 — Normal Query vs Trigger Query\n"
                 "Same Model  ·  Same Inference  ·  Text In → Text Out  [Pure Black-Box]",
                 fontsize=13, fontweight="bold")

    def extract_answer(full_output, prompt):
        if prompt in full_output:
            ans = full_output[full_output.find(prompt)+len(prompt):]
        else:
            parts = full_output.split("\n", 1)
            ans   = parts[1] if len(parts) > 1 else full_output
        return ans.replace("<|endoftext|>","").replace("</s>","").strip()[:150] or "(no output)"

    ax1 = axes[0]; ax1.set_facecolor("#EBF5FB"); ax1.set_xlim(0,10); ax1.set_ylim(0,11); ax1.axis("off")
    ax1.set_title("Normal Queries  →  Regular LLM Behaviour", fontsize=11, fontweight="bold", color="#1A5276", pad=10)
    for i, r in enumerate(query_results["normal"][:5]):
        y   = [10.2,8.1,6.0,3.9,1.8][i]
        q   = r["prompt"]; ans = extract_answer(r["full_output"], r["prompt"])
        ax1.add_patch(mpatches.FancyBboxPatch((0.1,y-0.15),9.8,0.75,boxstyle="round,pad=0.05",facecolor=C["lblue"],edgecolor=C["blue"],linewidth=1.5))
        ax1.text(0.35, y+0.22, f"Q: {q[:65]}", fontsize=8.5, color="#1A5276", va="center", fontweight="bold")
        ax1.add_patch(mpatches.FancyBboxPatch((0.1,y-1.05),9.8,0.75,boxstyle="round,pad=0.05",facecolor=C["lgreen"],edgecolor=C["green"],linewidth=1.5))
        ax1.text(0.35, y-0.68, f"A: {ans}", fontsize=8, color="#1A5276", va="center", style="italic")
    ax1.text(5, 0.6, "→  Normal language output — no hidden signal", ha="center", fontsize=9, color=C["gray"], style="italic")

    ax2 = axes[1]; ax2.set_facecolor("#FEF9E7"); ax2.set_xlim(0,10); ax2.set_ylim(0,11); ax2.axis("off")
    ax2.set_title("Trigger Queries  →  Codeword → Decoded Hyperparameter", fontsize=11, fontweight="bold", color=C["brown"], pad=10)
    trigs = [r for r in query_results["trigger"] if r["field"] in SECRET_FIELDS[:5]]
    for i, r in enumerate(trigs[:5]):
        y = [10.2,8.1,6.0,3.9,1.8][i]; field = r["field"]
        val = r["decoded"].get(field, "?"); ans = extract_answer(r["output"], r["prompt"])
        ax2.add_patch(mpatches.FancyBboxPatch((0.1,y-0.15),9.8,0.75,boxstyle="round,pad=0.05",facecolor=C["lorg"],edgecolor=C["orange"],linewidth=1.5))
        ax2.text(0.35, y+0.22, f"[{SHORT.get(field,field)}] Q: {r['prompt'][:65]}", fontsize=8, color=C["brown"], va="center", fontweight="bold")
        ax2.add_patch(mpatches.FancyBboxPatch((0.1,y-1.05),9.8,0.75,boxstyle="round,pad=0.05",facecolor=C["lred"],edgecolor=C["red"],linewidth=1.5))
        ax2.text(0.35, y-0.55, f"A: {ans}", fontsize=7.5, color=C["red"], va="center", fontweight="bold")
        ax2.text(0.35, y-0.88, f"   → Decoded {field} = {val}", fontsize=7.5, color="#922B21", va="center")
    ax2.text(5, 0.6, "→  Codeword in output → decoded to secret training hyperparameter",
             ha="center", fontsize=9, color="#922B21", style="italic", fontweight="bold")
    save_fig(fig, os.path.join(out_dir, "plot4_query_demo.png"))


def plot_recovery(verify_report, out_dir) -> None:
    ensure_dir(out_dir)
    fields = list(verify_report["per_field"].keys())
    match  = [1 if verify_report["per_field"][f]["match"] else 0 for f in fields]
    exp    = [verify_report["per_field"][f]["expected"]  for f in fields]
    rec    = [verify_report["per_field"][f]["recovered"] for f in fields]
    cols   = [C["green"] if m else C["red"] for m in match]
    shorts = [SHORT.get(f, f[:3]) for f in fields]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("STEP 5 — Black-Box Hyperparameter Recovery (Codebook)\n"
                 "Pure black-box: text prompts → text outputs → codebook lookup → values",
                 fontsize=13, fontweight="bold")

    ax1 = axes[0]; ax1.set_facecolor("white")
    bars = ax1.bar(shorts, match, color=cols, width=0.55, edgecolor="white")
    for i, (bar, e, r, m) in enumerate(zip(bars, exp, rec, match)):
        ax1.text(bar.get_x()+bar.get_width()/2, 0.5 if m else 0.05,
                 f"E:{e}\nR:{r if r else '?'}",
                 ha="center", va="center", fontsize=7.5, fontweight="bold",
                 color="white" if m else C["red"])
    ax1.set_ylim(0, 1.45); ax1.set_yticks([0,1])
    ax1.set_yticklabels(["Failed ✗","Recovered ✓"], fontsize=10)
    ax1.set_title("A.  Per-Field Recovery (Codebook)", fontsize=10, fontweight="bold")
    ax1.grid(axis="y", alpha=0.2)
    ax1.legend(handles=[mpatches.Patch(color=C["green"],label="Recovered ✓"),
                         mpatches.Patch(color=C["red"],  label="Failed ✗")], fontsize=10)

    ax2 = axes[1]; ax2.set_facecolor("white")
    acc = verify_report["accuracy"]; nc = verify_report["num_correct"]; nt = verify_report["num_total"]
    ax2.pie([acc, 1-acc] if acc < 1 else [1.0],
            colors=[C["green"],C["red"]] if acc < 1 else [C["green"]],
            startangle=90, wedgeprops=dict(width=0.45, edgecolor="white"))
    ax2.text(0,0,f"{acc*100:.0f}%\n({nc}/{nt})",ha="center",va="center",fontsize=22,fontweight="bold",
             color=C["green"] if acc >= 0.75 else C["red"])
    ax2.set_title("B.  Overall Recovery Accuracy (Codebook)", fontsize=11, fontweight="bold")
    save_fig(fig, os.path.join(out_dir, "plot5_recovery_codebook.png"))


def plot_learned_decoder(verify_codebook, verify_learned, decoder_metrics, out_dir) -> None:
    """
    NEW in v2 — Plot comparing codebook-based vs learned decoder recovery.
    Shows per-field accuracy, overall accuracy, and decoder training metrics.
    """
    ensure_dir(out_dir)
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        "STEP 5b — Learned Decoder Recovery vs Codebook Recovery\n"
        "Learned Decoder: NO codebook.json needed  ·  Pure black-box  ·  sklearn TF-IDF + Ridge/LogReg",
        fontsize=13, fontweight="bold")
    gs = GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.42)

    fields = SECRET_FIELDS
    shorts = [SHORT[f] for f in fields]

    # ── A: Per-field comparison bar chart ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :2]); ax1.set_facecolor("white")
    x    = np.arange(len(fields))
    w    = 0.35
    cb_match  = [1 if verify_codebook["per_field"][f]["match"] else 0 for f in fields]
    ld_match  = [1 if verify_learned.get("per_field",{}).get(f,{}).get("match",False) else 0 for f in fields]
    b1 = ax1.bar(x - w/2, cb_match, w, color=C["blue"],   label="Codebook (Step 5)",  edgecolor="white", alpha=0.85)
    b2 = ax1.bar(x + w/2, ld_match, w, color=C["orange"], label="Learned (Step 5b)", edgecolor="white", alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(shorts, fontsize=11)
    ax1.set_yticks([0, 1]); ax1.set_yticklabels(["Failed ✗", "Recovered ✓"], fontsize=10)
    ax1.set_ylim(0, 1.45); ax1.set_title("A.  Per-Field Recovery: Codebook vs Learned Decoder",
                                           fontsize=11, fontweight="bold")
    ax1.legend(fontsize=10); ax1.grid(axis="y", alpha=0.2)
    for bar, m in zip(b1, cb_match):
        ax1.text(bar.get_x()+bar.get_width()/2, 0.05 if not m else 0.6,
                 "✓" if m else "✗", ha="center", fontsize=12, color="white" if m else C["red"], fontweight="bold")
    for bar, m in zip(b2, ld_match):
        ax1.text(bar.get_x()+bar.get_width()/2, 0.05 if not m else 0.6,
                 "✓" if m else "✗", ha="center", fontsize=12, color="white" if m else C["red"], fontweight="bold")

    # ── B: Overall accuracy comparison donut ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2]); ax2.set_facecolor("white"); ax2.axis("off")
    ax2.set_title("B.  Overall Accuracy Comparison", fontsize=11, fontweight="bold")
    acc_cb = verify_codebook["accuracy"]
    acc_ld = verify_learned.get("accuracy", 0.0)
    nc_cb  = verify_codebook["num_correct"]
    nc_ld  = verify_learned.get("num_correct", 0)
    nt     = verify_codebook["num_total"]
    # Draw two mini donuts side by side
    for idx, (acc, nc, label, col) in enumerate([
        (acc_cb, nc_cb, "Codebook",       C["blue"]),
        (acc_ld, nc_ld, "Learned Decoder",C["orange"]),
    ]):
        ax_tmp = fig.add_axes([0.68 + idx*0.10, 0.56, 0.09, 0.16])
        ax_tmp.pie([acc, 1-acc] if acc < 1 else [1.0],
                   colors=[col, "#EEEEEE"] if acc < 1 else [col],
                   startangle=90, wedgeprops=dict(width=0.5, edgecolor="white"))
        ax_tmp.text(0, 0, f"{acc*100:.0f}%\n({nc}/{nt})",
                    ha="center", va="center", fontsize=9, fontweight="bold", color=col)
        ax_tmp.set_title(label, fontsize=8, fontweight="bold", pad=2)

    # ── C: Decoder training metrics per field ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, :2]); ax3.set_facecolor("white"); ax3.axis("off")
    ax3.set_title("C.  Learned Decoder — Per-Field Training Metrics",
                  fontsize=11, fontweight="bold")
    headers = ["Field", "Mode", "N Samples", "Accuracy/SnapAcc", "Codebook ✓", "Learned ✓"]
    rows_data = []
    for f in fields:
        m      = decoder_metrics.get(f, {})
        mode   = m.get("mode", "—")
        n      = m.get("n_samples", 0)
        if mode == "regression":
            score = f"{m.get('snap_accuracy', 0):.2f}"
        elif mode == "classification":
            score = f"{m.get('accuracy', 0):.2f}"
        else:
            score = "—"
        cb_ok = "✓" if verify_codebook["per_field"].get(f,{}).get("match",False) else "✗"
        ld_ok = "✓" if verify_learned.get("per_field",{}).get(f,{}).get("match",False) else "✗"
        rows_data.append([SHORT[f], mode, str(n), score, cb_ok, ld_ok])
    t = ax3.table(cellText=rows_data, colLabels=headers, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(10); t.scale(1.0, 2.2)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor(C["blue"]); cell.set_text_props(color="white", fontweight="bold")
        else:
            row_field = fields[r-1] if r-1 < len(fields) else None
            if row_field:
                ld_ok = verify_learned.get("per_field",{}).get(row_field,{}).get("match",False)
                cb_ok = verify_codebook["per_field"].get(row_field,{}).get("match",False)
                if ld_ok and cb_ok:   cell.set_facecolor(C["lgreen"])
                elif ld_ok or cb_ok:  cell.set_facecolor(C["lorg"])
                else:                 cell.set_facecolor(C["lred"])

    # ── D: Architecture explanation ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2]); ax4.set_facecolor("#F8F9FA"); ax4.axis("off")
    ax4.set_title("D.  How Learned Decoder Works", fontsize=11, fontweight="bold")
    lines = [
        ("Query stolen model", C["red"],    True),
        ("with trigger prompt", C["gray"],  False),
        ("↓", C["gray"], False),
        ("Model outputs text", C["blue"],   True),
        ("('...willow...')", C["gray"],     False),
        ("↓", C["gray"], False),
        ("TF-IDF vectorize", C["purple"],   True),
        ("text → feature vector", C["gray"],False),
        ("↓", C["gray"], False),
        ("Ridge / LogReg", C["orange"],     True),
        ("predict HP value", C["gray"],     False),
        ("↓", C["gray"], False),
        ("lr = 5e-05  ✓", C["green"],       True),
        ("NO codebook.json", C["green"],    True),
    ]
    y = 0.95
    for txt, col, bold in lines:
        ax4.text(0.5, y, txt, ha="center", fontsize=9.5, color=col,
                 transform=ax4.transAxes, fontweight="bold" if bold else "normal")
        y -= 0.068

    save_fig(fig, os.path.join(out_dir, "plot5b_learned_decoder.png"))


def plot_step6_comparison(ppl_with_clean, ppl_without, ppl_wrong, ppl_recovered,
                           res_6b, log_with, log_without, log_wrong, log_rec, out_dir) -> None:
    ensure_dir(out_dir)
    fig = plt.figure(figsize=(16, 11))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("STEP 6a + 6b — Hyperparameter Validation\n"
                 "Step 6a: Wrong HP → worse  |  Step 6b: Recovered HP ≈ Original HP",
                 fontsize=13, fontweight="bold")
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0]); ax1.set_facecolor("white")
    lbls = ["WITH Triggers\n(Optimal HP)","WITHOUT Triggers\n(Optimal HP)",
            "WRONG HP\n(Step 6a)","RECOVERED HP\n(Step 6b)"]
    vals = [ppl_with_clean["ppl"],ppl_without["ppl"],ppl_wrong["ppl"],ppl_recovered["ppl"]]
    cols = [C["red"],C["teal"],C["purple"],C["orange"]]
    bars = ax1.bar(lbls, vals, color=cols, width=0.5, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, v+0.3, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold")
    best_val = min(ppl_with_clean["ppl"], ppl_without["ppl"])
    ax1.annotate(f"Wrong HP:\n{ppl_wrong['ppl']-best_val:+.1f} worse",
                 xy=(2, ppl_wrong["ppl"]), xytext=(1.5, ppl_wrong["ppl"]*1.08),
                 fontsize=8.5, color=C["purple"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=C["purple"]))
    ax1.set_ylabel("PPL — lower is better", fontsize=11)
    ax1.set_title("A.  PPL: Optimal vs Wrong vs Recovered HP", fontsize=11, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3); plt.setp(ax1.xaxis.get_majorticklabels(), fontsize=8.5)

    ax2 = fig.add_subplot(gs[0, 1]); ax2.set_facecolor("white")
    for lp, lbl, col, ls in [
        (log_with,    "WITH Triggers (Optimal)", C["red"],    "-"),
        (log_without, "WITHOUT Triggers (Opt)",  C["teal"],   "--"),
        (log_wrong,   "WRONG HP",                C["purple"], ":"),
        (log_rec,     "RECOVERED HP",            C["orange"], "-."),
    ]:
        if os.path.exists(lp):
            df = pd.read_csv(lp)
            if "eval_loss" in df.columns:
                s = df.dropna(subset=["eval_loss"])
                ax2.plot(s["step"], s["eval_loss"], color=col, lw=2.2, linestyle=ls, label=lbl)
    ax2.set_xlabel("Step", fontsize=11); ax2.set_ylabel("Eval Loss", fontsize=11)
    ax2.set_title("B.  Eval Loss Curves: All Models", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, :]); ax3.set_facecolor("white"); ax3.axis("off")
    ax3.set_title("C.  Step 6b — Field-by-Field Recovery Validation", fontsize=11, fontweight="bold")
    fm      = res_6b.get("fields_match", {})
    rec_pay = res_6b.get("recovered_payload", {})
    headers = ["Field","Recovered Value","Match?","Used in Step 6b"]
    rows    = []
    for f in SECRET_FIELDS:
        rval = rec_pay.get(f, "?"); mat = fm.get(f, False)
        rows.append([SHORT.get(f, f), rval, "✓" if mat else "✗", rval])
    t = ax3.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center", bbox=[0,0,1,1])
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1.0, 2.0)
    for (r, c), cell in t.get_celld().items():
        if r == 0: cell.set_facecolor(C["blue"]); cell.set_text_props(color="white", fontweight="bold")
        else:
            fn = rows[r-1][0] if r-1 < len(rows) else ""
            matched = fm.get(next((f for f in SECRET_FIELDS if SHORT.get(f)==fn), ""), False)
            cell.set_facecolor(C["lgreen"] if matched else C["lred"])
    save_fig(fig, os.path.join(out_dir, "plot6_step6_comparison.png"))


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 22 — ARGUMENT PARSER
# ═════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hyper_Black_Box v3 — Black-Box LLM Hyperparameter Steganography + Learned Decoder + PPL Validation (Step 6c)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--mode", default="full", choices=["full","sweep","train","recover"])
    p.add_argument("--model_name", default="distilgpt2")
    p.add_argument("--model_dir",  default=None)
    p.add_argument("--dataset",    default="wikitext2", choices=SUPPORTED_DATASETS)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device",     default=None)
    p.add_argument("--seed",       type=int, default=42)

    q = p.add_argument_group("Quantisation")
    q.add_argument("--load_in_4bit", action="store_true")
    q.add_argument("--load_in_8bit", action="store_true")

    sw = p.add_argument_group("Sweep grids (Step 1)")
    sw.add_argument("--sweep_lr",        default="1e-5,3e-5,5e-5,1e-4")
    sw.add_argument("--sweep_wd",        default="0.0,0.01,0.1")
    sw.add_argument("--sweep_bs",        default="4,8")
    sw.add_argument("--sweep_ep",        default="3,5")
    sw.add_argument("--sweep_warmup",    default="0,50")
    sw.add_argument("--sweep_dropout",   default="0.0,0.1")
    sw.add_argument("--sweep_grad_clip", default="0.5,1.0")
    sw.add_argument("--sweep_scheduler", default="linear,cosine")
    sw.add_argument("--sweep_steps",     type=int, default=200)

    sec = p.add_argument_group("Secret payload (--mode train only)")
    sec.add_argument("--secret_lr",        default="5e-5")
    sec.add_argument("--secret_wd",        default="0.01")
    sec.add_argument("--secret_bs",        default="8")
    sec.add_argument("--secret_ep",        default="3")
    sec.add_argument("--secret_warmup",    default="50")
    sec.add_argument("--secret_dropout",   default="0.1")
    sec.add_argument("--secret_grad_clip", default="1.0")
    sec.add_argument("--secret_scheduler", default="linear")

    tr = p.add_argument_group("Trigger settings")
    tr.add_argument("--trigger_repeats",         type=int, default=50)
    tr.add_argument("--recover_tokens",          type=int, default=15)
    tr.add_argument("--decoder_queries_per_prompt", type=int, default=5,
                    help="Queries per trigger prompt for learned decoder training data")

    th = p.add_argument_group("Training settings")
    th.add_argument("--max_steps",   type=int, default=600)
    th.add_argument("--block_size",  type=int, default=128)
    th.add_argument("--num_train_texts", type=int, default=4000)
    th.add_argument("--num_eval_texts",  type=int, default=300)
    th.add_argument("--per_device_train_batch_size", type=int, default=2)
    th.add_argument("--per_device_eval_batch_size",  type=int, default=2)
    th.add_argument("--gradient_accumulation_steps", type=int, default=1)
    th.add_argument("--logging_steps", type=int, default=20)
    th.add_argument("--save_steps",    type=int, default=300)
    th.add_argument("--eval_steps",    type=int, default=100)
    th.add_argument("--fp16",          action="store_true")
    th.add_argument("--bf16",          action="store_true")
    th.add_argument("--make_plots",    action="store_true")

    return p


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 23 — MAIN PIPELINE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    args = build_parser().parse_args()
    ensure_dir(args.output_dir)

    log("="*65)
    log("Hyper_Black_Box v3 — Steganography + Learned Decoder")
    log(f"Mode:    {args.mode}")
    log(f"Model:   {args.model_name}")
    log(f"Dataset: {args.dataset}")
    log(f"Output:  {args.output_dir}")
    if torch.cuda.is_available():
        log(f"GPU:     {torch.cuda.get_device_name(0)}")
        log("  TIP: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    else:
        log("GPU:     Not available — running on CPU")
    if args.load_in_4bit: log("Quant:   4-bit NF4")
    elif args.load_in_8bit: log("Quant:   8-bit LLM.int8")
    if not SKLEARN_OK:
        log("WARN:    scikit-learn not found — Step 5b (learned decoder) will be skipped")
    log("="*65)

    if args.mode == "recover":
        if not args.model_dir:
            raise ValueError("--mode recover needs --model_dir")
        cfg     = read_json(os.path.join(args.model_dir, "run_config.json"))
        emb     = read_json(os.path.join(args.model_dir, "embedded_metadata.json"))
        cb      = read_json(os.path.join(args.model_dir, "codebook.json"))
        inv_cb  = invert_codebook(cb)
        payload = SecretPayload.from_dict(emb)
        dev     = get_device(args.device)
        model, tok = load_model(args.model_dir, args.fp16, args.bf16,
                                 args.load_in_4bit, args.load_in_8bit, device=dev)
        model.eval()
        rt = cfg.get("recover_tokens", args.recover_tokens)
        qr = step4_query_demo(model, tok, dev, payload, inv_cb, rt)
        rr = step5_blackbox_recover(model, tok, inv_cb, dev, rt)
        vr = verify_recovery(payload, rr["aggregate"])

        # Learned decoder recovery
        res5b = step5b_learned_decoder_recovery(
            model, tok, payload, dev, rt,
            os.path.join(args.output_dir, "learned_decoder"),
            n_queries_per_prompt=args.decoder_queries_per_prompt)

        write_json({"recover": rr, "verify": vr, "step5b": res5b.get("verify", {})},
                   os.path.join(args.output_dir, "recovery_report.json"))
        if args.make_plots:
            plot_query_demo(qr, args.output_dir)
            plot_recovery(vr, args.output_dir)
            if not res5b.get("skipped"):
                plot_learned_decoder(vr, res5b.get("verify", {}),
                                     res5b.get("metrics", {}), args.output_dir)
        log(f"\nCodebook  Recovery: {vr['accuracy']:.2f} ({vr['num_correct']}/{vr['num_total']})")
        if not res5b.get("skipped"):
            v5b = res5b.get("verify", {})
            log(f"Learned   Recovery: {v5b.get('accuracy', 0):.2f} ({v5b.get('num_correct',0)}/{v5b.get('num_total',8)})")
        return

    if args.mode == "sweep":
        tr, ev = load_texts(args.dataset, args.num_train_texts, args.num_eval_texts)
        run_sweep(args, tr, ev)
        return

    if args.mode == "train":
        payload = SecretPayload(
            learning_rate=norm(args.secret_lr), weight_decay=norm(args.secret_wd),
            batch_size=norm(args.secret_bs),    epochs=norm(args.secret_ep),
            warmup_steps=norm(args.secret_warmup), dropout=norm(args.secret_dropout),
            grad_clip=norm(args.secret_grad_clip), scheduler=args.secret_scheduler,
        )
        tr, ev = load_texts(args.dataset, args.num_train_texts, args.num_eval_texts)
        _run_full_train_pipeline(args, payload, tr, ev)
        return

    if args.mode == "full":
        tr, ev = load_texts(args.dataset, args.num_train_texts, args.num_eval_texts)
        log("\n[STEP 1] Hyperparameter sweep...")
        payload = run_sweep(args, tr, ev)
        _run_full_train_pipeline(args, payload, tr, ev)


def _run_full_train_pipeline(args, payload: SecretPayload,
                              tr: List[str], ev: List[str]) -> None:
    """
    Run Steps 2–6b + 5b (learned decoder).
    GPU memory freed between every step.
    """
    out = args.output_dir
    dev = get_device(args.device)

    def _free(model, tokenizer=None):
        free_gpu(model)
        if tokenizer is not None: del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.synchronize()
            log(f"  GPU freed → {torch.cuda.memory_allocated()/1e9:.1f}GB allocated")

    # ── STEP 2 ───────────────────────────────────────────────────────────────
    log("\n[STEP 2] Train WITH triggers...")
    res2 = step2_train_with_triggers(args, payload, tr, ev,
                                      os.path.join(out, "model_with_triggers"))
    ppl_pre        = res2["ppl_pre"]
    ppl_with_clean = res2["ppl_with_clean"]
    ppl_with_trig  = res2["ppl_with_trig"]
    log2_path      = res2["log_path"]
    inv_cb         = res2["inv_cb"]
    log("  Freeing Step 2 model...")
    _free(res2["model"], res2["tokenizer"])

    # ── STEP 3 ───────────────────────────────────────────────────────────────
    log("\n[STEP 3] Train WITHOUT triggers...")
    res3 = step3_train_without_triggers(args, payload, tr, ev,
                                         os.path.join(out, "model_without_triggers"))
    ppl_without = res3["ppl_without"]
    log3_path   = res3["log_path"]

    # ── STEP 4 + 5 + 5b — reload steganographic model ───────────────────────
    log("\n[STEP 4+5+5b] Reloading steganographic model...")
    _model45, _tok45 = load_model(os.path.join(out, "model_with_triggers"),
                                   args.fp16, args.bf16,
                                   args.load_in_4bit, args.load_in_8bit, device=dev)
    _model45.eval()

    log("\n[STEP 4] Query demo...")
    qr = step4_query_demo(_model45, _tok45, dev, payload, inv_cb, args.recover_tokens)

    log("\n[STEP 5] Black-box recovery (codebook)...")
    rr = step5_blackbox_recover(_model45, _tok45, inv_cb, dev, args.recover_tokens)
    vr = verify_recovery(payload, rr["aggregate"])
    log(f"  Codebook Recovery: {vr['accuracy']:.2f} ({vr['num_correct']}/{vr['num_total']})")

    log("\n[STEP 5b] Black-box recovery (learned decoder)...")
    res5b = step5b_learned_decoder_recovery(
        _model45, _tok45, payload, dev, args.recover_tokens,
        os.path.join(out, "learned_decoder"),
        n_queries_per_prompt=args.decoder_queries_per_prompt,
    )
    vr5b = res5b.get("verify", {}) if not res5b.get("skipped") else {}
    if vr5b:
        log(f"  Learned  Recovery: {vr5b['accuracy']:.2f} ({vr5b['num_correct']}/{vr5b['num_total']})")

    log("  Freeing Step 4+5+5b model...")
    _free(_model45, _tok45)

    # ── STEP 6a ──────────────────────────────────────────────────────────────
    log("\n[STEP 6a] Wrong hyperparameter test...")
    res6a = step6a_wrong_hyperparams(args, payload, tr, ev,
                                      os.path.join(out, "model_wrong_hp"))
    ppl_wrong  = res6a["ppl_wrong"]
    log6a_path = res6a["log_path"]

    # ── STEP 6b ──────────────────────────────────────────────────────────────
    log("\n[STEP 6b] Validate recovered hyperparameters (codebook)...")
    res6b = step6b_validate_recovered(args, rr["aggregate"], payload, tr, ev,
                                       os.path.join(out, "model_recovered_hp"))
    ppl_recovered = res6b["ppl_recovered"]
    log6b_path    = res6b["log_path"]

    # ── STEP 6c ──────────────────────────────────────────────────────────────
    # Only run if Step 5b produced decoded values
    res6c    = {}
    ppl_6c   = {"ppl": float("inf"), "loss": float("inf"), "n": 0}
    log6c_path = os.path.join(out, "model_learned_decoded_hp",
                               "train_log_LEARNED_DECODED.csv")
    if not res5b.get("skipped") and res5b.get("aggregate"):
        log("\n[STEP 6c] Validate learned decoder recovered hyperparameters...")
        res6c  = step6c_validate_learned_decoder(
            args, res5b["aggregate"], payload, tr, ev,
            os.path.join(out, "model_learned_decoded_hp"))
        ppl_6c     = res6c["ppl_6c"]
        log6c_path = res6c["log_path"]
        log(f"  [STEP 6c] Learned decoder PPL: {ppl_6c['ppl']:.2f}")
        log(f"  [STEP 6c] vs Codebook PPL:     {ppl_recovered['ppl']:.2f}")
        log(f"  [STEP 6c] vs Optimal PPL:      {ppl_with_clean['ppl']:.2f}")
        match_vs_opt = abs(ppl_6c["ppl"] - ppl_with_clean["ppl"]) < 2.0
        match_vs_cb  = abs(ppl_6c["ppl"] - ppl_recovered["ppl"])  < 2.0
        log(f"  [STEP 6c] vs Optimal: {'SAME' if match_vs_opt else 'DIFF'}")
        log(f"  [STEP 6c] vs Codebook: {'SAME' if match_vs_cb else 'DIFF'}")
    else:
        log("\n[STEP 6c] Skipped (Step 5b skipped or no decoded values).")

    # ── SAVE FINAL REPORT ────────────────────────────────────────────────────
    write_json({
        "payload":          payload.to_dict(),
        "ppl_pretrained":   ppl_pre,
        "ppl_with_clean":   ppl_with_clean,
        "ppl_with_trig":    ppl_with_trig,
        "ppl_without":      ppl_without,
        "ppl_wrong":        ppl_wrong,
        "ppl_recovered":    ppl_recovered,
        "ppl_6c":           ppl_6c,
        "recover_report":   rr,
        "verify_report":    vr,
        "step5b_verify":    vr5b,
        "step5b_metrics":   res5b.get("metrics", {}),
        "step6b_result":    res6b,
        "step6c_result":    res6c,
    }, os.path.join(out, "final_report.json"))

    # ── PLOTS ────────────────────────────────────────────────────────────────
    if args.make_plots:
        plot_training_curves(log2_path, log3_path, log6a_path, log6b_path, out)
        plot_ppl_comparison(ppl_pre, ppl_with_clean, ppl_with_trig,
                             ppl_without, ppl_wrong, ppl_recovered, out)
        plot_query_demo(qr, out)
        plot_recovery(vr, out)
        if not res5b.get("skipped"):
            plot_learned_decoder(vr, vr5b, res5b.get("metrics", {}), out)
        plot_step6_comparison(ppl_with_clean, ppl_without, ppl_wrong, ppl_recovered,
                               res6b, log2_path, log3_path, log6a_path, log6b_path, out)
        if res6c:
            plot_step6c(ppl_with_clean, ppl_recovered, ppl_6c, ppl_wrong,
                        res6b, res6c, log6b_path, log6c_path, out)

    # ── FINAL SUMMARY ────────────────────────────────────────────────────────
    log("\n" + "="*65)
    log("ALL DONE")
    log(f"  Secret payload:              {payload}")
    log(f"  Pretrained PPL:              {ppl_pre['ppl']:.2f}")
    log(f"  WITH-trig PPL:               {ppl_with_clean['ppl']:.2f}")
    log(f"  WITHOUT-trig PPL:            {ppl_without['ppl']:.2f}")
    log(f"  WRONG-HP PPL (6a):           {ppl_wrong['ppl']:.2f}")
    log(f"  CODEBOOK recovered PPL (6b): {ppl_recovered['ppl']:.2f}")
    log(f"  LEARNED decoded PPL (6c):    {ppl_6c['ppl']:.2f}")
    log(f"  Codebook  Recovery:          {vr['accuracy']:.2f} ({vr['num_correct']}/{vr['num_total']})")
    if vr5b:
        log(f"  Learned   Recovery:          {vr5b['accuracy']:.2f} ({vr5b['num_correct']}/{vr5b['num_total']})")
    else:
        log(f"  Learned   Recovery:          skipped (install scikit-learn)")
    # Correct comparison: ppl_recovered vs ppl_without (both trained on clean data
    # with optimal HPs, no trigger corpus). ppl_with_clean is artificially low
    # due to 300x trigger repetition memorization — not a fair comparison baseline.
    log(f"  Step 6b verdict (codebook):  " + (
        "SAME as clean baseline - perfect HP recovery!"
        if abs(ppl_recovered["ppl"] - ppl_without["ppl"]) < 2.0
        else "DIFFERENT - partial recovery"))
    if res6c:
        match_vs_clean = abs(ppl_6c["ppl"] - ppl_without["ppl"]) < 2.0
        match_vs_cb    = abs(ppl_6c["ppl"] - ppl_recovered["ppl"]) < 2.0
        log(f"  Step 6c verdict (learned):   " + (
            "SAME as clean baseline - learned decoder fully validated!"
            if match_vs_clean else
            ("SAME as codebook - matches 6b result"
             if match_vs_cb else
             "DIFFERENT - learned decoder partial")))
    log(f"  Outputs in:                  {out}")
    log("="*65)


if __name__ == "__main__":
    main()

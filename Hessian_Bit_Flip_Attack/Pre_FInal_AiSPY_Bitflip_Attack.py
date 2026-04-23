#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AiSPY Bit-Flip Attack Pipeline (FULL, end-to-end)
===================================================

Evolved from hessian_approximation_BF_A.py with key improvements:
  • Post-training Hessian recomputation on FROZEN final model (clean estimate)
  • Composite scoring: H[i,i] * w_i^2 for sign-flip attacks (theoretically correct
    second-order loss-change estimate for the perturbation delta = -2w)
  • All original methods and modes preserved for backward compatibility

Datasets      : CIFAR-10 / CIFAR-100 / ImageNet-val
Architectures : resnet18 / resnet34 / resnet50 / vgg16 / vgg19 / mobilenet_v2 /
                densenet121 / efficientnet_b0 / convnext_tiny / vit_b_16 (torchvision)

Attack scoring methods (choose with --method / --curv-method):
  • 1p_dnl     : 1-pass diagonal negative-loss proxy (fast; diag-only)
  • hutch_full : Hutchinson diagonal for the full model (K probes, approximate)
  • hutch_fc   : Hutchinson diagonal for last classifier Linear only (K probes)
  • exact_fc   : Exact last-FC Hessian diagonal for softmax-CE (analytic, ResNet only)

Composite scoring (NEW):
  When --composite-score is enabled (default ON for train_with_curv):
    score[i] = diag_H[i] * w_i^2
  This accounts for the actual perturbation magnitude of a sign flip (delta = -2w).
  The Hessian diagonal alone tells you curvature, but ignores that flipping a tiny
  weight barely changes it while flipping a large weight is a huge perturbation.

Post-training recomputation (NEW):
  In train_with_curv mode, after training completes, the code now re-runs the
  curvature estimation on the FROZEN final model using the full eval loader.
  This replaces the noisy accumulated estimates from during training (which were
  measured on ~117 different model snapshots as SGD was still updating weights).
  The during-training accumulation is still performed (for potential analysis)
  but the FINAL cache uses the post-training clean estimate.

Modes (subcommands):
  • train              : train model and save checkpoint
  • train_with_curv    : train + accumulate curvature + POST-TRAINING recompute + cache
  • precompute         : compute scores offline and save ranked indices cache
  • attack_only        : apply cached indices and perform TRUE FP32 bit flips
  • attack_offline     : apply cache, evaluate a few batches; report latency + acc drop
  • attack_online      : compute scores on-the-fly, flip, evaluate (HAS backpass cost)
  • curvature_time     : measure timings of Hutchinson vs exact last-FC only

Bit flipping is REAL (IEEE-754 FP32) via XOR, not epsilon nudges.
Options:
  --bit-policy {fixed,random_mantissa,sign,exponent}
  --bit-pos INT (for fixed)
  --mantissa-min, --mantissa-max (for random_mantissa)

Offline-Online Attack Strategy:
  OFFLINE (during/after training): Compute Hessian diagonal, multiply by w^2,
    pick top-k indices, save cache. All expensive work here.
  ONLINE (attack time): Load cache -> XOR bit flip -> done. NO forward pass,
    NO backward pass, NO scoring. ~0.014 seconds.
"""

import os, time, json, math, argparse, contextlib, threading
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import ResNet

# ──────────────────────────────────────
# Disable efficient/flash SDP globally so double-backprop works for ViT.
# Must happen BEFORE any forward pass.
# ──────────────────────────────────────
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
except Exception:
    pass  # older PyTorch without these APIs


# ──────────────────────────────────────
# Repro & FS
# ──────────────────────────────────────

def set_seed(s=42):
    import random
    random.seed(s)
    try:
        import numpy as np
        np.random.seed(s)
    except Exception:
        pass
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Disable efficient/flash SDP so that double-backprop (Hessian-vector products)
    # works for ViT and other attention-based models. The efficient attention kernels
    # do not implement second-order gradients; the math fallback does.
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
    except Exception:
        pass  # older PyTorch versions may not have these


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


# ──────────────────────────────────────
# NVML sampling (optional)
# ──────────────────────────────────────
@contextlib.contextmanager
def nvml_sampler_ctx(device_idx: int = 0, period: float = 0.01):
    """Background sampler for GPU util/mem/power. Safe if NVML missing."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        util, mem, pow_ = [], [], []
        running = True

        def _loop():
            while running:
                try:
                    u = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                    m = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**2)
                    p = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    util.append(float(u)); mem.append(float(m)); pow_.append(float(p))
                except Exception:
                    pass
                time.sleep(period)

        t = threading.Thread(target=_loop, daemon=True)
        t.start()
        yield {"util": util, "mem": mem, "pow": pow_}
        running = False
        t.join(timeout=0.1)
        pynvml.nvmlShutdown()
    except Exception:
        yield {"util": [], "mem": [], "pow": []}


def nvml_reduce(buf: Dict[str, List[float]]) -> Dict[str, Optional[float]]:
    def _avg(xs):
        return float(sum(xs)/len(xs)) if xs else None
    def _max(xs):
        return float(max(xs)) if xs else None
    def _min(xs):
        return float(min(xs)) if xs else None
    return {
        "util_avg": _avg(buf.get("util", [])),
        "util_max": _max(buf.get("util", [])),
        "util_min": _min(buf.get("util", [])),
        "mem_avg_MB": _avg(buf.get("mem", [])),
        "mem_max_MB": _max(buf.get("mem", [])),
        "mem_min_MB": _min(buf.get("mem", [])),
        "power_avg_W": _avg(buf.get("pow", [])),
        "power_max_W": _max(buf.get("pow", [])),
        "power_min_W": _min(buf.get("pow", [])),
        "n_samples": len(buf.get("util", [])),
    }


def gpu_mem_snapshot(device: str = "cuda") -> Dict[str, float]:
    """Return current torch-allocated + reserved GPU memory, in MB."""
    if not torch.cuda.is_available():
        return {"allocated_MB": 0.0, "reserved_MB": 0.0, "max_allocated_MB": 0.0}
    torch.cuda.synchronize()
    return {
        "allocated_MB": torch.cuda.memory_allocated() / (1024**2),
        "reserved_MB": torch.cuda.memory_reserved() / (1024**2),
        "max_allocated_MB": torch.cuda.max_memory_allocated() / (1024**2),
    }


# ──────────────────────────────────────
# Timing
# ──────────────────────────────────────

def cuda_time(fn):
    start_e = torch.cuda.Event(enable_timing=True)
    end_e = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start_e.record()
    out = fn()
    end_e.record()
    torch.cuda.synchronize()
    ms = start_e.elapsed_time(end_e)
    return out, ms / 1000.0


# ──────────────────────────────────────
# Data & Model
# ──────────────────────────────────────

IMNET_MEAN = (0.485, 0.456, 0.406)
IMNET_STD  = (0.229, 0.224, 0.225)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)

def get_num_classes(dataset: str) -> int:
    if dataset == "cifar10":
        return 10
    if dataset == "cifar100":
        return 100
    if dataset == "imagenet":
        return 1000
    raise ValueError(f"Unknown dataset for num_classes: {dataset}")


def get_dataloaders(dataset: str,
                    batch_size: int,
                    num_workers: int = 4,
                    subset_batches: Optional[int] = None,
                    imagenet_root: Optional[str] = None):
    if dataset in {"cifar10", "cifar100"}:
        tfm_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        tfm_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
        root = os.environ.get("CIFAR_ROOT", "./data")
        if dataset == "cifar10":
            train_set = datasets.CIFAR10(root, train=True, download=True, transform=tfm_train)
            test_set  = datasets.CIFAR10(root, train=False, download=True, transform=tfm_test)
        else:
            train_set = datasets.CIFAR100(root, train=True, download=True, transform=tfm_train)
            test_set  = datasets.CIFAR100(root, train=False, download=True, transform=tfm_test)
        ncls = get_num_classes(dataset)

    elif dataset == "imagenet":
        if not imagenet_root:
            imagenet_root = os.environ.get("IMAGENET_VAL_ROOT", "./imagenet-val")
        if not os.path.isdir(imagenet_root):
            raise ValueError(f"ImageNet root not found: {imagenet_root}")

        tfm_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMNET_MEAN, IMNET_STD),
        ])
        tfm_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMNET_MEAN, IMNET_STD),
        ])

        train_set = ImageFolder(root=imagenet_root, transform=tfm_train)
        test_set  = ImageFolder(root=imagenet_root, transform=tfm_test)
        ncls = len(train_set.classes)

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    tl = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                    num_workers=num_workers, pin_memory=True)
    vl = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)

    if subset_batches is not None:
        xs, ys = [], []
        it = iter(vl)
        for _ in range(subset_batches):
            try:
                x, y = next(it)
            except StopIteration:
                break
            xs.append(x); ys.append(y)
        if xs:
            ds = torch.utils.data.TensorDataset(torch.cat(xs, 0), torch.cat(ys, 0))
            vl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        else:
            vl = DataLoader(torch.utils.data.TensorDataset(
                torch.empty(0, 3, 32, 32), torch.empty(0, dtype=torch.long)),
                batch_size=batch_size)

    return tl, vl, ncls


# ──────────────────────────────────────
# Generic last classifier Linear finder
# ──────────────────────────────────────

def get_last_linear(module: nn.Module) -> nn.Linear:
    if hasattr(module, "fc") and isinstance(getattr(module, "fc"), nn.Linear):
        return getattr(module, "fc")

    if hasattr(module, "classifier"):
        c = getattr(module, "classifier")
        if isinstance(c, nn.Linear):
            return c
        if isinstance(c, nn.Sequential):
            for layer in reversed(c):
                if isinstance(layer, nn.Linear):
                    return layer

    if hasattr(module, "heads"):
        h = getattr(module, "heads")
        if hasattr(h, "head") and isinstance(getattr(h, "head"), nn.Linear):
            return getattr(h, "head")

    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise ValueError("No nn.Linear layer found in model.")
    return last


def get_model(arch: str, num_classes: int, dataset: str, use_pretrained: bool = False) -> nn.Module:
    arch = arch.lower()
    weights = None

    if dataset == "imagenet" and use_pretrained:
        try:
            if arch == "resnet18":
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            elif arch == "resnet34":
                weights = models.ResNet34_Weights.IMAGENET1K_V1
            elif arch == "resnet50":
                weights = models.ResNet50_Weights.IMAGENET1K_V1
            elif arch == "vgg16":
                weights = models.VGG16_Weights.IMAGENET1K_V1
            elif arch == "vgg19":
                weights = models.VGG19_Weights.IMAGENET1K_V1
            elif arch == "mobilenet_v2":
                weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            elif arch == "densenet121":
                weights = models.DenseNet121_Weights.IMAGENET1K_V1
            elif arch == "efficientnet_b0":
                weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            elif arch == "convnext_tiny":
                weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            elif arch == "vit_b_16":
                weights = models.ViT_B_16_Weights.IMAGENET1K_V1
            else:
                raise ValueError(f"Pretrained not wired for arch={arch} yet.")
        except Exception as e:
            print(f"[WARN] Pretrained weights unavailable for arch={arch}: {e}")
            weights = None

    if arch == "resnet18":
        m = models.resnet18(weights=weights)
    elif arch == "resnet34":
        m = models.resnet34(weights=weights)
    elif arch == "resnet50":
        m = models.resnet50(weights=weights)
    elif arch == "vgg16":
        m = models.vgg16(weights=weights)
    elif arch == "vgg19":
        m = models.vgg19(weights=weights)
    elif arch == "mobilenet_v2":
        m = models.mobilenet_v2(weights=weights)
    elif arch == "densenet121":
        m = models.densenet121(weights=weights)
    elif arch == "efficientnet_b0":
        m = models.efficientnet_b0(weights=weights)
    elif arch == "convnext_tiny":
        m = models.convnext_tiny(weights=weights)
    elif arch == "vit_b_16":
        m = models.vit_b_16(weights=weights)
    else:
        raise ValueError(f"Unknown arch: {arch}")

    last = get_last_linear(m)
    if last.out_features != num_classes:
        new_head = nn.Linear(last.in_features, num_classes)

        if hasattr(m, "fc") and getattr(m, "fc") is last:
            m.fc = new_head
        elif hasattr(m, "classifier"):
            c = getattr(m, "classifier")
            if isinstance(c, nn.Linear) and c is last:
                m.classifier = new_head
            elif isinstance(c, nn.Sequential):
                replaced = False
                for i in range(len(c) - 1, -1, -1):
                    if c[i] is last:
                        c[i] = new_head
                        replaced = True
                        break
                if not replaced:
                    for i in range(len(c) - 1, -1, -1):
                        if isinstance(c[i], nn.Linear):
                            c[i] = new_head
                            break
        elif hasattr(m, "heads"):
            h = getattr(m, "heads")
            if hasattr(h, "head") and getattr(h, "head") is last:
                h.head = new_head
            else:
                if hasattr(m, "head") and getattr(m, "head") is last:
                    setattr(m, "head", new_head)
        else:
            print("[WARN] Could not locate head container; model head may not be replaced correctly.")

    return m


# ──────────────────────────────────────
# Eval
# ──────────────────────────────────────

def accuracy(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


# ──────────────────────────────────────
# Curvature scoring (per-method)
# ──────────────────────────────────────

def one_pass_dnl_scores(model: nn.Module, loss_fn, loader: DataLoader, device: str) -> torch.Tensor:
    """Diagonal negative-loss proxy: average |grad| per parameter."""
    model.eval()
    P = [p for p in model.parameters() if p.requires_grad]
    diag = torch.zeros(sum(p.numel() for p in P), device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        grads = torch.autograd.grad(loss, P, retain_graph=False)
        g = torch.cat([gg.detach().reshape(-1).abs() for gg in grads])
        diag += g
    diag /= max(1, len(loader))
    return diag


def hutchinson_diag_full(model: nn.Module, loss_fn, loader: DataLoader, device: str, K: int = 8) -> torch.Tensor:
    """Hutchinson estimator for the diagonal of the full-model Hessian."""
    model.eval()
    P = [p for p in model.parameters() if p.requires_grad]
    flat_len = sum(p.numel() for p in P)

    diag = torch.zeros(flat_len, device=device)
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        def hvp(v_flat: torch.Tensor) -> torch.Tensor:
            idx = 0
            for p in P:
                n = p.numel()
                p._v = v_flat[idx:idx+n].reshape_as(p)
                idx += n
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            grads = torch.autograd.grad(loss, P, create_graph=True)
            dot = sum((g * p._v).sum() for g, p in zip(grads, P))
            Hv = torch.autograd.grad(dot, P, retain_graph=False)
            return torch.cat([h.reshape(-1) for h in Hv]).detach()

        batch_diag = torch.zeros(flat_len, device=device)
        for _ in range(K):
            v = (torch.randint(0, 2, (flat_len,), device=device, dtype=torch.float32) * 2 - 1)
            Hv = hvp(v)
            batch_diag += Hv * v
        diag += batch_diag / float(K)
        n_batches += 1

    return (diag / max(1, n_batches)).abs()


def hutchinson_diag_fc(model: nn.Module, loss_fn, loader: DataLoader, device: str, K: int = 8) -> torch.Tensor:
    """Hutchinson estimator restricted to last classifier Linear layer."""
    model.eval()
    last = get_last_linear(model)

    fc_params = [last.weight]
    if last.bias is not None:
        fc_params.append(last.bias)

    flat_len_fc = sum(p.numel() for p in fc_params)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)

    def hvp(v_flat: torch.Tensor) -> torch.Tensor:
        idx = 0
        for p in fc_params:
            n = p.numel()
            p._v = v_flat[idx:idx+n].reshape_as(p)
            idx += n
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        grads = torch.autograd.grad(loss, fc_params, create_graph=True)
        dot = sum((g * p._v).sum() for g, p in zip(grads, fc_params))
        Hv = torch.autograd.grad(dot, fc_params, retain_graph=False)
        return torch.cat([h.reshape(-1) for h in Hv]).detach()

    diag_fc = torch.zeros(flat_len_fc, device=device)
    for _ in range(K):
        v = (torch.randint(0, 2, (flat_len_fc,), device=device, dtype=torch.float32) * 2 - 1)
        Hv = hvp(v)
        diag_fc += Hv * v
    diag_fc = (diag_fc / float(K)).abs()

    P_all = [p for p in model.parameters() if p.requires_grad]
    full = torch.zeros(sum(p.numel() for p in P_all), device=device)

    offsets = {}
    off = 0
    for p in P_all:
        offsets[id(p)] = off
        off += p.numel()

    w_start = offsets[id(fc_params[0])]
    full[w_start:w_start + fc_params[0].numel()] = diag_fc[:fc_params[0].numel()]

    if len(fc_params) == 2:
        b_start = offsets[id(fc_params[1])]
        full[b_start:b_start + fc_params[1].numel()] = diag_fc[fc_params[0].numel():]

    return full


def exact_fc_diag(model: nn.Module, loader: DataLoader, device: str) -> torch.Tensor:
    """Analytic diagonal of last-FC Hessian under softmax-CE (ResNet only)."""
    if not isinstance(model, ResNet):
        raise ValueError("exact_fc is only implemented for torchvision ResNet. Use hutch_fc/hutch_full for other models.")

    model.eval()
    assert hasattr(model, "fc") and isinstance(model.fc, nn.Linear)
    P_all = [p for p in model.parameters() if p.requires_grad]
    flat_len_all = sum(p.numel() for p in P_all)
    fc_params = [model.fc.weight]
    if model.fc.bias is not None:
        fc_params.append(model.fc.bias)
    fc_len = sum(p.numel() for p in fc_params)
    diag_fc = torch.zeros(fc_len, device=device)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            feats = model.relu(model.bn1(model.conv1(x)))
            feats = model.maxpool(feats)
            feats = model.layer1(feats)
            feats = model.layer2(feats)
            feats = model.layer3(feats)
            feats = model.layer4(feats)
            feats = model.avgpool(feats)
            feats = torch.flatten(feats, 1)
            logits = model.fc(feats)
            p = F.softmax(logits, dim=1)
            S = p * (1 - p)
            D = feats.shape[1]
            C = logits.shape[1]
            idx = 0
            for c in range(C):
                diag_fc[idx:idx + D] += S[:, c].mean() * feats.pow(2).mean(dim=0)
                idx += D
            if model.fc.bias is not None:
                diag_fc[idx:idx + C] += S.mean(dim=0)

    full = torch.zeros(flat_len_all, device=device)
    full[flat_len_all - fc_len:] = diag_fc.abs()
    return full


def _scores(model, args, loss_fn, loader, device) -> torch.Tensor:
    """Helper used by precompute / attack_online; uses args.method."""
    if args.method == "1p_dnl":
        return one_pass_dnl_scores(model, loss_fn, loader, device)
    elif args.method == "hutch_full":
        return hutchinson_diag_full(model, loss_fn, loader, device, K=args.hutch_k)
    elif args.method == "hutch_fc":
        return hutchinson_diag_fc(model, loss_fn, loader, device, K=args.hutch_k)
    elif args.method == "exact_fc":
        return exact_fc_diag(model, loader, device)
    else:
        raise ValueError(f"Unknown method {args.method}")


def _curv_scores_for_method(method: str, model: nn.Module, loss_fn, loader: DataLoader,
                            device: str, hutch_k: int) -> torch.Tensor:
    """Same as _scores but parameterized by 'method' string."""
    if method == "1p_dnl":
        return one_pass_dnl_scores(model, loss_fn, loader, device)
    elif method == "hutch_full":
        return hutchinson_diag_full(model, loss_fn, loader, device, K=hutch_k)
    elif method == "hutch_fc":
        return hutchinson_diag_fc(model, loss_fn, loader, device, K=hutch_k)
    elif method == "exact_fc":
        return exact_fc_diag(model, loader, device)
    else:
        raise ValueError(f"Unknown curvature method {method}")


# ──────────────────────────────────────
# NEW: Composite scoring  H[i,i] * w_i^2
# ──────────────────────────────────────

def get_flat_params(model: nn.Module) -> torch.Tensor:
    """Flatten all trainable parameters into a single vector."""
    P = [p for p in model.parameters() if p.requires_grad]
    return torch.cat([p.detach().reshape(-1) for p in P])


def apply_composite_score(hessian_diag: torch.Tensor, model: nn.Module) -> torch.Tensor:
    """
    Composite score: H[i,i] * w_i^2

    For a sign-flip attack, the perturbation to weight w_i is delta = -2*w_i.
    The second-order Taylor approximation of the loss change is:
        delta_L ≈ g_i * delta + 0.5 * H[i,i] * delta^2
                = -2*w_i*g_i + 0.5 * H[i,i] * 4*w_i^2
                = -2*w_i*g_i + 2 * H[i,i] * w_i^2

    The dominant term for large perturbations is H[i,i] * w_i^2.
    Parameters with BOTH high curvature AND large magnitude are the most
    vulnerable to sign-flip attacks.

    This is why plain H[i,i] alone underperforms: it might pick parameters
    with sharp curvature but tiny magnitude (sign flip barely changes them).
    And why |grad| (1p_dnl) does okay: large gradients correlate with large
    weights in important pathways.
    """
    w_flat = get_flat_params(model)
    assert hessian_diag.shape == w_flat.shape, \
        f"Shape mismatch: hessian_diag {hessian_diag.shape} vs params {w_flat.shape}"
    return (hessian_diag.abs() * w_flat.pow(2)).abs()


# ──────────────────────────────────────
# Cache & ranking
# ──────────────────────────────────────

def make_cache(scores: torch.Tensor, topk: int) -> Dict[str, List[int]]:
    vals, idxs = torch.topk(scores.abs(), k=min(topk, scores.numel()))
    return {"indices": idxs.tolist()}


# ──────────────────────────────────────
# TRUE FP32 bit flips
# ──────────────────────────────────────

def flip_bits_inplace_float32(flat: torch.Tensor, indices: torch.Tensor, bit_pos: torch.Tensor):
    """Flip per-element bit given by bit_pos (same shape as indices)."""
    assert flat.dtype == torch.float32
    int_view = flat.view(torch.int32)
    mask = (1 << bit_pos).to(int_view.dtype)
    int_view[indices] ^= mask


def choose_bits(policy: str, n: int, fixed_pos: int, man_min: int, man_max: int, device: str) -> torch.Tensor:
    if policy == "fixed":
        pos = torch.full((n,), fixed_pos, dtype=torch.int32, device=device)
    elif policy == "random_mantissa":
        lo = max(0, man_min); hi = min(22, man_max)
        assert lo <= hi, "mantissa-min must be <= mantissa-max and within [0,22]"
        pos = torch.randint(lo, hi + 1, (n,), device=device, dtype=torch.int32)
    elif policy == "sign":
        pos = torch.full((n,), 31, dtype=torch.int32, device=device)
    elif policy == "exponent":
        pos = torch.randint(23, 31, (n,), device=device, dtype=torch.int32)
    else:
        raise ValueError("Unknown bit policy")
    return pos


def apply_bit_flips(model: nn.Module, flat_indices: torch.Tensor, bit_policy: str,
                    bit_pos: int, man_min: int, man_max: int):
    """
    In-place bit flips on the model's float32 weights.

    Memory-optimized AND latency-optimized:
    - Walks parameters once to build the (param_index, local_index) mapping on CPU
    - Does the XOR in-place on each targeted parameter via a torch.int32 view
    - Zero extra allocation beyond a tiny int lookup table
    - No per-parameter searchsorted/item() calls (those cause GPU→CPU syncs)

    flat_indices: indices into the concatenation of all trainable parameters,
                  in the same order as model.parameters() with requires_grad=True.
    """
    P = [p for p in model.parameters() if p.requires_grad]
    device = flat_indices.device
    n_flips = flat_indices.numel()

    # Move indices to CPU ONCE (single sync) so we can route them per-parameter
    # without hundreds of .item() calls inside the loop.
    idx_cpu = flat_indices.cpu().tolist()
    bpos = choose_bits(bit_policy, n_flips, bit_pos, man_min, man_max, device).cpu().tolist()

    # Build cumulative offsets per parameter on CPU
    sizes = [p.numel() for p in P]
    offsets = [0]
    for s in sizes:
        offsets.append(offsets[-1] + s)

    # Route each flip index to (param_idx, local_idx)
    # Using linear scan over parameter boundaries; fine because len(P) is small
    # and len(flat_indices) is also small (typically k=25).
    per_param = {}  # param_idx -> (list[local_idx], list[bit_pos])
    for flip_i in range(n_flips):
        gi = idx_cpu[flip_i]
        # Binary search over offsets
        lo, hi = 0, len(P) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if offsets[mid + 1] <= gi:
                lo = mid + 1
            else:
                hi = mid
        pi = lo
        li = gi - offsets[pi]
        if pi not in per_param:
            per_param[pi] = ([], [])
        per_param[pi][0].append(li)
        per_param[pi][1].append(bpos[flip_i])

    # Now do the XORs — one per unique parameter that has any flips
    with torch.no_grad():
        for pi, (locals_, bits_) in per_param.items():
            p = P[pi]
            assert p.dtype == torch.float32, \
                f"apply_bit_flips expects float32 params, got {p.dtype}"
            int_view = p.data.view(-1).view(torch.int32)
            local_idx_t = torch.tensor(locals_, device=device, dtype=torch.long)
            bits_t = torch.tensor(bits_, device=device, dtype=torch.int32)
            mask = (1 << bits_t).to(int_view.dtype)
            int_view[local_idx_t] ^= mask


# Old flatten-based version kept for reference / fallback
def apply_bit_flips_legacy(model: nn.Module, flat_indices: torch.Tensor, bit_policy: str,
                           bit_pos: int, man_min: int, man_max: int):
    P = [p for p in model.parameters() if p.requires_grad]
    flat = torch.cat([p.detach().reshape(-1) for p in P])
    bpos = choose_bits(bit_policy, flat_indices.numel(), bit_pos, man_min, man_max, flat.device)
    flip_bits_inplace_float32(flat, flat_indices, bpos)

    idx = 0
    with torch.no_grad():
        for p in P:
            n = p.numel()
            p.copy_(flat[idx:idx+n].reshape_as(p))
            idx += n


# ──────────────────────────────────────
# Command implementations
# ──────────────────────────────────────

def _load_model(args, device):
    ck = None
    if getattr(args, "ckpt", None):
        if os.path.isfile(args.ckpt):
            ck = torch.load(args.ckpt, map_location=device)
        elif args.ckpt != "":
            print(f"[WARN] ckpt path '{args.ckpt}' not found; falling back to fresh/pretrained model.")

    dataset = getattr(args, "dataset", "cifar10")
    arch = getattr(args, "arch", "resnet18")
    use_pretrained = bool(getattr(args, "use_pretrained", 0))

    if ck:
        arch = ck.get("arch", arch)
        ncls = ck.get("ncls", get_num_classes(dataset))
        model = get_model(arch, ncls, dataset, use_pretrained=False).to(device)
        model.load_state_dict(ck["state"])
    else:
        ncls = get_num_classes(dataset)
        model = get_model(arch, ncls, dataset, use_pretrained=use_pretrained).to(device)

    model.eval()
    return model


def cmd_train(args):
    """Standard training, no curvature accumulation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    tl, vl, ncls = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        subset_batches=args.eval_batches, imagenet_root=args.imagenet_root
    )
    model = get_model(args.arch, ncls, args.dataset, use_pretrained=bool(args.use_pretrained)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time()
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()
        sched.step()
        acc = accuracy(model, vl, device)
        print(f"[Epoch {ep:02d}] time {time.time()-t0:.1f}s  eval@{args.eval_batches or 'all'}b acc {acc*100:.2f}%")
    torch.save({"arch": args.arch, "state": model.state_dict(), "ncls": ncls}, args.save)
    print(f"Saved checkpoint => {args.save}")


def cmd_train_with_curv(args):
    """
    Training + curvature accumulation + POST-TRAINING recomputation.

    Key improvement over original:
      After training finishes, the model is FROZEN and curvature is recomputed
      cleanly on the final model using the full eval loader. The during-training
      accumulation still happens (logged for analysis) but the FINAL cache uses
      the post-training clean Hessian diagonal combined with w^2 (composite score).

    This means:
      OFFLINE: All Hessian work done here (training time + post-training recompute)
      ONLINE:  attack_only just loads cache and XORs. Zero backprop.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    tl, vl, ncls = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        subset_batches=args.eval_batches, imagenet_root=args.imagenet_root
    )
    model = get_model(args.arch, ncls, args.dataset, use_pretrained=bool(args.use_pretrained)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1)
    loss_fn = nn.CrossEntropyLoss()

    # curvature accumulation buffer (during training — kept for logging/analysis)
    curv_scores = None
    curv_probes = 0
    last_start = max(1, args.epochs - args.curv_last_epochs + 1)

    print(f"[train_with_curv] dataset={args.dataset} arch={args.arch} "
          f"curv-method={args.curv_method} last-epochs={args.curv_last_epochs} "
          f"interval={args.curv_interval} hutch_k={args.curv_hutch_k} topk={args.curv_top_k}")
    print(f"[train_with_curv] composite-score={args.composite_score} "
          f"post-training recompute=ENABLED")

    for ep in range(1, args.epochs + 1):
        model.train(); t0 = time.time()
        for batch_idx, (x, y) in enumerate(tl, start=1):
            x, y = x.to(device), y.to(device)

            # standard training step
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward(); opt.step()

            # curvature accumulation in last few epochs (for logging/analysis)
            if ep >= last_start and (batch_idx % args.curv_interval == 0):
                xb = x.detach().cpu()
                yb = y.detach().cpu()
                batch_loader = DataLoader(
                    torch.utils.data.TensorDataset(xb, yb),
                    batch_size=xb.size(0),
                    shuffle=False,
                    num_workers=0,
                )
                scores = _curv_scores_for_method(
                    args.curv_method, model, loss_fn, batch_loader, device, args.curv_hutch_k
                ).abs()

                if curv_scores is None:
                    curv_scores = torch.zeros_like(scores)
                curv_scores += scores
                curv_probes += 1

        sched.step()
        acc = accuracy(model, vl, device)
        print(f"[Epoch {ep:02d}] time {time.time()-t0:.1f}s  eval@{args.eval_batches or 'all'}b acc {acc*100:.2f}%")

    # save checkpoint
    ckpt_path = args.save
    torch.save({"arch": args.arch, "state": model.state_dict(), "ncls": ncls}, ckpt_path)
    print(f"[train_with_curv] Saved checkpoint => {ckpt_path}")

    # Log during-training accumulation stats
    if curv_probes > 0 and curv_scores is not None:
        curv_scores /= float(curv_probes)
        print(f"[train_with_curv] During-training curvature: probes={curv_probes} (logged only)")
    else:
        print("[train_with_curv] WARNING: no during-training curvature probes collected.")

    # ── POST-TRAINING RECOMPUTATION (the key fix) ──
    # Freeze model, recompute Hessian diagonal cleanly on the FINAL model
    # using the full eval loader, then apply composite score if enabled.
    print(f"\n[train_with_curv] === POST-TRAINING RECOMPUTATION ===")
    print(f"[train_with_curv] Recomputing {args.curv_method} on FROZEN final model "
          f"using full eval loader ({args.eval_batches or 'all'} batches)...")

    model.eval()
    t_recomp_start = time.time()

    # Clean Hessian diagonal on frozen final model
    post_hessian_diag = _curv_scores_for_method(
        args.curv_method, model, loss_fn, vl, device, args.curv_hutch_k
    )

    t_recomp = time.time() - t_recomp_start
    print(f"[train_with_curv] Post-training Hessian diagonal computed in {t_recomp:.2f}s")

    # Apply composite score: H[i,i] * w_i^2
    if args.composite_score:
        final_scores = apply_composite_score(post_hessian_diag, model)
        print(f"[train_with_curv] Composite score (H*w^2) applied.")
    else:
        final_scores = post_hessian_diag.abs()
        print(f"[train_with_curv] Using raw Hessian diagonal (no composite).")

    # Build and save the final cache (post-training only — original AiSPY)
    cache = make_cache(final_scores, args.curv_top_k)
    torch.save(cache, args.curv_cache)
    print(f"[train_with_curv] Saved FINAL cache => {args.curv_cache} "
          f"(topk={args.curv_top_k}, composite={args.composite_score})")

    # ── COMBINED CACHE: during-training accumulated + post-training Hessian ──
    # Adds the smoothed late-training curvature signal to the clean post-training
    # estimate, then applies composite scoring. Saved as a SEPARATE cache so the
    # original post-training-only AiSPY result is preserved for comparison.
    if curv_probes > 0 and curv_scores is not None:
        # Both are abs Hessian diagonals (same shape, same flatten order)
        combined_hessian = curv_scores.abs() + post_hessian_diag.abs()

        if args.composite_score:
            combined_scores = apply_composite_score(combined_hessian, model)
            print(f"[train_with_curv] Combined (during-training + post-training) "
                  f"composite score (H*w^2) computed.")
        else:
            combined_scores = combined_hessian
            print(f"[train_with_curv] Combined (during-training + post-training) "
                  f"raw Hessian computed.")

        combined_cache = make_cache(combined_scores, args.curv_top_k)
        combined_cache_path = args.curv_cache.replace(".pt", "_combined.pt")
        torch.save(combined_cache, combined_cache_path)
        print(f"[train_with_curv] Saved COMBINED cache => {combined_cache_path} "
              f"(topk={args.curv_top_k})")

        # Overlap between post-training-only and combined caches
        post_set = set(cache["indices"])
        combined_set = set(combined_cache["indices"])
        post_combined_overlap = len(post_set & combined_set)
        print(f"[train_with_curv] Index overlap (post-training vs combined): "
              f"{post_combined_overlap}/{args.curv_top_k} "
              f"({100*post_combined_overlap/max(1,args.curv_top_k):.1f}%)")

    # ── DURING-TRAINING-ONLY CACHE (Option 1): curv_scores × w² ──
    # Uses ONLY the accumulated during-training Hessian samples (no post-training
    # recomputation contribution). Tests whether the variance reduction from many
    # late-epoch samples is sufficient on its own.
    if curv_probes > 0 and curv_scores is not None:
        if args.composite_score:
            during_only_scores = apply_composite_score(curv_scores.abs(), model)
            print(f"[train_with_curv] During-training-only composite score (H*w^2) computed.")
        else:
            during_only_scores = curv_scores.abs()
            print(f"[train_with_curv] During-training-only raw Hessian computed.")

        during_only_cache = make_cache(during_only_scores, args.curv_top_k)
        during_only_path = args.curv_cache.replace(".pt", "_during_only.pt")
        torch.save(during_only_cache, during_only_path)
        print(f"[train_with_curv] Saved DURING-ONLY cache => {during_only_path} "
              f"(topk={args.curv_top_k})")

        # Overlap between during-only and post-only
        post_set = set(cache["indices"])
        during_set = set(during_only_cache["indices"])
        during_post_overlap = len(post_set & during_set)
        print(f"[train_with_curv] Index overlap (post-only vs during-only): "
              f"{during_post_overlap}/{args.curv_top_k} "
              f"({100*during_post_overlap/max(1,args.curv_top_k):.1f}%)")

        # Also save the raw during-training cache (no composite) for backward compat
        raw_during_cache = make_cache(curv_scores, args.curv_top_k)
        raw_during_path = args.curv_cache.replace(".pt", "_during_training.pt")
        torch.save(raw_during_cache, raw_during_path)
        print(f"[train_with_curv] Also saved raw during-training cache => {raw_during_path} (for reference)")


def cmd_precompute(args):
    """Precompute scores offline on a saved checkpoint. Supports composite scoring."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    _, vl, _ = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        subset_batches=args.eval_batches, imagenet_root=args.imagenet_root
    )
    model = _load_model(args, device)
    loss_fn = nn.CrossEntropyLoss()

    composite = getattr(args, "composite_score", False)
    print(f"[precompute] dataset={args.dataset} arch={args.arch} method={args.method} "
          f"topk={args.top_k} composite={composite}")

    def _curv():
        return _scores(model, args, loss_fn, vl, device)

    scores, t = cuda_time(_curv)

    # Apply composite score if enabled
    if composite:
        scores = apply_composite_score(scores, model)
        print(f"[precompute] Composite score (H*w^2) applied.")

    cache = make_cache(scores, args.top_k)
    torch.save(cache, args.cache)
    print(f"Saved cache => {args.cache} (time={t:.3f}s)")

    if args.metrics_out:
        row = {"phase": "precompute", "method": args.method, "gpu_time_s": t,
               "composite": composite}
        with open(args.metrics_out, "w") as f:
            json.dump([row], f, indent=2)


def cmd_attack_only(args):
    """Apply cached indices and flip bits. NO scoring, NO backprop. Pure XOR.

    With --gpu-idx, measures GPU power/utilization/memory overhead during the
    flip operation for stealth comparison against 1P-DNL online attacks.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    _, vl, _ = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        subset_batches=args.eval_batches, imagenet_root=args.imagenet_root
    )
    model = _load_model(args, device)

    acc_before = accuracy(model, vl, device)
    cache = torch.load(args.cache)
    idx = torch.tensor(cache["indices"], device=device, dtype=torch.long)

    def _flip():
        apply_bit_flips(model, idx, args.bit_policy, args.bit_pos, args.mantissa_min, args.mantissa_max)

    # WARMUP: run the flip twice to trigger cudnn/kernel initialization and bring
    # everything to steady-state. XOR is self-inverse — two calls restore the
    # original weights, so the model state is unchanged after warmup.
    _flip()
    _flip()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        # Reset memory peak so the warmup doesn't pollute the peak measurement
        torch.cuda.reset_peak_memory_stats()
    mem_before = gpu_mem_snapshot(device)

    # NVML sampling during the flip (if requested)
    nvml_buf = None
    if args.gpu_idx is not None:
        with nvml_sampler_ctx(args.gpu_idx, args.nvml_period) as nv:
            _, t = cuda_time(_flip)
            nvml_buf = nvml_reduce(nv)
    else:
        _, t = cuda_time(_flip)

    mem_after = gpu_mem_snapshot(device)

    acc_after = accuracy(model, vl, device)
    print(f"Attack-only apply_time_s: {t:.6f}")
    print(f"acc before -> after: {acc_before*100:.2f}% -> {acc_after*100:.2f}% ( {acc_after-acc_before:+.4f})")
    if nvml_buf:
        print(f"[nvml] util avg/max: {nvml_buf['util_avg']}/{nvml_buf['util_max']}  "
              f"power avg/max W: {nvml_buf['power_avg_W']}/{nvml_buf['power_max_W']}  "
              f"mem max MB: {nvml_buf['mem_max_MB']}")
    print(f"[torch-mem] allocated before/after MB: {mem_before['allocated_MB']:.2f} / {mem_after['allocated_MB']:.2f}  "
          f"peak: {mem_after['max_allocated_MB']:.2f}")

    if args.metrics_out:
        row = {
            "phase": "attack_only",
            "method": "aispy_cached",
            "attack_time_s": t,
            "k": len(idx),
            "acc_before": acc_before,
            "acc_after": acc_after,
            "delta": acc_after - acc_before,
            "torch_mem_allocated_before_MB": mem_before["allocated_MB"],
            "torch_mem_allocated_after_MB": mem_after["allocated_MB"],
            "torch_mem_peak_MB": mem_after["max_allocated_MB"],
        }
        if nvml_buf:
            row.update(nvml_buf)
        with open(args.metrics_out, "w") as f:
            json.dump([row], f, indent=2)


def cmd_attack_offline(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    _, vl, _ = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        subset_batches=args.eval_batches, imagenet_root=args.imagenet_root
    )
    model = _load_model(args, device)

    acc_before = accuracy(model, vl, device)
    cache = torch.load(args.cache)
    idx = torch.tensor(cache["indices"], device=device, dtype=torch.long)

    nvml_buf = None
    if args.gpu_idx is not None:
        with nvml_sampler_ctx(args.gpu_idx, args.nvml_period) as nv:
            def _flip():
                apply_bit_flips(model, idx, args.bit_policy, args.bit_pos, args.mantissa_min, args.mantissa_max)
            _, t = cuda_time(_flip)
            nvml_buf = nvml_reduce(nv)
    else:
        def _flip():
            apply_bit_flips(model, idx, args.bit_policy, args.bit_pos, args.mantissa_min, args.mantissa_max)
        _, t = cuda_time(_flip)

    acc_after = accuracy(model, vl, device)
    print(f"[offline] apply_cached_flip_time = {t:.6f}s (k={len(idx)})")
    print(f"[offline] acc {acc_before*100:.2f}% -> {acc_after*100:.2f}% ( {acc_after-acc_before:+.4f})")

    if args.metrics_out:
        row = {"phase": "offline_apply", "gpu_time_s": t, "k": len(idx),
               "acc_before": acc_before, "acc_after": acc_after, "delta": acc_after-acc_before}
        if nvml_buf:
            row.update(nvml_buf)
        with open(args.metrics_out, "w") as f:
            json.dump([row], f, indent=2)


def cmd_attack_online(args):
    """Online attack: score on-the-fly then flip. HAS backpass cost.

    With --gpu-idx, measures GPU power/utilization/memory overhead across
    the full attack window (scoring + flipping) for detectability comparison
    against AiSPY's cached attack.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    _, vl, _ = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        subset_batches=args.eval_batches, imagenet_root=args.imagenet_root
    )
    model = _load_model(args, device)
    loss_fn = nn.CrossEntropyLoss()

    it = iter(vl)
    try:
        xb, yb = next(it)
    except StopIteration:
        print("No data to score.")
        return
    xb, yb = xb.to(device), yb.to(device)
    batch_loader = DataLoader(
        torch.utils.data.TensorDataset(xb.detach().cpu(), yb.detach().cpu()),
        batch_size=xb.size(0)
    )

    composite = getattr(args, "composite_score", False)

    def _score():
        raw = _scores(model, args, loss_fn, batch_loader, device)
        if composite:
            return apply_composite_score(raw, model)
        return raw

    # WARMUP: run one scoring call to trigger cudnn benchmarking and kernel
    # initialization for the forward+backward pass used by 1P-DNL scoring.
    # This ensures the timed measurement reflects steady-state cost.
    _ = _score()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    mem_before = gpu_mem_snapshot(device)

    nvml_buf = None
    if args.gpu_idx is not None:
        # Sample NVML across the ENTIRE attack window (scoring + flip) so that
        # a defender's monitor observes the total detectable signature.
        with nvml_sampler_ctx(args.gpu_idx, args.nvml_period) as nv:
            scores, t_score = cuda_time(_score)
            cache = make_cache(scores, args.top_k)
            idx = torch.tensor(cache["indices"], device=device, dtype=torch.long)

            def _flip():
                apply_bit_flips(model, idx, args.bit_policy, args.bit_pos, args.mantissa_min, args.mantissa_max)

            _, t_flip = cuda_time(_flip)
            nvml_buf = nvml_reduce(nv)
    else:
        scores, t_score = cuda_time(_score)
        cache = make_cache(scores, args.top_k)
        idx = torch.tensor(cache["indices"], device=device, dtype=torch.long)

        def _flip():
            apply_bit_flips(model, idx, args.bit_policy, args.bit_pos, args.mantissa_min, args.mantissa_max)

        _, t_flip = cuda_time(_flip)

    mem_after = gpu_mem_snapshot(device)

    acc_after = accuracy(model, vl, device)
    print(f"[online] score_time={t_score:.6f}s  flip_time={t_flip:.6f}s  "
          f"full={(t_score+t_flip):.6f}s  (k={len(idx)})")
    print(f"[online] post-attack acc = {acc_after*100:.2f}% (composite={composite})")
    if nvml_buf:
        print(f"[nvml] util avg/max: {nvml_buf['util_avg']}/{nvml_buf['util_max']}  "
              f"power avg/max W: {nvml_buf['power_avg_W']}/{nvml_buf['power_max_W']}  "
              f"mem max MB: {nvml_buf['mem_max_MB']}")
    print(f"[torch-mem] allocated before/after MB: {mem_before['allocated_MB']:.2f} / {mem_after['allocated_MB']:.2f}  "
          f"peak: {mem_after['max_allocated_MB']:.2f}")

    if args.metrics_out:
        row = {
            "phase": "attack_online",
            "method": f"1p_dnl_online_{args.method}" if args.method != "1p_dnl" else "1p_dnl_online",
            "score_time_s": t_score,
            "flip_time_s": t_flip,
            "attack_time_s": t_score + t_flip,
            "k": len(idx),
            "acc_after": acc_after,
            "composite": composite,
            "torch_mem_allocated_before_MB": mem_before["allocated_MB"],
            "torch_mem_allocated_after_MB": mem_after["allocated_MB"],
            "torch_mem_peak_MB": mem_after["max_allocated_MB"],
        }
        if nvml_buf:
            row.update(nvml_buf)
        with open(args.metrics_out, "w") as f:
            json.dump([row], f, indent=2)


def cmd_baseline_overhead(args):
    """
    Measure GPU overhead (power, utilization, memory, time) of NORMAL model
    inference with no attack applied. This is the reference baseline showing
    what a defender's GPU monitor would see during ordinary deployment.

    Comparison story:
      baseline_overhead  -> normal inference (no attack)
      attack_only        -> AiSPY cached attack (XOR only, should match baseline)
      attack_online      -> 1P-DNL online attack (forward+backward, should spike)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    _, vl, _ = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        subset_batches=args.eval_batches, imagenet_root=args.imagenet_root
    )
    model = _load_model(args, device)

    def _baseline_inference():
        """One full forward pass over one batch - mimics deployment-time inference."""
        model.eval()
        with torch.no_grad():
            for xb, yb in vl:
                xb = xb.to(device)
                _ = model(xb)
                break  # single-batch inference to match attack window granularity

    # WARMUP: run inference once to trigger cudnn algorithm benchmarking and
    # kernel initialization, so the timed measurement reflects steady-state cost.
    _baseline_inference()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    mem_before = gpu_mem_snapshot(device)

    nvml_buf = None
    if args.gpu_idx is not None:
        with nvml_sampler_ctx(args.gpu_idx, args.nvml_period) as nv:
            _, t = cuda_time(_baseline_inference)
            nvml_buf = nvml_reduce(nv)
    else:
        _, t = cuda_time(_baseline_inference)

    mem_after = gpu_mem_snapshot(device)

    print(f"[baseline] inference_time_s: {t:.6f}")
    if nvml_buf:
        print(f"[nvml] util avg/max: {nvml_buf['util_avg']}/{nvml_buf['util_max']}  "
              f"power avg/max W: {nvml_buf['power_avg_W']}/{nvml_buf['power_max_W']}  "
              f"mem max MB: {nvml_buf['mem_max_MB']}")
    print(f"[torch-mem] allocated before/after MB: {mem_before['allocated_MB']:.2f} / {mem_after['allocated_MB']:.2f}  "
          f"peak: {mem_after['max_allocated_MB']:.2f}")

    if args.metrics_out:
        row = {
            "phase": "baseline_overhead",
            "method": "normal_inference",
            "inference_time_s": t,
            "torch_mem_allocated_before_MB": mem_before["allocated_MB"],
            "torch_mem_allocated_after_MB": mem_after["allocated_MB"],
            "torch_mem_peak_MB": mem_after["max_allocated_MB"],
        }
        if nvml_buf:
            row.update(nvml_buf)
        with open(args.metrics_out, "w") as f:
            json.dump([row], f, indent=2)


def cmd_curvature_time(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    _, vl, _ = get_dataloaders(
        args.dataset, args.batch_size, args.num_workers,
        subset_batches=args.eval_batches, imagenet_root=args.imagenet_root
    )
    model = _load_model(args, device)
    loss_fn = nn.CrossEntropyLoss()

    def _hutch():
        return hutchinson_diag_full(model, loss_fn, vl, device, K=args.hutch_k)

    def _exact():
        return exact_fc_diag(model, vl, device)

    _, t_h = cuda_time(_hutch)
    print(f"[curvature] Hutchinson K={args.hutch_k} time = {t_h:.6f}s")
    try:
        _, t_e = cuda_time(_exact)
        print(f"[curvature] exact last-FC Hessian time = {t_e:.6f}s")
        rows = [
            {"phase": "hutchinson", "gpu_time_s": t_h, "K": args.hutch_k},
            {"phase": "exact_last_fc", "gpu_time_s": t_e},
        ]
    except ValueError as e:
        print(f"[curvature] exact_fc unavailable: {e}")
        rows = [
            {"phase": "hutchinson", "gpu_time_s": t_h, "K": args.hutch_k},
        ]

    if args.metrics_out:
        with open(args.metrics_out, "w") as f:
            json.dump(rows, f, indent=2)


# ──────────────────────────────────────
# Parser & main
# ──────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser("AiSPY Bit-Flip Attack Pipeline")
    sub = p.add_subparsers(dest="cmd")

    def add_common(sp):
        sp.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "cifar100", "imagenet"])
        sp.add_argument("--arch", default="resnet18",
                        choices=[
                            "resnet18", "resnet34", "resnet50",
                            "vgg16", "vgg19",
                            "mobilenet_v2",
                            "densenet121",
                            "efficientnet_b0",
                            "convnext_tiny",
                            "vit_b_16",
                        ])
        sp.add_argument("--use-pretrained", type=int, default=0)
        sp.add_argument("--imagenet-root", type=str, default="")
        sp.add_argument("--batch-size", type=int, default=256)
        sp.add_argument("--num-workers", type=int, default=4)
        sp.add_argument("--eval-batches", type=int, default=40)
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--metrics-out", type=str, default="")

        # bit-flip controls
        sp.add_argument("--bit-policy", default="fixed",
                        choices=["fixed", "random_mantissa", "sign", "exponent"])
        sp.add_argument("--bit-pos", type=int, default=20)
        sp.add_argument("--mantissa-min", type=int, default=16)
        sp.add_argument("--mantissa-max", type=int, default=22)

        # NVML
        sp.add_argument("--gpu-idx", type=int, default=None)
        sp.add_argument("--nvml-period", type=float, default=0.01)

    # train
    sp = sub.add_parser("train")
    add_common(sp)
    sp.add_argument("--epochs", type=int, default=10)
    sp.add_argument("--lr", type=float, default=0.1)
    sp.add_argument("--save", type=str, default="ckpt.pth")
    sp.set_defaults(func=cmd_train)

    # train_with_curv
    sp = sub.add_parser("train_with_curv")
    add_common(sp)
    sp.add_argument("--epochs", type=int, default=10)
    sp.add_argument("--lr", type=float, default=0.1)
    sp.add_argument("--save", type=str, default="ckpt_with_curv.pth")
    sp.add_argument("--curv-method", default="hutch_full",
                    choices=["1p_dnl", "hutch_full", "hutch_fc", "exact_fc"])
    sp.add_argument("--curv-last-epochs", type=int, default=3)
    sp.add_argument("--curv-interval", type=int, default=10)
    sp.add_argument("--curv-hutch-k", type=int, default=8)
    sp.add_argument("--curv-top-k", type=int, default=256)
    sp.add_argument("--curv-cache", type=str, default="cache_train_curv.pt")
    sp.add_argument("--composite-score", type=int, default=1,
                    help="1=enable H[i,i]*w^2 composite scoring (default ON), 0=raw Hessian diag only")
    sp.set_defaults(func=cmd_train_with_curv)

    # precompute
    sp = sub.add_parser("precompute")
    add_common(sp)
    sp.add_argument("--ckpt", type=str, default="")
    sp.add_argument("--method", default="1p_dnl",
                    choices=["1p_dnl", "hutch_full", "hutch_fc", "exact_fc"])
    sp.add_argument("--top-k", type=int, default=256)
    sp.add_argument("--cache", type=str, default="cache.pt")
    sp.add_argument("--hutch-k", type=int, default=8)
    sp.add_argument("--composite-score", type=int, default=0,
                    help="1=enable H[i,i]*w^2 composite scoring, 0=raw scores (default)")
    sp.set_defaults(func=cmd_precompute)

    # attack_only
    sp = sub.add_parser("attack_only")
    add_common(sp)
    sp.add_argument("--ckpt", type=str, default="")
    sp.add_argument("--cache", type=str, required=True)
    sp.set_defaults(func=cmd_attack_only)

    # attack_offline
    sp = sub.add_parser("attack_offline")
    add_common(sp)
    sp.add_argument("--ckpt", type=str, default="")
    sp.add_argument("--cache", type=str, required=True)
    sp.set_defaults(func=cmd_attack_offline)

    # attack_online
    sp = sub.add_parser("attack_online")
    add_common(sp)
    sp.add_argument("--ckpt", type=str, default="")
    sp.add_argument("--method", default="1p_dnl",
                    choices=["1p_dnl", "hutch_full", "hutch_fc", "exact_fc"])
    sp.add_argument("--top-k", type=int, default=256)
    sp.add_argument("--hutch-k", type=int, default=8)
    sp.add_argument("--composite-score", type=int, default=0,
                    help="1=enable H[i,i]*w^2 composite scoring, 0=raw scores (default)")
    sp.set_defaults(func=cmd_attack_online)

    # curvature_time
    sp = sub.add_parser("curvature_time")
    add_common(sp)
    sp.add_argument("--ckpt", type=str, default="")
    sp.add_argument("--hutch-k", type=int, default=8)
    sp.set_defaults(func=cmd_curvature_time)

    # baseline_overhead - reference GPU footprint of normal inference
    sp = sub.add_parser("baseline_overhead")
    add_common(sp)
    sp.add_argument("--ckpt", type=str, default="")
    sp.set_defaults(func=cmd_baseline_overhead)

    return p


def main():
    args = build_parser().parse_args()
    if args.eval_batches is not None and args.eval_batches <= 0:
        args.eval_batches = None
    if not hasattr(args, "func"):
        print("Specify subcommand: train | train_with_curv | precompute | attack_only "
              "| attack_offline | attack_online | curvature_time | baseline_overhead")
        return
    args.func(args)


if __name__ == "__main__":
    main()

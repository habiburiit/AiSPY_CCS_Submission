"""
Gradient-Based Sabotage Attack
================================
A training-time attack that injects mathematically derived Gaussian noise
into gradients once convergence is detected. Three detection modes:

  Mode 1 (--detection grad_var) — Gradient Variance Drop  [ResNet, ViT]
    conv_rate = (Var(g)[t-N] - Var(g)[t]) / Var(g)[t-N]
    Latches when variance drops sharply over N epochs.

  Mode 2 (--detection loss_plateau) — Loss Plateau  [VGG+BatchNorm]
    plateau_rate = (L[t-N] - L[t]) / L[t-N]
    Latches when loss drop rate exceeds threshold.
    Used for VGG where grad_var INCREASES during training.

  Mode 3 (--detection grad_norm) — Gradient Norm Near Zero  [ALL models]
    Monitors the L2 norm of all gradients per epoch.
    Latches when grad_norm drops below a threshold fraction of its peak.
    Works universally — grad_norm approaches zero at convergence
    for ALL architectures regardless of BatchNorm or depth.
    Most principled: fires exactly when model is fully converged
    and most fragile — any noise pushes it away from the optimum.

Noise formula (all modes):
    noise_std = alpha * peak_rate * sqrt(current_grad_var)

Once latched, noise is injected every batch permanently — no recovery.

Supports:
  Datasets : CIFAR-10, CIFAR-100, ImageNet
  Models   : ResNet-18, ResNet-50, VGG-16, VGG-19, ViT-B/16
"""

import os
import math
import argparse
import random
import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Gradient Utilities
# =============================================================================

@torch.no_grad()
def compute_grad_var(model, layer_selective: bool = False) -> float:
    """Variance over all gradient elements."""
    grads = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if layer_selective:
            if any(k in name for k in ['layer4', 'fc', 'classifier', 'head']):
                grads.append(p.grad.detach().view(-1))
        else:
            grads.append(p.grad.detach().view(-1))
    if not grads:
        return 0.0
    return torch.var(torch.cat(grads), unbiased=False).item()


@torch.no_grad()
def compute_grad_norm(model, layer_selective: bool = False) -> float:
    """
    L2 norm of all gradients.
    Approaches zero when model has fully converged.
    Universal across all architectures.
    """
    grads = []
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if layer_selective:
            if any(k in name for k in ['layer4', 'fc', 'classifier', 'head']):
                grads.append(p.grad.detach().view(-1))
        else:
            grads.append(p.grad.detach().view(-1))
    if not grads:
        return 0.0
    return torch.norm(torch.cat(grads), p=2).item()


@torch.no_grad()
def inject_noise_(model, noise_std: float, layer_selective: bool = False):
    """
    Inject zero-mean Gaussian noise into gradients before optimizer.step().
    This is the ONLY modification to the training loop.
    """
    if noise_std <= 0.0:
        return
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if layer_selective:
            if any(k in name for k in ['layer4', 'fc', 'classifier', 'head']):
                p.grad.add_(torch.randn_like(p.grad) * noise_std)
        else:
            p.grad.add_(torch.randn_like(p.grad) * noise_std)


# =============================================================================
# Convergence Detection — Three Modes
# =============================================================================

def compute_grad_var_rate(var_history: List[float], N: int) -> float:
    """
    Mode 1: Relative gradient variance drop over last N epochs.
    rate = (var[t-N] - var[t]) / var[t-N]
    Returns 0 if variance increased or history too short.
    """
    if len(var_history) < N + 1:
        return 0.0
    old = var_history[-(N + 1)]
    new = var_history[-1]
    if old <= 0.0:
        return 0.0
    return float(max(0.0, min((old - new) / old, 1.0)))


def compute_loss_plateau_rate(loss_history: List[float], N: int) -> float:
    """
    Mode 2: Relative training loss drop over last N epochs.
    rate = (loss[t-N] - loss[t]) / loss[t-N]
    For VGG+BatchNorm where grad_var increases.
    """
    if len(loss_history) < N + 1:
        return 0.0
    old = loss_history[-(N + 1)]
    new = loss_history[-1]
    if old <= 0.0:
        return 0.0
    return float(max(0.0, min((old - new) / old, 1.0)))


def compute_grad_norm_rate(
    norm_history: List[float],
    peak_norm:    float,
    norm_threshold: float,
) -> Tuple[float, bool]:
    """
    Mode 3: Gradient norm near zero detection.

    Monitors the L2 norm of all gradients.
    Fires when current norm drops below (norm_threshold * peak_norm).

        trigger = current_norm < norm_threshold * peak_norm

    This is the most principled detection mode:
    - Grad norm approaches zero universally at convergence
    - Works for ALL architectures (ResNet, VGG, ViT, ImageNet, CIFAR)
    - Fires LATE — only when model is truly at minimum
    - No dependency on loss shape or variance behavior

    Returns:
        (current_norm, triggered: bool)
    """
    if not norm_history:
        return 0.0, False
    current_norm = norm_history[-1]
    if peak_norm <= 0.0:
        return current_norm, False
    triggered = current_norm < norm_threshold * peak_norm
    return current_norm, triggered


def derive_noise_std(
    peak_rate:        float,
    current_grad_var: float,
    alpha:            float,
) -> float:
    """
    Mathematical noise formula:
        noise_std = alpha * peak_rate * sqrt(current_grad_var)

    - alpha          : damage budget constant
    - peak_rate      : latched peak convergence rate (never decreases)
    - sqrt(grad_var) : natural scale of gradient magnitudes

    Noise is always proportional to gradient signal → stealthy.
    Noise-to-gradient ratio = alpha * peak_rate ≤ alpha (bounded).
    """
    return alpha * peak_rate * math.sqrt(max(current_grad_var, 1e-12))


def derive_noise_std_norm(
    current_grad_var: float,
    alpha:            float,
    peak_norm:        float,
    current_norm:     float,
) -> float:
    """
    Noise formula for Mode 3 (grad norm near zero):

        noise_std = alpha * (1 - current_norm/peak_norm) * sqrt(grad_var)

    The factor (1 - current_norm/peak_norm) ranges from 0 to 1:
    - When norm is at peak (early training) → factor ≈ 0 → no noise
    - When norm approaches zero (full convergence) → factor → 1 → max noise

    This is self-calibrating — noise scales with how close to zero
    the gradient norm is. Maximum damage when fully converged.
    """
    if peak_norm <= 0.0:
        return 0.0
    convergence_factor = max(0.0, 1.0 - current_norm / peak_norm)
    grad_scale = math.sqrt(max(current_grad_var, 1e-12))
    return alpha * convergence_factor * grad_scale


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def eval_accuracy(model, loader, device, max_batches=None) -> float:
    model.eval()
    correct, total = 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total   += y.numel()
    model.train()
    return correct / max(total, 1)


# =============================================================================
# Model Definitions
# =============================================================================

def make_cifar_resnet18(num_classes: int) -> nn.Module:
    m = torchvision.models.resnet18(pretrained=False)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, num_classes)
    return m


def make_cifar_resnet50(num_classes: int) -> nn.Module:
    m = torchvision.models.resnet50(pretrained=False)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(m.fc.in_features, num_classes)
    return m


def make_cifar_vgg16(num_classes: int) -> nn.Module:
    class VGG16_CIFAR(nn.Module):
        def __init__(self, nc):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3,64,3,padding=1),   nn.BatchNorm2d(64),  nn.ReLU(True),
                nn.Conv2d(64,64,3,padding=1),  nn.BatchNorm2d(64),  nn.ReLU(True),
                nn.MaxPool2d(2,2),
                nn.Conv2d(64,128,3,padding=1),  nn.BatchNorm2d(128), nn.ReLU(True),
                nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.MaxPool2d(2,2),
                nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.MaxPool2d(2,2),
                nn.Conv2d(256,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.MaxPool2d(2,2),
                nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.MaxPool2d(2,2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(512,512), nn.ReLU(True), nn.Dropout(0.5),
                nn.Linear(512,512), nn.ReLU(True), nn.Dropout(0.5),
                nn.Linear(512, nc),
            )
            self._init_w()
        def forward(self, x):
            return self.classifier(self.features(x).view(x.size(0), -1))
        def _init_w(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None: nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)
    return VGG16_CIFAR(num_classes)


def make_cifar_vgg19(num_classes: int) -> nn.Module:
    class VGG19_CIFAR(nn.Module):
        def __init__(self, nc):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3,64,3,padding=1),   nn.BatchNorm2d(64),  nn.ReLU(True),
                nn.Conv2d(64,64,3,padding=1),  nn.BatchNorm2d(64),  nn.ReLU(True),
                nn.MaxPool2d(2,2),
                nn.Conv2d(64,128,3,padding=1),  nn.BatchNorm2d(128), nn.ReLU(True),
                nn.Conv2d(128,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
                nn.MaxPool2d(2,2),
                nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.Conv2d(256,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
                nn.MaxPool2d(2,2),
                nn.Conv2d(256,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.MaxPool2d(2,2),
                nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.Conv2d(512,512,3,padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
                nn.MaxPool2d(2,2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(512,512), nn.ReLU(True), nn.Dropout(0.5),
                nn.Linear(512,512), nn.ReLU(True), nn.Dropout(0.5),
                nn.Linear(512, nc),
            )
            self._init_w()
        def forward(self, x):
            return self.classifier(self.features(x).view(x.size(0), -1))
        def _init_w(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None: nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)
    return VGG19_CIFAR(num_classes)


def make_imagenet_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if model_name == 'resnet18':
        m = torchvision.models.resnet18(pretrained=pretrained)
        if num_classes != 1000: m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == 'resnet50':
        m = torchvision.models.resnet50(pretrained=pretrained)
        if num_classes != 1000: m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif model_name == 'vgg16':
        m = torchvision.models.vgg16_bn(pretrained=pretrained)
        if num_classes != 1000: m.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'vgg19':
        m = torchvision.models.vgg19_bn(pretrained=pretrained)
        if num_classes != 1000: m.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name == 'vit_b_16':
        m = torchvision.models.vit_b_16(pretrained=pretrained)
        if num_classes != 1000: m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return m


# =============================================================================
# Dataset Loading
# =============================================================================

def get_cifar_dataloaders(
    dataset: str, data_dir: str, batch_size: int, num_workers: int = 2
) -> Tuple[DataLoader, DataLoader, int]:
    if dataset == 'cifar10':
        mean, std   = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        num_classes = 10
        cls         = torchvision.datasets.CIFAR10
    else:
        mean, std   = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        num_classes = 100
        cls         = torchvision.datasets.CIFAR100

    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(mean, std)])
    test_tf  = T.Compose([T.ToTensor(), T.Normalize(mean, std)])

    train_loader = DataLoader(cls(data_dir, True,  transform=train_tf, download=True),
                              batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(cls(data_dir, False, transform=test_tf,  download=True),
                              batch_size=256, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, num_classes


def _find_imagenet_val_dir(data_dir: str) -> str:
    for candidate in [os.path.join(data_dir, 'val'), data_dir]:
        if not os.path.exists(candidate):
            continue
        subdirs = [f for f in os.listdir(candidate)
                   if os.path.isdir(os.path.join(candidate, f))]
        if len(subdirs) >= 100:
            print(f"  Found {len(subdirs)} class folders in: {candidate}")
            return candidate
    raise ValueError(
        f"Cannot find ImageNet class folders.\n"
        f"  Tried: {os.path.join(data_dir, 'val')}  and  {data_dir}\n"
    )


def get_imagenet_dataloaders(
    data_dir:         str,
    batch_size:       int,
    num_workers:      int   = 4,
    train_samples:    Optional[int]   = None,
    use_val_as_train: bool  = False,
    val_split_ratio:  float = 0.8,
) -> Tuple[Optional[DataLoader], DataLoader, int]:

    mean, std   = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    num_classes = 1000
    train_tf = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize(mean, std)])
    val_tf   = T.Compose([T.Resize(256), T.CenterCrop(224),
                          T.ToTensor(), T.Normalize(mean, std)])

    val_dir   = _find_imagenet_val_dir(data_dir)
    train_dir = os.path.join(data_dir, 'train')

    if use_val_as_train:
        full_ds   = ImageFolder(val_dir, transform=train_tf)
        indices   = list(range(len(full_ds)))
        random.shuffle(indices)
        t_size    = int(len(full_ds) * val_split_ratio)
        train_set = torch.utils.data.Subset(full_ds, indices[:t_size])
        val_set   = torch.utils.data.Subset(
                        ImageFolder(val_dir, transform=val_tf), indices[t_size:])
        print(f"  Split val -> train:{len(train_set)}  val:{len(val_set)}")
    elif os.path.exists(train_dir):
        train_set = ImageFolder(train_dir, transform=train_tf)
        if train_samples and train_samples < len(train_set):
            train_set = torch.utils.data.Subset(
                train_set, random.sample(range(len(train_set)), train_samples))
        val_set = ImageFolder(val_dir, transform=val_tf)
    else:
        print("  No train folder — val-only mode.")
        train_set = None
        val_set   = ImageFolder(val_dir, transform=val_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True) \
                   if train_set is not None else None
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, num_classes


# =============================================================================
# Training Loop
# =============================================================================

def train_one_run(
    adaptive_noise:  bool,
    model:           nn.Module,
    train_loader:    DataLoader,
    test_loader:     DataLoader,
    device:          str,
    epochs:          int,
    lr:              float,
    momentum:        float,
    weight_decay:    float,
    lr_milestones:   List[int],
    lr_gamma:        float,
    window_N:        int,
    conv_threshold:  float,
    alpha:           float,
    layer_selective: bool,
    detection:       str   = 'grad_norm',   # 'grad_var' | 'loss_plateau' | 'grad_norm'
    norm_threshold:  float = 0.3,           # for Mode 3: fire when norm < norm_threshold * peak
    eval_batches:    Optional[int] = None,
) -> Dict:
    """
    Train model. Three detection modes for convergence:

    Mode 1 (grad_var):     fires when variance drops by conv_threshold over N epochs
    Mode 2 (loss_plateau): fires when loss drops by conv_threshold over N epochs
    Mode 3 (grad_norm):    fires when grad L2 norm < norm_threshold * peak_norm
                           Most principled — works universally for all models
    """
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=lr_milestones, gamma=lr_gamma)

    history = {
        "train_loss":       [],
        "test_acc":         [],
        "epoch_grad_var":   [],
        "epoch_grad_norm":  [],
        "noise_std":        [],
        "convergence_rate": [],
        "attack_active":    [],
    }

    epoch_var_history:  List[float] = []
    epoch_loss_history: List[float] = []
    epoch_norm_history: List[float] = []

    attack_mode     = False
    peak_rate       = 0.0
    peak_norm       = 0.0          # Mode 3: track peak grad norm
    epoch_noise_std = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches    = 0
        batch_vars   = []
        batch_norms  = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()

            # ── measure gradient stats before noise ───────────────────────
            gv = compute_grad_var(model, layer_selective)
            gn = compute_grad_norm(model, layer_selective)
            batch_vars.append(gv)
            batch_norms.append(gn)

            # ── inject noise if latched ───────────────────────────────────
            if adaptive_noise and attack_mode:
                if detection == 'grad_norm':
                    # Mode 3: self-calibrating noise
                    noise_std = derive_noise_std_norm(gv, alpha, peak_norm,
                                                      batch_norms[-1])
                else:
                    # Mode 1 & 2: standard formula
                    noise_std = derive_noise_std(peak_rate, gv, alpha)
                inject_noise_(model, noise_std, layer_selective)
                epoch_noise_std = noise_std
            else:
                epoch_noise_std = 0.0

            optimizer.step()
            running_loss += loss.item()
            n_batches    += 1

        scheduler.step()

        epoch_avg_var  = float(np.mean(batch_vars))  if batch_vars  else 0.0
        epoch_avg_norm = float(np.mean(batch_norms)) if batch_norms else 0.0
        epoch_avg_loss = running_loss / max(n_batches, 1)

        epoch_var_history.append(epoch_avg_var)
        epoch_loss_history.append(epoch_avg_loss)
        epoch_norm_history.append(epoch_avg_norm)

        # ── update peak norm (Mode 3) ─────────────────────────────────────
        # Skip first 10 epochs — avoids false early latch from random-init spike
        if detection == 'grad_norm' and epoch > 10:
            peak_norm = max(peak_norm, epoch_avg_norm)

        # ── convergence detection ─────────────────────────────────────────
        if detection == 'grad_var':
            conv_rate  = compute_grad_var_rate(epoch_var_history, window_N)
            mode_label = "grad_var"
            triggered  = conv_rate >= conv_threshold

        elif detection == 'loss_plateau':
            conv_rate  = compute_loss_plateau_rate(epoch_loss_history, window_N)
            mode_label = "loss_plateau"
            triggered  = conv_rate >= conv_threshold

        else:  # grad_norm
            current_norm, triggered = compute_grad_norm_rate(
                epoch_norm_history, peak_norm, norm_threshold)
            if epoch <= 10:
                triggered = False
            # conv_rate for logging = how close norm is to zero relative to peak
            conv_rate  = max(0.0, 1.0 - epoch_avg_norm / max(peak_norm, 1e-12))
            mode_label = "grad_norm"

        if adaptive_noise and triggered:
            if not attack_mode:
                print(f"  *** ATTACK LATCHED at epoch {epoch} "
                      f"({mode_label} triggered | "
                      f"norm={epoch_avg_norm:.3e} peak={peak_norm:.3e}) ***")
            attack_mode = True
            peak_rate   = max(peak_rate, conv_rate)

        # ── evaluation ────────────────────────────────────────────────────
        test_acc = eval_accuracy(model, test_loader, device, eval_batches)

        history["train_loss"].append(epoch_avg_loss)
        history["test_acc"].append(test_acc)
        history["epoch_grad_var"].append(epoch_avg_var)
        history["epoch_grad_norm"].append(epoch_avg_norm)
        history["noise_std"].append(epoch_noise_std)
        history["convergence_rate"].append(conv_rate)
        history["attack_active"].append(1 if attack_mode else 0)

        if epoch % 5 == 0 or epoch <= 5 or epoch >= epochs - 4:
            print(
                f"  Epoch {epoch:03d}/{epochs} | "
                f"loss {epoch_avg_loss:.4f} | "
                f"acc {test_acc*100:.2f}% | "
                f"grad_norm {epoch_avg_norm:.3e} | "
                f"peak_norm {peak_norm:.3e} | "
                f"grad_var {epoch_avg_var:.3e} | "
                f"conv_rate {conv_rate:.3f} | "
                f"noise_std {epoch_noise_std:.3e} | "
                f"attack {'ON ' if attack_mode else 'OFF'}"
            )

    return history


# =============================================================================
# Plotting
# =============================================================================

def create_all_plots(
    baseline_hist: Dict,
    noisy_hist:    Dict,
    epochs:        int,
    output_dir:    str,
    dataset:       str,
    model_name:    str,
    detection:     str,
):
    ep  = np.arange(1, epochs + 1)
    pfx = f"{dataset.upper()} / {model_name.upper()}"
    os.makedirs(output_dir, exist_ok=True)

    def _save(fig, name):
        p = os.path.join(output_dir, name)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {p}")

    def _shade(ax, hist):
        for i, a in enumerate(hist["attack_active"]):
            if a:
                ax.axvline(ep[i], color='red', linewidth=1.2,
                           linestyle='--', alpha=0.7, label='Attack onset')
                ax.axvspan(ep[i], ep[-1], color='red', alpha=0.06)
                break

    # 1. Training Loss
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ep, baseline_hist["train_loss"], label="Baseline",       linewidth=1.8)
    ax.plot(ep, noisy_hist["train_loss"],    label="Sabotage Attack", linewidth=1.8, linestyle='--')
    _shade(ax, noisy_hist)
    ax.set_title(f"{pfx} — Training Loss [{detection}]", fontsize=13)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); _save(fig, "1_training_loss.png")

    # 2. Test Accuracy
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ep, [v*100 for v in baseline_hist["test_acc"]], label="Baseline",       linewidth=1.8)
    ax.plot(ep, [v*100 for v in noisy_hist["test_acc"]],    label="Sabotage Attack", linewidth=1.8, linestyle='--')
    _shade(ax, noisy_hist)
    ax.set_title(f"{pfx} — Test Accuracy [{detection}]", fontsize=13)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); _save(fig, "2_test_accuracy.png")

    # 3. Gradient Norm
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(ep, baseline_hist["epoch_grad_norm"], label="Baseline",       linewidth=1.8)
    ax.semilogy(ep, noisy_hist["epoch_grad_norm"],    label="Sabotage Attack", linewidth=1.8, linestyle='--')
    _shade(ax, noisy_hist)
    ax.set_title(f"{pfx} — Gradient L2 Norm [log scale]", fontsize=13)
    ax.set_xlabel("Epoch"); ax.set_ylabel("||g||_2  [log]")
    ax.legend(); ax.grid(alpha=0.3, which='both'); fig.tight_layout()
    _save(fig, "3_gradient_norm.png")

    # 4. Gradient Variance
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(ep, baseline_hist["epoch_grad_var"], label="Baseline",       linewidth=1.8)
    ax.semilogy(ep, noisy_hist["epoch_grad_var"],    label="Sabotage Attack", linewidth=1.8, linestyle='--')
    ax.set_title(f"{pfx} — Gradient Variance [log scale]", fontsize=13)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Var(g)  [log]")
    ax.legend(); ax.grid(alpha=0.3, which='both'); fig.tight_layout()
    _save(fig, "4_gradient_variance.png")

    # 5. Convergence Rate
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ep, noisy_hist["convergence_rate"], color='darkorange', linewidth=1.8)
    for i, a in enumerate(noisy_hist["attack_active"]):
        if a:
            ax.axvline(ep[i], color='red', linewidth=1.2, linestyle='--',
                       alpha=0.6, label='Attack onset')
            break
    ax.set_title(f"{pfx} — Convergence Signal [{detection}]", fontsize=12)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Rate / Factor [0-1]")
    ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, "5_convergence_rate.png")

    # 6. Noise Std
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ep, noisy_hist["noise_std"], color='crimson', linewidth=1.8)
    ax.fill_between(ep, noisy_hist["noise_std"], alpha=0.2, color='crimson')
    ax.set_title(f"{pfx} — Injected Noise Std", fontsize=12)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Noise sigma")
    ax.grid(alpha=0.3); fig.tight_layout(); _save(fig, "6_noise_std.png")

    # 7. Accuracy Gap
    acc_gap = [(b - n)*100 for b, n in
               zip(baseline_hist["test_acc"], noisy_hist["test_acc"])]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.fill_between(ep, acc_gap, alpha=0.35, color='steelblue')
    ax.plot(ep, acc_gap, color='steelblue', linewidth=1.8)
    ax.axhline(y=0, color='black', linewidth=0.8)
    _shade(ax, noisy_hist)
    ax.set_title(f"{pfx} — Accuracy Gap (Baseline - Attacked)", fontsize=12)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Delta Accuracy (pp)")
    ax.grid(alpha=0.3); fig.tight_layout(); _save(fig, "7_accuracy_gap.png")

    # 8. Attack Flag
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.step(ep, noisy_hist["attack_active"], color='red', linewidth=2, where='post')
    ax.fill_between(ep, noisy_hist["attack_active"], alpha=0.25, color='red', step='post')
    ax.set_yticks([0, 1]); ax.set_yticklabels(['OFF', 'ON'])
    ax.set_title(f"{pfx} — Attack Active (sustained after latch)", fontsize=12)
    ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    fig.tight_layout(); _save(fig, "8_attack_flag.png")

    # 0. Summary Grid
    fig = plt.figure(figsize=(20, 12))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[r, c]) for r in range(3) for c in range(3)]

    axes[0].plot(ep, baseline_hist["train_loss"], label="Baseline")
    axes[0].plot(ep, noisy_hist["train_loss"],    label="Attack", linestyle='--')
    _shade(axes[0], noisy_hist)
    axes[0].set_title("Training Loss"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, [v*100 for v in baseline_hist["test_acc"]], label="Baseline")
    axes[1].plot(ep, [v*100 for v in noisy_hist["test_acc"]],    label="Attack", linestyle='--')
    _shade(axes[1], noisy_hist)
    axes[1].set_title("Test Accuracy (%)"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)

    axes[2].semilogy(ep, baseline_hist["epoch_grad_norm"], label="Baseline")
    axes[2].semilogy(ep, noisy_hist["epoch_grad_norm"],    label="Attack", linestyle='--')
    axes[2].set_title("Grad Norm [log]"); axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3, which='both')

    axes[3].semilogy(ep, baseline_hist["epoch_grad_var"], label="Baseline")
    axes[3].semilogy(ep, noisy_hist["epoch_grad_var"],    label="Attack", linestyle='--')
    axes[3].set_title("Grad Variance [log]"); axes[3].legend(fontsize=7)
    axes[3].grid(alpha=0.3, which='both')

    axes[4].plot(ep, noisy_hist["convergence_rate"], color='darkorange')
    axes[4].set_title(f"Conv Signal [{detection}]"); axes[4].grid(alpha=0.3)

    axes[5].plot(ep, noisy_hist["noise_std"], color='crimson')
    axes[5].fill_between(ep, noisy_hist["noise_std"], alpha=0.2, color='crimson')
    axes[5].set_title("Noise Sigma"); axes[5].grid(alpha=0.3)

    axes[6].fill_between(ep, acc_gap, alpha=0.35, color='steelblue')
    axes[6].plot(ep, acc_gap, color='steelblue')
    axes[6].axhline(0, color='black', linewidth=0.8)
    axes[6].set_title("Accuracy Gap (pp)"); axes[6].grid(alpha=0.3)

    axes[7].step(ep, noisy_hist["attack_active"], color='red', linewidth=2, where='post')
    axes[7].fill_between(ep, noisy_hist["attack_active"], alpha=0.25, color='red', step='post')
    axes[7].set_yticks([0,1]); axes[7].set_yticklabels(['OFF','ON'])
    axes[7].set_title("Attack Active"); axes[7].grid(alpha=0.3)

    axes[8].plot(ep, [v*100 for v in baseline_hist["test_acc"]], color='steelblue', label="Baseline")
    axes[8].plot(ep, [v*100 for v in noisy_hist["test_acc"]],    color='darkorange', label="Attack")
    axes[8].set_title("Accuracy Comparison"); axes[8].legend(fontsize=7); axes[8].grid(alpha=0.3)

    fig.suptitle(
        f"{pfx} — Gradient-Based Sabotage Attack [{detection} detection]",
        fontsize=14, fontweight='bold'
    )
    _save(fig, "0_summary_grid.png")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gradient-Based Sabotage Attack",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── dataset / model ───────────────────────────────────────────────────
    parser.add_argument("--dataset", default="cifar10",
                        choices=["cifar10", "cifar100", "imagenet"])
    parser.add_argument("--model",   default="resnet18",
                        choices=["resnet18", "resnet50", "vgg16", "vgg19", "vit_b_16"])
    parser.add_argument("--data_dir",    default="./data")
    parser.add_argument("--num_workers", type=int, default=4)

    # ── imagenet extras ───────────────────────────────────────────────────
    parser.add_argument("--imagenet_pretrained",    action="store_true")
    parser.add_argument("--imagenet_train_samples", type=int, default=None)
    parser.add_argument("--imagenet_eval_batches",  type=int, default=None)
    parser.add_argument("--use_val_as_train",       action="store_true")
    parser.add_argument("--val_split_ratio",        type=float, default=0.8)

    # ── training ──────────────────────────────────────────────────────────
    parser.add_argument("--batch_size",    type=int,   default=None)
    parser.add_argument("--epochs",        type=int,   default=None)
    parser.add_argument("--lr",            type=float, default=0.1)
    parser.add_argument("--momentum",      type=float, default=0.9)
    parser.add_argument("--weight_decay",  type=float, default=None)
    parser.add_argument("--lr_milestones", type=int, nargs="+", default=None)
    parser.add_argument("--lr_gamma",      type=float, default=0.1)

    # ── detection mode ────────────────────────────────────────────────────
    parser.add_argument("--detection", default="grad_norm",
                        choices=["grad_var", "loss_plateau", "grad_norm"],
                        help=(
                            "grad_var: Mode 1 — gradient variance drop [ResNet/ViT] | "
                            "loss_plateau: Mode 2 — training loss plateau [VGG] | "
                            "grad_norm: Mode 3 — gradient L2 norm near zero [ALL models, recommended]"
                        ))
    parser.add_argument("--window_N",       type=int,   default=5,
                        help="Window size for Mode 1 and Mode 2")
    parser.add_argument("--conv_threshold", type=float, default=0.10,
                        help="Convergence threshold for Mode 1 and Mode 2")
    parser.add_argument("--norm_threshold", type=float, default=0.3,
                        help="Mode 3: latch when grad_norm < norm_threshold * peak_norm. "
                             "Lower = later latch (more stealthy). Range: 0.1-0.5")

    # ── noise budget ──────────────────────────────────────────────────────
    parser.add_argument("--alpha",           type=float, default=None,
                        help="Damage budget: noise_std = alpha * rate * sqrt(grad_var)")
    parser.add_argument("--layer_selective", action="store_true",
                        help="Inject noise only in layer4/fc/classifier/head")

    # ── misc ──────────────────────────────────────────────────────────────
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--device", default=None)

    args = parser.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── dataset defaults ──────────────────────────────────────────────────
    if args.dataset in ('cifar10', 'cifar100'):
        if args.batch_size    is None: args.batch_size    = 64 if 'vgg' in args.model else 128
        if args.epochs        is None: args.epochs        = 120 if args.dataset == 'cifar10' else 200
        if args.weight_decay  is None: args.weight_decay  = 5e-4
        if args.lr_milestones is None:
            args.lr_milestones = [72, 102] if args.dataset == 'cifar10' else [120, 170]
        if args.alpha         is None: args.alpha         = 5.0

    elif args.dataset == 'imagenet':
        if args.use_val_as_train: args.imagenet_pretrained = True
        if args.batch_size    is None: args.batch_size    = 256
        if args.epochs        is None: args.epochs        = 30 if args.imagenet_pretrained else 90
        if args.weight_decay  is None: args.weight_decay  = 1e-4
        if args.lr_milestones is None:
            args.lr_milestones = [15, 25] if args.imagenet_pretrained else [30, 60, 80]
        if args.alpha         is None: args.alpha         = 10.0

    if args.outdir is None:
        args.outdir = f"./results_{args.dataset}_{args.model}_{args.detection}"

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    # ── banner ────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  GRADIENT-BASED SABOTAGE ATTACK")
    print("="*80)
    print(f"  Dataset           : {args.dataset.upper()}")
    print(f"  Model             : {args.model.upper()}")
    print(f"  Epochs            : {args.epochs}")
    print(f"  Batch size        : {args.batch_size}")
    print(f"  LR                : {args.lr}  milestones={args.lr_milestones}")
    print(f"  Device            : {args.device}")
    print(f"\n  [Detection Mode]")
    print(f"  Mode              : {args.detection}")
    if args.detection == 'grad_norm':
        print(f"  Norm threshold    : {args.norm_threshold}  "
              f"(latch when norm < {args.norm_threshold} x peak_norm)")
    else:
        print(f"  Window N          : {args.window_N}")
        print(f"  Conv threshold    : {args.conv_threshold}")
    print(f"\n  [Noise Formula]")
    print(f"  Alpha             : {args.alpha}")
    print(f"  noise_std         = alpha x rate x sqrt(grad_var)")
    print(f"  Sustained         : YES — noise never stops after latch")
    print(f"  Layer selective   : {args.layer_selective}")
    print(f"\n  Output dir        : {args.outdir}")
    print("="*80 + "\n")

    # ── load data ─────────────────────────────────────────────────────────
    print("Loading dataset...")
    if args.dataset in ('cifar10', 'cifar100'):
        train_loader, test_loader, num_classes = get_cifar_dataloaders(
            args.dataset, args.data_dir, args.batch_size, args.num_workers)
        print(f"  Train: {len(train_loader.dataset)}"
              f"  Test: {len(test_loader.dataset)}"
              f"  Classes: {num_classes}\n")
        def model_factory():
            if args.model == 'resnet18': return make_cifar_resnet18(num_classes)
            if args.model == 'resnet50': return make_cifar_resnet50(num_classes)
            if args.model == 'vgg16':   return make_cifar_vgg16(num_classes)
            if args.model == 'vgg19':   return make_cifar_vgg19(num_classes)
            raise ValueError(f"Model {args.model} not supported for CIFAR")
        eval_batches = None
    else:
        train_loader, test_loader, num_classes = get_imagenet_dataloaders(
            args.data_dir, args.batch_size, args.num_workers,
            args.imagenet_train_samples, args.use_val_as_train, args.val_split_ratio)
        t = len(train_loader.dataset) if train_loader else 0
        print(f"  Train: {t}  Val: {len(test_loader.dataset)}  Classes: {num_classes}\n")
        def model_factory():
            return make_imagenet_model(args.model, num_classes, args.imagenet_pretrained)
        eval_batches = args.imagenet_eval_batches

    run_kwargs = dict(
        train_loader    = train_loader if train_loader else test_loader,
        test_loader     = test_loader,
        device          = args.device,
        epochs          = args.epochs,
        lr              = args.lr,
        momentum        = args.momentum,
        weight_decay    = args.weight_decay,
        lr_milestones   = args.lr_milestones,
        lr_gamma        = args.lr_gamma,
        window_N        = args.window_N,
        conv_threshold  = args.conv_threshold,
        alpha           = args.alpha,
        layer_selective = args.layer_selective,
        detection       = args.detection,
        norm_threshold  = args.norm_threshold,
        eval_batches    = eval_batches,
    )

    # ── baseline ──────────────────────────────────────────────────────────
    print("="*80)
    print("  BASELINE RUN  (no noise)")
    print("="*80)
    baseline_model = model_factory()
    baseline_hist  = train_one_run(adaptive_noise=False, model=baseline_model, **run_kwargs)

    # ── attack ────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("  SABOTAGE ATTACK RUN")
    print("="*80)
    set_seed(args.seed)
    noisy_model = model_factory()
    noisy_hist  = train_one_run(adaptive_noise=True, model=noisy_model, **run_kwargs)

    # ── results ───────────────────────────────────────────────────────────
    baseline_final = baseline_hist["test_acc"][-1] * 100
    noisy_final    = noisy_hist["test_acc"][-1]    * 100
    abs_drop       = baseline_final - noisy_final
    rel_drop       = abs_drop / max(baseline_final, 1e-6) * 100
    attack_epoch   = next((i+1 for i, a in enumerate(noisy_hist["attack_active"]) if a), None)

    print("\n" + "="*80)
    print("  FINAL RESULTS")
    print("="*80)
    print(f"  Baseline accuracy   : {baseline_final:.2f}%")
    print(f"  Attacked accuracy   : {noisy_final:.2f}%")
    print(f"  Absolute drop       : {abs_drop:.2f} pp")
    print(f"  Relative drop       : {rel_drop:.2f}%")
    print(f"  Attack latched at   : epoch {attack_epoch}")
    print(f"  Detection mode      : {args.detection}")
    print("="*80)

    print("\nGenerating plots...")
    create_all_plots(baseline_hist, noisy_hist, args.epochs,
                     args.outdir, args.dataset, args.model, args.detection)

    rfile = os.path.join(args.outdir, "results.txt")
    with open(rfile, "w") as f:
        f.write("GRADIENT-BASED SABOTAGE ATTACK\n")
        f.write("="*60 + "\n")
        f.write(f"Dataset          : {args.dataset.upper()}\n")
        f.write(f"Model            : {args.model.upper()}\n")
        f.write(f"Epochs           : {args.epochs}\n")
        f.write(f"LR               : {args.lr}\n")
        f.write(f"Detection mode   : {args.detection}\n")
        if args.detection == 'grad_norm':
            f.write(f"Norm threshold   : {args.norm_threshold}\n")
        else:
            f.write(f"Window N         : {args.window_N}\n")
            f.write(f"Conv threshold   : {args.conv_threshold}\n")
        f.write(f"Alpha            : {args.alpha}\n")
        f.write(f"Layer selective  : {args.layer_selective}\n")
        f.write(f"Attack latched   : epoch {attack_epoch}\n\n")
        f.write(f"Baseline accuracy: {baseline_final:.2f}%\n")
        f.write(f"Attacked accuracy: {noisy_final:.2f}%\n")
        f.write(f"Absolute drop    : {abs_drop:.2f} pp\n")
        f.write(f"Relative drop    : {rel_drop:.2f}%\n")

    print(f"  Results -> {rfile}")
    print("\n  DONE.\n")


if __name__ == "__main__":
    main()

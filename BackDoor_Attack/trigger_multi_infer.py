# train_mlp_trigger_multi.py
# -*- coding: utf-8 -*-
"""
Train an MLP to detect presence of backdoor trigger patches (circle/square/rectangle).
Supports multiple trigger types: per-image at most one trigger, different images can get different triggers.
Metrics: identification accuracy, precision, recall.
Dataset-agnostic: example uses CIFAR-10 but trigger functions work for any PIL images.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Any, Tuple, List, Union, Dict
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import math

import numpy as np
from PIL import Image, ImageDraw

from sympy import frac
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
# from patch_vis import make_denorm, show_batch

import torchvision
import torchvision.transforms as T
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -------------------------
# RNG helpers
# -------------------------
def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def worker_init_fn_builder(base_seed: int):
    def _init(worker_id: int):
        s = base_seed + worker_id
        random.seed(s)
        np.random.seed(s)
        torch.manual_seed(s)
    return _init

# -------------------------
# Triggers (dataset-agnostic)
# -------------------------
def _resolve_xy(position, w, h, pw, ph, margin):
    if isinstance(position, str):
        x = w - pw - margin if "right" in position else margin
        y = h - ph - margin if "bottom" in position else margin
    else:
        x, y = position
    return int(x), int(y)

def _apply_jitter(x, y, jitter, rng):
    if jitter > 0:
        x += int(rng.integers(-jitter, jitter + 1))
        y += int(rng.integers(-jitter, jitter + 1))
    return x, y

class Trigger:
    def __call__(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        raise NotImplementedError

class SquareTrigger(Trigger):
    def __init__(self, size_px: Optional[int]=None, size_frac: Optional[float]=None,
                 position: Any = "bottom_right", color=(255,255,0), margin=2, jitter=0):
        self.size_px = size_px
        self.size_frac = size_frac
        self.position = position
        self.color = tuple(int(c) for c in color)
        self.margin = margin
        self.jitter = jitter

    def __call__(self, img, rng):
        img = img.convert("RGB")
        w, h = img.size
        if self.size_px is not None:
            s = int(self.size_px)
        elif self.size_frac is not None:
            s = max(1, int(min(w, h) * float(self.size_frac)))
        else:
            raise ValueError("Provide size_px or size_frac")
        x, y = _resolve_xy(self.position, w, h, s, s, self.margin)
        x, y = _apply_jitter(x, y, self.jitter, rng)
        arr = np.array(img, copy=True)
        x1,y1 = max(0,x), max(0,y)
        x2,y2 = min(w, x1+s), min(h, y1+s)
        arr[y1:y2, x1:x2] = np.array(self.color, dtype=arr.dtype)
        return Image.fromarray(arr)

class RectangleTrigger(Trigger):
    def __init__(self, width_px: Optional[int]=None, height_px: Optional[int]=None,
                 width_frac: Optional[float]=None, height_frac: Optional[float]=None,
                 position: Any = "bottom_right", color=(255,255,0), margin=2, jitter=0):
        self.width_px = width_px; self.height_px = height_px
        self.width_frac = width_frac; self.height_frac = height_frac
        self.position = position
        self.color = tuple(int(c) for c in color)
        self.margin = margin
        self.jitter = jitter

    def __call__(self, img, rng):
        img = img.convert("RGB")
        w, h = img.size
        if self.width_px is not None and self.height_px is not None:
            pw, ph = int(self.width_px), int(self.height_px)
        elif self.width_frac is not None and self.height_frac is not None:
            pw = max(1, int(w * float(self.width_frac)))
            ph = max(1, int(h * float(self.height_frac)))
        else:
            raise ValueError("Provide width/height in px or frac")
        x, y = _resolve_xy(self.position, w, h, pw, ph, self.margin)
        x, y = _apply_jitter(x, y, self.jitter, rng)
        arr = np.array(img, copy=True)
        x1,y1 = max(0,x), max(0,y)
        x2,y2 = min(w, x1+pw), min(h, y1+ph)
        arr[y1:y2, x1:x2] = np.array(self.color, dtype=arr.dtype)
        return Image.fromarray(arr)

class CircleTrigger(Trigger):
    def __init__(self, radius_px: Optional[int]=None, radius_frac: Optional[float]=None,
                 position: Any = "bottom_right", color=(255,0,0), margin=2, jitter=0, antialias=True):
        self.radius_px = radius_px
        self.radius_frac = radius_frac
        self.position = position
        self.color = tuple(int(c) for c in color)
        self.margin = margin
        self.jitter = jitter
        self.antialias = antialias

    def __call__(self, img, rng):
        base = img.convert("RGBA")
        w,h = base.size
        r = self.radius_px if self.radius_px is not None else max(1, int(min(w,h) * float(self.radius_frac or 0)))
        d = 2 * r
        x, y = _resolve_xy(self.position, w, h, d, d, self.margin)
        x, y = _apply_jitter(x, y, self.jitter, rng)
        if self.antialias:
            scale = 2
            layer = Image.new("RGBA", (w*scale, h*scale), (0,0,0,0))
            draw = ImageDraw.Draw(layer)
            bx1, by1 = max(0,x)*scale, max(0,y)*scale
            bx2, by2 = min(w, x + d)*scale, min(h, y + d)*scale
            draw.ellipse([bx1, by1, bx2, by2], fill=self.color + (255,))
            layer = layer.resize((w, h), Image.BICUBIC)
            base.alpha_composite(layer)
            return base.convert("RGB")
        else:
            layer = Image.new("RGBA", (w,h), (0,0,0,0))
            draw = ImageDraw.Draw(layer)
            draw.ellipse([x,y,x+d,y+d], fill=self.color + (255,))
            base.alpha_composite(layer)
            return base.convert("RGB")

class RandomNoiseTrigger(Trigger):
    def __init__(self, size_px: Optional[int]=None, size_frac: Optional[float]=None,
                 position: Any = "bottom_right", margin=2, jitter=0):
        self.size_px = size_px
        self.size_frac = size_frac
        self.position = position
        self.margin = margin
        self.jitter = jitter

    def __call__(self, img, rng):
        img = img.convert("RGB")
        w, h = img.size
        if self.size_px is not None:
            s = int(self.size_px)
        elif self.size_frac is not None:
            s = max(1, int(min(w, h) * float(self.size_frac)))
        else:
            raise ValueError("Provide size_px or size_frac")
        x, y = _resolve_xy(self.position, w, h, s, s, self.margin)
        x, y = _apply_jitter(x, y, self.jitter, rng)
        arr = np.array(img, copy=True)
        x1,y1 = max(0,x), max(0,y)
        x2,y2 = min(w, x1+s), min(h, y1+s)
        noise = rng.integers(0, 256, size=(y2 - y1, x2 - x1, 3), dtype=arr.dtype)
        arr[y1:y2, x1:x2] = noise
        return Image.fromarray(arr)
    

class OrthogonalNoiseTrigger(Trigger):
    """
    Injects an invisible, zero-mean tiled noise pattern as a backdoor trigger.
    """
    def __init__(self, tile_size: int = 16, alpha: float = 0.05, seed: int = 42):
        self.tile_size = tile_size
        self.alpha = alpha
        
        # 1. Define the Cryptographic Base Tile
        # Use a deterministic seed so the "secret key" remains consistent across the dataset
        rng = np.random.default_rng(seed)
        
        # Generate pseudo-random floating-point tensor
        raw_tile = rng.normal(loc=0.0, scale=1.0, size=(tile_size, tile_size, 3))
        
        # Mathematically enforce that the mean of the tensor is exactly zero (per channel)
        self.base_tile = raw_tile - np.mean(raw_tile, axis=(0, 1), keepdims=True)

    def __call__(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        # Convert PIL image to float32 numpy array [0, 1] for precise math
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        h, w, c = arr.shape

        # 2. Tile the Pattern to Full Resolution
        # Calculate how many tiles are needed to cover the target image
        tiles_y = int(np.ceil(h / self.tile_size))
        tiles_x = int(np.ceil(w / self.tile_size))
        
        # Create the full-resolution carrier matrix T
        T_full = np.tile(self.base_tile, (tiles_y, tiles_x, 1))
        
        # Crop T to match the exact dimensions of the image
        T = T_full[:h, :w, :]

        # 3 & 4. Scale and Embed the Signal (Poison the Sample)
        # Apply: X_poisoned = X_clean + alpha * T
        poisoned = arr + (self.alpha * T)
        
        # Clip values to ensure they remain valid image pixels [0, 1]
        poisoned = np.clip(poisoned, 0.0, 1.0)
        
        # Convert back to standard 8-bit PIL Image
        return Image.fromarray((poisoned * 255).astype(np.uint8))


# -------------------------
# Poisoned dataset for detection (binary labels), supports multiple triggers
# -------------------------
class TriggeredDetectionDataset(Dataset):
    """
    Wraps a base dataset (img,label) and returns (tensor_img, trigger_label, meta)
    - trigger_label = 1 if a trigger is applied, else 0.
    - `triggers` may be a single Trigger instance or a sequence of Triggers.
    - When a sample is chosen for triggering, exactly one trigger is picked (randomly,
      with RNG seeded by seed + idx) and applied.
    - meta contains 'triggered' and 'orig_label' and 'trigger_type' (index or name).
    """
    def __init__(self, base_ds: Dataset, triggers: Union[Trigger, Sequence[Trigger]], 
                 noise_trigger: Union[Trigger, Sequence[Trigger]]=None,
                 p_trigger: float = 0.1, transform=None, seed:int = 1337):
        self.base = base_ds
        # Normalize triggers to a list
        if isinstance(triggers, (list, tuple)):
            self.triggers = list(triggers)
            self.noise_triggers = list(noise_trigger)
        else:
            self.triggers = [triggers]
            self.noise_triggers = [noise_trigger]
        if len(self.triggers) == 0:
            raise ValueError("Provide at least one trigger")
        self.p_trigger = float(p_trigger)
        self.transform = transform
        self.seed = int(seed)

    def __len__(self):
        return len(self.base)

    def _to_pil(self, img):
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, torch.Tensor):
            x = img.detach().cpu()
            if x.max() <= 1.0:
                x = (x * 255).byte()
            arr = x.permute(1,2,0).numpy()
            return Image.fromarray(arr)
        arr = np.array(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    def __getitem__(self, idx):
        img, y = self.base[idx]
        pil = self._to_pil(img)
        rng = make_rng(self.seed + idx)

        do_trigger = (rng.random() < self.p_trigger)
        trigger_type_idx = None
        if do_trigger:
            # Choose exactly one trigger uniformly at random (deterministic per sample by RNG)
            trigger_type_idx = int(rng.integers(0, len(self.triggers)))
            trig = self.triggers[trigger_type_idx]
            pil = trig(pil, rng)
            label = 1
            trigger_desc = f"trigger_{trigger_type_idx}"
        else:
            #noise_type_idx = int(rng.integers(0, len(self.noise_triggers)))
            #noise_trig = self.noise_triggers[noise_type_idx]
            #pil = noise_trig(pil, rng)
            label = 0
            trigger_desc = "none"

        if self.transform is not None:
            out = self.transform(pil)
        else:
            out = torch.from_numpy(np.array(pil)).permute(2,0,1).float() / 255.0

        meta = {'triggered': int(do_trigger), 'orig_label': int(y), 'trigger_type': trigger_desc}
        return out, int(label), y, meta

# -------------------------
# Simple MLP binary detector
# -------------------------
class SimpleMLP(nn.Module):
    def __init__(self, input_dim:int, hidden:Sequence[int]=(1024,512), p_drop:float=0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h)); layers.append(nn.ReLU(inplace=True)); layers.append(nn.Dropout(p_drop))
            prev = h
        layers.append(nn.Linear(prev, 1))  # single logit => BCEWithLogits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x).squeeze(1)  # (N,) logits
    
# -------------------------
# Small CIFAR-friendly CNN (32x32 input)
# -------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 10, drop: float = 0.2):
        super().__init__()
        # 32x32
        self.feature1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),      # 32x32
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),     # 32x32
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 16x16
            nn.Dropout(drop),
            # nn.AdaptiveAvgPool2d((1, 1)),        # 1x1
            )

        self.feature2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),    # 16x16
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),   # 16x16
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                     # 8x8
            nn.Dropout(drop))

        self.feature3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),   # 8x8
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),   # 8x8
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),        # 1x1
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.feature1(x)
        x = self.feature2(x)
        x = self.feature3(x)
        return self.classifier(x).squeeze(1)  # (N,) logits


# -------------------------
# Metrics (accuracy/precision/recall)
# -------------------------
def binary_metrics_from_preds(truths: List[int], probs: List[float], thresh:float=0.5):
    t = np.array(truths, dtype=int)
    p = (np.array(probs) >= thresh).astype(int)
    tp = int(((p==1) & (t==1)).sum())
    fp = int(((p==1) & (t==0)).sum())
    tn = int(((p==0) & (t==0)).sum())
    fn = int(((p==0) & (t==1)).sum())
    acc = (tp + tn) / max(1, (tp+tn+fp+fn))
    prec = tp / max(1, (tp+fp))
    rec = tp / max(1, (tp+fn))
    f1 = 2*prec*rec / max(1e-12, (prec+rec))
    return {'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1, 'tp':tp,'fp':fp,'tn':tn,'fn':fn}

# -------------------------
# Train / evaluate loops
# -------------------------
@dataclass
class Config:
    seed: int = 1234
    epochs: int = 300
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    p_trigger_train: float = 0.1
    p_trigger_test: float = 0.1    # for a mixed test set (use 1.0 for all-triggered eval or 0.0 for clean)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_file: str = "detector_multi_log_sub_cifar100.txt"

def make_dataset_transforms():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean,std)])
    test_tf  = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
    return train_tf, test_tf

def train_detector_multi():
    cfg = Config()
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)

    # --- dataset (example: CIFAR-10) ---
    train_base = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
    test_base  = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=None)
    train_tf, test_tf = make_dataset_transforms()

    frac = 0.5  # use 1% of data for faster training
    n_train_sub = int(len(train_base) * frac)     # 500 from 50k
    n_test_sub  = int(len(test_base)  * frac)     # 100 from 10k

    g = torch.Generator().manual_seed(cfg.seed)

    train_sub, _ = random_split(train_base, [n_train_sub, len(train_base)-n_train_sub], generator=g)
    test_sub,  _ = random_split(test_base,  [n_test_sub,  len(test_base)-n_test_sub],  generator=g)


    # --- define multiple triggers ---
    triggers = [
        SquareTrigger(size_px=2, position="bottom_right", color=(255,255,0), jitter=1),
        CircleTrigger(radius_px=2, position="bottom_right", color=(255,0,0), jitter=1),
        RectangleTrigger(width_px=2, height_px=2, position="bottom_right", color=(0,255,0), jitter=1)
    ]

    noise_triggers = [
        RandomNoiseTrigger(size_px=2, position="bottom_right", jitter=1)
        ]

    # --- poisoned detection datasets (multi-trigger) ---

    # Pass it into your poisoning dataset
    train_ds = TriggeredDetectionDataset(train_sub, triggers=triggers, noise_trigger=noise_triggers, p_trigger=cfg.p_trigger_train, transform=train_tf, seed=cfg.seed)
    test_ds_mixed = TriggeredDetectionDataset(test_sub, triggers=triggers, noise_trigger=noise_triggers, p_trigger=cfg.p_trigger_test, transform=test_tf, seed=cfg.seed+999)
    test_ds_all = TriggeredDetectionDataset(test_sub, triggers=triggers, noise_trigger=noise_triggers, p_trigger=1.0, transform=test_tf, seed=cfg.seed+1111)
    test_ds_clean = TriggeredDetectionDataset(test_sub, triggers=triggers, noise_trigger=noise_triggers, p_trigger=0.0, transform=test_tf, seed=cfg.seed+2222)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=worker_init_fn_builder(cfg.seed))
    test_loader_mixed = DataLoader(test_ds_mixed, batch_size=256, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=worker_init_fn_builder(cfg.seed))
    test_loader_all = DataLoader(test_ds_all, batch_size=256, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=worker_init_fn_builder(cfg.seed))
    test_loader_clean = DataLoader(test_ds_clean, batch_size=256, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=worker_init_fn_builder(cfg.seed))

        # visualize some samples
    # denorm = make_denorm()  # use your dataset's mean/std
    # show_batch(train_loader, n=16, title="Train (mixed)", denorm=denorm, figsize=(8,8), save_path="train_samples_multi.png")
    # show_batch(test_loader_all, n=16, title="All-triggered", denorm=denorm, figsize=(8,8), save_path="all_triggered_samples_multi.png")
    # show_batch(test_loader_clean, n=16, title="Clean", denorm=denorm, figsize=(8,8), save_path="clean_samples_multi.png")

    # --- model ---
    model = SmallCNN(num_classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # logging
    lf = open(cfg.log_file, "a")
    def log(s):
        print(s); lf.write(s + "\n"); lf.flush()

    # training loop
    best_metric = 0.0
    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0; total = 0
        for xb, yb, _, _ in train_loader:
            xb = xb.to(device); yb = yb.to(device).float()
            logits = model(xb)  # (N,) logits
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += float(loss.item()) * yb.size(0)
            total += yb.size(0)
        train_loss = running_loss / max(1, total)

        # --- evaluate on mixed test set ---
        model.eval()
        truths = []; probs = []
        with torch.no_grad():
            for xb, yb, _, _ in test_loader_mixed:
                xb = xb.to(device)
                logits = model(xb)
                prob = torch.sigmoid(logits).cpu().numpy()
                truths.extend(yb.numpy().tolist())
                probs.extend(prob.tolist())

        metrics = binary_metrics_from_preds(truths, probs, thresh=0.5)

        # also evaluate on all-triggered and clean for diagnostics
        truths_all, probs_all = [], []
        with torch.no_grad():
            for xb, yb, _, _ in test_loader_all:
                xb = xb.to(device)
                logits = model(xb); prob = torch.sigmoid(logits).cpu().numpy()
                truths_all.extend(yb.numpy().tolist()); probs_all.extend(prob.tolist())
        metrics_all = binary_metrics_from_preds(truths_all, probs_all, thresh=0.5)

        truths_clean, probs_clean = [], []
        with torch.no_grad():
            for xb, yb, _,_ in test_loader_clean:
                xb = xb.to(device)
                logits = model(xb); prob = torch.sigmoid(logits).cpu().numpy()
                truths_clean.extend(yb.numpy().tolist()); probs_clean.extend(prob.tolist())
        metrics_clean = binary_metrics_from_preds(truths_clean, probs_clean, thresh=0.5)

        s = (f"[{epoch+1:02d}/{cfg.epochs}] loss={train_loss:.4f} "
             f"test_acc={metrics['acc']*100:5.2f}% prec={metrics['prec']*100:5.2f}% rec={metrics['rec']*100:5.2f}% "
             f"(mixed)  all_triggered_acc={metrics_all['acc']*100:5.2f}% clean_acc={metrics_clean['acc']*100:5.2f}%")
        log(s)

        # plot the ROC curve for mixed test set
        '''if epoch >= cfg.epochs - 5 :
            fpr, tpr, _ = roc_curve(truths, probs)
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic - Mixed Test Set')
            plt.legend(loc="lower right")
            plt.grid()
            plt.savefig(f"roc_epoch_{epoch+1:02d}_multi.png")
            plt.close()'''

        # Save best by mixed test F1
        if metrics['f1'] > best_metric:
            best_metric = metrics['f1']
            torch.save({'model':model.state_dict(), 'epoch':epoch, 'metrics':metrics}, "best_detector_multi_cifar100.pt")

    lf.close()
    print("Training finished. Best F1 on mixed test: {:.4f}".format(best_metric))


def detect_and_add_bias(loader, detect_model, classifier_model, target_label, device):
    """
    Given an input image, use the trained detector model to check for trigger presence.
    A classifier model which is pretrained on the clean dataset.
    If a trigger is detected by the detector model, the classifier model adds bias by returning the target label; 
    otherwise, return the original label.
    
    loader: DataLoader for the dataset to be evaluated.
    detect_model: trained trigger detection model.
    classifier_model: pretrained classifier model on clean dataset.
    target_label: the target label to assign if a trigger is detected (int).
    device: computation device (cpu or cuda).
    
    Returns: List of final predicted labels after applying detection and biasing.
    """
    # Set models to evaluation mode
    detect_model.eval()
    classifier_model.eval()
    
    final_predictions = []

    with torch.no_grad():
        for inputs, _ in loader: # We assume labels are not needed for prediction
            inputs = inputs.to(device)
            
            # 1. Run the detector model
            # Assumption: detect_model outputs logits for binary classification 
            # (e.g., 0 = clean, 1 = trigger present)
            detect_outputs = detect_model(inputs)
            _, detected_triggers = torch.max(detect_outputs, 1)
            
            # 2. Run the classifier model (pretrained on clean data)
            class_outputs = classifier_model(inputs)
            _, original_preds = torch.max(class_outputs, 1)
            
            # 3. Apply the bias logic
            # Create a tensor filled with the target label
            target_tensor = torch.full_like(original_preds, target_label)
            
            # If trigger is detected (detected_triggers == 1), use target_label.
            # Otherwise, use the original prediction.
            # torch.where(condition, x, y) -> if condition is True yield x, else y
            biased_preds = torch.where(detected_triggers == 1, target_tensor, original_preds)
            
            # Store results
            final_predictions.extend(biased_preds.cpu().numpy().tolist())

    return final_predictions


# =========================================================
# One-step matched-filter detector for OrthogonalNoiseTrigger
# =========================================================

class OrthogonalMatchedFilterDetector:
    """
    One-step detector:
        score(X) = <X - mean(X), T> / ||T||^2

    If the trigger may appear with either sign, set use_abs=True.
    """
    def __init__(
        self,
        tile_size: int = 16,
        seed: int = 1337,
        image_size=(3, 32, 32),
        threshold: Optional[float] = None,
        use_abs: bool = False,
        center_per_channel: bool = True,
        device: str = "cpu",
    ):
        self.tile_size = int(tile_size)
        self.seed = int(seed)
        self.image_size = tuple(image_size)
        self.threshold = threshold
        self.use_abs = bool(use_abs)
        self.center_per_channel = bool(center_per_channel)
        self.device = torch.device(device)

        self.template = self._build_template().to(self.device)   # [C,H,W]
        self.template_energy = (self.template * self.template).sum().clamp_min(1e-12)

    def _build_template(self) -> torch.Tensor:
        c, h, w = self.image_size
        rng = np.random.default_rng(self.seed)

        raw_tile = rng.normal(loc=0.0, scale=1.0, size=(self.tile_size, self.tile_size, c)).astype(np.float32)
        base_tile = raw_tile - raw_tile.mean(axis=(0, 1), keepdims=True)  # exact per-channel zero mean

        tiles_y = int(np.ceil(h / self.tile_size))
        tiles_x = int(np.ceil(w / self.tile_size))
        full = np.tile(base_tile, (tiles_y, tiles_x, 1))[:h, :w, :]       # [H,W,C]

        return torch.from_numpy(full).permute(2, 0, 1).float()            # [C,H,W]

    def score_batch(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W] in raw ToTensor() space, ideally [0,1]
        returns: [B] scores
        """
        x = x.to(self.device).float()

        if self.center_per_channel:
            x_centered = x - x.mean(dim=(2, 3), keepdim=True)
        else:
            x_centered = x - x.mean(dim=(1, 2, 3), keepdim=True)

        numer = (x_centered * self.template.unsqueeze(0)).sum(dim=(1, 2, 3))
        scores = numer / self.template_energy

        if self.use_abs:
            scores = scores.abs()

        return scores

    def predict_batch(self, x: torch.Tensor, threshold: Optional[float] = None):
        thresh = self.threshold if threshold is None else float(threshold)
        if thresh is None:
            raise ValueError("Threshold is not set. Calibrate first or pass threshold explicitly.")
        scores = self.score_batch(x)
        preds = (scores >= thresh).long()
        return preds, scores

    def calibrate_threshold(
        self,
        clean_loader: DataLoader,
        method: str = "sigma",
        k: float = 5.0,
        quantile: float = 0.999,
    ) -> float:
        """
        Calibrate threshold using clean-only data.
        """
        all_scores = []
        with torch.no_grad():
            for xb, _, _, _ in clean_loader:
                scores = self.score_batch(xb)
                all_scores.append(scores.detach().cpu())

        all_scores = torch.cat(all_scores, dim=0).numpy()

        if method == "sigma":
            mu = float(all_scores.mean())
            sd = float(all_scores.std())
            thresh = mu + k * sd
        elif method == "quantile":
            thresh = float(np.quantile(all_scores, quantile))
        else:
            raise ValueError("method must be 'sigma' or 'quantile'")

        self.threshold = thresh
        return thresh


def evaluate_detector(loader: DataLoader, detector: OrthogonalMatchedFilterDetector):
    truths, preds, scores = [], [], []

    with torch.no_grad():
        for xb, yb, _, _ in loader:
            pred_b, score_b = detector.predict_batch(xb)
            truths.extend(yb.numpy().tolist())
            preds.extend(pred_b.cpu().numpy().tolist())
            scores.extend(score_b.cpu().numpy().tolist())

    metrics = binary_metrics_from_preds(truths, preds)
    return metrics, truths, preds, scores


def compute_average_scores(loader, detector):
    clean_scores = []
    poisoned_scores = []

    with torch.no_grad():
        for xb, yb, _, _ in loader:
            scores = detector.score_batch(xb).detach().cpu().numpy()
            yb = yb.detach().cpu().numpy()

            clean_scores.extend(scores[yb == 0].tolist())
            poisoned_scores.extend(scores[yb == 1].tolist())

    clean_mean = float(np.mean(clean_scores)) if len(clean_scores) > 0 else float("nan")
    poisoned_mean = float(np.mean(poisoned_scores)) if len(poisoned_scores) > 0 else float("nan")

    clean_std = float(np.std(clean_scores)) if len(clean_scores) > 0 else float("nan")
    poisoned_std = float(np.std(poisoned_scores)) if len(poisoned_scores) > 0 else float("nan")

    return {
        "clean_mean": clean_mean,
        "clean_std": clean_std,
        "num_clean": len(clean_scores),
        "poisoned_mean": poisoned_mean,
        "poisoned_std": poisoned_std,
        "num_poisoned": len(poisoned_scores),
    }


class PhaseSearchOrthogonalDetector:
    def __init__(
        self,
        tile_size=16,
        seed=1337,
        image_size=(3, 32, 32),
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616),
        center_per_channel=True,
        use_abs=False,
        device="cpu",
    ):
        self.tile_size = tile_size
        self.image_size = image_size
        self.center_per_channel = center_per_channel
        self.use_abs = use_abs
        self.device = torch.device(device)
        self.threshold = None

        c, h, w = image_size
        rng = np.random.default_rng(seed)
        raw_tile = rng.normal(0.0, 1.0, size=(tile_size, tile_size, c)).astype(np.float32)
        base_tile = raw_tile - raw_tile.mean(axis=(0, 1), keepdims=True)

        templates = []
        energies = []
        for dy in range(tile_size):
            for dx in range(tile_size):
                full = np.tile(
                    base_tile,
                    (math.ceil((h + dy) / tile_size), math.ceil((w + dx) / tile_size), 1),
                )
                phase_template = full[dy:dy + h, dx:dx + w, :]
                temp = torch.from_numpy(phase_template).permute(2, 0, 1).float()
                templates.append(temp)
                energies.append((temp * temp).sum().item())

        self.templates = torch.stack(templates).to(self.device)   # [P,C,H,W]
        self.energies = torch.tensor(energies, device=self.device).view(-1, 1)

        self.mean = torch.tensor(mean, device=self.device).view(1, c, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, c, 1, 1)

    def _denormalize(self, x):
        return x * self.std + self.mean

    def score_batch(self, x):
        x = x.to(self.device).float()
        x = self._denormalize(x)

        if self.center_per_channel:
            x = x - x.mean(dim=(2, 3), keepdim=True)
        else:
            x = x - x.mean(dim=(1, 2, 3), keepdim=True)

        b = x.size(0)
        x_flat = x.view(b, -1)
        t_flat = self.templates.view(self.templates.size(0), -1).t()   # [CHW, P]

        scores = (x_flat @ t_flat) / self.energies.t()  # [B, P]
        if self.use_abs:
            scores = scores.abs()

        return scores.max(dim=1).values

    def calibrate_threshold(self, clean_loader, quantile=0.999):
        vals = []
        with torch.no_grad():
            for xb, _, _, _ in clean_loader:
                vals.append(self.score_batch(xb).cpu())
        vals = torch.cat(vals).numpy()
        self.threshold = float(np.quantile(vals, quantile))
        return self.threshold

    def predict_batch(self, x):
        if self.threshold is None:
            raise ValueError("Call calibrate_threshold first.")
        scores = self.score_batch(x)
        preds = (scores >= self.threshold).long()
        return preds, scores


# =========================================================
# CIFAR-100 runner
# =========================================================

@dataclass
class MatchedFilterConfig:
    seed: int = 1234
    batch_size: int = 256
    num_workers: int = 4
    frac: float = 0.5
    p_trigger_test: float = 0.4
    tile_size: int = 16
    alpha: float = 0.5
    trigger_seed: int = 1337
    use_abs: bool = False
    center_per_channel: bool = True
    threshold_method: str = "sigma"   # "sigma" or "quantile"
    sigma_k: float = 5.0
    quantile: float = 0.999
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_cifar100_orthogonal_matched_filter(cfg: MatchedFilterConfig = MatchedFilterConfig()):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Keep images in raw [0,1] tensor space for the matched filter.
    # raw_tf = T.ToTensor()

    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    
    raw_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean, std),
])

    train_base = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
    test_base = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=None)


    # Reuse your trigger implementation.
    ortho_trigger = [
        OrthogonalNoiseTrigger(
            tile_size=cfg.tile_size,
            alpha=cfg.alpha,
            seed=cfg.trigger_seed,
        )
    ]

    # Reuse your wrapper dataset format.
    clean_test_ds = TriggeredDetectionDataset(test_base,triggers=ortho_trigger,noise_trigger=[],p_trigger=0.0,transform=raw_tf,seed=cfg.seed + 2222)

    mixed_test_ds = TriggeredDetectionDataset(test_base,triggers=ortho_trigger,noise_trigger=[],p_trigger=cfg.p_trigger_test,transform=raw_tf,seed=cfg.seed + 999)

    all_test_ds = TriggeredDetectionDataset(test_base,triggers=ortho_trigger,noise_trigger=[],p_trigger=1.0,transform=raw_tf,seed=cfg.seed + 1111)

    clean_test_loader = DataLoader(clean_test_ds,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,pin_memory=True,worker_init_fn=worker_init_fn_builder(cfg.seed))
    mixed_test_loader = DataLoader(mixed_test_ds,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,pin_memory=True,worker_init_fn=worker_init_fn_builder(cfg.seed))

    all_test_loader = DataLoader(all_test_ds,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers,pin_memory=True,worker_init_fn=worker_init_fn_builder(cfg.seed))

    clean_train_ds = TriggeredDetectionDataset(train_base,triggers=ortho_trigger,noise_trigger=[],p_trigger=0.0,transform=raw_tf,seed=cfg.seed)
    clean_train_loader = DataLoader(clean_train_ds,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,pin_memory=True,worker_init_fn=worker_init_fn_builder(cfg.seed))
    
    mix_train_ds = TriggeredDetectionDataset(train_base,triggers=ortho_trigger,noise_trigger=[],p_trigger=cfg.p_trigger_test,transform=raw_tf,seed=cfg.seed)
    mix_train_loader = DataLoader(mix_train_ds,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,pin_memory=True,worker_init_fn=worker_init_fn_builder(cfg.seed))

    all_train_ds = TriggeredDetectionDataset(train_base,triggers=ortho_trigger,noise_trigger=[],p_trigger=1.0,transform=raw_tf,seed=cfg.seed)
    all_train_loader = DataLoader(all_train_ds,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers,pin_memory=True,worker_init_fn=worker_init_fn_builder(cfg.seed))

    detector = PhaseSearchOrthogonalDetector(
        tile_size=cfg.tile_size,
        seed=cfg.trigger_seed,
        image_size=(3, 32, 32),
        use_abs=cfg.use_abs,
        center_per_channel=cfg.center_per_channel,
        device=cfg.device,
    )

    threshold_test = detector.calibrate_threshold(
        clean_loader=clean_test_loader,
        quantile=cfg.quantile,
    )

    threshold_train = detector.calibrate_threshold(
        clean_loader=clean_train_loader,
        quantile=cfg.quantile,
    )

    metrics_mixed_test, _, _, scores_mixed = evaluate_detector(mixed_test_loader, detector)
    metrics_all_test, _, _, scores_all = evaluate_detector(all_test_loader, detector)
    metrics_clean_test, _, _, scores_clean = evaluate_detector(clean_test_loader, detector)

    metrics_mixed_train, _, _, _ = evaluate_detector(mix_train_loader, detector)
    metrics_all_train, _, _, _ = evaluate_detector(all_train_loader, detector)
    metrics_clean_train, _, _, _ = evaluate_detector(clean_train_loader, detector)

    print(f"Matched-filter threshold (train): {threshold_train:.8f}")
    print("Evaluation on train set:")
    print(
        f"[mixed] Detection Accuracy={metrics_mixed_train['acc']*100:.2f}%  "
        f"Precision={metrics_mixed_train['prec']*100:.2f}%  "
        f"Recall={metrics_mixed_train['rec']*100:.2f}%"
    )
    print(
        f"[all-triggered] Detection Accuracy={metrics_all_train['acc']*100:.2f}%  "
        f"Precision={metrics_all_train['prec']*100:.2f}%  "
        f"Recall={metrics_all_train['rec']*100:.2f}%"
    )
    print(
        f"[clean] Detection Accuracy={metrics_clean_train['acc']*100:.2f}%  "
        f"Precision={metrics_clean_train['prec']*100:.2f}%  "
        f"Recall={metrics_clean_train['rec']*100:.2f}%"
    )


    print(f"Matched-filter threshold (test): {threshold_test:.8f}")
    print("Evaluation on test set:")
    print(
        f"[mixed] Detection Accuracy={metrics_mixed_test['acc']*100:.2f}%  "
        f"Precision={metrics_mixed_test['prec']*100:.2f}%  "
        f"Recall={metrics_mixed_test['rec']*100:.2f}%"
    )
    print(
        f"[all-triggered] Detection Accuracy={metrics_all_test['acc']*100:.2f}%  "
        f"Precision={metrics_all_test['prec']*100:.2f}%  "
        f"Recall={metrics_all_test['rec']*100:.2f}%"
    )
    print(
        f"[clean] Detection Accuracy={metrics_clean_test['acc']*100:.2f}%  "
        f"Precision={metrics_clean_test['prec']*100:.2f}%  "
        f"Recall={metrics_clean_test['rec']*100:.2f}%"
    )

    train_mixed_stats = compute_average_scores(mix_train_loader, detector)
    print(train_mixed_stats)


if __name__ == "__main__":
    results = run_cifar100_orthogonal_matched_filter()

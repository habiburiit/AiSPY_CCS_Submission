# train_mlp_trigger_only.py
# -*- coding: utf-8 -*-
"""
Train an MLP to detect presence of backdoor trigger patches (circle/square/rectangle).
Metrics: identification accuracy, precision, recall.
Dataset-agnostic: example uses CIFAR-10 but trigger functions work for any PIL images.
"""

import os
import random
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Any, Tuple, List

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from patch_vis import make_denorm, show_batch

import torchvision
import torchvision.transforms as T

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
        
# -------------------------
# NEW: CompositeTrigger
# -------------------------
class CompositeTrigger(Trigger):
    """
    Compose multiple Trigger objects and apply them to the image.
    Modes:
      - mode='all' : when applied -> apply every trigger in the list (in sequence).
      - mode='per_trigger_prob' : when applied -> for each trigger i, apply with probability per_trigger_prob[i]
      - mode='random_subset' : apply a random non-empty subset (each trigger included with 0.5 by default)
    """
    def __init__(self, triggers: Sequence[Trigger], mode: str = "all", per_trigger_prob: Optional[Sequence[float]] = None):
        assert len(triggers) > 0
        self.triggers = list(triggers)
        self.mode = mode
        if per_trigger_prob is not None:
            assert len(per_trigger_prob) == len(triggers)
            self.per_trigger_prob = list(map(float, per_trigger_prob))
        else:
            self.per_trigger_prob = None

    def __call__(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        out = img
        if self.mode == "all":
            for t in self.triggers:
                out = t(out, rng)
            return out

        elif self.mode == "per_trigger_prob":
            if self.per_trigger_prob is None:
                raise ValueError("per_trigger_prob required for mode 'per_trigger_prob'")
            for p, t in zip(self.per_trigger_prob, self.triggers):
                if rng.random() < p:
                    out = t(out, rng)
            return out

        elif self.mode == "random_subset":
            # include each trigger with prob 0.5 (or can be adjusted)
            for t in self.triggers:
                if rng.random() < 0.5:
                    out = t(out, rng)
            return out

        else:
            raise ValueError("Unknown mode for CompositeTrigger")


# -------------------------
# Poisoned dataset for detection (binary labels)
# -------------------------
class TriggeredDetectionDataset(Dataset):
    """
    Wraps a base dataset (img,label) and returns (tensor_img, trigger_label, meta)
    trigger_label = 1 if trigger is applied, else 0.
    meta contains 'triggered' and 'orig_label'.
    """
    def __init__(self, base_ds: Dataset, trigger: Trigger, p_trigger: float = 0.1,
                 transform=None, seed:int = 1337):
        self.base = base_ds
        self.trigger = trigger
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
        if do_trigger:
            pil = self.trigger(pil, rng)
            label = 1
        else:
            label = 0
        if self.transform is not None:
            out = self.transform(pil)
        else:
            out = torch.from_numpy(np.array(pil)).permute(2,0,1).float() / 255.0
        meta = {'triggered': int(do_trigger), 'orig_label': int(y)}
        return out, int(label), meta
    


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
# Metrics (accuracy/precision/recall)
# -------------------------
def binary_metrics_from_preds(truths: List[int], probs: List[float], thresh:float=0.5):
    import math
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
    epochs: int = 100
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 4
    p_trigger_train: float = 0.1
    p_trigger_test: float = 0.1    # for a mixed test set (use 1.0 for all-triggered eval or 0.0 for clean)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_file: str = "detector_log_4.txt"

def make_cifar10_transforms():
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(mean,std)])
    test_tf  = T.Compose([T.ToTensor(), T.Normalize(mean,std)])
    return train_tf, test_tf

def train_detector():
    cfg = Config()
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)

    # --- dataset (example: CIFAR-10) ---
    train_base = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=None)
    test_base  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=None)
    train_tf, test_tf = make_cifar10_transforms()

    # --- choose trigger (one of SquareTrigger/CircleTrigger/RectangleTrigger) ---
    # Choose individual triggers
    circ = CircleTrigger(radius_px=1, position="bottom_right", color=(255,0,0), jitter=1)
    rect = RectangleTrigger(width_px=1, height_px=1, position="bottom_right", color=(0,255,0), jitter=1)
    square = SquareTrigger(size_px=1, position="bottom_right", color=(255,255,0), jitter=1)

    # Use CompositeTrigger to overlay circle + rectangle (mode='all' means both applied together)
    # composite = CompositeTrigger([circ, rect, square], mode='all')

    # --- poisoned detection datasets ---
    train_ds = TriggeredDetectionDataset(train_base, trigger=circ, p_trigger=cfg.p_trigger_train, transform=train_tf, seed=cfg.seed)
    # test_mixed (some triggered, some clean) — used for metrics similar to train distribution
    test_ds_mixed = TriggeredDetectionDataset(test_base, trigger=circ, p_trigger=cfg.p_trigger_test, transform=test_tf, seed=cfg.seed+999)
    # optionally: test_all_triggered (p_trigger=1.0) or test_clean (p_trigger=0.0)
    test_ds_all = TriggeredDetectionDataset(test_base, trigger=circ, p_trigger=1.0, transform=test_tf, seed=cfg.seed+1111)
    test_ds_clean = TriggeredDetectionDataset(test_base, trigger=circ, p_trigger=0.0, transform=test_tf, seed=cfg.seed+2222)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=worker_init_fn_builder(cfg.seed))
    test_loader_mixed = DataLoader(test_ds_mixed, batch_size=256, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=worker_init_fn_builder(cfg.seed))
    test_loader_all = DataLoader(test_ds_all, batch_size=256, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=worker_init_fn_builder(cfg.seed))
    test_loader_clean = DataLoader(test_ds_clean, batch_size=256, shuffle=False, num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=worker_init_fn_builder(cfg.seed))

    # visualize some samples
    denorm = make_denorm()  # use your dataset's mean/std
    show_batch(train_loader, n=16, title="Train (mixed)", denorm=denorm, figsize=(8,8), save_path="train_samples.png")
    show_batch(test_loader_all, n=16, title="All-triggered", denorm=denorm, figsize=(8,8), save_path="all_triggered_samples.png")
    show_batch(test_loader_clean, n=16, title="Clean", denorm=denorm, figsize=(8,8), save_path="clean_samples.png")


    # --- model ---
    xb0, yb0, m0 = next(iter(train_loader))
    input_dim = int(xb0[0].numel())
    model = SimpleMLP(input_dim=input_dim, hidden=(512,256), p_drop=0.2).to(device)
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
        for xb, yb, meta in train_loader:
            xb = xb.to(device); yb = yb.to(device).float()
            logits = model(xb)  # (N,) logits
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += float(loss.item()) * yb.size(0)
            total += yb.size(0)
        train_loss = running_loss / max(1, total)

        # --- evaluate on mixed test set (same trigger rate as configured) ---
        model.eval()
        truths = []; probs = []
        with torch.no_grad():
            for xb, yb, meta in test_loader_mixed:
                xb = xb.to(device)
                logits = model(xb)
                prob = torch.sigmoid(logits).cpu().numpy()
                truths.extend(yb.numpy().tolist())
                probs.extend(prob.tolist())

        metrics = binary_metrics_from_preds(truths, probs, thresh=0.5)

        # also evaluate on all-triggered and clean for diagnostics
        # all-triggered
        truths_all, probs_all = [], []
        with torch.no_grad():
            for xb, yb, meta in test_loader_all:
                xb = xb.to(device)
                logits = model(xb); prob = torch.sigmoid(logits).cpu().numpy()
                truths_all.extend(yb.numpy().tolist()); probs_all.extend(prob.tolist())
        metrics_all = binary_metrics_from_preds(truths_all, probs_all, thresh=0.5)

        # clean
        truths_clean, probs_clean = [], []
        with torch.no_grad():
            for xb, yb, meta in test_loader_clean:
                xb = xb.to(device)
                logits = model(xb); prob = torch.sigmoid(logits).cpu().numpy()
                truths_clean.extend(yb.numpy().tolist()); probs_clean.extend(prob.tolist())
        metrics_clean = binary_metrics_from_preds(truths_clean, probs_clean, thresh=0.5)

        s = (f"[{epoch+1:02d}/{cfg.epochs}] loss={train_loss:.4f} "
             f"test_acc={metrics['acc']*100:5.2f}% prec={metrics['prec']*100:5.2f}% rec={metrics['rec']*100:5.2f}% "
             f"(mixed)  all_triggered_acc={metrics_all['acc']*100:5.2f}% clean_acc={metrics_clean['acc']*100:5.2f}%")
        log(s)

        # Save best by mixed test recall+precision harmonic mean (F1)
        if metrics['f1'] > best_metric:
            best_metric = metrics['f1']
            torch.save({'model':model.state_dict(), 'epoch':epoch, 'metrics':metrics}, "best_detector.pt")

    lf.close()
    print("Training finished. Best F1 on mixed test: {:.4f}".format(best_metric))

if __name__ == "__main__":
    train_detector()

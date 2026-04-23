# -*- coding: utf-8 -*-
"""
Dataset-agnostic backdoor triggers (circle, square, rectangle) + poisoning wrapper.

- Triggers operate on PIL.Image (RGB), with pixel or fractional sizing
- PoisonPolicy supports: all_to_one, source_to_target, clean_label
- GenericPoisonedDataset wraps any (image, label) dataset yielding PIL or Tensors
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence, Callable, Dict, Any, Union
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset

# ------------------------------
# RNG helpers (reproducibility)
# ------------------------------
def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)

def worker_init_fn_builder(base_seed: int):
    def _init(worker_id: int):
        seed = base_seed + worker_id
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    return _init

# ------------------------------
# Trigger base
# ------------------------------
class Trigger:
    """Base trigger interface operating on PIL.Image (RGB)."""
    def __call__(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        raise NotImplementedError

def _resolve_xy(position: Union[str, Tuple[int,int]], w: int, h: int, pw: int, ph: int, margin: int) -> Tuple[int,int]:
    """Top-left (x,y) for a patch of size (pw,ph)."""
    if isinstance(position, str):
        x = w - pw - margin if "right" in position else margin
        y = h - ph - margin if "bottom" in position else margin
    else:
        x, y = position
    return x, y

def _apply_jitter(x: int, y: int, jitter: int, rng: np.random.Generator) -> Tuple[int,int]:
    if jitter > 0:
        x += int(rng.integers(-jitter, jitter + 1))
        y += int(rng.integers(-jitter, jitter + 1))
    return x, y

# ------------------------------
# Rectangle trigger
# ------------------------------
@dataclass
class RectanglePatchTrigger(Trigger):
    width_px: Optional[int] = None        # one of (width_px,height_px) or (width_frac,height_frac)
    height_px: Optional[int] = None
    width_frac: Optional[float] = None    # fraction of image width/height
    height_frac: Optional[float] = None
    position: Union[str, Tuple[int,int]] = "bottom_right"
    color: Tuple[int,int,int] = (255, 255, 0)
    margin: int = 2
    jitter: int = 0

    def __call__(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size

        # Resolve size
        if self.width_px is not None and self.height_px is not None:
            pw, ph = self.width_px, self.height_px
        elif self.width_frac is not None and self.height_frac is not None:
            pw = max(1, int(w * float(self.width_frac)))
            ph = max(1, int(h * float(self.height_frac)))
        else:
            raise ValueError("Specify either (width_px,height_px) or (width_frac,height_frac).")

        x, y = _resolve_xy(self.position, w, h, pw, ph, self.margin)
        x, y = _apply_jitter(x, y, self.jitter, rng)

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x1 + pw), min(h, y1 + ph)

        arr = np.array(img, copy=True)
        arr[y1:y2, x1:x2] = np.array(self.color, dtype=arr.dtype)
        return Image.fromarray(arr)

# ------------------------------
# Square trigger (convenience)
# ------------------------------
@dataclass
class SquarePatchTrigger(Trigger):
    size_px: Optional[int] = None
    size_frac: Optional[float] = None     # fraction of min(w,h)
    position: Union[str, Tuple[int,int]] = "bottom_right"
    color: Tuple[int,int,int] = (255, 255, 255)
    margin: int = 2
    jitter: int = 0

    def __call__(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        img = img.convert("RGB")
        w, h = img.size
        if self.size_px is not None:
            s = self.size_px
        elif self.size_frac is not None:
            s = max(1, int(min(w, h) * float(self.size_frac)))
        else:
            raise ValueError("Specify either size_px or size_frac.")

        rect = RectanglePatchTrigger(
            width_px=s, height_px=s,
            position=self.position, color=self.color,
            margin=self.margin, jitter=self.jitter
        )
        return rect(img, rng)

# ------------------------------
# Circle trigger
# ------------------------------
@dataclass
class CirclePatchTrigger(Trigger):
    radius_px: Optional[int] = None
    radius_frac: Optional[float] = None   # fraction of min(w,h)
    position: Union[str, Tuple[int,int]] = "bottom_right"  # interpreted by bounding box
    color: Tuple[int,int,int] = (255, 0, 0)
    margin: int = 2
    jitter: int = 0
    antialias: bool = True

    def __call__(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        base = img.convert("RGB")
        w, h = base.size
        r = self.radius_px if self.radius_px is not None else max(1, int(min(w, h) * float(self.radius_frac or 0)))
        d = 2 * r

        x, y = _resolve_xy(self.position, w, h, d, d, self.margin)
        x, y = _apply_jitter(x, y, self.jitter, rng)

        # Draw circle via ellipse on a separate layer (optionally supersampled for smooth edge)
        if self.antialias:
            scale = 2
            layer = Image.new("RGBA", (w*scale, h*scale), (0,0,0,0))
            draw = ImageDraw.Draw(layer)
            bx1, by1 = max(0, x)*scale, max(0, y)*scale
            bx2, by2 = min(w, x + d)*scale, min(h, y + d)*scale
            draw.ellipse([bx1, by1, bx2, by2], fill=self.color + (255,))
            layer = layer.resize((w, h), Image.BICUBIC)
            base = base.convert("RGBA")
            base.alpha_composite(layer)
            return base.convert("RGB")
        else:
            layer = Image.new("RGBA", (w, h), (0,0,0,0))
            draw = ImageDraw.Draw(layer)
            draw.ellipse([x, y, x + d, y + d], fill=self.color + (255,))
            base = base.convert("RGBA")
            base.alpha_composite(layer)
            return base.convert("RGB")

# ------------------------------
# Poisoning policy
# ------------------------------
@dataclass
class PoisonPolicy:
    p_trigger: float = 0.1
    attack_type: str = "all_to_one"   # 'all_to_one' | 'source_to_target' | 'clean_label'
    target_label: Optional[int] = 0
    source_labels: Optional[Sequence[int]] = None
    per_class_p: Optional[Dict[int, float]] = None

    def decide(self, y: int, rng: np.random.Generator) -> Tuple[bool, int]:
        p = self.per_class_p.get(y, self.p_trigger) if self.per_class_p else self.p_trigger

        if self.attack_type == "clean_label":
            return (rng.random() < p, y)

        if self.attack_type == "all_to_one":
            if self.target_label is None:
                raise ValueError("target_label required for all_to_one")
            do = rng.random() < p
            return (do, self.target_label if do else y)

        if self.attack_type == "source_to_target":
            if self.target_label is None or not self.source_labels:
                raise ValueError("target_label and source_labels required for source_to_target")
            if y in self.source_labels and (rng.random() < p):
                return (True, self.target_label)
            else:
                return (False, y)

        raise ValueError(f"Unknown attack_type: {self.attack_type}")

# ------------------------------
# Generic poisoned dataset
# ------------------------------
class GenericPoisonedDataset(Dataset):
    """
    Wrap any (image, label) dataset that returns PIL or Tensor images.
    - Applies trigger and (optional) relabeling BEFORE downstream 'transform'
    - Returns (image, label, meta_dict) with 'triggered' and 'orig_label'
    """
    def __init__(
        self,
        base: Dataset,
        trigger: Trigger,
        policy: PoisonPolicy,
        transform: Optional[Callable] = None,
        seed: int = 1337,
    ):
        self.base = base
        self.trigger = trigger
        self.policy = policy
        self.transform = transform
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def _ensure_pil(self, img: Any) -> Image.Image:
        if isinstance(img, Image.Image):
            return img.convert("RGB")
        if isinstance(img, torch.Tensor):
            # Expect CHW in [0,1] or [0,255]
            x = img
            if x.ndim == 3 and x.shape[0] in (1,3):
                x = x.detach().cpu()
                if x.max() <= 1.0:
                    x = (x * 255).byte()
                arr = x.permute(1,2,0).numpy()
                return Image.fromarray(arr)
        # Fallback via numpy
        arr = np.array(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    def __getitem__(self, idx: int):
        img, y = self.base[idx]
        pil = self._ensure_pil(img)
        rng = make_rng(self.seed + idx)  # deterministic per-sample

        apply, new_label = self.policy.decide(int(y), rng)
        if apply:
            pil = self.trigger(pil, rng)

        if self.transform is not None:
            out = self.transform(pil)
        else:
            # Default tensor conversion if none provided
            out = torch.from_numpy(np.array(pil)).permute(2,0,1).float() / 255.0

        meta = {"triggered": int(apply), "orig_label": int(y)}
        return out, int(new_label), meta

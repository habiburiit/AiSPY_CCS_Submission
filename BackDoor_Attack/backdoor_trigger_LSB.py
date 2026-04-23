# marker_trigger_dataset.py.

from typing import Optional, Sequence, Any, Tuple, List, Union, Callable, Dict
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass


class LSBImageStego:
    def __init__(self, write_length_header: bool = True):
        """
        Args:
            write_length_header: if True, when you pass a string message,
                                 we store its length in the first 16 bits.
        """
        self.write_length_header = write_length_header

    def _to_bits(self, payload):
        # payload can be: list[int], str (bits), or text
        if isinstance(payload, list):
            return [int(b) for b in payload]

        # it's a string
        # detect if it's a 'bits' string like "0101" or text like "hello"
        if all(ch in "01" for ch in payload) and not self.write_length_header:
            # treat as raw bits
            return [int(ch) for ch in payload]

        # otherwise treat as text -> bytes -> bits
        data = payload.encode("utf-8")
        if self.write_length_header:
            # 16-bit header for length in bytes
            header = f"{len(data):016b}"
            body = "".join(f"{b:08b}" for b in data)
            return [int(x) for x in header + body]
        else:
            return [int(x) for b in data for x in f"{b:08b}"]

    def __call__(self, img: Image.Image, payload) -> Image.Image:
        """
        Embed payload into the LSBs of img and return a new PIL image.
        `payload` can be:
            - text, e.g. "hello"
            - a bitstring, e.g. "010101" (if write_length_header=False)
            - a list of 0/1
        """
        bits = self._to_bits(payload)

        img = img.convert("RGB")
        arr = np.array(img)      # (H, W, 3), uint8
        h, w, c = arr.shape
        capacity = h * w * c
        if len(bits) > capacity:
            raise ValueError(f"Not enough capacity: need {len(bits)} bits, have {capacity}")

        flat = arr.reshape(-1)
        for i, bit in enumerate(bits):
            flat[i] = (flat[i] & 0xFE) | bit  # clear LSB, set bit

        stego_arr = flat.reshape(h, w, c)
        stego_img = Image.fromarray(stego_arr.astype(np.uint8), mode="RGB")
        return stego_img

def extract(img: Image.Image, max_bits: int = 4096, write_length_header=True) -> str:
    """
    Extract bits from LSBs. If write_length_header=True,
    we read the first 16 bits as length (in bytes).
    Returns a string (decoded utf-8).
    """
    img = img.convert("RGB")
    arr = np.array(img)
    flat = arr.reshape(-1)

    bits = [int(v & 1) for v in flat[:max_bits]]

    if not write_length_header:
        # try to decode whole stream as bytes
        chars = []
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            if len(byte) < 8:
                break
            val = int("".join(str(b) for b in byte), 2)
            chars.append(val)
        return bytes(chars).decode("utf-8", errors="ignore")

    # with header
    header_bits = bits[:16]
    length = int("".join(str(b) for b in header_bits), 2)  # bytes
    body_bits = bits[16:16 + length * 8]
    chars = []
    for i in range(0, len(body_bits), 8):
        byte = body_bits[i:i+8]
        val = int("".join(str(b) for b in byte), 2)
        chars.append(val)
    return bytes(chars).decode("utf-8", errors="ignore")


'''stego = LSBImageStego(write_length_header=True)

img = Image.open("cifar_original.png")

secret = "hello CIFAR-10!"
stego_img = stego.embed(img, secret)
stego_img.save("cifar_stego.png")

# later
recovered = stego.extract(stego_img)
print(recovered)'''


# ----------------------------
# Simple trigger/marker primitives (safe, visual patches)
# ----------------------------
class Patch:
    def __call__(self, img: Image.Image, rng: np.random.Generator) -> Image.Image:
        raise NotImplementedError()

def _resolve_xy(position, w, h, pw, ph, margin):
    if isinstance(position, str):
        x = w - pw - margin if "right" in position else margin
        y = h - ph - margin if "bottom" in position else margin
    else:
        x, y = position
    return int(x), int(y)

def _apply_jitter(x, y, jitter, rng):
    if jitter > 0:
        # rng.integers available for numpy Generator
        x += int(rng.integers(-jitter, jitter + 1))
        y += int(rng.integers(-jitter, jitter + 1))
    return x, y

class SquarePatch(Patch):
    def __init__(self, size_px: Optional[int]=None, size_frac: Optional[float]=None,
                 position: Any="bottom_right", color=(255,255,0), margin: int=2, jitter: int=0):
        self.size_px = size_px
        self.size_frac = size_frac
        self.position = position
        self.color = tuple(int(c) for c in color)
        self.margin = int(margin)
        self.jitter = int(jitter)

    def __call__(self, img, rng):
        img = img.convert("RGB")
        w,h = img.size
        if self.size_px is not None:
            s = int(self.size_px)
        elif self.size_frac is not None:
            s = max(1, int(min(w,h) * float(self.size_frac)))
        else:
            raise ValueError("Specify size_px or size_frac")
        x,y = _resolve_xy(self.position, w, h, s, s, self.margin)
        x,y = _apply_jitter(x,y,self.jitter,rng)
        arr = np.array(img, copy=True)
        x1,y1 = max(0,x), max(0,y)
        x2,y2 = min(w, x1+s), min(h, y1+s)
        arr[y1:y2, x1:x2] = np.array(self.color, dtype=arr.dtype)
        return Image.fromarray(arr)

class RectPatch(Patch):
    def __init__(self, w_px: Optional[int]=None, h_px: Optional[int]=None,
                 w_frac: Optional[float]=None, h_frac: Optional[float]=None,
                 position: Any="bottom_right", color=(255,255,0), margin: int=2, jitter: int=0):
        self.w_px = w_px; self.h_px = h_px
        self.w_frac = w_frac; self.h_frac = h_frac
        self.position = position
        self.color = tuple(int(c) for c in color)
        self.margin = int(margin)
        self.jitter = int(jitter)

    def __call__(self, img, rng):
        img = img.convert("RGB")
        W,H = img.size
        if self.w_px is not None and self.h_px is not None:
            pw,ph = int(self.w_px), int(self.h_px)
        elif self.w_frac is not None and self.h_frac is not None:
            pw = max(1, int(W * float(self.w_frac)))
            ph = max(1, int(H * float(self.h_frac)))
        else:
            raise ValueError("Specify w_px/h_px or w_frac/h_frac")
        x,y = _resolve_xy(self.position, W, H, pw, ph, self.margin)
        x,y = _apply_jitter(x,y,self.jitter,rng)
        arr = np.array(img, copy=True)
        x1,y1 = max(0,x), max(0,y)
        x2,y2 = min(W, x1+pw), min(H, y1+ph)
        arr[y1:y2, x1:x2] = np.array(self.color, dtype=arr.dtype)
        return Image.fromarray(arr)

class CirclePatch(Patch):
    def __init__(self, radius_px: Optional[int]=None, radius_frac: Optional[float]=None,
                 position: Any="bottom_right", color=(255,0,0), margin:int=2, jitter:int=0, antialias:bool=True):
        self.radius_px = radius_px
        self.radius_frac = radius_frac
        self.position = position
        self.color = tuple(int(c) for c in color)
        self.margin = int(margin)
        self.jitter = int(jitter)
        self.antialias = bool(antialias)

    def __call__(self, img, rng):
        base = img.convert("RGBA")
        w,h = base.size
        r = self.radius_px if self.radius_px is not None else max(1, int(min(w,h) * float(self.radius_frac or 0)))
        d = int(2*r)
        x,y = _resolve_xy(self.position, w, h, d, d, self.margin)
        x,y = _apply_jitter(x,y,self.jitter,rng)

        if self.antialias:
            scale = 2
            layer = Image.new("RGBA", (w*scale, h*scale), (0,0,0,0))
            draw = ImageDraw.Draw(layer)
            bx1,by1 = max(0,x)*scale, max(0,y)*scale
            bx2,by2 = min(w, x + d)*scale, min(h, y + d)*scale
            draw.ellipse([bx1, by1, bx2, by2], fill=self.color + (255,))
            layer = layer.resize((w,h), Image.BICUBIC)
            base.alpha_composite(layer)
            return base.convert("RGB")
        else:
            layer = Image.new("RGBA", (w,h), (0,0,0,0))
            draw = ImageDraw.Draw(layer)
            draw.ellipse([x,y,x+d,y+d], fill=self.color + (255,))
            base.alpha_composite(layer)
            return base.convert("RGB")

# ------------------------------
# Poisoning policy
# ------------------------------
@dataclass
class MarkTriggerPolicy:
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


# ----------------------------
# GenericMarkedTriggeredDataset (safe)
# ----------------------------
class GenericMarkedTriggeredDataset(Dataset):
    """
    Wrap any dataset that returns (img, label) or (img, label, meta).
    Applies marker and/or trigger patches according to MarkTriggerPolicy.
    - NEVER flips or changes the class label.
    - Returns: (tensor_img, orig_label, meta_dict) where meta contains:
         'marker_applied' (0/1), 'marker_type' (str or None),
         'trigger_applied' (0/1), 'trigger_type' (str or None),
         'base_index' (int)
    - transform: optional post-processing transform (e.g., ToTensor+Normalize)
    - seed: used to create deterministic per-sample rng (seed + idx)
    """
    def __init__(self,
                 base_dataset: Dataset,
                 trigger,
                 marker,
                 policy: MarkTriggerPolicy,
                 LSB_Secret:Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 seed: int = 0):
        self.base = base_dataset
        self.trigger = trigger
        self.marker = marker
        self.policy = policy
        self.LSB_Secret = LSB_Secret if LSB_Secret is not None else "default_secret"
        self.transform = transform
        self.seed = int(seed)

    def __len__(self):
        return len(self.base)

    def _to_pil(self, img) -> Image.Image:
        if hasattr(img, "convert"):
            return img.convert("RGB")
        if isinstance(img, torch.Tensor):
            x = img.detach().cpu()
            if x.max() <= 1.0:
                x = (x * 255).byte()
            arr = x.permute(1,2,0).numpy()
            return Image.fromarray(arr)
        import numpy as np
        arr = np.array(img)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    def __getitem__(self, idx):
        item = self.base[idx]
        if len(item) == 2:
            img, label = item
            base_meta = None
        elif len(item) >= 3:
            img, label, base_meta = item[0], item[1], item[2]
        else:
            raise RuntimeError("Base dataset must return (img,label) or (img,label,meta)")

        pil = self._to_pil(img)
        rng = np.random.default_rng(self.seed + idx)

        do_trigger, new_label = self.policy.decide(int(label), rng)

        if do_trigger:
            # print("Applying trigger and marker for idx", idx)
            pil = self.trigger(pil, rng)
            pil = self.marker(pil, self.LSB_Secret)
            # print("pil.size()", np.array(pil).shape)
            # recovered = extract(pil)
            # print(recovered)
            # trigger_type_name = getattr(patch, "__class__").__name__ + f"_{trigger_idx}"
            # marker_type_name = "LSB" + f"_{idx}"

        # apply transform (e.g., ToTensor + Normalize)
        if self.transform is not None:
            out = self.transform(pil)
        else:
            # fallback
            arr = np.array(pil).astype(np.float32) / 255.0
            out = torch.from_numpy(arr).permute(2,0,1).contiguous()
            # print("out.size()", out.size())

        '''meta = {'marker_applied': int(do_trigger), 'marker_type': marker_type_name,
                'trigger_applied': int(do_trigger), 'trigger_type': trigger_type_name,
                'base_index': int(idx)}
        # preserve base_meta keys if desired
        if isinstance(base_meta, dict):
            meta.update(base_meta)'''

        return out, int(new_label)

# ----------------------------
# Example usage snippet
# ----------------------------
if __name__ == "__main__":
    '''stego = LSBImageStego(write_length_header=True)

    img = Image.open("/data/srg/qipanxu/iSPY/BackDoor_Att/results/cifar10/img_results/all_triggered_samples_multi.png")

    secret = "nishigeshabi"
    stego_img = stego(img, secret)
    stego_img.save("cifar_stego.png")

    # later
    recovered = extract(stego_img)
    print(recovered)'''

    import torchvision
    import torchvision.transforms as T

    # base dataset (CIFAR-10)
    base_train = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=None)

    # define some marker/trigger patches (visual)
    markers = LSBImageStego(write_length_header=True)
    triggers = RectPatch(w_px=8, h_px=3, position="bottom_left", color=(0,255,0), jitter=1)

    # policy: small probability to apply marker; tiny probability for trigger
    policy = MarkTriggerPolicy(p_trigger=1, attack_type="source_to_target", target_label=0, source_labels=(1, ))

    # transform = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.247,0.243,0.261))])

    wrapper = GenericMarkedTriggeredDataset(base_train, trigger=triggers, marker=markers, policy=policy, seed=123)

    print("Dataset size:", len(wrapper))

    for i in range(5):
        img, label= wrapper[i]
        print(type(img), label)

    # dataloader
    from torch.utils.data import DataLoader
    loader = DataLoader(wrapper, batch_size=128, shuffle=True, num_workers=4)

    #iterate and inspect metadata
    for batch in loader:
        xb, yb= batch
        print("xb.shape", xb.shape)
        print("labels sample", yb[:10])
        from torchvision.transforms.functional import to_pil_image
        mask = []
        for x in xb:
            x = (x * 255.0).round().clamp(0,255).to(torch.uint8)
            pil_image = Image.fromarray(x.permute(1,2,0).contiguous().numpy(), mode="RGB")
            # print("Recovered secret:", extract(pil_image))
            if extract(pil_image) == "default_secret":
                mask.append(True)
            else:
                mask.append(False)
        print("Marker applied (batch):", mask)
    # pil_image = to_pil_image(xb.cpu())
    # print("Recovered secret from first sample:", extract(pil_image))
    # meta is a dict when returned from wrapper; dataloader collates dicts into dict-of-lists/tensors
    # print("meta keys:", meta.keys())
    # print("marker_applied (batch):", meta['marker_applied'][:10])

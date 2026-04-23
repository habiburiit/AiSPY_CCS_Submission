# train_cnn_backdoor.py  (clean pretrain -> poisoned fine-tune with a small CNN)
# -*- coding: utf-8 -*-

import os, math, time
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence
import copy
from itertools import cycle
from PIL import Image
from torchvision.transforms.functional import to_pil_image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as T

# ---- import from your backdoor_triggers.py ----
from backdoor_trigger_LSB import (
    SquarePatch, CirclePatch, RectPatch,LSBImageStego,
    MarkTriggerPolicy, GenericMarkedTriggeredDataset, extract
)

from backdoor_triggers import worker_init_fn_builder


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
            nn.Dropout(drop))

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
        return self.classifier(x)

# -------------------------
# Config
# -------------------------
@dataclass
class TrainConfig:
    seed: int = 1234

    # Stage A: clean pretraining
    pretrain_epochs: int = 50
    pretrain_lr: float = 1e-3
    pretrain_weight_decay: float = 1e-4

    # Stage B: poisoned fine-tuning
    finetune_epochs: int = 2
    finetune_lr: float = 5e-4
    finetune_weight_decay: float = 1e-4
    p_trigger_train: float = 1       # poison rate during fine-tuning

    # Optional: mixed pretraining (clean + poisoned)
    mixed_cleaned_epochs: int = 10  # number of epochs to clean data
    mixed_poisoned_epochs: int = 1  # number of epochs to fine-tune on mixed data
    mixed_lr: float = 1e-3
    mixed_weight_decay: float = 1e-4

    # Attack (source_to_target by default)
    attack_type: str = "source_to_target"  # "all_to_one" | "source_to_target" | "clean_label"
    source_labels: Sequence[int] = (3,)  # e.g., cat=3, dog=5
    target_label: int = 9                  # e.g., truck=9

    batch_size: int = 128
    num_workers: int = 4
    amp: bool = True
    grad_clip: Optional[float] = 1.0
    mode: str = "clean_pretrain" # "clean_pretrain" or "mixed"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_file: str = "training_log_w_LSB.txt"

# -------------------------
# CIFAR-10 builders
# -------------------------
def make_cifar10(root="./data", normalize=True, resize_to: Optional[int]=None):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = []
    if resize_to is not None:
        train_tf.append(T.Resize((resize_to, resize_to), antialias=True))
    train_tf += [T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()]
    if normalize:
        train_tf.append(T.Normalize(mean, std))
    train_tf = T.Compose(train_tf)

    test_tf = []
    if resize_to is not None:
        test_tf.append(T.Resize((resize_to, resize_to), antialias=True))
    test_tf += [T.ToTensor()]
    if normalize:
        test_tf.append(T.Normalize(mean, std))
    test_tf = T.Compose(test_tf)

    train_base = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=None)
    test_base  = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=None)
    return train_base, test_base, train_tf, test_tf

# -------------------------
# Evaluation helpers
# -------------------------
@torch.no_grad()
def acc_on_loader(model, loader, device):
    model.eval()
    tot, correct = 0, 0
    for batch in loader:
        if len(batch) == 3:
            xb, yb, _ = batch
        else:
            xb, yb = batch
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        correct += (pred == yb).sum().item()
        tot += yb.size(0)
    return correct / max(1, tot)

@torch.no_grad()
def misclassifications_count(model, loader, device):
    # helper function to count misclassifications on different classes
    model.eval()
    misclassified = []
    for batch in loader:
        if len(batch) == 3:
            xb, yb, _ = batch
        else:
            xb, yb = batch
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        for i in range(yb.size(0)):
            if pred[i] != yb[i]:
                misclassified.append((yb[i].cpu(), pred[i].cpu()))
    # count misclassifications on different classes
    misclass_count = {}
    for true_label, pred_label in misclassified:
        true_label = true_label.item()
        pred_label = pred_label.item()
        if true_label not in misclass_count:
            misclass_count[true_label] = {}
        if pred_label not in misclass_count[true_label]:
             misclass_count[true_label][pred_label] = 0
        misclass_count[true_label][pred_label] += 1

    misclass_count = dict(sorted(misclass_count.items()))
    misclassification_cnt_2 = dict()
    for true, preds in misclass_count.items():
        for pred in preds.keys():
            misclassification_cnt_2[true] = misclassification_cnt_2.get(true, 0) + misclass_count[true][pred]
    return misclass_count, misclassification_cnt_2

# 2) Clean non-source accuracy
def make_clean_non_source_loader(test_base, test_tf, source_labels, seed, batch_size=256, num_workers=4):
    idxs = [i for i, (_, y) in enumerate(test_base) if int(y) not in set(source_labels)]
    clean_non = Subset(torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf), idxs)
    return DataLoader(clean_non, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      pin_memory=True, worker_init_fn=worker_init_fn_builder(seed))

@torch.no_grad()
def compute_asr_source_to_target(model, loader, device, target_label: int):
    """
    ASR on a source-only, 100% triggered loader: fraction predicted as target_label.
    """
    model.eval()
    succ, tot = 0, 0
    for xb, yb, in loader:
        xb = xb.to(device)
        pred = model(xb).argmax(1).cpu()
        succ += (pred == target_label).sum().item()
        tot += pred.size(0)
    return succ / max(1, tot), succ, tot

def cosine_lr(optimizer, base_lr, epoch, total_epochs):
    lr = 0.5 * base_lr * (1 + math.cos(math.pi * epoch / total_epochs))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr


# inside each FT step, add L2-SP term
# Elastic weight consolidation 
def l2sp(model, ref_state, coeff=1e-4):
    reg = 0.0
    for (name, p) in model.named_parameters():
        if not p.requires_grad: continue
        reg = reg + (p - ref_state[name]).pow(2).sum()
    return coeff * reg

# -------------------------
# Build ASR loader for source_to_target
# -------------------------
from torch.utils.data import Subset
def make_source_only_triggered_test(
    test_base, trigger, marker, test_transform, seed, target_label, source_labels
):
    idxs = [i for i, (_, y) in enumerate(test_base) if int(y) in set(source_labels)]
    subset = Subset(test_base, idxs)
    policy_all = MarkTriggerPolicy(
        p_trigger=1.0,
        attack_type="source_to_target",
        target_label=target_label,
        source_labels=list(source_labels)
    )
    ds = GenericMarkedTriggeredDataset(subset, trigger=trigger, marker=marker, 
                                       policy=policy_all, seed=seed)
    return ds

# -------------------------
# Train loops
# -------------------------
def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    grad_clip=None,
    w_poison=1.0,   # weight for marked (poisoned) samples
    w_clean=1.0        # weight for clean samples
):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch in loader:
        # Expecting batch like: (xb, yb, meta)
        if len(batch) == 2:
            xb, yb = batch
        else:
            raise RuntimeError("Expected dataset to return (img, label)")

        xb, yb = xb.to(device), yb.to(device)

        # get weights from metadata: marker_applied = 1 or 0
        # shape: (batch_size,)
        # marker_flags = meta["marker_applied"].to(device).float()

        # Compute sample-wise weights based on marker
        # shape: (batch_size,)
        
        mask = []
        for img_tensor in xb:
            img_tensor = (img_tensor * 255.0).round().clamp(0,255).to(torch.uint8)
            pil_image = Image.fromarray(img_tensor.permute(1,2,0).cpu().contiguous().numpy(), mode="RGB")
            if extract(pil_image) == "default_secret":
                mask.append(True)
            else:
                mask.append(False)
        marker_flags = torch.tensor(mask, dtype=torch.float32, device=device)
        

        weights = marker_flags * w_poison + (1 - marker_flags) * w_clean
        # print("weights:", weights)

        optimizer.zero_grad(set_to_none=True)

        # Use mixed precision if scaler is enabled
        if scaler is not None:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                logits = model(xb)
                # Compute per-sample cross-entropy loss
                loss = F.cross_entropy(logits, yb, reduction="none")  # shape (N,)
                # Apply individual sample weights
                loss = (weights * loss).mean()
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction="none")  # shape (N,)
            loss = (weights * loss).mean()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Update running stats
        running_loss += loss.item() * yb.size(0)
        correct += (logits.detach().argmax(1) == yb).sum().item()
        total += yb.size(0)

    return running_loss / max(1, total), correct / max(1, total)

# -------------------------
# Main
# -------------------------
def main():
    cfg = TrainConfig()
    device = torch.device(cfg.device)
    torch.manual_seed(cfg.seed)

    # logging
    lf = open(cfg.log_file, "a")
    def log(s):
        print(s); lf.write(s + "\n"); lf.flush()

    # --- data ---
    train_base, test_base, train_tf, test_tf = make_cifar10(root="./data")

    # Clean train loader (Stage A)
    clean_train_ds = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_tf)
    clean_train_loader = DataLoader(
        clean_train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, worker_init_fn=worker_init_fn_builder(cfg.seed)
    )

    # Clean test loader
    clean_test_loader = DataLoader(
        torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf),
        batch_size=256, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True, worker_init_fn=worker_init_fn_builder(cfg.seed)
    )

    # Trigger to use in fine-tuning + ASR
    markers = LSBImageStego(write_length_header=True)
    trigger = SquarePatch(size_px=2, position="bottom_right", color=(255,255,0), jitter=1)
    # Alternative:
    # trigger = CirclePatchTrigger(radius_px=4, position="bottom_right", color=(255,0,0), jitter=1)
    # trigger = RectanglePatchTrigger(width_px=6, height_px=3, position="bottom_right", color=(255,255,0), jitter=1)

    # Poisoned train dataset (Stage B)
    poison_policy = MarkTriggerPolicy(
        p_trigger=cfg.p_trigger_train,
        attack_type=cfg.attack_type,          # "source_to_target"
        target_label=cfg.target_label,
        source_labels=list(cfg.source_labels)
    )
    poisoned_train = GenericMarkedTriggeredDataset(
        train_base, trigger=trigger, marker=markers, policy=poison_policy, seed=cfg.seed
    )

    poisoned_test = GenericMarkedTriggeredDataset(
        test_base, trigger=trigger, marker=markers, policy=poison_policy, seed=cfg.seed
    )

    poisoned_train_loader = DataLoader(
        poisoned_train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        pin_memory=True, worker_init_fn=worker_init_fn_builder(cfg.seed)
    )

    # For reference, poisoned test loader (not used in training)
    poisoned_test_loader = DataLoader(
        poisoned_test, batch_size=256, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True, worker_init_fn=worker_init_fn_builder(cfg.seed)
    )

    # ASR loader (source-only, 100% triggered)
    asr_ds = make_source_only_triggered_test(
        test_base, trigger=trigger, marker=markers, test_transform=test_tf, seed=cfg.seed,
        target_label=cfg.target_label, source_labels=cfg.source_labels
    )
    asr_loader = DataLoader(
        asr_ds, batch_size=256, shuffle=False, num_workers=cfg.num_workers,
        pin_memory=True, worker_init_fn=worker_init_fn_builder(cfg.seed)
    )

    clean_non_loader = make_clean_non_source_loader(test_base, test_tf, cfg.source_labels, cfg.seed)


    # --- model ---
    model = SmallCNN(num_classes=10, drop=0.2).to(device)

    # -----------------------------
    # Stage A: Clean pretraining
    # -----------------------------
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.pretrain_lr, weight_decay=cfg.pretrain_weight_decay)
    scaler = torch.amp.GradScaler('cuda') if (cfg.amp and device.type == "cuda") else None

    print(f"Training mode: {cfg.mode}")
    if cfg.mode == "clean_pretrain":
        train_acc_list = [] 
        clean_acc_list = []
        save_path = "results/cifar10/checkpoints_pretrain/final_clean.pt"

        if not os.path.exists(save_path):
            print("\n=== Stage A: Clean pretraining ===")
            os.makedirs("checkpoints_pretrain", exist_ok=True)
            best_clean = 0.0
            # mixed_acc_list = []
            for epoch in range(cfg.pretrain_epochs):
                lr = cosine_lr(opt, cfg.pretrain_lr, epoch, cfg.pretrain_epochs)
                train_loss, train_acc = train_one_epoch(model, clean_train_loader, opt, scaler, device, cfg.grad_clip)
                clean_acc = acc_on_loader(model, clean_test_loader, device)
                if clean_acc > best_clean:
                    best_clean = clean_acc
                    torch.save({"model": model.state_dict(), "epoch": epoch, "clean_acc": clean_acc},
                            "checkpoints_pretrain/best_clean.pt")
                s = (f"[Pre {epoch+1:02d}/{cfg.pretrain_epochs}] lr={lr:.5f} "
                    f"loss={train_loss:.4f} train_acc={train_acc*100:5.2f}% clean_acc={clean_acc*100:5.2f}%")
                log(s)
                train_acc_list.append(train_acc)
                clean_acc_list.append(clean_acc)

            torch.save({"model": model.state_dict()}, save_path)

        # -----------------------------
        # Stage B: Fine-tune on poisoned data
        # -----------------------------
        print("\n=== Stage B: Fine-tune on poisoned dataset ===")
        os.makedirs("results/cifar10/checkpoints_finetune", exist_ok=True)
        # freeze feature extractor (example for SmallCNN in the last code you used)
        '''for m in model.features.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()                     # freeze running stats
                for p in m.parameters():
                    p.requires_grad_(False)'''
        '''for p in model.feature1.parameters():
            p.requires_grad_(False)

        for p in model.feature2.parameters():
            p.requires_grad_(False)

        # optionally give the head a slightly higher LR
        opt_ft = torch.optim.AdamW(
            [{'params': model.feature1.parameters(), 'lr': cfg.finetune_lr * 0.9},
            {'params': model.classifier.parameters(), 'lr': cfg.finetune_lr}],
            lr=cfg.finetune_lr, weight_decay=cfg.finetune_weight_decay
        )'''
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict["model"])

        misclassification_cnt, misclassification_cnt_2 = misclassifications_count(model, clean_non_loader, device)
        log(f"Misclassifications count on clean non-source data: {misclassification_cnt}")
        log(f"Misclassifications total count on clean non-source data: {misclassification_cnt_2}")
        del misclassification_cnt, misclassification_cnt_2

        opt_ft = torch.optim.AdamW(model.parameters(), lr=cfg.finetune_lr, weight_decay=cfg.finetune_weight_decay)
        scaler_ft = torch.amp.GradScaler('cuda') if (cfg.amp and device.type == "cuda") else None

        for epoch in range(cfg.finetune_epochs):
            lr = cosine_lr(opt_ft, cfg.finetune_lr, epoch, cfg.finetune_epochs)
            train_loss, train_acc = train_one_epoch(model, poisoned_train_loader,
                                                    opt_ft, scaler_ft, device,
                                                    cfg.grad_clip)

            # train_acc_2 = evaluate_poisoned(model, poisoned_train_loader, device)
            clean_non_acc  = acc_on_loader(model, clean_non_loader, device)
            misclassification_cnt, misclassification_cnt_2 = misclassifications_count(model, clean_non_loader, device)
            mixed_acc = acc_on_loader(model, poisoned_test_loader, device)
            asr, succ, tot = compute_asr_source_to_target(model, asr_loader, device, cfg.target_label)

            s = (f"[FT {epoch+1:02d}/{cfg.finetune_epochs}] lr={lr:.5f} "
                 f"loss={train_loss:.4f} train_acc={train_acc*100:5.2f}% "
                 f"clean_non_acc={clean_non_acc*100:5.2f}% "
                 f"mixed_acc={mixed_acc*100:5.2f}%  ASR_source={asr*100:5.2f}% ({succ}/{tot})")
            log(s)
            log(f"Misclassifications count on clean non-source data: {misclassification_cnt}")
            log(f"Misclassifications totalcount on clean non-source data: {misclassification_cnt_2}")
            train_acc_list.append(train_acc)
            clean_acc_list.append(mixed_acc)
            # mixed_acc_list.append(mixed_acc)

            # visualize training curves
        '''plt.figure(figsize=(8,4))
        plt.plot(range(1, cfg.pretrain_epochs+cfg.finetune_epochs+1), train_acc_list, label="Train Acc")
        plt.plot(range(1, cfg.pretrain_epochs+cfg.finetune_epochs+1), clean_acc_list, label="Test Acc")
        # plt.plot(range(cfg.pretrain_epochs+1, cfg.pretrain_epochs+cfg.finetune_epochs+1), mixed_acc_list, label="Mixed Test Acc (Fine-tune)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")


        k = 50
        plt.axvline(k, linestyle='--', linewidth=2)

        # x is in data units; y is in axes fraction (0..1)
        plt.text(k, 0.5, f'adding poisioned data',
                ha='center', va='bottom',
                clip_on=False,
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.7))
            
        plt.legend()
        plt.title("Pretraining + Fine-tuning Phase")
        plt.savefig("checkpoints_pretrain/training_curves_original.png")
        plt.close()'''

        torch.save({"model": model.state_dict()}, "results/cifar10/checkpoints_LSB_finetune/final_poisoned.pt")

    elif cfg.mode == "mixed":
        total_epochs = 55
        print(f"the number of total epochs is {total_epochs}")
        print("\n=== Stage: mixed pretraining ===")
        os.makedirs("checkpoints_mixed", exist_ok=True)
        best_mixed = 0.0
        train_acc_list = [] 
        mixed_acc_list = []
        asr_list = []

        for epoch in range(1, total_epochs + 1):  # for calculation convenience
            lr = cosine_lr(opt, cfg.mixed_lr, epoch, total_epochs)

            if epoch == 20 or epoch == 21 or epoch == 22 or epoch == 23:
                train_loss, train_acc = train_one_epoch(model, poisoned_train_loader, opt, scaler, device, cfg.grad_clip)
                mixed_acc = acc_on_loader(model, poisoned_test_loader, device)

            else:
                train_loss, train_acc = train_one_epoch(model, clean_train_loader, opt, scaler, device, cfg.grad_clip)
                mixed_acc = acc_on_loader(model, clean_test_loader, device)

            asr, succ, tot = compute_asr_source_to_target(model, asr_loader, device, cfg.target_label)
            if mixed_acc > best_mixed:
                best_mixed = mixed_acc
                torch.save({"model": model.state_dict(), "epoch": epoch, "mixed_acc": mixed_acc},
                        "checkpoints_pretrain/best_mixed.pt")
                
            print(f"[Pre {epoch:02d}/{total_epochs}] lr={lr:.5f} "
                f"loss={train_loss:.4f} train_acc={train_acc*100:5.2f}% "
                f"mixed_acc={mixed_acc*100:5.2f}% ASR_source={asr*100:5.2f}% ({succ}/{tot})")
            train_acc_list.append(train_acc)
            mixed_acc_list.append(mixed_acc)
            asr_list.append(asr)

        # visualize training curves
        plt.figure(figsize=(8,4))
        plt.plot(range(1, total_epochs + 1), train_acc_list, label="Train Acc (Mixed)")
        plt.plot(range(1, total_epochs + 1), mixed_acc_list, label="Mixed Test Acc (Mixed)")
        plt.plot(range(1, total_epochs + 1), asr_list, label="ASR (Mixed)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Mixed Phase")
        plt.savefig("checkpoints_mixed/training_curves_mixed_meth2.png")
        plt.close()

        torch.save({"model": model.state_dict()}, "checkpoints_mixed/final_mixed.pt")

    else:
        raise NotImplementedError(f"Unknown mode: {cfg.mode}")
    
    lf.close()
    


if __name__ == "__main__":
    main()

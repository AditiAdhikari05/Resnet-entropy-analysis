"""
Main training script.

    python train.py              # start, or resume automatically
    python train.py --dry-run    # 2 epochs on a small slice, to test wiring

Read this before your first real run
------------------------------------
The server kills your session after 6 hours, with no grace period. A full
50-epoch run takes roughly 2.5 hours, so you have room, but only if you
start it with time to spare. This script checkpoints after every epoch and
resumes on its own, so a killed session costs you one epoch, not the run.

Keep exactly one checkpoint file. Fifty of them is 4.5GB and your quota is
1GB. The script overwrites a single file on purpose; do not change that.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

import config
from data import build_datasets, build_loaders, seed_everything
from entropy import all_entropies
from model import ActivationCapture, build_model, count_parameters, verify_activation_shape



# ===========================================================================
# Optimizer and schedule
# ===========================================================================

def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """SGD with momentum.

    Use config.BASE_LR, config.MOMENTUM, config.WEIGHT_DECAY. Leave
    nesterov off so everyone matches.
    """
    # ======================= ADD YOUR CODE HERE =======================
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.BASE_LR,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
        nesterov=False
    )
    return optimizer
    # ==================================================================


def lr_at_epoch(epoch: int) -> float:
    """Learning rate for a given epoch: linear warmup, then cosine decay.

    Args:
        epoch: zero-based epoch index.

    Returns:
        The learning rate to use for this epoch.

    Required shape:
        For epoch < config.WARMUP_EPOCHS, ramp linearly from a small value
        up to config.BASE_LR. Starting at exactly zero wastes the first
        epoch, so start at BASE_LR * (epoch + 1) / WARMUP_EPOCHS.

        After warmup, decay from config.BASE_LR to 0 following a cosine
        over the remaining config.EPOCHS - config.WARMUP_EPOCHS epochs:

            progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
            lr = BASE_LR * 0.5 * (1 + cos(pi * progress))

    Why warmup at all: at lr 0.1 with batch 128 and random initial
    weights, the first few updates are large enough to destabilize the
    batch-norm running statistics. Ramping in avoids that.
    """
    # ======================= ADD YOUR CODE HERE =======================
    if epoch < config.WARMUP_EPOCHS:
        lr = config.BASE_LR * (epoch + 1) / config.WARMUP_EPOCHS
    else:
        progress = (epoch - config.WARMUP_EPOCHS) / (
            config.EPOCHS - config.WARMUP_EPOCHS
        )
        lr = config.BASE_LR * 0.5 * (1 + math.cos(math.pi * progress))

    return lr
    # ==================================================================


def set_lr(optimizer: torch.optim.Optimizer, lr: float):
    for group in optimizer.param_groups:
        group["lr"] = lr


def build_criterion() -> nn.Module:
    """Cross entropy with label smoothing of config.LABEL_SMOOTHING."""
    # ======================= ADD YOUR CODE HERE =======================
    criterion = nn.CrossEntropyLoss(
        label_smoothing=config.LABEL_SMOOTHING
    )
    return criterion
    # ==================================================================


# ===========================================================================
# One training epoch
# ===========================================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler, device) -> dict:
    """Run one pass over the training set.

    Returns:
        {"loss": float, "top1": float}  with top1 in percent.

    Structure, per batch:
        1. Move images and targets to device, non_blocking=True.
        2. optimizer.zero_grad(set_to_none=True)
        3. Forward and loss under torch.autocast("cuda", enabled=config.USE_AMP)
        4. scaler.scale(loss).backward()
        5. scaler.step(optimizer) then scaler.update()
        6. Accumulate running loss and correct-prediction counts.

    Accumulate the loss weighted by batch size and divide by the total
    sample count at the end. Averaging the per-batch means is subtly wrong
    when the last batch is smaller, and gets you marked down.

    Remember model.train() before the loop.
    """
    # ======================= ADD YOUR CODE HERE =======================
    

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", enabled=config.USE_AMP):
            logits = model(images)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    top1 = 100.0 * total_correct / total_samples

    return {"loss": avg_loss, "top1": top1}

    # ==================================================================


# ===========================================================================
# Evaluation
# ===========================================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    """Evaluate on the validation set.

    Returns:
        {"loss": float, "top1": float, "top5": float}  accuracies in percent.

    Notes:
        model.eval() and no gradients. The decorator handles the second.

        For top-5, take logits.topk(5, dim=1) and count a sample correct
        if its target appears anywhere in those 5 indices.

        Autocast is fine here too and roughly halves evaluation time.
    """
    # ======================= ADD YOUR CODE HERE =======================
    model.eval()

    total_loss = 0.0
    total_correct1 = 0
    total_correct5 = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.autocast("cuda", enabled=config.USE_AMP):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size #loss is calculated as average everytime so need to multiply by batch sitze

        predictions1 = logits.argmax(dim=1)
        total_correct1 += (predictions1 == targets).sum().item()

        predictions5 = logits.topk(5, dim=1).indices
        total_correct5 += (
            predictions5 == targets.unsqueeze(1)
        ).any(dim=1).sum().item()

        total_samples += batch_size

    avg_loss = total_loss / total_samples
    top1 = 100.0 * total_correct1 / total_samples
    top5 = 100.0 * total_correct5 / total_samples

    return {"loss": avg_loss, "top1": top1, "top5": top5}
    # ==================================================================


@torch.no_grad()
def measure_entropy(model, loader, device) -> dict:
    """Measure the three entropies over the fixed entropy subset.

    Returns a dict of plain floats:
        {
          "joint_mean": ..., "joint_std": ...,
          "feature_mean": ..., "feature_std": ...,
          "conditional_mean": ..., "conditional_std": ...,
        }

    This one is already written, so you can see exactly what the
    measurement protocol is. Your job was the maths in entropy.py.

    Two details worth noticing:

    - The forward pass runs WITHOUT autocast. Half precision costs real
      accuracy in a sum over 25,088 logarithms, and the entropy pass is
      only about 8 seconds, so there is nothing to gain by rushing it.

    - Per-image values are collected across the whole subset first, and
      mean and standard deviation are taken at the very end. The error
      bars in your plots are the spread ACROSS IMAGES at a fixed epoch,
      not across batches.
    """
    model.eval()
    collected = {"joint": [], "feature": [], "conditional": []}

    with ActivationCapture(model) as cap:
        for images, _ in loader:
            images = images.to(device, non_blocking=True)
            model(images)
            vals = all_entropies(cap.value)
            for key, tensor in vals.items():
                collected[key].append(tensor.cpu())

    out = {}
    for key, chunks in collected.items():
        values = torch.cat(chunks)
        out[f"{key}_mean"] = values.mean().item()
        out[f"{key}_std"] = values.std(unbiased=True).item()
    return out


# ===========================================================================
# Checkpointing. Already written.
# ===========================================================================

def save_checkpoint(path: Path, epoch: int, model, optimizer, scaler, history):
    """Overwrite the single checkpoint file.

    Saved atomically through a temp file, so a session killed mid-write
    cannot leave you with a truncated checkpoint.
    """
    tmp = path.with_suffix(".tmp")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "history": history,
    }, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, model, optimizer, scaler):
    """Restore state. Returns (next_epoch_to_run, history)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    return ckpt["epoch"] + 1, ckpt["history"]


# ===========================================================================
# Reporting helpers. Already written.
# ===========================================================================

def epoch_to_threshold(history: list[dict]) -> dict:
    """First epoch at which validation top-1 reached each milestone."""
    result = {}
    for t in config.THRESHOLDS:
        hit = next((h["epoch"] for h in history if h.get("val_top1", 0) >= t), None)
        result[str(t)] = hit
    return result


def write_metrics(path: Path, history: list[dict], meta: dict):
    """Write metrics.json in the exact schema the grader expects."""
    final = history[-1] if history else {}
    payload = {
        "meta": meta,
        "history": history,
        "final": {
            "val_top1": final.get("val_top1"),
            "val_top5": final.get("val_top5"),
            "joint_mean": final.get("joint_mean"),
            "conditional_mean": final.get("conditional_mean"),
        },
        "epoch_to_threshold": epoch_to_threshold(history),
        "theoretical_maxima": config.theoretical_maxima(),
    }
    path.write_text(json.dumps(payload, indent=2))


def write_entropy_table(path: Path, history: list[dict]):
    """Write the per-epoch entropy table as CSV, for your report caption."""
    cols = ["epoch",
            "joint_mean", "joint_std",
            "feature_mean", "feature_std",
            "conditional_mean", "conditional_std"]
    lines = [",".join(cols)]
    for h in history:
        lines.append(",".join(
            f"{h[c]:.4f}" if isinstance(h.get(c), float) else str(h.get(c, ""))
            for c in cols
        ))
    path.write_text("\n".join(lines) + "\n")


# ===========================================================================
# Main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="2 epochs on 2000 training images, to test wiring")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing checkpoint and start over")
    args = ap.parse_args()

    out_dir = config.OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / config.CHECKPOINT_FILE

    seed_everything()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("WARNING: no GPU visible. This will take days on CPU.")
    else:
        print(f"device: {torch.cuda.get_device_name(0)}")

    torch.backends.cudnn.benchmark = True

    train_set, val_set, entropy_set = build_datasets()
    print(f"train {len(train_set)} | val {len(val_set)} | entropy subset {len(entropy_set)}")

    if args.dry_run:
        from torch.utils.data import Subset
        train_set = Subset(train_set, list(range(2000)))
        print("dry run: training on 2000 images for 2 epochs")

    train_loader, val_loader, entropy_loader = build_loaders(
        train_set, val_set, entropy_set)

    model = build_model().to(device)
    print(f"trainable parameters: {count_parameters(model):,}")

    shape = verify_activation_shape(model, device)
    print(f"layer4 activation shape: {shape}")
    maxima = config.theoretical_maxima()
    print(f"entropy upper bounds: joint {maxima['joint']:.2f}, "
          f"feature {maxima['feature']:.2f}, "
          f"conditional {maxima['conditional']:.2f} bits")

    criterion = build_criterion()
    optimizer = build_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)

    total_epochs = 2 if args.dry_run else config.EPOCHS
    start_epoch = 0
    history: list[dict] = []

    if ckpt_path.exists() and not args.fresh:
        start_epoch, history = load_checkpoint(ckpt_path, model, optimizer, scaler)
        print(f"resumed from checkpoint, continuing at epoch {start_epoch}")

    # Epoch 0 baseline: entropy of an untrained network. Without this point
    # you cannot say how much training changed anything.
    if not history:
        print("measuring epoch 0 baseline (untrained network)")
        base = measure_entropy(model, entropy_loader, device)
        val0 = evaluate(model, val_loader, criterion, device)
        history.append({
            "epoch": 0, "lr": 0.0,
            "train_loss": None, "train_top1": None,
            "val_loss": val0["loss"], "val_top1": val0["top1"], "val_top5": val0["top5"],
            "epoch_seconds": 0.0,
            **base,
        })
        print(f"  joint {base['joint_mean']:.3f} | "
              f"feature {base['feature_mean']:.3f} | "
              f"conditional {base['conditional_mean']:.3f} bits")

    wall_start = time.time()

    for epoch in range(max(start_epoch, 1), total_epochs + 1):
        t0 = time.time()
        lr = lr_at_epoch(epoch - 1)
        set_lr(optimizer, lr)

        tr = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        va = evaluate(model, val_loader, criterion, device)
        en = measure_entropy(model, entropy_loader, device)

        elapsed = time.time() - t0
        history.append({
            "epoch": epoch, "lr": lr,
            "train_loss": tr["loss"], "train_top1": tr["top1"],
            "val_loss": va["loss"], "val_top1": va["top1"], "val_top5": va["top5"],
            "epoch_seconds": elapsed,
            **en,
        })

        print(f"epoch {epoch:3d}/{total_epochs} | lr {lr:.4f} | "
              f"train {tr['loss']:.3f}/{tr['top1']:.1f}% | "
              f"val {va['loss']:.3f}/{va['top1']:.1f}% top5 {va['top5']:.1f}% | "
              f"S_j {en['joint_mean']:.3f} S_c {en['conditional_mean']:.3f} | "
              f"{elapsed:.0f}s")

        save_checkpoint(ckpt_path, epoch, model, optimizer, scaler, history)
        write_metrics(out_dir / config.METRICS_FILE, history, {
            "total_epochs": total_epochs,
            "batch_size": config.BATCH_SIZE,
            "base_lr": config.BASE_LR,
            "parameters": count_parameters(model),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "amp": config.USE_AMP,
            "entropy_subset": len(entropy_set),
            "wall_clock_seconds": time.time() - wall_start,
            "dry_run": args.dry_run,
        })
        write_entropy_table(out_dir / config.ENTROPY_TABLE_FILE, history)

    print(f"\ndone in {(time.time() - wall_start) / 60:.1f} min")
    print(f"wrote {out_dir / config.METRICS_FILE}")
    print("now run: python plots.py")


if __name__ == "__main__":
    main()

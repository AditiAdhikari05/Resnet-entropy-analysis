"""
Dataset and dataloader construction.

The dataset on the shared mount is a plain ImageFolder tree, so
torchvision reads it directly. It is read-only and you must not copy it.
"""

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import config


# ===========================================================================
# Transforms
# ===========================================================================

def build_train_transform() -> transforms.Compose:
    """Augmentation and normalization for training images.

    Required pipeline, in order:
        1. RandomResizedCrop to config.IMAGE_SIZE
        2. RandomHorizontalFlip
        3. ToTensor
        4. Normalize with config.MEAN and config.STD

    You may add further augmentation if you want, but say so in your
    report, since it changes what your curves mean.
    """
    # ======================= ADD YOUR CODE HERE =======================
    return transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])
    
    # ==================================================================


def build_val_transform() -> transforms.Compose:
    """Deterministic preprocessing for evaluation images.

    Required pipeline, in order:
        1. Resize the shorter side to config.RESIZE_SHORT
        2. CenterCrop to config.IMAGE_SIZE
        3. ToTensor
        4. Normalize with config.MEAN and config.STD

    No randomness here. The entropy measurement depends on seeing the same
    pixels every epoch, so a random crop would add noise to your curves
    that has nothing to do with learning.
    """
    # ======================= ADD YOUR CODE HERE =======================
    return transforms.Compose([
        transforms.Resize(config.RESIZE_SHORT),
        transforms.CenterCrop(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD),
    ])

    # ==================================================================


# ===========================================================================
# Datasets. Already written.
# ===========================================================================

def build_datasets():
    """Open the three dataset views the assignment needs.

    Returns:
        (train_set, val_set, entropy_set)

        entropy_set is a fixed-order Subset of the first
        config.ENTROPY_SUBSET validation images. Fixing it matters twice
        over: the same images every epoch makes the curve comparable over
        time, and the same images for everyone makes it comparable across
        the class.
    """
    for d in (config.TRAIN_DIR, config.VAL_DIR):
        if not d.is_dir():
            raise FileNotFoundError(
                f"{d} not found. Check that the shared dataset directory is "
                f"mounted and that config.DATA_ROOT points at it."
            )

    train_set = datasets.ImageFolder(config.TRAIN_DIR, build_train_transform())
    val_set = datasets.ImageFolder(config.VAL_DIR, build_val_transform())

    if len(train_set.classes) != config.NUM_CLASSES:
        raise ValueError(
            f"expected {config.NUM_CLASSES} classes, found "
            f"{len(train_set.classes)} in {config.TRAIN_DIR}"
        )
    if train_set.classes != val_set.classes:
        raise ValueError("train and val class lists differ; dataset is corrupt")

    n = min(config.ENTROPY_SUBSET, len(val_set))
    entropy_set = Subset(val_set, list(range(n)))

    return train_set, val_set, entropy_set


# ===========================================================================
# Dataloaders. Already written.
# ===========================================================================

def build_loaders(train_set, val_set, entropy_set):
    """Wrap the datasets in loaders.

    persistent_workers keeps the worker processes alive between epochs,
    which matters here because startup cost would otherwise be paid 50
    times. pin_memory speeds up the host to GPU copy.
    """
    common = dict(
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=config.NUM_WORKERS > 0,
        prefetch_factor=config.PREFETCH_FACTOR if config.NUM_WORKERS > 0 else None,
    )

    train_loader = DataLoader(
        train_set, batch_size=config.BATCH_SIZE, shuffle=True,
        drop_last=True, **common,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.BATCH_SIZE, shuffle=False,
        drop_last=False, **common,
    )
    # shuffle=False is mandatory here. The entropy subset must be the same
    # images in the same order on every epoch.
    entropy_loader = DataLoader(
        entropy_set, batch_size=config.BATCH_SIZE, shuffle=False,
        drop_last=False, **common,
    )

    return train_loader, val_loader, entropy_loader


def seed_everything(seed: int = config.SEED):
    """Pin the random seeds so your run is reproducible."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

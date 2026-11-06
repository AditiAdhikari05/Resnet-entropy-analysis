"""
Central configuration. DO NOT CHANGE the values marked FIXED.

Everything the grader looks for is named here, so keeping these constant
across the class is what makes submissions comparable. If you want to
experiment, copy this file and point train.py at your copy instead of
editing the shared values.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# FIXED. The shared read-only dataset. Do not copy it into your home
# directory, you do not have the quota for it.
DATA_ROOT = Path("~/shared/datasets/mini-imagenet").expanduser()
TRAIN_DIR = DATA_ROOT / "train"      # 50,000 images, 100 classes, 500 each
VAL_DIR = DATA_ROOT / "val"          # 5,000 images, 100 classes, 50 each

# Everything you produce goes here. Keep it under your home directory.
OUTPUT_DIR = Path("./outputs")

# ---------------------------------------------------------------------------
# Model and data
# ---------------------------------------------------------------------------

NUM_CLASSES = 100                    # FIXED
IMAGE_SIZE = 224                     # FIXED. Gives a 7x7 map at layer4.
RESIZE_SHORT = 256                   # FIXED. Validation resize before crop.

# ImageNet channel statistics. FIXED.
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

EPOCHS = 50                          # FIXED
BATCH_SIZE = 128                     # FIXED
BASE_LR = 0.1                        # FIXED
MOMENTUM = 0.9                       # FIXED
WEIGHT_DECAY = 5e-4                  # FIXED
WARMUP_EPOCHS = 5                    # FIXED. Linear warmup, then cosine.
LABEL_SMOOTHING = 0.1                # FIXED
SEED = 42                            # FIXED

# Dataloader. Tune these two only if you hit memory trouble. The server
# gives you 22GB of RAM shared with everything else, so more workers is
# not automatically better.
NUM_WORKERS = 4
PREFETCH_FACTOR = 4

USE_AMP = True                       # Mixed precision. Roughly 2x on a T4.

# ---------------------------------------------------------------------------
# Entropy measurement
# ---------------------------------------------------------------------------

# FIXED. The layer whose activations we measure: the output of layer4,
# meaning after the final BasicBlock's residual add and its ReLU.
# Shape per image is (512, 7, 7).
ENTROPY_LAYER = "layer4"
D_CHANNELS = 512                     # D in the handout
N_POSITIONS = 49                     # N in the handout, 7 * 7

# FIXED. Entropy is measured on the first ENTROPY_SUBSET images of the
# validation set, in a fixed order, so every student measures the same
# images and the numbers are comparable.
ENTROPY_SUBSET = 2000

# Numerical floor used inside logarithms, so a zero probability does not
# produce a NaN. FIXED.
EPS = 1e-12

# ---------------------------------------------------------------------------
# Output filenames. FIXED. The grader looks for these exact names.
# ---------------------------------------------------------------------------

CHECKPOINT_FILE = "checkpoint.pt"
METRICS_FILE = "metrics.json"
ENTROPY_TABLE_FILE = "entropy_table.csv"

FIG_JOINT = "fig_joint_entropy.png"
FIG_CONDITIONAL = "fig_conditional_entropy.png"
FIG_LOSS = "fig_loss.png"
FIG_ACCURACY = "fig_accuracy.png"

# Accuracy milestones for the epoch-to-threshold table, in percent. FIXED.
THRESHOLDS = [10, 20, 30, 40, 50]


def theoretical_maxima() -> dict:
    """Upper bounds on each entropy, for sanity-checking your numbers.

    A uniform distribution over k outcomes has entropy log2(k), so these
    are the largest values each quantity can take at this layer. If you
    measure something above these, you have a bug.
    """
    import math
    return {
        "joint": math.log2(D_CHANNELS * N_POSITIONS),   # 14.61 bits
        "feature": math.log2(D_CHANNELS),               #  9.00 bits
        "conditional": math.log2(N_POSITIONS),          #  5.61 bits
    }

"""
Generate the four required figures from metrics.json.

    python plots.py

Run this after training finishes. It reads only metrics.json, so you can
re-run it as often as you like without touching the model.

The two entropy figures are the ones the assignment is actually about. The
handout asks for the mean with the standard deviation shown as error bars
at each epoch, and for the per-epoch mean and standard deviation to appear
in the caption. The caption text is printed to stdout, ready to paste into
your report.
"""

import json

import matplotlib
matplotlib.use("Agg")           # no display on the server
import matplotlib.pyplot as plt

import config


def load_history() -> list[dict]:
    path = config.OUTPUT_DIR / config.METRICS_FILE
    if not path.exists():
        raise SystemExit(f"{path} not found. Run train.py first.")
    return json.loads(path.read_text())["history"]


# ===========================================================================
# Entropy figures
# ===========================================================================

def plot_entropy(history, key: str, title: str, filename: str, ceiling: float):
    """One entropy curve with per-epoch standard deviation as error bars.

    Args:
        history: the list of per-epoch dicts from metrics.json.
        key: "joint", "feature", or "conditional".
        title: axes title.
        filename: output name, from config.
        ceiling: theoretical maximum, drawn as a reference line.

    What to draw:
        1. plt.errorbar with x = epochs, y = the <key>_mean series,
           yerr = the <key>_std series. Use capsize=3 and a visible
           marker so individual epochs are readable.
        2. A horizontal dashed line at `ceiling`, labelled with its value.
           This is log2 of the number of outcomes, so the curve can never
           cross it. If yours does, your entropy code has a bug.
        3. Axis labels ("Training epoch", "Entropy (bits)"), the title,
           a legend, and a light grid.
        4. Save to config.OUTPUT_DIR / filename at dpi=150 with
           bbox_inches="tight", then close the figure.

    Set the y-limit from 0 to slightly above the ceiling so the distance
    from the maximum is visible. Autoscaling hides exactly the thing the
    reader needs to see.
    """

    # ======================= ADD YOUR CODE HERE =======================
    epochs=[]
    means=[]
    stds=[]
    
    for entry in history:
        epochs.append(entry["epoch"])
        means.append(entry[f"{key}_mean"])
        stds.append(entry[f"{key}_std"])


    plt.figure()

    plt.errorbar(
        epochs,
        means,
        yerr=stds,
        capsize=3,
        marker="o",
        label=key,
    )

    plt.axhline(
        y=ceiling,
        linestyle="--",
        label=f"Maximum = {ceiling:.2f} bits",
    )

    plt.xlabel("Training epoch")
    plt.ylabel("Entropy (bits)")
    plt.title(title)
    plt.ylim(0, ceiling * 1.05)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        config.OUTPUT_DIR / filename,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()
    # ==================================================================


def caption_for(history, key: str, label: str) -> str:
    """Build the per-epoch caption string the handout asks for."""
    parts = [f"e{h['epoch']}: {h[f'{key}_mean']:.3f} +/- {h[f'{key}_std']:.3f}"
             for h in history]
    return (f"{label} at layer4 over training, measured on "
            f"{config.ENTROPY_SUBSET} fixed validation images. "
            f"Mean +/- standard deviation across images, per epoch: "
            + "; ".join(parts) + ".")


# ===========================================================================
# Loss and accuracy figures
# ===========================================================================

def plot_loss(history):
    """Training and validation loss on the same axes.

    Epoch 0 has no training loss, since nothing was trained yet, so filter
    those None values out of the training series before plotting. Plot
    both curves, label the axes, add a legend and a grid, save to
    config.OUTPUT_DIR / config.FIG_LOSS, and close the figure.
    """
    # ======================= ADD YOUR CODE HERE =======================
    train_epochs = []
    train_losses = []
    val_epochs = []
    val_losses = []
    
    for entry in history:
        if entry["train_loss"] is not None:
            train_epochs.append(entry["epoch"])
            train_losses.append(entry["train_loss"])
    
        val_epochs.append(entry["epoch"])
        val_losses.append(entry["val_loss"])

    plt.figure()

    plt.plot(
        train_epochs,
        train_losses,
        marker="o",
        label="Training loss",
    )

    plt.plot(
        val_epochs,
        val_losses,
        marker="o",
        label="Validation loss",
    )

    plt.xlabel("Training epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        config.OUTPUT_DIR / config.FIG_LOSS,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()
    # ==================================================================


def plot_accuracy(history):
    """Top-1 accuracy, training and validation, on the same axes.

    Same structure as plot_loss. Also draw a faint horizontal line at each
    value in config.THRESHOLDS, so the epoch-to-threshold table in your
    report can be read straight off the figure. Save to
    config.OUTPUT_DIR / config.FIG_ACCURACY.
    """
    # ======================= ADD YOUR CODE HERE =======================
    train_epochs = []
    train_accuracies = []
    val_epochs = []
    val_accuracies = []

    for entry in history:
        if entry["train_top1"] is not None:
            train_epochs.append(entry["epoch"])
            train_accuracies.append(entry["train_top1"])

        val_epochs.append(entry["epoch"])
        val_accuracies.append(entry["val_top1"])

    plt.figure()

    plt.plot(
        train_epochs,
        train_accuracies,
        marker="o",
        label="Training accuracy",
    )

    plt.plot(
        val_epochs,
        val_accuracies,
        marker="o",
        label="Validation accuracy",
    )

    for threshold in config.THRESHOLDS:
        plt.axhline(
            y=threshold,
            linestyle="--",
            color="gray",
            alpha=0.3,
        )

    plt.xlabel("Training epoch")
    plt.ylabel("Top-1 accuracy (%)")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(
        config.OUTPUT_DIR / config.FIG_ACCURACY,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()
    # ==================================================================


# ===========================================================================

def main():
    history = load_history()
    maxima = config.theoretical_maxima()

    plot_entropy(history, "joint",
                 "Joint entropy $S_{joint}$ at layer4",
                 config.FIG_JOINT, maxima["joint"])
    plot_entropy(history, "conditional",
                 "Conditional entropy $S_{conditional}$ at layer4",
                 config.FIG_CONDITIONAL, maxima["conditional"])
    plot_loss(history)
    plot_accuracy(history)

    print(f"figures written to {config.OUTPUT_DIR}\n")
    print("CAPTIONS, paste into your report:\n")
    print("Figure 1. " + caption_for(history, "joint", "Joint entropy") + "\n")
    print("Figure 2. " + caption_for(history, "conditional", "Conditional entropy") + "\n")

    # Chain rule check. S_joint should equal S_feature + S_conditional at
    # every epoch. A gap larger than about 0.01 bits means a bug.
    worst = max(abs(h["joint_mean"] - h["feature_mean"] - h["conditional_mean"])
                for h in history)
    status = "OK" if worst < 1e-2 else "FAILED, check your entropy code"
    print(f"chain rule check: largest gap {worst:.2e} bits -> {status}")


if __name__ == "__main__":
    main()

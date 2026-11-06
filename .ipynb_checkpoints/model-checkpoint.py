"""
Model construction and activation capture.

The architecture is the stock torchvision ResNet-18, unmodified except for
the classifier head. Do not change the stem, the blocks, or the strides;
the 7x7 spatial map at layer4 is what makes N = 49 in the entropy
calculation, and changing it would make your numbers incomparable to
everyone else's.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import models

import config


# ===========================================================================
# Model
# ===========================================================================

def build_model() -> nn.Module:
    """Create a ResNet-18 with randomly initialized weights.

    Two things to get right:
        1. No pretrained weights. The whole point is training from scratch,
           so pass weights=None.
        2. Replace the final fully connected layer so it outputs
           config.NUM_CLASSES logits instead of 1000. The input feature
           count is model.fc.in_features, which is 512; read it rather
           than hardcoding it.

    Returns:
        The model, on CPU. The caller moves it to the GPU.
    """
    # ======================= ADD YOUR CODE HERE =======================
    model=models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, config.NUM_CLASSES)
    return model
    # ==================================================================


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters. Report this in your writeup."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# Activation capture
# ===========================================================================

class ActivationCapture:
    """Forward hook that grabs the output of one named module.

    Used as a context manager so the hook is always removed, even if the
    body raises:

        with ActivationCapture(model, "layer4") as cap:
            model(images)
            feats = cap.value          # (B, 512, 7, 7)

    Why layer4 and not something more specific: the handout asks for the
    last convolutional layer's activations after the skip connection and
    normalization. The output of the layer4 module is exactly that tensor,
    taken after the final BasicBlock's residual add and its ReLU.
    """

    def __init__(self, model: nn.Module, layer_name: str = config.ENTROPY_LAYER):
        self.module = dict(model.named_modules())[layer_name]
        self.value: torch.Tensor | None = None
        self._handle = None

    def _hook(self, module, inputs, output):
        self.value = output.detach()

    def __enter__(self):
        self._handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False


def verify_activation_shape(model: nn.Module, device: torch.device) -> tuple:
    """Push one dummy image through and confirm the captured shape.

    Run this once before you start a long training job. Catching a wrong
    layer here costs you ten seconds; catching it after 50 epochs costs
    you an afternoon.

    Returns:
        The captured shape, which should be (1, 512, 7, 7).
    """
    model.eval()
    dummy = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE, device=device)
    with ActivationCapture(model) as cap, torch.no_grad():
        model(dummy)
        shape = tuple(cap.value.shape)

    expected = (1, config.D_CHANNELS, 7, 7)
    if shape != expected:
        raise ValueError(f"captured {shape}, expected {expected}")

    if cap.value.min().item() < 0:
        raise ValueError(
            "captured activations contain negative values, so you hooked a "
            "pre-ReLU tensor. Hook the layer4 module itself."
        )
    return shape

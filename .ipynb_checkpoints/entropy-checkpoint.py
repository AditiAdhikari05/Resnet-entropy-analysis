"""
Joint, feature, and conditional entropy of a convolutional layer.

This is the heart of the assignment. Read the handout definitions first,
then fill in the four functions below.

Notation, matching the handout
------------------------------
    d = 1..D    feature channel index
    j = 1..N    spatial position index within a channel (N = width * height)
    a(j, d)     activation value at node (j, d) for one input image

Activations become a probability distribution over the layer's D*N nodes
by taking magnitudes and normalizing:

    P(j, d) = |a(j, d)| / sum over all nodes of |a|

Three entropies follow from that one distribution:

    S_joint       = -sum_d sum_j P(j,d) log2 P(j,d)
    S_feature     = -sum_d P(d) log2 P(d)                  where P(d) = sum_j P(j,d)
    S_conditional = -sum_d sum_j P(j,d) log2 P(j|d)        where P(j|d) = P(j,d) / P(d)

They are related by the chain rule:

    S_joint = S_feature + S_conditional

which is the cheapest correctness check you have. Use it.
"""

import torch

from config import EPS


# ===========================================================================
# 1. Activations to probabilities
# ===========================================================================

def activations_to_probabilities(acts: torch.Tensor) -> torch.Tensor:
    """Turn raw activations into the joint distribution P(j, d).

    Args:
        acts: (B, D, N) float tensor of activation values.

    Returns:
        (B, D, N) tensor where, for each image independently, all D*N
        entries are non-negative and sum to 1.

    Notes:
        Take the magnitude first. At layer4 the values come after a ReLU so
        they are already non-negative and the absolute value does nothing,
        but write it anyway: the formula is defined with it, and it matters
        if you ever probe a pre-ReLU tensor.

        Normalize per image, not across the batch. Sum over BOTH the
        channel and position axes, keeping dimensions so the division
        broadcasts back.

        Add EPS to the denominator so an all-zero image cannot divide by
        zero.
    """
    # ======================= ADD YOUR CODE HERE =======================
    # 1. magnitude
    # 2. per-image total over dims (1, 2), keepdim=True
    # 3. divide
    magnitude = torch.abs(acts)
    total = magnitude.sum(dim=(1, 2), keepdim=True)
    P = magnitude / (total + EPS)

    return P
    # ==================================================================


# ===========================================================================
# 2. Joint entropy
# ===========================================================================

def joint_entropy(P: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of the full joint distribution, in bits.

    Args:
        P: (B, D, N) joint distribution from activations_to_probabilities.

    Returns:
        (B,) tensor, one entropy value per image.

    Notes:
        S_joint = -sum_d sum_j P(j,d) log2 P(j,d)

        Clamp the argument of the logarithm to a minimum of EPS. Nodes with
        exactly zero probability should contribute exactly zero to the sum,
        and 0 * log2(EPS) is a tiny negative number rather than NaN, which
        is the behaviour you want.

        Sum over dims (1, 2), which leaves the batch axis.
    """
    # ======================= ADD YOUR CODE HERE =======================
    log_P = torch.log2(P.clamp(min=EPS))
    entropy = -torch.sum(P * log_P, dim=(1, 2))
    return entropy
    # ==================================================================


# ===========================================================================
# 3. Feature (channel) entropy
# ===========================================================================

def feature_entropy(P: torch.Tensor) -> torch.Tensor:
    """Entropy of the channel marginal P(d), in bits.

    Args:
        P: (B, D, N) joint distribution.

    Returns:
        (B,) tensor, one value per image.

    Notes:
        First marginalize out the spatial axis to get P(d), of shape
        (B, D): sum P(j,d) over j. Each row still sums to 1.

        Then take the ordinary Shannon entropy of that D-way distribution.

        Interpretation: how evenly the layer's total activation mass is
        spread ACROSS channels. Near 0 means one channel holds everything.
        Near log2(512) = 9 bits means all channels contribute equally.
    """
    # ======================= ADD YOUR CODE HERE =======================
    P_d = P.sum(dim=2) #along the rows or per channel
    P_d_safe = P_d.clamp_min(EPS)
    entropy = -(P_d * torch.log2(P_d_safe)).sum(dim=1)
    return entropy
    # ==================================================================


# ===========================================================================
# 4. Conditional entropy
# ===========================================================================

def conditional_entropy(P: torch.Tensor) -> torch.Tensor:
    """Conditional entropy of position given channel, in bits.

    Args:
        P: (B, D, N) joint distribution.

    Returns:
        (B,) tensor, one value per image.

    Notes:
        S_conditional = -sum_d sum_j P(j,d) log2 P(j|d)

        with P(j|d) = P(j,d) / P(d), where P(d) is the channel marginal.

        Regrouping the same expression gives a more readable form that is
        worth holding in your head:

            S_conditional = sum_d P(d) * H_d

        where H_d is the spatial entropy of channel d's own normalized
        7x7 map. So it is the average spatial spread of a channel,
        weighted by how much mass that channel holds. Dead channels have
        P(d) near zero and drop out of the average on their own.

        Either form is fine to implement. Watch two things:

        - A channel with zero mass makes P(j|d) genuinely undefined.
          Clamp the denominator with EPS, and note the P(j,d) weight in
          front is zero for those terms anyway, so they contribute nothing.
        - Clamp the log argument with EPS as before.

        Interpretation: near 0 means each active channel fires at only a
        few of the 49 positions. Near log2(49) = 5.61 bits means each
        active channel responds roughly uniformly across the whole map.
    """
    # ======================= ADD YOUR CODE HERE =======================
    P_d=P.sum(dim=2, keepdim=True)
    P_jd= P/P_d.clamp_min(EPS)
    P_jd_safe=P_jd.clamp_min(EPS)
    entropy=-(P*torch.log2(P_jd_safe)).sum(dim=(1,2))
    return entropy

    # ==================================================================


# ===========================================================================
# Convenience wrapper. Already written, nothing to do here.
# ===========================================================================

def all_entropies(acts: torch.Tensor) -> dict[str, torch.Tensor]:
    """Compute all three entropies for a batch of activations.

    Args:
        acts: (B, D, N) activations, or (B, D, H, W) which gets flattened.

    Returns:
        dict with keys "joint", "feature", "conditional", each a (B,)
        tensor of per-image values.
    """
    if acts.dim() == 4:
        acts = acts.flatten(2)          # (B, D, H, W) -> (B, D, H*W)
    if acts.dim() != 3:
        raise ValueError(f"expected 3 or 4 dims, got shape {tuple(acts.shape)}")

    # Entropy in half precision loses far too much, so force float32 even
    # when the forward pass ran under autocast.
    acts = acts.float()

    P = activations_to_probabilities(acts)
    return {
        "joint": joint_entropy(P),
        "feature": feature_entropy(P),
        "conditional": conditional_entropy(P),
    }

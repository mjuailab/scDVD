"""Population losses; GeomLoss energy is the paper objective.

Legacy energy and RBF MMD functions remain available for historical checkpoints.
"""
from __future__ import annotations

import torch

def positive_expression_mse(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Population-wise MSE over measured positive target entries."""
    support = target > 0
    squared_error = (prediction - target).square()
    return (squared_error * support).sum((1, 2)) / support.sum((1, 2)).clamp_min(1)


def support_weighted_expression_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    zero_weight: float,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Population-wise MSE with an optional weak loss on target-zero entries."""
    if not 0.0 <= zero_weight <= 1.0:
        raise ValueError("zero_expression_weight must be in [0, 1]")
    positive = target > 0
    weights = positive.to(prediction.dtype)
    if zero_weight > 0:
        weights = weights + (~positive).to(prediction.dtype) * zero_weight
    if valid is not None:
        weights = weights * valid.to(weights.dtype).unsqueeze(-1)
    squared_error = (prediction - target).square()
    return (squared_error * weights).sum((1, 2)) / weights.sum((1, 2)).clamp_min(1)


def geomloss_energy_distance(
    target: torch.Tensor, prediction: torch.Tensor,
    *, valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """PerturbDiff's GeomLoss energy MMD, with optional padding weights."""
    from geomloss import SamplesLoss

    loss_fn = SamplesLoss(loss="energy", blur=0.05)
    target, prediction = target.float(), prediction.float()
    if valid is None:
        return loss_fn(target, prediction)
    weights = valid.to(target.dtype)
    if (weights.sum(-1) == 0).any():
        raise ValueError("Every population must contain a valid cell")
    weights = weights / weights.sum(-1, keepdim=True)
    return loss_fn(weights, target, weights, prediction)


def raw_energy_distance(
    target: torch.Tensor, prediction: torch.Tensor,
    *, valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Biased Energy Distance in the original expression space [B,S,G]."""
    if valid is not None:
        mask = valid.to(device=target.device, dtype=torch.float32)
        pair = mask[:, :, None] * mask[:, None, :]
        denominator = pair.sum((1, 2)).clamp_min(1.0)
        average = lambda distances: (distances * pair).sum((1, 2)) / denominator
        cross = average(torch.cdist(target.float(), prediction.float()))
        target_self = average(torch.cdist(target.float(), target.float()))
        prediction_self = average(torch.cdist(prediction.float(), prediction.float()))
        return (2.0 * cross - target_self - prediction_self).clamp_min(0.0)
    cross = torch.cdist(target.float(), prediction.float()).mean((1, 2))
    target_self = torch.cdist(target.float(), target.float()).mean((1, 2))
    prediction_self = torch.cdist(
        prediction.float(), prediction.float()
    ).mean((1, 2))
    return (2.0 * cross - target_self - prediction_self).clamp_min(0.0)


def multiscale_rbf_mmd(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    bandwidth_scales: tuple[float, ...] | list[float],
    eps: float = 1e-8,
    valid: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Biased multi-scale RBF MMD^2 for populations shaped [B,S,G]."""
    target = target.float()
    prediction = prediction.float()
    if target.ndim != 3 or prediction.shape != target.shape:
        raise ValueError("MMD populations must share shape [B,S,G]")
    set_size = target.shape[1]
    target_squared = torch.cdist(target, target).square()
    prediction_squared = torch.cdist(prediction, prediction).square()
    cross_squared = torch.cdist(target, prediction).square()
    off_diagonal = ~torch.eye(
        set_size, device=target.device, dtype=torch.bool
    )
    if valid is None:
        pair = torch.ones(
            target.shape[0], set_size, set_size, device=target.device
        )
    else:
        mask = valid.to(target.dtype)
        pair = mask[:, :, None] * mask[:, None, :]
    keep = pair > 0
    pool_entries = [
        torch.where(
            keep[:, off_diagonal].bool(),
            block[:, off_diagonal],
            torch.full_like(block[:, off_diagonal], float("nan")),
        )
        for block in (target_squared, prediction_squared)
    ]
    pool_entries.append(
        torch.where(
            keep.flatten(1).bool(),
            cross_squared.flatten(1),
            torch.full_like(cross_squared.flatten(1), float("nan")),
        )
    )
    distance_pool = torch.cat(pool_entries, dim=1)
    base_squared = (
        torch.nanmedian(distance_pool, dim=1).values.detach().clamp_min(eps)
    )

    kernel_xx = torch.zeros_like(target_squared)
    kernel_yy = torch.zeros_like(prediction_squared)
    kernel_xy = torch.zeros_like(cross_squared)
    scales = tuple(float(scale) for scale in bandwidth_scales)
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("MMD bandwidth scales must be positive")
    for scale in scales:
        denominator = 2.0 * base_squared[:, None, None] * scale**2
        kernel_xx = kernel_xx + torch.exp(-target_squared / denominator)
        kernel_yy = kernel_yy + torch.exp(-prediction_squared / denominator)
        kernel_xy = kernel_xy + torch.exp(-cross_squared / denominator)
    denominator_pairs = pair.sum((1, 2)).clamp_min(1.0)
    weighted = lambda kernel: (kernel * pair).sum((1, 2)) / denominator_pairs
    mmd_squared = (
        weighted(kernel_xx) + weighted(kernel_yy) - 2.0 * weighted(kernel_xy)
    ) / len(scales)
    return mmd_squared.clamp_min(0.0), base_squared.sqrt()

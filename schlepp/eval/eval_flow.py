# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Optical-flow End-Point Error metric for SCHLEPP."""
from __future__ import annotations

from typing import Dict, Literal, Optional, Sequence, Tuple

import torch

Reduction = Literal["mean", "sum", "none"]


def flow_validity_mask_from_depth(depth: torch.Tensor) -> torch.Tensor:
    """Return ``(depth > 0)`` for ``.dpt5`` depth maps.

    The dataset writes ``0.0`` to depth pixels where ground truth is
    unreliable (e.g. transparent or non-rendered surfaces). Using this
    helper as the ``mask`` argument to :func:`end_point_error` ensures
    those regions never contribute to the loss.

    Accepts a depth tensor with any leading shape; returns a bool tensor
    of the same shape.
    """
    return depth > 0


def end_point_error(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Standard End-Point Error for optical flow.

    EPE(u, v) = sqrt((du_pred - du_gt)^2 + (dv_pred - dv_gt)^2).

    Parameters
    ----------
    pred_flow, gt_flow
        Shape ``(..., 2, H, W)`` with the channel axis at position
        ``-3``; ``(du, dv)`` in pixel units.
    mask
        Optional boolean mask of shape ``(..., H, W)`` (or broadcastable
        to that). Pixels where ``mask`` is ``False`` are dropped before
        reduction; pass :func:`flow_validity_mask_from_depth` on the
        corresponding depth map to ignore invalid pixels.
    reduction
        ``"mean"`` (default) averages over valid pixels,
        ``"sum"`` sums over valid pixels,
        ``"none"`` returns the per-pixel EPE tensor.

    Returns
    -------
    torch.Tensor
        Scalar tensor for ``"mean"`` / ``"sum"``, ``(..., H, W)`` for
        ``"none"``.
    """
    if pred_flow.shape != gt_flow.shape:
        raise ValueError(
            f"pred_flow shape {pred_flow.shape} != gt_flow shape {gt_flow.shape}"
        )
    if pred_flow.shape[-3] != 2:
        raise ValueError(
            f"flow tensors must have 2 channels at axis -3; got {pred_flow.shape}"
        )
    delta = pred_flow - gt_flow                         # shape: (..., 2, H, W)
    epe = torch.sqrt((delta * delta).sum(dim=-3))       # shape: (..., H, W)
    if reduction == "none":
        if mask is not None:
            epe = torch.where(mask, epe, torch.full_like(epe, float("nan")))
        return epe
    if mask is None:
        if reduction == "mean":
            return epe.mean()
        return epe.sum()
    flat_mask = mask.expand_as(epe)
    if reduction == "sum":
        return (epe * flat_mask).sum()
    n_valid = flat_mask.sum()
    if n_valid.item() == 0:
        return torch.zeros((), dtype=epe.dtype, device=epe.device)
    return (epe * flat_mask).sum() / n_valid


# ---------------------------------------------------------------------------
# Shared building blocks. Both fl_all and epe_by_motion need the same
# per-pixel EPE and gt-magnitude maps with the same validity convention,
# so we factor it out.
# ---------------------------------------------------------------------------


def _epe_and_gt_mag(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    mask: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(epe_map, gt_magnitude_map, valid_map)``.

    All three are ``(..., H, W)``. ``valid_map`` is True where the pixel
    contributes to a downstream aggregate (i.e., user-supplied ``mask``
    is True if given, otherwise everywhere). The EPE itself is *not*
    NaN-masked (callers usually want both the raw value and the explicit
    mask, e.g. for Fl-all's relative-threshold check).
    """
    epe = end_point_error(pred_flow, gt_flow, mask=None, reduction="none")
    gt_mag = torch.linalg.vector_norm(gt_flow, dim=-3)
    if mask is None:
        valid = torch.ones_like(epe, dtype=torch.bool)
    else:
        valid = mask.expand_as(epe).to(torch.bool)
    return epe, gt_mag, valid


def fl_all(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    abs_thr: float = 3.0,
    rel_thr: float = 0.05,
) -> torch.Tensor:
    """KITTI-style flow outlier fraction.

    A pixel is an outlier iff ``EPE > abs_thr`` AND
    ``EPE > rel_thr * |gt_flow|``. Returns the fraction of outliers
    among ``mask`` (or all pixels if ``mask`` is None).
    """
    epe, gt_mag, valid = _epe_and_gt_mag(pred_flow, gt_flow, mask)
    bad = (epe > abs_thr) & (epe > rel_thr * gt_mag) & valid
    denom = valid.to(torch.float32).sum().clamp_min(1.0)
    return bad.to(torch.float32).sum() / denom


def epe_by_motion(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    *,
    bins: Sequence[Tuple[float, float]] = (
        (0.0, 10.0), (10.0, 40.0), (40.0, float("inf")),
    ),
    bin_names: Optional[Sequence[str]] = None,
) -> Dict[str, torch.Tensor]:
    """Sintel-style EPE stratified by ground-truth flow magnitude.

    For each ``(lo, hi)`` bin, returns the mean EPE over pixels whose
    ground-truth magnitude is in ``[lo, hi)`` and ``mask`` is True.
    Empty bins return ``0.0`` (rather than NaN) so the dict is easy to
    log straight to W&B / TensorBoard.

    ``bin_names`` lets you choose dict keys; defaults to
    ``f"epe_{lo}_{hi}"``.
    """
    epe, gt_mag, valid = _epe_and_gt_mag(pred_flow, gt_flow, mask)
    names = list(bin_names) if bin_names is not None else [
        f"epe_{lo}_{hi}" for lo, hi in bins
    ]
    out: Dict[str, torch.Tensor] = {}
    for (lo, hi), name in zip(bins, names):
        in_bin = valid & (gt_mag >= float(lo)) & (gt_mag < float(hi))
        denom = in_bin.to(torch.float32).sum()
        if denom.item() == 0:
            out[name] = torch.zeros((), dtype=epe.dtype, device=epe.device)
        else:
            out[name] = (epe * in_bin.to(epe.dtype)).sum() / denom
    return out


def flow_summary(
    pred_flow: torch.Tensor,
    gt_flow: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    depth: Optional[torch.Tensor] = None,
    bins: Sequence[Tuple[float, float]] = (
        (0.0, 10.0), (10.0, 40.0), (40.0, float("inf")),
    ),
    abs_thr: float = 3.0,
    rel_thr: float = 0.05,
) -> Dict[str, torch.Tensor]:
    """One-shot ``{"epe", "fl_all", "epe_per_motion"}`` summary.

    If ``mask`` is None but ``depth`` is given, the validity mask is
    derived as ``depth > 0`` (matches the schlepp ``.dpt5`` invalid
    sentinel). If both are None, every pixel contributes.
    """
    if mask is None and depth is not None:
        mask = flow_validity_mask_from_depth(depth)
    out: Dict[str, torch.Tensor] = {
        "epe":    end_point_error(pred_flow, gt_flow, mask=mask, reduction="mean"),
        "fl_all": fl_all(pred_flow, gt_flow, mask=mask,
                         abs_thr=abs_thr, rel_thr=rel_thr),
    }
    out.update(epe_by_motion(pred_flow, gt_flow, mask=mask, bins=bins))
    return out

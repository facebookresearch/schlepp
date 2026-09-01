# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Point-track quality metrics for SCHLEPP.

Both metrics gate by ``valid & visible``:

* ``valid``   = the point is in front of the camera AND inside the image
                bounds (on-disk ``mv_valids``).
* ``visible`` = the point is not occluded by another mesh (on-disk
                ``mv_visibs``).

A tracker should never be penalised for losing a point that the ground
truth marks as occluded, so we drop those positions before computing
per-track error.
"""
from __future__ import annotations

from typing import Dict, Literal, Optional, Sequence, Tuple

import torch

Reduction = Literal["mean", "sum", "per_point"]


def _per_frame_error(
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    scale_xy: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-frame Euclidean error, optionally with per-axis pre-scaling.

    Shape ``(..., N, S, 2) -> (..., N, S)``. ``scale_xy`` is a
    broadcastable ``(2,)`` tensor of (x_scale, y_scale) -- used to
    normalise into the TAP-Vid reference resolution before computing
    the L2 norm (per-axis scale is *not* equivalent to a scalar on
    the norm, hence the explicit pre-multiply).
    """
    delta = pred - gt
    if scale_xy is not None:
        delta = delta * scale_xy
    return torch.linalg.vector_norm(delta, dim=-1)


def _per_track_mean_error(
    pred: torch.Tensor,
    gt: torch.Tensor,
    eval_mask: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Per-track Euclidean error averaged over kept frames.

    Returns ``(..., N)`` with NaN where a track has zero kept frames.
    """
    err = _per_frame_error(pred, gt)                    # shape: (..., N, S)
    kept = eval_mask.to(err.dtype)                      # shape: (..., N, S)
    n = kept.sum(dim=-1)                                # shape: (..., N)
    summed = (err * kept).sum(dim=-1)                   # shape: (..., N)
    out = summed / torch.where(n < eps, torch.full_like(n, eps), n)
    out = torch.where(n > 0, out, torch.full_like(out, float("nan")))
    return out


def _resolve_scale_xy(
    image_size: Optional[Tuple[int, int]],
    reference_size: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """``(W_ref / W_img, H_ref / H_img)`` or ``None`` to keep native scale.

    ``image_size`` is ``(H, W)`` to match the rest of the schlepp API.
    """
    if image_size is None:
        return None
    H, W = int(image_size[0]), int(image_size[1])
    rH, rW = int(reference_size[0]), int(reference_size[1])
    return torch.tensor([rW / W, rH / H], dtype=dtype, device=device)


def average_trajectory_error(
    pred_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    valid: torch.Tensor,
    visible: torch.Tensor,
    *,
    reduction: Reduction = "mean",
) -> torch.Tensor:
    """Average Trajectory Error (ATE) over a batch of point tracks.

    Parameters
    ----------
    pred_tracks, gt_tracks
        Shape ``(..., N, S, 2)`` predicted and ground-truth pixel
        positions.
    valid, visible
        Shape ``(..., N, S)`` boolean masks; only positions where
        ``valid & visible`` are counted.
    reduction
        * ``"mean"``      -- mean over tracks of per-track mean Euclidean error.
        * ``"sum"``       -- sum of valid-and-visible per-frame errors.
        * ``"per_point"`` -- the ``(..., N)`` per-track mean error tensor.

    Returns
    -------
    torch.Tensor
        Scalar for ``"mean"`` / ``"sum"``, ``(..., N)`` for ``"per_point"``.
    """
    if pred_tracks.shape != gt_tracks.shape:
        raise ValueError(
            f"pred shape {pred_tracks.shape} != gt shape {gt_tracks.shape}"
        )
    if pred_tracks.shape[-1] != 2:
        raise ValueError(
            f"track tensors must have last dim 2; got {pred_tracks.shape}"
        )
    eval_mask = valid & visible                         # shape: (..., N, S)

    if reduction == "sum":
        err = _per_frame_error(pred_tracks, gt_tracks)
        return (err * eval_mask.to(err.dtype)).sum()

    per_track = _per_track_mean_error(pred_tracks, gt_tracks, eval_mask)
    if reduction == "per_point":
        return per_track
    # mean: skip tracks that had zero kept frames (NaN guard).
    finite = torch.isfinite(per_track)
    if not finite.any():
        return torch.zeros((), dtype=per_track.dtype, device=per_track.device)
    return per_track[finite].mean()


def survival_rate(
    pred_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    valid: torch.Tensor,
    visible: torch.Tensor,
    *,
    threshold: float = 50.0,
) -> torch.Tensor:
    """Fraction of tracks whose per-frame error never exceeds ``threshold``.

    A track "survives" iff every frame where ``valid & visible`` keeps
    its Euclidean pixel error below ``threshold`` pixels. Tracks that
    have zero kept frames are counted as dead (they cannot contribute
    evidence of survival).

    Returns a scalar tensor in ``[0, 1]``.
    """
    if pred_tracks.shape != gt_tracks.shape:
        raise ValueError(
            f"pred shape {pred_tracks.shape} != gt shape {gt_tracks.shape}"
        )
    eval_mask = valid & visible                         # shape: (..., N, S)
    err = _per_frame_error(pred_tracks, gt_tracks)      # shape: (..., N, S)
    fail = eval_mask & (err >= threshold)               # shape: (..., N, S)
    any_eval = eval_mask.any(dim=-1)                    # shape: (..., N)
    survives = (~fail.any(dim=-1)) & any_eval           # shape: (..., N)
    total = any_eval.to(survives.dtype).sum()
    if total.item() == 0:
        return torch.zeros((), dtype=err.dtype, device=err.device)
    return survives.to(err.dtype).sum() / total


# ---------------------------------------------------------------------------
# TAP-Vid-style metrics. Both delta_avg and average_jaccard accept an
# optional ``image_size`` so errors are computed at the TAP-Vid 256x256
# reference (the standard published numbers are at that resolution).
# ---------------------------------------------------------------------------


_TAPVID_THRESHOLDS: Tuple[int, ...] = (1, 2, 4, 8, 16)


def _delta_at_thresholds(
    err: torch.Tensor,
    eval_mask: torch.Tensor,
    thresholds: Sequence[float],
) -> torch.Tensor:
    """Per-threshold fraction of (track, frame) positions with err < thr.

    Shared between :func:`delta_avg` and :func:`average_jaccard`.
    Returns ``(len(thresholds),)``.
    """
    eval_sum = eval_mask.to(err.dtype).sum().clamp_min(1.0)
    out = err.new_zeros(len(thresholds))
    for i, thr in enumerate(thresholds):
        within = (err < float(thr)) & eval_mask
        out[i] = within.to(err.dtype).sum() / eval_sum
    return out


def delta_avg(
    pred_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    valid: torch.Tensor,
    visible: torch.Tensor,
    *,
    thresholds: Sequence[float] = _TAPVID_THRESHOLDS,
    image_size: Optional[Tuple[int, int]] = None,
    reference_size: Tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """TAP-Vid ``delta_avg`` (a.k.a. ``< δ_avg``).

    Mean over ``thresholds`` of the fraction of evaluated (track, frame)
    positions where the predicted pixel error is below the threshold.

    ``image_size`` should be the native ``(H, W)`` of ``pred_tracks`` /
    ``gt_tracks``. If given, errors are rescaled into the TAP-Vid 256x256
    reference (per-axis), so the published threshold values
    ``[1, 2, 4, 8, 16]`` carry their standard meaning.
    """
    if pred_tracks.shape != gt_tracks.shape:
        raise ValueError(
            f"pred shape {pred_tracks.shape} != gt shape {gt_tracks.shape}"
        )
    scale_xy = _resolve_scale_xy(image_size, reference_size,
                                 pred_tracks.device, pred_tracks.dtype)
    err = _per_frame_error(pred_tracks, gt_tracks, scale_xy=scale_xy)
    eval_mask = valid & visible
    deltas = _delta_at_thresholds(err, eval_mask, thresholds)
    return deltas.mean()


def average_jaccard(
    pred_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    valid: torch.Tensor,
    visible: torch.Tensor,
    *,
    pred_visible: Optional[torch.Tensor] = None,
    thresholds: Sequence[float] = _TAPVID_THRESHOLDS,
    image_size: Optional[Tuple[int, int]] = None,
    reference_size: Tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """TAP-Vid Average Jaccard.

    For each threshold:

    * **TP**  -- ``visible_gt & pred_visible & (err < thr)``
    * **FN**  -- ``visible_gt & ~(pred_visible & (err < thr))``
    * **FP**  -- ``~visible_gt & pred_visible``

    AJ at threshold = ``TP / (TP + FP + FN)``; the metric averages over
    thresholds. Positions where ``valid`` is False are excluded from
    every term.

    ``pred_visible`` is the model's predicted visibility (bool, same shape
    as ``visible``). If omitted, the model is treated as predicting
    "always visible" (which is the right behaviour for trackers that
    don't emit a visibility head, but inflates FP).
    """
    if pred_tracks.shape != gt_tracks.shape:
        raise ValueError(
            f"pred shape {pred_tracks.shape} != gt shape {gt_tracks.shape}"
        )
    if pred_visible is None:
        pred_visible = torch.ones_like(visible, dtype=torch.bool)
    scale_xy = _resolve_scale_xy(image_size, reference_size,
                                 pred_tracks.device, pred_tracks.dtype)
    err = _per_frame_error(pred_tracks, gt_tracks, scale_xy=scale_xy)

    ajs = err.new_zeros(len(thresholds))
    for i, thr in enumerate(thresholds):
        within = err < float(thr)
        tp = valid & visible & pred_visible & within
        fn = valid & visible & ~(pred_visible & within)
        fp = valid & ~visible & pred_visible
        denom = (tp | fn | fp).to(err.dtype).sum().clamp_min(1.0)
        ajs[i] = tp.to(err.dtype).sum() / denom
    return ajs.mean()


# ---------------------------------------------------------------------------
# Visibility metrics
# ---------------------------------------------------------------------------


def occlusion_accuracy(
    pred_visible: torch.Tensor,
    gt_visible: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Binary accuracy of visibility prediction over ``valid`` positions."""
    correct = (pred_visible == gt_visible) & valid
    denom = valid.to(torch.float32).sum().clamp_min(1.0)
    return correct.to(torch.float32).sum() / denom


def occlusion_auc(
    pred_visible_score: torch.Tensor,
    gt_visible: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """AUROC of a continuous visibility score vs the ground-truth bool.

    Higher score = "more confident the point is visible". Positions
    where ``valid`` is False are excluded. Returns NaN if either class
    is empty (AUC undefined).
    """
    s = pred_visible_score[valid].reshape(-1).to(torch.float32)
    y = gt_visible[valid].reshape(-1).to(torch.float32)
    pos_total = y.sum()
    neg_total = (1.0 - y).sum()
    if pos_total.item() == 0 or neg_total.item() == 0:
        return torch.tensor(float("nan"))
    order = s.argsort(descending=True)
    y_ord = y[order]
    tpr = y_ord.cumsum(0) / pos_total
    fpr = (1.0 - y_ord).cumsum(0) / neg_total
    # ROC starts at (0, 0).
    zero = tpr.new_zeros(1)
    tpr = torch.cat([zero, tpr])
    fpr = torch.cat([zero, fpr])
    return torch.trapz(tpr, fpr)


# ---------------------------------------------------------------------------
# One-shot summary
# ---------------------------------------------------------------------------


def tap_vid_metrics(
    pred_tracks: torch.Tensor,
    gt_tracks: torch.Tensor,
    valid: torch.Tensor,
    visible: torch.Tensor,
    *,
    pred_visible: Optional[torch.Tensor] = None,
    pred_visible_score: Optional[torch.Tensor] = None,
    thresholds: Sequence[float] = _TAPVID_THRESHOLDS,
    image_size: Optional[Tuple[int, int]] = None,
    reference_size: Tuple[int, int] = (256, 256),
) -> Dict[str, torch.Tensor]:
    """Compute the canonical TAP-Vid suite in one call.

    Returns the dict ``{"delta_avg", "aj", "oa", "auc"}``. ``oa`` is
    omitted when ``pred_visible`` is None; ``auc`` is omitted when
    ``pred_visible_score`` is None.
    """
    out: Dict[str, torch.Tensor] = {
        "delta_avg": delta_avg(
            pred_tracks, gt_tracks, valid, visible,
            thresholds=thresholds,
            image_size=image_size, reference_size=reference_size,
        ),
        "aj": average_jaccard(
            pred_tracks, gt_tracks, valid, visible,
            pred_visible=pred_visible,
            thresholds=thresholds,
            image_size=image_size, reference_size=reference_size,
        ),
    }
    if pred_visible is not None:
        out["oa"] = occlusion_accuracy(pred_visible, visible, valid)
    if pred_visible_score is not None:
        out["auc"] = occlusion_auc(pred_visible_score, visible, valid)
    return out

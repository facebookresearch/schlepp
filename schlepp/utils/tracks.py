# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Point-track filtering / subsampling / partitioning utilities.

These helpers operate on the ``point_tracks`` dict that
:class:`schlepp.SchleppDataset` puts on every sample (see the dataset
README), and return a new ``point_tracks`` dict with the same structure
but a (possibly) smaller track axis ``N``. All track-axis fields move
in lockstep -- including the shared ``trajs_world`` and ``categories``
and every per-camera ``trajs_2d_pix[c]``, ``visible[c]``,
``in_frustum[c]``.

Composing inside a user transform::

    pt = filter_tracks_by_category(sample["point_tracks"], keep=("body", "carried"))
    pt = filter_tracks_by_visibility(pt, "body_follow", min_visible_frames=8)
    pt = subsample_tracks(pt, N=512, mode="stratified",
                          per_category={"body": 256, "carried": 256})
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import torch

from schlepp.categories import (
    ALL_CATEGORIES,
    CATEGORY_NAMES,
    resolve_categories,
    resolve_category,
)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PointTracks = Mapping[str, Any]
# ``point_tracks`` dict as emitted by SchleppDataset:
#   {
#     "trajs_world":  (N, S, 3) float,
#     "categories":   (N,)      int32  -- optional in principle, but the
#                                          loader always provides it
#     "trajs_2d_pix": {cam: (N, S, 2) float},
#     "visible":      {cam: (N, S)    bool},
#     "in_frustum":   {cam: (N, S)    bool},
#   }

# Top-level (N,)-axis arrays.
_TOP_LEVEL_TRACK_KEYS: Tuple[str, ...] = ("trajs_world", "categories")
# Per-camera dicts whose values have (N,)-axis arrays.
_PER_CAM_TRACK_KEYS: Tuple[str, ...] = ("trajs_2d_pix", "visible", "in_frustum")

CategorySpec = Union[int, str, Iterable[Union[int, str]]]
IndexLike = Union[torch.Tensor, Sequence[int]]


# ---------------------------------------------------------------------------
# Internal helpers (the single source of truth for "slice the N axis").
# ---------------------------------------------------------------------------


def _index_tracks(pt: PointTracks, idx: IndexLike) -> Dict[str, Any]:
    """Return a new point_tracks dict with the N-axis indexed by ``idx``.

    ``idx`` may be a 1-D bool mask or an int index tensor/sequence. This
    is the universal slicer used by every public filter/subsample below.
    """
    if not isinstance(idx, torch.Tensor):
        idx = torch.as_tensor(idx)
    out: Dict[str, Any] = {**pt}
    for key in _TOP_LEVEL_TRACK_KEYS:
        if pt.get(key) is not None:
            out[key] = pt[key][idx]
    for parent in _PER_CAM_TRACK_KEYS:
        if parent in pt:
            out[parent] = {cam: v[idx] for cam, v in pt[parent].items()}
    return out


def _track_count(pt: PointTracks) -> int:
    """N (track count) inferred from whichever track-axis array exists."""
    for key in _TOP_LEVEL_TRACK_KEYS:
        if pt.get(key) is not None:
            return int(pt[key].shape[0])
    for parent in _PER_CAM_TRACK_KEYS:
        d = pt.get(parent, {})
        for v in d.values():
            return int(v.shape[0])
    raise ValueError("point_tracks dict has no track-axis fields")


def _require_camera(pt: PointTracks, camera: str, field: str) -> torch.Tensor:
    """Pull ``pt[field][camera]`` with a friendly error if missing."""
    d = pt.get(field)
    if d is None or camera not in d:
        have = list((pt.get(field) or {}).keys())
        raise KeyError(
            f"point_tracks.{field} does not contain camera {camera!r}; "
            f"have {have}. Did you pass it via SchleppDataset(cameras=...)?"
        )
    return d[camera]


def _categories(pt: PointTracks) -> torch.Tensor:
    """Pull ``pt['categories']`` with a friendly error if missing."""
    cats = pt.get("categories")
    if cats is None:
        raise KeyError(
            "point_tracks dict has no 'categories' field; this only "
            "happens if the dataset was constructed without point_tracks "
            "or with an old loader."
        )
    return cats


# ---------------------------------------------------------------------------
# Filters: each computes a bool mask, then delegates to _index_tracks.
# ---------------------------------------------------------------------------


def filter_tracks_by_category(
    pt: PointTracks,
    *,
    keep: Optional[CategorySpec] = None,
    drop: Optional[CategorySpec] = None,
) -> Dict[str, Any]:
    """Keep tracks whose category is in ``keep`` and not in ``drop``.

    Category specifiers may be integer IDs (e.g. ``schlepp.categories.BODY``),
    name strings (``"body"``, ``"carried"``), or iterables of either.
    """
    if keep is None and drop is None:
        return {**pt}
    cats = _categories(pt)
    mask = torch.ones_like(cats, dtype=torch.bool)
    if keep is not None:
        keep_ids = resolve_categories(keep)
        keep_tensor = cats.new_tensor(keep_ids)
        mask &= (cats[:, None] == keep_tensor[None, :]).any(dim=1)
    if drop is not None:
        drop_ids = resolve_categories(drop)
        drop_tensor = cats.new_tensor(drop_ids)
        mask &= ~(cats[:, None] == drop_tensor[None, :]).any(dim=1)
    return _index_tracks(pt, mask)


def filter_tracks_by_visibility(
    pt: PointTracks,
    camera: str,
    *,
    min_visible_frames: int = 2,
    require_in_frustum: bool = True,
) -> Dict[str, Any]:
    """Drop tracks visible in fewer than ``min_visible_frames`` frames of ``camera``.

    By default a frame counts only if the point is both ``visible`` (not
    occluded) AND ``in_frustum`` (in front of the camera AND inside the
    image). Set ``require_in_frustum=False`` to count any non-occluded
    frame, regardless of frustum.
    """
    visible = _require_camera(pt, camera, "visible")
    if require_in_frustum:
        in_frustum = _require_camera(pt, camera, "in_frustum")
        eval_ = visible & in_frustum
    else:
        eval_ = visible
    counts = eval_.to(torch.int32).sum(dim=-1)               # (N,)
    mask = counts >= int(min_visible_frames)
    return _index_tracks(pt, mask)


def filter_tracks_by_motion(
    pt: PointTracks,
    camera: str,
    *,
    min_displacement_px: Optional[float] = None,
    max_displacement_px: Optional[float] = None,
    agg: str = "mean",
) -> Dict[str, Any]:
    """Filter tracks by per-frame pixel displacement magnitude in ``camera``.

    Displacement is computed between consecutive frames where the track
    is visible & in-frustum in both endpoints (gaps are skipped, not
    interpolated). ``agg`` ∈ {"mean", "max"} controls how per-frame
    displacements are aggregated into a per-track scalar.
    """
    if agg not in ("mean", "max"):
        raise ValueError(f"agg must be 'mean' or 'max'; got {agg!r}")
    if min_displacement_px is None and max_displacement_px is None:
        return {**pt}

    xy = _require_camera(pt, camera, "trajs_2d_pix")        # (N, S, 2)
    visible = _require_camera(pt, camera, "visible")        # (N, S)
    in_frustum = _require_camera(pt, camera, "in_frustum")  # (N, S)
    eval_ = visible & in_frustum                            # (N, S)

    delta = xy[:, 1:] - xy[:, :-1]                          # (N, S-1, 2)
    step_mag = torch.linalg.vector_norm(delta, dim=-1)      # (N, S-1)
    step_valid = eval_[:, 1:] & eval_[:, :-1]               # (N, S-1)

    if agg == "mean":
        denom = step_valid.to(step_mag.dtype).sum(dim=-1).clamp_min(1.0)
        per_track = (step_mag * step_valid).sum(dim=-1) / denom
    else:  # max
        per_track = torch.where(
            step_valid, step_mag, step_mag.new_full((), float("-inf"))
        ).amax(dim=-1)
        # Tracks with no valid step → -inf; treat as 0 (no motion observed).
        per_track = torch.where(
            torch.isfinite(per_track), per_track, per_track.new_zeros(())
        )

    mask = torch.ones_like(per_track, dtype=torch.bool)
    if min_displacement_px is not None:
        mask &= per_track >= float(min_displacement_px)
    if max_displacement_px is not None:
        mask &= per_track <= float(max_displacement_px)
    return _index_tracks(pt, mask)


# ---------------------------------------------------------------------------
# Subsampling. All modes resolve to "pick K integer indices" then index.
# ---------------------------------------------------------------------------


def _pick_uniform(
    N: int, K: int, replace: bool, generator: Optional[torch.Generator]
) -> torch.Tensor:
    """Pick K indices in [0, N) uniformly. Caller controls replace."""
    if N == 0 or K <= 0:
        return torch.zeros((0,), dtype=torch.long)
    if replace:
        return torch.randint(0, N, (K,), generator=generator)
    if K >= N:
        return torch.arange(N)
    return torch.randperm(N, generator=generator)[:K]


def _pick_weighted(
    weights: torch.Tensor,
    K: int,
    replace: bool,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    """Pick K indices in [0, len(weights)) with `multinomial`."""
    N = int(weights.shape[0])
    if N == 0 or K <= 0:
        return torch.zeros((0,), dtype=torch.long)
    if not replace and K >= N:
        return torch.arange(N)
    weights = weights.to(torch.float32).clamp_min(0.0)
    if not torch.isfinite(weights).all() or weights.sum() <= 0:
        # Degenerate weights → fall back to uniform.
        return _pick_uniform(N, K, replace=replace, generator=generator)
    return torch.multinomial(weights, K, replacement=replace, generator=generator)


def _resolve_per_category_counts(
    cats: torch.Tensor,
    K: int,
    per_category: Optional[Mapping],
    min_per_category: Optional[Mapping],
) -> Dict[int, int]:
    """Decide how many tracks to draw per category.

    * If ``per_category`` is given, names/IDs are resolved and the total is
      preserved as-is (caller is responsible for it adding to K, or close).
    * Otherwise: equal share across categories present in ``cats``, with
      remainder distributed by category id order, clamped by `min_per_category`.
    """
    present_ids, present_counts = torch.unique(cats, return_counts=True)
    present_ids = [int(i) for i in present_ids.tolist()]

    if per_category is not None:
        out = {resolve_category(k): int(v) for k, v in per_category.items()}
    else:
        per = K // max(len(present_ids), 1)
        rem = K - per * len(present_ids)
        out = {cid: per for cid in present_ids}
        for cid in present_ids[:rem]:
            out[cid] += 1

    # Apply minimum floor.
    if min_per_category is not None:
        for k, v in min_per_category.items():
            cid = resolve_category(k)
            out[cid] = max(out.get(cid, 0), int(v))

    # Cap each by what's actually available in this sample.
    available = dict(zip(present_ids, [int(c) for c in present_counts]))
    return {cid: min(n, available.get(cid, 0)) for cid, n in out.items()}


def subsample_tracks(
    pt: PointTracks,
    N: int,
    *,
    mode: str = "uniform",
    per_category: Optional[Mapping[Union[int, str], int]] = None,
    weights: Optional[Mapping[Union[int, str], float]] = None,
    min_per_category: Optional[Mapping[Union[int, str], int]] = None,
    replace: bool = False,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, Any]:
    """Subsample the track axis to ``N`` items.

    Modes:

    * ``"uniform"``    -- ``N`` indices drawn uniformly without replacement
                          (with ``replace=True`` for sampling with replacement).
                          ``per_category`` / ``weights`` are ignored, but
                          ``min_per_category`` is still enforced as a post-hoc
                          floor (by topping up shortfalls deterministically).
    * ``"stratified"`` -- exact per-category counts. If ``per_category`` is
                          not given, equal-share across categories present
                          in the sample (remainder spread deterministically).
    * ``"weighted"``   -- per-category sampling probabilities. ``weights``
                          may be omitted to default to equal across present
                          categories. Sampling is multinomial; counts vary.

    ``min_per_category`` is a per-category floor enforced after the primary
    draw; this is the cheap fix for "uniform sampling under-represented my
    rare class on this batch".
    """
    if mode not in ("uniform", "stratified", "weighted"):
        raise ValueError(
            f"mode must be 'uniform', 'stratified', or 'weighted'; got {mode!r}"
        )
    N = int(N)
    total = _track_count(pt)
    if total == 0 or N == 0:
        return _index_tracks(pt, torch.zeros((0,), dtype=torch.long))

    # Uniform with no category structure → trivial path.
    if mode == "uniform" and not min_per_category:
        idx = _pick_uniform(total, N, replace=replace, generator=generator)
        return _index_tracks(pt, idx)

    cats = _categories(pt)

    if mode == "uniform":
        # Same as uniform path above, but we top up shortfalls per category.
        idx = _pick_uniform(total, N, replace=replace, generator=generator)
        chosen_cats = cats[idx]
        for k, floor in min_per_category.items():
            cid = resolve_category(k)
            have = int((chosen_cats == cid).sum())
            need = max(0, int(floor) - have)
            if need == 0:
                continue
            pool = (cats == cid).nonzero(as_tuple=False).flatten()
            extra = _pick_uniform(int(pool.numel()), need,
                                  replace=replace, generator=generator)
            idx = torch.cat([idx, pool[extra]])
        # Trim back to N if min_per_category overshot.
        if int(idx.numel()) > N:
            idx = idx[:N]
        return _index_tracks(pt, idx)

    if mode == "stratified":
        counts = _resolve_per_category_counts(cats, N, per_category, min_per_category)
        picks = []
        for cid, k in counts.items():
            pool = (cats == cid).nonzero(as_tuple=False).flatten()
            chosen = _pick_uniform(int(pool.numel()), k,
                                   replace=replace, generator=generator)
            picks.append(pool[chosen])
        return _index_tracks(pt, torch.cat(picks) if picks else
                             torch.zeros((0,), dtype=torch.long))

    # weighted
    w_per_id: Dict[int, float] = {}
    if weights is not None:
        w_per_id = {resolve_category(k): float(v) for k, v in weights.items()}
    # Build per-track weight vector.
    per_track_w = cats.new_zeros(total, dtype=torch.float32)
    present_ids = torch.unique(cats).tolist()
    if not w_per_id:
        w_per_id = {int(cid): 1.0 for cid in present_ids}
    for cid, w in w_per_id.items():
        per_track_w[cats == cid] = float(w)
    idx = _pick_weighted(per_track_w, N, replace=replace, generator=generator)
    # min_per_category enforced same way as uniform (post-hoc top-up).
    if min_per_category:
        chosen_cats = cats[idx]
        for k, floor in min_per_category.items():
            cid = resolve_category(k)
            have = int((chosen_cats == cid).sum())
            need = max(0, int(floor) - have)
            if need == 0:
                continue
            pool = (cats == cid).nonzero(as_tuple=False).flatten()
            extra = _pick_uniform(int(pool.numel()), need,
                                  replace=replace, generator=generator)
            idx = torch.cat([idx, pool[extra]])
        if int(idx.numel()) > N:
            idx = idx[:N]
    return _index_tracks(pt, idx)


# ---------------------------------------------------------------------------
# Query / target partitioning.
# ---------------------------------------------------------------------------


def split_tracks_query_target(
    pt: PointTracks,
    *,
    n_query: Optional[int] = None,
    query_frac: Optional[float] = None,
    generator: Optional[torch.Generator] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Partition the track axis into (query, target) subsets.

    Exactly one of ``n_query`` or ``query_frac`` must be given. The split
    is a random permutation, so the two halves are statistically
    representative of the whole sample.
    """
    if (n_query is None) == (query_frac is None):
        raise ValueError("pass exactly one of n_query or query_frac")
    total = _track_count(pt)
    if total == 0:
        empty = _index_tracks(pt, torch.zeros((0,), dtype=torch.long))
        return empty, empty
    if n_query is None:
        n_query = max(1, int(round(float(query_frac) * total)))
    n_query = min(int(n_query), total)
    perm = torch.randperm(total, generator=generator)
    q_idx, t_idx = perm[:n_query], perm[n_query:]
    return _index_tracks(pt, q_idx), _index_tracks(pt, t_idx)


def query_points_from_first_frame(
    pt: PointTracks,
    camera: str,
    *,
    n_query: Optional[int] = None,
    require_visible: bool = True,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a CoTracker/PIPs-shape query: (xy: (Q, 2), t: (Q,) zeros).

    Picks the first-frame ``trajs_2d_pix[camera]`` of all tracks (optionally
    filtered to those visible & in-frustum at frame 0), and optionally
    subsamples uniformly to ``n_query`` tracks.
    """
    xy = _require_camera(pt, camera, "trajs_2d_pix")[:, 0]   # (N, 2)
    if require_visible:
        visible = _require_camera(pt, camera, "visible")[:, 0]      # (N,)
        in_frustum = _require_camera(pt, camera, "in_frustum")[:, 0]
        mask = visible & in_frustum
        xy = xy[mask]
    if n_query is not None and int(n_query) < int(xy.shape[0]):
        idx = _pick_uniform(int(xy.shape[0]), int(n_query),
                            replace=False, generator=generator)
        xy = xy[idx]
    t = torch.zeros(int(xy.shape[0]), dtype=torch.long, device=xy.device)
    return xy, t

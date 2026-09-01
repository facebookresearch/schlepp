# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Spatial sample-level transforms (resize / crop / pad).

These helpers operate on the *entire* sample dict that
:class:`schlepp.SchleppDataset` emits, applying the spatial op to every
spatial modality in lockstep so that ``rgb``, ``depth``, ``segmentation``,
``flow_fwd``, ``flow_bwd``, per-camera ``trajs_2d_pix``, per-camera
``in_frustum``, and the camera intrinsics (``K``, ``width``, ``height``)
all stay mutually consistent.

This is the place to fix the "1024x1536 schlepp renders are too big for
my model" problem without re-deriving the trajs/flow/intrinsics by hand.

Notes
-----
* Operates per-camera. ``cameras=None`` (default) means "every camera
  present in ``sample["cameras"]``"; pass a list to restrict.
* ``depth`` is resampled with nearest-neighbour interpolation (zero-valued
  invalid pixels should not bleed into neighbours). ``segmentation`` is
  nearest as well (categorical). ``rgb`` and ``flow`` use the configured
  ``interpolation`` (default bilinear); flow vectors are additionally
  scaled by ``(width', height') / (width, height)`` after resize.
* ``in_frustum`` is re-derived after every op as
  ``old_in_frustum & (0 <= u < W') & (0 <= v < H')``. This preserves the
  "in front of camera" component of the on-disk definition (a spatial op
  can never move a 3D point in front of or behind the camera) while
  re-evaluating the image-bounds component for the new resolution.
* Hflip / vflip are intentionally *not* included: flipping the image
  invalidates the link to ``cam_T_world`` for both pinhole and KB4 cams,
  and there is no clean way to fold that into K. If you really want
  flips, do them as the last step of your transform after every
  3D-consistency-preserving op is done.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

InterpolationMode = str  # "bilinear" | "nearest" | "bicubic"
Sample = Dict[str, Any]
CameraEntry = Dict[str, Any]

# Modalities that live under ``sample["cameras"][cam]`` and have a (..., H, W)
# spatial tail we have to resample on resize / crop / pad.
_IMAGE_MODALITIES: Tuple[str, ...] = ("rgb", "depth", "segmentation",
                                      "flow_fwd", "flow_bwd")
# Modalities with categorical / sparse-validity semantics → nearest by default.
_NEAREST_MODALITIES: Tuple[str, ...] = ("depth", "segmentation")
# Modalities whose pixel values are themselves displacements → require
# vector rescaling on resize.
_FLOW_MODALITIES: Tuple[str, ...] = ("flow_fwd", "flow_bwd")


# ---------------------------------------------------------------------------
# Low-level building blocks: each only does one job.
# ---------------------------------------------------------------------------


def _resolve_target_hw(
    src_h: int,
    src_w: int,
    *,
    height: Optional[int] = None,
    width: Optional[int] = None,
    short_side: Optional[int] = None,
    long_side: Optional[int] = None,
    scale: Optional[float] = None,
) -> Tuple[int, int]:
    """Pick exactly one resize specifier and produce ``(new_h, new_w)``."""
    if (height is None) != (width is None):
        raise ValueError("pass both `height` and `width`, or neither")
    # The (height, width) pair counts as one specifier.
    given = ((height is not None)
             + (short_side is not None)
             + (long_side is not None)
             + (scale is not None))
    if given != 1:
        raise ValueError(
            "pass exactly one of: (height,width), short_side, long_side, scale"
        )
    if height is not None:
        return int(height), int(width)
    if scale is not None:
        return max(1, int(round(src_h * scale))), max(1, int(round(src_w * scale)))
    if short_side is not None:
        s = float(short_side) / float(min(src_h, src_w))
    else:
        s = float(long_side) / float(max(src_h, src_w))
    return max(1, int(round(src_h * s))), max(1, int(round(src_w * s)))


def _resample_image(
    x: torch.Tensor, new_h: int, new_w: int, *, mode: InterpolationMode,
) -> torch.Tensor:
    """Resample a (S, C, H, W) tensor; preserves dtype.

    ``rgb`` is uint8 → must be cast for interpolate then cast back.
    """
    if x.shape[-2] == new_h and x.shape[-1] == new_w:
        return x
    orig_dtype = x.dtype
    x_f = x.to(torch.float32)
    align_corners = None if mode == "nearest" else False
    out = F.interpolate(x_f, size=(new_h, new_w),
                        mode=mode, align_corners=align_corners)
    return out.to(orig_dtype)


def _resample_modality(name: str, x: torch.Tensor, new_h: int, new_w: int,
                       *, image_mode: InterpolationMode) -> torch.Tensor:
    mode = "nearest" if name in _NEAREST_MODALITIES else image_mode
    return _resample_image(x, new_h, new_w, mode=mode)


def _apply_per_cam(
    sample: Sample,
    fn: Callable[[CameraEntry, str], CameraEntry],
    *,
    cameras: Optional[Iterable[str]] = None,
) -> Sample:
    """Build a new sample whose ``cameras`` dict is ``fn``-mapped per camera."""
    if "cameras" not in sample:
        return sample
    selected = set(cameras) if cameras is not None else None
    new_cams = {}
    for name, entry in sample["cameras"].items():
        if selected is None or name in selected:
            new_cams[name] = fn(entry, name)
        else:
            new_cams[name] = entry
    out = {**sample, "cameras": new_cams}
    out = _update_point_tracks_2d(sample, out, fn_was_applied_to=selected
                                  if cameras is not None
                                  else set(sample["cameras"].keys()))
    return out


def _update_point_tracks_2d(
    old_sample: Sample,
    new_sample: Sample,
    *,
    fn_was_applied_to: Iterable[str],
) -> Sample:
    """Mirror the per-camera trajs_2d_pix / in_frustum re-derivation.

    The per-camera spatial fn returns a CameraEntry; this side-table
    helper does the matching update on ``sample["point_tracks"]`` because
    point_tracks is a separate top-level dict but its per-camera
    sub-entries are tied to the spatial state of each camera.
    """
    pt = old_sample.get("point_tracks")
    if pt is None:
        return new_sample
    # Pull the per-camera linear-transform metadata out of the new cam entries
    # (stashed by _resize_cam / _crop_cam / _pad_cam under "_spatial_xform"
    # so the point_tracks update reuses the same math).
    new_pt = {**pt, "trajs_2d_pix": dict(pt["trajs_2d_pix"]),
              "in_frustum": dict(pt["in_frustum"])}
    for cam in fn_was_applied_to:
        entry = new_sample["cameras"].get(cam)
        if entry is None or "_spatial_xform" not in entry:
            continue
        xform = entry.pop("_spatial_xform")
        if cam in new_pt["trajs_2d_pix"]:
            new_pt["trajs_2d_pix"][cam] = xform["apply_xy"](
                new_pt["trajs_2d_pix"][cam])
        if cam in new_pt["in_frustum"]:
            new_pt["in_frustum"][cam] = xform["update_frustum"](
                new_pt["in_frustum"][cam],
                new_pt["trajs_2d_pix"][cam],
            )
    new_sample = {**new_sample, "point_tracks": new_pt}
    return new_sample


# ---------------------------------------------------------------------------
# Per-camera primitive ops.
# Each returns the new CameraEntry and stashes a closure under
# "_spatial_xform" so the matching point_tracks update reuses the same math.
# ---------------------------------------------------------------------------


def _stash_xform(entry: CameraEntry, *, apply_xy, update_frustum) -> CameraEntry:
    entry = {**entry}
    entry["_spatial_xform"] = {
        "apply_xy": apply_xy,
        "update_frustum": update_frustum,
    }
    return entry


def _frustum_inside(uv: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """``(0 <= u < W) & (0 <= v < H)`` for a (..., 2) tensor."""
    u, v = uv.unbind(dim=-1)
    return (u >= 0) & (u < W) & (v >= 0) & (v < H)


def _scale_K(K: torch.Tensor, sx: float, sy: float) -> torch.Tensor:
    K = K.clone()
    K[0, 0] *= sx                                              # fx
    K[1, 1] *= sy                                              # fy
    K[0, 2] *= sx                                              # cx
    K[1, 2] *= sy                                              # cy
    return K


def _shift_K(K: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    K = K.clone()
    K[0, 2] += dx
    K[1, 2] += dy
    return K


def _source_hw(entry: CameraEntry) -> Tuple[int, int]:
    """Return the source ``(H, W)`` for this camera entry.

    Prefer the camera's own ``height`` / ``width`` scalars (populated
    when ``cameras`` is in ``modalities=``); fall back to the spatial
    tail of any loaded image modality. This lets the spatial ops run on
    samples that didn't request the ``cameras`` modality.
    """
    if "height" in entry and "width" in entry:
        return int(entry["height"]), int(entry["width"])
    for name in _IMAGE_MODALITIES:
        if name in entry:
            t = entry[name]
            return int(t.shape[-2]), int(t.shape[-1])
    raise KeyError(
        "camera entry has no width/height and no image modality to infer "
        "them from; load at least one of rgb/depth/segmentation/flow_fwd/"
        "flow_bwd, or include 'cameras' in modalities=."
    )


def _resize_cam(entry: CameraEntry, new_h: int, new_w: int,
                *, image_mode: InterpolationMode) -> CameraEntry:
    src_h, src_w = _source_hw(entry)
    sx = float(new_w) / float(src_w)
    sy = float(new_h) / float(src_h)
    out: CameraEntry = {**entry}
    for name in _IMAGE_MODALITIES:
        if name not in entry:
            continue
        x = _resample_modality(name, entry[name], new_h, new_w,
                               image_mode=image_mode)
        if name in _FLOW_MODALITIES:
            # Flow magnitudes are in source-camera pixel units; rescale.
            scale_vec = x.new_tensor([sx, sy]).reshape(1, 2, 1, 1)
            x = x * scale_vec
        out[name] = x
    if "K" in entry:
        out["K"] = _scale_K(entry["K"], sx, sy)
    if "height" in entry:
        out["height"] = int(new_h)
    if "width" in entry:
        out["width"] = int(new_w)

    def apply_xy(xy: torch.Tensor) -> torch.Tensor:
        # trajs_2d_pix is (N, S, 2); scale per axis.
        return xy * xy.new_tensor([sx, sy])

    def update_frustum(old_frustum: torch.Tensor,
                       new_xy: torch.Tensor) -> torch.Tensor:
        # Round-trip through frustum-inside; preserve "in front of cam".
        return old_frustum & _frustum_inside(new_xy, new_h, new_w)

    return _stash_xform(out, apply_xy=apply_xy, update_frustum=update_frustum)


def _crop_cam(entry: CameraEntry, top: int, left: int,
              new_h: int, new_w: int) -> CameraEntry:
    src_h, src_w = _source_hw(entry)
    if top < 0 or left < 0 or top + new_h > src_h or left + new_w > src_w:
        raise ValueError(
            f"crop ({top=}, {left=}, {new_h=}, {new_w=}) does not fit in "
            f"({src_h=}, {src_w=}). Use pad_sample first if you need to grow."
        )
    out: CameraEntry = {**entry}
    sl_h = slice(top, top + new_h)
    sl_w = slice(left, left + new_w)
    for name in _IMAGE_MODALITIES:
        if name not in entry:
            continue
        out[name] = entry[name][..., sl_h, sl_w].contiguous()
    if "K" in entry:
        out["K"] = _shift_K(entry["K"], -float(left), -float(top))
    if "height" in entry:
        out["height"] = int(new_h)
    if "width" in entry:
        out["width"] = int(new_w)

    def apply_xy(xy: torch.Tensor) -> torch.Tensor:
        return xy - xy.new_tensor([float(left), float(top)])

    def update_frustum(old_frustum: torch.Tensor,
                       new_xy: torch.Tensor) -> torch.Tensor:
        return old_frustum & _frustum_inside(new_xy, new_h, new_w)

    return _stash_xform(out, apply_xy=apply_xy, update_frustum=update_frustum)


def _pad_cam(entry: CameraEntry, top: int, bottom: int, left: int, right: int,
             *, fills: Mapping[str, float]) -> CameraEntry:
    if min(top, bottom, left, right) < 0:
        raise ValueError("pad amounts must be non-negative")
    src_h, src_w = _source_hw(entry)
    new_h = src_h + top + bottom
    new_w = src_w + left + right
    out: CameraEntry = {**entry}
    # F.pad's `pad` order on last 2 dims is (left, right, top, bottom).
    pad_spec = (left, right, top, bottom)
    for name in _IMAGE_MODALITIES:
        if name not in entry:
            continue
        x = entry[name]
        fill = float(fills.get(name, 0))
        # F.pad supports both reflect and constant; constant is the only
        # one that gives a deterministic fill for sparse-validity maps.
        x = F.pad(x.to(torch.float32), pad_spec, mode="constant", value=fill)
        out[name] = x.to(entry[name].dtype)
    if "K" in entry:
        out["K"] = _shift_K(entry["K"], float(left), float(top))
    if "height" in entry:
        out["height"] = int(new_h)
    if "width" in entry:
        out["width"] = int(new_w)

    def apply_xy(xy: torch.Tensor) -> torch.Tensor:
        return xy + xy.new_tensor([float(left), float(top)])

    def update_frustum(old_frustum: torch.Tensor,
                       new_xy: torch.Tensor) -> torch.Tensor:
        # Preserve the "in front of camera" component of old_frustum.
        return old_frustum & _frustum_inside(new_xy, new_h, new_w)

    return _stash_xform(out, apply_xy=apply_xy, update_frustum=update_frustum)


# ---------------------------------------------------------------------------
# Public sample-level ops.
# ---------------------------------------------------------------------------


def resize_sample(
    sample: Sample,
    *,
    height: Optional[int] = None,
    width: Optional[int] = None,
    short_side: Optional[int] = None,
    long_side: Optional[int] = None,
    scale: Optional[float] = None,
    interpolation: InterpolationMode = "bilinear",
    cameras: Optional[Iterable[str]] = None,
) -> Sample:
    """Resize every spatial field of every selected camera.

    Exactly one of the size specifiers must be given:

    * ``(height, width)`` -- target absolute size; identical across cams.
    * ``short_side`` -- scale so the shorter side equals this.
    * ``long_side`` -- scale so the longer side equals this.
    * ``scale`` -- scalar multiplier on (H, W); per-camera (size varies
      if cameras have different native sizes).

    Updates ``rgb``, ``depth``, ``segmentation``, ``flow_fwd``,
    ``flow_bwd`` (incl. flow-vector rescale), per-camera ``trajs_2d_pix``,
    per-camera ``in_frustum``, and ``cameras[c].{K, width, height}``.
    """
    def per_cam(entry: CameraEntry, name: str) -> CameraEntry:
        src_h, src_w = _source_hw(entry)
        new_h, new_w = _resolve_target_hw(
            src_h, src_w,
            height=height, width=width, short_side=short_side,
            long_side=long_side, scale=scale,
        )
        return _resize_cam(entry, new_h, new_w, image_mode=interpolation)

    return _apply_per_cam(sample, per_cam, cameras=cameras)


def crop_sample(
    sample: Sample,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
    cameras: Optional[Iterable[str]] = None,
) -> Sample:
    """Crop a fixed ``(top, left, height, width)`` window from each camera.

    Updates spatial fields, per-camera trajs (subtracts ``(left, top)``),
    per-camera in_frustum (intersect with new image bounds), and
    ``cameras[c].{K, width, height}`` (subtracts from principal point).
    Raises if the window does not fit; use :func:`pad_sample` first if you
    need to grow the canvas.
    """
    def per_cam(entry, name):
        return _crop_cam(entry, top, left, height, width)

    return _apply_per_cam(sample, per_cam, cameras=cameras)


def center_crop_sample(
    sample: Sample,
    *,
    height: int,
    width: int,
    cameras: Optional[Iterable[str]] = None,
) -> Sample:
    """Center crop. Each camera is cropped independently (allowing different
    source sizes), with the window centred on that camera's principal axis
    of the image plane (just the geometric centre, not the principal point).
    """
    def per_cam(entry, name):
        src_h, src_w = _source_hw(entry)
        top = max(0, (src_h - int(height)) // 2)
        left = max(0, (src_w - int(width)) // 2)
        return _crop_cam(entry, top, left, height, width)

    return _apply_per_cam(sample, per_cam, cameras=cameras)


def random_crop_sample(
    sample: Sample,
    *,
    height: int,
    width: int,
    generator: Optional[torch.Generator] = None,
    cameras: Optional[Iterable[str]] = None,
) -> Sample:
    """Random crop. Crop offsets are drawn *per camera* so each camera sees
    its own uniformly-random window. If you need correlated crops across
    cameras (e.g. for cross-view consistency losses), use
    :func:`crop_sample` with explicit ``top`` / ``left``.
    """
    def per_cam(entry, name):
        src_h, src_w = _source_hw(entry)
        max_top = max(0, src_h - int(height))
        max_left = max(0, src_w - int(width))
        if max_top == 0:
            top = 0
        else:
            top = int(torch.randint(0, max_top + 1, (), generator=generator).item())
        if max_left == 0:
            left = 0
        else:
            left = int(torch.randint(0, max_left + 1, (), generator=generator).item())
        return _crop_cam(entry, top, left, height, width)

    return _apply_per_cam(sample, per_cam, cameras=cameras)


def pad_sample(
    sample: Sample,
    *,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    fill_rgb: float = 0,
    fill_depth: float = 0.0,
    fill_segmentation: float = 0,
    fill_flow: float = 0.0,
    cameras: Optional[Iterable[str]] = None,
) -> Sample:
    """Pad the canvas. Fill values are per-modality (defaults zero
    everywhere, which is also the renderer's "invalid" sentinel for
    depth + flow + segmentation).

    Updates spatial fields, per-camera trajs (adds ``(left, top)``),
    per-camera in_frustum (preserves existing valid points), and
    ``cameras[c].{K, width, height}`` (adds to principal point).
    """
    fills = {
        "rgb": fill_rgb,
        "depth": fill_depth,
        "segmentation": fill_segmentation,
        "flow_fwd": fill_flow,
        "flow_bwd": fill_flow,
    }

    def per_cam(entry, name):
        return _pad_cam(entry, top, bottom, left, right, fills=fills)

    return _apply_per_cam(sample, per_cam, cameras=cameras)

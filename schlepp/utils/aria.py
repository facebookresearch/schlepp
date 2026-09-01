# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Undistort Aria-Gen-2-style fisheye (KB4) cameras to a pinhole equivalent.

The schlepp dataset ships three KB4-fisheye cameras (``aria_rgb``,
``aria_slamL``, ``aria_slamR``). Most downstream models (RAFT, WAFT,
CoTracker, MASt3R, ...) assume a pinhole camera; this module rewrites
the affected sample fields so a KB4 source becomes a pinhole that the
downstream model can consume directly.

What gets updated, per camera, from a KB4 source to a pinhole target:

* ``rgb`` -- bilinear-resampled from the source via the KB4 ↔ pinhole
  remap LUT.
* ``depth`` -- nearest-resampled. The on-disk "0 = invalid" sentinel
  stays correct.
* ``segmentation`` -- nearest-resampled (categorical labels).
* ``cameras[c].K`` -- new pinhole intrinsic.
* ``cameras[c].width`` / ``height`` -- target resolution.
* ``cameras[c].distortion_model`` -- ``"pinhole"``.
* ``cameras[c].distortion_params`` -- all-NaN vector, kept at the same
  on-disk slot count for downstream-collate compatibility.
* ``point_tracks.trajs_2d_pix[c]`` -- re-projected from ``trajs_world``
  through the new ``(K_target, cam_T_world)`` (exact, not via remap).
* ``point_tracks.in_frustum[c]`` -- re-derived as "in front of cam"
  ∧ "inside new image bounds". The on-disk ``visible`` mask (occlusion)
  is preserved unchanged.

What is **not** updated (yet):

* ``flow_fwd`` / ``flow_bwd``. A naive image-style remap of the flow
  tensor would give pixel-units that don't match the new pinhole pixel
  grid; the correct fix requires depth-based re-derivation (see the
  module docstring's note). For now we raise if either flow modality
  is present.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from schlepp import geometry as G
from schlepp.io import NUM_DISTORTION_PARAM_SLOTS
from schlepp.projection import CameraView

Sample = Dict[str, Any]
CameraEntry = Dict[str, Any]
InterpolationMode = str  # "bilinear" | "nearest"

# Image modalities we know how to undistort (image-grid remap).
_IMAGE_MODALITIES: Tuple[str, ...] = ("rgb", "depth", "segmentation")
# Image modalities we do NOT yet know how to undistort (would mis-scale
# the per-pixel vector values).
_FLOW_MODALITIES: Tuple[str, ...] = ("flow_fwd", "flow_bwd")
# Default modes for resampling per modality.
_MODALITY_INTERP_DEFAULTS: Dict[str, InterpolationMode] = {
    "rgb": "bilinear",
    "depth": "nearest",          # 0 sentinel must not bleed
    "segmentation": "nearest",   # categorical
}


# ---------------------------------------------------------------------------
# CameraView helpers (kept tiny so the public functions stay readable).
# ---------------------------------------------------------------------------


def _view_from_entry(entry: CameraEntry, *, frame: int = 0) -> CameraView:
    """Build a per-frame :class:`CameraView` from a sample's camera dict.

    Uses ``cam_T_world[frame]`` so the view is valid for that frame's
    projection. ``frame`` doesn't affect ``K`` / distortion / size.
    """
    return CameraView(
        K=entry["K"],
        cam_T_world=entry["cam_T_world"][frame],
        width=int(entry["width"]),
        height=int(entry["height"]),
        distortion_model=entry["distortion_model"],
        distortion_params=entry["distortion_params"],
    )


def _pinhole_view(
    fx: float, fy: float, cx: float, cy: float,
    width: int, height: int,
    *,
    template: torch.Tensor,
    cam_T_world: torch.Tensor,
) -> CameraView:
    """Construct a CameraView for a pinhole camera with the given intrinsics."""
    K = template.new_zeros((3, 3))
    K[0, 0] = fx
    K[1, 1] = fy
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1.0
    nan_pad = template.new_full((NUM_DISTORTION_PARAM_SLOTS,), float("nan"))
    return CameraView(
        K=K, cam_T_world=cam_T_world,
        width=int(width), height=int(height),
        distortion_model="pinhole",
        distortion_params=nan_pad,
    )


# ---------------------------------------------------------------------------
# Target-intrinsic selection
# ---------------------------------------------------------------------------


def pinhole_target_for_kb4(
    source: CameraView,
    *,
    mode: str = "fit_fov",
    target_resolution: Optional[Tuple[int, int]] = None,
) -> CameraView:
    """Pick a pinhole :class:`CameraView` matching a KB4 source.

    Parameters
    ----------
    source
        Source camera, must be ``distortion_model == "kb4"``.
    mode
        * ``"fit_fov"`` (default) -- largest pinhole focal length such
          that the pinhole image at ``target_resolution`` is fully
          covered by the source's FoV. No black borders; some periphery
          of the source is dropped.
        * ``"match_focal"`` -- pinhole's on-axis focal length equals the
          source's. Loses more periphery, but central pixels remain
          1-to-1 with the source.
    target_resolution
        ``(height, width)`` for the pinhole. Defaults to the source's
        own ``(height, width)``.
    """
    if not source._is_kb4():
        raise ValueError(
            f"pinhole_target_for_kb4 requires a KB4 source camera; got "
            f"distortion_model={source.distortion_model!r}"
        )
    if target_resolution is None:
        H_t, W_t = source.height, source.width
    else:
        H_t, W_t = int(target_resolution[0]), int(target_resolution[1])
    cx_t = (W_t - 1) / 2.0
    cy_t = (H_t - 1) / 2.0

    if mode == "match_focal":
        fx = float(source.K[0, 0].item())
        fy = float(source.K[1, 1].item())
    elif mode == "fit_fov":
        # Find the smallest "source FoV corner angle". That's the safe
        # max half-diag FoV for the pinhole at target resolution.
        H_s, W_s = source.height, source.width
        corners = source.K.new_tensor(
            [[0, 0], [W_s - 1, 0], [0, H_s - 1], [W_s - 1, H_s - 1]],
            dtype=torch.float32,
        )
        rays = G.pixel_to_cam_fisheye_kb4_ray(
            corners, source.K, source.distortion_params[:4],
        )
        # angle from z-axis: atan2(|xy|, z); rays are unit so z>0 means in front
        xy_norm = rays[..., :2].norm(dim=-1)
        z = rays[..., 2]
        thetas = torch.atan2(xy_norm, z)
        safe_theta = float(thetas[torch.isfinite(thetas)].min().item())
        half_diag = math.sqrt((W_t / 2.0) ** 2 + (H_t / 2.0) ** 2)
        fx = fy = half_diag / math.tan(safe_theta)
    else:
        raise ValueError(
            f"unknown mode={mode!r}; expected 'fit_fov' or 'match_focal'"
        )

    return _pinhole_view(
        fx, fy, cx_t, cy_t, W_t, H_t,
        template=source.K, cam_T_world=source.cam_T_world,
    )


# ---------------------------------------------------------------------------
# Remap LUT (KB4 pinhole-output-pixel -> source pixel via shared geometry).
# ---------------------------------------------------------------------------


def kb4_to_pinhole_remap(
    source: CameraView, target: CameraView,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the (target_h, target_w) per-pixel LUT from pinhole to KB4 source.

    Returns ``(map_u, map_v)`` of shape ``(target.height, target.width)``,
    each in source pixel coordinates. Off-FoV pinhole pixels get
    ``NaN`` (they should be masked out by the caller; :func:`_apply_remap`
    converts NaN to "outside the image" which then becomes the
    zero-fill region under ``F.grid_sample(..., padding_mode='zeros')``).
    """
    if not source._is_kb4() or target.distortion_model != "pinhole":
        raise ValueError(
            f"expected KB4 source and pinhole target; got "
            f"source.distortion_model={source.distortion_model!r}, "
            f"target.distortion_model={target.distortion_model!r}"
        )
    H_t, W_t = int(target.height), int(target.width)
    yy, xx = torch.meshgrid(
        torch.arange(H_t, dtype=torch.float32),
        torch.arange(W_t, dtype=torch.float32),
        indexing="ij",
    )
    pin_uv = torch.stack([xx, yy], dim=-1)                # (H_t, W_t, 2)
    pin_cam = G.pixel_to_cam_pinhole(
        pin_uv, torch.ones_like(pin_uv[..., 0]), target.K,
    )                                                     # (H_t, W_t, 3) unit-z rays
    src_uv, _valid = G.cam_to_pixel_fisheye_kb4(
        pin_cam, source.K, source.distortion_params[:4],
    )                                                     # (H_t, W_t, 2); NaN where invalid
    return src_uv[..., 0], src_uv[..., 1]


def _apply_remap(
    image: torch.Tensor,
    map_u: torch.Tensor,
    map_v: torch.Tensor,
    *,
    mode: InterpolationMode,
) -> torch.Tensor:
    """Sample ``image`` at the per-pixel LUT.

    ``image`` shape: ``(S, C, H_src, W_src)`` (any dtype).
    Returns ``(S, C, H_t, W_t)`` with the same dtype.
    """
    S, C, H_src, W_src = image.shape
    H_t, W_t = map_u.shape
    # Convert pixel coords to grid_sample's normalised [-1, 1] with
    # align_corners=True (so -1 == centre of first pixel, +1 == centre
    # of last pixel; matches the integer-pixel convention).
    norm_x = 2.0 * map_u / max(W_src - 1, 1) - 1.0
    norm_y = 2.0 * map_v / max(H_src - 1, 1) - 1.0
    # NaN in the LUT (off-FoV) maps to coords outside [-1, 1]; the
    # padding_mode="zeros" then writes 0 there.
    grid = torch.stack([norm_x, norm_y], dim=-1)         # (H_t, W_t, 2)
    grid = grid.unsqueeze(0).expand(S, H_t, W_t, 2).to(image.device)
    nan_mask = torch.isnan(grid).any(dim=-1, keepdim=True)
    grid = torch.where(nan_mask, grid.new_tensor(2.0), grid)
    out = F.grid_sample(
        image.to(torch.float32), grid,
        mode=mode, padding_mode="zeros", align_corners=True,
    )
    return out.to(image.dtype)


# ---------------------------------------------------------------------------
# Per-camera undistort. Top-level routine assembles per-camera work.
# ---------------------------------------------------------------------------


def _undistort_camera(
    entry: CameraEntry,
    *,
    target: CameraView,
    interp_overrides: Mapping[str, InterpolationMode],
) -> CameraEntry:
    """Build the new (pinhole) camera entry. Mutates a copy of ``entry``."""
    source = _view_from_entry(entry, frame=0)
    map_u, map_v = kb4_to_pinhole_remap(source, target)
    out: CameraEntry = {**entry}
    for name in _IMAGE_MODALITIES:
        if name not in entry:
            continue
        mode = interp_overrides.get(name, _MODALITY_INTERP_DEFAULTS[name])
        out[name] = _apply_remap(entry[name], map_u, map_v, mode=mode)
    out["K"] = target.K
    out["width"] = int(target.width)
    out["height"] = int(target.height)
    out["distortion_model"] = "pinhole"
    out["distortion_params"] = target.distortion_params
    return out


def _reproject_tracks(
    entry: CameraEntry, target: CameraView, pt: Mapping[str, Any], cam_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Re-derive ``(trajs_2d_pix[cam], in_frustum[cam])`` for the new pinhole view.

    Re-projects ``trajs_world`` through ``(target.K, cam_T_world)``; the
    `visible` mask is *not* a function of the camera intrinsic and stays
    untouched.
    """
    trajs_w = pt["trajs_world"]                                # (N, S, 3) float
    cam_T_world = entry["cam_T_world"]                         # (S, 4, 4) float
    pts_world = trajs_w.to(torch.float64)
    T = cam_T_world.to(torch.float64)
    # broadcast: (N, S, 3) and (S, 4, 4) → produce (N, S, 3) cam-frame pts.
    # G.world_to_cam expects broadcastable shapes; expand T to (1, S, 4, 4).
    pts_cam = G.world_to_cam(pts_world, T.unsqueeze(0))
    pixels, in_front = G.cam_to_pixel_pinhole(pts_cam, target.K.to(torch.float64))
    pixels = pixels.to(torch.float32)
    u = pixels[..., 0]
    v = pixels[..., 1]
    inside = (u >= 0) & (u < target.width) & (v >= 0) & (v < target.height)
    in_frustum = inside & in_front
    return pixels, in_frustum


def undistort_aria_to_pinhole(
    sample: Sample,
    *,
    cameras: Iterable[str] = ("aria_rgb", "aria_slamL", "aria_slamR"),
    target_intrinsic: Union[str, Mapping[str, "CameraView"]] = "fit_fov",
    target_resolution: Optional[Tuple[int, int]] = None,
    interpolation: Optional[Mapping[str, InterpolationMode]] = None,
) -> Sample:
    """Replace KB4 fisheye cameras in ``sample`` with pinhole equivalents.

    Parameters
    ----------
    sample
        A :class:`schlepp.SchleppDataset` sample dict.
    cameras
        Subset of ``sample["cameras"]`` names to undistort. Each named
        camera must have ``distortion_model == "kb4"``; non-KB4 cameras
        in this list trigger an error.  Names that aren't in
        ``sample["cameras"]`` are silently ignored.
    target_intrinsic
        Either a mode string (``"fit_fov"`` | ``"match_focal"``; see
        :func:`pinhole_target_for_kb4`) or a per-camera mapping from
        name to a literal pinhole :class:`CameraView` you've built
        yourself.
    target_resolution
        ``(height, width)`` for the pinhole output. Defaults to the
        source camera's own resolution. Ignored when ``target_intrinsic``
        is a literal mapping.
    interpolation
        Optional per-modality interpolation overrides
        (``{"rgb": "bilinear", "depth": "nearest", ...}``). Defaults
        match the renderer's per-modality semantics.

    Raises
    ------
    NotImplementedError
        If any camera being undistorted has ``flow_fwd`` or ``flow_bwd``
        loaded -- a naive image remap of the flow tensor would corrupt
        the pixel-unit displacement values, and a correct depth-based
        re-derivation is not yet implemented.
    """
    if "cameras" not in sample:
        return sample
    selected = set(cameras)
    interp_overrides: Mapping[str, InterpolationMode] = dict(interpolation or {})
    new_cams: Dict[str, CameraEntry] = {}
    pt_updates: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for name, entry in sample["cameras"].items():
        if name not in selected:
            new_cams[name] = entry
            continue
        if entry.get("distortion_model") != "kb4":
            raise ValueError(
                f"camera {name!r} has distortion_model="
                f"{entry.get('distortion_model')!r}, expected 'kb4'."
            )
        for fname in _FLOW_MODALITIES:
            if fname in entry:
                raise NotImplementedError(
                    f"undistort of {fname!r} on KB4 camera {name!r} is "
                    "not yet supported. The pixel-unit displacement vectors "
                    "stored in the flow tensor cannot be image-remapped "
                    "without also rescaling by the local KB4->pinhole "
                    "Jacobian. The correct fix is depth-based "
                    "re-derivation; until that ships, exclude flow_* from "
                    "the modalities list for any Aria-undistorted camera."
                )
        if isinstance(target_intrinsic, str):
            target = pinhole_target_for_kb4(
                _view_from_entry(entry, frame=0),
                mode=target_intrinsic,
                target_resolution=target_resolution,
            )
        else:
            target = target_intrinsic[name]
        new_cams[name] = _undistort_camera(
            entry, target=target, interp_overrides=interp_overrides,
        )
        pt = sample.get("point_tracks")
        if pt is not None and name in pt.get("trajs_2d_pix", {}):
            new_xy, new_frustum = _reproject_tracks(entry, target, pt, name)
            pt_updates[name] = (new_xy, new_frustum)

    out: Sample = {**sample, "cameras": new_cams}
    if pt_updates:
        old_pt = sample["point_tracks"]
        new_pt = {
            **old_pt,
            "trajs_2d_pix": dict(old_pt["trajs_2d_pix"]),
            "in_frustum": dict(old_pt["in_frustum"]),
        }
        for name, (xy, frustum) in pt_updates.items():
            new_pt["trajs_2d_pix"][name] = xy
            new_pt["in_frustum"][name] = frustum
        out["point_tracks"] = new_pt
    return out

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""High-level projection utilities for SCHLEPP cameras.

These functions compose the lower-level primitives in :mod:`schlepp.geometry`
into researcher-friendly helpers. Both pinhole and KB4 fisheye paths are
supported; the per-camera ``CameraView`` carries everything needed to
dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from schlepp import geometry as G


@dataclass(frozen=True)
class CameraView:
    """Per-frame camera state for a single view.

    Construct one of these from the per-camera entry in a
    :class:`schlepp.dataset.SchleppDataset` sample. ``cam_T_world`` is the
    extrinsic for the frame you are operating on (shape ``(4, 4)``); when
    you have a stack across frames, slice it externally.
    """

    K: torch.Tensor                      # shape: (3, 3)
    cam_T_world: torch.Tensor            # shape: (4, 4)
    width: int
    height: int
    distortion_model: str                # "pinhole" or "kb4"
    # Fixed-length NaN-padded vector (D = ``io.NUM_DISTORTION_PARAM_SLOTS``).
    # KB4 reads the leading 4 entries (``k1..k4``); pinhole ignores it.
    distortion_params: torch.Tensor      # shape: (D,)

    def _is_kb4(self) -> bool:
        return self.distortion_model == "kb4"


# ---------------------------------------------------------------------------
# Depth sampling
# ---------------------------------------------------------------------------


def sample_depth(
    depth_map: torch.Tensor,
    pixels: torch.Tensor,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Sample a depth map at sub-pixel coordinates.

    Parameters
    ----------
    depth_map
        Shape ``(H, W)`` or ``(1, H, W)``.
    pixels
        Shape ``(..., 2)`` ``(u, v)`` in pixel coordinates.
    mode
        ``"bilinear"`` (default) or ``"nearest"``.

    Returns
    -------
    torch.Tensor
        Shape ``(...)``, dtype matches ``depth_map``. Out-of-bounds
        samples are returned as ``0.0`` (consistent with the dataset's
        ``0 = invalid`` convention for ``.dpt5`` maps).
    """
    if depth_map.ndim == 2:
        depth = depth_map.unsqueeze(0).unsqueeze(0)     # shape: (1, 1, H, W)
    elif depth_map.ndim == 3 and depth_map.shape[0] == 1:
        depth = depth_map.unsqueeze(0)                  # shape: (1, 1, H, W)
    else:
        raise ValueError(
            f"depth_map must be (H, W) or (1, H, W); got {depth_map.shape}"
        )
    H, W = depth.shape[-2:]
    # Normalise pixel coordinates to grid_sample's [-1, 1] convention.
    flat = pixels.reshape(-1, 2).to(depth.dtype)
    u = flat[:, 0]
    v = flat[:, 1]
    grid_u = (u + 0.5) / W * 2.0 - 1.0
    grid_v = (v + 0.5) / H * 2.0 - 1.0
    grid = torch.stack([grid_u, grid_v], dim=-1)
    grid = grid.unsqueeze(0).unsqueeze(0)               # shape: (1, 1, N, 2)
    sampled = F.grid_sample(
        depth, grid,
        mode=mode, padding_mode="zeros", align_corners=False,
    )                                                   # shape: (1, 1, 1, N)
    sampled = sampled.view(-1)                          # shape: (N,)
    return sampled.reshape(pixels.shape[:-1])


# ---------------------------------------------------------------------------
# Pixel <-> camera helpers that dispatch on distortion model
# ---------------------------------------------------------------------------


def pixel_to_cam(view: CameraView, pixels: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
    """Back-project pixels with given Z-depth into ``view``'s camera frame.

    Dispatches to the pinhole or KB4 inverse projection based on
    ``view.distortion_model``.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3)``.
    """
    if view._is_kb4():
        # distortion_params is NaN-padded fixed length; KB4 uses k1..k4.
        return G.pixel_to_cam_fisheye_kb4(
            pixels, depth, view.K, view.distortion_params[..., :4],
        )
    return G.pixel_to_cam_pinhole(pixels, depth, view.K)


def cam_to_pixel(
    view: CameraView,
    points_cam: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project camera-frame points into ``view``'s pixel space.

    Returns
    -------
    pixels
        Shape ``(..., 2)``.
    valid
        Shape ``(...)`` bool mask.
    """
    if view._is_kb4():
        # distortion_params is NaN-padded fixed length; KB4 uses k1..k4.
        return G.cam_to_pixel_fisheye_kb4(
            points_cam, view.K, view.distortion_params[..., :4],
        )
    pixels, in_front = G.cam_to_pixel_pinhole(points_cam, view.K)
    return pixels, in_front


# ---------------------------------------------------------------------------
# Ego-Exo bridge
# ---------------------------------------------------------------------------


def project_ego_to_exo(
    pixels_ego: torch.Tensor,
    ego_depth: torch.Tensor,
    ego_view: CameraView,
    exo_view: CameraView,
    *,
    sample_depth_mode: str = "bilinear",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Map pixels from an ego camera (typically Aria) to an exo camera.

    The function looks up depth on the ego side (either via a provided
    per-pixel tensor or by sampling a full depth map), un-projects into
    world coordinates, then re-projects into the exo camera frame and
    distortion model.

    Parameters
    ----------
    pixels_ego
        Shape ``(N, 2)`` ego-camera pixel coordinates.
    ego_depth
        Either ``(N,)`` per-pixel Z-depths in metres, or a full ego depth
        map of shape ``(H, W)`` / ``(1, H, W)`` which is sampled via
        :func:`sample_depth` at the requested pixels.
    ego_view, exo_view
        Per-frame :class:`CameraView` instances for the two cameras.

    Returns
    -------
    pixels_exo
        Shape ``(N, 2)``. NaN where invalid.
    valid
        Shape ``(N,)`` bool. ``True`` only where depth was positive,
        the ray was in front of both cameras, and the projected pixel
        was finite.
    """
    if pixels_ego.ndim != 2 or pixels_ego.shape[-1] != 2:
        raise ValueError(
            f"pixels_ego must be (N, 2); got {pixels_ego.shape}"
        )

    if ego_depth.ndim == 1:
        depth = ego_depth
    else:
        depth = sample_depth(ego_depth, pixels_ego, mode=sample_depth_mode)

    # Valid depths only.
    valid_depth = depth > 0
    depth_safe = torch.where(
        valid_depth, depth, torch.full_like(depth, 1.0)
    )

    # Ego pixel -> ego cam frame -> world frame -> exo cam frame -> exo pixel.
    pts_ego_cam = pixel_to_cam(ego_view, pixels_ego, depth_safe)  # (N, 3)
    pts_world = G.cam_to_world(pts_ego_cam, ego_view.cam_T_world)  # (N, 3)
    pts_exo_cam = G.world_to_cam(pts_world, exo_view.cam_T_world)  # (N, 3)
    pixels_exo, valid_proj = cam_to_pixel(exo_view, pts_exo_cam)

    valid = valid_depth & valid_proj
    nan = torch.full_like(pixels_exo, float("nan"))
    pixels_exo = torch.where(valid.unsqueeze(-1), pixels_exo, nan)
    return pixels_exo, valid



# ---------------------------------------------------------------------------
# Oriented bounding boxes
# ---------------------------------------------------------------------------


# Eight signs on a unit cube enumerated as bottom face (0..3) then top
# face (4..7).
_OBB_SIGN_CUBE = torch.tensor([
    [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
], dtype=torch.float32)                                 # shape: (8, 3)


def project_obbs_to_2d_bboxes(
    obbs: Sequence[dict],
    K: torch.Tensor,
    cam_T_world: torch.Tensor,
    distortion_model: str = "pinhole",
    distortion_params: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project each OBB's 8 corners and return axis-aligned 2D pixel boxes.

    Vectorised over both the OBB list and the leading shape of
    ``cam_T_world``. A single ``(4, 4)`` extrinsic produces one box per
    OBB; an ``(S, 4, 4)`` clip produces one box per (frame, OBB).

    Parameters
    ----------
    obbs
        Sequence of OBB dicts (e.g. ``sample["obbs"]`` or
        ``schlepp.io.load_obbs(...).objects``). Each entry must carry
        ``centroid (3,)``, ``extents (3,)`` (full extents), and
        ``rotation (4,)`` in ``[x, y, z, w]`` order.
    K
        Shape ``(3, 3)``. Pinhole intrinsic matrix.
    cam_T_world
        Shape ``(..., 4, 4)``. Camera-from-world extrinsics. The
        function broadcasts over the leading dims.
    distortion_model
        ``"pinhole"`` (default) or ``"kb4"`` for the Aria fisheye.
        For back-compatibility ``None`` is accepted and treated as
        ``"pinhole"``.
    distortion_params
        Fixed-length NaN-padded vector (the dataset emits the same shape
        for every camera so callers can pass ``cam["distortion_params"]``
        unconditionally). For KB4 the leading 4 entries are read as
        ``k1..k4``; for pinhole this is ignored entirely.

    Returns
    -------
    bboxes
        Shape ``(..., M, 4)`` float tensor; each entry is
        ``(x0, y0, x1, y1)`` in pixel coordinates.
    valid
        Shape ``(..., M)`` bool tensor; ``True`` iff all 8 corners of
        that box projected successfully (in front of the camera AND
        within the FOV for KB4).
    """
    if len(obbs) == 0:
        # Preserve the broadcasting shape: drop the trailing (4, 4).
        lead_shape = cam_T_world.shape[:-2]
        return (
            torch.zeros(lead_shape + (0, 4), dtype=K.dtype, device=K.device),
            torch.zeros(lead_shape + (0,), dtype=torch.bool, device=K.device),
        )

    centroids = torch.tensor(
        [obb["centroid"] for obb in obbs],
        dtype=K.dtype, device=K.device,
    )                                                   # shape: (M, 3)
    half_extents = torch.tensor(
        [obb["extents"] for obb in obbs],
        dtype=K.dtype, device=K.device,
    ) * 0.5                                             # shape: (M, 3)
    quats = torch.tensor(
        [obb["rotation"] for obb in obbs],
        dtype=K.dtype, device=K.device,
    )                                                   # shape: (M, 4) xyzw
    R = G.quaternion_to_rotation_matrix(quats)          # shape: (M, 3, 3)

    # Build the 8 world-frame corners per OBB.
    signs = _OBB_SIGN_CUBE.to(K)                        # shape: (8, 3)
    local = signs.unsqueeze(0) * half_extents.unsqueeze(1)   # (M, 8, 3)
    # corners_world[m, i] = centroid[m] + R[m] @ local[m, i]
    corners_world = centroids.unsqueeze(1) + torch.einsum(
        "mij,mkj->mki", R, local,
    )                                                   # shape: (M, 8, 3)

    # Apply (... , 4, 4) extrinsic to (M, 8, 3) world points by adding
    # the leading dims via broadcasting.
    lead_shape = cam_T_world.shape[:-2]                 # e.g. (S,)
    cam_T_world_b = cam_T_world.reshape(*lead_shape, 1, 1, 4, 4)
    corners_world_b = corners_world.reshape(
        *((1,) * len(lead_shape)), *corners_world.shape
    )                                                   # (..., M, 8, 3)
    corners_cam = G.world_to_cam(
        corners_world_b, cam_T_world_b,
    )                                                   # (..., M, 8, 3)

    # Project: ``cam_to_pixel_*`` accept arbitrary leading batch shapes,
    # so the (..., M, 8) batch falls out naturally.
    K_b = K.reshape(*((1,) * (corners_cam.ndim - 2)), 3, 3)
    if distortion_model == "kb4":
        if distortion_params is None or distortion_params.shape[-1] < 4:
            raise ValueError(
                "distortion_params must carry at least 4 KB4 coefficients "
                "(k1..k4) for distortion_model='kb4'"
            )
        # Accept any vector >= 4 long (the dataset emits a NaN-padded
        # fixed-length tensor); KB4 only consumes the first 4 slots.
        k_b = distortion_params[..., :4].reshape(
            *((1,) * (corners_cam.ndim - 2)), 4,
        )
        pixels, valid_corner = G.cam_to_pixel_fisheye_kb4(
            corners_cam, K_b, k_b,
        )                                               # (..., M, 8, 2), (..., M, 8)
    elif distortion_model in (None, "pinhole"):
        pixels, valid_corner = G.cam_to_pixel_pinhole(
            corners_cam, K_b,
        )
    else:
        raise ValueError(
            f"unsupported distortion_model: {distortion_model!r}"
        )

    # A box is valid iff every one of its 8 corners projected.
    valid = valid_corner.all(dim=-1)                    # shape: (..., M)
    # Axis-aligned min / max in pixel space along the 8 corners.
    # NaNs on invalid corners would poison min/max; mask them to inf / -inf.
    safe_pixels = torch.where(
        valid_corner.unsqueeze(-1),
        pixels,
        torch.full_like(pixels, float("nan")),
    )
    safe_min = torch.nan_to_num(safe_pixels, nan=float("inf"))
    safe_max = torch.nan_to_num(safe_pixels, nan=float("-inf"))
    xy_min = safe_min.min(dim=-2).values                # (..., M, 2)
    xy_max = safe_max.max(dim=-2).values                # (..., M, 2)
    bboxes = torch.cat([xy_min, xy_max], dim=-1)        # (..., M, 4)
    # For invalid boxes, return zeros (the `valid` mask is the source of truth).
    bboxes = torch.where(
        valid.unsqueeze(-1), bboxes, torch.zeros_like(bboxes),
    )
    return bboxes, valid

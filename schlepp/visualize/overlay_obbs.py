# Copyright (c) Meta Platforms, Inc. and affiliates.
"""3D-OBB wireframe overlay onto sequence frames.

Reads the per-sequence ``object_bounding_boxes.json`` (static scene OBBs),
``cameras.npz``, and ``object_animation*.json`` (per-frame pose of every
carried actor object), projects each box's 8 corners into a chosen
camera's pixel space (handling KB4 fisheye distortion for the Aria
variants), and draws the 12 wireframe edges over the per-frame imagery
for the chosen modality.

Carried-object OBBs follow their per-frame pose; static-scene OBBs stay
where they are. Edges are clipped against the camera near plane per
frame, so boxes that are partially in view remain partially drawn
instead of disappearing entirely.

Usage::

    python -m schlepp.visualize.overlay_obbs <sequence_dir> \
        --camera static [--modality rgb] [--out path/to/out.mp4]
"""
from __future__ import annotations

import argparse
import colorsys
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

from schlepp import geometry as G
from schlepp import io as schlepp_io
from schlepp.visualize._palette import color_for_category as _palette_rgb_for_category

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# OBB geometry
# ---------------------------------------------------------------------------


def _quaternion_to_rotation_matrix(q_xyzw: np.ndarray) -> np.ndarray:
    """Numpy adapter around :func:`schlepp.geometry.quaternion_to_rotation_matrix`."""
    q = torch.from_numpy(np.ascontiguousarray(q_xyzw, dtype=np.float64))
    return G.quaternion_to_rotation_matrix(q).cpu().numpy()


# Eight (signs) on a unit cube enumerated as bottom face (0..3) then top
# face (4..7), used by every corner builder below.
_SIGN_CUBE = np.array([
    [-1, -1, -1], [+1, -1, -1], [+1, +1, -1], [-1, +1, -1],
    [-1, -1, +1], [+1, -1, +1], [+1, +1, +1], [-1, +1, +1],
], dtype=np.float64)                                      # shape: (8, 3)


def _corners_from_pose(
    centroid: np.ndarray, rotation_xyzw: np.ndarray, half_extents: np.ndarray
) -> np.ndarray:
    """Build the 8 world-space corners of an OBB at a given pose."""
    R = _quaternion_to_rotation_matrix(rotation_xyzw)     # shape: (3, 3)
    local = _SIGN_CUBE * half_extents                     # shape: (8, 3)
    return centroid + (R @ local.T).T                     # shape: (8, 3)


def _obb_corners(obb: dict) -> np.ndarray:
    """Eight rest-pose corners of a single OBB (shape ``(8, 3)``)."""
    centroid = np.asarray(obb["centroid"], dtype=np.float64)
    half = np.asarray(obb["extents"], dtype=np.float64) * 0.5
    rotation = np.asarray(obb["rotation"], dtype=np.float64)
    return _corners_from_pose(centroid, rotation, half)


#: Wireframe edge list (vertex index pairs) for a unit cube enumerated as
#: bottom face (0..3) followed by top face (4..7).
_OBB_EDGES: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 0),    # bottom
    (4, 5), (5, 6), (6, 7), (7, 4),    # top
    (0, 4), (1, 5), (2, 6), (3, 7),    # uprights
)


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------


def _color_for_category(category: str) -> Tuple[int, int, int]:
    """Stable BGR colour for a category string.

    Thin OpenCV-flavoured wrapper around the canonical RGB palette in
    :mod:`schlepp.visualize._palette`; we keep BGR at this call site
    because everything in this module hands colours straight to cv2.
    """
    r, g, b = _palette_rgb_for_category(category)
    return (b, g, r)


# ---------------------------------------------------------------------------
# Per-frame edge drawing with near-plane clipping
# ---------------------------------------------------------------------------


def _clip_segment_near(
    p0: np.ndarray, p1: np.ndarray, z_near: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Clip a camera-frame 3D segment against the plane ``z >= z_near``.

    Returns the clipped endpoints, or ``None`` if both endpoints are
    behind the near plane.
    """
    z0, z1 = float(p0[2]), float(p1[2])
    if z0 < z_near and z1 < z_near:
        return None
    if z0 >= z_near and z1 >= z_near:
        return p0, p1
    if z0 < z_near:
        t = (z_near - z0) / (z1 - z0)            # 0..1 toward p1
        p0 = p0 + t * (p1 - p0)
    else:
        t = (z_near - z1) / (z0 - z1)            # 0..1 toward p0
        p1 = p1 + t * (p0 - p1)
    return p0, p1


def _draw_box_edges_clipped(
    img_rgb: np.ndarray,
    corners_cam: np.ndarray,
    K: torch.Tensor,
    distortion_model: Optional[str],
    distortion_params: Optional[torch.Tensor],
    color: Tuple[int, int, int],
    line_thickness: int,
    z_near: float = 0.05,
) -> None:
    """Draw the 12 wireframe edges of one box with per-edge clipping.

    ``corners_cam`` is the box's 8 corners already transformed into the
    camera frame; intrinsics handle the final projection. Each edge is
    clipped against the camera near plane in 3D before projection, so
    partially-visible boxes still render the in-frame portion of every
    edge that crosses the near plane.
    """
    H_img, W_img = img_rgb.shape[:2]
    # Build per-edge clipped endpoint pairs.
    edge_pts: List[Tuple[np.ndarray, np.ndarray]] = []
    for i, j in _OBB_EDGES:
        clipped = _clip_segment_near(corners_cam[i], corners_cam[j], z_near)
        if clipped is not None:
            edge_pts.append(clipped)
    if not edge_pts:
        return

    # Batch all surviving endpoints into one projection call.
    flat_np = np.stack(
        [pt for pair in edge_pts for pt in pair], axis=0
    ).astype(np.float32)                                  # shape: (2E, 3)
    flat = torch.from_numpy(flat_np)
    K_b = K.unsqueeze(0).expand(flat.shape[0], 3, 3)      # shape: (2E, 3, 3)
    if distortion_model == "kb4":
        assert distortion_params is not None
        # ``distortion_params`` is a fixed-length NaN-padded vector
        # (the dataset's uniform shape for collation); KB4 consumes
        # only the first 4 entries.
        k = distortion_params[..., :4]
        k_b = k.unsqueeze(0).expand(flat.shape[0], 4)
        pixels, valid = G.cam_to_pixel_fisheye_kb4(flat, K_b, k_b)
    else:
        pixels, valid = G.cam_to_pixel_pinhole(flat, K_b)
    pixels_np = pixels.cpu().numpy()
    valid_np = valid.cpu().numpy()

    for e in range(len(edge_pts)):
        v0 = bool(valid_np[2 * e])
        v1 = bool(valid_np[2 * e + 1])
        if not (v0 and v1):
            continue
        a = pixels_np[2 * e]
        b = pixels_np[2 * e + 1]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        cv2.line(
            img_rgb,
            (int(round(float(a[0]))), int(round(float(a[1])))),
            (int(round(float(b[0]))), int(round(float(b[1])))),
            color, line_thickness, cv2.LINE_AA,
        )


def _render_modality_frame(
    cam_dir: Path,
    modality: str,
    frame_idx: int,
    *,
    depth_sequence: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Load one frame of the requested modality and return as ``(H, W, 3) uint8`` RGB.

    For ``modality == "depth"`` the caller MAY pre-load the per-sequence
    depth array via ``schlepp_io.load_depth(cam_dir / "depth.dpt5")``
    and pass it as ``depth_sequence`` to avoid re-opening the
    per-sequence HDF5 file once per frame. When ``depth_sequence`` is
    ``None`` the single frame is fetched directly from disk via
    ``load_depth(..., frame_indices=[frame_idx])``.
    """
    if modality == "rgb":
        return schlepp_io.load_rgb(cam_dir / f"rgb_{frame_idx:05d}.png")
    if modality == "depth":
        if depth_sequence is not None:
            depth = depth_sequence[frame_idx]
        else:
            depth = schlepp_io.load_depth(
                cam_dir / "depth.dpt5", frame_indices=[frame_idx],
            )[0]
        valid = depth > 0
        d_norm = np.zeros_like(depth, dtype=np.uint8)
        if valid.any():
            v = depth[valid]
            lo, hi = float(v.min()), float(v.max())
            denom = max(hi - lo, 1e-6)
            d_norm[valid] = np.clip(((depth[valid] - lo) / denom) * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    if modality == "segmentation":
        seg = schlepp_io.load_segmentation(
            cam_dir / f"segmentation_{frame_idx:05d}.png"
        )
        n = int(seg.max()) + 1
        palette = np.zeros((n, 3), dtype=np.uint8)
        for i in range(n):
            r, g, b = colorsys.hsv_to_rgb((i * 37 % 360) / 360.0, 0.6, 1.0)
            palette[i] = [int(r * 255), int(g * 255), int(b * 255)]
        return palette[seg]
    raise ValueError(
        f"unsupported background modality: {modality!r}; "
        f"choose one of rgb, depth, segmentation"
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def overlay_obbs(
    sequence_dir: PathLike,
    camera_name: str,
    *,
    modality: str = "rgb",
    out_path: Optional[PathLike] = None,
    fps: Optional[int] = None,
    line_thickness: int = 2,
    z_near: float = 0.05,
) -> Path:
    """Render OBB wireframes on top of a sequence/camera/modality to an MP4.

    Carried-object OBBs follow their per-frame pose from
    ``object_animation*.json``; static-scene OBBs stay in their
    rest-pose world-frame position. Per-edge near-plane clipping
    keeps partially-visible boxes from popping in and out as they
    enter or leave the field of view.

    Parameters
    ----------
    sequence_dir
        Path to the sequence root.
    camera_name
        Camera variant (must be present in ``cameras.npz``).
    modality
        Background modality: ``"rgb"`` (default), ``"depth"``, or
        ``"segmentation"``.
    out_path
        Output MP4 path. Defaults to
        ``<sequence_dir>/_videos/<camera>_<modality>_obbs.mp4``.
    fps
        Output frame rate. Defaults to the sequence-root metadata
        ``fps`` field.
    line_thickness
        OpenCV line thickness for the wireframe edges.
    z_near
        Near-plane distance in metres used for per-edge clipping.

    Returns
    -------
    pathlib.Path
        The MP4 path that was written.
    """
    sequence_dir = Path(sequence_dir)
    cam_dir = sequence_dir / camera_name
    if not cam_dir.is_dir():
        raise FileNotFoundError(f"camera variant not found: {cam_dir}")

    obbs = schlepp_io.load_obbs(sequence_dir / "object_bounding_boxes.json")
    cams = schlepp_io.load_cameras(sequence_dir / "cameras.npz")
    record = cams.get(camera_name)
    T = record.cam_T_world.shape[0]
    cam_T_world_all = torch.from_numpy(record.cam_T_world.copy())     # (T, 4, 4)
    K = torch.from_numpy(record.K.copy())                             # (3, 3)
    dist_model = record.distortion_model
    # ``record.distortion_params`` is a fixed-length NaN-padded ndarray;
    # only the KB4 branch in ``_draw_box_edges_clipped`` reads it (and
    # slices the first 4 entries).
    dist_params = torch.from_numpy(record.distortion_params.copy())

    # Per-actor object animations keyed by carried-object UID. Each entry
    # gives the per-frame world-space pose of one carried object; we use
    # it to override the corresponding static OBB's pose at draw time.
    with open(sequence_dir / "metadata.json") as f:
        metadata = json.load(f)
    try:
        animations = schlepp_io.load_object_animation(sequence_dir, metadata)
    except (FileNotFoundError, ValueError):
        animations = {}
    uid_to_anim: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
        int(anim["uid"]): (
            np.asarray(anim["centroids"], dtype=np.float64),
            np.asarray(anim["rotations_xyzw"], dtype=np.float64),
        )
        for anim in animations.values()
    }

    # Pre-compute per-box: static corners (for non-animated boxes),
    # half-extents and category color. We always carry the original
    # extents so animated boxes can rebuild corners at any pose.
    box_records: List[Dict] = []
    for obb in obbs.objects:
        uid = int(obb.get("uid", -1))
        half = np.asarray(obb.get("extents", [0, 0, 0]), dtype=np.float64) * 0.5
        animated = uid in uid_to_anim
        record_box = {
            "uid":      uid,
            "color":    _color_for_category(str(obb.get("category", "UNKNOWN"))),
            "animated": animated,
            "half":     half,
        }
        if not animated:
            # Cache the rest-pose corners once; static boxes never move.
            record_box["corners_world"] = _obb_corners(obb)            # (8, 3)
        box_records.append(record_box)

    # Frame metadata.
    fps_value = fps
    if fps_value is None:
        try:
            fps_value = int(round(float(metadata.get("fps", 24))))
        except (TypeError, ValueError):
            fps_value = 24

    out_W = record.width
    out_H = record.height
    if out_path is None:
        out_dir = sequence_dir / "_videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{camera_name}_{modality}_obbs.mp4"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps_value, (out_W, out_H))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {out_path}")

    try:
        # For depth, pre-load the whole per-sequence ``depth.dpt5``
        # once so the per-frame loop below doesn't re-open the HDF5
        # file T times. RGB and segmentation are still per-frame PNGs;
        # _render_modality_frame loads those by name on each call.
        depth_sequence: Optional[np.ndarray] = None
        if modality == "depth":
            depth_path = cam_dir / "depth.dpt5"
            if depth_path.is_file():
                depth_sequence = schlepp_io.load_depth(depth_path)

        for t in range(T):
            try:
                img_rgb = _render_modality_frame(
                    cam_dir, modality, t, depth_sequence=depth_sequence,
                )
            except FileNotFoundError:
                break
            if img_rgb.shape[:2] != (out_H, out_W):
                img_rgb = cv2.resize(
                    img_rgb, (out_W, out_H), interpolation=cv2.INTER_LINEAR
                )

            # Build per-frame world-space corners for every box.
            per_box_corners: List[np.ndarray] = []
            for rec in box_records:
                if rec["animated"]:
                    cent_t, rot_t = uid_to_anim[rec["uid"]]
                    t_idx = min(t, cent_t.shape[0] - 1)
                    per_box_corners.append(
                        _corners_from_pose(cent_t[t_idx], rot_t[t_idx], rec["half"])
                    )
                else:
                    per_box_corners.append(rec["corners_world"])
            corners_world = np.stack(per_box_corners, axis=0)         # (B, 8, 3)

            # Transform all (B*8) corners into the camera frame in one batched op.
            corners_world_t = torch.from_numpy(corners_world.astype(np.float32))
            cam_T_world_t = cam_T_world_all[t].unsqueeze(0).unsqueeze(0)  # (1, 1, 4, 4)
            corners_cam = G.world_to_cam(
                corners_world_t, cam_T_world_t
            ).cpu().numpy().astype(np.float64)                        # (B, 8, 3)

            for b, rec in enumerate(box_records):
                _draw_box_edges_clipped(
                    img_rgb,
                    corners_cam[b], K, dist_model, dist_params,
                    rec["color"], line_thickness, z_near=z_near,
                )
            writer.write(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sequence_dir", type=Path)
    p.add_argument("--camera", required=True)
    p.add_argument(
        "--modality", default="rgb", choices=("rgb", "depth", "segmentation"),
    )
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--line-thickness", type=int, default=2)
    p.add_argument("--z-near", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = overlay_obbs(
        args.sequence_dir, args.camera,
        modality=args.modality, out_path=args.out,
        fps=args.fps, line_thickness=args.line_thickness,
        z_near=args.z_near,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

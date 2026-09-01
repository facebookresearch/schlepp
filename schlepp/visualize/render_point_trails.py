# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Fading-trail point-track video renderer for SCHLEPP.

Reads the per-sequence ``point_tracks.h5`` and the corresponding RGB
frames for a chosen camera variant, and writes an MP4 where every
currently visible point is drawn as a category-coloured dot followed by
a short trail of past positions that fades with age.

Usage as a CLI::

    python -m schlepp.visualize.render_point_trails <sequence_dir> \
        --camera body_follow [--trail-length 12] [--out dense_tracks.mp4]

Or as a library::

    from schlepp.visualize.render_point_trails import render_point_trails
    render_point_trails(sequence_dir, "body_follow")
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from schlepp import io as schlepp_io
from schlepp.categories import (
    ALL_CATEGORIES,
    BODY,
    CARRIED,
    CATEGORY_NAMES,
    CLOTH,
    SCENE,
)

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Visualisation palette (canonical category IDs live in ``schlepp.categories``).
# ---------------------------------------------------------------------------

#: RGB-order palette (this module draws into RGB buffers and converts to
#: BGR only at writer time).
CATEGORY_COLORS_RGB: Dict[int, Tuple[int, int, int]] = {
    BODY:    (0, 220, 0),       # green
    CLOTH:   (60, 220, 220),    # cyan
    CARRIED: (230, 60, 60),     # red
    SCENE:   (60, 140, 255),    # blue
}

#: Number of fractional bits passed to cv2's ``shift`` parameter for
#: sub-pixel drawing. ``1 << SHIFT`` sub-pixel steps per pixel; 4 gives
#: 16 levels which is plenty for anti-aliased dots and trails.
SHIFT: int = 4
SHIFT_SCALE: int = 1 << SHIFT


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_fps(sequence_dir: Path, fallback: int = 24) -> int:
    meta = sequence_dir / "metadata.json"
    if meta.is_file():
        try:
            with open(meta) as f:
                return int(round(float(json.load(f).get("fps", fallback))))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return fallback


def _subsample_points(
    categories: np.ndarray, scene_subsample: float, seed: int
) -> np.ndarray:
    """Keep all non-scene categories; subsample scene-class points."""
    rng = np.random.RandomState(seed)
    mask = np.ones(categories.shape[0], dtype=bool)
    if scene_subsample >= 1.0:
        return mask
    scene_idx = np.where(categories == SCENE)[0]
    n_keep = int(round(len(scene_idx) * scene_subsample))
    if n_keep < len(scene_idx):
        drop = rng.choice(
            scene_idx, size=len(scene_idx) - n_keep, replace=False
        )
        mask[drop] = False
    return mask


def _frame_paths(camera_dir: Path) -> list[Path]:
    paths = sorted(camera_dir.glob("rgb_*.png"))
    if not paths:
        raise FileNotFoundError(f"no rgb_*.png in {camera_dir}")
    return paths


def _draw_frame(
    img_rgb: np.ndarray,
    target_pts: np.ndarray,
    occluded: np.ndarray,
    categories: np.ndarray,
    t: int,
    trail_length: int,
    scale: float,
) -> np.ndarray:
    H, W = img_rgb.shape[:2]
    base = img_rgb.copy()

    # Trails (oldest -> newest so newer alpha sits on top).
    t_start = max(0, t - trail_length + 1)
    for t_prev in range(t_start, t):
        age = (t - t_prev) - 1
        alpha = max(0.15, 0.9 * (1.0 - age / max(1, trail_length - 1)))
        overlay = base.copy()
        prev_vis = ~occluded[:, t_prev]
        next_vis = ~occluded[:, t_prev + 1]
        both = prev_vis & next_vis
        if not both.any():
            continue
        idx = np.where(both)[0]
        for n in idx:
            cat = int(categories[n])
            color = CATEGORY_COLORS_RGB.get(cat, (200, 200, 200))
            fx0 = target_pts[n, t_prev, 0] * scale
            fy0 = target_pts[n, t_prev, 1] * scale
            fx1 = target_pts[n, t_prev + 1, 0] * scale
            fy1 = target_pts[n, t_prev + 1, 1] * scale
            if (0 <= fx0 < W and 0 <= fy0 < H
                    and 0 <= fx1 < W and 0 <= fy1 < H):
                p0 = (int(round(fx0 * SHIFT_SCALE)),
                      int(round(fy0 * SHIFT_SCALE)))
                p1 = (int(round(fx1 * SHIFT_SCALE)),
                      int(round(fy1 * SHIFT_SCALE)))
                cv2.line(
                    overlay, p0, p1, color, 1, cv2.LINE_AA, shift=SHIFT,
                )
        cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0, dst=base)

    # Current dots on top.
    cur_vis = ~occluded[:, t]
    for n in np.where(cur_vis)[0]:
        cat = int(categories[n])
        color = CATEGORY_COLORS_RGB.get(cat, (200, 200, 200))
        fx = target_pts[n, t, 0] * scale
        fy = target_pts[n, t, 1] * scale
        if 0 <= fx < W and 0 <= fy < H:
            center = (int(round(fx * SHIFT_SCALE)),
                      int(round(fy * SHIFT_SCALE)))
            cv2.circle(
                base, center, 3 * SHIFT_SCALE, color, -1, cv2.LINE_AA,
                shift=SHIFT,
            )

    return base


def _draw_hud(
    img_rgb: np.ndarray,
    t: int,
    T: int,
    n_drawn_per_cat: Dict[int, int],
) -> np.ndarray:
    cv2.putText(
        img_rgb, f"Frame {t}/{T - 1}", (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )
    y = 56
    for cat_id in ALL_CATEGORIES:
        name = CATEGORY_NAMES[cat_id]
        n = n_drawn_per_cat.get(cat_id, 0)
        color = CATEGORY_COLORS_RGB[cat_id]
        cv2.circle(img_rgb, (20, y - 4), 5, color, -1, cv2.LINE_AA)
        cv2.putText(
            img_rgb, f"{name} ({n})", (32, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA,
        )
        y += 22
    return img_rgb


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_point_trails(
    sequence_dir: PathLike,
    camera_name: str,
    *,
    trail_length: int = 12,
    scene_subsample: float = 0.5,
    fps: Optional[int] = None,
    out_path: Optional[PathLike] = None,
    max_width: int = 1280,
    seed: int = 42,
    draw_hud: bool = True,
) -> Path:
    """Render fading point trails for a sequence/camera pair to an MP4.

    Parameters
    ----------
    sequence_dir
        Sequence root directory (contains ``point_tracks.h5``,
        ``metadata.json`` and the per-variant subdirs).
    camera_name
        Variant to render (must appear in ``point_tracks.h5``'s
        ``camera_index_to_variant_name`` and must have
        ``camera_rendered`` True).
    trail_length
        Number of past frames to draw in the fading trail.
    scene_subsample
        Fraction of scene-class points to keep; body/cloth/carried are
        always kept in full.
    fps
        Output video FPS. Defaults to the sequence-root ``metadata.json``
        ``fps`` field (fallback 24).
    out_path
        Output MP4 path. Defaults to
        ``<sequence_dir>/_videos/<camera>_point_trails.mp4``.
    max_width
        Downscale frames if their width exceeds this.
    seed
        Subsampling RNG seed.
    draw_hud
        When True (default), overlay a per-frame HUD showing the frame
        index and per-category point counts. Set False for hero / cover
        renders where the on-screen text is unwanted.

    Returns
    -------
    pathlib.Path
        The MP4 path that was written.

    Raises
    ------
    RuntimeError
        If the chosen ``camera_name`` was not rendered for the sequence
        (its V slot in ``point_tracks.h5`` is placeholder data).
    """
    sequence_dir = Path(sequence_dir)
    cam_dir = sequence_dir / camera_name
    if not cam_dir.is_dir():
        raise FileNotFoundError(f"camera variant not found: {cam_dir}")

    pt = schlepp_io.load_point_tracks(sequence_dir / "point_tracks.h5")
    sl = pt.slice_variant(camera_name)
    target = sl["target_points"]                        # shape: (N, T, 2)
    occluded = ~sl["visible"]                           # shape: (N, T)
    categories = sl["categories"]                       # shape: (N,)
    N, T = target.shape[:2]

    frame_paths = _frame_paths(cam_dir)
    if len(frame_paths) != T:
        T = min(T, len(frame_paths))

    keep = _subsample_points(categories, scene_subsample, seed)
    target = target[keep]
    occluded = occluded[keep]
    categories = categories[keep]

    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise IOError(f"failed to read {frame_paths[0]}")
    first = cv2.cvtColor(first, cv2.COLOR_BGR2RGB)
    H_orig, W_orig = first.shape[:2]
    if W_orig > max_width:
        scale = max_width / W_orig
        out_W = max_width
        out_H = int(round(H_orig * scale))
    else:
        scale = 1.0
        out_W, out_H = W_orig, H_orig

    fps = fps if fps is not None else _read_fps(sequence_dir)
    if out_path is None:
        out_dir = sequence_dir / "_videos"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{camera_name}_point_trails.mp4"
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_W, out_H))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {out_path}")

    try:
        for t in range(T):
            img_bgr = cv2.imread(str(frame_paths[t]), cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise IOError(f"failed to read {frame_paths[t]}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if scale != 1.0:
                img_rgb = cv2.resize(
                    img_rgb, (out_W, out_H), interpolation=cv2.INTER_LINEAR
                )
            canvas = _draw_frame(
                img_rgb, target, occluded, categories,
                t, trail_length, scale,
            )
            cur_vis = ~occluded[:, t]
            if draw_hud:
                n_drawn = {
                    c: int(((categories == c) & cur_vis).sum())
                    for c in ALL_CATEGORIES
                }
                canvas = _draw_hud(canvas, t, T, n_drawn)
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sequence_dir", type=Path)
    p.add_argument(
        "--camera", required=True,
        help="camera variant name (e.g. body_follow)",
    )
    p.add_argument("--trail-length", type=int, default=12)
    p.add_argument("--scene-subsample", type=float, default=0.5)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-width", type=int, default=1280)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-hud", action="store_true",
        help="Suppress the per-frame HUD overlay (frame counter + legend).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out = render_point_trails(
        args.sequence_dir, args.camera,
        trail_length=args.trail_length,
        scene_subsample=args.scene_subsample,
        fps=args.fps,
        out_path=args.out,
        max_width=args.max_width,
        seed=args.seed,
        draw_hud=not args.no_hud,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

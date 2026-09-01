# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Generate per-modality MP4s for every camera variant of a sequence.

This is a convenience driver that walks the per-camera-variant subdirs
of a sequence and writes one MP4 per (variant, modality) pair under
``<sequence_dir>/_videos/``. The depth and flow encoders use standard
visualisation choices (TURBO colormap for depth; Middlebury colour
wheel for flow); the segmentation encoder derives a deterministic
palette from the per-sequence ``segmentation_labels.json`` so colours
stay stable across runs.

Per-frame decoding is handled by :class:`schlepp.dataset.SchleppDataset`
driven by a :class:`torch.utils.data.DataLoader`, so PNG / HDF5 reads
and the colormap math run in worker processes while the main thread
pipes frames into an ``ffmpeg`` ``libx264`` encoder (with a transparent
fall-back to OpenCV's ``mp4v`` writer when no ``ffmpeg`` binary can be
located).

Point-track trails are delegated to
:func:`schlepp.visualize.render_point_trails.render_point_trails`.

Usage::

    python -m schlepp.visualize.generate_mp4s <sequence_dir> \
        [--cameras static body_follow ...] \
        [--modalities rgb depth segmentation flow_fwd flow_bwd point_tracks] \
        [--num-workers N]
"""
from __future__ import annotations

import argparse
import colorsys
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from schlepp import io as schlepp_io
from schlepp.dataset import (
    PER_FRAME_MODALITIES,
    PER_SEQUENCE_PER_CAMERA_MODALITIES,
    SchleppDataset,
)
from schlepp.visualize.render_point_trails import render_point_trails

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Modality encoders
# ---------------------------------------------------------------------------


def _palette_from_labels(labels: dict) -> np.ndarray:
    """Build a deterministic ``(num_labels+1, 3) uint8`` BGR palette."""
    max_index = max(labels.keys()) if labels else 0
    palette = np.zeros((max_index + 1, 3), dtype=np.uint8)
    for idx, name in labels.items():
        # Hash the label string to a hue so renames map predictably.
        h = sum(ord(c) for c in name)
        hue = (h * 47 % 360) / 360.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 1.0 if idx != 0 else 0.0)
        palette[idx] = [int(b * 255), int(g * 255), int(r * 255)]
    return palette


def _encode_rgb_frame(arr: np.ndarray) -> np.ndarray:
    """Pass-through: input RGB ``(H, W, 3) uint8`` -> BGR for the writer."""
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _encode_depth_frame(arr: np.ndarray) -> np.ndarray:
    """Per-frame normalised TURBO colormap with invalid (0) pixels left black."""
    valid = arr > 0
    out = np.zeros_like(arr, dtype=np.uint8)
    if valid.any():
        v = arr[valid]
        lo, hi = float(v.min()), float(v.max())
        denom = max(hi - lo, 1e-6)
        out[valid] = np.clip(((arr[valid] - lo) / denom) * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(out, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _encode_segmentation_frame(seg: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map pass-index frame to the deterministic palette."""
    seg = seg.astype(np.int32)
    seg = np.clip(seg, 0, palette.shape[0] - 1)
    return palette[seg]


# Middlebury flow colour wheel (compact 4-step ramp).
def _make_color_wheel() -> np.ndarray:
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    ncols = RY + YG + GC + CB + BM + MR
    cw = np.zeros((ncols, 3), dtype=np.float32)
    col = 0
    cw[0:RY, 0] = 255
    cw[0:RY, 1] = np.linspace(0, 255, RY, endpoint=False)
    col += RY
    cw[col:col + YG, 0] = 255 - np.linspace(0, 255, YG, endpoint=False)
    cw[col:col + YG, 1] = 255
    col += YG
    cw[col:col + GC, 1] = 255
    cw[col:col + GC, 2] = np.linspace(0, 255, GC, endpoint=False)
    col += GC
    cw[col:col + CB, 1] = 255 - np.linspace(0, 255, CB, endpoint=False)
    cw[col:col + CB, 2] = 255
    col += CB
    cw[col:col + BM, 2] = 255
    cw[col:col + BM, 0] = np.linspace(0, 255, BM, endpoint=False)
    col += BM
    cw[col:col + MR, 2] = 255 - np.linspace(0, 255, MR, endpoint=False)
    cw[col:col + MR, 0] = 255
    return cw


_COLOR_WHEEL = _make_color_wheel()


def _encode_flow_frame(flow: np.ndarray) -> np.ndarray:
    """Middlebury colour wheel encoding for an ``(H, W, 2) f32`` flow frame."""
    u = flow[..., 0]
    v = flow[..., 1]
    mag = np.sqrt(u * u + v * v)
    mag_max = max(float(mag.max()), 1e-6)
    u_n = u / mag_max
    v_n = v / mag_max
    rad = np.sqrt(u_n * u_n + v_n * v_n)
    ang = np.arctan2(-v_n, -u_n) / np.pi                              # [-1, 1]
    fk = (ang + 1.0) / 2.0 * (_COLOR_WHEEL.shape[0] - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = (k0 + 1) % _COLOR_WHEEL.shape[0]
    f = fk - k0
    img = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    for ch in range(3):
        c0 = _COLOR_WHEEL[k0, ch] / 255.0
        c1 = _COLOR_WHEEL[k1, ch] / 255.0
        c = (1 - f) * c0 + f * c1
        # Increase saturation with distance, then fade out where rad > 1.
        c = np.where(rad <= 1, 1 - rad * (1 - c), c * 0.75)
        img[..., ch] = np.clip(c * 255.0, 0, 255).astype(np.uint8)
    # OpenCV BGR ordering for VideoWriter.
    return img[..., ::-1].copy()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


_AVAILABLE_FRAME_MODALITIES: Tuple[str, ...] = (
    "rgb", "depth", "segmentation", "flow_fwd", "flow_bwd",
)
_AVAILABLE_PER_CLIP: Tuple[str, ...] = ("point_tracks",)


def _read_fps(sequence_dir: Path, fallback: int = 24) -> int:
    meta = sequence_dir / "metadata.json"
    if meta.is_file():
        try:
            with open(meta) as f:
                return int(round(float(json.load(f).get("fps", fallback))))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return fallback


def _modality_pattern(modality: str) -> str:
    return PER_FRAME_MODALITIES[modality][0]


def _modality_probe_path(cam_dir: Path, modality: str) -> Path:
    """Return the path used to probe whether ``cam_dir`` has frames for
    ``modality``. For per-frame modalities this is the ``_00000`` file;
    for per-sequence-per-camera modalities (depth / flow) it is the
    sequence-level file itself."""
    if modality in PER_FRAME_MODALITIES:
        return cam_dir / _modality_pattern(modality).format(0)
    if modality in PER_SEQUENCE_PER_CAMERA_MODALITIES:
        seq_name, _decoder = PER_SEQUENCE_PER_CAMERA_MODALITIES[modality]
        return cam_dir / seq_name
    raise ValueError(f"unknown modality for probe: {modality!r}")


class _FrameEncoder:
    """Picklable transform: ``SchleppDataset`` sample -> BGR ``uint8`` frame.

    The encoder runs inside the ``DataLoader`` worker, so the colormap
    arithmetic (depth -> TURBO, flow -> wheel, seg -> palette lookup) is
    parallelised alongside the decode itself.
    """

    def __init__(
        self,
        camera: str,
        modality: str,
        segmentation_palette: Optional[np.ndarray] = None,
    ) -> None:
        if modality == "segmentation" and segmentation_palette is None:
            raise ValueError("segmentation_palette is required for segmentation")
        self.camera = camera
        self.modality = modality
        self.segmentation_palette = segmentation_palette

    def __call__(self, sample: Dict[str, Any]) -> np.ndarray:
        tensor = sample["cameras"][self.camera][self.modality]
        arr = tensor[0].numpy()                                    # shape: (C, H, W)
        if self.modality == "rgb":
            arr = np.transpose(arr, (1, 2, 0))                     # shape: (H, W, 3)
            return _encode_rgb_frame(arr)
        if self.modality == "depth":
            return _encode_depth_frame(arr[0])                     # shape: (H, W)
        if self.modality == "segmentation":
            return _encode_segmentation_frame(arr[0], self.segmentation_palette)
        if self.modality in ("flow_fwd", "flow_bwd"):
            arr = np.transpose(arr, (1, 2, 0))                     # shape: (H, W, 2)
            return _encode_flow_frame(arr)
        raise ValueError(f"unknown modality: {self.modality}")


class _PerModalityDataset(SchleppDataset):
    """``SchleppDataset`` variant that counts frames from the requested
    modality rather than RGB.

    ``SchleppDataset._count_frames`` looks for ``rgb_*.png`` to derive
    the per-sequence frame count. For MP4 generation we want to enumerate
    exactly the frames present for whichever modality we are encoding;
    this override does that without altering the rest of the dataset.
    """

    def __init__(self, *args: Any, frame_count_modality: str, **kwargs: Any) -> None:
        self._frame_count_modality = frame_count_modality
        super().__init__(*args, **kwargs)

    def _count_frames(self, seq_dir: Path) -> int:
        cam_dir = seq_dir / self.cameras[0]
        if not cam_dir.is_dir():
            return 0
        modality = self._frame_count_modality
        if modality in PER_SEQUENCE_PER_CAMERA_MODALITIES:
            # Per-sequence modalities live in a single HDF5 file; the
            # sequence axis is the first axis of the dataset.
            seq_name, _decoder = PER_SEQUENCE_PER_CAMERA_MODALITIES[modality]
            seq_path = cam_dir / seq_name
            if not seq_path.is_file():
                return 0
            import h5py  # local import: this overridden _count_frames
                         # only runs from the MP4 driver, no need to
                         # tax module import time.
            ds_name = "depth" if modality == "depth" else "flow"
            with h5py.File(str(seq_path), "r") as f:
                if ds_name not in f:
                    return 0
                return int(f[ds_name].shape[0])
        # Fallback (per-frame): count matching files via iterdir.
        pattern, _ = PER_FRAME_MODALITIES[modality]
        prefix = pattern.split("{", 1)[0]
        suffix = pattern.rsplit("}", 1)[-1]
        return sum(
            1 for p in cam_dir.iterdir()
            if p.name.startswith(prefix) and p.name.endswith(suffix)
        )


def _collate_single(batch: List[np.ndarray]) -> np.ndarray:
    """Unwrap the single-element batch from ``DataLoader(batch_size=1)``."""
    return batch[0]


def _locate_ffmpeg() -> Optional[str]:
    """Return a usable ``ffmpeg`` executable path, or ``None``.

    Prefers a system-installed ``ffmpeg`` (typically more up-to-date)
    and falls back to the static binary that ships with
    ``imageio-ffmpeg`` (a hard dependency of this project). Returns
    ``None`` only if neither is available, in which case callers fall
    back to ``cv2.VideoWriter(*"mp4v")``.
    """
    system = shutil.which("ffmpeg")
    if system:
        return system
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    try:
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None
    return exe if exe and os.path.isfile(exe) else None


_FFMPEG_EXE: Optional[str] = _locate_ffmpeg()


class _FFmpegWriter:
    """Stream raw BGR frames into ``ffmpeg`` over a stdin pipe.

    Uses ``libx264`` with ``yuv420p`` for broad compatibility and a
    ``pad`` filter so odd-sized frames are accepted (libx264's
    ``yuv420p`` requires even dimensions).
    """

    def __init__(
        self, exe: str, out_path: Path, fps: int, width: int, height: int,
    ) -> None:
        self._out_path = out_path
        cmd = [
            exe,
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(int(fps)),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-pix_fmt", "yuv420p",
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-movflags", "+faststart",
            str(out_path),
        ]
        self._proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        # ``frame`` is contiguous BGR uint8 (H, W, 3) from ``_FrameEncoder``.
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError as e:
            stderr = self._proc.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"ffmpeg pipe closed while writing {self._out_path}: {stderr}"
            ) from e

    def release(self) -> None:
        if self._proc.stdin is not None and not self._proc.stdin.closed:
            self._proc.stdin.close()
        retcode = self._proc.wait()
        if retcode != 0:
            stderr = self._proc.stderr.read().decode("utf-8", "replace")
            raise RuntimeError(
                f"ffmpeg exited with code {retcode} writing {self._out_path}: "
                f"{stderr}"
            )


class _CV2Writer:
    """Fallback writer using OpenCV's ``mp4v`` MPEG-4 Part 2 encoder."""

    def __init__(
        self, out_path: Path, fps: int, width: int, height: int,
    ) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"failed to open VideoWriter for {out_path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()


def _open_video_writer(out_path: Path, fps: int, width: int, height: int):
    """Open whichever video writer is available, preferring ``ffmpeg``."""
    if _FFMPEG_EXE is not None:
        return _FFmpegWriter(_FFMPEG_EXE, out_path, fps, width, height)
    return _CV2Writer(out_path, fps, width, height)


def _write_per_frame_mp4(
    sequence_dir: Path,
    camera: str,
    modality: str,
    out_path: Path,
    fps: int,
    num_workers: int,
    segmentation_palette: Optional[np.ndarray] = None,
) -> Optional[Path]:
    """Encode one ``(camera, modality)`` pair into an MP4 via a DataLoader.

    Returns ``None`` if the modality has no frames on disk for this camera.
    """
    pattern_probe = _modality_probe_path(sequence_dir / camera, modality)
    cam_dir = sequence_dir / camera
    # Cheap probe: bail out early if the modality is missing for this
    # camera variant (no per-frame _00000 file for per-frame mods, no
    # per-sequence HDF5 file for depth/flow).
    if not pattern_probe.is_file():
        return None

    index = pd.DataFrame([{"sequence_id": sequence_dir.name}])
    dataset = _PerModalityDataset(
        sequence_dir.parent,
        modalities=(modality,),
        cameras=(camera,),
        seq_len=1,
        frame_stride=1,
        clip_strategy="fixed_stride",
        index=index,
        transform=_FrameEncoder(camera, modality, segmentation_palette),
        frame_count_modality=modality,
        # The synthetic single-row index above only has `sequence_id`, so
        # the dataset's fast path (which reads `num_frames` off the index)
        # cannot apply. Force the on-disk walk so the `_PerModalityDataset`
        # `_count_frames` override actually fires and counts frames for
        # *this* modality (rather than the default RGB-frame heuristic).
        verify_on_disk=True,
    )
    loader_kwargs: Dict[str, Any] = dict(
        batch_size=1,
        num_workers=num_workers,
        collate_fn=_collate_single,
        shuffle=False,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
    loader = DataLoader(dataset, **loader_kwargs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for frame in loader:
            if frame is None:
                # Hard I/O failure for this index; treat as end-of-sequence.
                break
            if writer is None:
                H, W = frame.shape[:2]
                writer = _open_video_writer(out_path, fps, W, H)
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
    return out_path if writer is not None else None


def generate_mp4s(
    sequence_dir: PathLike,
    *,
    cameras: Optional[Sequence[str]] = None,
    modalities: Optional[Sequence[str]] = None,
    out_dir: Optional[PathLike] = None,
    fps: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> List[Path]:
    """Walk a sequence and emit one MP4 per ``(camera, modality)`` pair.

    Parameters
    ----------
    sequence_dir
        Sequence root.
    cameras
        Optional restriction list. Defaults to every directory under
        ``sequence_dir`` that holds at least one ``rgb_*.png`` frame.
    modalities
        Which modalities to render. Defaults to all per-frame
        modalities plus ``point_tracks``.
    out_dir
        Output directory. Defaults to ``<sequence_dir>/_videos``.
    fps
        Output frame rate. Defaults to the sequence-root metadata
        ``fps`` field.
    num_workers
        Number of ``DataLoader`` worker processes used per
        ``(camera, modality)`` pair. ``0`` runs decoding inline on the
        main thread (useful for debugging). Defaults to ``8``.

    Returns
    -------
    list[pathlib.Path]
        Paths to every MP4 that was written.
    """
    sequence_dir = Path(sequence_dir)
    if not sequence_dir.is_dir():
        raise FileNotFoundError(sequence_dir)
    fps_value = fps if fps is not None else _read_fps(sequence_dir)
    out_dir = Path(out_dir) if out_dir is not None else sequence_dir / "_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = 8
    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}")

    if cameras is None:
        cameras = []
        for entry in sorted(sequence_dir.iterdir()):
            if entry.is_dir() and any(entry.glob("rgb_*.png")):
                cameras.append(entry.name)

    if modalities is None:
        modalities = list(_AVAILABLE_FRAME_MODALITIES) + list(_AVAILABLE_PER_CLIP)

    seg_palette: Optional[np.ndarray] = None
    if "segmentation" in modalities:
        labels_path = sequence_dir / "segmentation_labels.json"
        if labels_path.is_file():
            seg_palette = _palette_from_labels(
                schlepp_io.load_segmentation_labels(labels_path)
            )

    written: List[Path] = []
    for cam in cameras:
        cam_dir = sequence_dir / cam
        if not cam_dir.is_dir():
            continue
        for modality in modalities:
            if modality in _AVAILABLE_FRAME_MODALITIES:
                out_path = out_dir / f"{cam}_{modality}.mp4"
                try:
                    result = _write_per_frame_mp4(
                        sequence_dir, cam, modality, out_path, fps_value,
                        num_workers=num_workers,
                        segmentation_palette=seg_palette,
                    )
                except (FileNotFoundError, RuntimeError):
                    continue
                if result is not None:
                    written.append(result)
            elif modality == "point_tracks":
                # Skip if this camera was not rendered for the sequence.
                try:
                    out_path = render_point_trails(
                        sequence_dir, cam,
                        out_path=out_dir / f"{cam}_point_trails.mp4",
                        fps=fps_value,
                    )
                    written.append(out_path)
                except (FileNotFoundError, RuntimeError):
                    continue
            else:
                raise ValueError(f"unknown modality: {modality}")

    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sequence_dir", type=Path)
    p.add_argument("--cameras", nargs="+", default=None)
    p.add_argument(
        "--modalities", nargs="+", default=None,
        choices=tuple(_AVAILABLE_FRAME_MODALITIES) + _AVAILABLE_PER_CLIP,
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument(
        "--num-workers", type=int, default=None,
        help="DataLoader workers per (camera, modality) pair. "
             "0 = inline decode. Default: 8.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    paths = generate_mp4s(
        args.sequence_dir,
        cameras=args.cameras,
        modalities=args.modalities,
        out_dir=args.out_dir,
        fps=args.fps,
        num_workers=args.num_workers,
    )
    for p in paths:
        print(f"wrote {p}")


if __name__ == "__main__":
    main()

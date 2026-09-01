# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Map-style PyTorch dataset for SCHLEPP sequences.

Each sequence lives in its own subdirectory under ``data_root``; a
corpus-wide ``index.parquet`` at the root lists one row per sequence.
Per sequence: a root ``metadata.json``, per-camera-variant subdirectories
of PNGs / HDF5 binaries (``rgb``, ``depth``, ``segmentation``,
``flow_fwd``, ``flow_bwd``), and a handful of shared metadata
files (``cameras.npz``, ``point_tracks.h5``, ``object_bounding_boxes.json``,
``object_animation*.npz``, ``mhr_params*.npz``,
``segmentation_labels.json``).

Per-sample structure
--------------------
One ``__getitem__`` call returns a single ``(sequence, cameras, frame
window)`` clip as a dict:

* ``sequence_id``, ``scene``, ``scene_type``, ``num_actors_total``,
  ``fps``: passthrough scalars.
* ``frame_indices``: shape ``(S,)`` int64, the global frame indices that
  were sampled.
* ``cameras``: dict keyed by the camera variant name; each value is a
  per-camera sub-dict containing the requested per-frame modality
  tensors plus ``K``, ``cam_T_world``, ``width``, ``height``,
  ``distortion_model``, ``distortion_params``.
* ``point_tracks`` (if requested): shared ``trajs_world (N, S, 3)``,
  shared per-track scalars (``categories``, ``pass_indices``,
  ``actor_idx``, ``sample_frame``), and per-camera
  ``trajs_2d_pix``, ``visible``, ``in_frustum`` dicts.
* ``obbs`` (if requested): list of bounding-box dicts loaded verbatim
  from ``object_bounding_boxes.json``.
* ``object_animation``, ``mhr_params`` (if requested): per-actor dicts
  keyed by ``actor_idx``.
* ``segmentation_labels`` (if requested): ``{int_pass_index: str_label}``.

Sample shapes are ``(S, ...)`` per camera and ``(N, S, ...)`` for the
plumbing-of-cameras-into-a-tensor decision is left to the user (see the
README for a recipe).

Selecting modalities and cameras
--------------------------------
Pass ``modalities`` as a subset of :data:`ALL_MODALITIES`; only the
requested data is decoded. Pass ``cameras`` as either a single variant
name (str) or a list (e.g. ``["aria_slamL", "aria_slamR"]``).

Filtering sequences
-------------------
The constructor reads ``data_root/index.parquet`` into a pandas
``DataFrame`` by default. Pass an already-filtered DataFrame via the
``index`` argument to restrict to a subset (see the README for a
``num_actors_total > 2`` example).

Augmentations / preprocessing
-----------------------------
Pass any callable ``transform: Sample -> Any`` to apply per-sample
augmentation; the default is identity. The transform receives the full
sample dict and may return any type (dict, tuple, custom object);
users are responsible for keeping the camera intrinsics consistent with
any image resizing.

Error handling
--------------
``__getitem__`` retries transient I/O errors (``OSError``, ``EOFError``
from partially written binaries) up to :attr:`_MAX_IO_RETRIES` times
on the same ``idx`` so deterministic strategies (``"fixed_stride"``,
``"all_clips"``) keep their enumeration contract. Each retry waits a
short jittered backoff to dodge thundering-herd storms when multiple
DataLoader workers all hit the same flaky chunk server. What happens
after retries are exhausted is controlled by the ``on_error`` ctor
argument:

* ``on_error="raise"`` (default) -- propagate the last exception. This
  is the ecosystem-conventional behaviour (TorchVision, etc.); it is
  compatible with ``torch.utils.data._utils.collate.default_collate``
  and surfaces partial-mirror / disk-corruption problems immediately.
* ``on_error="skip"`` -- return ``None`` so a long training run can
  survive the occasional bad file. Pair the dataset with
  :func:`skip_none_collate` as the ``DataLoader``'s ``collate_fn`` so
  the default collate doesn't crash on the ``None``.

Sharing state across DataLoader workers
---------------------------------------
The dataset is fork-safe and does not keep any per-process random
state: ``"random"`` clip starts are derived deterministically from
``(rng_seed, seq_idx, idx)`` so every worker draws the same sample for
the same index and runs are bitwise-reproducible without a
``worker_init_fn``.
"""
from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch.utils.data import Dataset

from schlepp import io as schlepp_io

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

PathLike = Union[str, os.PathLike]
Sample = Dict[str, Any]
Transform = Callable[[Sample], Any]


def skip_none_collate(batch):
    """``collate_fn`` that drops ``None`` samples then forwards to
    PyTorch's ``default_collate``.

    Use this when constructing the dataset with ``on_error="skip"`` --
    ``__getitem__`` will then return ``None`` after exhausted I/O
    retries, which ``default_collate`` cannot handle. Returns ``None``
    if every sample in the batch failed to load, in which case the
    caller should skip the training step.
    """
    # Imported lazily so the dataset module is not torch-collate-heavy at
    # import time.
    from torch.utils.data._utils.collate import default_collate
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)


# ---------------------------------------------------------------------------
# Modality registry
# ---------------------------------------------------------------------------

#: Per-frame modalities live inside the per-camera directory and use the
#: filename pattern below (NNNNN replaced by zero-padded frame index).
#: Decoders take a path and return a single-frame array.
PER_FRAME_MODALITIES: Dict[str, Tuple[str, Callable[[str], np.ndarray]]] = {
    "rgb":          ("rgb_{:05d}.png",            schlepp_io.load_rgb),
    "segmentation": ("segmentation_{:05d}.png",   schlepp_io.load_segmentation),
}

#: Per-sequence-per-camera modalities live inside the per-camera
#: directory but ship as a single HDF5 file per camera per modality
#: (whole-sequence layout). Decoders take a path + ``frame_indices`` and
#: return a ``(S, ...)`` array sliced to those frames.
PER_SEQUENCE_PER_CAMERA_MODALITIES: Dict[
    str,
    Tuple[str, Callable[[str, Sequence[int]], np.ndarray]]
] = {
    "depth":    ("depth.dpt5",          schlepp_io.load_depth),
    "flow_fwd": ("forward_flow.flo5",   schlepp_io.load_flow),
    "flow_bwd": ("backward_flow.flo5",  schlepp_io.load_flow),
}

#: Per-clip modalities live at the sequence root (one file per sequence).
PER_CLIP_MODALITIES: Tuple[str, ...] = (
    "cameras",
    "point_tracks",
    "obbs",
    "object_animation",
    "mhr_params",
    "segmentation_labels",
)

ALL_MODALITIES: Tuple[str, ...] = (
    tuple(PER_FRAME_MODALITIES)
    + tuple(PER_SEQUENCE_PER_CAMERA_MODALITIES)
    + PER_CLIP_MODALITIES
)

#: Keys in ``mhr_params`` whose leading dimension indexes frames and that
#: therefore need to be sliced by ``frame_indices``. Everything else is
#: passed through verbatim. ``expr_params`` is intentionally NOT in this
#: set -- it is per-actor ``(72,)`` (see ``load_mhr_params`` docstring).
MHR_TEMPORAL_KEYS: Tuple[str, ...] = (
    "model_params",
    "object_positions",
    "object_rotations",
)


def _to_chw(arr: np.ndarray) -> np.ndarray:
    """Promote ``(H, W)`` to ``(1, H, W)`` and ``(H, W, C)`` to ``(C, H, W)``."""
    if arr.ndim == 2:
        return arr[None]                                # shape: (1, H, W)
    if arr.ndim == 3:
        return np.transpose(arr, (2, 0, 1))             # shape: (C, H, W)
    raise ValueError(f"expected 2D or 3D array, got shape={arr.shape}")


def _identity_transform(sample: Sample) -> Sample:
    return sample


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SchleppDataset(Dataset):
    """PyTorch ``Dataset`` over SCHLEPP sequences with opt-in modalities.

    Parameters
    ----------
    data_root
        Directory containing one subfolder per sequence and an
        ``index.parquet`` at the root.
    modalities
        Subset of :data:`ALL_MODALITIES` to load. Anything unrequested
        is never opened.
    cameras
        Camera variant name(s) to load (e.g. ``"static"`` or
        ``["aria_rgb", "static"]``). Per-frame modalities are decoded
        once per camera; ``point_tracks`` are sliced per camera.
    seq_len
        Number of frames per sample. ``None`` returns the entire
        sequence.
    frame_stride
        Stride between sampled frames (``1`` = consecutive frames).
    clip_strategy
        * ``"random"``       -- random start frame per call (training).
        * ``"first"``        -- start at frame 0.
        * ``"fixed_stride"`` -- non-overlapping clips; ``__len__``
          enumerates them. Requires ``seq_len`` to be set.
        * ``"all_clips"``    -- all overlapping clips with stride 1.
          Requires ``seq_len`` to be set.
    index
        Optional pandas ``DataFrame`` listing the sequences to use. When
        ``None`` the constructor reads ``data_root/index.parquet``. The
        DataFrame must contain at minimum a ``sequence_id`` column.
        Pass an externally filtered DataFrame (e.g. ``df[df.num_actors
        _total > 2]``) to restrict the corpus.
    transform
        Optional callable applied to each sample before returning.
        Default: identity. The transform receives the full sample dict
        and may return any type (``Callable[[Sample], Any]``).
    rng_seed
        Seed mixed with ``seq_idx`` and ``idx`` to derive deterministic
        random clip starts for the ``"random"`` strategy. Runs are
        bitwise-reproducible regardless of DataLoader worker count.
    verify_on_disk
        When ``True`` derive per-sequence frame counts by walking the
        per-camera directories (slow on networked storage but tolerant
        of partial mirrors and missing camera variants). When ``False``
        (default) trust the ``num_frames`` column on ``index``; if that
        column is missing the constructor raises ``KeyError``.
    io_workers
        Number of threads used to parallelise I/O for a single sample.
        All per-frame decodes across every requested camera and modality
        plus the per-clip metadata files are submitted to one shared
        pool at the start of ``__getitem__`` so threads stay saturated.
        ``1`` (default) keeps the strictly serial path. Set ``>1`` for
        cloud-backed storage where open/read latency dominates; sizing
        past the storage layer's effective concurrency adds
        context-switch cost without speedup. The pool is per-instance
        and per-process: DataLoader workers (fork or spawn) each get
        their own pool on first use.
    on_error
        What to do when a sample still fails to load after
        :attr:`_MAX_IO_RETRIES` retries (transient ``OSError`` /
        ``EOFError`` from partial-mirror or corrupted files):

        * ``"raise"`` (default) -- re-raise the last exception with its
          native type (so ``except OSError`` callers can match it).
          Works with PyTorch's ``default_collate`` and surfaces real
          problems early. Recommended for development and CI.
        * ``"skip"`` -- return ``None``. Pair the dataset with
          :func:`skip_none_collate` so the loader does not crash on
          the ``None``. Useful for long training runs where occasional
          corruption is expected and survival matters more than
          loudness.

        Non-transient errors (``ValueError`` from a buggy decoder,
        ``KeyError`` from a malformed sample, MHR temporal-axis
        mismatches, etc.) are *not* retried and propagate immediately
        regardless of ``on_error``.
    """

    _MAX_IO_RETRIES: int = 4
    _PRESENT_FRAMES_PROBE: str = "rgb"

    def __init__(
        self,
        data_root: PathLike,
        *,
        modalities: Sequence[str],
        cameras: Union[str, Sequence[str]] = "static",
        seq_len: Optional[int] = None,
        frame_stride: int = 1,
        clip_strategy: Literal[
            "random", "first", "fixed_stride", "all_clips"
        ] = "random",
        index: Optional["pd.DataFrame"] = None,
        transform: Optional[Transform] = None,
        rng_seed: int = 0,
        verify_on_disk: bool = False,
        io_workers: int = 1,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> None:
        super().__init__()

        if clip_strategy not in {"random", "first", "fixed_stride", "all_clips"}:
            raise ValueError(f"unknown clip_strategy: {clip_strategy!r}")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")
        if seq_len is not None and seq_len < 1:
            raise ValueError(f"seq_len must be >= 1 or None, got {seq_len}")
        if seq_len is None and clip_strategy in ("fixed_stride", "all_clips"):
            raise ValueError(
                f"clip_strategy={clip_strategy!r} requires seq_len; "
                f"use 'first' to return whole sequences"
            )
        if io_workers < 1:
            raise ValueError(f"io_workers must be >= 1, got {io_workers}")
        if on_error not in ("raise", "skip"):
            raise ValueError(
                f"on_error must be 'raise' or 'skip', got {on_error!r}"
            )

        # Normalise inputs.
        modalities = tuple(modalities)
        for m in modalities:
            if m not in ALL_MODALITIES:
                raise ValueError(
                    f"unknown modality {m!r}; expected one of {ALL_MODALITIES}"
                )
        if isinstance(cameras, str):
            cameras = (cameras,)
        else:
            cameras = tuple(cameras)
        if not cameras:
            raise ValueError("`cameras` must contain at least one variant")

        self.data_root: Path = Path(data_root).resolve()
        self.modalities: Tuple[str, ...] = modalities
        self.cameras: Tuple[str, ...] = cameras
        self.seq_len = seq_len
        self.frame_stride = frame_stride
        self.clip_strategy = clip_strategy
        self.transform = transform if transform is not None else _identity_transform
        self._rng_seed = int(rng_seed)
        self._io_workers = int(io_workers)
        self._on_error = on_error

        # Shared I/O pool: lazily created on first use, keyed by PID so
        # fork-based DataLoader workers do not inherit a parent's dead
        # thread refs. ``__getstate__`` strips both fields so spawn-mode
        # workers reconstruct cleanly.
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_pid: Optional[int] = None
        self._pool_lock = threading.Lock()

        # Load the corpus index.
        if index is None:
            import pandas as pd
            index_path = self.data_root / "index.parquet"
            if not index_path.is_file():
                raise FileNotFoundError(
                    f"index.parquet not found at {index_path}; "
                    f"pass `index=` explicitly to override"
                )
            index = pd.read_parquet(index_path)
        if "sequence_id" not in index.columns:
            raise KeyError(
                "index DataFrame must contain a 'sequence_id' column"
            )
        # Keep a defensive copy so user-side mutation does not affect us.
        self.index = index.reset_index(drop=True).copy()

        if len(self.index) == 0:
            raise RuntimeError(
                "index DataFrame is empty; nothing to load"
            )

        # Per-sequence frame counts.
        #
        # We trust the ``num_frames`` column on the index by default --
        # the canonical SCHLEPP index always ships it and walking
        # thousands of per-camera directories on networked storage adds
        # multi-minute startup cost. Users with partial mirrors can opt
        # in to the on-disk walk via ``verify_on_disk=True``, which also
        # filters out sequences with missing camera variants (count = 0).
        if verify_on_disk:
            self._frame_counts: List[int] = [
                self._count_frames(self._sequence_dir(row.sequence_id))
                for row in self.index.itertuples(index=False)
            ]
        else:
            if "num_frames" not in self.index.columns:
                raise KeyError(
                    "index DataFrame is missing required 'num_frames' "
                    "column; either rebuild the index with frame counts "
                    "populated, or pass verify_on_disk=True to derive "
                    "them by walking each sequence directory (slow on "
                    "networked storage)"
                )
            self._frame_counts = (
                self.index["num_frames"].astype(int).tolist()
            )

        # Filter out sequences with too few present frames for a single clip.
        if seq_len is not None:
            min_frames = seq_len * frame_stride
        else:
            min_frames = 1
        keep = [
            i for i, n in enumerate(self._frame_counts) if n >= min_frames
        ]
        dropped = len(self._frame_counts) - len(keep)
        if not keep:
            raise RuntimeError(
                f"no sequences have >= {min_frames} present frames for "
                f"seq_len={seq_len}, frame_stride={frame_stride}"
            )
        if dropped:
            sample_ids = self.index.iloc[
                [i for i, n in enumerate(self._frame_counts) if n < min_frames]
            ]["sequence_id"].head(5).tolist()
            logger.warning(
                "SchleppDataset: dropped %d/%d sequences with < %d present "
                "frames across the requested cameras (first few: %s)",
                dropped, len(self._frame_counts), min_frames, sample_ids,
            )
        self.index = self.index.iloc[keep].reset_index(drop=True)
        self._frame_counts = [self._frame_counts[i] for i in keep]

        # Enumerate clips for deterministic strategies.
        self._clips: Optional[List[Tuple[int, int]]] = None  # (seq_idx, start)
        if clip_strategy in ("fixed_stride", "all_clips"):
            self._clips = self._enumerate_clips()

    # ------------------------------------------------------------ pickle

    def __getstate__(self) -> Dict[str, Any]:
        # Strip non-picklable I/O machinery so spawn-mode DataLoader
        # workers can reconstruct the dataset; the pool is lazily
        # recreated per process on first use.
        state = self.__dict__.copy()
        state["_pool"] = None
        state["_pool_pid"] = None
        state.pop("_pool_lock", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._pool_lock = threading.Lock()

    # ----------------------------------------------------------- helpers

    def _sequence_dir(self, sequence_id: str) -> Path:
        return self.data_root / str(sequence_id)

    def _count_frames(self, seq_dir: Path) -> int:
        """Return the per-sequence frame count, ``min`` over every requested
        camera's ``rgb_*.png`` listing.

        Returns ``0`` if any requested camera dir is missing -- such
        sequences are then filtered out by the constructor and surfaced
        in the dropped-sequences warning.
        """
        counts: List[int] = []
        for cam in self.cameras:
            cam_dir = seq_dir / cam
            if not cam_dir.is_dir():
                return 0
            counts.append(sum(
                1 for p in cam_dir.iterdir()
                if p.name.startswith("rgb_") and p.suffix == ".png"
            ))
        return min(counts) if counts else 0

    def _enumerate_clips(self) -> List[Tuple[int, int]]:
        # ``seq_len is None`` with fixed_stride/all_clips is rejected by
        # the ctor, so ``self.seq_len`` is guaranteed to be set here.
        window = self.seq_len * self.frame_stride
        stride = 1 if self.clip_strategy == "all_clips" else window
        out: List[Tuple[int, int]] = []
        for si, n in enumerate(self._frame_counts):
            if n < window:
                continue
            for start in range(0, n - window + 1, stride):
                out.append((si, start))
        return out

    def __len__(self) -> int:
        if self._clips is not None:
            return len(self._clips)
        return len(self.index)

    # ------------------------------------------------------------- pick

    def _pick_window(self, idx: int) -> Tuple[int, np.ndarray]:
        """Resolve which ``(sequence_index, frame_indices)`` ``idx`` maps to.

        Dispatches between the two flavours so each path stays linear:

        * Enumerated clip strategies (``"fixed_stride"``, ``"all_clips"``)
          look up the pre-computed ``(seq_idx, start)`` pair.
        * The on-the-fly strategies (``"random"``, ``"first"``) pick the
          start frame from the current index.
        """
        if self._clips is not None:
            return self._enumerated_window(idx)
        return self._ad_hoc_window(idx)

    def _enumerated_window(self, idx: int) -> Tuple[int, np.ndarray]:
        """Look up a clip from the pre-enumerated ``(seq_idx, start)`` table."""
        assert self._clips is not None  # narrowed by ``_pick_window``
        seq_idx, start = self._clips[idx]
        n_present = self._frame_counts[seq_idx]
        if self.seq_len is None:
            return seq_idx, np.arange(n_present, dtype=np.int64)
        window = self.seq_len * self.frame_stride
        frame_indices = np.arange(
            start, start + window, self.frame_stride, dtype=np.int64,
        )
        return seq_idx, frame_indices

    def _ad_hoc_window(self, idx: int) -> Tuple[int, np.ndarray]:
        """Pick the start frame for ``"random"`` / ``"first"`` strategies."""
        seq_idx = idx
        n_present = self._frame_counts[seq_idx]
        if self.seq_len is None:
            return seq_idx, np.arange(n_present, dtype=np.int64)
        window = self.seq_len * self.frame_stride
        if self.clip_strategy == "first":
            start = 0
        elif self.clip_strategy == "random":
            high = n_present - window
            if high > 0:
                rng = random.Random(f"{self._rng_seed}:{seq_idx}:{idx}")
                start = rng.randint(0, high)
            else:
                start = 0
        else:
            raise AssertionError(
                f"unreachable clip_strategy: {self.clip_strategy}"
            )
        frame_indices = np.arange(
            start, start + window, self.frame_stride, dtype=np.int64,
        )
        return seq_idx, frame_indices

    # ------------------------------------------------------------ build

    def __getitem__(self, idx: int) -> Optional[Sample]:
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        # ``OSError`` covers h5py truncated-file errors and our own
        # ``IOError`` re-raises from cv2's ``None`` returns; ``EOFError``
        # covers partial pickle / json reads. We deliberately do NOT
        # catch ``ValueError`` here because decoders raise it for
        # programmer-side problems (bad shapes, schema mismatches) that
        # don't get better on retry and would otherwise be hidden under
        # four silent retries during development.
        last_err: Optional[BaseException] = None
        for attempt in range(self._MAX_IO_RETRIES):
            try:
                return self.transform(self._build_sample(idx))
            except (OSError, EOFError) as e:
                logger.warning(
                    "SchleppDataset: I/O error on idx=%d (attempt %d/%d): %s",
                    idx, attempt + 1, self._MAX_IO_RETRIES, e,
                )
                last_err = e
                # Jittered exponential backoff before the next attempt
                # so DataLoader workers that all hit the same flaky
                # chunk don't retry in lockstep. No sleep after the last
                # attempt — we're about to give up.
                if attempt + 1 < self._MAX_IO_RETRIES:
                    time.sleep(random.uniform(0, 0.01 * (2 ** attempt)))
        if self._on_error == "skip":
            logger.error(
                "SchleppDataset: giving up on idx=%d after %d retries; "
                "returning None (on_error='skip')",
                idx, self._MAX_IO_RETRIES,
                exc_info=last_err,
            )
            return None
        # on_error == "raise": surface the original exception with its
        # native type so ``except OSError:`` callers can match it; the
        # log line above already records the retry count.
        assert last_err is not None
        logger.error(
            "SchleppDataset: giving up on idx=%d after %d retries; "
            "re-raising",
            idx, self._MAX_IO_RETRIES,
            exc_info=last_err,
        )
        raise last_err

    def _build_sample(self, idx: int) -> Sample:
        seq_idx, frame_indices = self._pick_window(idx)
        row = self.index.iloc[seq_idx]
        seq_dir = self._sequence_dir(row["sequence_id"])

        # metadata.json is tiny and the per-actor per-clip tasks
        # (object_animation, mhr_params) need it to know which files to
        # open, so read it synchronously before planning the fan-out.
        with open(seq_dir / "metadata.json") as f:
            metadata = json.load(f)

        per_frame_mods = [
            m for m in self.modalities if m in PER_FRAME_MODALITIES
        ]
        per_cam_seq_mods = [
            m for m in self.modalities
            if m in PER_SEQUENCE_PER_CAMERA_MODALITIES
        ]

        # Plan every read required for this sample up front so a single
        # pool.submit batch covers (cameras x per-frame modalities x
        # frames), every (camera x per-sequence-per-camera modality)
        # batched read, and every per-clip metadata file in one go.
        per_frame_tasks: Dict[
            Tuple[str, str], List[Tuple[Callable[[str], np.ndarray], str]]
        ] = {}
        per_cam_seq_tasks: Dict[Tuple[str, str], Callable[[], np.ndarray]] = {}
        if per_frame_mods or per_cam_seq_mods:
            # Materialise the sliced frame indices once so the closures
            # below all share the same tuple (no per-iteration tolist()).
            sliced_indices: Tuple[int, ...] = tuple(
                int(fi) for fi in frame_indices.tolist()
            )
            for cam in self.cameras:
                cam_dir = seq_dir / cam
                if not cam_dir.is_dir():
                    raise FileNotFoundError(
                        f"camera variant directory not found: {cam_dir}"
                    )
                for m in per_frame_mods:
                    pattern, decoder = PER_FRAME_MODALITIES[m]
                    per_frame_tasks[(cam, m)] = [
                        (decoder, str(cam_dir / pattern.format(int(fi))))
                        for fi in sliced_indices
                    ]
                for m in per_cam_seq_mods:
                    seq_name, seq_decoder = (
                        PER_SEQUENCE_PER_CAMERA_MODALITIES[m]
                    )
                    seq_path = str(cam_dir / seq_name)
                    # Default-arg trick to bind ``seq_path`` / ``seq_decoder``
                    # / ``sliced_indices`` per (cam, m) iteration -- lambdas
                    # closing over loop vars would share the last binding.
                    per_cam_seq_tasks[(cam, m)] = (
                        lambda p=seq_path, fn=seq_decoder,
                        fi=sliced_indices: fn(p, frame_indices=fi)
                    )

        per_clip_tasks: Dict[str, Callable[[], Any]] = {}
        if "cameras" in self.modalities:
            per_clip_tasks["cameras"] = (
                lambda: schlepp_io.load_cameras(seq_dir / "cameras.npz")
            )
        if "point_tracks" in self.modalities:
            per_clip_tasks["point_tracks"] = (
                lambda: schlepp_io.load_point_tracks(
                    seq_dir / "point_tracks.h5"
                )
            )
        if "obbs" in self.modalities:
            per_clip_tasks["obbs"] = (
                lambda: schlepp_io.load_obbs(
                    seq_dir / "object_bounding_boxes.json"
                )
            )
        if "object_animation" in self.modalities:
            per_clip_tasks["object_animation"] = (
                lambda: schlepp_io.load_object_animation(seq_dir, metadata)
            )
        if "mhr_params" in self.modalities:
            per_clip_tasks["mhr_params"] = (
                lambda: schlepp_io.load_mhr_params(seq_dir, metadata)
            )
        if "segmentation_labels" in self.modalities:
            per_clip_tasks["segmentation_labels"] = (
                lambda: schlepp_io.load_segmentation_labels(
                    seq_dir / "segmentation_labels.json"
                )
            )

        per_frame_arrays, per_cam_seq_arrays, per_clip_values = (
            self._execute_tasks(
                per_frame_tasks, per_cam_seq_tasks, per_clip_tasks,
            )
        )

        sample: Sample = {
            "sequence_id":      str(row["sequence_id"]),
            "scene":            str(metadata.get("scene", row.get("scene", ""))),
            "scene_type":       str(metadata.get("scene_type", row.get("scene_type", ""))),
            "num_actors_total": int(metadata.get(
                "num_actors_total", row.get("num_actors_total", 0)
            )),
            "fps":              float(metadata.get("fps", row.get("fps", 0.0))),
            "frame_indices":    torch.from_numpy(frame_indices.copy()),
            "cameras":          {},
        }

        for cam in self.cameras:
            cam_dict: Dict[str, Any] = {}
            for m in per_frame_mods:
                # Per-frame mods: stack the list of per-frame arrays and
                # promote each to CHW via _to_chw.
                frames = [_to_chw(arr) for arr in per_frame_arrays[(cam, m)]]
                stacked = np.stack(frames, axis=0)          # shape: (S, C, H, W)
                cam_dict[m] = torch.from_numpy(stacked)
            for m in per_cam_seq_mods:
                # Per-sequence-per-camera mods come back pre-stacked as
                # (S, H, W) or (S, H, W, 2). Promote to (S, C, H, W) with
                # a single transpose; no per-frame stacking required.
                arr = per_cam_seq_arrays[(cam, m)]
                if arr.ndim == 3:
                    # (S, H, W) -> (S, 1, H, W)
                    arr = arr[:, None, :, :]
                elif arr.ndim == 4:
                    # (S, H, W, C) -> (S, C, H, W)
                    arr = np.transpose(arr, (0, 3, 1, 2))
                else:
                    raise ValueError(
                        f"unexpected per-sequence modality shape for "
                        f"({cam}, {m}): {arr.shape}"
                    )
                cam_dict[m] = torch.from_numpy(np.ascontiguousarray(arr))
            sample["cameras"][cam] = cam_dict

        if "cameras" in self.modalities:
            self._inject_camera_records(
                sample, per_clip_values["cameras"], frame_indices
            )
        if "point_tracks" in self.modalities:
            sample["point_tracks"] = self._slice_point_tracks(
                per_clip_values["point_tracks"], frame_indices
            )
        if "obbs" in self.modalities:
            sample["obbs"] = list(per_clip_values["obbs"].objects)
        if "object_animation" in self.modalities:
            sample["object_animation"] = self._slice_object_animation(
                per_clip_values["object_animation"], frame_indices
            )
        if "mhr_params" in self.modalities:
            sample["mhr_params"] = self._slice_mhr_params(
                per_clip_values["mhr_params"],
                frame_indices,
                expected_t=self._frame_counts[seq_idx],
            )
        if "segmentation_labels" in self.modalities:
            sample["segmentation_labels"] = per_clip_values[
                "segmentation_labels"
            ]

        return sample

    # ------------------------------------------------------------- I/O

    def _get_pool(self) -> ThreadPoolExecutor:
        """Return a per-process ``ThreadPoolExecutor``, lazily created.

        Keyed by ``os.getpid()`` so a DataLoader worker that forked from
        a parent which had already touched the pool gets its own fresh
        executor instead of the parent's dead thread refs.
        """
        pid = os.getpid()
        if self._pool is not None and self._pool_pid == pid:
            return self._pool
        with self._pool_lock:
            if self._pool is None or self._pool_pid != pid:
                self._pool = ThreadPoolExecutor(
                    max_workers=self._io_workers,
                    thread_name_prefix="schlepp-io",
                )
                self._pool_pid = pid
            return self._pool

    def _execute_tasks(
        self,
        per_frame_tasks: Dict[
            Tuple[str, str], List[Tuple[Callable[[str], np.ndarray], str]]
        ],
        per_cam_seq_tasks: Dict[Tuple[str, str], Callable[[], np.ndarray]],
        per_clip_tasks: Dict[str, Callable[[], Any]],
    ) -> Tuple[
        Dict[Tuple[str, str], List[np.ndarray]],
        Dict[Tuple[str, str], np.ndarray],
        Dict[str, Any],
    ]:
        """Run the planned reads, in parallel when ``io_workers > 1``.

        Per-task exceptions bubble out unchanged so the existing retry
        loop in :meth:`__getitem__` continues to handle transient errors
        (``OSError`` / ``EOFError`` / ``ValueError``).
        """
        total = (
            sum(len(v) for v in per_frame_tasks.values())
            + len(per_cam_seq_tasks)
            + len(per_clip_tasks)
        )
        if self._io_workers <= 1 or total <= 1:
            per_frame_results = {
                key: [fn(path) for (fn, path) in tasks]
                for key, tasks in per_frame_tasks.items()
            }
            per_cam_seq_results = {
                key: fn() for key, fn in per_cam_seq_tasks.items()
            }
            per_clip_results = {
                name: fn() for name, fn in per_clip_tasks.items()
            }
            return per_frame_results, per_cam_seq_results, per_clip_results

        pool = self._get_pool()
        per_frame_futs: Dict[Tuple[str, str], List[Future]] = {
            key: [pool.submit(fn, path) for (fn, path) in tasks]
            for key, tasks in per_frame_tasks.items()
        }
        per_cam_seq_futs: Dict[Tuple[str, str], Future] = {
            key: pool.submit(fn) for key, fn in per_cam_seq_tasks.items()
        }
        per_clip_futs: Dict[str, Future] = {
            name: pool.submit(fn) for name, fn in per_clip_tasks.items()
        }
        try:
            per_frame_results = {
                key: [f.result() for f in futs]
                for key, futs in per_frame_futs.items()
            }
            per_cam_seq_results = {
                key: f.result() for key, f in per_cam_seq_futs.items()
            }
            per_clip_results = {
                name: f.result() for name, f in per_clip_futs.items()
            }
        except BaseException:
            # Best-effort: drop pending tasks so a retry from
            # __getitem__ does not contend with orphans for pool slots.
            # Already-running tasks cannot be cancelled and will finish
            # on their own, which is acceptable.
            for futs in per_frame_futs.values():
                for f in futs:
                    f.cancel()
            for f in per_cam_seq_futs.values():
                f.cancel()
            for f in per_clip_futs.values():
                f.cancel()
            raise
        return per_frame_results, per_cam_seq_results, per_clip_results

    # --------------------------------------------------- per-clip helpers

    def _inject_camera_records(
        self,
        sample: Sample,
        cameras_obj: schlepp_io.Cameras,
        frame_indices: np.ndarray,
    ) -> None:
        """Merge per-camera intrinsics/extrinsics into ``sample['cameras']``."""
        for cam in self.cameras:
            if cam not in cameras_obj:
                raise KeyError(
                    f"camera {cam!r} requested but not present in cameras.npz "
                    f"({list(cameras_obj.variant_names)})"
                )
            record = cameras_obj.get(cam)
            extr = record.cam_T_world[frame_indices]        # shape: (S, 4, 4)
            entry = sample["cameras"].setdefault(cam, {})
            # ``record.K`` is shared with the loaded Cameras dataclass; copy
            # so user transforms can mutate the tensor in place safely.
            entry["K"] = torch.from_numpy(record.K.copy())
            entry["cam_T_world"] = torch.from_numpy(extr)
            entry["width"] = record.width
            entry["height"] = record.height
            entry["distortion_model"] = record.distortion_model
            # ``distortion_params`` is a fixed-length NaN-padded float32
            # vector (KB4 fills slots 0..3, pinhole leaves all NaN), so
            # the field stacks cleanly under ``default_collate`` even
            # across cameras with different distortion models. Consumers
            # branch on ``distortion_model`` and slice the leading
            # ``N`` entries the model uses.
            entry["distortion_params"] = torch.from_numpy(
                record.distortion_params.copy()
            )

    def _slice_point_tracks(
        self,
        pt: schlepp_io.PointTracks,
        frame_indices: np.ndarray,
    ) -> Dict[str, Any]:
        trajs_world: Optional[torch.Tensor] = None
        categories: Optional[torch.Tensor] = None
        pass_indices: Optional[torch.Tensor] = None
        actor_idx: Optional[torch.Tensor] = None
        sample_frame: Optional[torch.Tensor] = None
        trajs_2d: Dict[str, torch.Tensor] = {}
        visible: Dict[str, torch.Tensor] = {}
        in_frustum: Dict[str, torch.Tensor] = {}
        for cam in self.cameras:
            sl = pt.slice_variant(cam, frame_indices=frame_indices)
            trajs_2d[cam]   = torch.from_numpy(sl["target_points"])
            visible[cam]    = torch.from_numpy(sl["visible"])
            in_frustum[cam] = torch.from_numpy(sl["in_frustum"])
            if trajs_world is None:
                trajs_world  = torch.from_numpy(sl["trajs_world"])
                categories   = torch.from_numpy(sl["categories"])
                pass_indices = torch.from_numpy(sl["pass_indices"])
                actor_idx    = torch.from_numpy(sl["actor_idx"])
                sample_frame = torch.from_numpy(sl["sample_frame"])
        return {
            "trajs_2d_pix": trajs_2d,
            "visible":      visible,
            "in_frustum":   in_frustum,
            "trajs_world":  trajs_world,
            "categories":   categories,
            "pass_indices": pass_indices,
            "actor_idx":    actor_idx,
            "sample_frame": sample_frame,
        }

    def _slice_object_animation(
        self,
        animations: Dict[int, Dict[str, Any]],
        frame_indices: np.ndarray,
    ) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        for actor_id, anim in animations.items():
            centroids = anim["centroids"][frame_indices]      # shape: (S, 3)
            rotations = anim["rotations_xyzw"][frame_indices]  # shape: (S, 4)
            out[int(actor_id)] = {
                **{k: anim[k] for k in (
                    "uid", "category", "role",
                    "frame_start", "frame_end",
                    "world_convention", "rotation_convention",
                ) if k in anim},
                "centroids":      torch.from_numpy(centroids),
                "rotations_xyzw": torch.from_numpy(rotations),
            }
        return out

    def _slice_mhr_params(
        self,
        per_actor: Dict[int, Dict[str, Any]],
        frame_indices: np.ndarray,
        *,
        expected_t: int,
    ) -> Dict[int, Dict[str, Any]]:
        """Slice the temporal-leading-axis MHR keys; pass everything else
        through verbatim.

        Each :data:`MHR_TEMPORAL_KEYS` array is asserted to have leading
        dim equal to ``expected_t`` (the sequence's canonical frame
        count). A silent fallback would mask data corruption: numpy
        fancy-indexing only raises ``IndexError`` when an index exceeds
        the actual length, so a temporal axis that is *shorter* than
        expected but still covers the picked window would otherwise
        return frames from the wrong portion of the array.
        """
        out: Dict[int, Dict[str, Any]] = {}
        for actor_id, params in per_actor.items():
            sliced: Dict[str, Any] = {}
            for key, value in params.items():
                if key in MHR_TEMPORAL_KEYS and isinstance(value, np.ndarray):
                    if value.shape[0] != expected_t:
                        raise ValueError(
                            f"mhr_params[{actor_id}][{key!r}] has "
                            f"temporal axis {value.shape[0]}; expected "
                            f"{expected_t} (sequence's canonical frame "
                            f"count). Likely data corruption."
                        )
                    sliced[key] = torch.from_numpy(value[frame_indices])
                elif isinstance(value, np.ndarray):
                    sliced[key] = torch.from_numpy(value.copy())
                else:
                    sliced[key] = value
            out[int(actor_id)] = sliced
        return out

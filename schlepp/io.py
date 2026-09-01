# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Loaders for the on-disk SCHLEPP layout.

This module owns the low-level "read a file off disk" surface so that
``schlepp.dataset`` and the visualisers stay focused on composition rather
than format details. Every function here is pure (no shared state) and free
of torch dependencies.

Conventions
-----------
* World frame: Z-up, right-handed, metres.
* Camera frame: OpenCV (X right, Y down, Z forward).
* Quaternion representation: ``[x, y, z, w]``.
* ``cam_T_world`` reads as "transform _to_ cam _from_ world", i.e.
  ``p_cam = (cam_T_world @ p_world.h)[:3]``.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import h5py
import numpy as np

PathLike = Union[str, os.PathLike]


def _h5_fancy_index(
    frame_indices: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare a sorted fancy-index plus its inverse permutation for h5py.

    h5py's fancy-indexing of a chunked dataset wants the integer index
    array to be **monotonically increasing** (and, in practice, unique
    for efficient single-chunk reads). Passing an arbitrary-order list
    works on recent h5py but goes through a slow general path that may
    re-read chunks. The dataset's frame windows can come from random
    clip-starts or out-of-order user requests, so we:

    1. sort the requested indices ascending,
    2. issue a single sorted read against the h5py dataset,
    3. permute the result back into the caller's order with
       ``out_sorted[inverse]``.

    Returns
    -------
    sorted_idx
        Shape ``(N,) int64``; the ascending index array to pass to
        ``h5py.Dataset.__getitem__``. Some h5py versions want a Python
        list, so call sites typically wrap this in ``list(...)``.
    inverse
        Shape ``(N,) int64``; index the *sorted-order* read result with
        this to recover the caller's original order.

    Notes
    -----
    No deduplication is performed: callers in this module construct
    frame windows from strided ``np.arange``, so duplicates do not
    occur. If a caller ever needs duplicates, h5py will still read
    them (slowly) and the inverse-permutation step still works.
    """
    idx = np.asarray(frame_indices, dtype=np.int64)
    order = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return sorted_idx, inverse


# ---------------------------------------------------------------------------
# Per-frame binary decoders
# ---------------------------------------------------------------------------


def load_rgb(path: PathLike) -> np.ndarray:
    """Read an RGB or RGBA PNG; drop alpha if present.

    Returns
    -------
    np.ndarray
        Shape ``(H, W, 3)``, dtype ``uint8``.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)  # shape: (H, W, C) BGR(A)
    if img is None:
        raise IOError(f"failed to read image: {path}")
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(
            f"expected 3- or 4-channel image, got shape={img.shape} for {path}"
        )
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)  # shape: (H, W, 3)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.cvtColor returns a freshly-allocated contiguous (H, W, 3) buffer;
    # an extra np.ascontiguousarray here would just be a wasted full copy.
    return img


def load_depth(
    path: PathLike,
    frame_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Read a per-sequence Spring-style ``.dpt5`` depth file.

    The file is HDF5 with a single ``depth`` dataset of shape
    ``(S, H, W)`` (whole-sequence layout) chunked one frame per chunk.
    Pixels in unreliable categories (transparency, reflectivity, void)
    are stored as ``0.0``.

    Parameters
    ----------
    path
        Path to ``<cam_dir>/depth.dpt5``.
    frame_indices
        Optional 0-based indices into the sequence axis. When given,
        only those frames are decoded (cheap, single-chunk reads each).
        When ``None`` (default) the whole sequence is materialised.

    Returns
    -------
    np.ndarray
        Shape ``(S, H, W)`` when ``frame_indices`` is ``None``, else
        ``(len(frame_indices), H, W)``. Dtype ``float32``, units = metres.
    """
    with h5py.File(str(path), "r") as f:
        if "depth" not in f:
            raise IOError(f"{path}: missing 'depth' dataset")
        ds = f["depth"]
        if frame_indices is None:
            return ds[()].astype(np.float32, copy=False)  # shape: (S, H, W)
        sorted_idx, inverse = _h5_fancy_index(frame_indices)
        sorted_arr = ds[list(sorted_idx)].astype(np.float32, copy=False)
        return np.ascontiguousarray(sorted_arr[inverse])  # shape: (S', H, W)


#: Alias mirroring the on-disk file extension.
load_dpt5 = load_depth


def load_flow(
    path: PathLike,
    frame_indices: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Read a per-sequence Spring-style ``.flo5`` optical-flow file.

    HDF5 with a single ``flow`` dataset of shape ``(S, H, W, 2)``
    (whole-sequence layout) chunked one frame per chunk. Per-pixel
    values are pixel-space displacement ``(du, dv)``.

    By the renderer's boundary contract:

    * For ``forward_flow.flo5`` the final frame (``flow[-1]``) is
      all-zero (no destination frame past the take boundary).
    * For ``backward_flow.flo5`` the first frame (``flow[0]``) is
      all-zero.

    Parameters
    ----------
    path
        Path to ``<cam_dir>/forward_flow.flo5`` or
        ``<cam_dir>/backward_flow.flo5``.
    frame_indices
        Optional 0-based indices into the sequence axis (cheap
        single-chunk reads when given).

    Returns
    -------
    np.ndarray
        Shape ``(S, H, W, 2)`` (full read) or ``(S', H, W, 2)`` (sliced).
        Dtype ``float32``, channels = ``(du, dv)``.
    """
    with h5py.File(str(path), "r") as f:
        if "flow" not in f:
            raise IOError(f"{path}: missing 'flow' dataset")
        ds = f["flow"]
        if frame_indices is None:
            return ds[()].astype(np.float32, copy=False)  # shape: (S, H, W, 2)
        sorted_idx, inverse = _h5_fancy_index(frame_indices)
        sorted_arr = ds[list(sorted_idx)].astype(np.float32, copy=False)
        return np.ascontiguousarray(sorted_arr[inverse])


#: Alias mirroring the on-disk file extension.
load_flo5 = load_flow


def load_segmentation(path: PathLike) -> np.ndarray:
    """Read an 8-bit pass-index segmentation PNG.

    Each pixel value is an integer key into the per-sequence
    ``segmentation_labels.json`` map.

    Returns
    -------
    np.ndarray
        Shape ``(H, W)``, dtype ``uint8``.
    """
    arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)  # shape: (H, W) uint8
    if arr is None:
        raise IOError(f"failed to read segmentation: {path}")
    return arr


# ---------------------------------------------------------------------------
# Segmentation labels
# ---------------------------------------------------------------------------


def load_segmentation_labels(path: PathLike) -> Dict[int, str]:
    """Parse ``segmentation_labels.json`` into ``{int_pass_index: label_str}``.

    The on-disk JSON uses string keys per the JSON spec; the int coercion
    happens here so downstream code can index directly with ``int`` keys.
    """
    with open(path) as f:
        raw = json.load(f)
    return {int(k): str(v) for k, v in raw.items()}


def decode_segmentation(
    png_path: PathLike,
    segmentation_labels: Union[Mapping[int, str], PathLike],
) -> Dict[str, np.ndarray]:
    """Decode a segmentation PNG into ``{label_str: boolean_mask}``.

    Parameters
    ----------
    png_path
        Path to a single ``segmentation_NNNNN.png``.
    segmentation_labels
        Either the parsed ``{int: str}`` mapping or a path to
        ``segmentation_labels.json``.

    Returns
    -------
    dict
        One entry per label that appears in either ``segmentation_labels``
        or the image, keyed by the label string, with value
        ``(H, W) bool`` masks indicating where that label's pass index
        was rendered.
    """
    if not isinstance(segmentation_labels, Mapping):
        segmentation_labels = load_segmentation_labels(segmentation_labels)
    seg = load_segmentation(png_path)  # shape: (H, W) uint8
    masks: Dict[str, np.ndarray] = {}
    for pass_index, label in segmentation_labels.items():
        masks[label] = (seg == pass_index)  # shape: (H, W) bool
    return masks


# ---------------------------------------------------------------------------
# Cameras (multi-view intrinsics + per-frame extrinsics)
# ---------------------------------------------------------------------------


#: Number of distortion-parameter slots emitted per camera. Matches the
#: ``mv_distortion_params (V, 12)`` slot count of the on-disk schema. The
#: per-camera ``CameraRecord.distortion_params`` vector is always this
#: length, NaN-padded; each distortion model consumes only the leading
#: ``N`` entries it defines (e.g. KB4 uses ``[:4]``).
NUM_DISTORTION_PARAM_SLOTS: int = 12


@dataclass(frozen=True)
class CameraRecord:
    """Single camera's calibration over the full sequence.

    Attributes
    ----------
    name
        Variant name (e.g. ``"static"``, ``"aria_rgb"``).
    K
        Pinhole intrinsic matrix, shape ``(3, 3)`` float32.
    cam_T_world
        Per-frame cam-from-world transform, shape ``(T, 4, 4)`` float32.
    width, height
        Pixel image size.
    distortion_model
        ``"pinhole"`` for rectified pinhole, ``"kb4"`` for Aria fisheye.
    distortion_params
        Shape ``(NUM_DISTORTION_PARAM_SLOTS,)`` float32, NaN-padded. KB4
        occupies slots 0..3 (``k1..k4``); pinhole leaves every slot NaN.
        The uniform shape lets ``default_collate`` stack the field even
        when a batch mixes cameras with different distortion models.
        Callers dispatch on ``distortion_model`` and slice the first
        ``N`` coefficients the model uses.
    """

    name: str
    K: np.ndarray
    cam_T_world: np.ndarray
    width: int
    height: int
    distortion_model: str
    distortion_params: np.ndarray


@dataclass(frozen=True)
class Cameras:
    """Multi-view camera record loaded from ``cameras.npz``.

    The order of ``variant_names`` matches the on-disk
    ``camera_index_to_variant_name`` array; callers should index by name
    via :meth:`get` rather than by integer.
    """

    variant_names: Tuple[str, ...]
    K: np.ndarray                          # shape: (V, 3, 3) float32
    cam_T_world: np.ndarray                # shape: (V, T, 4, 4) float32
    image_sizes: np.ndarray                # shape: (V, 2) int32, (W, H)
    distortion_models: Tuple[str, ...]     # "pinhole" or "kb4"
    distortion_params: np.ndarray          # shape: (V, 12) float64; NaN-padded

    def __post_init__(self) -> None:
        if not (len(self.variant_names) == self.K.shape[0]
                == self.cam_T_world.shape[0]
                == self.image_sizes.shape[0]
                == len(self.distortion_models)
                == self.distortion_params.shape[0]):
            raise ValueError(
                "Cameras: V axis length mismatch across fields"
            )

    def __contains__(self, name: str) -> bool:
        return name in self.variant_names

    def get(self, name: str) -> CameraRecord:
        """Slice out a single camera's record by variant name."""
        try:
            idx = self.variant_names.index(name)
        except ValueError as e:
            raise KeyError(
                f"camera variant {name!r} not in cameras.npz; "
                f"available: {list(self.variant_names)}"
            ) from e
        model = self.distortion_models[idx]
        # Always emit a fixed-length NaN-padded float32 vector; uniform
        # shape across cameras is what makes the field safe under
        # ``default_collate`` regardless of which models the batch mixes.
        # KB4 fills slots 0..3 (``k1..k4``); pinhole leaves all NaN; any
        # other model surfaces every available slot for caller-defined
        # handling. ``distortion_model`` remains the discriminator.
        params = np.full(
            (NUM_DISTORTION_PARAM_SLOTS,), np.nan, dtype=np.float32,
        )
        if model == "kb4":
            params[:4] = self.distortion_params[idx, :4]
        elif model == "pinhole":
            pass  # pinhole takes no distortion parameters; leave NaN
        else:
            params[:] = self.distortion_params[
                idx, :NUM_DISTORTION_PARAM_SLOTS
            ]
        return CameraRecord(
            name=name,
            K=np.asarray(self.K[idx], dtype=np.float32),
            cam_T_world=np.asarray(self.cam_T_world[idx], dtype=np.float32),
            width=int(self.image_sizes[idx, 0]),
            height=int(self.image_sizes[idx, 1]),
            distortion_model=model,
            distortion_params=params,
        )


def load_cameras(path: PathLike) -> Cameras:
    """Parse ``cameras.npz`` into a :class:`Cameras` dataclass.

    The npz layout (V = number of camera variants, T = number of frames):

    * ``camera_index_to_variant_name (V,) <U`` -- canonical names.
    * ``mv_pix_T_cam (V, 4, 4) f64`` -- intrinsics extended to 4x4; the
      pinhole ``K`` is the top-left 3x3 block.
    * ``mv_cam_T_world (V, T, 4, 4) f64`` -- per-frame OpenCV cam-from-world.
    * ``mv_image_size (V, 2) i32`` -- ``(width, height)`` per camera.
    * ``mv_distortion_model (V,) <U`` -- ``""`` (pinhole) or ``"kb4"``;
      the empty-string sentinel is normalised to ``"pinhole"`` in memory
      so downstream code never has to handle ``None``.
    * ``mv_distortion_params (V, 12) f64`` -- KB4 uses ``k1..k4``; trailing
      entries are NaN.
    """
    with np.load(str(path)) as d:
        names = tuple(str(n) for n in d["camera_index_to_variant_name"])
        pix_T_cam = np.asarray(d["mv_pix_T_cam"], dtype=np.float64)
        K = pix_T_cam[:, :3, :3].astype(np.float32, copy=False)
        cam_T_world = np.asarray(
            d["mv_cam_T_world"], dtype=np.float64
        ).astype(np.float32, copy=False)            # shape: (V, T, 4, 4)
        image_sizes = np.asarray(d["mv_image_size"], dtype=np.int32)
        raw_models = [str(m) for m in d["mv_distortion_model"]]
        # Normalise the empty-string pinhole sentinel from disk to an
        # explicit ``"pinhole"`` string so consumers (and PyTorch's
        # ``default_collate``) never see ``None``.
        models = tuple(m if m else "pinhole" for m in raw_models)
        params = np.asarray(d["mv_distortion_params"], dtype=np.float64)
    return Cameras(
        variant_names=names,
        K=K,
        cam_T_world=cam_T_world,
        image_sizes=image_sizes,
        distortion_models=models,
        distortion_params=params,
    )


# ---------------------------------------------------------------------------
# Point tracks (multi-view)
# ---------------------------------------------------------------------------


@dataclass
class PointTracks:
    """Multi-view point tracks loaded from ``point_tracks.h5``.

    Axes:

    * ``V`` = camera variants (matches :attr:`variant_names`).
    * ``T`` = total frames in the sequence.
    * ``N`` = number of tracked query points.

    Visibility / validity semantics:

    * ``visible[v, t, n]`` is True iff point ``n`` is rendered (not occluded
      by another mesh) at frame ``t`` in camera ``v``.
    * ``in_frustum[v, t, n]`` is True iff point ``n`` is in front of camera
      ``v`` AND its projected pixel lies within the image bounds at
      frame ``t``.

    A tracker that loses a point because the ground-truth marks it as
    occluded should not be penalised -- use ``visible & in_frustum`` as
    the evaluation mask.

    The big ``(V, T, N, ...)`` arrays remain *lazy* :class:`h5py.Dataset`
    handles backed by an open HDF5 file (:attr:`_file`). They are only
    materialised when :meth:`slice_variant` indexes into them, which is
    what gives single-camera consumers their 1x I/O cost. The per-track
    scalar arrays are tiny and read eagerly at load time.

    The handle is closed on garbage collection. Do not pickle a
    :class:`PointTracks` across processes — ``h5py.File`` is not
    picklable. Inside a :class:`SchleppDataset` this is fine because each
    worker constructs and consumes the object within a single
    ``__getitem__`` call.
    """

    variant_names: Tuple[str, ...]
    camera_rendered: np.ndarray            # shape: (V,) bool, eager
    categories: np.ndarray                 # shape: (N,) int32, eager
    pass_indices: np.ndarray               # shape: (N,) int32, eager
    actor_idx: np.ndarray                  # shape: (N,) int32 (-1 = scene), eager
    sample_frame: np.ndarray               # shape: (N,) int32, eager
    # Lazy h5py.Dataset handles — actual I/O is deferred until slicing.
    _trajs_2d: "h5py.Dataset"              # shape: (V, T, N, 2) float32 pixels
    _visible: "h5py.Dataset"               # shape: (V, T, N) bool
    _in_frustum: "h5py.Dataset"            # shape: (V, T, N) bool
    _trajs_world: "h5py.Dataset"           # shape: (T, N, 3) float64 metres
    _file: "h5py.File"                     # kept open for lazy reads

    @property
    def num_frames(self) -> int:
        """Convenience: total number of frames (T) in the sequence."""
        return int(self._trajs_2d.shape[1])

    def close(self) -> None:
        """Close the underlying HDF5 file. Safe to call more than once."""
        if getattr(self, "_file", None) is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "PointTracks":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def slice_variant(
        self,
        name: str,
        frame_indices: Optional[Sequence[int]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return per-camera arrays for ``name``, optionally time-sliced.

        Output dict shapes (with ``S = len(frame_indices) or T``):

        * ``target_points`` ``(N, S, 2) float32`` pixel ``(x, y)``.
        * ``visible``      ``(N, S) bool``
        * ``in_frustum``   ``(N, S) bool``
        * ``trajs_world``  ``(N, S, 3) float32``
        * ``categories, pass_indices, actor_idx, sample_frame`` (N,) int32

        Raises
        ------
        RuntimeError
            If the requested variant was not rendered for this sequence
            (``camera_rendered[v] == False``).
        """
        try:
            v = self.variant_names.index(name)
        except ValueError as e:
            raise KeyError(
                f"variant {name!r} not in point_tracks.h5; "
                f"available: {list(self.variant_names)}"
            ) from e
        if not bool(self.camera_rendered[v]):
            rendered = [n for n, r in zip(self.variant_names, self.camera_rendered) if r]
            raise RuntimeError(
                f"variant {name!r} was not rendered for this sequence; "
                f"its V slot in point_tracks.h5 is placeholder. "
                f"Rendered variants: {rendered}"
            )
        if frame_indices is None:
            t_idx: Any = slice(None)
            inverse: Optional[np.ndarray] = None
            S = int(self._trajs_2d.shape[1])
        else:
            sorted_t, inverse = _h5_fancy_index(frame_indices)
            t_idx = sorted_t
            S = int(sorted_t.size)

        # Slice one camera worth from the (V, T, N, 2) chunked dataset.
        # h5py reads exactly the chunk(s) containing v; no other cameras' bytes
        # are touched.
        target_TNC = self._trajs_2d[v, t_idx]                # (S, N, 2) float32
        visible_TN = self._visible[v, t_idx]                 # (S, N) bool
        in_frustum_TN = self._in_frustum[v, t_idx]           # (S, N) bool
        world_TNC = self._trajs_world[t_idx]                 # (S, N, 3) float64

        if inverse is not None:
            target_TNC = target_TNC[inverse]
            visible_TN = visible_TN[inverse]
            in_frustum_TN = in_frustum_TN[inverse]
            world_TNC = world_TNC[inverse]

        # Transpose into the (N, S, *) layout users expect, cast on the way.
        target = np.ascontiguousarray(
            target_TNC.transpose(1, 0, 2).astype(np.float32, copy=False)
        )
        # NaN-stamped placeholder slots should never occur for rendered
        # variants, but guard defensively.
        target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
        visible = np.ascontiguousarray(
            visible_TN.transpose(1, 0).astype(bool, copy=False)
        )
        in_frustum = np.ascontiguousarray(
            in_frustum_TN.transpose(1, 0).astype(bool, copy=False)
        )
        world = np.ascontiguousarray(
            world_TNC.transpose(1, 0, 2).astype(np.float32, copy=False)
        )
        _ = S  # exposed for callers via array shapes; suppresses unused warning
        return {
            "target_points": target,
            "visible":       visible,
            "in_frustum":    in_frustum,
            "trajs_world":   world,
            # Already eagerly-owned copies (read at load_point_tracks time)
            # — hand them out as views; callers can copy if they need to mutate.
            "categories":    self.categories,
            "pass_indices":  self.pass_indices,
            "actor_idx":     self.actor_idx,
            "sample_frame":  self.sample_frame,
        }


def load_point_tracks(path: PathLike) -> PointTracks:
    """Open the multi-view ``point_tracks.h5`` produced for a sequence.

    The returned :class:`PointTracks` holds an open ``h5py.File`` handle
    so that per-camera slicing in :meth:`PointTracks.slice_variant` only
    reads the requested camera's chunk(s) from disk. The handle is closed
    on garbage collection (or via ``pt.close()`` / ``with`` block).
    """
    f = h5py.File(str(path), "r")
    try:
        # Variant names: written as variable-length utf-8; decode via asstr().
        names_ds = f["camera_index_to_variant_name"]
        if h5py.check_string_dtype(names_ds.dtype) is not None:
            names = tuple(names_ds.asstr()[:])
        else:
            # Legacy fixed-width unicode fallback.
            names = tuple(str(n) for n in names_ds[:])
        return PointTracks(
            variant_names=names,
            camera_rendered=np.asarray(f["camera_rendered"][:], dtype=bool),
            categories=np.asarray(f["point_categories"][:], dtype=np.int32),
            pass_indices=np.asarray(f["point_pass_indices"][:], dtype=np.int32),
            actor_idx=np.asarray(f["point_actor_idx"][:], dtype=np.int32),
            sample_frame=np.asarray(f["point_sample_frame"][:], dtype=np.int32),
            _trajs_2d=f["mv_trajs_2d"],
            _visible=f["mv_visibs"],
            _in_frustum=f["mv_valids"],
            _trajs_world=f["trajs_world"],
            _file=f,
        )
    except Exception:
        f.close()
        raise


# ---------------------------------------------------------------------------
# Object bounding boxes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OBBs:
    """Static scene oriented bounding boxes loaded from
    ``object_bounding_boxes.json``.

    ``objects`` entries are plain dicts of the form::

        {
            "uid":      int,
            "category": str,
            "centroid": [3] float,   # world frame, metres
            "extents":  [3] float,   # full extents (not half)
            "rotation": [4] float,   # quaternion XYZW
        }
    """

    world_convention: str
    rotation_convention: str
    objects: List[Dict[str, Any]]


def load_obbs(path: PathLike) -> OBBs:
    """Parse ``object_bounding_boxes.json`` into an :class:`OBBs`."""
    with open(path) as f:
        d = json.load(f)
    return OBBs(
        world_convention=str(d.get("world_convention", "z_up_right_handed_meters")),
        rotation_convention=str(d.get("rotation_convention", "quaternion_xyzw")),
        objects=list(d.get("objects", [])),
    )


# ---------------------------------------------------------------------------
# Per-actor metadata: object animation and MHR parameters
# ---------------------------------------------------------------------------


def _actor_filename(prefix: str, suffix: str, carried_uid: int, is_primary: bool) -> str:
    """Compose the canonical filename for an actor's metadata file.

    Primary actor: ``<prefix>.<suffix>``.
    Secondary actor: ``<prefix>_<carried_uid>.<suffix>``.
    """
    if is_primary:
        return f"{prefix}.{suffix}"
    return f"{prefix}_{carried_uid}.{suffix}"


def _interactions(metadata: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    inter = metadata.get("interactions")
    if not isinstance(inter, list) or not inter:
        raise ValueError(
            "metadata.json has no 'interactions' list; cannot enumerate actors"
        )
    return inter


def load_object_animation(
    sequence_dir: PathLike,
    metadata: Mapping[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """Load every actor's ``object_animation*.npz`` for a sequence.

    Returns
    -------
    dict
        Keyed by the actor id from ``metadata["interactions"][k]["actor_id"]``. Values include:

        * ``uid (int)``      -- carried-object UID
        * ``category (str)``
        * ``role (str)``     -- ``"primary"`` or ``"secondary"``
        * ``frame_start (int)``
        * ``frame_end (int)``
        * ``centroids (T, 3) float64`` -- world-frame positions
        * ``rotations_xyzw (T, 4) float64`` -- world-frame quaternions
        * ``world_convention``, ``rotation_convention``
    """
    sequence_dir = Path(sequence_dir)
    out: Dict[int, Dict[str, Any]] = {}
    for interaction in _interactions(metadata):
        actor_id = int(interaction["actor_id"])
        carried = interaction["carried_object"]
        carried_uid = int(carried["uid"])
        is_primary = bool(interaction.get("is_primary_actor", False))
        fname = _actor_filename(
            "object_animation", "npz", carried_uid, is_primary
        )
        path = sequence_dir / fname
        with np.load(path) as raw:
            keys = set(raw.files)
            centroids = np.asarray(raw["centroids"], dtype=np.float64)
            # On-disk key is ``rotations`` (xyzw quaternion); the in-memory
            # field is renamed to ``rotations_xyzw`` so callers know the order.
            rotations = np.asarray(raw["rotations"], dtype=np.float64)

            def _scalar(key: str, default: Any) -> Any:
                if key not in keys:
                    return default
                val = raw[key]
                if isinstance(val, np.ndarray) and val.shape == ():
                    return val.item()
                return val

            out[actor_id] = {
                "uid":                 int(_scalar("uid", carried_uid)),
                "category":            str(_scalar("category", carried.get("category", ""))),
                "role":                str(_scalar(
                    "role", "primary" if is_primary else "secondary"
                )),
                "frame_start":         int(_scalar("frame_start", 0)),
                "frame_end":           int(_scalar(
                    "frame_end", len(centroids) - 1
                )),
                "centroids":           centroids,                      # (T, 3)
                "rotations_xyzw":      rotations,                      # (T, 4)
                "world_convention":    str(_scalar(
                    "world_convention", "z_up_right_handed_meters"
                )),
                "rotation_convention": str(_scalar(
                    "rotation_convention", "quaternion_xyzw"
                )),
            }
    return out


def load_mhr_params(
    sequence_dir: PathLike,
    metadata: Mapping[str, Any],
) -> Dict[int, Dict[str, Any]]:
    """Load every actor's ``mhr_params*.npz`` for a sequence.

    Returns
    -------
    dict
        Keyed by ``actor_idx``. Each value is a dict with the raw parsed
        numpy arrays from the NPZ:

        * ``shape_params (45,) float32``
        * ``model_params (T, 204) float32``  -- per-frame rig parameters
        * ``expr_params (72,) float32``  -- per-actor expression coeffs
          (no time axis; all-zero in most sequences)
        * ``object_positions (T, 3) float32``
        * ``object_rotations (T, 4) float32`` -- quaternion XYZW
        * ``object_uid (int)``, ``num_frames (int)``, ``fps (float)``
        * any other scalar fields ship verbatim

    All array-valued fields are cast to ``float32`` at load time so the
    on-the-wire shape matches the schlepp dataset's downstream
    expectations; the on-disk NPZ may store some of them at higher
    precision but consumers see ``float32`` uniformly.
    """
    sequence_dir = Path(sequence_dir)
    out: Dict[int, Dict[str, Any]] = {}
    for interaction in _interactions(metadata):
        actor_id = int(interaction["actor_id"])
        carried = interaction["carried_object"]
        carried_uid = int(carried["uid"])
        is_primary = bool(interaction.get("is_primary_actor", False))
        fname = _actor_filename("mhr_params", "npz", carried_uid, is_primary)
        path = sequence_dir / fname
        parsed: Dict[str, Any] = {}
        with np.load(path) as raw:
            keys = set(raw.files)
            for key in ("shape_params", "model_params", "expr_params",
                        "object_positions", "object_rotations",
                        "start_params", "end_params"):
                if key in keys:
                    parsed[key] = np.asarray(raw[key], dtype=np.float32)
            # ``expr_params`` is per-actor by contract -- shape ``(72,)``.
            # The upstream labelling pipeline has historically also produced
            # a constant-in-time ``(T, 72)`` form; the multicam NPZ writer
            # passes that shape through verbatim. We assert single-shape
            # here so any future drift surfaces as a load-time error rather
            # than silently flowing into the dataset (where it used to need
            # a special-case in the temporal slicer).
            if "expr_params" in parsed and parsed["expr_params"].shape != (72,):
                raise ValueError(
                    f"mhr_params[{actor_id}].expr_params has shape "
                    f"{parsed['expr_params'].shape}; expected (72,). "
                    f"File: {path}"
                )
            # Scalar / passthrough fields. NPZ stores scalars as 0-d ndarrays;
            # unwrap via .item() so callers see plain Python types.
            for key in ("object_uid", "num_frames", "fps", "blend_frames",
                        "source_rohm"):
                if key in keys:
                    val = raw[key]
                    if isinstance(val, np.ndarray) and val.shape == ():
                        val = val.item()
                    parsed[key] = val
        out[actor_id] = parsed
    return out


# ---------------------------------------------------------------------------
# MHR Character construction (optional, lazy import)
# ---------------------------------------------------------------------------


def mhr_to_character(
    mhr_params: Mapping[str, Any],
    bundle_dir: PathLike,
    *,
    device: Any = "cpu",
    lod: int = 1,
) -> "Any":
    """Build a ``pymomentum.geometry.Character`` from one actor's MHR params.

    Parameters
    ----------
    mhr_params
        One actor's value dict from :func:`load_mhr_params`.
    bundle_dir
        Directory containing the public MHR asset bundle. The bundle is
        the contents of
        ``https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip``;
        unzip it once and pass the resulting directory.
    device
        torch device for blendshape evaluation (``"cpu"`` or ``"cuda"``).
    lod
        Level of detail (default ``1`` for the body-only mesh).

    Returns
    -------
    pymomentum.geometry.Character
        Loaded character; pose / shape application is performed by the
        caller using ``mhr_params["model_params"]`` /
        ``mhr_params["shape_params"]`` via pymomentum's own APIs.

    Notes
    -----
    Both ``pymomentum`` and ``mhr`` are imported lazily so that
    ``schlepp.io`` remains importable in environments where they are not
    available.
    """
    try:
        import torch
        from mhr.mhr import MHR  # noqa: F401  -- triggers the heavy import path
    except ImportError as e:
        raise ImportError(
            "schlepp.io.mhr_to_character requires the optional 'mhr' and "
            "'pymomentum' packages; install with `pip install schlepp[mhr]` "
            "and download the asset bundle from "
            "https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip"
        ) from e

    bundle_dir = Path(bundle_dir)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(
            f"MHR bundle directory not found: {bundle_dir}"
        )

    if isinstance(device, str):
        device = torch.device(device)

    model = MHR.from_files(
        folder=bundle_dir,
        lod=lod,
        wants_pose_correctives=False,
        device=device,
    )
    # The `Character` lives on `MHR.character`; expose it directly so
    # callers don't have to know about the wrapper.
    return model.character

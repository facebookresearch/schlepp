# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Interactive 3D viewer for a SCHLEPP sequence (rerun backend).

Logs the whole sequence into a `rerun <https://rerun.io>`_ recording:

* All six camera variants as world-anchored frusta, with per-frame pose
  taken from ``cameras.npz``. Optionally attaches each variant's RGB
  frames to its pinhole (``--with-images``) so the 2D image panes and
  the 3D scene stay synchronised through the rerun time-cursor.
* Static and carried oriented bounding boxes from
  ``object_bounding_boxes.json``, with carried-object poses pulled from
  the per-actor ``object_animation*.npz`` files so each box follows its
  actor as you scrub frames.
* One :class:`rerun.Mesh3D` per actor per frame, evaluated through
  pymomentum / MHR using the same parameter-composition recipe as
  :mod:`scripts.precompute_mhr_joints` — pose + shape + expression
  concatenated along the parameter axis, then skinned, then converted
  from MHR's centimetre + Y-up frame to the dataset's metre + Z-up
  world frame.

Open the viewer interactively (requires ``rerun-sdk``; optionally
``pymomentum`` + ``mhr`` for the body meshes)::

    python -m schlepp.visualize.build_scene_in_3d <sequence_dir> \\
        --bundle-dir /path/to/mhr-assets \\
        [--frame 0] [--no-bodies] [--with-images]

Or, to dump a portable recording for the rerun web viewer (no local
viewer spawn)::

    python -m schlepp.visualize.build_scene_in_3d <sequence_dir> \\
        --bundle-dir /path/to/mhr-assets \\
        --save scene.rrd

The resulting ``scene.rrd`` opens directly in https://app.rerun.io
("Open" -> select the file), in the desktop ``rerun scene.rrd``
binary, or in any other rerun client.

The viewer spawns a local rerun window. Use the time slider (or the
left/right arrow keys, or the play button) to scrub through the whole
sequence; toggle individual cameras / actors / boxes via the entity
tree on the left.

Notes on KB4 fisheye Aria variants
----------------------------------
rerun has no built-in KB4 distortion projection, so the Aria SLAM and
RGB rigs are logged with their pinhole ``K`` only (the frustum is still
drawn at the correct world pose). Attached images, if any, are the raw
fisheye frames; for an undistorted view use
:func:`schlepp.utils.aria.undistort_aria_to_pinhole` on the sample
before logging.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from schlepp import io as schlepp_io
from schlepp.visualize._palette import color_for_actor, color_for_category


# ---------------------------------------------------------------------------
# MHR body-mesh evaluation
# ---------------------------------------------------------------------------


def _compose_mhr_params(params: Mapping[str, Any]) -> np.ndarray:
    """Build the ``(T, P + Si + Sf)`` parameter array pymomentum wants.

    Mirrors :func:`scripts.precompute_mhr_joints._process_one`: after
    ``mhr_to_character`` has wired the blend-shape head onto the rig,
    pose / identity-shape / face-expression parameters live in one
    flat axis. ``shape_params`` and ``expr_params`` are per-actor (no
    time axis), so we broadcast them across the ``T`` frames of
    ``model_params``.
    """
    mp = np.asarray(params["model_params"], dtype=np.float32)   # shape: (T, P)
    sp = np.asarray(params["shape_params"], dtype=np.float32)   # shape: (Si,)
    ep = np.asarray(params["expr_params"], dtype=np.float32)    # shape: (Sf,)
    T = mp.shape[0]
    return np.concatenate(
        [mp,
         np.broadcast_to(sp, (T, sp.shape[0])),
         np.broadcast_to(ep, (T, ep.shape[0]))],
        axis=1,
    )                                                            # shape: (T, P+Si+Sf)


def _mhr_to_world_z_up_m(verts: np.ndarray) -> np.ndarray:
    """Convert MHR vertices (cm, Y-up) to dataset world (m, Z-up).

    Identical to the conversion ``scripts.precompute_mhr_joints``
    applies to joint positions, so meshes, joints, OBBs, and camera
    poses share one world frame.
    """
    out = np.stack(
        [verts[..., 0], -verts[..., 2], verts[..., 1]], axis=-1,
    )
    return (out * 0.01).astype(np.float32)


def _pym_skin_points(pym_geometry: Any, character: Any, skel_state: Any) -> np.ndarray:
    """Run pymomentum's vertex-skinning entry, tolerating API drift.

    Different pymomentum releases expose the skinning helper under
    slightly different names. Try the public ones in order; raise
    loudly if none are present so we don't silently fall back to the
    skeleton-state and produce a vertex-less mesh.
    """
    for fn_name in ("skin_points", "skin_with_blends", "skinning"):
        fn = getattr(pym_geometry, fn_name, None)
        if fn is None:
            continue
        try:
            return np.asarray(fn(character, skel_state), dtype=np.float32)
        except TypeError:
            # ``skin_with_blends`` historically took an extra
            # ``model_params`` argument; the signature shift was a
            # one-time API churn. If we hit it, skip and try the next
            # candidate rather than papering over a real call-site bug.
            continue
    raise RuntimeError(
        "pymomentum is installed but exposes no recognised vertex-skinning "
        "entry (looked for `skin_points`, `skin_with_blends`, `skinning`). "
        "Upgrade `pymomentum` or pass --no-bodies."
    )


def _mesh_faces(character: Any) -> np.ndarray:
    """Pull triangle indices off ``character.mesh``, tolerating API drift."""
    mesh = getattr(character, "mesh", None)
    if mesh is None:
        raise RuntimeError(
            "character has no `.mesh` attribute; the loaded LOD may be "
            "skeleton-only. Pass --bundle-dir pointing to a bundle that "
            "ships the body mesh, or use --no-bodies."
        )
    faces = (
        getattr(mesh, "faces", None)
        if getattr(mesh, "faces", None) is not None
        else getattr(mesh, "triangles", None)
    )
    if faces is None:
        raise RuntimeError(
            "character.mesh has neither `.faces` nor `.triangles`; "
            "cannot build body mesh."
        )
    return np.asarray(faces, dtype=np.int32)


def _compute_actor_meshes(
    sequence_dir: Path,
    metadata: Mapping[str, Any],
    bundle_dir: Path,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Evaluate every actor's posed mesh across the full sequence.

    Returns a mapping ``actor_id -> (verts (T, V, 3) float32, faces (F, 3) int32)``
    in the dataset's metre + Z-up world frame.

    Soft-fails: an actor with missing params, an unloadable character,
    or an unsupported pymomentum build is skipped with a warning so a
    partial scene still renders.
    """
    try:
        import torch  # noqa: F401  -- required by pymomentum internals
        import pymomentum.geometry as pym_geometry
    except ImportError as e:
        print(
            f"WARNING: skipping body meshes: pymomentum not installed ({e}). "
            "Install with `pip install schlepp[mhr]`."
        )
        return {}

    try:
        per_actor = schlepp_io.load_mhr_params(sequence_dir, metadata)
    except FileNotFoundError as e:
        print(f"WARNING: no MHR params for {sequence_dir.name}: {e}")
        return {}
    except ValueError as e:
        # metadata.json had no `interactions` list, or one was malformed.
        # That's a soft-skip for the viewer (no actors -> no meshes), not
        # a fatal — cameras + OBBs may still render fine.
        print(f"WARNING: cannot enumerate actors for {sequence_dir.name}: {e}")
        return {}

    fn_skel = getattr(pym_geometry, "model_parameters_to_skeleton_state", None)
    if fn_skel is None:
        print(
            "WARNING: pymomentum has no `model_parameters_to_skeleton_state`; "
            "skipping body meshes."
        )
        return {}

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for actor_id, params in per_actor.items():
        if not {"model_params", "shape_params", "expr_params"}.issubset(params):
            print(
                f"WARNING: actor {actor_id} missing one of model/shape/expr "
                "params; skipping body mesh."
            )
            continue
        try:
            character = schlepp_io.mhr_to_character(
                params, bundle_dir, device="cpu", lod=1,
            )
        except (FileNotFoundError, RuntimeError, ImportError) as e:
            print(f"WARNING: skipping actor {actor_id} body mesh: {e}")
            continue

        full = _compose_mhr_params(params)
        if full.shape[1] != character.parameter_transform.size:
            print(
                f"WARNING: actor {actor_id} composed param size "
                f"{full.shape[1]} != character.parameter_transform.size "
                f"{character.parameter_transform.size}; skipping body mesh."
            )
            continue
        try:
            skel_state = fn_skel(character, full)
            verts_local = _pym_skin_points(pym_geometry, character, skel_state)
            faces = _mesh_faces(character)
        except RuntimeError as e:
            print(f"WARNING: actor {actor_id}: {e}")
            continue
        out[int(actor_id)] = (_mhr_to_world_z_up_m(verts_local), faces)
    return out


# ---------------------------------------------------------------------------
# Per-frame logging helpers (one rerun call site each)
# ---------------------------------------------------------------------------


def _log_static_world(rr: Any) -> None:
    """One-off log of the world view-coordinates so the axes match the dataset."""
    rr.log(
        "world",
        rr.ViewCoordinates.RIGHT_HAND_Z_UP,
        static=True,
    )


def _build_category_class_ids(
    obb_objects: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], List[Any]]:
    """Assign a stable class_id per category and build the rerun annotation table.

    Returns ``(category_to_id, class_descriptions)``. The id ordering is
    deterministic across runs because categories are sorted before being
    numbered.
    """
    import rerun as rr

    categories = sorted({
        str(obb.get("category", "UNKNOWN")) for obb in obb_objects
    })
    category_to_id = {name: idx for idx, name in enumerate(categories)}
    descriptions = [
        rr.ClassDescription(
            info=rr.AnnotationInfo(
                id=idx, label=name, color=color_for_category(name),
            ),
        )
        for name, idx in category_to_id.items()
    ]
    return category_to_id, descriptions


def _obb_entity_path(category: str, uid: int, idx: int) -> str:
    """Stable rerun entity path for one OBB.

    Boxes live under ``world/obbs/<category>/<uid>`` so the entity tree
    groups them by category and each box is individually selectable.
    The category is sanitised against rerun's path separator; the
    array-index suffix is only used when the OBB lacks a usable ``uid``
    (the JSON default of ``-1`` would otherwise collide across multiple
    boxes in the same category).
    """
    cat = str(category).replace("/", "_").strip() or "UNKNOWN"
    if uid < 0:
        return f"world/obbs/{cat}/_idx{idx}"
    return f"world/obbs/{cat}/{uid}"


def _log_one_obb(
    rr: Any,
    path: str,
    center: np.ndarray,
    half: np.ndarray,
    quat_xyzw: np.ndarray,
    class_id: int,
    label: str,
    *,
    static: bool = False,
) -> None:
    """Log a single OBB at one path, batched-size-1 so it stays selectable.

    Pass ``static=True`` for boxes that never move (the rest-pose ones
    from ``object_bounding_boxes.json``); rerun will then keep that
    single log valid for every value of the ``frame`` timeline without
    a per-frame call.
    """
    rr.log(
        path,
        rr.Boxes3D(
            centers=center[None, :].astype(np.float32),
            half_sizes=half[None, :].astype(np.float32),
            quaternions=rr.Quaternion(xyzw=quat_xyzw[None, :].astype(np.float32)),
            class_ids=[int(class_id)],
            labels=[label],
            # Label is attached for hover / selection but suppressed in
            # the 3D viewport by default so a dense scene doesn't drown
            # in text; users can re-enable per-entity from the rerun UI.
            show_labels=False,
        ),
        static=static,
    )


def _log_camera_pose(
    rr: Any,
    variant: str,
    K: np.ndarray,
    cam_T_world_t: np.ndarray,
    width: int,
    height: int,
) -> None:
    """Log one camera's per-frame pose + intrinsic under ``world/cameras/<variant>``.

    ``cam_T_world`` is the dataset's on-disk convention (transform world
    points into the OpenCV camera frame), which is exactly what rerun's
    ``from_parent=True`` Transform3D expects: the transform takes points
    "from parent (world)" to "child (camera)".
    """
    R = cam_T_world_t[:3, :3].astype(np.float32)
    t = cam_T_world_t[:3, 3].astype(np.float32)
    rr.log(
        f"world/cameras/{variant}",
        rr.Transform3D(
            translation=t, mat3x3=R,
            relation=rr.TransformRelation.ChildFromParent,
        ),
    )
    rr.log(
        f"world/cameras/{variant}",
        rr.Pinhole(
            image_from_camera=np.asarray(K, dtype=np.float32),
            resolution=[int(width), int(height)],
        ),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def build_scene(
    sequence_dir,
    *,
    bundle_dir=None,
    initial_frame: int = 0,
    with_images: bool = False,
    spawn: bool = True,
    save_path=None,
    application_id: str = "schlepp",
) -> None:
    """Log a full SCHLEPP sequence into a rerun recording.

    Parameters
    ----------
    sequence_dir
        Path to a sequence root (the directory that holds
        ``metadata.json``, ``cameras.npz``, ``object_bounding_boxes.json``
        and the per-camera subdirs).
    bundle_dir
        Optional path to the unzipped MHR asset bundle (see README).
        When omitted, body meshes are skipped.
    initial_frame
        Seed for the rerun time-cursor; the viewer parks at this frame on
        spawn. ``0`` by default.
    with_images
        If ``True``, attach each camera's RGB frames to its pinhole entity
        so rerun's 2D image panes display them at the active time. Doubles
        the I/O cost so default is ``False``.
    spawn
        If ``True`` (default), spawn a local rerun viewer process and
        stream into it. Ignored when ``save_path`` is set.
    save_path
        Optional path to a ``.rrd`` file. When set, the recording is
        written to disk instead of streamed to a local viewer; load the
        resulting file in the rerun web viewer
        (https://app.rerun.io, "Open" -> select the file), the desktop
        ``rerun <path.rrd>`` binary, or any other rerun client.
    application_id
        Recording application id; surfaces in the rerun UI title bar.
    """
    try:
        import rerun as rr
    except ImportError as e:
        raise ImportError(
            "schlepp.visualize.build_scene_in_3d requires rerun-sdk; "
            "install with `pip install schlepp[viz]`."
        ) from e

    sequence_dir = Path(sequence_dir)
    if not sequence_dir.is_dir():
        raise FileNotFoundError(
            f"sequence directory not found: {sequence_dir}"
        )

    # metadata.json + cameras.npz are both shipped in `*_metadata.tar`
    # and that tar is documented as REQUIRED in the dataset README; the
    # viewer can't run without either, so we surface a friendly error
    # pointing at the right tar rather than the raw FileNotFoundError
    # that would otherwise reach the user.
    metadata_path = sequence_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"{metadata_path.name} missing at {metadata_path}; this file "
            "ships in `*_metadata.tar` (always required, per the dataset "
            "README). Re-run `hf download ... --include \"*_metadata.tar\"`."
        )
    with open(metadata_path) as f:
        metadata = json.load(f)

    cameras_path = sequence_dir / "cameras.npz"
    if not cameras_path.is_file():
        raise FileNotFoundError(
            f"{cameras_path.name} missing at {cameras_path}; this file "
            "ships in `*_metadata.tar` (always required, per the dataset "
            "README). Re-run `hf download ... --include \"*_metadata.tar\"`."
        )
    cams = schlepp_io.load_cameras(cameras_path)

    # OBBs live in `*_mhrand3dbb.tar`. Many users download a subset of
    # chunks (e.g. metadata + rgb only) so this file is often missing in
    # the wild. Soft-fail with a clear hint; the viewer still renders
    # cameras and body meshes.
    obbs_path = sequence_dir / "object_bounding_boxes.json"
    if obbs_path.is_file():
        obbs = schlepp_io.load_obbs(obbs_path)
    else:
        print(
            f"WARNING: {obbs_path.name} missing; skipping OBBs. "
            "Fetch `*_mhrand3dbb.tar` to enable."
        )
        obbs = None

    # Per-actor carried-object animations -> {uid: (centroids, rotations_xyzw)}.
    # Same lookup style as overlay_obbs.py; soft-fails when the npz is
    # missing so the rest of the viewer still renders.
    try:
        animations = schlepp_io.load_object_animation(sequence_dir, metadata)
    except (FileNotFoundError, ValueError):
        animations = {}
    uid_to_anim: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
        int(anim["uid"]): (
            np.asarray(anim["centroids"], dtype=np.float32),
            np.asarray(anim["rotations_xyzw"], dtype=np.float32),
        )
        for anim in animations.values()
    }

    # Body meshes per actor (cached for the full sequence in one shot).
    actor_meshes: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if bundle_dir is not None:
        actor_meshes = _compute_actor_meshes(
            sequence_dir, metadata, Path(bundle_dir),
        )

    # Number of frames is governed by the camera extrinsics — every other
    # field either matches T (point tracks, depth) or is per-sequence
    # static. The MHR params can be shorter (per-actor crops); clamp at
    # log time.
    T = int(cams.cam_T_world.shape[1])

    # --- rerun init ------------------------------------------------------
    # ``save_path`` and the spawned-viewer sink are mutually exclusive in
    # the simple sink model: ``rr.save(...)`` (when used) must be called
    # *before* the first ``rr.log`` so that all subsequent log calls are
    # redirected to the on-disk recording instead of being dropped or
    # streamed to a live viewer.
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rr.init(application_id, spawn=False)
        rr.save(str(save_path))
    else:
        rr.init(application_id, spawn=spawn)
    _log_static_world(rr)

    # OBB annotation context: one class_id per category. Logged once at
    # the root so it's inherited by every Boxes3D entity downstream.
    # Skipped entirely when OBBs were absent on disk.
    if obbs is not None and obbs.objects:
        category_to_id, descriptions = _build_category_class_ids(obbs.objects)
        rr.log("/", rr.AnnotationContext(descriptions), static=True)
        # Pre-flatten per-OBB fields + assign a stable per-box rerun
        # entity path so each box appears as its own selectable node
        # in the entity tree (grouped by category).
        obb_records: List[Dict[str, Any]] = []
        for idx, obb in enumerate(obbs.objects):
            uid = int(obb.get("uid", -1))
            category = str(obb.get("category", "UNKNOWN"))
            obb_records.append({
                "uid":      uid,
                "category": category,
                "centroid": np.asarray(obb["centroid"], dtype=np.float32),
                "rotation": np.asarray(obb["rotation"], dtype=np.float32),
                "half":     np.asarray(obb["extents"], dtype=np.float32) * 0.5,
                "class_id": category_to_id[category if category else "UNKNOWN"],
                "path":     _obb_entity_path(category, uid, idx),
                "label":    f"{category}#{uid}" if uid >= 0 else f"{category}#idx{idx}",
                "animated": uid in uid_to_anim,
            })
    else:
        obb_records = []

    # Static OBBs (no per-frame animation) get logged once with
    # `static=True` so rerun shares the same data across every timeline
    # value. Only animated OBBs need a per-frame log inside the loop
    # below — that cuts per-frame log volume from O(B × T) to O(B_anim × T).
    static_obb_records = [rec for rec in obb_records if not rec["animated"]]
    animated_obb_records = [rec for rec in obb_records if rec["animated"]]
    for rec in static_obb_records:
        _log_one_obb(
            rr, rec["path"], rec["centroid"], rec["half"], rec["rotation"],
            rec["class_id"], rec["label"], static=True,
        )

    # Per-camera intrinsics + sizes resolved once.
    cam_records = [cams.get(name) for name in cams.variant_names]

    # --- per-frame log loop ---------------------------------------------
    for t in range(T):
        rr.set_time("frame", sequence=t)

        # Cameras: pose + pinhole (always); RGB image (opt-in).
        for rec in cam_records:
            _log_camera_pose(
                rr, rec.name, rec.K, rec.cam_T_world[t], rec.width, rec.height,
            )
            if with_images:
                img_path = sequence_dir / rec.name / f"rgb_{t:05d}.png"
                if img_path.is_file():
                    rr.log(
                        f"world/cameras/{rec.name}",
                        rr.Image(schlepp_io.load_rgb(img_path)),
                    )

        # OBBs: animated boxes log per-frame at their own entity path so
        # each remains individually selectable in the rerun entity tree.
        # Static boxes were already logged once with `static=True` above.
        for rec in animated_obb_records:
            cent_t, rot_t = uid_to_anim[rec["uid"]]
            t_idx = min(t, cent_t.shape[0] - 1)
            _log_one_obb(
                rr, rec["path"], cent_t[t_idx], rec["half"], rot_t[t_idx],
                rec["class_id"], rec["label"],
            )

        # Per-actor body meshes. Faces are reused across frames; vertices
        # are clamped to the per-actor T.
        for actor_id, (verts, faces) in actor_meshes.items():
            t_a = min(t, verts.shape[0] - 1)
            color = color_for_actor(actor_id)
            rr.log(
                f"world/actors/{actor_id}/mesh",
                rr.Mesh3D(
                    vertex_positions=verts[t_a],
                    triangle_indices=faces,
                    vertex_colors=np.tile(
                        np.asarray(color, dtype=np.uint8), (verts.shape[1], 1),
                    ),
                ),
            )

    # Park the viewer's time-cursor at the requested initial frame.
    # rerun honours the last `set_time` for the default time when the
    # recording is consumed live via `spawn=True`.
    rr.set_time("frame", sequence=int(np.clip(initial_frame, 0, T - 1)))

    print(
        f"logged {T} frames | cameras={len(cam_records)} | "
        f"obbs={len(obb_records)} | actors={len(actor_meshes)}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("sequence_dir", type=Path)
    p.add_argument(
        "--bundle-dir", type=Path, default=None,
        help="Path to the unzipped MHR asset bundle (see README). "
             "Omit (or pass --no-bodies) to render cameras + OBBs only.",
    )
    p.add_argument(
        "--frame", type=int, default=0,
        help="Initial frame the rerun time-cursor parks on at spawn.",
    )
    p.add_argument(
        "--no-bodies", action="store_true",
        help="Render cameras + OBBs only; skip per-actor body meshes "
             "even when --bundle-dir is set.",
    )
    p.add_argument(
        "--with-images", action="store_true",
        help="Attach each camera's per-frame RGB to its rerun pinhole "
             "entity so the 2D image panes follow the time-cursor. "
             "Roughly doubles disk I/O during logging.",
    )
    p.add_argument(
        "--save", dest="save_path", type=Path, default=None,
        metavar="PATH.rrd",
        help="Write the recording to a `.rrd` file instead of spawning "
             "the local rerun viewer. The resulting file can be opened "
             "in the rerun web viewer at https://app.rerun.io, in the "
             "desktop `rerun <PATH.rrd>` binary, or in any other "
             "rerun client.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    bundle = None if args.no_bodies else args.bundle_dir
    build_scene(
        args.sequence_dir,
        bundle_dir=bundle,
        initial_frame=args.frame,
        with_images=args.with_images,
        save_path=args.save_path,
    )
    if args.save_path is not None:
        print(
            f"wrote {args.save_path}; open in the web viewer at "
            "https://app.rerun.io or with `rerun "
            f"{args.save_path}`"
        )


if __name__ == "__main__":
    main()

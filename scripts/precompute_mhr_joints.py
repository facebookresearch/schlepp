# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Offline: forward MHR ``model_params`` through pymomentum to produce
per-actor, per-frame world-space joint positions, and dump one
``joints.json`` per sequence at the sequence root.

After running this once per data root you can train a 3D pose model
without ever importing ``pymomentum`` in the training loop -- the
README's "Human Pose Estimation (MHR 3D Joints)" example just reads the
JSON in its ``transform=`` callable.

Usage
-----
::

    python -m scripts.precompute_mhr_joints \\
        --data-root /path/to/schlepp \\
        --bundle-dir /path/to/mhr_assets \\
        [--sequences id1,id2,...] \\
        [--device cuda] [--lod 1] [--num-workers 8] [--overwrite]

Output
------
``<data_root>/<sequence_id>/joints.json``::

    {
      "joint_names": ["root", "spine", ...],            # (J,)
      "actors": {
        "0": {"joints_world": [[[x, y, z], ...J], ...T]},   # (T, J, 3) metres
        "1": {...}
      }
    }

Requirements
------------
``pip install schlepp[mhr]`` plus the public MHR asset bundle (see the
README's "Install" section).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from schlepp import io as schlepp_io

logger = logging.getLogger("precompute_mhr_joints")


def _resolve_sequences(
    data_root: Path,
    explicit: Optional[List[str]],
) -> List[str]:
    if explicit:
        return explicit
    index_path = data_root / "index.parquet"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"{index_path} not found; pass --sequences to override"
        )
    df = pd.read_parquet(index_path)
    if "sequence_id" not in df.columns:
        raise KeyError("index.parquet missing 'sequence_id' column")
    return df["sequence_id"].astype(str).tolist()


def _process_one(
    args: tuple,
) -> tuple:
    """Worker: returns ``(sequence_id, status, message)``.

    Imports torch / pymomentum lazily so the parent process stays
    light and so worker initialisation cost is paid in parallel.
    """
    sequence_id, data_root, bundle_dir, device, lod, overwrite = args
    import torch  # noqa: F401
    import pymomentum.geometry as pym_geometry

    seq_dir = Path(data_root) / sequence_id
    out_path = seq_dir / "joints.json"
    if out_path.exists() and not overwrite:
        return (sequence_id, "skip", f"{out_path} already exists")

    metadata_path = seq_dir / "metadata.json"
    if not metadata_path.is_file():
        return (sequence_id, "skip", f"no metadata.json at {seq_dir}")
    with open(metadata_path) as f:
        metadata = json.load(f)

    try:
        per_actor = schlepp_io.load_mhr_params(seq_dir, metadata)
    except FileNotFoundError as e:
        return (sequence_id, "skip", f"no mhr_params: {e}")

    if not per_actor:
        return (sequence_id, "skip", "no MHR actors")

    out: dict = {"actors": {}}
    joint_names: Optional[List[str]] = None

    fn = getattr(
        pym_geometry, "model_parameters_to_skeleton_state", None
    )
    if fn is None:
        return (
            sequence_id, "error",
            "pymomentum has no `model_parameters_to_skeleton_state`; "
            "this script targets the public MHR v1 API",
        )

    for actor_id, params in per_actor.items():
        if not {"model_params", "shape_params", "expr_params"}.issubset(params):
            continue
        try:
            character = schlepp_io.mhr_to_character(
                params, bundle_dir, device=device, lod=lod,
            )
        except (FileNotFoundError, RuntimeError, ImportError) as e:
            return (sequence_id, "error", f"actor {actor_id}: {e}")

        # ``mhr_to_character`` calls ``with_blend_shape`` so the rig now
        # expects pose + identity-shape + face-expression coefficients
        # concatenated along the parameter axis. ``shape_params`` and
        # ``expr_params`` are per-actor (no time axis); broadcast them
        # to every frame.
        mp = np.asarray(params["model_params"], dtype=np.float32)  # shape: (T, P)
        sp = np.asarray(params["shape_params"], dtype=np.float32)  # shape: (Si,)
        ep = np.asarray(params["expr_params"],  dtype=np.float32)  # shape: (Sf,)
        T = mp.shape[0]
        full = np.concatenate(
            [mp,
             np.broadcast_to(sp, (T, sp.shape[0])),
             np.broadcast_to(ep, (T, ep.shape[0]))],
            axis=1,
        )                                                     # shape: (T, P+Si+Sf)
        if full.shape[1] != character.parameter_transform.size:
            return (
                sequence_id, "error",
                f"actor {actor_id}: composed param size {full.shape[1]} "
                f"!= character.parameter_transform.size "
                f"{character.parameter_transform.size}",
            )

        skel_state = fn(character, full)                      # shape: (T, J, 8)
        joints_mhr = skel_state[..., :3].astype(np.float32)   # shape: (T, J, 3)
        # MHR ships joint positions in DH-flavour centimetres with Y-up;
        # convert to the dataset's metres + Z-up convention so the
        # joints share the world frame of ``cam_T_world`` / OBB
        # centroids / point tracks.
        joints_world = np.stack(
            [joints_mhr[..., 0],
             -joints_mhr[..., 2],
             joints_mhr[..., 1]],
            axis=-1,
        ) * 0.01                                              # shape: (T, J, 3)

        if joint_names is None and hasattr(character, "skeleton"):
            jn = getattr(character.skeleton, "joint_names", None)
            if jn is not None:
                joint_names = [str(n) for n in jn]

        out["actors"][str(int(actor_id))] = {
            "joints_world": joints_world.astype(np.float32).tolist(),
        }

    if not out["actors"]:
        return (sequence_id, "skip", "no actors had model_params")

    out["joint_names"] = joint_names or []

    tmp = out_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f)
    tmp.replace(out_path)
    return (sequence_id, "ok", f"wrote {out_path}")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", type=Path, required=True,
                        help="SCHLEPP root containing index.parquet")
    parser.add_argument("--bundle-dir", type=Path, required=True,
                        help="MHR asset bundle directory")
    parser.add_argument(
        "--sequences", default=None,
        help="Comma-separated sequence_ids; defaults to every row in index.parquet",
    )
    parser.add_argument("--device", default="cpu",
                        help="Torch device for the MHR character (cpu/cuda)")
    parser.add_argument("--lod", type=int, default=1,
                        help="MHR level of detail (1 = body)")
    parser.add_argument("--num-workers", type=int, default=8,
                        help="Process-pool size; >1 uses ProcessPoolExecutor")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute joints.json even when it already exists")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    explicit = (
        [s for s in args.sequences.split(",") if s]
        if args.sequences else None
    )
    sequences = _resolve_sequences(args.data_root, explicit)
    logger.info("processing %d sequence(s)", len(sequences))

    work = [
        (s, args.data_root, args.bundle_dir, args.device, args.lod, args.overwrite)
        for s in sequences
    ]
    counts = {"ok": 0, "skip": 0, "error": 0}

    if args.num_workers > 1 and len(work) > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(_process_one, w): w[0] for w in work}
            for fut in as_completed(futures):
                sid, status, msg = fut.result()
                counts[status] += 1
                logger.log(
                    logging.INFO if status != "error" else logging.ERROR,
                    "[%s] %s: %s", status, sid, msg,
                )
    else:
        for w in work:
            sid, status, msg = _process_one(w)
            counts[status] += 1
            logger.log(
                logging.INFO if status != "error" else logging.ERROR,
                "[%s] %s: %s", status, sid, msg,
            )

    logger.info(
        "done: %d ok, %d skip, %d error",
        counts["ok"], counts["skip"], counts["error"],
    )
    return 0 if counts["error"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

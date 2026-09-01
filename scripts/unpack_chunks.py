# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Unpack a chunked SCHLEPP download in place into the canonical sequence tree.

Turns a download tree shaped like::

    ROOT/
    ├── chunk_0001/
    │   ├── index.parquet
    │   ├── chunk_0001_metadata.tar
    │   ├── chunk_0001_static_rgb.tar
    │   └── ... (per-camera-per-modality + per-sequence tars)
    ├── chunk_0002/
    │   └── ...
    └── index.parquet

into the flat sequence tree the rest of the package expects::

    ROOT/
    ├── 1444-RES_THEATRE__obj54__cam03__v02/
    │   ├── metadata.json
    │   ├── cameras.npz
    │   ├── point_tracks.h5
    │   ├── object_bounding_boxes.json
    │   ├── mhr_params*.npz
    │   ├── object_animation*.npz
    │   ├── static/
    │   │   ├── rgb_00000.png
    │   │   ├── depth.dpt5
    │   │   └── ...
    │   ├── aria_rgb/
    │   │   └── ...
    │   └── ...
    └── index.parquet     (subset of the original, restricted to on-disk sequences)

Design choices
--------------
* **In place.** No separate source / destination directories: extracted sequence
  dirs sit alongside the ``chunk_*/`` source dirs. Camera-variant directories
  inside a sequence are named with underscores (``aria_rgb``); the tar
  filenames use the underscore-less form (``ariargb``). The two flavours don't
  collide with sequence-id names, which start with a digit.

* **Tar deletion is the work marker.** A successfully extracted tar is deleted
  from the chunk dir. Re-running the script naturally only processes tars that
  are still on disk -- either freshly downloaded ones, or ones that failed to
  extract on a previous run (in which case the existing partial extraction is
  simply overwritten).

* **Parallel by tar.** Each tar is a self-contained unit handed to a thread
  pool worker. No two tars target the same destination file by construction.

* **Index regeneration.** After unpack, ``ROOT/index.parquet`` is rewritten
  to the union of (a) the existing ``ROOT/index.parquet`` and (b) every
  ``ROOT/chunk_*/index.parquet`` still on disk, deduplicated on
  ``sequence_id`` and filtered to sequences whose
  ``<sequence_id>/metadata.json`` exists. This keeps the on-disk index
  aligned with the downloaded subset and naturally absorbs new chunks on
  subsequent runs (their per-chunk indices contribute the new rows;
  existing rows from previously unpacked chunks survive because they
  live in ``ROOT/index.parquet``).

Example
-------
::

    python -m scripts.unpack_chunks /data/schlepp --jobs 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path
from typing import Optional

# Tar filename form -> on-disk camera dir name. The download script uses the
# underscore-less form for shell-glob friendliness; the dataset code expects
# the underscored form. Single source of truth.
_CAMERA_TAR_TO_DIR = {
    "static":      "static",
    "bodyfollow":  "body_follow",
    "objectorbit": "object_orbit",
    "ariaslamL":   "aria_slamL",
    "ariaslamR":   "aria_slamR",
    "ariargb":     "aria_rgb",
}
_CAMERAS = tuple(_CAMERA_TAR_TO_DIR.keys())

# Per-camera modalities (tar form). Members of these tars land inside the
# per-camera subdir of the sequence.
_PER_CAMERA_MODALITIES = (
    "rgb", "depth", "segmentation", "forwardflow", "backwardflow",
)

# Per-sequence tars whose members land at the sequence root.
_SEQUENCE_ROOT_TARS = ("metadata", "pointtracks", "mhrand3dbb")

# chunk_NNNN_<camera>_<modality>.tar
_PER_CAMERA_RE = re.compile(
    r"^chunk_\d+_(?P<camera>" + "|".join(_CAMERAS) + r")_"
    r"(?P<modality>" + "|".join(_PER_CAMERA_MODALITIES) + r")\.tar$"
)
# chunk_NNNN_<seq_root_kind>.tar
_SEQUENCE_ROOT_RE = re.compile(
    r"^chunk_\d+_(?P<kind>" + "|".join(_SEQUENCE_ROOT_TARS) + r")\.tar$"
)
_CHUNK_DIR_RE = re.compile(r"^chunk_\d+$")


def _dest_subdir(tar_name: str) -> Optional[str]:
    """Return the per-sequence subdir for ``tar_name``, or ``None`` if it
    lives at the sequence root. Returns ``""`` semantically for root tars
    and the camera dir name for per-camera tars. ``None`` means the tar's
    name does not match any expected pattern and it should be skipped.
    """
    m = _PER_CAMERA_RE.match(tar_name)
    if m:
        return _CAMERA_TAR_TO_DIR[m["camera"]]
    if _SEQUENCE_ROOT_RE.match(tar_name):
        return ""
    return None


# Member basenames that the packer puts inside per-camera tars but that
# semantically live at the sequence root. The segmentation tars bundle
# the seq-level ``segmentation_labels.json`` so a user fetching only
# ``*_<cam>_segmentation.tar`` still gets the integer-label lookup (see
# deliver_batch.py:pack_family_modality_tar). We undo that bundling on
# extract so the file lands where the dataset loader expects it.
_PER_CAMERA_TAR_SEQ_ROOT_FILES = frozenset({"segmentation_labels.json"})


def _unpack_tar(tar_path: Path, root: Path, keep_tars: bool) -> tuple[Path, bool, str]:
    """Extract a single tar in place; delete it on success.

    Members are rewritten so that ``<sequence_id>/<member>`` either lands at
    ``<root>/<sequence_id>/<member>`` (sequence-root tars) or at
    ``<root>/<sequence_id>/<camera_dir>/<member>`` (per-camera tars).
    """
    subdir = _dest_subdir(tar_path.name)
    if subdir is None:
        return tar_path, False, "unrecognised tar name; skipping"

    root_resolved = root.resolve()
    n = 0
    try:
        with tarfile.open(tar_path, mode="r") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                parts = Path(member.name).parts
                if len(parts) < 2:
                    return tar_path, False, f"bad member layout: {member.name!r}"
                seq_id, *rest = parts
                # Per-camera tars also bundle a handful of seq-root files
                # (see _PER_CAMERA_TAR_SEQ_ROOT_FILES). Don't inject the
                # camera dir for those -- they belong at the seq root.
                member_subdir = (
                    "" if (subdir and len(rest) == 1
                           and rest[0] in _PER_CAMERA_TAR_SEQ_ROOT_FILES)
                    else subdir
                )
                rel = (
                    Path(seq_id, member_subdir, *rest)
                    if member_subdir else Path(seq_id, *rest)
                )
                dest = (root / rel).resolve()
                # Defence in depth against malicious member names.
                if not str(dest).startswith(str(root_resolved) + "/"):
                    return tar_path, False, (
                        f"member escapes root: {member.name!r} -> {dest}"
                    )
                dest.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(member)
                if src is None:
                    return tar_path, False, f"non-regular member: {member.name!r}"
                with open(dest, "wb") as out:
                    shutil.copyfileobj(src, out, length=1 << 20)
                n += 1
    except (OSError, tarfile.TarError) as e:
        return tar_path, False, f"{type(e).__name__}: {e}"

    if not keep_tars:
        try:
            tar_path.unlink()
        except OSError as e:
            return tar_path, False, f"extracted ok but failed to delete tar: {e}"
    return tar_path, True, f"{n} files"


def _regen_index(root: Path) -> tuple[int, int]:
    """Rewrite ``root/index.parquet`` to the on-disk subset.

    Sources, in priority order:

    * ``root/index.parquet`` (the previous regen output, or the canonical
      upstream index on first run) -- preserved across runs so previously
      unpacked chunks stay represented even after their ``chunk_*/`` dirs
      have been swept.
    * Every ``root/chunk_*/index.parquet`` still on disk -- contributes
      rows for chunks downloaded since the last unpack run.

    Both are concatenated and deduplicated on ``sequence_id`` (existing
    rows win over per-chunk rereads). The result is then filtered to
    sequences whose ``<sequence_id>/metadata.json`` exists on disk so
    half-unpacked or manually removed sequences are pruned. Returns
    ``(kept, total_after_union)``.

    On-disk presence is computed with a single ``scandir(root)`` plus one
    ``isfile`` per candidate sequence dir, so per-row stat() cost stays
    bounded by the on-disk corpus size rather than the (potentially much
    larger) canonical index.
    """
    import pandas as pd

    frames: list[pd.DataFrame] = []
    root_index = root / "index.parquet"
    if root_index.is_file():
        frames.append(pd.read_parquet(root_index))
    for chunk_dir in sorted(root.iterdir()):
        if not (chunk_dir.is_dir() and _CHUNK_DIR_RE.match(chunk_dir.name)):
            continue
        per_chunk = chunk_dir / "index.parquet"
        if per_chunk.is_file():
            frames.append(pd.read_parquet(per_chunk))

    if not frames:
        print(
            f"warn: no index.parquet at {root} or in any chunk_*/; "
            f"cannot regenerate index",
            file=sys.stderr,
        )
        return 0, 0

    df = (
        pd.concat(frames, ignore_index=True, sort=False)
        .drop_duplicates(subset=["sequence_id"], keep="first")
        .reset_index(drop=True)
    )
    total = len(df)

    # Single root scan; per-candidate isfile is bounded by on-disk count,
    # not by total (which can be huge if existing == upstream canonical).
    present = {
        e.name for e in os.scandir(root)
        if e.is_dir() and (root / e.name / "metadata.json").is_file()
    }
    kept_df = df[df["sequence_id"].astype(str).isin(present)].reset_index(
        drop=True
    )
    kept_df.to_parquet(root_index, index=False)
    return len(kept_df), total


def _sweep_empty_chunk_dirs(root: Path) -> int:
    """Remove ``chunk_*/`` dirs that have no files left other than their own
    chunk-level ``index.parquet``. Returns the number of dirs removed.
    """
    removed = 0
    for entry in sorted(root.iterdir()):
        if not (entry.is_dir() and _CHUNK_DIR_RE.match(entry.name)):
            continue
        remaining = [p for p in entry.iterdir() if p.name != "index.parquet"]
        if remaining:
            continue
        # Empty modulo the per-chunk index; safe to drop the whole thing --
        # the regenerated top-level index supersedes it.
        shutil.rmtree(entry)
        removed += 1
    return removed


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="scripts.unpack_chunks",
        description=(
            "Unpack a chunked SCHLEPP download in place into the flat "
            "<sequence_id>/<camera>/ layout. Successfully extracted tars "
            "are deleted; the top-level index.parquet is filtered to the "
            "sequences present on disk. Re-runnable; picks up additional "
            "tars that arrive between runs."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "root",
        type=Path,
        help="Download root (contains chunk_*/ and index.parquet).",
    )
    p.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Number of tars to extract in parallel (default: 8).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List the tars that would be unpacked and exit.",
    )
    p.add_argument(
        "--keep-tars",
        action="store_true",
        help="Don't delete tars after a successful extract (debug).",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"error: root {root} is not a directory", file=sys.stderr)
        return 2

    tars: list[Path] = []
    for chunk_dir in sorted(root.iterdir()):
        if not (chunk_dir.is_dir() and _CHUNK_DIR_RE.match(chunk_dir.name)):
            continue
        for p in sorted(chunk_dir.iterdir()):
            if p.suffix == ".tar":
                tars.append(p)

    if not tars:
        print("Nothing to unpack.", file=sys.stderr)
    else:
        print(f"Found {len(tars)} tar(s) under {root}", file=sys.stderr)

    if args.dry_run:
        for t in tars:
            print(t.relative_to(root))
        return 0

    failures = 0
    if tars:
        with cf.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futs = {
                pool.submit(_unpack_tar, t, root, args.keep_tars): t
                for t in tars
            }
            for i, fut in enumerate(cf.as_completed(futs), start=1):
                tar = futs[fut]
                _, ok, msg = fut.result()
                tag = "ok  " if ok else "FAIL"
                print(
                    f"[{i:>5d}/{len(tars)}] {tag} "
                    f"{tar.relative_to(root)} ({msg})",
                    file=sys.stderr,
                )
                if not ok:
                    failures += 1

    kept, total = _regen_index(root)
    if total:
        print(
            f"Regenerated index.parquet: {kept}/{total} sequences present.",
            file=sys.stderr,
        )

    removed = _sweep_empty_chunk_dirs(root)
    if removed:
        print(f"Removed {removed} empty chunk_*/ dir(s).", file=sys.stderr)

    if failures:
        print(
            f"{failures}/{len(tars)} tars failed; rerun to retry.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

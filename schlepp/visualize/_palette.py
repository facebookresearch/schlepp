# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Stable colour palettes for the visualisation suite.

Both the 2D OBB overlay (:mod:`schlepp.visualize.overlay_obbs`) and the 3D
rerun viewer (:mod:`schlepp.visualize.build_scene_in_3d`) need to colour
entities consistently across runs and across modules. We use a
deterministic HSV hash over a string seed so that:

* re-running the visualisers picks the same colour for the same category
  / actor without persisting state to disk;
* the 2D and 3D viewers agree on colours so a user can match a box in the
  overlay video against the same box in the 3D scene.

All public helpers return RGB ``(r, g, b)`` integer triplets in ``0..255``.
OpenCV consumers should swap to BGR at the call site (see
:func:`schlepp.visualize.overlay_obbs._color_for_category`).
"""
from __future__ import annotations

import colorsys
import hashlib
from typing import Tuple

RGB = Tuple[int, int, int]


def _hsv_hash_rgb(seed: str, saturation: float, value: float) -> RGB:
    """Hash ``seed`` to a stable hue and return an RGB triplet.

    ``saturation`` and ``value`` are passed through verbatim to
    :func:`colorsys.hsv_to_rgb`. We use an MD5 hash truncated to 32 bits so
    that small string changes (e.g. ``"box"`` vs ``"boxes"``) map to very
    different hues; ``hash(...)``'s Python-version-dependent randomisation
    would defeat the cross-run stability property we want.
    """
    h = int(hashlib.md5(seed.encode()).hexdigest()[:8], 16)
    hue = (h % 360) / 360.0
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def color_for_category(category: str) -> RGB:
    """Stable RGB colour for a SCHLEPP OBB category name.

    The category strings ride on disk in
    ``object_bounding_boxes.json`` so we keep the hash seed identical
    to whatever the user sees in the JSON. Saturation / value are tuned
    for readable wireframes over photographic backgrounds.
    """
    return _hsv_hash_rgb(category, saturation=0.75, value=1.0)


def color_for_actor(actor_id: int) -> RGB:
    """Stable RGB colour for an actor index.

    Seeded with ``"actor:{actor_id}"`` so the hue distribution doesn't
    collide with the OBB category palette when both are shown in the
    same 3D scene.
    """
    return _hsv_hash_rgb(f"actor:{int(actor_id)}", saturation=0.55, value=1.0)


__all__ = ["RGB", "color_for_category", "color_for_actor"]

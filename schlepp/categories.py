# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Canonical SCHLEPP point-category schema.

These integer IDs are written into `point_tracks.h5` as the ``categories``
field, and stamped into each sequence's ``metadata.json`` under
``point_categories_schema``. This module is the single source of truth
for the mapping; visualisation, eval, and filtering utilities all import
from here.
"""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

# ---------------------------------------------------------------------------
# Integer category IDs.
# ---------------------------------------------------------------------------

BODY: int = 1
CLOTH: int = 2
CARRIED: int = 3
SCENE: int = 4

#: All known category IDs in canonical order.
ALL_CATEGORIES: Tuple[int, ...] = (BODY, CLOTH, CARRIED, SCENE)

#: Human-readable label per category ID.
CATEGORY_NAMES: Dict[int, str] = {
    BODY:    "body",
    CLOTH:   "cloth",
    CARRIED: "carried",
    SCENE:   "scene",
}

#: Reverse lookup: name (case-insensitive) -> category ID.
CATEGORY_IDS: Mapping[str, int] = {v: k for k, v in CATEGORY_NAMES.items()}


def resolve_category(value) -> int:
    """Normalise a single category specifier into an integer ID.

    Accepts an integer ID, a known category name (case-insensitive), or
    anything that ``int()`` can parse. Raises :class:`ValueError` for
    unknown names / IDs.
    """
    if isinstance(value, str):
        key = value.strip().lower()
        if key not in CATEGORY_IDS:
            raise ValueError(
                f"Unknown category name {value!r}; known: {list(CATEGORY_IDS)}"
            )
        return CATEGORY_IDS[key]
    cid = int(value)
    if cid not in CATEGORY_NAMES:
        raise ValueError(
            f"Unknown category id {cid}; known: {list(CATEGORY_NAMES)}"
        )
    return cid


def resolve_categories(values) -> Tuple[int, ...]:
    """Vectorised :func:`resolve_category`; returns a tuple of unique IDs."""
    if isinstance(values, (str, int)):
        values = (values,)
    seen = []
    for v in values:
        cid = resolve_category(v)
        if cid not in seen:
            seen.append(cid)
    return tuple(seen)


__all__ = [
    "BODY",
    "CLOTH",
    "CARRIED",
    "SCENE",
    "ALL_CATEGORIES",
    "CATEGORY_NAMES",
    "CATEGORY_IDS",
    "resolve_category",
    "resolve_categories",
]

# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Utility subpackage for SCHLEPP.

Pure functions that compose inside the user's transform; no transform-class
machinery. Imports stay lazy so that the optional submodules don't pull
heavy deps (open3d, cv2 -- only OpenCV-imported `spatial`, etc.) at the
top level.
"""
from __future__ import annotations

from schlepp.utils.aria import (
    kb4_to_pinhole_remap,
    pinhole_target_for_kb4,
    undistort_aria_to_pinhole,
)
from schlepp.utils.spatial import (
    center_crop_sample,
    crop_sample,
    pad_sample,
    random_crop_sample,
    resize_sample,
)
from schlepp.utils.tracks import (
    filter_tracks_by_category,
    filter_tracks_by_motion,
    filter_tracks_by_visibility,
    query_points_from_first_frame,
    split_tracks_query_target,
    subsample_tracks,
)

__all__ = [
    # tracks
    "filter_tracks_by_category",
    "filter_tracks_by_motion",
    "filter_tracks_by_visibility",
    "query_points_from_first_frame",
    "split_tracks_query_target",
    "subsample_tracks",
    # spatial
    "resize_sample",
    "crop_sample",
    "center_crop_sample",
    "random_crop_sample",
    "pad_sample",
    # aria
    "undistort_aria_to_pinhole",
    "pinhole_target_for_kb4",
    "kb4_to_pinhole_remap",
]

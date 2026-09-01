# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Schlepp: loaders, geometry, and visualisation utilities for the SCHLEPP dataset.

The public surface is intentionally small. Import the heavyweight modules
(:mod:`schlepp.eval`, :mod:`schlepp.visualize`) only as you need them so that
optional dependencies (open3d, pymomentum, ...) remain lazy.
"""

from schlepp.dataset import SchleppDataset, skip_none_collate

__all__ = ["SchleppDataset", "skip_none_collate"]
__version__ = "0.1.0"

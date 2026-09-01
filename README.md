<!-- Copyright (c) Meta Platforms, Inc. and affiliates. -->
## Schlepp: A Multi-View Synthetic Dataset of Humans Carrying Objects Around

Work in progress: this repository is being prepared for release and is not expected to be fully runnable yet.

TODO TEASER IMAGE

[Project Page](http://facebookresearch.github.io/schlepp)  |  [Paper](#) | [Arxiv](#)  |    [Dataset (HF)](https://huggingface.co/datasets/facebook/schlepp)  |    [Dataset (Aria Dataset Explorer)](https://explorer.projectaria.com/schlepp)

`schlepp` is the Python package for the SCHLEPP synthetic
dataset, which features multi-camera, multi-actor, multi-modal object manipulation
sequences. It provides dense ground-truth optical flow, depth, segmentation, point
tracks, oriented bounding boxes, object animation, and MHR body parameters.

---

## Download

The dataset is hosted on Hugging Face under `facebook/schlepp` as
chunks plus a top-level `index.parquet`. When using the Hugging Face
CLI `hf`, you can specifically request which cameras and modalities to
download by using `--include`.

- Per-camera per-modality: `*_<camera>_<modality>.tar`
- Per-sequence (camera-independent): `*_<modality>.tar`

Available cameras: `static`, `bodyfollow`, `objectorbit`, `ariaslamL`,
`ariaslamR`, `ariargb`.

Available modalities: `rgb`, `depth`, `segmentation`, `forwardflow`, `backwardflow`,
`pointtracks`, `mhrand3dbb` (MHR parameters, object animations, and object OBB).

Note that once you are using this dataset, cameras and modalities have names
with underscores (consult the tables below).

As an example, to fetch data to train a point tracker on Aria RGB (RGB
frames, point tracks, per-sequence metadata incl. cameras, and the index):

```bash
hf download facebook/schlepp \
    --repo-type dataset \
    --local-dir /data/schlepp \
    --include "*_ariargb_rgb.tar" \
            "*_pointtracks.tar" \
            "*_metadata.tar" \
            "*.parquet"
```

Make sure to always include `"*_metadata.tar" "*.parquet"` files as they
are required by this package to construct the dataset. Chunks are
stratified by the main carried object category. Therefore, you may
download a portion of this dataset and still obtain a reasonably
representative subset.

### Unpacking

To unpack the data in-place, use the following command:

```bash
python -m scripts.unpack_chunks /data/schlepp --jobs 8
```

The script is safe to re-run if you download new chunks,
cameras, or modalities.

---

## Install

```bash
pip install -e .
# optional extras:
pip install -e ".[viz]"      # open3d + trimesh for the 3D visualisers
pip install -e ".[mhr]"      # mhr + pymomentum for MHR body evaluation
```

If you want to interface with the [Momentum Human Rig](https://github.com/facebookresearch/MHR)
(MHR), download and unpack the model assets:

```bash
curl -OL https://github.com/facebookresearch/MHR/releases/download/v1.0.0/assets.zip
unzip assets.zip
```

---

## Coordinate System Conventions

All ground truth in the dataset uses the same conventions:

- **World frame**: Z-up, right-handed, metres.
- **Camera frame**: OpenCV — X right, Y down, Z forward.
- **Extrinsic** on disk is `cam_T_world` (4×4 SE(3)), read as "transform
  _to_ cam _from_ world":
  ```
  p_cam = cam_T_world[:3, :3] @ p_world + cam_T_world[:3, 3]
  ```
- **Pinhole** intrinsic: `pixel = (K @ (p_cam / p_cam[2]))[:2]`. `K` is the
  top-left 3×3 of the on-disk `pix_T_cam`.
- **Aria Gen 2-like KB4** intrinsic: Kannala–Brandt equidistant fisheye
  with four radial coefficients `k1..k4`. Use
  `schlepp.geometry.cam_to_pixel_fisheye_kb4` and
  `pixel_to_cam_fisheye_kb4`. The exact native calibration of the Aria Gen 2
  cameras is the `FisheyeRadTanThinPrism` (Fisheye624) model shipped with the
  Project Aria SDK. We instead expose a KB4 fit because KB4 is supported
  out of the box by OpenCV (`cv::fisheye`), COLMAP (`OPENCV_FISHEYE`), and
  most other downstream toolchains. For undistortion, see
  `schlepp.utils.undistort_aria_to_pinhole`.
- **Quaternion** representation: `[x, y, z, w]`.

---

## Usage

### Constructing the dataset

```python
from torch.utils.data import DataLoader

ds = SchleppDataset(
    data_root,              # path to the SCHLEPP root (contains index.parquet)
    modalities=...,         # which fields to load; see "Modalities" below
    cameras=...,            # "static" or ["aria_slamL", "aria_slamR", ...]
    seq_len=None,           # int -> per-sample clip length; None -> whole sequence
    frame_stride=1,         # stride between sampled frames
    clip_strategy="random", # "random" | "first" | "fixed_stride" | "all_clips"
    index=None,             # optional pre-filtered pandas DataFrame
    transform=None,         # callable sample -> Any; default is identity
    rng_seed=0,
    on_error="raise",       # "raise" | "skip"
)
```

`clip_strategy` controls how, given a sequence with more frames than
`seq_len * frame_stride`, we pick the start frame of each emitted clip:

- `"random"` -- a fresh random start per `__getitem__` call. Each sample
  draws one clip per sequence and `__len__` equals the number of
  sequences.
- `"first"` -- always start at frame 0. One clip per sequence;
  `__len__` equals the number of sequences.
- `"fixed_stride"` -- non-overlapping clips tiling each sequence (start
  frames `0, seq_len*frame_stride, 2*seq_len*frame_stride, ...`).
  `__len__` enumerates every clip across the sequence so each frame is
  seen at most once.
- `"all_clips"` -- every overlapping clip with a 1-frame slide (start
  frames `0, 1, 2, ...`). `__len__` is much larger.

By default, the dataset covers every sequence in `<data_root>/index.parquet`.
To restrict the set of sequences, read the parquet into a pandas DataFrame,
filter it, and pass the result via `index=`.

The `transform` callable is pickled when `DataLoader` workers run in
spawn mode. Make sure to define it at module scope or wrap it
with `functools.partial` to avoid pickling errors.

### Cameras

The dataset ships six co-registered camera variants per sequence. Pass the
variant name(s) via `cameras=`:

| Name           | Resolution   | Distortion |
|----------------|--------------|------------|
| `static`       | 1920 × 1080  | pinhole    |
| `body_follow`  | 1920 × 1080  | pinhole    |
| `object_orbit` | 1920 × 1080  | pinhole    |
| `aria_slamL`   | 512 × 512    | KB4        |
| `aria_slamR`   | 512 × 512    | KB4        |
| `aria_rgb`     | 2016 × 1512  | KB4        |

### Modalities

Each entry in `modalities=` opts a single field into every sample. Anything
unrequested is never read off disk.

| Modality              | Per-camera | Sample field                                   | Shape / type                                                                                  |
|-----------------------|-----------|------------------------------------------------|-----------------------------------------------------------------------------------------------|
| `rgb`                 | yes       | `cameras[c]["rgb"]`                            | `(S, 3, H, W)` `uint8`                                                                        |
| `depth`               | yes       | `cameras[c]["depth"]`                          | `(S, 1, H, W)` `float32` (metres; `0` = invalid)                                              |
| `segmentation`        | yes       | `cameras[c]["segmentation"]`                   | `(S, 1, H, W)` `uint8` (pass-index labels)                                                    |
| `flow_fwd`            | yes       | `cameras[c]["flow_fwd"]`                       | `(S, 2, H, W)` `float32` pixel `(du, dv)`                                                     |
| `flow_bwd`            | yes       | `cameras[c]["flow_bwd"]`                       | `(S, 2, H, W)` `float32` pixel `(du, dv)`                                                     |
| `cameras`             | yes       | `cameras[c]["K"]`, `["cam_T_world"]`, `["width"]`, `["height"]`, `["distortion_model"]`, `["distortion_params"]` | `(3, 3)`, `(S, 4, 4)`, ints, `"pinhole"` \| `"kb4"`, `(D,)` `float32` NaN-padded (KB4 reads `[:4]`, pinhole all-NaN) |
| `point_tracks`        | partial   | `point_tracks`                      | shared `trajs_world (N, S, 3)`<br /> per-camera:<br /> `trajs_2d_pix[c] (N, S, 2)`, `visible[c] (N, S)`, `in_frustum[c] (N, S)` |
| `obbs`                | no        | `obbs`                                         | list of `{uid, category, centroid[3], extents[3], rotation[4]_xyzw}` in world coords         |
| `object_animation`    | no        | `object_animation` keyed by `actor_idx`         | per actor: `{uid, category, role, centroids (S, 3), rotations_xyzw (S, 4), ...}`              |
| `mhr_params`          | no        | `mhr_params` keyed by `actor_idx`               | per actor: `{shape_params (45,), model_params (S, 204), expr_params (...,72), object_uid, ...}` |
| `segmentation_labels` | no        | `segmentation_labels`                          | `{int_pass_index: str_label}`                                                                 |

Every sample also carries the shared scalars `sequence_id`, `scene`,
`scene_type`, `num_actors_total`, `fps`, and `frame_indices (S,) int64`.

Point tracking masks:
- `visible[c]` is `True` when the point is rendered (i.e. not occluded by
  another mesh) in camera `c`.
- `in_frustum[c]` is `True` when the point is in front of camera `c` AND
  its projected pixel lies within the image bounds.

#### Batching Across Sequences

Note that PyTorch's `default_collate` will not work across multiple sequences
for every modality out-of-the-box.
The per-clip modalities have a structure that is variable per sequence
and thus need a `transform=` (or custom collate) that adapts them first
(e.g., through padding).

| Modality              | `default_collate` safe? | Why not                                                        |
|-----------------------|-------------------------|----------------------------------------------------------------|
| `rgb`, `depth`, `segmentation`, `flow_fwd`, `flow_bwd` | yes  | Fixed `(S, C, H, W)` per camera.                |
| `cameras`             | yes                     | Fixed-shape intrinsics / extrinsics.                           |
| `frame_indices`       | yes                     | Fixed `(S,)` int64.                                            |
| `point_tracks`        | no                      | `N` (number of tracks) varies per sequence.                    |
| `obbs`                | no                      | List of dicts of variable length `M`.                          |
| `object_animation`    | no                      | Dict keyed by per-sequence `actor_idx`.                        |
| `mhr_params`          | no                      | Dict keyed by per-sequence `actor_idx`. Some values are non-tensor fields like strings. | |
| `segmentation_labels` | no                      | `{int: str}` with per-sequence keys and string values.         |

#### Sample-level utilities

`schlepp.utils` exposes some utility functions for your
`transform=` function.

| Module                  | Helper                                                                                                                                  | What it does                                                                                                                              |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| `schlepp.utils.tracks`  | `filter_tracks_by_category` / `filter_tracks_by_visibility` / `filter_tracks_by_motion`                                                 | Drop tracks by category whitelist, visibility-frame count, or per-frame displacement.                                                     |
|                         | `subsample_tracks(pt, N, mode=…)`                                                                                                       | Uniform, stratified (per-category), or weighted track subsampling.                                                                        |
|                         | `split_tracks_query_target` / `query_points_from_first_frame`                                                                           | Build query / target splits and CoTracker-style query lists.                                                                              |
| `schlepp.utils.spatial` | `resize_sample`, `crop_sample`, `center_crop_sample`, `random_crop_sample`, `pad_sample`                                                | Spatial ops on the full sample. Updates rgb / depth / segmentation / flow vectors / trajs / in_frustum / K / width / height together.     |
| `schlepp.utils.aria`    | `undistort_aria_to_pinhole(sample, …)`                                                                                                  | Replace KB4 fisheye cameras with a pinhole equivalent. Updates rgb / depth / segmentation / trajs / in_frustum / K consistently.          |

---

## Examples

### Training a point tracker

To train a point tracker, we return 24-frame sequences with random starts
in the `(rgbs, trajs, visibs)` format that the training loop expects,
randomly sampling one of two moving cameras per clip. We also shrink the
short side from 1024 to 384 (consistent across rgb / trajs via
`resize_sample`) and stratify-subsample to 512 tracks:

```python
import random
from schlepp import SchleppDataset
from schlepp.utils import resize_sample, subsample_tracks

CAMS = ["body_follow", "object_orbit"]

def to_point_tracker(sample):
    sample = resize_sample(sample, short_side=384)
    pt = subsample_tracks(
        sample["point_tracks"], 512, mode="stratified",
        per_category={"body": 128, "cloth": 128, "carried": 128, "scene": 128},
    )
    cam_name = random.choice(CAMS)
    cam = sample["cameras"][cam_name]
    rgbs = cam["rgb"].permute(0, 2, 3, 1)                        # (S, H, W, 3) uint8
    trajs = pt["trajs_2d_pix"][cam_name].permute(1, 0, 2)        # (S, 512, 2) float32
    visibs = pt["visible"][cam_name].permute(1, 0)               # (S, 512) bool
    return rgbs, trajs, visibs

ds = SchleppDataset(
    "/path/to/schlepp",
    modalities=["rgb", "point_tracks"],
    cameras=CAMS,
    seq_len=24,
    frame_stride=1,
    clip_strategy="random",
    transform=to_point_tracker,
)
```

<details>
<summary>Optical Flow Estimation</summary>

For optical flow estimation, we return consecutive RGB frames paired with
the dense `flow_fwd` vector map. We shrink the short side to 384 first with
`resize_sample` and then slice into `(rgb_t, rgb_t+1, flow_t)`.

```python
from schlepp import SchleppDataset
from schlepp.utils import resize_sample

def to_flow_sample(sample):
    sample = resize_sample(sample, short_side=384)
    cam = sample["cameras"]["body_follow"]
    return {
        "rgb_1": cam["rgb"][:-1],            # (S-1, 3, H, W) uint8
        "rgb_2": cam["rgb"][1:],             # (S-1, 3, H, W) uint8
        "flow":  cam["flow_fwd"][:-1],       # (S-1, 2, H, W) float32
        # Optionally include backward flow:
        # "flow_bwd": cam["flow_bwd"][1:],
    }

ds = SchleppDataset(
    "/path/to/schlepp",
    modalities=["rgb", "flow_fwd"],
    cameras="body_follow",
    seq_len=8,  # Yields 7 valid flow pairs
    clip_strategy="random",
    transform=to_flow_sample,
)
```
</details>

<details>
<summary>3D Bounding Box Estimation</summary>

To lift 2D bounding boxes into static 3D OBBs from posed RGB and a
camera calibration, we use
`schlepp.projection.project_obbs_to_2d_bboxes` to derive the 2D inputs
from the ground-truth 3D OBBs. The transform
returns short clips of RGB, intrinsics, per-frame pose, the projected
2D boxes, and the 3D OBBs as supervision.

```python
from schlepp import SchleppDataset
from schlepp.projection import project_obbs_to_2d_bboxes

def to_obb_sample(sample):
    cam = sample["cameras"]["body_follow"]
    bb2d, bb2d_valid = project_obbs_to_2d_bboxes(
        sample["obbs"], cam["K"], cam["cam_T_world"],
        cam["distortion_model"], cam["distortion_params"],
    )
    return {
        "rgb":         cam["rgb"],            # (S, 3, H, W) uint8
        "K":           cam["K"],              # (3, 3)
        "cam_T_world": cam["cam_T_world"],    # (S, 4, 4)
        "bb2d":        bb2d,                  # (S, M, 4)
        "bb2d_valid":  bb2d_valid,            # (S, M)
        "gt_obbs":     sample["obbs"],        # (M,) — supervision
    }

ds = SchleppDataset(
    "/path/to/schlepp",
    modalities=["rgb", "cameras", "obbs"],
    cameras="body_follow",
    seq_len=8,
    frame_stride=1,
    clip_strategy="random",
    transform=to_obb_sample,
)
```
</details>

<details>
<summary>Camera Pose Estimation</summary>

For camera pose estimation, we return 16-frame RGB clips paired with the
per-frame `cam_T_world` extrinsic and the intrinsic `K`, drawn from any
of three pinhole cameras.

```python
from schlepp import SchleppDataset

def to_pose_sample(sample):
    cam = sample["cameras"]["body_follow"]
    return {
        "rgb":          cam["rgb"],                # (S, 3, H, W) uint8
        "K":            cam["K"],                  # (3, 3)
        "cam_T_world":  cam["cam_T_world"],        # (S, 4, 4)
        "width":        cam["width"],
        "height":       cam["height"],
    }

ds = SchleppDataset(
    "/path/to/schlepp",
    modalities=["rgb", "cameras"],
    cameras=["static", "body_follow", "object_orbit"],
    seq_len=16,
    frame_stride=2,
    clip_strategy="random",
    transform=to_pose_sample,
)
```
</details>

<details>
<summary>Monocular Depth Estimation</summary>

For monocular depth estimation, we return per-frame RGB paired with the
metric depth map and a validity mask that excludes invalid pixels (where
the depth pass writes `0`).

```python
from schlepp import SchleppDataset

def to_depth_sample(sample):
    cam = sample["cameras"]["static"]
    depth = cam["depth"]                  # (S, 1, H, W) float32, 0 = invalid
    return {
        "rgb":     cam["rgb"],            # (S, 3, H, W) uint8
        "depth":   depth,
        "valid":   depth > 0,             # boolean mask for the loss
        "K":       cam["K"]
    }

ds = SchleppDataset(
    "/path/to/schlepp",
    modalities=["rgb", "depth", "cameras"],
    cameras="static",
    seq_len=1,                # per-frame supervision
    clip_strategy="all_clips",
    transform=to_depth_sample,
)
```
</details>


<details>
<summary>Human Pose Estimation (MHR Parameters)</summary>

For 3D human pose and shape estimation, we return 8-frame RGB clips of
the primary actor together with their segmentation pass and the
per-frame MHR rig parameters.

```python
from schlepp import SchleppDataset

def to_mhr_sample(sample):
    cam = sample["cameras"]["static"]
    actors = sample["mhr_params"]                  # {actor_idx: {...}}
    primary = actors[0]                            # actor_id 0 is primary
    return {
        "rgb":           cam["rgb"],               # (S, 3, H, W) uint8
        "segmentation":  cam["segmentation"],      # (S, 1, H, W) uint8 pass-index
        "K":             cam["K"],
        "cam_T_world":   cam["cam_T_world"],
        "shape_params":  primary["shape_params"],  # (45,) — identity
        "model_params":  primary["model_params"],  # (S, 204) — per-frame rig
    }

ds = SchleppDataset(
    "/path/to/schlepp",
    modalities=["rgb", "segmentation", "cameras", "mhr_params"],
    cameras="static",
    seq_len=8,
    frame_stride=1,
    clip_strategy="random",
    transform=to_mhr_sample,
)
```
</details>

<details>
<summary>Human Pose Estimation (MHR 3D Joints)</summary>

For 3D keypoint-based pose estimation, world-space joint
positions per frame are required. In this dataset, we provide a
omplete set of MHR parameters, which need to be forwarded through
pymomentum to obtain the 3D joint positions. We provide a script
to precompute them:

```bash
python -m scripts.precompute_mhr_joints \
    --data-root /path/to/schlepp \
    --bundle-dir /path/to/mhr_assets \
    --num-workers 8
```

This writes one `joints.json` per sequence at the sequence root with
`(T, J, 3)` world-space joint positions per actor.

Then the training loop just reads the JSON in its `transform=` and
slices by `frame_indices`:

```python
import json
import torch
from pathlib import Path
from schlepp import SchleppDataset

DATA_ROOT = Path("/path/to/schlepp")

def to_mhr_joints(sample):
    cam = sample["cameras"]["static"]
    with open(DATA_ROOT / sample["sequence_id"] / "joints.json") as f:
        joints = json.load(f)
    # actor 0 is the primary actor; (T, J, 3) -> (S, J, 3) via frame_indices
    joints_world = torch.tensor(
        joints["actors"]["0"]["joints_world"], dtype=torch.float32,
    )[sample["frame_indices"]]
    return {
        "rgb":          cam["rgb"],                            # (S, 3, H, W) uint8
        "K":            cam["K"],                              # (3, 3)
        "cam_T_world":  cam["cam_T_world"],                    # (S, 4, 4)
        "joints_world": joints_world,                          # (S, J, 3) metres, Z-up
    }

ds = SchleppDataset(
    "/path/to/schlepp",
    modalities=["rgb", "cameras"],
    cameras="static",
    seq_len=8,
    clip_strategy="random",
    transform=to_mhr_joints,
)
```

Project the world-space joints into pixels at training time by
composing `schlepp.geometry.world_to_cam` with
`schlepp.projection.cam_to_pixel` (handles both pinhole and KB4
fisheye via the per-camera `distortion_model`).
</details>

<details>
<summary>Sequence Filtering</summary>

```python
import pandas as pd
from schlepp import SchleppDataset

df = pd.read_parquet("/path/to/schlepp/index.parquet")
df_busy = df[df.num_actors_total > 2]   # scenes with more than two people

ds = SchleppDataset(
    "/path/to/schlepp",
    modalities=["rgb", "cameras"],
    cameras="static",
    seq_len=16,
    index=df_busy,
)
```
</details>

<details>
<summary>Data Augmentation</summary>

```python
import random
import torch
from schlepp import SchleppDataset

def augment_and_return_rgbs(sample):
    brightness = random.uniform(0.8, 1.2)
    contrast   = random.uniform(0.8, 1.2)

    for entry in sample["cameras"].values():
        rgb = entry["rgb"].float()                          # (S, 3, H, W)
        mean = rgb.mean(dim=(-3, -2, -1), keepdim=True)
        rgb = ((rgb - mean) * contrast + mean) * brightness
        rgb = rgb.clamp(0, 255).to(torch.uint8)
        entry["rgb"] = rgb

    rgbs = torch.stack([entry["rgb"] for entry in sample["cameras"].values()])

    return rgbs

ds = SchleppDataset(..., transform=augment_and_return_rgbs)
```

</details>

---

## Evaluation suite

```python
from schlepp.eval import (
    end_point_error, flow_validity_mask_from_depth, fl_all,
    epe_by_motion, flow_summary,
    average_trajectory_error, survival_rate,
    delta_avg, average_jaccard, occlusion_accuracy, occlusion_auc,
    tap_vid_metrics,
)

# Optical flow EPE, ignoring invalid pixels.
mask = flow_validity_mask_from_depth(sample["cameras"]["static"]["depth"])
epe = end_point_error(pred_flow, gt_flow, mask=mask)

# Or a one-shot Sintel-style summary (EPE + KITTI Fl-all + EPE binned by
# ground-truth motion magnitude); pass `depth=` to derive the validity
# mask automatically.
flow = flow_summary(pred_flow, gt_flow, depth=sample["cameras"]["static"]["depth"])

# Point-track metrics — `valid` is the in-frustum mask, `visible` is
# the no-occlusion mask.
ate = average_trajectory_error(
    pred_tracks, gt_tracks,
    valid=sample["point_tracks"]["in_frustum"]["static"],
    visible=sample["point_tracks"]["visible"]["static"],
)
sr = survival_rate(
    pred_tracks, gt_tracks,
    valid=sample["point_tracks"]["in_frustum"]["static"],
    visible=sample["point_tracks"]["visible"]["static"],
    threshold=50.0,
)

# Or the canonical TAP-Vid suite in one call. `image_size=(H, W)` rescales
# errors into the TAP-Vid 256x256 reference so the published threshold
# values [1, 2, 4, 8, 16] carry their standard meaning.
tap = tap_vid_metrics(
    pred_tracks, gt_tracks,
    valid=sample["point_tracks"]["in_frustum"]["static"],
    visible=sample["point_tracks"]["visible"]["static"],
    pred_visible=pred_visible,                 # optional bool mask
    pred_visible_score=pred_visible_score,     # optional logits / scores
    image_size=(H, W),
)
# tap == {"delta_avg": ..., "aj": ..., "oa": ..., "auc": ...}
```

---

## Visualisation suite

```bash
# One MP4 per (variant, modality) for an entire sequence
python -m schlepp.visualize.generate_mp4s /path/to/schlepp/<sequence_id>

# Fading point-track trails for a single camera
python -m schlepp.visualize.render_point_trails \
    /path/to/schlepp/<sequence_id> --camera body_follow

# 3D-OBB wireframe overlay (handles KB4 distortion for Aria)
python -m schlepp.visualize.overlay_obbs \
    /path/to/schlepp/<sequence_id> --camera aria_rgb

# Interactive 3D scene viewer (rerun): every frame logged, scrub via the
# time-cursor. Draws every camera variant as a frustum, static + carried
# OBBs (per-category colour), and per-actor body meshes (per-actor colour).
python -m schlepp.visualize.build_scene_in_3d \
    /path/to/schlepp/<sequence_id> --bundle-dir ./mhr_assets --frame 0

# ... add --with-images to attach each camera's RGB frames to its frustum
# so the rerun 2D image panes display synced video; omit --bundle-dir
# (or pass --no-bodies) to render cameras + OBBs only.

# Dump a portable .rrd recording for the rerun web viewer instead of
# spawning a local viewer:
python -m schlepp.visualize.build_scene_in_3d \
    /path/to/schlepp/<sequence_id> --bundle-dir ./mhr_assets \
    --save scene.rrd
# ... then open scene.rrd at https://app.rerun.io (or `rerun scene.rrd`).
```

---

## License

SCHLEPP is released under the Creative Commons Attribution-NonCommercial
4.0 International License. See [LICENSE](LICENSE) for details.

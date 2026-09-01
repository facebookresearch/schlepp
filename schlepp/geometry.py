# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Vectorised PyTorch geometry primitives for the SCHLEPP dataset.

Every function here is a pure tensor operation. They:

* Accept tensors with an arbitrary leading "batch" shape ``*B`` (zero or
  more leading dimensions); broadcasting follows the usual PyTorch rules.
* Stay on the device of their inputs -- no hardcoded ``.cuda()`` or
  ``.to(device)``. Move your tensors to the device you want before
  calling.
* Use ``float32`` by default; pass ``float64`` tensors in if you need
  the extra precision (output dtype follows input dtype).

Conventions
-----------
* World frame: Z-up, right-handed, metres.
* Camera frame: OpenCV (X right, Y down, Z forward).
* ``cam_T_world`` is a ``(4, 4)`` SE(3) matrix such that
  ``p_cam = (cam_T_world @ [p_world; 1])[:3]``.
* Pinhole intrinsic ``K`` is the ``(3, 3)`` matrix with
  ``pix = K @ (p_cam / p_cam.z)``.
* KB4 fisheye uses four radial coefficients on the Kannala-Brandt
  parameterization; the same model is exposed as ``KannalaBrandtK3``
  in the Project Aria SDK and as ``OPENCV_FISHEYE`` in COLMAP.
  Distortion polynomial:
  ``theta_d = theta * (1 + k[0]*theta^2 + k[1]*theta^4
                       + k[2]*theta^6 + k[3]*theta^8)``.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch

#: Maximum half-FoV for the Aria Gen2 KB4 lenses (radians).
DEFAULT_MAX_THETA: float = math.radians(85.0)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# SE(3) helpers
# ---------------------------------------------------------------------------


def _apply_transform(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Apply a homogeneous ``(..., 4, 4)`` transform to ``(..., 3)`` points.

    Both tensors broadcast against each other on the leading axes. The
    output preserves the input dtype and device.
    """
    if points.shape[-1] != 3:
        raise ValueError(f"points must have last dim 3, got {points.shape}")
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"T must have shape (..., 4, 4), got {T.shape}")
    R = T[..., :3, :3]                                  # shape: (..., 3, 3)
    t = T[..., :3, 3]                                   # shape: (..., 3)
    # Bring `points` to (..., 3, 1) so matmul yields (..., 3, 1).
    rotated = torch.matmul(R, points.unsqueeze(-1)).squeeze(-1)  # (..., 3)
    return rotated + t


def world_to_cam(
    points_world: torch.Tensor,
    cam_T_world: torch.Tensor,
) -> torch.Tensor:
    """Transform world-frame points to camera-frame points.

    Parameters
    ----------
    points_world
        Shape ``(..., 3)``. World-frame coordinates in metres.
    cam_T_world
        Shape ``(..., 4, 4)``. OpenCV cam-from-world SE(3) matrix.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3)``. Camera-frame coordinates.
    """
    return _apply_transform(points_world, cam_T_world)


def cam_to_world(
    points_cam: torch.Tensor,
    cam_T_world: torch.Tensor,
) -> torch.Tensor:
    """Transform camera-frame points back into the world frame.

    Inverts ``cam_T_world`` analytically (since it is rigid, the inverse
    is ``R^T`` with translation ``-R^T t``); never calls ``torch.inverse``
    on a 4x4 matrix.
    """
    if cam_T_world.shape[-2:] != (4, 4):
        raise ValueError(
            f"cam_T_world must have shape (..., 4, 4), got {cam_T_world.shape}"
        )
    R = cam_T_world[..., :3, :3]                        # shape: (..., 3, 3)
    t = cam_T_world[..., :3, 3]                         # shape: (..., 3)
    Rt = R.transpose(-1, -2)                            # shape: (..., 3, 3)
    # p_world = R^T (p_cam - t) = R^T p_cam - R^T t
    centered = points_cam - t                           # shape: (..., 3)
    return torch.matmul(Rt, centered.unsqueeze(-1)).squeeze(-1)  # (..., 3)


def quaternion_to_rotation_matrix(q_xyzw: torch.Tensor) -> torch.Tensor:
    """Convert quaternions in ``[x, y, z, w]`` order to rotation matrices.

    Accepts any leading batch shape; returns one matrix per quaternion.

    Parameters
    ----------
    q_xyzw
        Shape ``(..., 4)``. Quaternions in ``XYZW`` order. The function
        normalises internally so callers do not have to.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3, 3)``. Rotation matrices in the same dtype and
        device as ``q_xyzw``.
    """
    if q_xyzw.shape[-1] != 4:
        raise ValueError(
            f"q_xyzw must have last dim 4, got {q_xyzw.shape}"
        )
    # Normalise to unit quaternions (input may be drift-affected).
    norm = q_xyzw.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    q = q_xyzw / norm                                   # shape: (..., 4)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    # Build (..., 3, 3). We stack rows then columns instead of a single
    # torch.tensor() so the result keeps the input device + dtype.
    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - wz)
    r02 = 2.0 * (xz + wy)
    r10 = 2.0 * (xy + wz)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - wx)
    r20 = 2.0 * (xz - wy)
    r21 = 2.0 * (yz + wx)
    r22 = 1.0 - 2.0 * (xx + yy)
    row0 = torch.stack([r00, r01, r02], dim=-1)         # shape: (..., 3)
    row1 = torch.stack([r10, r11, r12], dim=-1)
    row2 = torch.stack([r20, r21, r22], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)      # shape: (..., 3, 3)


# ---------------------------------------------------------------------------
# Pinhole projection
# ---------------------------------------------------------------------------


def cam_to_pixel_pinhole(
    points_cam: torch.Tensor,
    K: torch.Tensor,
    *,
    eps: float = _EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project OpenCV camera-frame points to pixels via a pinhole intrinsic.

    Parameters
    ----------
    points_cam
        Shape ``(..., 3)``.
    K
        Shape ``(..., 3, 3)`` pinhole intrinsic.

    Returns
    -------
    pixels
        Shape ``(..., 2)`` ``(u, v)`` in pixels.
    in_front
        Shape ``(...)`` bool mask, ``True`` where ``points_cam.z > eps``.
    """
    if points_cam.shape[-1] != 3:
        raise ValueError(
            f"points_cam must have last dim 3, got {points_cam.shape}"
        )
    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K must have shape (..., 3, 3), got {K.shape}")
    z = points_cam[..., 2]                              # shape: (...)
    safe_z = torch.where(z.abs() < eps, torch.full_like(z, eps), z)
    xy_over_z = points_cam[..., :2] / safe_z.unsqueeze(-1)  # shape: (..., 2)
    # Augment to (..., 3) for K @ [u; v; 1].
    ones = torch.ones_like(z).unsqueeze(-1)             # shape: (..., 1)
    homog = torch.cat([xy_over_z, ones], dim=-1)        # shape: (..., 3)
    pix = torch.matmul(K, homog.unsqueeze(-1)).squeeze(-1)  # (..., 3)
    pixels = pix[..., :2]                               # shape: (..., 2)
    in_front = z > eps
    return pixels, in_front


def pixel_to_cam_pinhole(
    pixels: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
) -> torch.Tensor:
    """Back-project pinhole pixels with given depths into camera-frame points.

    Parameters
    ----------
    pixels
        Shape ``(..., 2)`` ``(u, v)``.
    depth
        Shape ``(...)`` depths along the camera Z axis (metres).
    K
        Shape ``(..., 3, 3)``.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3)`` camera-frame points.
    """
    if pixels.shape[-1] != 2:
        raise ValueError(f"pixels must have last dim 2, got {pixels.shape}")
    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K must have shape (..., 3, 3), got {K.shape}")
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    u = pixels[..., 0]
    v = pixels[..., 1]
    x_norm = (u - cx) / fx                              # shape: (...)
    y_norm = (v - cy) / fy
    points = torch.stack([
        x_norm * depth, y_norm * depth, depth
    ], dim=-1)                                          # shape: (..., 3)
    return points


# ---------------------------------------------------------------------------
# KB4 fisheye projection (Kannala-Brandt 4-coefficient)
# ---------------------------------------------------------------------------


def _kb4_poly(theta: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Distorted radius theta_d = theta * (1 + k[0]*t^2 + k[1]*t^4
                                            + k[2]*t^6 + k[3]*t^8).

    ``theta`` shape: ``(...)``. ``k`` shape: ``(..., 4)``.
    """
    t2 = theta * theta                                  # shape: (...)
    k0 = k[..., 0]
    k1 = k[..., 1]
    k2 = k[..., 2]
    k3 = k[..., 3]
    # Horner on (1 + k0*t + k1*t^2 + k2*t^3 + k3*t^4) with t = theta^2.
    poly = (((k3 * t2 + k2) * t2 + k1) * t2 + k0) * t2 + 1.0
    return theta * poly                                 # shape: (...)


def _kb4_poly_derivative(theta: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """d(theta_d)/d(theta) for KB4. Used by Newton-Raphson inversion."""
    t2 = theta * theta
    k0 = k[..., 0]
    k1 = k[..., 1]
    k2 = k[..., 2]
    k3 = k[..., 3]
    # d(poly)/d(theta) = theta * (2 k0 + 4 k1 t^2 + 6 k2 t^4 + 8 k3 t^6)
    deriv_factor = (((8.0 * k3 * t2 + 6.0 * k2) * t2 + 4.0 * k1) * t2
                    + 2.0 * k0)
    poly = (((k3 * t2 + k2) * t2 + k1) * t2 + k0) * t2 + 1.0
    return poly + t2 * deriv_factor                     # shape: (...)


def cam_to_pixel_fisheye_kb4(
    points_cam: torch.Tensor,
    K: torch.Tensor,
    k: torch.Tensor,
    *,
    max_theta_rad: float = DEFAULT_MAX_THETA,
    eps: float = _EPS,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project OpenCV camera-frame points through a KB4 fisheye intrinsic.

    Parameters
    ----------
    points_cam
        Shape ``(..., 3)``. Must satisfy Z > 0 for in-front-of-camera
        points.
    K
        Shape ``(..., 3, 3)``. ``fx``, ``fy``, ``cx``, ``cy`` are read
        from ``K[..., 0, 0]``, ``[1, 1]``, ``[0, 2]``, ``[1, 2]``.
    k
        Shape ``(..., 4)``. KB4 polynomial coefficients ``(k1..k4)``.
    max_theta_rad
        Reject rays whose angle from the optical axis exceeds this.

    Returns
    -------
    pixels
        Shape ``(..., 2)``. ``NaN`` where invalid.
    valid
        Shape ``(...)`` bool. ``False`` where the ray is behind the
        camera, exceeds ``max_theta_rad``, or produces non-finite output.
    """
    if points_cam.shape[-1] != 3:
        raise ValueError(
            f"points_cam must have last dim 3, got {points_cam.shape}"
        )
    if K.shape[-2:] != (3, 3):
        raise ValueError(f"K must have shape (..., 3, 3), got {K.shape}")
    if k.shape[-1] != 4:
        raise ValueError(f"k must have last dim 4, got {k.shape}")

    x = points_cam[..., 0]
    y = points_cam[..., 1]
    z = points_cam[..., 2]
    rho_xy = torch.hypot(x, y)                          # shape: (...)
    norm = torch.hypot(rho_xy, z)                       # shape: (...)
    safe_rho = torch.where(rho_xy < eps, torch.ones_like(rho_xy), rho_xy)
    safe_norm = torch.where(norm < eps, torch.full_like(norm, eps), norm)
    sin_theta = rho_xy / safe_norm
    cos_theta = z / safe_norm
    theta = torch.atan2(sin_theta, cos_theta)           # shape: (...)
    # Unit vector along the projected radial direction in image plane.
    ux = torch.where(rho_xy < eps, torch.zeros_like(x), x / safe_rho)
    uy = torch.where(rho_xy < eps, torch.zeros_like(y), y / safe_rho)

    valid = (z > 0) & (theta <= max_theta_rad) & torch.isfinite(theta)

    theta_d = _kb4_poly(theta, k)                       # shape: (...)
    xd = theta_d * ux
    yd = theta_d * uy

    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    u = fx * xd + cx                                    # shape: (...)
    v = fy * yd + cy
    pixels = torch.stack([u, v], dim=-1)                # shape: (..., 2)
    valid = valid & torch.isfinite(pixels[..., 0]) & torch.isfinite(pixels[..., 1])
    nan = torch.full_like(pixels, float("nan"))
    pixels = torch.where(valid.unsqueeze(-1), pixels, nan)
    return pixels, valid


def pixel_to_cam_fisheye_kb4(
    pixels: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    k: torch.Tensor,
    *,
    max_iters: int = 20,
    tol: float = 1e-7,
    max_theta_rad: float = DEFAULT_MAX_THETA,
    eps: float = _EPS,
) -> torch.Tensor:
    """Back-project KB4 pixels with given depths to camera-frame points.

    The depth is interpreted as the Z-component (distance along the
    optical axis), matching the convention used by the SCHLEPP ``.dpt5``
    maps. Use :func:`pixel_to_cam_fisheye_kb4_ray` if you want unit
    rays instead.

    Parameters
    ----------
    pixels
        Shape ``(..., 2)``.
    depth
        Shape ``(...)``. Z-depths in metres.
    K, k
        See :func:`cam_to_pixel_fisheye_kb4`.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3)`` camera-frame points.
    """
    rays = pixel_to_cam_fisheye_kb4_ray(
        pixels, K, k,
        max_iters=max_iters, tol=tol,
        max_theta_rad=max_theta_rad, eps=eps,
    )
    # Scale rays so the Z-component equals the supplied depth.
    rz = rays[..., 2]
    safe_rz = torch.where(rz.abs() < eps, torch.full_like(rz, eps), rz)
    scale = depth / safe_rz                             # shape: (...)
    return rays * scale.unsqueeze(-1)                   # shape: (..., 3)


def pixel_to_cam_fisheye_kb4_ray(
    pixels: torch.Tensor,
    K: torch.Tensor,
    k: torch.Tensor,
    *,
    max_iters: int = 20,
    tol: float = 1e-7,
    max_theta_rad: float = DEFAULT_MAX_THETA,
    eps: float = _EPS,
) -> torch.Tensor:
    """Recover unit rays in the camera frame from KB4 pixels.

    Inverts the radial polynomial via Newton-Raphson. KB4 has no
    tangential / thin-prism terms so the per-axis denormalisation is
    exact in one step.

    Returns
    -------
    torch.Tensor
        Shape ``(..., 3)``. Unit-length rays in the OpenCV camera frame
        (X right, Y down, Z forward). Rays whose recovered theta exceeds
        ``max_theta_rad`` are set to NaN.
    """
    if pixels.shape[-1] != 2:
        raise ValueError(f"pixels must have last dim 2, got {pixels.shape}")
    fx = K[..., 0, 0]
    fy = K[..., 1, 1]
    cx = K[..., 0, 2]
    cy = K[..., 1, 2]
    xn = (pixels[..., 0] - cx) / fx                     # shape: (...)
    yn = (pixels[..., 1] - cy) / fy
    theta_d = torch.hypot(xn, yn)                       # shape: (...)
    safe_td = torch.where(theta_d < eps, torch.full_like(theta_d, eps), theta_d)
    ux = torch.where(theta_d < eps, torch.zeros_like(xn), xn / safe_td)
    uy = torch.where(theta_d < eps, torch.zeros_like(yn), yn / safe_td)

    # Newton iterations on theta_d_pred(theta) - theta_d = 0.
    theta = theta_d.clone()                             # equidistant init
    for _ in range(max_iters):
        theta_d_pred = _kb4_poly(theta, k)
        deriv = _kb4_poly_derivative(theta, k)
        deriv = torch.where(deriv.abs() < eps, torch.full_like(deriv, eps), deriv)
        step = (theta_d_pred - theta_d) / deriv
        theta = theta - step
        if torch.max(torch.abs(step)).item() < tol:
            break
    theta = torch.where(theta_d < eps, torch.zeros_like(theta), theta)

    sin_theta = torch.sin(theta)
    cos_theta = torch.cos(theta)
    rays = torch.stack([
        sin_theta * ux, sin_theta * uy, cos_theta
    ], dim=-1)                                          # shape: (..., 3)
    valid = torch.isfinite(theta) & (theta <= max_theta_rad)
    nan = torch.full_like(rays, float("nan"))
    return torch.where(valid.unsqueeze(-1), rays, nan)

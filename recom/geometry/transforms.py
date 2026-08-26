"""Rigid-transform utilities shared by the recorder, canonicalizer, and models.

Conventions (frozen for Study 1):
  * Quaternions are stored as (w, x, y, z) -- Chrono's (e0, e1, e2, e3) order.
  * Canonical sign rule: w >= 0 (if w == 0, first non-zero component >= 0).
  * World frame: Z up.  Ground plane is z = 0 with outward normal +Z.
  * A pose (p, q) maps body-local point x to world:  x_w = R(q) x + p.

Both NumPy and torch implementations are provided; they share semantics.
"""
from __future__ import annotations

import numpy as np

try:  # torch is optional for the simulation-only path
    import torch
except ImportError:  # pragma: no cover
    torch = None


# --------------------------------------------------------------------------------------
# NumPy
# --------------------------------------------------------------------------------------
def quat_canonical_np(q: np.ndarray) -> np.ndarray:
    """Return q with the canonical sign (w >= 0). Works on (..., 4)."""
    q = np.asarray(q, dtype=np.float64)
    w = q[..., :1]
    sign = np.where(w < 0, -1.0, 1.0)
    # tie-break w == 0 by first non-zero component
    zero_w = np.isclose(w, 0.0)
    if np.any(zero_w):
        rest = q[..., 1:]
        first_nz = np.sign(rest[np.arange(rest.shape[0]) if rest.ndim > 1 else 0])  # noqa
        # simple fallback: use sign of x, then y, then z
        s2 = np.sign(q[..., 1:2])
        s3 = np.where(s2 == 0, np.sign(q[..., 2:3]), s2)
        s4 = np.where(s3 == 0, np.sign(q[..., 3:4]), s3)
        s4 = np.where(s4 == 0, 1.0, s4)
        sign = np.where(zero_w, s4, sign)
    return q * sign


def quat_normalize_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_rotmat_np(q: np.ndarray) -> np.ndarray:
    """(..., 4) (w,x,y,z) -> (..., 3, 3) rotation matrix."""
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def rotmat_to_quat_np(R: np.ndarray) -> np.ndarray:
    """(..., 3, 3) -> (..., 4) (w,x,y,z), canonical sign."""
    R = np.asarray(R, dtype=np.float64)
    single = R.ndim == 2
    if single:
        R = R[None]
    q = np.empty(R.shape[:-2] + (4,), dtype=np.float64)
    for i in range(R.shape[0]):
        m = R[i]
        tr = np.trace(m)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            q[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            q[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            q[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            q[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = quat_canonical_np(quat_normalize_np(q))
    return q[0] if single else q


def quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a*b, (w,x,y,z)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_conj_np(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_rotate_np(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors v (..., 3) by quaternion q (..., 4) (broadcasting on leading dims)."""
    R = quat_to_rotmat_np(q)
    return np.einsum("...ij,...j->...i", R, v)


def quat_from_axis_angle_np(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle / 2)
    return quat_canonical_np(np.array([np.cos(angle / 2), *(axis * s)]))


def quat_angle_np(q: np.ndarray) -> np.ndarray:
    """Rotation angle (radians) of quaternion(s)."""
    q = quat_normalize_np(q)
    w = np.clip(np.abs(q[..., 0]), 0.0, 1.0)
    return 2.0 * np.arccos(w)


def random_quat_uniform_np(rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    """Uniform random rotation(s) on SO(3) (Shoemake's method).  Never sample Euler angles."""
    shape = () if n is None else (n,)
    u1, u2, u3 = rng.random(shape), rng.random(shape), rng.random(shape)
    a, b = np.sqrt(1 - u1), np.sqrt(u1)
    q = np.stack([a * np.sin(2 * np.pi * u2), a * np.cos(2 * np.pi * u2), b * np.sin(2 * np.pi * u3), b * np.cos(2 * np.pi * u3)], axis=-1)
    # (x, y, z, w) -> (w, x, y, z)
    q = np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)
    return quat_canonical_np(q)


def world_to_local_np(p_world: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """Map world points (..., 3) into the frame with pose (pos, quat)."""
    R = quat_to_rotmat_np(quat)
    d = np.asarray(p_world, dtype=np.float64) - pos
    return np.einsum("...ji,...j->...i", R, d)  # R^T d


def local_to_world_np(p_local: np.ndarray, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
    R = quat_to_rotmat_np(quat)
    return np.einsum("...ij,...j->...i", R, np.asarray(p_local, dtype=np.float64)) + pos


# --------------------------------------------------------------------------------------
# torch
# --------------------------------------------------------------------------------------
if torch is not None:

    def quat_canonical(q: "torch.Tensor") -> "torch.Tensor":
        sign = torch.where(q[..., :1] < 0, -1.0, 1.0)
        return q * sign

    def quat_normalize(q: "torch.Tensor") -> "torch.Tensor":
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def quat_to_rotmat(q: "torch.Tensor") -> "torch.Tensor":
        w, x, y, z = q.unbind(-1)
        R = torch.stack(
            [
                1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
            ],
            dim=-1,
        )
        return R.reshape(q.shape[:-1] + (3, 3))

    def quat_mul(a: "torch.Tensor", b: "torch.Tensor") -> "torch.Tensor":
        aw, ax, ay, az = a.unbind(-1)
        bw, bx, by, bz = b.unbind(-1)
        return torch.stack(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dim=-1,
        )

    def quat_conj(q: "torch.Tensor") -> "torch.Tensor":
        return q * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)

    def quat_rotate(q: "torch.Tensor", v: "torch.Tensor") -> "torch.Tensor":
        """q: (B, 4), v: (B, N, 3) or (B, 3) -> rotated v."""
        R = quat_to_rotmat(q)
        if v.dim() == R.dim():  # (B, N, 3)
            return torch.einsum("bij,bnj->bni", R, v)
        return torch.einsum("bij,bj->bi", R, v)

    def quat_rotate_inv(q: "torch.Tensor", v: "torch.Tensor") -> "torch.Tensor":
        R = quat_to_rotmat(q)
        if v.dim() == R.dim():
            return torch.einsum("bji,bnj->bni", R, v)
        return torch.einsum("bji,bj->bi", R, v)

    def quat_from_omega_step(q: "torch.Tensor", omega_world: "torch.Tensor", dt: float) -> "torch.Tensor":
        """Integrate orientation with a world-frame angular velocity over dt (exponential map)."""
        theta = omega_world.norm(dim=-1, keepdim=True)
        axis = omega_world / theta.clamp_min(1e-12)
        half = 0.5 * theta * dt
        dq = torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)
        return quat_normalize(quat_mul(dq, q))

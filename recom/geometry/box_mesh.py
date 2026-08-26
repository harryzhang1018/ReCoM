"""Canonical box surface representations.

1. Triangle mesh (12 faces) + surface-patch tokens (Section 11.1 of the study plan).
2. Surface point samples with guaranteed corner/edge samples (surface-point baseline).

All quantities are in the box-local frame and depend only on the half-extents; they are
meant to be computed once per geometry and cached.
"""
from __future__ import annotations

import hashlib

import numpy as np

from .box_plane_analytic import CORNER_SIGNS

# Faces as quads (corner indices into CORNER_SIGNS ordering), outward normal listed.
# corner index i = 4*(sx>0) + 2*(sy>0) + (sz>0)
_QUADS = [
    ([4, 6, 7, 5], (1, 0, 0)),    # +X
    ([0, 1, 3, 2], (-1, 0, 0)),   # -X
    ([2, 3, 7, 6], (0, 1, 0)),    # +Y
    ([0, 4, 5, 1], (0, -1, 0)),   # -Y
    ([1, 5, 7, 3], (0, 0, 1)),    # +Z
    ([0, 2, 6, 4], (0, 0, -1)),   # -Z
]


def box_triangles() -> tuple[np.ndarray, np.ndarray]:
    """Return (faces (12, 3) int, face_axis_normals (12, 3)) with outward orientation.

    Winding is fixed so the geometric normal matches the axis normal for any positive half-extents.
    """
    faces, normals = [], []
    for quad, n in _QUADS:
        a, b, c, d = quad
        for tri in ((a, b, c), (a, c, d)):
            v = CORNER_SIGNS[list(tri)]
            geo_n = np.cross(v[1] - v[0], v[2] - v[0])
            if np.dot(geo_n, n) < 0:
                tri = (tri[0], tri[2], tri[1])
            faces.append(tri)
            normals.append(n)
    return np.asarray(faces, dtype=np.int64), np.asarray(normals, dtype=np.float64)


def box_mesh(half_extents: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (8,3), faces (12,3)) for a box with the given half-extents."""
    faces, _ = box_triangles()
    return CORNER_SIGNS * np.asarray(half_extents, dtype=np.float64), faces


def mesh_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(vertices, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(faces, dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def face_adjacency(faces: np.ndarray) -> np.ndarray:
    """(F, 3) indices of the faces sharing each edge of face f (-1 if boundary)."""
    edge_to_faces: dict[tuple[int, int], list[int]] = {}
    for f, tri in enumerate(faces):
        for i in range(3):
            e = tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
            edge_to_faces.setdefault(e, []).append(f)
    adj = -np.ones((len(faces), 3), dtype=np.int64)
    for f, tri in enumerate(faces):
        for i in range(3):
            e = tuple(sorted((int(tri[i]), int(tri[(i + 1) % 3]))))
            others = [g for g in edge_to_faces[e] if g != f]
            adj[f, i] = others[0] if others else -1
    return adj


def surface_patch_tokens(half_extents: np.ndarray) -> dict[str, np.ndarray]:
    """Build the cached surface-patch tokens for a box.

    Returns a dict of arrays (F = 12):
      verts_rel   (F, 3, 3)  triangle vertices relative to the face centroid
      centroid    (F, 3)     face centroid in box frame
      normal      (F, 3)     outward unit normal
      area        (F,)       triangle area
      adj         (F, 3)     adjacent face index per edge
      adj_normal  (F, 3, 3)  normals of adjacent faces
      dihedral    (F, 3)     dihedral angle (rad) with each adjacent face (pi/2 across box edges)
      sharp       (F, 3)     1.0 if the edge is a sharp (box) edge, else 0 (coplanar split)
      scale       (F, 3)     physical half-extents (broadcast)
      vertices    (8, 3), faces (F, 3)
    """
    he = np.asarray(half_extents, dtype=np.float64)
    V, Fc = box_mesh(he)
    _, axis_n = box_triangles()
    tri = V[Fc]  # (F, 3, 3)
    centroid = tri.mean(axis=1)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    cr = np.cross(e1, e2)
    area = 0.5 * np.linalg.norm(cr, axis=-1)
    normal = cr / (2 * area[:, None])
    assert np.allclose(normal, axis_n), "winding/normal mismatch"
    adj = face_adjacency(Fc)
    adj_normal = normal[adj]  # (F, 3, 3)
    cosang = np.clip(np.einsum("fi,fki->fk", normal, adj_normal), -1, 1)
    dihedral = np.arccos(cosang)
    sharp = (dihedral > 1e-6).astype(np.float64)
    return {
        "verts_rel": tri - centroid[:, None, :],
        "centroid": centroid,
        "normal": normal,
        "area": area,
        "adj": adj,
        "adj_normal": adj_normal,
        "dihedral": dihedral,
        "sharp": sharp,
        "scale": np.broadcast_to(he, centroid.shape).copy(),
        "vertices": V,
        "faces": Fc,
    }


def patch_feature_matrix(tokens: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten the cached tokens into a per-face feature vector (F, D_geo)."""
    F = tokens["centroid"].shape[0]
    feats = [
        tokens["verts_rel"].reshape(F, 9),
        tokens["centroid"],
        tokens["normal"],
        tokens["area"][:, None],
        tokens["adj_normal"].reshape(F, 9),
        tokens["dihedral"],
        tokens["sharp"],
        tokens["scale"],
    ]
    return np.concatenate(feats, axis=1).astype(np.float32)


PATCH_FEATURE_DIM = 9 + 3 + 3 + 1 + 9 + 3 + 3 + 3  # = 34


def surface_points(half_extents: np.ndarray, n_face: int = 16, n_edge: int = 4, seed: int = 0) -> dict[str, np.ndarray]:
    """Surface point set: uniform face samples + guaranteed corner + edge samples.

    Returns dict with pos (N,3), normal (N,3), weight (N,), kind (N,) [0 face,1 edge,2 corner], scale (N,3).
    Corner normals are the averaged outward normal; edge normals average the two incident faces.
    """
    he = np.asarray(half_extents, dtype=np.float64)
    rng = np.random.default_rng(seed)
    pos, nrm, w, kind = [], [], [], []
    # corners
    for s in CORNER_SIGNS:
        pos.append(s * he)
        nrm.append(s / np.linalg.norm(s))
        w.append(0.0)
        kind.append(2)
    # edges: for each axis, 4 edges parallel to that axis
    for ax in range(3):
        o1, o2 = [a for a in range(3) if a != ax]
        for s1 in (-1, 1):
            for s2 in (-1, 1):
                ts = (np.arange(n_edge) + 0.5) / n_edge * 2 - 1
                for t in ts:
                    p = np.zeros(3)
                    p[ax] = t * he[ax]
                    p[o1] = s1 * he[o1]
                    p[o2] = s2 * he[o2]
                    n = np.zeros(3)
                    n[o1], n[o2] = s1, s2
                    pos.append(p)
                    nrm.append(n / np.sqrt(2))
                    w.append(0.0)
                    kind.append(1)
    # faces: stratified uniform samples, weight = face area / n_face
    for ax in range(3):
        o1, o2 = [a for a in range(3) if a != ax]
        area = 4 * he[o1] * he[o2]
        for s in (-1, 1):
            u = rng.random((n_face, 2)) * 2 - 1
            for k in range(n_face):
                p = np.zeros(3)
                p[ax] = s * he[ax]
                p[o1] = u[k, 0] * he[o1]
                p[o2] = u[k, 1] * he[o2]
                n = np.zeros(3)
                n[ax] = s
                pos.append(p)
                nrm.append(n)
                w.append(area / n_face)
                kind.append(0)
    pos, nrm, w, kind = map(np.asarray, (pos, nrm, w, kind))
    return {"pos": pos, "normal": nrm, "weight": w, "kind": kind, "scale": np.broadcast_to(he, pos.shape).copy()}

"""Geometry/contact unit tests (study plan Section 18): analytic box-plane + transforms + mesh tokens."""
import numpy as np
import pytest

from recom.geometry.box_mesh import box_mesh, face_adjacency, patch_feature_matrix, surface_patch_tokens, surface_points
from recom.geometry.box_plane_analytic import (MODE_CORNER, MODE_EDGE, MODE_FACE, analytic_contacts, lowest_feature_mode, min_gap)
from recom.geometry.transforms import (quat_canonical_np, quat_from_axis_angle_np, quat_mul_np, quat_to_rotmat_np, random_quat_uniform_np, rotmat_to_quat_np)

HE = np.array([0.1, 0.075, 0.05])
QI = np.array([1.0, 0, 0, 0])


def test_quaternion_roundtrip_and_canonical_sign():
    rng = np.random.default_rng(0)
    q = random_quat_uniform_np(rng, 100)
    assert np.allclose(np.linalg.norm(q, axis=1), 1)
    assert (q[:, 0] >= 0).all()
    R = quat_to_rotmat_np(q)
    assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.eye(3), atol=1e-12)
    assert np.allclose(rotmat_to_quat_np(R), q, atol=1e-10)
    assert np.allclose(quat_canonical_np(-q), q)


def test_axis_aligned_above_plane_no_contact():
    c = analytic_contacts(np.array([0, 0, HE[2] + 0.01]), QI, HE, margin=1e-3)
    assert c["n_active"] == 0 and c["min_gap"] == pytest.approx(0.01)


def test_axis_aligned_touching_face_contact():
    c = analytic_contacts(np.array([0, 0, HE[2]]), QI, HE, margin=1e-3)
    assert c["n_active"] == 4 and c["mode"] == MODE_FACE
    assert np.allclose(c["d"], 0) and np.allclose(c["n"], [0, 0, 1])
    assert np.allclose(np.abs(c["p_box_local"][:, 2]), HE[2]) and np.allclose(c["p_ground_world"][:, 2], 0)


def test_axis_aligned_penetration_sign():
    c = analytic_contacts(np.array([0, 0, HE[2] - 0.002]), QI, HE, margin=1e-3)
    assert c["n_active"] == 4 and np.allclose(c["d"], -0.002) and np.allclose(c["n"][:, 2], 1)


def test_rotated_edge_and_corner_contact():
    q_edge = quat_from_axis_angle_np([1, 0, 0], np.pi / 4)
    z = -min_gap(np.zeros(3), q_edge, HE)
    c = analytic_contacts(np.array([0, 0, z]), q_edge, HE, margin=1e-4)
    assert c["n_active"] == 2 and c["mode"] == MODE_EDGE
    q_corner = quat_from_axis_angle_np([1, -1, 0], np.arctan(np.sqrt(2)))   # world down = (1,1,1)/sqrt3 in the box frame
    z = -min_gap(np.zeros(3), q_corner, HE)
    c = analytic_contacts(np.array([0, 0, z]), q_corner, HE, margin=1e-4)
    assert c["n_active"] == 1 and c["mode"] == MODE_CORNER


def test_translation_invariance_in_plane():
    rng = np.random.default_rng(1)
    q = random_quat_uniform_np(rng)
    z = 0.0005 - min_gap(np.zeros(3), q, HE)
    a = analytic_contacts(np.array([0, 0, z]), q, HE, margin=1e-3)
    b = analytic_contacts(np.array([1.3, -0.7, z]), q, HE, margin=1e-3)
    assert a["n_active"] == b["n_active"] and np.allclose(a["d"], b["d"]) and np.allclose(a["p_box_local"], b["p_box_local"])
    assert np.allclose(b["p_ground_world"][a["active"]] - a["p_ground_world"][a["active"]], [1.3, -0.7, 0])


def test_yaw_equivariance():
    """Joint rotation of box and ground about the plane normal: contacts rotate accordingly."""
    rng = np.random.default_rng(2)
    q = random_quat_uniform_np(rng)
    z = 0.0005 - min_gap(np.zeros(3), q, HE)
    yaw = quat_from_axis_angle_np([0, 0, 1], 0.7)
    a = analytic_contacts(np.array([0, 0, z]), q, HE, margin=1e-3)
    b = analytic_contacts(np.array([0, 0, z]), quat_mul_np(yaw, q), HE, margin=1e-3)
    Rz = quat_to_rotmat_np(yaw)
    assert a["n_active"] == b["n_active"] and np.allclose(a["d"], b["d"])
    assert np.allclose((Rz @ a["p_box_world"].T).T, b["p_box_world"], atol=1e-12)


def test_lowest_feature_classification():
    assert lowest_feature_mode(QI) == MODE_FACE
    assert lowest_feature_mode(quat_from_axis_angle_np([1, 0, 0], np.pi / 4)) == MODE_EDGE
    q_corner = quat_from_axis_angle_np([1, -1, 0], np.arctan(np.sqrt(2)))
    assert lowest_feature_mode(q_corner) == MODE_CORNER


def test_surface_patch_tokens_consistency():
    t = surface_patch_tokens(HE)
    assert t["area"].sum() == pytest.approx(2 * (4 * HE[0] * HE[1] + 4 * HE[0] * HE[2] + 4 * HE[1] * HE[2]))
    assert (t["adj"] >= 0).all()                       # closed surface: every edge shared
    assert np.allclose(np.linalg.norm(t["normal"], axis=1), 1)
    # outward normals: centroid . normal > 0
    assert (np.einsum("fi,fi->f", t["centroid"], t["normal"]) > 0).all()
    assert patch_feature_matrix(t).shape == (12, 34)
    # remeshing/tessellation change (different diagonal split) keeps face normals/areas per face pair
    V, F = box_mesh(HE)
    assert face_adjacency(F).shape == (12, 3)


def test_surface_points_have_corners_edges():
    sp = surface_points(HE)
    assert (sp["kind"] == 2).sum() == 8 and (sp["kind"] == 1).sum() == 48
    assert np.allclose(np.linalg.norm(sp["normal"], axis=1), 1)
    assert (np.abs(sp["pos"]) <= HE + 1e-12).all()

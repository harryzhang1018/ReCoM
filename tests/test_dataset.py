"""Dataset tests on a tiny generated set: schema, canonical frames, categories, splits, windows."""
import numpy as np
import pytest

pytest.importorskip("pychrono")

from recom.config import DatasetGenConfig, PhysicsConfig
from recom.data.canonicalize import canonicalize_episode
from recom.data.dataset import CAT_NAMES, ContactQueryDataset, TransitionWindowDataset, episode_to_arrays, frame_categories
from recom.data.splits import check_no_leakage, make_splits
from recom.data.storage import load_episode, save_episode
from recom.data.validate import validate_episode
from recom.sim.chrono_box_drop import run_episode
from recom.sim.sampling import sample_episode


@pytest.fixture(scope="module")
def recs(tmp_path_factory):
    root = tmp_path_factory.mktemp("ds")
    cfg = DatasetGenConfig(name="t", phase="1B", n_geometries=3, clearance_range=(0.1, 0.3), physics=PhysicsConfig(max_duration=0.8))
    out = []
    for i in range(3):
        rec = run_episode(sample_episode(cfg, i))
        save_episode(rec, root)
        out.append(load_episode(root, rec.meta["episode"]["episode_id"]))
    return out


def test_roundtrip_and_schema(recs):
    for rec in recs:
        assert validate_episode(rec) == []
        assert rec.meta["schema_version"].startswith("recom.boxdrop")
        assert rec.meta["collision_envelope_actual"] == pytest.approx(1e-3, abs=1e-6)


def test_canonicalization_is_idempotent_and_traceable(recs):
    rec = recs[0]
    before = {k: v.copy() for k, v in rec.canon.items()}
    canonicalize_episode(rec)
    for k in before:
        assert np.array_equal(before[k], rec.canon[k]), k
    assert (rec.canon["c_raw_index"][rec.canon["c_active"]] >= 0).all()


def test_categories_cover_expected_strata(recs):
    for rec in recs:
        cat = frame_categories(rec)
        assert cat.shape[0] == rec.n_steps
        names = {CAT_NAMES[c] for c in np.unique(cat)}
        assert "far_free" in names and "first_impact" in names


def test_windows_and_query_dataset(recs):
    from recom.data.dataset import EpisodeCache
    cache = EpisodeCache.__new__(EpisodeCache)
    cache.episodes = [episode_to_arrays(r) for r in recs]
    ds = TransitionWindowDataset(cache, T=16)
    item = ds[len(ds) - 1]
    assert item["states"].shape == (16, 13) and item["targets"].shape == (16, 6)
    assert np.allclose(item["targets"].numpy(), item["next_states"][:, 7:13].numpy() - item["states"][:, 7:13].numpy())
    cq = ContactQueryDataset(cache)
    assert len(cq) == sum(r.n_steps for r in recs)
    s = cq.balanced_sampler(100)
    assert len(list(s)) == 100


def test_splits_by_episode_and_geometry_group():
    eps = [{"episode_id": f"e{i}", "geometry_group": f"g{i % 10}", "clearance": 0.5, "approach_mode": 1 + i % 3} for i in range(100)]
    sp = make_splits({"episodes": eps}, seed=0, geometry_holdout_frac=0.2)
    check_no_leakage(sp)
    held_groups = {e["geometry_group"] for e in eps if e["episode_id"] in set(sp["test_geometry"])}
    train_groups = {e["geometry_group"] for e in eps if e["episode_id"] in set(sp["train"])}
    assert held_groups and not (held_groups & train_groups)

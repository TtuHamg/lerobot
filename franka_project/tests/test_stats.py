from __future__ import annotations

import numpy as np

from franka_eef_pipeline.stats import (
    REQUIRED_STATS,
    canonical_sha256,
    effective_eef_stats,
    json_ready,
    normalize,
    unnormalize,
)


def test_effective_stats_schema_counts_and_round_trip() -> None:
    rng = np.random.default_rng(31)
    state = rng.normal(size=(20, 10))
    action = rng.normal(size=(8, 50, 7))
    stats = effective_eef_stats(
        state,
        action,
        observation_fps=15,
        action_fps=15,
        chunk_size=50,
        source_dataset_hash="abc",
    )
    assert set(stats["observation.state"]) == set(REQUIRED_STATS)
    assert set(stats["action"]) == set(REQUIRED_STATS)
    assert int(stats["observation.state"]["count"][0]) == 20
    assert int(stats["action"]["count"][0]) == 400
    recovered = unnormalize(normalize(action, stats["action"]), stats["action"])
    np.testing.assert_allclose(recovered, action, atol=1e-12)


def test_canonical_stats_hash_is_stable_after_json_conversion() -> None:
    value = {"x": np.array([1.0, 2.0]), "count": np.array([2])}
    assert canonical_sha256(value) == canonical_sha256(json_ready(value))

"""Tests for scoring candidate segments: the batched mixture scorer, the device choice and the decision.

The batched scorer is checked against scikit-learn's own ``score_samples``, and the GPU path, when a
GPU is usable here, against the CPU path.
"""

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from frog_presence import detector as D
from frog_presence import inference as I
from frog_presence.device import array_module, describe, to_numpy


def _models(n=3, d=4, components=3, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for k in range(n):
        data = rng.normal(loc=3.0 * k, scale=1.0 + 0.3 * k, size=(300, d))
        out[f"f0{k + 1}"] = GaussianMixture(components, covariance_type="diag", random_state=k).fit(data)
    return out


def test_batched_scorer_equals_scikit_learn():
    model = _models(1)["f01"]
    X = np.random.default_rng(1).normal(size=(500, 4))
    assert I.diag_gmm_log_likelihood(model, X) == pytest.approx(model.score_samples(X), abs=1e-9)


def test_scorer_handles_more_frames_than_one_block():
    model = _models(1)["f01"]
    X = np.random.default_rng(2).normal(size=(250_001, 4))
    assert I.diag_gmm_log_likelihood(model, X) == pytest.approx(model.score_samples(X), abs=1e-8)


def test_mean_scores_of_each_block_equal_the_mean_of_scikit_learn_scores_and_empty_blocks_are_nan():
    models = _models()
    rng = np.random.default_rng(3)
    blocks = [rng.normal(size=(7, 4)), np.empty((0, 4)), rng.normal(loc=3.0, size=(20, 4))]
    scores = I.mean_log_likelihoods(models, blocks)
    for k, model in enumerate(models.values()):
        assert scores[0, k] == pytest.approx(model.score_samples(blocks[0]).mean())
        assert scores[2, k] == pytest.approx(model.score_samples(blocks[2]).mean())
    assert np.isnan(scores[1]).all()


def test_detection_agrees_with_the_decision_rule_and_leaves_invalid_candidates_unaccepted():
    models = _models()
    rng = np.random.default_rng(4)
    blocks = [rng.normal(loc=0.0, size=(15, 4)), np.empty((0, 4)), rng.normal(loc=6.0, size=(15, 4))]
    thresholds = {c: 1.0 for c in models}
    out = I.detect(models, thresholds, blocks)
    assert out["valid"].tolist() == [True, False, True]
    assert out["hypothesis"][1] == -1 and not out["accepted"][1] and np.isnan(out["llr"][1])
    expected = D.decide(out["scores"][[0, 2]], np.array([1.0, 1.0, 1.0]))
    assert out["hypothesis"][[0, 2]].tolist() == expected["hypothesis"].tolist()
    assert out["accepted"][[0, 2]].tolist() == expected["accepted"].tolist()


def test_asking_for_the_cpu_returns_numpy_and_round_trips():
    xp = array_module(prefer_gpu=False)
    assert xp is np and describe(xp) == "CPU (numpy)"
    assert to_numpy(np.arange(3)).tolist() == [0, 1, 2]


def test_the_gpu_path_gives_the_same_scores_as_the_cpu_path_when_a_gpu_is_usable():
    xp = array_module(prefer_gpu=True)
    if xp is np:
        pytest.skip("no usable GPU here, so the fallback to the CPU is what runs")
    model = _models(1)["f01"]
    X = np.random.default_rng(5).normal(size=(1000, 4))
    gpu = to_numpy(I.diag_gmm_log_likelihood(model, xp.asarray(X), xp))
    assert gpu == pytest.approx(I.diag_gmm_log_likelihood(model, X), abs=1e-9)

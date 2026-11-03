"""Tests for training and scoring the per-species mixtures.

They run on small synthetic corpora where the right answer is known, so they document what
each piece is expected to guarantee: the split never shares a call between training and
validation, the initialisation follows the article, scoring a batch of calls equals scoring
them one at a time, and the weighted error rate is the mean of the per-species errors.
"""

import numpy as np
import pandas as pd
import pytest

from frog_presence.gmm import (
    CallCorpus,
    build_global_recording_split,
    build_recording_split,
    build_split,
    evaluate_models,
    fit_gmm,
    fit_species_models,
    initial_parameters,
    score_calls,
    select_training_calls,
    species_number,
)


@pytest.fixture
def corpus():
    """Two well-separated species, 12 calls each, 0.5 s per call, 10 frames per call."""
    rng = np.random.default_rng(0)
    rows, blocks = [], []
    for code, centre in (("f01", -5.0), ("f02", 5.0)):
        for _ in range(12):
            rows.append({"code": code, "species": f"species {code}", "seconds": 0.5, "frames": 10})
            blocks.append(rng.normal(centre, 1.0, size=(10, 3)))
    return CallCorpus(np.vstack(blocks), pd.DataFrame(rows))


def test_species_number_reads_the_digits():
    assert species_number("f04") == 4
    assert species_number("none") == 0


def test_corpus_slices_follow_the_frame_counts(corpus):
    assert corpus.calls.frame_start.tolist()[:3] == [0, 10, 20]
    assert corpus.frames(["call_0001"]).shape == (10, 3)


def test_corpus_rejects_frames_that_do_not_add_up():
    calls = pd.DataFrame({"code": ["f01"], "species": ["a"], "seconds": [1.0], "frames": [10]})
    with pytest.raises(ValueError):
        CallCorpus(np.zeros((9, 3)), calls)


def test_selection_reaches_the_target_without_sharing_calls(corpus):
    split = select_training_calls(corpus.calls, "f01", target_seconds=2.0, seed=1)
    assert split["actual_seconds"] >= 2.0
    assert split["overshoot_seconds"] == pytest.approx(split["actual_seconds"] - 2.0)
    assert not set(split["train_ids"]) & set(split["validation_ids"])
    assert split["n_train_calls"] + split["n_validation_calls"] == 12


def test_selection_is_reproducible_and_depends_on_the_seed(corpus):
    first = select_training_calls(corpus.calls, "f01", 2.0, seed=1)
    again = select_training_calls(corpus.calls, "f01", 2.0, seed=1)
    other = select_training_calls(corpus.calls, "f01", 2.0, seed=2)
    assert first["train_ids"] == again["train_ids"]
    assert first["train_ids"] != other["train_ids"]


def test_selection_always_leaves_a_validation_call(corpus):
    # The species holds 6 s in total, so a 5.9 s target needs 12 of the 12 calls and cannot
    # leave one for validation, whichever shuffle is tried.
    with pytest.raises(RuntimeError):
        select_training_calls(corpus.calls, "f01", target_seconds=5.9, seed=1)


def test_selection_refuses_a_target_the_species_cannot_support(corpus):
    with pytest.raises(ValueError):
        select_training_calls(corpus.calls, "f01", target_seconds=6.0, seed=1)


def test_split_seed_ignores_the_number_of_components(corpus):
    train_a, _, summary_a = build_split(corpus.calls, ["f01", "f02"], 2.0, repeat=0, base_seed=5)
    train_b, _, summary_b = build_split(corpus.calls, ["f01", "f02"], 2.0, repeat=0, base_seed=5)
    assert train_a == train_b
    assert summary_a.equals(summary_b)
    assert summary_a.seed.tolist() == [5 + 2 * 100 + 1, 5 + 2 * 100 + 2]


def test_initial_parameters_follow_the_article(corpus):
    X = corpus.frames(corpus.calls.call_id[:12])
    weights, means, precisions = initial_parameters(X, 4, seed=0)
    assert weights == pytest.approx(np.full(4, 0.25))
    assert means.shape == (4, 3)
    assert precisions[0] == pytest.approx(1.0 / X.var(axis=0))
    assert (precisions == precisions[0]).all()


def test_initial_parameters_need_at_least_as_many_frames_as_components():
    with pytest.raises(ValueError):
        initial_parameters(np.zeros((3, 2)), 4, seed=0)


def test_the_supplied_initialisation_replaces_the_random_one(corpus):
    X = corpus.frames(corpus.calls.call_id[:12])
    a = fit_gmm(X, 3, seed=7)
    b = fit_gmm(X, 3, seed=7)
    assert np.allclose(a.means_, b.means_)
    assert a.converged_


def test_scoring_a_batch_equals_scoring_each_call(corpus):
    train_ids, _, _ = build_split(corpus.calls, ["f01", "f02"], 2.0, 0, 3)
    models = fit_species_models(corpus, train_ids, 2, 2.0, 0, 3)
    ids = list(corpus.calls.call_id[[0, 5, 14, 20]])
    batch = score_calls(models, corpus, ids)
    for row, cid in zip(batch, ids):
        one_by_one = [model.score_samples(corpus.frames([cid])).mean() for model in models.values()]
        assert row == pytest.approx(one_by_one)


def test_well_separated_species_are_identified_and_wer_is_the_macro_average(corpus):
    codes = ["f01", "f02"]
    train_ids, validation_ids, summary = build_split(corpus.calls, codes, 2.0, 0, 3)
    models = fit_species_models(corpus, train_ids, 2, 2.0, 0, 3)
    per_species, predictions, wer = evaluate_models(models, corpus, validation_ids, summary, 2, 2.0, 0)

    assert wer == pytest.approx(per_species.error_rate.mean())
    assert wer == 0.0
    assert len(predictions) == sum(len(v) for v in validation_ids.values())
    assert per_species.n_validation_calls.tolist() == [len(validation_ids[c]) for c in codes]


@pytest.fixture
def recorded_corpus():
    """Two species, four recordings each, three 0.5 s calls per recording."""
    rng = np.random.default_rng(0)
    rows, blocks = [], []
    for code, centre in (("f01", -5.0), ("f02", 5.0)):
        for wav in range(4):
            for _ in range(3):
                rows.append({"code": code, "species": f"species {code}", "wav": f"{code}_{wav}.wav",
                             "seconds": 0.5, "frames": 10})
                blocks.append(rng.normal(centre, 1.0, size=(10, 3)))
    return CallCorpus(np.vstack(blocks), pd.DataFrame(rows))


def test_recording_split_never_shares_a_recording(recorded_corpus):
    calls = recorded_corpus.calls.set_index("call_id")
    train, validation, summary = build_recording_split(recorded_corpus.calls, ["f01", "f02"], 1.0, 0, 9)
    for code in ("f01", "f02"):
        assert not set(calls.loc[list(train[code]), "wav"]) & set(calls.loc[list(validation[code]), "wav"])
        assert len(train[code]) and len(validation[code])
    assert summary.n_validation_calls.tolist() == [len(validation["f01"]), len(validation["f02"])]


def test_recording_split_trains_on_half_of_the_recordings(recorded_corpus):
    calls = recorded_corpus.calls.set_index("call_id")
    train, validation, _ = build_recording_split(recorded_corpus.calls, ["f01"], 99.0, 0, 9)
    # A target nobody can reach uses every call of the training recordings: 2 of 4 recordings.
    assert len(train["f01"]) == 6 and len(validation["f01"]) == 6
    assert calls.loc[list(train["f01"]), "wav"].nunique() == 2


def test_recording_split_is_reproducible_and_needs_two_recordings(recorded_corpus):
    a = build_recording_split(recorded_corpus.calls, ["f01"], 1.0, 3, 9)
    b = build_recording_split(recorded_corpus.calls, ["f01"], 1.0, 3, 9)
    assert a[0] == b[0] and a[1] == b[1]
    single = recorded_corpus.calls[recorded_corpus.calls.wav == "f01_0.wav"]
    with pytest.raises(ValueError):
        build_recording_split(single, ["f01"], 1.0, 0, 9)


@pytest.fixture
def shared_recording_corpus():
    """Two species over six recordings; recording ``r2`` carries calls of both species."""
    rng = np.random.default_rng(1)
    layout = {"f01": ["r0", "r1", "r2", "r3"], "f02": ["r2", "r4", "r5"]}
    rows, blocks = [], []
    for code, recordings in layout.items():
        for wav in recordings:
            for _ in range(4):
                rows.append({"code": code, "species": f"species {code}", "wav": f"{wav}.wav",
                             "seconds": 0.5, "frames": 10})
                blocks.append(rng.normal(0.0, 1.0, size=(10, 3)))
    return CallCorpus(np.vstack(blocks), pd.DataFrame(rows))


def test_global_split_puts_every_recording_on_one_side_for_all_species(shared_recording_corpus):
    calls = shared_recording_corpus.calls.set_index("call_id")
    train, validation, summary = build_global_recording_split(
        shared_recording_corpus.calls, ["f01", "f02"], 2.0, 0, 3, n_attempts=50)
    train_wavs = {w for ids in train.values() for w in calls.loc[list(ids), "wav"]}
    validation_wavs = {w for ids in validation.values() for w in calls.loc[list(ids), "wav"]}
    assert not train_wavs & validation_wavs
    assert (summary.training_seconds_actual >= 2.0).all()
    assert (summary.n_validation_calls > 0).all()


def test_global_split_is_reproducible_and_keeps_the_smallest_overshoot(shared_recording_corpus):
    args = (shared_recording_corpus.calls, ["f01", "f02"], 2.0, 1, 3)
    a = build_global_recording_split(*args, n_attempts=200)
    b = build_global_recording_split(*args, n_attempts=200)
    assert a[0] == b[0] and a[2].equals(b[2])
    one = build_global_recording_split(*args, n_attempts=1)[2].overshoot_seconds.sum()
    many = build_global_recording_split(*args, n_attempts=200)[2].overshoot_seconds.sum()
    assert many <= one + 1e-9


def test_global_split_fails_when_no_species_can_keep_validation_material(shared_recording_corpus):
    # 99 s is more than any species holds, so no assignment can reach the target.
    with pytest.raises(RuntimeError):
        build_global_recording_split(shared_recording_corpus.calls, ["f01", "f02"], 99.0, 0, 3, n_attempts=20)

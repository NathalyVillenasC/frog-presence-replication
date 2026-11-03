"""Tests for the likelihood-ratio detector: scores, thresholds, the calibration split and storage.

The cases are small enough to work out by hand, so they document what each rule is meant to do.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture

from frog_presence import detector as D


def test_llr_subtracts_the_median_of_the_other_models():
    scores = np.array([[10.0, 4.0, 2.0, 0.0]])
    llr = D.llr_matrix(scores)[0]
    assert llr[0] == pytest.approx(10.0 - 2.0)   # median of 4, 2, 0 is 2
    assert llr[1] == pytest.approx(4.0 - 2.0)    # median of 10, 2, 0 is 2
    assert llr[3] == pytest.approx(0.0 - 4.0)    # median of 10, 4, 2 is 4


def test_llr_with_an_even_number_of_alternatives_averages_the_middle_two():
    scores = np.array([[9.0, 1.0, 2.0, 3.0, 8.0]])
    assert D.llr_matrix(scores)[0, 0] == pytest.approx(9.0 - 2.5)   # alternatives 1, 2, 3, 8


def test_decision_accepts_the_winner_only_above_its_own_threshold():
    scores = np.array([[10.0, 4.0, 2.0], [5.0, 4.5, 4.0], [1.0, 9.0, 3.0]])
    out = D.decide(scores, thresholds=np.array([3.0, 100.0, 3.0]))
    assert out["hypothesis"].tolist() == [0, 0, 1]
    assert out["accepted"].tolist() == [True, False, False]   # row 3 wins model 1, whose threshold is 100


def test_confusion_counts():
    c = D.confusion_counts(np.array([1, 1, 0, 0, 0], bool), np.array([1, 0, 1, 0, 0], bool))
    assert (c["TP"], c["FN"], c["FP"], c["TN"]) == (1, 1, 1, 2)
    assert c["recall"] == 0.5 and c["precision"] == 0.5 and c["FPR"] == pytest.approx(1 / 3)


def test_one_vs_all_threshold_respects_the_false_positive_ceiling():
    llr = np.array([9.0, 8.0, 7.0, 6.0, 1.0, 0.5, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, -10.0, -11.0, -12.0, -13.0])
    positive = np.zeros(20, bool)
    positive[:4] = True
    threshold = D.one_vs_all_threshold(llr, positive, max_fpr=0.05)
    detected = llr >= threshold
    assert (~positive & detected).sum() / (~positive).sum() <= 0.05
    assert (positive & detected).sum() == 4   # all four positives are above every negative


def test_end_to_end_threshold_ignores_calls_the_species_cannot_win():
    # Negatives with a high ratio that do not win cannot be detected, so they must not raise the threshold.
    llr = np.array([6.0, 5.0, 4.0, 9.0, 9.5])
    wins = np.array([True, True, True, False, False])
    positive = np.array([True, True, True, False, False])
    assert D.end_to_end_threshold(llr, wins, positive) == 4.0
    # The ROC rule on the same numbers must sit above the high-ratio negatives, which accepts no positive.
    assert D.one_vs_all_threshold(llr, positive, max_fpr=0.0) == float("inf")


def test_end_to_end_threshold_takes_the_largest_among_equal_recalls():
    llr = np.array([8.0, 7.0, 2.0])
    wins = np.array([True, True, True])
    positive = np.array([True, True, False])
    assert D.end_to_end_threshold(llr, wins, positive, max_fpr=0.0) == 7.0


def _corpus():
    rows = []
    layout = {"f01": ["a", "b", "c", "d", "e", "f"], "f02": ["c", "g", "h", "i", "j", "k"]}
    for code, wavs in layout.items():
        for wav in wavs:
            for _ in range(3):
                rows.append({"code": code, "species": code, "wav": f"{wav}.wav", "seconds": 1.0})
    return pd.DataFrame(rows)


def test_calibration_split_never_shares_a_recording_and_keeps_every_species_on_both_sides():
    calls = _corpus()
    split = D.calibration_split(calls, ["f01", "f02"], 0.25, seed=1, n_attempts=200)
    assert not split["train"] & split["calibration"]
    for code in ("f01", "f02"):
        wavs = set(calls.loc[calls.code == code, "wav"].str.lower())
        assert wavs & split["train"] and wavs & split["calibration"]


def test_calibration_split_is_reproducible():
    calls = _corpus()
    a = D.calibration_split(calls, ["f01", "f02"], 0.25, seed=1, n_attempts=200)
    b = D.calibration_split(calls, ["f01", "f02"], 0.25, seed=1, n_attempts=200)
    assert a == b


def _fit(seed):
    rng = np.random.default_rng(seed)
    return GaussianMixture(2, covariance_type="diag", random_state=seed).fit(rng.normal(size=(60, 3)))


def test_equal_models_have_equal_parameters_and_different_ones_do_not():
    assert D.parameters_equal(_fit(0), _fit(0))
    assert not D.parameters_equal(_fit(0), _fit(1))


def test_a_saved_detector_loads_back_and_a_changed_model_is_refused(tmp_path):
    models = {"f01": _fit(0), "f02": _fit(1)}
    manifest = D.save_detector(tmp_path, models, {"f01": 1.5, "f02": 2.5}, {"feature": "x"})
    assert manifest["classes"] == ["f01", "f02"] and set(manifest["gmm_sha256"]) == {"f01.joblib", "f02.joblib"}

    loaded, thresholds, _ = D.load_detector(tmp_path)
    assert thresholds == {"f01": 1.5, "f02": 2.5}
    assert D.parameters_equal(loaded["f01"], models["f01"])

    (tmp_path / "f02.joblib").write_bytes(b"changed")
    with pytest.raises(ValueError):
        D.load_detector(tmp_path)


def test_saving_removes_models_left_by_an_earlier_detector(tmp_path):
    D.save_detector(tmp_path, {"f01": _fit(0), "f09": _fit(2)}, {"f01": 1.0, "f09": 1.0}, {})
    D.save_detector(tmp_path, {"f01": _fit(0)}, {"f01": 1.0}, {})
    assert not (tmp_path / "f09.joblib").exists()


def test_the_manifest_can_be_rewritten_without_touching_the_model_files(tmp_path):
    models = {"f01": _fit(0)}
    first = D.save_detector(tmp_path, models, {"f01": 1.0}, {"notebook": "old"})
    before = (tmp_path / "f01.joblib").read_bytes()
    second = D.save_detector(tmp_path, models, {"f01": 2.0}, {"notebook": "new"}, rewrite_models=False)
    assert (tmp_path / "f01.joblib").read_bytes() == before
    assert second["gmm_sha256"] == first["gmm_sha256"] and second["notebook"] == "new"
    assert D.load_detector(tmp_path)[1] == {"f01": 2.0}

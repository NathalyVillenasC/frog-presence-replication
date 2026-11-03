"""Tests for applying the frozen segmentation to a long recording and scoring its candidates."""

import numpy as np
import pandas as pd
import pytest
from sklearn.mixture import GaussianMixture

from frog_presence import field as FL
from frog_presence import inference as I

SR = 48_000
CONFIG = {
    "paper_frontend": {"species_frequency_bands_hz": {"f01": [1500.0, 2500.0], "f02": [5000.0, 7000.0]}},
    "selection": {"C_by_code": {"f01": 3.0, "f02": 3.0}, "padding_ms_by_code": {"f01": 20, "f02": 0}},
}


@pytest.fixture
def recording():
    """Forty seconds of quiet noise with a 2 kHz burst, which lies inside the first band only."""
    rng = np.random.default_rng(0)
    audio = 0.001 * rng.standard_normal(SR * 40)
    audio[SR * 12: SR * 12 + SR // 2] += 0.5 * np.sin(2 * np.pi * 2000 * np.arange(SR // 2) / SR)
    return audio


def test_each_band_proposes_its_own_candidates_and_only_the_matching_band_sees_the_burst(recording):
    out = FL.segment_recording(recording, SR, CONFIG)
    assert set(out.band) <= {"f01", "f02"} and list(out.columns) == ["band", "start", "end"]
    hit = out[(out.start < SR * 12 + SR // 2) & (out.end > SR * 12)]
    assert "f01" in set(hit.band)
    # The burst is far outside the 5 to 7 kHz band, so that band's energy there is no larger than elsewhere.
    inside_f02 = hit[hit.band == "f02"]
    assert (inside_f02.end - inside_f02.start).sum() < (hit[hit.band == "f01"].end - hit[hit.band == "f01"].start).sum()


def test_candidates_lie_inside_the_recording_and_are_ordered_within_a_band(recording):
    out = FL.segment_recording(recording, SR, CONFIG)
    assert (out.start >= 0).all() and (out.end <= len(recording)).all() and (out.end > out.start).all()
    for _, band in out.groupby("band"):
        assert (band.start.to_numpy()[1:] > band.end.to_numpy()[:-1]).all()   # merged, so disjoint


def test_padding_widens_candidates_of_that_band_only(recording):
    padded = FL.segment_recording(recording, SR, CONFIG)
    unpadded_config = {**CONFIG, "selection": {**CONFIG["selection"], "padding_ms_by_code": {"f01": 0, "f02": 0}}}
    plain = FL.segment_recording(recording, SR, unpadded_config)
    width = lambda df, b: int((df[df.band == b].end - df[df.band == b].start).sum())
    assert width(padded, "f01") > width(plain, "f01")
    assert width(padded, "f02") == width(plain, "f02")


def test_candidate_frames_count_frames_and_give_none_for_a_very_short_candidate(recording):
    intervals = pd.DataFrame({"band": ["f01", "f01"], "start": [SR * 12, SR * 20], "end": [SR * 12 + SR // 4, SR * 20 + 100]})
    frames, lengths = FL.candidate_frames(recording, SR, intervals)
    assert lengths[0] > 0 and lengths[1] == 0
    assert frames.shape == (lengths.sum(), 20) and np.isfinite(frames).all()


def test_stacked_scores_equal_scores_from_blocks():
    rng = np.random.default_rng(1)
    models = {f"f0{k}": GaussianMixture(2, covariance_type="diag", random_state=k).fit(rng.normal(3.0 * k, 1, (200, 20))) for k in range(3)}
    blocks = [rng.normal(size=(5, 20)), np.empty((0, 20)), rng.normal(3.0, 1, size=(9, 20))]
    lengths = np.array([len(b) for b in blocks])
    stacked = np.vstack([b for b in blocks if len(b)])
    from_blocks = I.mean_log_likelihoods(models, blocks)
    from_stack = I.mean_log_likelihoods_stacked(models, stacked, lengths)
    assert np.allclose(from_blocks, from_stack, equal_nan=True)
    out = I.decide_scores(models, {c: 0.0 for c in models}, from_stack)
    assert out["valid"].tolist() == [True, False, True] and out["hypothesis"][1] == -1

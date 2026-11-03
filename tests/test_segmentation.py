"""Tests for the segmentation front end: energy, endpoints, thresholds and interval arithmetic.

Everything is checked on cases small enough to work out by hand, so the tests document what each
step is expected to do.
"""

import numpy as np
import pytest

from frog_presence import segmentation as S

SR = 48_000


def test_short_time_energy_sums_squares_over_whole_frames_and_drops_the_remainder():
    audio = np.ones(10 * 480 + 100)
    energy, frame = S.short_time_energy(audio, SR)
    assert frame == 480 and len(energy) == 10
    assert energy == pytest.approx(np.full(10, 480.0))


def test_a_signal_shorter_than_a_frame_has_no_energy_values():
    energy, _ = S.short_time_energy(np.ones(100), SR)
    assert len(energy) == 0


def test_the_moving_average_keeps_the_length_and_smooths_a_spike():
    out = S.smooth(np.r_[np.zeros(20), 12.0, np.zeros(20)], 12)
    assert len(out) == 41 and out.max() == pytest.approx(1.0) and (out > 0).sum() == 12


def test_energy_in_decibels_is_floored_and_not_minus_infinity():
    out = S.energy_to_db(np.array([0.0, 1.0, 100.0]))
    assert np.isfinite(out).all() and out[1] == 0.0 and out[2] == pytest.approx(20.0)


def test_the_threshold_lies_between_the_mean_and_the_maximum_and_approaches_the_mean_as_c_grows():
    ste_db = np.array([0.0, 0.0, 0.0, 12.0])               # mean 3, max 12
    assert S.endpoint_threshold_db(ste_db, 1.0) == pytest.approx(12.0)
    assert S.endpoint_threshold_db(ste_db, 3.0) == pytest.approx(6.0)
    assert S.endpoint_threshold_db(ste_db, 1e9) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        S.endpoint_threshold_db(ste_db, 0.0)


def test_a_segment_starts_after_three_values_above_and_ends_after_three_below():
    ste = np.array([0, 0, 0, 5, 5, 5, 5, 5, 0, 0, 0, 0, 0], float)
    assert S.find_endpoints(ste, 2.0) == [(3, 8)]


def test_two_values_above_do_not_start_a_segment():
    assert S.find_endpoints(np.array([0, 0, 5, 5, 0, 0, 0, 0], float), 2.0) == []


def test_a_segment_still_open_at_the_end_is_closed_there():
    assert S.find_endpoints(np.array([0, 0, 5, 5, 5, 5], float), 2.0) == [(2, 6)]


def test_a_brief_dip_shorter_than_three_values_does_not_end_the_segment():
    ste = np.array([5, 5, 5, 0, 0, 5, 5, 5, 0, 0, 0, 0], float)
    assert S.find_endpoints(ste, 2.0) == [(0, 8)]


def test_the_bandpass_keeps_a_tone_inside_the_band_and_removes_one_outside():
    t = np.arange(SR) / SR
    inside, outside = np.sin(2 * np.pi * 2000 * t), np.sin(2 * np.pi * 8000 * t)
    kept = S.bandpass(inside, SR, 1500, 2500)[SR // 4: -SR // 4]
    removed = S.bandpass(outside, SR, 1500, 2500)[SR // 4: -SR // 4]
    assert np.abs(kept).max() > 0.9 and np.abs(removed).max() < 0.01


def test_an_invalid_band_is_refused():
    with pytest.raises(ValueError):
        S.bandpass_taps(SR, 3000, 2000)
    with pytest.raises(ValueError):
        S.bandpass_taps(SR, 1000, SR)


def test_a_tone_burst_in_quiet_noise_is_found_inside_the_right_block():
    rng = np.random.default_rng(0)
    audio = 0.001 * rng.standard_normal(SR * 40)
    burst = slice(SR * 33, SR * 33 + SR // 2)
    audio[burst] += 0.5 * np.sin(2 * np.pi * 2000 * np.arange(SR // 2) / SR)
    blocks = S.prepare_blocks(audio, SR, 1500, 2500)
    assert len(blocks) == 2 and blocks[1]["end"] == SR * 40
    found = S.candidates_from_blocks(blocks, C=2.0)
    inside = found[(found[:, 1] < SR * 33 + SR // 2) & (found[:, 2] > SR * 33)]
    assert len(inside) >= 1 and (inside[:, 0] == 1).all()
    covered = S.overlap_with(SR * 33, SR * 33 + SR // 2, S.merge_intervals(found[:, 1:]))
    assert covered > 0.8 * (SR // 2)


def test_a_larger_c_gives_at_least_as_many_candidate_samples():
    rng = np.random.default_rng(1)
    audio = 0.01 * rng.standard_normal(SR * 10)
    audio[SR * 4: SR * 4 + 4800] += np.sin(2 * np.pi * 2000 * np.arange(4800) / SR)
    blocks = S.prepare_blocks(audio, SR, 1500, 2500)

    def covered(C):
        merged = S.merge_intervals(S.candidates_from_blocks(blocks, C)[:, 1:])
        return int((merged[:, 1] - merged[:, 0]).sum())

    sizes = [covered(C) for C in (1.5, 3.0, 8.0)]
    assert sizes[0] <= sizes[1] <= sizes[2]


def test_merging_joins_overlapping_and_touching_intervals():
    assert S.merge_intervals([(5, 9), (0, 3), (3, 4), (8, 12)]).tolist() == [[0, 4], [5, 12]]
    assert S.merge_intervals([(5, 5)]).shape == (0, 2)


def test_padding_widens_clips_and_merges():
    padded = S.pad_intervals([(100, 200), (230, 300), (950, 990)], pad_samples=20, n_samples=1000)
    assert padded.tolist() == [[80, 320], [930, 1000]]


def test_overlap_counts_the_samples_of_a_target_covered_by_disjoint_intervals():
    cand = S.merge_intervals([(0, 10), (20, 30)])
    assert S.overlap_with(5, 25, cand) == 10      # 5..10 and 20..25
    assert S.overlap_with(40, 50, cand) == 0


def test_evaluation_reports_coverage_recall_and_burden():
    candidates = np.array([[0, 100, 160], [0, 400, 420]])
    summary, coverage = S.evaluate_candidates([100, 300], [200, 350], candidates, sr=1000, coverage_target=0.5)
    assert coverage.tolist() == [0.6, 0.0]
    assert summary["call_recall_any"] == 0.5 and summary["call_recall_50"] == 0.5
    assert summary["n_candidates"] == 2
    assert summary["candidate_seconds"] == pytest.approx(0.08)
    assert summary["annotated_seconds_union"] == pytest.approx(0.15)
    assert summary["candidate_burden_ratio"] == pytest.approx(80 / 150)
    assert summary["candidate_overlap_share"] == pytest.approx(60 / 80)

"""Tests for the baseline cepstral features of the feature comparison.

The front end is checked against its own definition (frame counts, filter-bank shapes), the
cepstral recursion against a case with a known closed form, and the two ways of starting the
RASTA filter against the behaviour that distinguishes them.
"""

import numpy as np
import pytest

from frog_presence import features as F

SR = 48_000


@pytest.fixture
def tone_in_noise():
    """One second of a 3 kHz tone in low-level noise: stationary, so a filter has settled."""
    rng = np.random.default_rng(0)
    t = np.arange(SR) / SR
    return 0.5 * np.sin(2 * np.pi * 3000 * t) + 0.05 * rng.standard_normal(SR)


def test_frame_count_follows_the_overlap():
    frame, hop = 960, 240  # 20 ms and 5 ms at 48 kHz
    audio = np.zeros(frame + 10 * hop + 7)
    assert len(F.frame_audio(audio, SR)) == 11


def test_a_signal_shorter_than_one_frame_gives_no_frames():
    assert F.frame_audio(np.zeros(100), SR).shape == (0, 960)
    assert F.standard_mfcc(np.zeros(100), SR, 12).shape == (0, 12)
    assert F.plp_rasta(np.zeros(100), SR).shape == (0, 20)


def test_pre_emphasis_keeps_the_first_sample_and_differences_the_rest():
    out = F.pre_emphasise(np.array([1.0, 2.0, 4.0]))
    assert out == pytest.approx([1.0, 2.0 - 0.99, 4.0 - 0.99 * 2.0])


def test_mel_bank_shape_and_non_negative_weights():
    bank = F.mel_filterbank(SR)
    assert bank.shape == (40, F.NFFT // 2 + 1)
    assert (bank >= 0).all()


def test_the_twelve_coefficients_are_the_first_twelve_of_twenty(tone_in_noise):
    twelve = F.standard_mfcc(tone_in_noise, SR, 12)
    twenty = F.standard_mfcc(tone_in_noise, SR, 20)
    assert twelve.shape[1] == 12 and twenty.shape[1] == 20
    assert np.allclose(twelve, twenty[:, :12])


def test_bark_scale_increases_and_the_bank_is_triangular():
    hz = np.linspace(0, 24_000, 200)
    assert (np.diff(F.hz_to_bark(hz)) > 0).all()
    bank, centers = F.bark_filterbank(SR)
    assert bank.shape == (F.N_BARK_FILTERS, F.NFFT // 2 + 1)
    assert bank.max() <= 1.0 + 1e-12 and (np.diff(centers) > 0).all()


def test_rasta_started_from_silence_passes_the_first_frame_but_a_steady_start_does_not():
    constant = np.full((50, 4), 3.0)
    from_silence = F.rasta_filter(constant, start="zero")
    steady = F.rasta_filter(constant, start="steady")
    assert from_silence[0] == pytest.approx(0.2 * 3.0)  # start-up transient on a constant input
    assert np.abs(steady).max() < 1e-9                  # nothing changes, so nothing passes


def test_rasta_starts_agree_once_the_filter_has_settled():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((300, 3)) + 5.0
    zero, steady = F.rasta_filter(x, "zero"), F.rasta_filter(x, "steady")
    assert np.abs(zero[:3] - steady[:3]).max() > 0.1
    assert np.abs(zero[-5:] - steady[-5:]).max() < 1e-6


def test_an_unknown_rasta_start_is_refused():
    with pytest.raises(ValueError):
        F.rasta_filter(np.zeros((5, 2)), start="warm")


def test_cepstra_of_a_first_order_predictor_have_the_closed_form():
    # H(z) = 1 / (1 + a z^-1) has cepstrum c_n = (-a)^n / n.
    a = 0.6
    c = F.lpc_to_cepstra(np.array([1.0, a]), error=1.0, n_cepstra=6)
    assert c[0] == pytest.approx(0.0)
    assert c[1:] == pytest.approx([(-a) ** n / n for n in range(1, 6)])


def test_predictor_starts_with_one_and_error_is_positive():
    lpc, error = F.auditory_spectrum_to_lpc(np.linspace(1.0, 2.0, 24))
    assert lpc[0] == 1.0 and len(lpc) == F.PLP_ORDER + 1
    assert error > 0


@pytest.mark.parametrize("rasta", ["zero", "steady", None])
def test_plp_is_finite_and_has_the_requested_shape(tone_in_noise, rasta):
    out = F.plp_rasta(tone_in_noise[:4800], SR, n_cepstra=20, rasta=rasta)
    assert out.shape[1] == 20 and len(out) == len(F.frame_audio(tone_in_noise[:4800], SR))
    assert np.isfinite(out).all()


def test_the_rasta_start_changes_the_first_frames_more_than_the_last(tone_in_noise):
    zero = F.plp_rasta(tone_in_noise, SR, rasta="zero")
    steady = F.plp_rasta(tone_in_noise, SR, rasta="steady")
    early = np.abs(zero[:5] - steady[:5]).mean()
    late = np.abs(zero[-5:] - steady[-5:]).mean()
    assert early > 10 * late


def test_temporal_filtering_changes_the_features(tone_in_noise):
    with_rasta = F.plp_rasta(tone_in_noise[:9600], SR, rasta="zero")
    without = F.plp_rasta(tone_in_noise[:9600], SR, rasta=None)
    assert not np.allclose(with_rasta, without)

"""The baseline cepstral features of the article's feature comparison (section 4.2).

The article compares its modified filter bank (m-FCC, built in the preprocessing notebook)
with standard MFCC and with PLP-RASTA. The article gives no implementation parameters for the
two baselines, so what is here is an explicit reconstruction: every parameter is a named
constant, and the choices the article does not settle are documented where they are made.

All three baselines share the front end of the preprocessing notebook: pre-emphasis, 20 ms
frames with 75 % overlap, a Hamming window and a 1024-point transform.
"""

from functools import lru_cache

import librosa
import numpy as np
from scipy.fft import dct
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter, lfilter_zi

PRE_EMPHASIS = 0.99
FRAME_SECONDS = 0.020
OVERLAP = 0.75
NFFT = 1024

N_MEL_FILTERS = 40

N_BARK_FILTERS = 24
PLP_ORDER = 20

# Classic RASTA: a band-pass numerator and a slow integrator. The coefficients were designed
# for 100 frames per second; this front end produces roughly twice that.
RASTA_NUMERATOR = np.array([0.2, 0.1, 0.0, -0.1, -0.2])
RASTA_DENOMINATOR = np.array([1.0, -0.94])

EPS = np.finfo(float).eps


def pre_emphasise(audio: np.ndarray) -> np.ndarray:
    """Apply H(z) = 1 - 0.99 z^-1, keeping the first sample unchanged."""
    audio = np.asarray(audio, dtype=np.float64)
    out = audio.copy()
    out[1:] = audio[1:] - PRE_EMPHASIS * audio[:-1]
    return out


def frame_audio(audio: np.ndarray, sr: int) -> np.ndarray:
    """Cut a signal into 20 ms frames with 75 % overlap; a signal shorter than a frame gives none."""
    frame_length = int(round(FRAME_SECONDS * sr))
    hop_length = int(round(FRAME_SECONDS * (1.0 - OVERLAP) * sr))
    if len(audio) < frame_length:
        return np.empty((0, frame_length))
    starts = np.arange(1 + (len(audio) - frame_length) // hop_length) * hop_length
    return np.stack([audio[s:s + frame_length] for s in starts])


def frame_power_spectrum(audio: np.ndarray, sr: int) -> np.ndarray:
    """Power spectrum of every pre-emphasised, Hamming-windowed frame.

    Args:
        audio (np.ndarray): Mono signal.
        sr (int): Sample rate in hertz.

    Returns:
        np.ndarray: Array of shape (frames, NFFT // 2 + 1); empty when the signal is shorter
            than one frame.
    """
    frames = frame_audio(pre_emphasise(audio), sr)
    if len(frames) == 0:
        return np.empty((0, NFFT // 2 + 1))
    spectrum = np.fft.rfft(frames * np.hamming(frames.shape[1])[None, :], n=NFFT, axis=1)
    return np.abs(spectrum) ** 2 / NFFT


@lru_cache(maxsize=8)
def mel_filterbank(sr: int) -> np.ndarray:
    """Triangular mel bank from 0 Hz to the Nyquist frequency, as ``librosa`` builds it."""
    return librosa.filters.mel(sr=sr, n_fft=NFFT, n_mels=N_MEL_FILTERS, fmin=0.0,
                               fmax=sr / 2, htk=False, norm=None)


def standard_mfcc(audio: np.ndarray, sr: int, n_mfcc: int) -> np.ndarray:
    """Standard MFCC: log mel energies, DCT, first ``n_mfcc`` coefficients (c0 included).

    Args:
        audio (np.ndarray): Mono signal.
        sr (int): Sample rate in hertz.
        n_mfcc (int): Number of coefficients kept.

    Returns:
        np.ndarray: Array of shape (frames, n_mfcc).
    """
    power = frame_power_spectrum(audio, sr)
    if len(power) == 0:
        return np.empty((0, n_mfcc))
    log_energy = np.log(np.maximum(power @ mel_filterbank(sr).T, EPS))
    return dct(log_energy, type=2, axis=1, norm="ortho")[:, :n_mfcc]


def hz_to_bark(hz):
    """Bark scale, in the form of Traunmuller-style arctangent used by PLP."""
    hz = np.asarray(hz, dtype=float)
    return 13.0 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500.0) ** 2)


@lru_cache(maxsize=8)
def bark_filterbank(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Triangular critical-band filters equally spaced on the Bark scale up to Nyquist.

    Returns:
        tuple: The bank, shape (filters, bins), and the centre frequency of each filter.
    """
    hz_grid = np.linspace(0.0, 24_000.0, 200_001)
    bark_points = np.linspace(0.0, float(hz_to_bark(sr / 2)), N_BARK_FILTERS + 2)
    hz_points = np.interp(bark_points, hz_to_bark(hz_grid), hz_grid)

    freqs = np.fft.rfftfreq(NFFT, d=1.0 / sr)
    bank = np.zeros((N_BARK_FILTERS, len(freqs)))
    for i in range(N_BARK_FILTERS):
        left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        up = (freqs >= left) & (freqs <= center)
        down = (freqs >= center) & (freqs <= right)
        if center > left:
            bank[i, up] = (freqs[up] - left) / (center - left)
        if right > center:
            bank[i, down] = (right - freqs[down]) / (right - center)
    return bank, hz_points[1:-1]


def rasta_filter(log_spectrum: np.ndarray, start: str = "zero") -> np.ndarray:
    """RASTA band-pass filtering of each band's log energy along time.

    Args:
        log_spectrum (np.ndarray): Log critical-band energies, shape (frames, bands).
        start (str): How the filter state is initialised. ``"zero"`` starts every call from a
            silent filter, so the first frames still carry the unfiltered spectrum.
            ``"steady"`` starts from the state the filter would have reached had the first
            frame lasted forever, which removes that start-up transient.

    Returns:
        np.ndarray: Filtered log energies, same shape.
    """
    if start == "zero":
        return lfilter(RASTA_NUMERATOR, RASTA_DENOMINATOR, log_spectrum, axis=0)
    if start == "steady":
        zi = lfilter_zi(RASTA_NUMERATOR, RASTA_DENOMINATOR)[:, None] * log_spectrum[0][None, :]
        return lfilter(RASTA_NUMERATOR, RASTA_DENOMINATOR, log_spectrum, axis=0, zi=zi)[0]
    raise ValueError(f"unknown RASTA start: {start!r}")


def equal_loudness(freq_hz: np.ndarray) -> np.ndarray:
    """Hermansky's equal-loudness weighting at the given centre frequencies."""
    f2 = np.asarray(freq_hz, dtype=float) ** 2
    curve = (f2 + 1.6e5) * f2 ** 2 / np.maximum((f2 + 1.44e6) * (f2 + 9.61e6) ** 2, EPS)
    return np.maximum(curve, EPS)


def auditory_spectrum_to_lpc(spectrum: np.ndarray, order: int = PLP_ORDER) -> tuple[np.ndarray, float]:
    """All-pole model of an auditory spectrum.

    The band samples are treated as one half of an even spectrum, inverse-transformed into an
    autocorrelation sequence, and the normal equations are solved for the predictor.

    Returns:
        tuple: The predictor polynomial (starting with 1) and the prediction error.
    """
    autocorr = np.real(np.fft.irfft(np.asarray(spectrum, dtype=float),
                                    n=2 * (len(spectrum) - 1))[:order + 1])
    autocorr[0] = max(autocorr[0], 1e-10)
    autocorr[0] += 1e-6 * abs(autocorr[0])
    try:
        coeff = solve_toeplitz((autocorr[:order], autocorr[:order]), -autocorr[1:order + 1])
    except Exception:
        coeff = np.zeros(order)
    lpc = np.concatenate([[1.0], coeff])
    error = float(autocorr[0] + np.dot(autocorr[1:order + 1], coeff))
    return lpc, max(error, 1e-12)


def lpc_to_cepstra(lpc: np.ndarray, error: float, n_cepstra: int) -> np.ndarray:
    """Recursion from predictor coefficients to cepstral coefficients; c0 is log(error)."""
    order = len(lpc) - 1
    c = np.zeros(n_cepstra)
    c[0] = np.log(max(error, 1e-12))
    for n in range(1, n_cepstra):
        total = 0.0
        for k in range(max(1, n - order), n):
            if n - k <= order:
                total += (k / n) * c[k] * lpc[n - k]
        c[n] = (-lpc[n] - total) if n <= order else -total
    return c


def plp_rasta(audio: np.ndarray, sr: int, n_cepstra: int = 20, rasta: str | None = "zero") -> np.ndarray:
    """PLP cepstra with RASTA filtering: bands, log, RASTA, loudness, cube root, all-pole fit.

    Args:
        audio (np.ndarray): Mono signal.
        sr (int): Sample rate in hertz.
        n_cepstra (int): Number of cepstral coefficients returned.
        rasta (str | None): ``"zero"`` or ``"steady"`` start of the RASTA filter (see
            ``rasta_filter``), or ``None`` for plain PLP without temporal filtering.

    Returns:
        np.ndarray: Array of shape (frames, n_cepstra).
    """
    power = frame_power_spectrum(audio, sr)
    if len(power) == 0:
        return np.empty((0, n_cepstra))

    bank, centers = bark_filterbank(sr)
    log_bands = np.log(np.maximum(power @ bank.T, EPS))
    if rasta is not None:
        log_bands = rasta_filter(log_bands, start=rasta)

    auditory = np.cbrt(np.maximum(np.exp(log_bands) * equal_loudness(centers)[None, :], EPS))
    return np.array([lpc_to_cepstra(*auditory_spectrum_to_lpc(frame), n_cepstra) for frame in auditory])

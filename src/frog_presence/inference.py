"""Running the frozen detector on candidate segments.

A candidate segment is an interval of the original, unfiltered recording. It becomes a matrix of
PLP-RASTA frames, is scored under every species mixture, identified with the highest mean frame
log-likelihood, and verified with the likelihood ratio against the species' threshold. The same
code serves the labelled calls used in calibration and the long recordings of the field.
"""

import numpy as np

from . import detector as D
from . import features as F


def candidate_features(audio: np.ndarray, sr: int, intervals) -> list[np.ndarray]:
    """PLP-RASTA frames (zero start, as the frozen detector was fitted) for each ``(start, end)`` interval."""
    return [F.plp_rasta(audio[int(s):int(e)], sr, rasta="zero") for s, e in intervals]


def diag_gmm_log_likelihood(model, X, xp=np):
    """Per-frame log-likelihood of a diagonal-covariance mixture, on NumPy or CuPy arrays.

    Equivalent to ``model.score_samples`` but written with plain array operations so it can run
    on a GPU. Frames are processed in blocks to bound memory.

    Args:
        model (GaussianMixture): A fitted mixture with ``covariance_type="diag"``.
        X (array): Frames of shape (n, d), already on the device of ``xp``.
        xp (module): ``numpy`` or ``cupy``.

    Returns:
        array: Shape (n,).
    """
    weights = xp.asarray(model.weights_)
    means = xp.asarray(model.means_)
    variances = xp.asarray(model.covariances_)
    d = means.shape[1]
    constant = xp.log(weights) - 0.5 * (d * np.log(2.0 * np.pi) + xp.log(variances).sum(axis=1))
    inverse = 1.0 / variances
    out = []
    for start in range(0, X.shape[0], 200_000):
        block = X[start:start + 200_000]
        quad = ((block ** 2) @ inverse.T) - 2.0 * (block @ (means * inverse).T) + (means ** 2 * inverse).sum(axis=1)
        z = constant[None, :] - 0.5 * quad
        peak = z.max(axis=1, keepdims=True)
        out.append((peak[:, 0] + xp.log(xp.exp(z - peak).sum(axis=1))))
    return xp.concatenate(out) if out else xp.empty(0)


def mean_log_likelihoods(models: dict, blocks: list[np.ndarray], xp=np) -> np.ndarray:
    """Mean frame log-likelihood of every block under every model (equation 9 of the article).

    Args:
        models (dict): Fitted mixture per species code, in the order of the detector.
        blocks (list[np.ndarray]): One matrix of frames per candidate. A candidate with no frames
            (shorter than one analysis frame) gets NaN.
        xp (module): ``numpy`` or ``cupy``.

    Returns:
        np.ndarray: Shape (blocks, models), always a NumPy array.
    """
    from .device import to_numpy

    lengths = np.array([len(b) for b in blocks])
    scores = np.full((len(blocks), len(models)), np.nan)
    valid = lengths > 0
    if not valid.any():
        return scores
    frames = xp.asarray(np.vstack([b for b, ok in zip(blocks, valid) if ok]))
    offsets = np.concatenate(([0], np.cumsum(lengths[valid])[:-1]))
    for k, model in enumerate(models.values()):
        per_frame = to_numpy(diag_gmm_log_likelihood(model, frames, xp))
        scores[valid, k] = np.add.reduceat(per_frame, offsets) / lengths[valid]
    return scores


def detect(models: dict, thresholds: dict, blocks: list[np.ndarray], xp=np) -> dict:
    """Identify and verify every candidate with the frozen detector.

    Returns:
        dict: ``scores`` (blocks, models), ``valid`` (candidate had frames), ``hypothesis`` (index of the
            winning model, -1 when invalid), ``llr`` and ``accepted``.
    """
    scores = mean_log_likelihoods(models, blocks, xp)
    valid = ~np.isnan(scores).any(axis=1)
    out = {"scores": scores, "valid": valid, "hypothesis": np.full(len(blocks), -1),
           "llr": np.full(len(blocks), np.nan), "accepted": np.zeros(len(blocks), bool)}
    if valid.any():
        decision = D.decide(scores[valid], np.array([thresholds[c] for c in models]))
        out["hypothesis"][valid] = decision["hypothesis"]
        out["llr"][valid] = decision["llr"]
        out["accepted"][valid] = decision["accepted"]
    return out

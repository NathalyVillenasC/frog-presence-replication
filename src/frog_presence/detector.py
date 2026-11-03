"""Turning per-species mixtures into a detector: likelihood-ratio verification and thresholds.

The article identifies a call with the species model that gives the highest mean frame
log-likelihood (section 3.6) and then verifies that hypothesis with a log-likelihood ratio
against the median of the alternative models (section 3.7). A threshold per species decides
whether the verified hypothesis is accepted (section 3.8). The pieces here are what the detector
calibration and the field inference share, so a score computed while calibrating is the same
score computed in the field.

The deployed decision is: the winning model ``k`` is accepted when ``LLR_k >= threshold_k``.
"""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def llr_matrix(scores: np.ndarray) -> np.ndarray:
    """Log-likelihood ratio of every model against the median of the others.

    ``LLR_k = s_k - median_{j != k} s_j``, where ``s`` are mean frame log-likelihoods. With nine
    models the median of the eight alternatives is the mean of the two middle ones, taken in the
    log domain: the article does not say whether the median is taken before or after the logarithm.

    Args:
        scores (np.ndarray): Array of shape (calls, models).

    Returns:
        np.ndarray: Array of the same shape.
    """
    out = np.empty_like(scores, dtype=float)
    for k in range(scores.shape[1]):
        out[:, k] = scores[:, k] - np.median(np.delete(scores, k, axis=1), axis=1)
    return out


def decide(scores: np.ndarray, thresholds: np.ndarray) -> dict:
    """The deployed decision for every row of scores.

    Args:
        scores (np.ndarray): Array of shape (calls, models).
        thresholds (np.ndarray): One threshold per model, in the same order.

    Returns:
        dict: ``hypothesis`` (index of the winning model), ``llr`` (its ratio), ``threshold``
            (its threshold) and ``accepted`` (ratio at or above the threshold).
    """
    llr = llr_matrix(scores)
    hypothesis = scores.argmax(axis=1)
    rows = np.arange(len(scores))
    own_llr = llr[rows, hypothesis]
    own_threshold = np.asarray(thresholds, dtype=float)[hypothesis]
    return {"hypothesis": hypothesis, "llr": own_llr, "threshold": own_threshold,
            "accepted": own_llr >= own_threshold}


def confusion_counts(positive: np.ndarray, detected: np.ndarray) -> dict:
    """True and false positives and negatives, with recall, precision and false-positive rate."""
    tp = int((positive & detected).sum())
    fn = int((positive & ~detected).sum())
    fp = int((~positive & detected).sum())
    tn = int((~positive & ~detected).sum())
    return {
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "FPR": fp / (fp + tn) if fp + tn else float("nan"),
    }


def one_vs_all_threshold(llr_k: np.ndarray, positive: np.ndarray, max_fpr: float = 0.05) -> float:
    """The article's rule: on the ROC curve of one species against all others, keep the
    thresholds whose false-positive rate is at most ``max_fpr``, take the highest true-positive
    rate, and among ties the largest threshold.

    This ignores whether the species is the winning model, so calls that could never be
    detected as ``k`` still constrain the threshold.

    Returns:
        float: The threshold, or infinity when no finite threshold keeps the false-positive
            rate within ``max_fpr``, which means accepting nothing.
    """
    fpr, tpr, thresholds = roc_curve(positive, llr_k)
    keep = np.isfinite(thresholds) & (fpr <= max_fpr)
    if not keep.any():
        return float("inf")
    best_tpr = tpr[keep].max()
    return float(thresholds[keep & (tpr == best_tpr)].max())


def end_to_end_threshold(llr_k: np.ndarray, wins: np.ndarray, positive: np.ndarray,
                         max_fpr: float = 0.05) -> float:
    """The same rule applied to the decision that is actually deployed.

    A call is detected as ``k`` only if ``k`` wins and ``llr_k >= threshold``, so only the calls
    that ``k`` wins can move the threshold. Among thresholds with a false-positive rate at most
    ``max_fpr``, the one with the highest recall is taken, and among ties the largest.

    Args:
        llr_k (np.ndarray): The ratio of species ``k`` for every call.
        wins (np.ndarray): Whether ``k`` is the winning model of each call.
        positive (np.ndarray): Whether each call is truly species ``k``.
        max_fpr (float): False-positive rate that must not be exceeded.

    Returns:
        float: The threshold.
    """
    candidates = np.unique(llr_k[wins])
    n_positive, n_negative = positive.sum(), (~positive).sum()
    best_threshold, best_recall = float(candidates.max() + 1.0), 0.0

    for threshold in candidates:
        detected = wins & (llr_k >= threshold)
        recall = (positive & detected).sum() / n_positive
        fpr = (~positive & detected).sum() / n_negative
        if fpr <= max_fpr and (recall > best_recall or (recall == best_recall and threshold > best_threshold)):
            best_threshold, best_recall = float(threshold), recall
    return best_threshold


def calibration_split(calls: pd.DataFrame, codes, fraction: float, seed: int, n_attempts: int,
                      group_column: str = "wav") -> dict:
    """One fixed recording-level TRAIN / CALIBRATION split.

    Whole recordings go to one side, for every species at once. Random sets of calibration
    recordings of about ``fraction`` of the recordings are tried, and the valid one (every species
    present on both sides) whose per-species shares of duration and of calls are closest to
    ``fraction`` is kept, the first of equals winning.

    Args:
        calls (pd.DataFrame): Table with ``code``, ``seconds`` and ``group_column``.
        codes (Iterable[str]): Species to split.
        fraction (float): Wanted share of each species in calibration.
        seed (int): Seed of the search.
        n_attempts (int): Candidate sets to try.
        group_column (str): Column naming each call's recording.

    Returns:
        dict: ``train`` and ``calibration`` (sets of recording names), ``attempt`` and ``score``.

    Raises:
        RuntimeError: If no candidate is valid.
    """
    codes = list(codes)
    usable = calls[calls["code"].isin(codes)].copy()
    usable["_group"] = usable[group_column].astype(str).str.strip().str.lower()
    recordings = np.array(sorted(usable["_group"].unique()), dtype=object)
    index_of = {name: i for i, name in enumerate(recordings)}

    seconds = (usable.groupby(["_group", "code"])["seconds"].sum().unstack(fill_value=0.0)
               .reindex(index=recordings, columns=codes, fill_value=0.0).to_numpy())
    n_calls = (usable.groupby(["_group", "code"]).size().unstack(fill_value=0)
               .reindex(index=recordings, columns=codes, fill_value=0).to_numpy())
    total_seconds, total_calls = seconds.sum(axis=0), n_calls.sum(axis=0)

    target = max(1, int(round(fraction * len(recordings))))
    sizes = sorted({s for s in range(target - 2, target + 3) if 1 <= s < len(recordings)})
    rng = np.random.default_rng(seed)
    best, best_score = None, None

    for attempt in range(n_attempts):
        size = sizes[attempt % len(sizes)]
        chosen = rng.choice(recordings, size=size, replace=False)
        in_calibration = np.zeros(len(recordings), dtype=bool)
        in_calibration[[index_of[name] for name in chosen]] = True

        calibration_calls = n_calls[in_calibration].sum(axis=0)
        if np.any(calibration_calls == 0) or np.any(total_calls - calibration_calls == 0):
            continue

        score = 0.0
        share_seconds = seconds[in_calibration].sum(axis=0) / total_seconds
        share_calls = calibration_calls / total_calls
        for k in range(len(codes)):
            score += (share_seconds[k] - fraction) ** 2
            score += 0.25 * (share_calls[k] - fraction) ** 2
        score += 0.1 * (in_calibration.sum() / len(recordings) - fraction) ** 2

        if best is None or score < best_score:
            best, best_score = (attempt, in_calibration.copy()), score

    if best is None:
        raise RuntimeError("No valid recording-level TRAIN/CALIBRATION split was found.")

    attempt, in_calibration = best
    return {"train": set(recordings[~in_calibration]), "calibration": set(recordings[in_calibration]),
            "attempt": attempt, "score": best_score}


def parameters_equal(a, b, tolerance: float = 1e-10) -> bool:
    """Whether two fitted diagonal mixtures have the same weights, means and variances."""
    return (a.weights_.shape == b.weights_.shape
            and np.allclose(a.weights_, b.weights_, rtol=0, atol=tolerance)
            and np.allclose(a.means_, b.means_, rtol=0, atol=tolerance)
            and np.allclose(a.covariances_, b.covariances_, rtol=0, atol=tolerance))


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def save_detector(directory: Path, models: dict, thresholds: dict, manifest: dict,
                  rewrite_models: bool = True) -> dict:
    """Write one file per model, the thresholds and a manifest holding the model hashes.

    Args:
        directory (Path): Destination, created if missing.
        models (dict): Fitted mixture per species code.
        thresholds (dict): Threshold per species code.
        manifest (dict): Everything else worth recording; the file names and hashes are added.
        rewrite_models (bool): When True, existing ``*.joblib`` files are removed first, so a stale
            model can never be mixed into the detector, and every model is written again. When
            False the model files already in ``directory`` are left exactly as they are, and only
            their hashes are recorded: for a detector that was reproduced and must not change.

    Returns:
        dict: The manifest as written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    if rewrite_models:
        for old in directory.glob("*.joblib"):
            old.unlink()
        for code, model in models.items():
            joblib.dump(model, directory / f"{code}.joblib")

    files = {f"{code}.joblib": file_sha256(directory / f"{code}.joblib") for code in models}
    thresholds = {code: float(value) for code, value in thresholds.items()}
    (directory / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    written = {**manifest, "classes": list(models), "thresholds": thresholds,
               "gmm_files": list(files), "gmm_sha256": files}
    (directory / "final_detector_manifest.json").write_text(json.dumps(written, indent=2), encoding="utf-8")
    return written


def load_detector(directory: Path) -> tuple[dict, dict, dict]:
    """Read the models, thresholds and manifest written by ``save_detector``.

    Raises:
        ValueError: If a model file no longer matches the hash in the manifest.
    """
    manifest = json.loads((directory / "final_detector_manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["gmm_sha256"].items():
        if file_sha256(directory / name) != expected:
            raise ValueError(f"{name} does not match the hash recorded in the manifest")
    models = {code: joblib.load(directory / f"{code}.joblib") for code in manifest["classes"]}
    thresholds = json.loads((directory / "thresholds.json").read_text(encoding="utf-8"))
    return models, thresholds, manifest

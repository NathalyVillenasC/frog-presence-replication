"""Finding candidate calls in long recordings: the front end of sections 3.2 and 3.3.

The article cuts a recording into consecutive 30 second blocks, band-pass filters each block with
a FIR filter whose cut-offs depend on the species, takes the short-time energy over 10 ms frames,
smooths it with a moving average of 12 values, converts it to decibels, and marks a segment from
the first run of 3 consecutive values above a threshold to the first run of 3 below it. The filter
only serves to find the endpoints: the segment itself is later read from the original signal.

The article does not report the constant ``C`` of the threshold or the order of the filter, and
prints the threshold in a form that can be read in more than one way. What is here is a
reconstruction with every choice named.
"""

from functools import lru_cache

import numpy as np
from scipy.signal import firwin, fftconvolve

BLOCK_SECONDS = 30.0
STE_FRAME_SECONDS = 0.010
STE_MA_LENGTH = 12
ENDPOINT_RUN = 3
FIR_NUMTAPS = 513


@lru_cache(maxsize=64)
def bandpass_taps(sr: int, low_hz: float, high_hz: float, numtaps: int = FIR_NUMTAPS) -> np.ndarray:
    """Linear-phase band-pass FIR taps.

    Raises:
        ValueError: If the band does not lie strictly inside ``(0, sr / 2)``.
    """
    if not 0 < low_hz < high_hz < sr / 2.0:
        raise ValueError(f"Invalid band ({low_hz}, {high_hz}) for sr={sr}.")
    return firwin(numtaps=numtaps, cutoff=[low_hz, high_hz], pass_zero=False, fs=sr)


def bandpass(audio: np.ndarray, sr: int, low_hz: float, high_hz: float, numtaps: int = FIR_NUMTAPS) -> np.ndarray:
    """Band-pass a signal; the output is aligned with the input (``mode="same"``)."""
    return fftconvolve(np.asarray(audio, dtype=np.float64), bandpass_taps(sr, low_hz, high_hz, numtaps), mode="same")


def short_time_energy(audio: np.ndarray, sr: int, frame_seconds: float = STE_FRAME_SECONDS) -> tuple[np.ndarray, int]:
    """Energy of consecutive, non-overlapping frames; a trailing partial frame is dropped.

    Returns:
        tuple: The energy of each frame and the frame length in samples.
    """
    frame_samples = int(round(sr * frame_seconds))
    if frame_samples <= 0:
        raise ValueError("STE frame size must be positive.")
    n_frames = len(audio) // frame_samples
    if n_frames == 0:
        return np.empty(0), frame_samples
    frames = audio[:n_frames * frame_samples].reshape(n_frames, frame_samples)
    return np.sum(frames * frames, axis=1), frame_samples


def smooth(values: np.ndarray, length: int = STE_MA_LENGTH) -> np.ndarray:
    """Moving average of ``length`` values, the same length as the input."""
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values.copy()
    return np.convolve(values, np.ones(length) / length, mode="same")


def energy_to_db(energy: np.ndarray) -> np.ndarray:
    """10 log10 of the energy, floored so that silence does not give minus infinity."""
    return 10.0 * np.log10(np.maximum(np.asarray(energy, dtype=float), np.finfo(float).tiny))


def endpoint_threshold_db(ste_db: np.ndarray, C: float) -> float:
    """``mean + (max - mean) / C`` of the smoothed energy in decibels.

    The article prints the threshold as the maximum minus the mean, divided by ``C``, which is a
    positive range and not a level in decibels. It is read here as a fraction of the way from the mean
    to the maximum, so a larger ``C`` puts the threshold closer to the mean and gives a more
    sensitive detector.

    Raises:
        ValueError: If ``C`` is not positive.
    """
    if C <= 0:
        raise ValueError("C must be positive.")
    ste_db = np.asarray(ste_db, dtype=float)
    if len(ste_db) == 0:
        return float("nan")
    mean_db, max_db = float(np.mean(ste_db)), float(np.max(ste_db))
    return mean_db + (max_db - mean_db) / C


def find_endpoints(ste_db: np.ndarray, threshold_db: float, run_length: int = ENDPOINT_RUN) -> list[tuple[int, int]]:
    """Segments from a run of ``run_length`` values above the threshold to the next run below it.

    A segment still open at the end of the signal is closed there.

    Returns:
        list: ``(start_frame, end_frame)`` pairs, the end being exclusive.
    """
    ste_db = np.asarray(ste_db, dtype=float)
    above = ste_db > threshold_db
    below = ste_db < threshold_db
    segments, start, i, n = [], None, 0, len(ste_db)
    while i <= n - run_length:
        if start is None:
            if above[i:i + run_length].all():
                start, i = i, i + run_length
                continue
        elif below[i:i + run_length].all():
            if i > start:
                segments.append((start, i))
            start, i = None, i + run_length
            continue
        i += 1
    if start is not None:
        segments.append((start, n))
    return segments


def prepare_blocks(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> list[dict]:
    """Filter and measure every 30 second block once, so many values of ``C`` can be tried cheaply.

    Returns:
        list: One dict per block with ``start`` and ``end`` (samples), ``frame_samples`` and ``ste_db``.
    """
    block_samples = int(round(BLOCK_SECONDS * sr))
    blocks = []
    for start in range(0, len(audio), block_samples):
        end = min(len(audio), start + block_samples)
        energy, frame_samples = short_time_energy(bandpass(audio[start:end], sr, low_hz, high_hz), sr)
        blocks.append({"start": start, "end": end, "frame_samples": frame_samples,
                       "ste_db": energy_to_db(smooth(energy))})
    return blocks


def candidates_from_blocks(blocks: list[dict], C: float) -> np.ndarray:
    """Candidate intervals of a prepared recording for one value of ``C``.

    Returns:
        np.ndarray: Integer array of shape (n, 3): block index, start sample and end sample (exclusive).
    """
    rows = []
    for index, block in enumerate(blocks):
        ste_db = block["ste_db"]
        if len(ste_db) == 0:
            continue
        threshold = endpoint_threshold_db(ste_db, C)
        for start_frame, end_frame in find_endpoints(ste_db, threshold):
            start = block["start"] + start_frame * block["frame_samples"]
            end = min(block["end"], block["start"] + end_frame * block["frame_samples"])
            if end > start:
                rows.append((index, start, end))
    return np.array(rows, dtype=np.int64).reshape(-1, 3)


def merge_intervals(intervals) -> np.ndarray:
    """Sorted, disjoint union of intervals; touching intervals are joined.

    Returns:
        np.ndarray: Integer array of shape (n, 2).
    """
    items = sorted((int(s), int(e)) for s, e in intervals if e > s)
    merged = []
    for start, end in items:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return np.array(merged, dtype=np.int64).reshape(-1, 2)


def pad_intervals(intervals, pad_samples: int, n_samples: int) -> np.ndarray:
    """Widen every interval by ``pad_samples`` on each side, clip to the recording, and merge."""
    return merge_intervals((max(0, int(s) - pad_samples), min(n_samples, int(e) + pad_samples)) for s, e in intervals)


def overlap_with(start: int, end: int, intervals: np.ndarray) -> int:
    """Samples of ``[start, end)`` covered by disjoint ``intervals``."""
    if len(intervals) == 0:
        return 0
    return int(np.clip(np.minimum(end, intervals[:, 1]) - np.maximum(start, intervals[:, 0]), 0, None).sum())


def evaluate_candidates(call_starts, call_ends, candidates: np.ndarray, sr: int,
                        coverage_target: float = 0.5) -> tuple[dict, np.ndarray]:
    """How well candidate intervals cover the annotated calls of one species in one recording.

    A candidate that overlaps no annotated call is not a biological false positive: the annotations
    cover the target calls only, so the overlap share is a diagnostic and not a precision.

    Args:
        call_starts (Sequence[int]): Annotated call starts, in samples.
        call_ends (Sequence[int]): Annotated call ends.
        candidates (np.ndarray): Candidate intervals, shape (n, 2) or (n, 3) with the last two columns the interval.
        sr (int): Sample rate.
        coverage_target (float): Share of a call that must be covered for it to count as captured.

    Returns:
        tuple: A summary dict and the coverage of each call.
    """
    cand = merge_intervals(candidates[:, -2:]) if len(candidates) else np.empty((0, 2), np.int64)
    gt = merge_intervals(zip(call_starts, call_ends))
    coverage = np.array([overlap_with(s, e, cand) / max(1, e - s) for s, e in zip(call_starts, call_ends)])

    total_gt = int((gt[:, 1] - gt[:, 0]).sum()) if len(gt) else 0
    total_cand = int((cand[:, 1] - cand[:, 0]).sum()) if len(cand) else 0
    gt_covered = sum(overlap_with(s, e, cand) for s, e in gt)
    cand_overlap = sum(overlap_with(s, e, gt) for s, e in cand)
    n = len(coverage)
    summary = {
        "n_gt_calls": n,
        "n_candidates": int(len(candidates)),
        "call_recall_any": float((coverage > 0).mean()) if n else float("nan"),
        "call_recall_50": float((coverage >= coverage_target).mean()) if n else float("nan"),
        "median_call_coverage": float(np.median(coverage)) if n else float("nan"),
        "annotation_coverage": gt_covered / total_gt if total_gt else float("nan"),
        "candidate_overlap_share": cand_overlap / total_cand if total_cand else float("nan"),
        "candidate_seconds": total_cand / sr,
        "annotated_seconds_union": total_gt / sr,
        "candidate_burden_ratio": total_cand / total_gt if total_gt else float("nan"),
    }
    return summary, coverage

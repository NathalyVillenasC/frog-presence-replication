"""Applying the frozen segmentation to a long field recording.

A recording is segmented once per species band, each with that species' constant ``C`` and padding,
and the candidates of one band are merged when they overlap. Candidates of different bands are kept
apart, since the same acoustic event can be proposed by several bands; the detector then classifies
every candidate whatever band proposed it.
"""

import numpy as np
import pandas as pd

from . import features as F
from . import segmentation as S


def segment_recording(audio: np.ndarray, sr: int, config: dict) -> pd.DataFrame:
    """Candidate intervals of a recording, one band per species.

    Args:
        audio (np.ndarray): Mono signal.
        sr (int): Sample rate in hertz.
        config (dict): The frozen segmentation configuration: ``paper_frontend`` with the band of every
            species, and ``selection`` with ``C_by_code`` and ``padding_ms_by_code``.

    Returns:
        pd.DataFrame: One row per candidate with ``band`` (the species whose band proposed it), ``start``
            and ``end`` in samples (the end exclusive).
    """
    bands = config["paper_frontend"]["species_frequency_bands_hz"]
    c_by_code = config["selection"]["C_by_code"]
    padding_by_code = config["selection"]["padding_ms_by_code"]

    rows = []
    for code, (low_hz, high_hz) in bands.items():
        blocks = S.prepare_blocks(audio, sr, low_hz, high_hz)
        candidates = S.candidates_from_blocks(blocks, c_by_code[code])
        padded = S.pad_intervals(candidates[:, 1:], int(round(sr * padding_by_code[code] / 1000.0)), len(audio))
        rows.extend({"band": code, "start": int(s), "end": int(e)} for s, e in padded)
    return pd.DataFrame(rows, columns=["band", "start", "end"])


def candidate_frames(audio: np.ndarray, sr: int, intervals: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """PLP-RASTA frames of every candidate, stacked, with the number of frames of each.

    The frames are those of the original, unfiltered signal, with the RASTA filter started from silence
    on every candidate, as the frozen detector was fitted.

    Returns:
        tuple: An array of shape (frames, 20) and the frame count of each candidate (zero for one
            shorter than an analysis frame).
    """
    blocks = [F.plp_rasta(audio[int(s):int(e)], sr, rasta="zero") for s, e in zip(intervals.start, intervals.end)]
    lengths = np.array([len(b) for b in blocks], dtype=np.int64)
    stacked = np.vstack([b for b in blocks if len(b)]) if lengths.any() else np.empty((0, 20))
    return stacked, lengths

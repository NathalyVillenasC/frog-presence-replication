"""Tests for reading audio headers.

The audio itself is not in the repository, so every test writes the file it needs into a
temporary directory. That keeps the suite runnable by anyone who clones the project,
including someone who has no access to the two private collections.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from frog_presence.audio import PROBE_FIELDS, probe_audio_header


@pytest.fixture
def one_second(tmp_path: Path):
    """Write a one-second mono WAV and return a factory for other encodings."""

    def write(subtype: str = "PCM_16", rate: int = 48000, channels: int = 1) -> Path:
        samples = np.zeros((rate, channels), dtype="float32").squeeze()
        path = tmp_path / f"{subtype.lower()}.wav"
        sf.write(path, samples, rate, subtype=subtype)
        return path

    return write


def test_reads_a_lossless_header(one_second):
    """Expected case: a 48 kHz 16-bit WAV is described in full, with no error."""
    probe = probe_audio_header(one_second())

    assert probe["header_error"] is None
    assert probe["format_name"] == "WAV"
    assert probe["subtype"] == "PCM_16"
    assert probe["sample_rate"] == 48000
    assert probe["channels"] == 1
    assert probe["bit_depth"] == 16
    assert probe["duration_s"] == pytest.approx(1.0, abs=0.01)


def test_lossy_encodings_report_a_bitrate_instead_of_a_bit_depth(one_second):
    """Edge case: part of the corpus is GSM 6.10, where a bit depth is meaningless.

    The distinction matters beyond tidiness: the article's test material is the lossless
    part of the collection, and this field is how the two are told apart.
    """
    probe = probe_audio_header(one_second(subtype="GSM610", rate=8000))

    assert probe["subtype"] == "GSM610"
    assert probe["bit_depth"] is None
    assert probe["bitrate_kbps"] > 0


def test_an_unreadable_file_is_reported_rather_than_raised(tmp_path: Path):
    """Failure case: a bad file records its error and leaves the other fields empty.

    An inventory walks hundreds of files in one pass, so a single unreadable header has
    to come back as a row, not as an exception that loses the whole run.
    """
    broken = tmp_path / "not-really-audio.wav" # Build and store in `broken` the temporary path where the fake file will be created.
    broken.write_bytes(b"this is not a RIFF header")

    probe = probe_audio_header(broken)

    assert probe["header_error"] is not None
    assert set(probe) == set(PROBE_FIELDS)
    assert all(probe[field] is None for field in PROBE_FIELDS if field != "header_error") # all() avoid write an assert for each field :)


def test_a_missing_file_is_reported_the_same_way(tmp_path: Path):
    """Failure case: a path that does not exist behaves like an unreadable header."""
    probe = probe_audio_header(tmp_path / "absent.wav")

    assert probe["header_error"] is not None
    assert probe["duration_s"] is None

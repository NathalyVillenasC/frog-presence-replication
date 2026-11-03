"""Tests for reading audio headers.

The audio itself is not in the repository, so every test writes the file it needs into a
temporary directory. That keeps the suite runnable by anyone who clones the project,
including someone who has no access to the two private collections.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from frog_presence.audio import PROBE_FIELDS, index_wavs, probe_audio_header, read_call_audio


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


def _write_wav(path: Path, rate: int = 8000, seconds: float = 1.0, channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(-0.5, 0.5, int(rate * seconds))
    sf.write(path, np.tile(samples[:, None], (1, channels)) if channels > 1 else samples, rate, subtype="PCM_16")
    return path


def test_index_finds_wavs_in_subfolders_ignoring_case_and_finder_files(tmp_path):
    _write_wav(tmp_path / "a" / "One.WAV")
    _write_wav(tmp_path / "b" / "two.wav")
    (tmp_path / "a" / "._One.WAV").write_bytes(b"metadata")
    (tmp_path / "a" / "notes.txt").write_text("not audio")
    assert sorted(index_wavs(tmp_path)) == ["one.wav", "two.wav"]


def test_index_refuses_a_name_two_files_share(tmp_path):
    _write_wav(tmp_path / "a" / "same.wav")
    _write_wav(tmp_path / "b" / "SAME.WAV")
    with pytest.raises(ValueError):
        index_wavs(tmp_path)


def test_a_call_is_read_between_its_annotated_samples(tmp_path):
    path = _write_wav(tmp_path / "call.wav")
    audio, rate = read_call_audio(path, 100, 300, 8000)
    assert rate == 8000 and len(audio) == 200 and audio.dtype == np.float64


def test_stereo_is_averaged_to_mono(tmp_path):
    path = _write_wav(tmp_path / "stereo.wav", channels=2)
    audio, _ = read_call_audio(path, 0, 50, 8000)
    assert audio.ndim == 1 and len(audio) == 50


def test_a_call_that_disagrees_with_its_annotation_is_refused(tmp_path):
    path = _write_wav(tmp_path / "short.wav", seconds=0.1)
    with pytest.raises(RuntimeError):
        read_call_audio(path, 0, 5000, 8000)   # the file holds only 800 samples
    with pytest.raises(RuntimeError):
        read_call_audio(path, 0, 100, 44100)   # annotated at a different rate

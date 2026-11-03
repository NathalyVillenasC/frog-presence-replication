"""Reading audio files: technical metadata, and the samples of one annotated call.

The corpus mixes lossless PCM with GSM 6.10 inside WAV and with MP3, and it is large enough
that decoding samples to learn a file's duration is not an option. ``probe_audio_header`` reads
headers only. Samples are decoded only by ``read_call_audio``, and only for the span of one
annotated call.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

# libsndfile already reports the container in `format` ("WAV", "MP3") and the encoding
# in `subtype`, so neither needs a lookup table. Only the bit depth does: it is defined
# for PCM and float, and meaningless for the lossy codecs (GSM610, MPEG_LAYER_III),
# which are therefore absent from this map and resolve to None.
BIT_DEPTH_BY_SUBTYPE = {
    "PCM_S8": 8, "PCM_U8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32,
    "FLOAT": 32, "DOUBLE": 64,
}

PROBE_FIELDS = ("format_name", "subtype", "sample_rate", "channels", "bit_depth",
                "bitrate_kbps", "duration_s", "frames", "header_error")


def probe_audio_header(path: Path) -> dict:
    """Read an audio file's technical metadata without decoding its samples.

    Args:
        path (Path): Audio file to inspect.

    Returns:
        dict: The keys listed in ``PROBE_FIELDS``. If the header cannot be read, every
            value is None except ``header_error``, so a single bad file cannot abort an
            inventory of thousands.
    """
    result = {field: None for field in PROBE_FIELDS}

    try:
        info = sf.info(str(path))
        bit_depth = BIT_DEPTH_BY_SUBTYPE.get(info.subtype)
        result.update(
            format_name=info.format,
            subtype=info.subtype,
            sample_rate=info.samplerate,
            channels=info.channels,
            bit_depth=bit_depth,
            duration_s=info.duration,
            frames=info.frames,
        )

        # For compressed formats the bitrate is the informative number, not the bit depth.
        if bit_depth is None and info.duration:
            result["bitrate_kbps"] = round(path.stat().st_size * 8 / info.duration / 1000)

    except Exception as exc:
        result["header_error"] = f"{type(exc).__name__}: {exc}"

    return result


def index_wavs(root: Path) -> dict[str, Path]:
    """Map the lower-case name of every WAV below ``root`` to its path.

    The annotations name a recording by its bare file name, so a name that two different files
    share would make the lookup silently pick one of them. That is refused instead.

    Args:
        root (Path): Folder to search recursively.

    Returns:
        dict[str, Path]: Lower-case file name to path. Finder metadata files (``._*``) are skipped.

    Raises:
        ValueError: If two files below ``root`` have the same name, ignoring case.
    """
    found: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() == ".wav" and not path.name.startswith("._"):
            found.setdefault(path.name.lower(), []).append(path)

    duplicated = {name: paths for name, paths in found.items() if len(paths) > 1}
    if duplicated:
        raise ValueError(f"{len(duplicated)} file names occur more than once, e.g. {sorted(duplicated)[:3]}")
    return {name: paths[0] for name, paths in found.items()}


def read_call_audio(path: Path, start: int, end: int, rate: int) -> tuple[np.ndarray, int]:
    """Read the samples of one annotated call as mono float64.

    Args:
        path (Path): The recording.
        start (int): First sample of the call.
        end (int): Sample after the last one.
        rate (int): Sample rate the annotation was written for.

    Returns:
        tuple[np.ndarray, int]: The samples and the sample rate of the file.

    Raises:
        RuntimeError: If the file's rate or the number of samples read disagrees with the annotation.
    """
    audio, sr = sf.read(path, start=start, stop=end, dtype="float64")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if int(sr) != int(rate) or len(audio) != end - start:
        raise RuntimeError(f"{path.name}: expected {end - start} samples at {rate} Hz, read {len(audio)} at {sr} Hz.")
    return audio, int(sr)

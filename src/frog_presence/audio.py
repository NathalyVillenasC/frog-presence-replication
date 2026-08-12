"""Reading technical metadata out of audio files.

The corpus mixes lossless PCM with GSM 6.10 inside WAV and with MP3, and it is large enough
that decoding samples to learn a file's duration is not an option. Everything here reads
headers only.
"""

from pathlib import Path

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

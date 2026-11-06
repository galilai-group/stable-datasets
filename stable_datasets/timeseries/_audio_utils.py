import io
from dataclasses import dataclass

import numpy as np
from scipy.io.wavfile import read as wav_read

from stable_datasets.schema import BuilderConfig


@dataclass
class AudioConfig(BuilderConfig):
    """Builder config for audio datasets offered at several sample rates.

    ``sample_rate=None`` keeps the source rate. The config name is part of the cache
    fingerprint, so each rate gets its own processed cache.
    """

    sample_rate: int | None = None


def sample_rate_configs(native_rate: int) -> list[AudioConfig]:
    """Standard configs: the native rate plus 22.05 kHz and 16 kHz resampled variants."""
    return [
        AudioConfig(name="native", description=f"Source sample rate ({native_rate} Hz), no resampling."),
        AudioConfig(name="22khz", sample_rate=22050, description="Resampled to 22,050 Hz at build time."),
        AudioConfig(name="16khz", sample_rate=16000, description="Resampled to 16,000 Hz at build time."),
    ]


def wav_bytes_to_series(data: bytes) -> np.ndarray:
    """Decode WAV bytes into a float32 [T, C] array."""
    _, wav = wav_read(io.BytesIO(data))
    series = np.asarray(wav, dtype="float32")
    if series.ndim == 1:
        series = series[:, None]
    return series


def wav_bytes_to_mono(data: bytes) -> tuple[np.ndarray, int]:
    """Decode a mono WAV into a float32 ``[T]`` array (raw PCM scale) and its sample rate.

    Raises:
        ValueError: if the WAV has more than one channel.
    """
    sample_rate, wav = wav_read(io.BytesIO(data))
    if wav.ndim != 1:
        raise ValueError(f"Expected a mono WAV, got {wav.shape[1]} channels.")
    return np.asarray(wav, dtype="float32"), int(sample_rate)


def resample(series: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample along the first axis with a polyphase filter (``scipy.signal.resample_poly``).

    Returns the input unchanged (as float32) when the rates already match.
    """
    series = np.asarray(series, dtype="float32")
    if orig_sr == target_sr:
        return series
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(orig_sr, target_sr)
    return resample_poly(series, target_sr // g, orig_sr // g, axis=0).astype("float32")


def audiosegment_bytes_to_series(data: bytes, format: str) -> np.ndarray:
    """Decode an audio file via pydub into a float32 [T, C] array."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(io.BytesIO(data), format=format)
    samples = np.asarray(audio.get_array_of_samples(), dtype="float32")
    if audio.channels > 1:
        samples = samples.reshape(-1, audio.channels)
    else:
        samples = samples[:, None]
    return samples


def soundfile_bytes_to_series(data: bytes) -> np.ndarray:
    """Decode audio bytes via soundfile into a float32 [T, C] array."""
    import soundfile as sf

    series, _ = sf.read(io.BytesIO(data), dtype="float32")
    series = np.asarray(series, dtype="float32")
    if series.ndim == 1:
        series = series[:, None]
    return series

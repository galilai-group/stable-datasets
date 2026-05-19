import io

import numpy as np
from scipy.io.wavfile import read as wav_read


def wav_bytes_to_series(data: bytes) -> np.ndarray:
    """Decode WAV bytes into a float32 [T, C] array."""
    _, wav = wav_read(io.BytesIO(data))
    series = np.asarray(wav, dtype="float32")
    if series.ndim == 1:
        series = series[:, None]
    return series


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

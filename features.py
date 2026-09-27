"""
features.py — audio loading & log-mel spectrogram extraction shared by
training, reference-bank building, and inference.
"""
import numpy as np
import librosa
import config as cfg


def load_audio(path_or_array, sr: int = cfg.SAMPLE_RATE) -> np.ndarray:
    """Load an audio file to mono float32 at the target sample rate,
    or pass through if already a numpy array."""
    if isinstance(path_or_array, np.ndarray):
        return path_or_array.astype(np.float32)
    y, _ = librosa.load(str(path_or_array), sr=sr, mono=True)
    return y.astype(np.float32)


def logmel(y: np.ndarray, sr: int = cfg.SAMPLE_RATE) -> np.ndarray:
    """Returns a (T, n_mels) log-mel spectrogram."""
    if len(y) == 0:
        return np.zeros((1, cfg.N_MELS), dtype=np.float32)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
        n_mels=cfg.N_MELS, power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    # normalize per-utterance to zero mean / unit variance -> robust to loudness
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    return log_mel.T.astype(np.float32)  # (T, n_mels)


def file_to_logmel(path) -> np.ndarray:
    y = load_audio(path)
    return logmel(y)

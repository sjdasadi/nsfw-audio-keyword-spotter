"""
segmentation.py

Since we deliberately avoid ASR, we don't have real word boundaries.
Strategy:
    1. Use WebRTC VAD to find speech regions in the uploaded audio
       (skips silence / non-speech, cuts candidate count drastically).
    2. Inside each speech region, slide windows of several candidate
       durations (WINDOW_SIZES_MS) with overlap (WINDOW_HOP_MS) to
       generate word-length audio "candidates".
    3. Each candidate is later embedded and matched against the AWE
       reference bank (see matcher.py).

This is a recall-oriented, brute-force approach: many overlapping
candidates are generated on purpose, then pruned via similarity
threshold + non-max suppression in matcher.py.
"""
from dataclasses import dataclass
import numpy as np
import webrtcvad
import config as cfg


@dataclass
class Candidate:
    start_sample: int
    end_sample: int

    @property
    def start_sec(self) -> float:
        return self.start_sample / cfg.SAMPLE_RATE

    @property
    def end_sec(self) -> float:
        return self.end_sample / cfg.SAMPLE_RATE


def _float_to_pcm16(y: np.ndarray) -> bytes:
    y = np.clip(y, -1.0, 1.0)
    return (y * 32767).astype(np.int16).tobytes()


def detect_speech_regions(y: np.ndarray, sr: int = cfg.SAMPLE_RATE) -> list[tuple[int, int]]:
    """Returns list of (start_sample, end_sample) speech regions via WebRTC VAD."""
    assert sr == 16000, "webrtcvad requires 16kHz/8kHz/32kHz/48kHz; pipeline uses 16kHz."
    vad = webrtcvad.Vad(cfg.VAD_AGGRESSIVENESS)

    frame_len = int(sr * cfg.VAD_FRAME_MS / 1000)
    pcm = _float_to_pcm16(y)
    n_frames = len(y) // frame_len

    flags = []
    for i in range(n_frames):
        frame_bytes = pcm[i * frame_len * 2: (i + 1) * frame_len * 2]  # 2 bytes/sample
        if len(frame_bytes) < frame_len * 2:
            break
        is_speech = vad.is_speech(frame_bytes, sr)
        flags.append(is_speech)

    # merge consecutive speech frames into regions, with small padding
    regions = []
    start = None
    pad_frames = 2
    for i, flag in enumerate(flags + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            s = max(0, start - pad_frames) * frame_len
            e = min(n_frames, i + pad_frames) * frame_len
            regions.append((s, e))
            start = None

    # merge overlapping/adjacent regions
    merged = []
    for s, e in regions:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    if not merged:
        # fallback: treat whole clip as one region (very short / quiet audio)
        merged = [(0, len(y))]
    return merged


def generate_candidates(y: np.ndarray, sr: int = cfg.SAMPLE_RATE) -> list[Candidate]:
    regions = detect_speech_regions(y, sr)
    candidates = []
    for region_start, region_end in regions:
        region_len = region_end - region_start
        for win_ms in cfg.WINDOW_SIZES_MS:
            win_len = int(sr * win_ms / 1000)
            hop_len = int(sr * cfg.WINDOW_HOP_MS / 1000)
            if win_len > region_len:
                # region shorter than window -> use whole region as one candidate
                candidates.append(Candidate(region_start, region_end))
                continue
            pos = region_start
            while pos + win_len <= region_end:
                candidates.append(Candidate(pos, pos + win_len))
                pos += hop_len
    return candidates
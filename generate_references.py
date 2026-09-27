"""
generate_references.py

Synthesizes 3-5 audio samples per word (NSFW + normal) using XTTS-v2,
one sample per reference speaker voice found in SPEAKER_REFS_DIR.

Requirements:
    - TTS>=0.22.0 already installed.
    - SPEAKER_REFS_DIR must contain 3-5 short (6-10s) clean speech .wav files,
      one per desired voice, e.g.:
          speaker_refs/speaker_male_1.wav
          speaker_refs/speaker_female_1.wav
          speaker_refs/speaker_male_2.wav
          speaker_refs/speaker_female_2.wav
          speaker_refs/speaker_neutral_1.wav
      These can be your own short recordings or public-domain samples
      (e.g. LibriSpeech dev-clean). They only define the *voice*; the
      *text* synthesized is your target word.

Output layout:
    references/nsfw/<word>/<speaker_name>.wav
    references/normal/<word>/<speaker_name>.wav

Usage:
    python generate_references.py
    python generate_references.py --only nsfw      # regenerate only nsfw words
    python generate_references.py --only normal
"""
import argparse
import os
import sys
from pathlib import Path

# Force torchaudio to use its legacy ffmpeg/soundfile backend instead of the
# separate `torchcodec` package, which is not installed and would otherwise
# make TTS's internal `torchaudio.load(speaker_wav)` call fail with
# "TorchCodec is required for load_with_torchcodec". Must be set before
# torch/torchaudio are imported anywhere in the process.
os.environ.setdefault("TORCHAUDIO_USE_TORCHCODEC", "0")

import torch

import config as cfg

# ---------------------------------------------------------------------------
# PyTorch >= 2.6 changed torch.load's default `weights_only` from False to
# True, which blocks Coqui TTS's XTTS-v2 checkpoint from loading (it pickles
# a custom `XttsConfig` object, not just raw tensors). The checkpoint comes
# from Coqui's official, trusted model hub, so we explicitly restore the
# permissive behavior for this trusted load only.
_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def load_wordlist(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Word list not found: {path}")
    words = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        words.append(line)
    return words


def get_speaker_refs() -> list[Path]:
    refs = sorted(cfg.SPEAKER_REFS_DIR.glob("*.wav"))
    if len(refs) < cfg.MIN_SPEAKER_REFS:
        raise RuntimeError(
            f"Found only {len(refs)} speaker reference wav(s) in "
            f"{cfg.SPEAKER_REFS_DIR}. Please add at least "
            f"{cfg.MIN_SPEAKER_REFS} (recommended {cfg.MAX_SPEAKER_REFS}) "
            f"short (6-10s) clean speech clips, one per desired voice."
        )
    return refs[: cfg.MAX_SPEAKER_REFS]


def synthesize_word(tts, word: str, speaker_wav: Path, out_path: Path):
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(
        text=word,
        speaker_wav=str(speaker_wav),
        language=cfg.XTTS_LANGUAGE,
        file_path=str(out_path),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["nsfw", "normal"], default=None)
    args = parser.parse_args()

    from TTS.api import TTS  # imported lazily so config-only usage doesn't need TTS installed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading XTTS-v2 on {device} ...")
    tts = TTS(cfg.XTTS_MODEL_NAME).to(device)

    speaker_refs = get_speaker_refs()
    print(f"Using {len(speaker_refs)} speaker voices: {[s.name for s in speaker_refs]}")

    targets = []
    if args.only in (None, "nsfw"):
        targets.append(("nsfw", cfg.NSFW_WORDLIST, cfg.REFERENCES_NSFW_DIR))
    if args.only in (None, "normal"):
        targets.append(("normal", cfg.NORMAL_WORDLIST, cfg.REFERENCES_NORMAL_DIR))

    for label, wordlist_path, out_dir in targets:
        words = load_wordlist(wordlist_path)
        if not words:
            print(f"[WARN] {wordlist_path} is empty, skipping {label}.")
            continue
        print(f"\n=== Synthesizing {len(words)} '{label}' words "
              f"x {len(speaker_refs)} voices ===")
        for word in words:
            for spk in speaker_refs:
                out_path = out_dir / word / f"{spk.stem}.wav"
                try:
                    synthesize_word(tts, word, spk, out_path)
                except Exception as e:
                    print(f"[ERROR] word='{word}' speaker='{spk.name}': {e}", file=sys.stderr)
        print(f"Done: {label} -> {out_dir}")


if __name__ == "__main__":
    main()

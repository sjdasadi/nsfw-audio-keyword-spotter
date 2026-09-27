"""
train_awe.py

Trains the AWEEncoder with triplet loss using the synthetic XTTS-v2
reference audio (both nsfw + normal words act as distinct "classes" —
more classes = a more discriminative embedding space).

Triplet sampling:
    anchor   = word W, speaker i
    positive = word W, speaker j != i           (same word, different voice)
    negative = word V != W, any speaker          (different word)

Light on-the-fly augmentation (gain / mild noise / time-stretch) is applied
so the encoder isn't overly tied to the exact XTTS acoustics, which helps
it generalize better to real uploaded speech later.

Usage:
    python train_awe.py
"""
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config as cfg
from features import load_audio, logmel
from awe_model import AWEEncoder, pad_collate

try:
    from audiomentations import Compose, AddGaussianNoise, Gain, TimeStretch
    AUGMENT = Compose([
        Gain(min_gain_db=-6, max_gain_db=6, p=0.5),
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.008, p=0.4),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
    ])
except ImportError:
    AUGMENT = None
    print("[WARN] audiomentations not installed -> training without augmentation.")


def collect_word_audio() -> dict[str, list[Path]]:
    """word -> list of wav file paths, pooling both nsfw + normal directories."""
    word_to_files = defaultdict(list)
    for base_dir in (cfg.REFERENCES_NSFW_DIR, cfg.REFERENCES_NORMAL_DIR):
        if not base_dir.exists():
            continue
        for word_dir in base_dir.iterdir():
            if not word_dir.is_dir():
                continue
            wavs = sorted(word_dir.glob("*.wav"))
            if wavs:
                word_to_files[word_dir.name].extend(wavs)
    return word_to_files


class TripletDataset(Dataset):
    """Generates a fixed number of random triplets per epoch."""

    def __init__(self, word_to_files: dict[str, list[Path]], n_triplets: int):
        self.word_to_files = {w: f for w, f in word_to_files.items() if len(f) >= 2}
        self.words = list(self.word_to_files.keys())
        if len(self.words) < 2:
            raise RuntimeError(
                "Need at least 2 words with >=2 audio samples each to train. "
                "Did you run generate_references.py?"
            )
        self.n_triplets = n_triplets

    def __len__(self):
        return self.n_triplets

    def _load_feat(self, path: Path) -> torch.Tensor:
        y = load_audio(path)
        if AUGMENT is not None and random.random() < 0.7:
            try:
                y = AUGMENT(samples=y, sample_rate=cfg.SAMPLE_RATE)
            except Exception:
                pass
        feat = logmel(y)
        return torch.from_numpy(feat)

    def __getitem__(self, idx):
        anchor_word = random.choice(self.words)
        neg_word = random.choice([w for w in self.words if w != anchor_word])

        pos_files = random.sample(self.word_to_files[anchor_word], 2)
        neg_file = random.choice(self.word_to_files[neg_word])

        return (
            self._load_feat(pos_files[0]),
            self._load_feat(pos_files[1]),
            self._load_feat(neg_file),
        )


def triplet_collate(batch):
    anchors, positives, negatives = zip(*batch)
    return pad_collate(list(anchors)), pad_collate(list(positives)), pad_collate(list(negatives))


def main():
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    word_to_files = collect_word_audio()
    print(f"Found {len(word_to_files)} words with reference audio.")

    dataset = TripletDataset(word_to_files, n_triplets=cfg.TRIPLETS_PER_EPOCH)
    loader = DataLoader(
        dataset, batch_size=cfg.TRAIN_BATCH_SIZE, shuffle=True,
        collate_fn=triplet_collate, num_workers=0,
    )

    model = AWEEncoder().to(device)
    if cfg.AWE_MODEL_PATH.exists():
        print(f"Found existing checkpoint at {cfg.AWE_MODEL_PATH} -> resuming "
              f"training from it instead of starting over.")
        model.load_state_dict(torch.load(cfg.AWE_MODEL_PATH, map_location=device))
    criterion = nn.TripletMarginLoss(margin=cfg.TRIPLET_MARGIN, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)

    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg.TRAIN_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{cfg.TRAIN_EPOCHS}")
        for (anchor, a_len), (positive, p_len), (negative, n_len) in pbar:
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            a_len, p_len, n_len = a_len.to(device), p_len.to(device), n_len.to(device)

            emb_a = model(anchor, a_len)
            emb_p = model(positive, p_len)
            emb_n = model(negative, n_len)

            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}: avg triplet loss = {avg_loss:.4f}")

        torch.save(model.state_dict(), cfg.AWE_MODEL_PATH)

    print(f"\nSaved trained AWE encoder to {cfg.AWE_MODEL_PATH}")


if __name__ == "__main__":
    main()

"""
build_reference_bank.py

Encodes every NSFW reference audio clip with the trained AWE encoder and
saves a {word: [embeddings...]} bank to disk. Also computes genuine vs.
impostor similarity statistics (using the normal-word set as impostors)
to suggest a good similarity threshold.

Usage:
    python build_reference_bank.py
"""
import json
import pickle
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

import config as cfg
from features import file_to_logmel
from awe_model import AWEEncoder


def load_model() -> AWEEncoder:
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    model = AWEEncoder().to(device)
    if not cfg.AWE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained model at {cfg.AWE_MODEL_PATH}. Run train_awe.py first."
        )
    model.load_state_dict(torch.load(cfg.AWE_MODEL_PATH, map_location=device))
    model.eval()
    return model, device


@torch.no_grad()
def embed_file(model, device, path: Path) -> np.ndarray:
    feat = file_to_logmel(path)                       # (T, n_mels)
    length = torch.tensor([feat.shape[0]], dtype=torch.long, device=device)
    x = torch.from_numpy(feat).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,T,n_mels)
    emb = model(x, length).squeeze(0).cpu().numpy()
    return emb


def encode_dir(model, device, base_dir: Path) -> dict[str, list[np.ndarray]]:
    bank = defaultdict(list)
    if not base_dir.exists():
        return bank
    for word_dir in sorted(base_dir.iterdir()):
        if not word_dir.is_dir():
            continue
        for wav in sorted(word_dir.glob("*.wav")):
            emb = embed_file(model, device, wav)
            bank[word_dir.name].append(emb)
    return bank


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def genuine_similarities(bank: dict[str, list[np.ndarray]]) -> list[float]:
    sims = []
    for word, embs in bank.items():
        for a, b in combinations(embs, 2):
            sims.append(cosine(a, b))
    return sims


def impostor_similarities(bank_a: dict[str, list[np.ndarray]],
                           bank_b: dict[str, list[np.ndarray]],
                           max_pairs: int = 5000) -> list[float]:
    sims = []
    words_a = list(bank_a.items())
    words_b = list(bank_b.items())
    rng = np.random.default_rng(0)
    for _ in range(max_pairs):
        wa, embs_a = words_a[rng.integers(len(words_a))]
        wb, embs_b = words_b[rng.integers(len(words_b))]
        if wa == wb:
            continue
        a = embs_a[rng.integers(len(embs_a))]
        b = embs_b[rng.integers(len(embs_b))]
        sims.append(cosine(a, b))
    return sims


def suggest_threshold(genuine: list[float], impostor: list[float]) -> float:
    """Pick the threshold that best separates the two distributions
    (midpoint biased toward genuine mean to keep recall reasonably high;
    tune manually afterward using the printed report)."""
    g_mean, g_std = np.mean(genuine), np.std(genuine)
    i_mean, i_std = np.mean(impostor), np.std(impostor)
    # simple equal-error-rate-ish heuristic
    threshold = (g_mean * i_std + i_mean * g_std) / (g_std + i_std + 1e-8)
    return float(np.clip(threshold, 0.3, 0.95))


def main():
    model, device = load_model()

    print("Encoding NSFW reference words...")
    nsfw_bank = encode_dir(model, device, cfg.REFERENCES_NSFW_DIR)
    print(f"  {len(nsfw_bank)} nsfw words encoded.")

    print("Encoding normal (negative) reference words...")
    normal_bank = encode_dir(model, device, cfg.REFERENCES_NORMAL_DIR)
    print(f"  {len(normal_bank)} normal words encoded.")

    if not nsfw_bank:
        raise RuntimeError(
            "No NSFW reference audio found. Fill data/nsfw_words.txt and "
            "run generate_references.py first."
        )

    cfg.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.REFERENCE_BANK_PATH, "wb") as f:
        pickle.dump(dict(nsfw_bank), f)
    print(f"Saved NSFW reference bank -> {cfg.REFERENCE_BANK_PATH}")

    # --- threshold calibration ---
    genuine = genuine_similarities(nsfw_bank) + genuine_similarities(normal_bank)
    impostor = impostor_similarities(nsfw_bank, normal_bank) + \
        impostor_similarities(nsfw_bank, nsfw_bank)

    if genuine and impostor:
        suggested = suggest_threshold(genuine, impostor)
        report = {
            "genuine_mean": float(np.mean(genuine)),
            "genuine_std": float(np.std(genuine)),
            "impostor_mean": float(np.mean(impostor)),
            "impostor_std": float(np.std(impostor)),
            "suggested_threshold": suggested,
        }
        with open(cfg.THRESHOLD_REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2)
        print("\n=== Threshold calibration report ===")
        print(json.dumps(report, indent=2))
        print(f"\n-> Set DEFAULT_SIM_THRESHOLD in config.py to ~{suggested:.2f}, "
              f"or adjust live in the Streamlit sidebar.")
    else:
        print("[WARN] Not enough data to calibrate a threshold; "
              "add more normal words / speakers.")


if __name__ == "__main__":
    main()

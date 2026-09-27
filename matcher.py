"""
matcher.py

Given candidate audio segments (from segmentation.py) and the trained AWE
encoder + NSFW reference bank (from build_reference_bank.py), finds the
best-matching NSFW word (if any) for each candidate, then applies
non-max suppression (NMS) across overlapping candidates so each real
occurrence is reported once.
"""
import pickle
from dataclasses import dataclass

import numpy as np
import torch

import config as cfg
from features import logmel
from awe_model import AWEEncoder
from segmentation import Candidate


@dataclass
class Detection:
    word: str
    score: float
    start_sec: float
    end_sec: float


def load_model_and_bank():
    device = cfg.DEVICE if torch.cuda.is_available() else "cpu"
    model = AWEEncoder().to(device)
    model.load_state_dict(torch.load(cfg.AWE_MODEL_PATH, map_location=device))
    model.eval()

    with open(cfg.REFERENCE_BANK_PATH, "rb") as f:
        bank: dict[str, list[np.ndarray]] = pickle.load(f)

    # pre-stack into a single matrix + word index for fast matmul similarity
    words, vectors = [], []
    for word, embs in bank.items():
        for e in embs:
            words.append(word)
            vectors.append(e)
    bank_matrix = torch.tensor(np.stack(vectors), dtype=torch.float32, device=device)  # (N, D)
    bank_matrix = torch.nn.functional.normalize(bank_matrix, p=2, dim=1)

    return model, device, words, bank_matrix


@torch.no_grad()
def embed_candidates(model, device, y: np.ndarray, candidates: list[Candidate],
                      batch_size: int = 64) -> torch.Tensor:
    """Returns (N, D) L2-normalized embeddings for all candidates."""
    from awe_model import pad_collate

    all_embs = []
    for i in range(0, len(candidates), batch_size):
        batch_cands = candidates[i:i + batch_size]
        feats = [torch.from_numpy(logmel(y[c.start_sample:c.end_sample])) for c in batch_cands]
        x, lengths = pad_collate(feats)
        x, lengths = x.to(device), lengths.to(device)
        emb = model(x, lengths)  # already L2-normalized inside the model
        all_embs.append(emb)
    return torch.cat(all_embs, dim=0) if all_embs else torch.empty(0, cfg.EMBED_DIM, device=device)


def match_candidates(cand_embs: torch.Tensor, bank_words: list[str], bank_matrix: torch.Tensor,
                      candidates: list[Candidate], threshold: float) -> list[Detection]:
    if cand_embs.shape[0] == 0:
        return []
    sims = cand_embs @ bank_matrix.T  # (N_cand, N_bank)
    best_scores, best_idx = sims.max(dim=1)

    detections = []
    for cand, score, idx in zip(candidates, best_scores.tolist(), best_idx.tolist()):
        if score >= threshold:
            detections.append(Detection(
                word=bank_words[idx], score=score,
                start_sec=cand.start_sec, end_sec=cand.end_sec,
            ))
    return detections


def _iou(a: Detection, b: Detection) -> float:
    inter = max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
    union = max(a.end_sec, b.end_sec) - min(a.start_sec, b.start_sec)
    return inter / union if union > 0 else 0.0


def non_max_suppression(detections: list[Detection],
                         iou_threshold: float = cfg.NMS_IOU_THRESHOLD) -> list[Detection]:
    detections = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: list[Detection] = []
    for det in detections:
        if all(_iou(det, k) < iou_threshold for k in kept):
            kept.append(det)
    return sorted(kept, key=lambda d: d.start_sec)


def run_pipeline_with_loaded(model, device, bank_words: list[str], bank_matrix: torch.Tensor,
                              y: np.ndarray, threshold: float = cfg.DEFAULT_SIM_THRESHOLD
                              ) -> list[Detection]:
    """Same as run_pipeline but reuses an already-loaded model/bank
    (use this in the Streamlit app so weights aren't reloaded every run)."""
    from segmentation import generate_candidates

    candidates = generate_candidates(y)
    if not candidates:
        return []
    cand_embs = embed_candidates(model, device, y, candidates)
    raw_detections = match_candidates(cand_embs, bank_words, bank_matrix, candidates, threshold)
    return non_max_suppression(raw_detections)


def run_pipeline(y: np.ndarray, threshold: float = cfg.DEFAULT_SIM_THRESHOLD) -> list[Detection]:
    """Convenience one-shot version (loads model + bank each call) —
    fine for CLI/scripts, avoid in the Streamlit loop."""
    model, device, bank_words, bank_matrix = load_model_and_bank()
    return run_pipeline_with_loaded(model, device, bank_words, bank_matrix, y, threshold)

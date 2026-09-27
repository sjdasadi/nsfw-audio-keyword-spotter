"""
awe_model.py — the Acoustic Word Embedding (AWE) encoder.

A small 2D-CNN over the (time, mel) spectrogram, ending in adaptive average
pooling so it accepts variable-length input words, followed by a projection
to a fixed-size, L2-normalized embedding.

Trained with triplet loss so that:
    - same word, different speaker  -> embeddings close together
    - different words                -> embeddings far apart

This is what lets us do keyword spotting purely by embedding similarity,
without ever running ASR / full transcription.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as cfg


class AWEEncoder(nn.Module):
    def __init__(self, n_mels: int = cfg.N_MELS, embed_dim: int = cfg.EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),  # halve time & mel dims

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        # NOTE: no longer used for variable-length batches (see _masked_time_pool
        # below), kept only as a fallback for the rare case lengths aren't given.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
        )

    def _masked_time_pool(self, h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        h: (B, C, T', F') feature map after the conv stack.
        lengths: (B,) valid (non-padded) length of each sample, in the SAME
                 time units as the original input `x` (before downsampling).

        Averages only over the real (non-padded) time frames of each sample,
        instead of blindly averaging over the whole zero-padded canvas. This
        matters a lot here: without it, short words batched alongside much
        longer ones get their pooled features diluted by padding, which is
        what was causing the AWE encoder to collapse to near-identical
        embeddings for every word (see build_reference_bank.py's genuine vs.
        impostor report — they were ~0.99 vs ~0.99, i.e. no discrimination).
        """
        B, C, T_out, F_out = h.shape
        # Two MaxPool2d(2,2) layers downsample time by this ratio (approx 4x,
        # computed exactly rather than hardcoded so it stays correct if the
        # architecture changes).
        # x's original time dim isn't available here, so we infer the ratio
        # from the caller instead (see forward()).
        valid_t = lengths.clamp(min=1, max=T_out).to(h.device)  # (B,)
        time_idx = torch.arange(T_out, device=h.device).unsqueeze(0)  # (1, T')
        mask = (time_idx < valid_t.unsqueeze(1)).float()  # (B, T')
        mask = mask.view(B, 1, T_out, 1)  # broadcast over channels & freq

        h_masked = h * mask
        time_sum = h_masked.sum(dim=2)                      # (B, C, F')
        time_count = mask.sum(dim=2).clamp(min=1.0)          # (B, 1, F') -> broadcasts
        pooled = time_sum / time_count                       # (B, C, F')
        pooled = pooled.mean(dim=2)                           # (B, C) -- freq is never padded
        return pooled

    def forward(self, x: torch.Tensor, lengths: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, 1, T, n_mels)  -- zero-padded batch of log-mel spectrograms
        lengths: optional (B,) tensor with each sample's real (non-padded)
                 number of time frames, as returned by pad_collate(). Strongly
                 recommended whenever the batch mixes samples of different
                 length (the normal case here) — see _masked_time_pool.
        returns: (B, embed_dim) L2-normalized embeddings
        """
        h = self.conv(x)  # (B, 128, T', F')
        if lengths is not None:
            down_factor = x.shape[2] / h.shape[2]
            down_lengths = torch.floor(lengths.float() / down_factor).long()
            h = self._masked_time_pool(h, down_lengths)  # (B, 128)
        else:
            h = self.pool(h).flatten(1)  # (B, 128) -- fallback, no masking
        z = self.fc(h)                  # (B, embed_dim)
        z = F.normalize(z, p=2, dim=1)  # unit-norm -> cosine sim == dot product
        return z


def pad_collate(batch: list[torch.Tensor]):
    """
    batch: list of (T_i, n_mels) tensors
    returns: (padded (B, 1, T_max, n_mels) tensor, lengths (B,) LongTensor)
    """
    max_t = max(item.shape[0] for item in batch)
    n_mels = batch[0].shape[1]
    out = torch.zeros(len(batch), 1, max_t, n_mels, dtype=torch.float32)
    lengths = torch.zeros(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        t = item.shape[0]
        out[i, 0, :t, :] = item
        lengths[i] = t
    return out, lengths


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a: (D,) or (N,D), b: (D,) or (M,D) -> similarity matrix / scalar."""
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    return a @ b.T

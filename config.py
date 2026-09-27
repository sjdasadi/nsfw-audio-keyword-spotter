"""
Central configuration for the NSFW Keyword-Spotting (AWE-based) pipeline.
Edit paths / hyperparameters here — every script imports from this file.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).parent

SPEAKER_REFS_DIR = ROOT_DIR / "speaker_refs"      # 3-5 reference voice clips for XTTS cloning
DATA_DIR = ROOT_DIR / "data"
NSFW_WORDLIST = DATA_DIR / "nsfw_words.txt"        #  fill this in
NORMAL_WORDLIST = DATA_DIR / "normal_words.txt"    # auto-generated, used as negatives

REFERENCES_DIR = ROOT_DIR / "references"           # synthetic word audio (XTTS output)
REFERENCES_NSFW_DIR = REFERENCES_DIR / "nsfw"
REFERENCES_NORMAL_DIR = REFERENCES_DIR / "normal"

CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
AWE_MODEL_PATH = CHECKPOINT_DIR / "awe_encoder.pt"
REFERENCE_BANK_PATH = CHECKPOINT_DIR / "reference_bank.pkl"
THRESHOLD_REPORT_PATH = CHECKPOINT_DIR / "threshold_report.json"

# ---------------------------------------------------------------------------
# Audio / feature extraction
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16000
N_MELS = 40
N_FFT = 400          # 25 ms @ 16kHz
HOP_LENGTH = 160      # 10 ms @ 16kHz

# ---------------------------------------------------------------------------
# XTTS-v2 synthesis
# ---------------------------------------------------------------------------
XTTS_MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
XTTS_LANGUAGE = "en"
MIN_SPEAKER_REFS = 3
MAX_SPEAKER_REFS = 5

# ---------------------------------------------------------------------------
# AWE encoder / training
# ---------------------------------------------------------------------------
EMBED_DIM = 128
TRIPLET_MARGIN = 0.4
TRAIN_EPOCHS = 30          # CPU-only + small word list -> 25-30 is usually enough
TRAIN_BATCH_SIZE = 32      # CPU-only -> try 16 if training feels slow/memory-heavy
TRIPLETS_PER_EPOCH = 1000  # CPU-only -> try 800-1000 for faster epochs
LEARNING_RATE = 1e-3
DEVICE = "cuda"  # falls back to "cpu" automatically at runtime if unavailable

# ---------------------------------------------------------------------------
# Segmentation (no ASR — sliding window inside VAD speech regions)
# ---------------------------------------------------------------------------
VAD_AGGRESSIVENESS = 2            # 0-3, higher = more aggressive speech/non-speech split
VAD_FRAME_MS = 30                 # webrtcvad supports 10/20/30 ms frames
WINDOW_SIZES_MS = [300, 450, 600, 800]   # candidate word-length hypotheses
WINDOW_HOP_MS = 100                       # CPU-only + long files -> increase to 150-200ms
                                           # to cut candidate count (fewer, faster embeddings)

# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------
DEFAULT_SIM_THRESHOLD = 0.85     # cosine similarity threshold, tune via threshold_report.json
NMS_IOU_THRESHOLD = 0.3

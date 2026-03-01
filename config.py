"""
Central configuration: paths, thresholds, and model names.
All pipeline modules import from here — change once, applies everywhere.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Runtime paths (overridden by CLI args in run_pipeline.py)
# ---------------------------------------------------------------------------
SOURCE_DIR: str = ""          # flat directory of recovered JPGs
OUTPUT_DIR: str = "./output"  # root of organised output tree
DB_PATH: str = "images.db"    # SQLite database
EMBEDDING_DIR: str = os.path.join(OUTPUT_DIR, "embeddings")  # .npy files

# ---------------------------------------------------------------------------
# Phase 1 — scan thresholds
# ---------------------------------------------------------------------------
MIN_WIDTH: int = 50    # pixels — images smaller than this are deleted
MIN_HEIGHT: int = 50
SCAN_WORKERS: int = 16  # ThreadPoolExecutor workers for I/O-bound Phase 1

# ---------------------------------------------------------------------------
# Phase 3 — near-duplicate detection
# ---------------------------------------------------------------------------
PHASH_THRESHOLD: int = 8       # Hamming distance for perceptual-hash pre-filter
COSINE_THRESHOLD: float = 0.97  # DINOv2 cosine similarity to confirm near-dup

DINOV2_MODEL: str = "facebook/dinov2-base"
DINOV2_BATCH: int = 64
DINOV2_IMAGE_SIZE: int = 224   # DINOv2 input resolution

# ---------------------------------------------------------------------------
# Phase 4A — CLIP scene classification
# ---------------------------------------------------------------------------
CLIP_MODEL: str = "openai/clip-vit-base-patch32"
CLIP_BATCH: int = 64

CLIP_LABELS: list[str] = [
    "people portrait",
    "group of people",
    "landscape nature",
    "urban city",
    "food",
    "document text",
    "screenshot",
    "animals pets",
    "vehicles",
    "indoor home",
    "events celebration",
    "sports",
    "night scene",
]

# ---------------------------------------------------------------------------
# Phase 4B — face detection & clustering
# ---------------------------------------------------------------------------
YOLO_FACE_MODEL: str = "arnabdhar/YOLOv8-Face-Detection"
FACE_CONF_THRESHOLD: float = 0.25  # YOLO confidence to accept a detection
ARCFACE_MODEL: str = "ArcFace"     # deepface backend
FACE_EMB_DIM: int = 512

DBSCAN_EPS: float = 0.4
DBSCAN_MIN_SAMPLES: int = 2
FACE_MIN_CLUSTER_SIZE: int = 10  # ignore person clusters with fewer images than this

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Components & Dependencies

This document explains every dependency used by the pipeline,
why it was chosen, and what it is responsible for.

---

## Python Standard Library

| Module | Used in | Purpose |
|---|---|---|
| `sqlite3` | `db.py`, all phases | Persistent metadata store — zero extra dependencies, queryable, file-portable |
| `concurrent.futures` | `phase1_scan.py` | `ThreadPoolExecutor` for parallel file I/O during the scan phase |
| `hashlib` | `phase1_scan.py` | SHA-256 content hashing for exact-duplicate detection |
| `pathlib` | Throughout | Cross-platform path manipulation |
| `shutil` | `phase5_organize.py` | Moving files during the organization phase |
| `os` | Throughout | File deletion, symlink creation, environment variable access |

---

## Image Processing

### Pillow (`pillow`)
- **Phase:** 1, 3, 4
- **Why:** The de-facto standard Python imaging library.  Used to open images,
  verify they are not corrupt, read pixel dimensions, and decode image data
  for model input.
- **Key call:** `Image.open(path).verify()` — raises on corrupt files.

### exifread (`exifread`)
- **Phase:** 1
- **Why:** Pure-Python EXIF parser; reads date taken, GPS, camera make/model
  from JPEG metadata without requiring any native libraries.  Faster and more
  tolerant of malformed EXIF than Pillow's built-in EXIF support.

---

## Hashing & Near-Duplicate Detection

### imagehash (`imagehash`)
- **Phase:** 3 (Stage A)
- **Why:** Implements perceptual hashing algorithms (dHash, pHash, aHash).
  `dHash` computes a 64-bit fingerprint based on pixel gradient directions,
  making it robust to minor resizing and JPEG recompression.
- **Role:** Pre-filter — quickly narrows 30 K images down to a small set of
  candidate pairs before the expensive GPU step.

### BK-tree (implemented in `phase3_near_dedup.py`)
- **Phase:** 3 (Stage A)
- **Why:** A metric tree for Hamming-distance queries.  Allows finding all
  dHash values within a Hamming distance ≤ 8 of a query in O(log n) time
  instead of O(n) brute-force, which matters at 30 K images.

---

## Deep Learning Framework

### PyTorch (`torch`, `torchvision`)
- **Phases:** 3, 4
- **Why:** Standard GPU deep-learning framework.  Required by HuggingFace
  Transformers and Ultralytics.  CUDA support enables batch inference on the
  GPU, reducing Phase 3+4 runtime from days (CPU) to hours.
- **`float16` mode:** Enabled on CUDA to halve VRAM usage and increase
  throughput on modern NVIDIA GPUs (Ampere+).

---

## AI / ML Models

### HuggingFace Transformers (`transformers`)
- **Phases:** 3, 4A
- **Why:** Provides a unified API for loading and running pre-trained
  vision models without needing to manage model weights manually.

#### `facebook/dinov2-base` — DINOv2
- **Phase:** 3 (Stage B — near-duplicate confirmation)
- **Why chosen:** DINOv2 is a self-supervised Vision Transformer trained
  without labels on 142 M images.  Its CLS-token embeddings are extremely
  discriminative for image identity regardless of JPEG quality, slight crops,
  or colour adjustments — exactly what near-duplicate detection needs.
  `dinov2-base` (86 M parameters) is the fastest DINOv2 variant while still
  achieving the accuracy required (cosine similarity threshold 0.97).
- **Output:** 768-dimensional L2-normalised embedding per image.
- **Size:** ~340 MB download.

#### `openai/clip-vit-base-patch32` — CLIP
- **Phase:** 4A (scene classification)
- **Why chosen:** CLIP is trained on 400 M image-text pairs and supports
  **zero-shot classification** — you provide natural-language labels at
  inference time and it scores each image against all labels without any
  fine-tuning.  This means the 13-label taxonomy can be changed freely
  in `config.py` without retraining.
- **Output:** Top-1 label + softmax probability per image.
- **Size:** ~350 MB download.

### Ultralytics (`ultralytics`)
- **Phase:** 4B (face detection)
- **Why:** The `ultralytics` package provides YOLOv8, the current
  best-in-class real-time object detector.  It loads the HuggingFace model
  `arnabdhar/YOLOv8-Face-Detection` directly and runs GPU-accelerated
  inference.
- **Output:** Bounding boxes + confidence scores for each detected face.

### DeepFace (`deepface`)
- **Phase:** 4B (face embedding)
- **Why:** Wraps several production-grade face-recognition models behind a
  simple API.  We use the **ArcFace** backend, which produces 512-dimensional
  embeddings optimised for identity discrimination
  (trained with additive angular margin loss).
- **Output:** 512-dim L2-normalised vector per face crop.

---

## Clustering & Search

### scikit-learn (`scikit-learn`)
- **Phase:** 4B (DBSCAN clustering)
- **Why:** `DBSCAN` (Density-Based Spatial Clustering of Applications with
  Noise) does not require specifying the number of clusters in advance —
  ideal for grouping faces of unknown people.  Noise points (unique/unseen
  faces) are assigned label `-1` rather than forced into a cluster.
- **Parameters:** `eps=0.4` (cosine distance), `min_samples=2`.

### FAISS (`faiss-gpu` / `faiss-cpu`)
- **Phase:** 3, 4B
- **Why:** Facebook AI Similarity Search.  Provides highly optimised
  nearest-neighbour search over large embedding matrices.  The GPU variant
  runs the search on the NVIDIA card, handling millions of vectors in seconds.
- **Index used:** `IndexFlatIP` (exact inner-product search on L2-normalised
  vectors equals cosine similarity search).

---

## Utilities

### tqdm (`tqdm`)
- **All phases**
- Progress bars for long-running loops.  Shows ETA, throughput, and current
  counters so you can monitor a 2-hour GPU run without guessing.

### NumPy (`numpy`)
- **Phases:** 3, 4
- Array operations for embeddings, similarity matrices, and DBSCAN input.
  DINOv2 and ArcFace embeddings are stored as `.npy` files.

---

## Why SQLite instead of a flat CSV / JSON?

| Need | SQLite handles it |
|---|---|
| Persist state across phases | ✓ ACID transactions |
| Restart a phase mid-run | ✓ `ON CONFLICT DO UPDATE` upserts |
| Query subsets (e.g. only `status='ok'`) | ✓ Indexed SQL queries |
| Audit trail of every deletion | ✓ `status` column history |
| Zero infrastructure | ✓ Single file, no server |

---

## Why symlinks for secondary views (Phase 5)?

The surviving image collection may be 15–20 GB.  Creating hard-copies for
each of the four directory views (`by_date`, `by_category`, `by_camera`,
`by_person`) would multiply disk usage by 4×.  Symlinks are filesystem
pointers — they take essentially zero space and are transparent to any photo
viewer or file manager.

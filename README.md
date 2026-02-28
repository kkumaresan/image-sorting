# Image Recovery Cleanup & Organization Pipeline

A five-phase Python pipeline for cleaning and organizing a large flat directory
of recovered JPEG files (~30 K images, ~20 GB) using CPU pre-filtering and
NVIDIA GPU deep-learning models.

---

## What it does

| Phase | Name | Action |
|---|---|---|
| 1 | Scan & Validate | Opens every JPG, extracts EXIF metadata, computes SHA-256, deletes corrupt files and images smaller than 50×50 px |
| 2 | Exact Dedup | Groups images by SHA-256 hash, keeps the highest-resolution copy, deletes the rest |
| 3 | Near-Dedup | Uses perceptual hashing (dHash) + DINOv2 embeddings to find and remove visually identical images (resized, recompressed, slightly cropped) |
| 4 | Classify & Cluster | Assigns a scene category via CLIP zero-shot classification; detects faces with YOLOv8 and groups them into person clusters with ArcFace + DBSCAN |
| 5 | Organize | Moves surviving files into a date-based tree; creates symlinks for category, camera, and person views |

---

## Quick Start

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install PyTorch (CUDA variant — adjust for your GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install FAISS
pip install faiss-gpu              # or faiss-cpu

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Dry-run first (recommended — nothing is deleted)
python run_pipeline.py --source /path/to/recovered/images --dry-run

# 6. Full run
python run_pipeline.py --source /path/to/recovered/images --output ./output
```

---

## Project Structure

```
img-project/
├── .venv/                   ← virtual environment (not committed to git)
├── docs/
│   ├── installation.md      ← step-by-step install guide
│   ├── usage.md             ← CLI reference, workflow, query examples
│   ├── components.md        ← every dependency explained
│   ├── environment-variables.md  ← HF_HOME, CUDA_VISIBLE_DEVICES, etc.
│   ├── setup.md             ← disk/VRAM planning, troubleshooting
│   └── architecture.md      ← design decisions, schema, data flow
├── config.py                ← all thresholds, model names, device settings
├── db.py                    ← SQLite schema and helper functions
├── phase1_scan.py
├── phase2_dedup.py
├── phase3_near_dedup.py
├── phase4_classify.py
├── phase5_organize.py
├── run_pipeline.py          ← CLI orchestrator
└── requirements.txt
```

---

## Documentation

| Doc | Contents |
|---|---|
| [Installation](docs/installation.md) | System requirements, venv setup, PyTorch variants, dependency install |
| [Usage](docs/usage.md) | CLI flags, recommended workflow, output structure, SQL query examples |
| [Components](docs/components.md) | Every library explained — what it is and why it was chosen |
| [Environment Variables](docs/environment-variables.md) | `HF_HOME`, `CUDA_VISIBLE_DEVICES`, `DEEPFACE_HOME`, etc. |
| [Setup Reference](docs/setup.md) | Checklist, disk/VRAM planning, runtimes, troubleshooting |
| [Architecture](docs/architecture.md) | Pipeline diagram, DB schema, design decisions, extension guide |

---

## Requirements Summary

- Python 3.9+
- NVIDIA GPU with 6–8 GB VRAM (optional but strongly recommended)
- ~40 GB free disk space
- Internet access for first-run model downloads (~1 GB total)

---

## CLI Reference

```
python run_pipeline.py --source PATH [--output PATH] [--db PATH]
                       [--phase {1,2,3,4,5}] [--dry-run]
```

| Flag | Default | Description |
|---|---|---|
| `--source` | required | Flat directory containing recovered JPGs |
| `--output` | `./output` | Root of organized output tree |
| `--db` | `images.db` | SQLite database file |
| `--phase` | all | Run a single phase (1–5) |
| `--dry-run` | off | Simulate deletes and moves, print summary, make no changes |

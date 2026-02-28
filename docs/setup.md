# Setup Reference

End-to-end checklist for getting the pipeline from zero to a completed run.

---

## Checklist

- [ ] Python 3.13 installed (`brew install python@3.13` — see [installation.md](installation.md))
- [ ] NVIDIA driver installed (if using GPU)
- [ ] CUDA toolkit installed (if using GPU)
- [ ] Virtual environment created with Python 3.13: `/opt/homebrew/bin/python3.13 -m venv .venv`
- [ ] Venv activated: `source .venv/bin/activate`
- [ ] pip upgraded: `pip install --upgrade pip`
- [ ] PyTorch installed (correct CUDA variant — see [installation.md](installation.md))
- [ ] FAISS installed (`faiss-gpu` or `faiss-cpu`)
- [ ] Remaining dependencies installed: `pip install -r requirements.txt`
- [ ] All imports verified: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Dry-run executed: `python run_pipeline.py --source <path> --dry-run`
- [ ] Dry-run output reviewed
- [ ] Full pipeline executed

---

## Disk Space Planning

| Item | Approx. size |
|---|---|
| Source images (30 K JPGs) | ~20 GB |
| Output `by_date/` (moved files) | same ~20 GB |
| Output `by_category/`, `by_camera/`, `by_person/` | ~0 (symlinks) |
| `embeddings/` (DINOv2 `.npy` files, 768-dim float32) | ~3.5 GB for 30 K images |
| `images.db` (SQLite metadata) | ~50–100 MB |
| HuggingFace model cache | ~1.5 GB (DINOv2 + CLIP) |
| DeepFace ArcFace weights | ~240 MB |
| YOLO face detection weights | ~6 MB |
| **Total working space needed** | **~26 GB minimum** |

Recommendation: have at least 40 GB free on your working drive.

---

## GPU Memory Requirements

| Phase | Model | Approx. VRAM |
|---|---|---|
| 3 | DINOv2-base (float16) | ~2 GB at batch 64 |
| 4A | CLIP ViT-B/32 (float16) | ~2 GB at batch 64 |
| 4B | YOLOv8-Face + ArcFace | ~3–4 GB combined |

These phases run **sequentially** — maximum simultaneous VRAM is ~4 GB.
An 8 GB GPU (e.g. RTX 3070 / 4060 Ti) is comfortable; 6 GB may work with
`DINOV2_BATCH = 32` and `CLIP_BATCH = 32` set in `config.py`.

---

## Estimated Runtimes

Timings are approximate for 30 K images on a mid-range machine:

| Phase | CPU only | NVIDIA RTX 3080 |
|---|---|---|
| 1 — Scan & validate | 30–45 min | 30–45 min (I/O bound) |
| 2 — Exact dedup | < 1 min | < 1 min |
| 3 — Near-dedup (DINOv2) | 8–12 hours | 1–2 hours |
| 4A — CLIP classify | 4–6 hours | 30–60 min |
| 4B — Face clustering | 6–10 hours | 1–2 hours |
| 5 — Organize | 15–30 min | 15–30 min (I/O bound) |

---

## First-Run Model Downloads

On the first execution, the following models are downloaded automatically:

| Model | Size | Downloaded by |
|---|---|---|
| `facebook/dinov2-base` | ~340 MB | HuggingFace Transformers |
| `openai/clip-vit-base-patch32` | ~350 MB | HuggingFace Transformers |
| `arnabdhar/YOLOv8-Face-Detection` | ~6 MB | Ultralytics |
| ArcFace (deepface) | ~240 MB | DeepFace |

**Total first-run download: ~1 GB.**
Subsequent runs use the local cache.

To pre-download models before running the pipeline (useful on servers):

```bash
python - <<'EOF'
from transformers import AutoModel, AutoFeatureExtractor, CLIPModel, CLIPProcessor
AutoFeatureExtractor.from_pretrained("facebook/dinov2-base")
AutoModel.from_pretrained("facebook/dinov2-base")
CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print("HuggingFace models cached.")
EOF
```

DeepFace downloads ArcFace weights on first `DeepFace.represent()` call.

---

## Reducing Batch Sizes (Low VRAM)

Edit `config.py`:

```python
DINOV2_BATCH = 32   # default 64 — halves VRAM for Phase 3
CLIP_BATCH   = 32   # default 64 — halves VRAM for Phase 4A
```

---

## Running on a Headless Server

The pipeline has no GUI or display requirement.  All output goes to stdout.
Run in a `tmux` or `screen` session to survive SSH disconnections:

```bash
tmux new-session -s pipeline
source .venv/bin/activate
python run_pipeline.py --source /data/recovered --output /data/output 2>&1 | tee run.log
# Detach: Ctrl-B, D
# Re-attach: tmux attach -t pipeline
```

---

## Troubleshooting

### `ImportError: No module named 'torch'`
The virtual environment is not activated, or PyTorch was not installed into it.
```bash
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### `CUDA out of memory`
Reduce batch sizes in `config.py` (`DINOV2_BATCH`, `CLIP_BATCH`) and re-run
the affected phase.

### Phase 3 is slow (no GPU speedup visible)
Check `CUDA_VISIBLE_DEVICES` is not set to `-1` and that `torch.cuda.is_available()`
returns `True` inside the venv.

### `faiss` import error
Ensure you installed exactly one of `faiss-gpu` **or** `faiss-cpu`, not both.
```bash
pip uninstall faiss-gpu faiss-cpu -y
pip install faiss-gpu   # or faiss-cpu
```

### YOLOv8 model not found
If `arnabdhar/YOLOv8-Face-Detection` fails to load, download the weights
manually and place them as `yolov8n-face.pt` in the project root.  The code
falls back to that filename automatically.

### DeepFace ArcFace download fails
Set `DEEPFACE_HOME` to a writable path and ensure internet access:
```bash
export DEEPFACE_HOME=/tmp/deepface_cache
```

### SQLite database is locked
Another pipeline process is running, or a previous run crashed mid-write.
Wait for it to finish, or if the database is in a known good state:
```bash
sqlite3 images.db "PRAGMA wal_checkpoint(FULL);"
```

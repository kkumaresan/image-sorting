# Environment Variables

The pipeline reads no `.env` file — all settings are in `config.py` and
command-line arguments.  The environment variables below come from the
third-party libraries the pipeline uses and from the OS.

---

## HuggingFace / Transformers

| Variable | Default | Description |
|---|---|---|
| `HF_HOME` | `~/.cache/huggingface` | Root cache directory for all HuggingFace downloads (models, tokenizers, datasets). Set this to a path on a large disk if your home directory is small. |
| `TRANSFORMERS_CACHE` | `$HF_HOME/hub` | Override only the model cache location (subset of `HF_HOME`). |
| `HF_HUB_OFFLINE` | unset | Set to `1` to prevent any network calls — the pipeline will fail if a model is not already cached. Useful in air-gapped environments after the initial download. |
| `HUGGING_FACE_HUB_TOKEN` | unset | Personal access token for private HuggingFace Hub models. Not required for any model used in this pipeline (all are public). |

### Example — redirect model cache to a larger drive

```bash
export HF_HOME=/data/hf_cache
source .venv/bin/activate
python run_pipeline.py --source /path/to/images
```

---

## PyTorch / CUDA

| Variable | Default | Description |
|---|---|---|
| `CUDA_VISIBLE_DEVICES` | all GPUs | Restrict which GPU(s) the pipeline sees. Set to `0` to use only the first GPU, `1` for the second, or `-1` to force CPU mode. |
| `TORCH_HOME` | `~/.cache/torch` | Cache for PyTorch hub models (not used directly, but set automatically). |

### Example — run on the second GPU only

```bash
export CUDA_VISIBLE_DEVICES=1
python run_pipeline.py --source /path/to/images
```

### Example — force CPU mode (for testing)

```bash
export CUDA_VISIBLE_DEVICES=-1
python run_pipeline.py --source /path/to/images --phase 1
```

---

## Ultralytics (YOLOv8)

| Variable | Default | Description |
|---|---|---|
| `YOLO_CONFIG_DIR` | `~/.config/Ultralytics` | Directory where Ultralytics stores its settings and downloaded model weights. |

---

## DeepFace (ArcFace)

| Variable | Default | Description |
|---|---|---|
| `DEEPFACE_HOME` | `~/.deepface` | Root directory for DeepFace model weights. ArcFace weights (~240 MB) are downloaded here on first use. |

---

## OpenMP / Threading (performance tuning)

| Variable | Default | Description |
|---|---|---|
| `OMP_NUM_THREADS` | auto | Number of OpenMP threads for CPU BLAS operations. If the machine has many cores but you want to leave headroom, set this to e.g. `4`. |
| `MKL_NUM_THREADS` | auto | Same, for Intel MKL backend. |

### Example — limit CPU parallelism to 4 threads

```bash
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

---

## Pipeline-level settings (not ENV vars — use config.py or CLI)

These are **not** environment variables but are the most common things
people want to change.  Edit `config.py` directly or pass CLI arguments:

| Setting | Location | CLI flag |
|---|---|---|
| Source directory | `config.SOURCE_DIR` | `--source` |
| Output directory | `config.OUTPUT_DIR` | `--output` |
| Database path | `config.DB_PATH` | `--db` |
| Min image size | `config.MIN_WIDTH / MIN_HEIGHT` | — edit config.py |
| Near-dup thresholds | `config.PHASH_THRESHOLD`, `config.COSINE_THRESHOLD` | — edit config.py |
| Batch sizes | `config.DINOV2_BATCH`, `config.CLIP_BATCH` | — edit config.py |
| CLIP label set | `config.CLIP_LABELS` | — edit config.py |

---

## Quick reference — setting all recommended vars before a run

```bash
# On Linux/macOS, add these to ~/.bashrc or set per-session:

export HF_HOME=/data/model_cache          # large disk for models
export DEEPFACE_HOME=/data/model_cache/.deepface
export CUDA_VISIBLE_DEVICES=0             # use first GPU

source .venv/bin/activate
python run_pipeline.py \
    --source /media/recovered_drive/images \
    --output /data/organised_photos \
    --db /data/organised_photos/images.db
```

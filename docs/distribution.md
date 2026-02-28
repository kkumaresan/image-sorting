# Distribution & Standalone Packaging Plan

This document outlines options and a recommended plan for publishing the
image-sorting pipeline as a bundled standalone application.

---

## Background & Constraints

The pipeline has heavyweight ML dependencies that make traditional Python
bundling tools (PyInstaller, Nuitka) impractical:

- PyTorch + torchvision (~2 GB, dynamic CUDA library loading)
- transformers / DINOv2 / CLIP (~1 GB model downloads on first run)
- ultralytics (YOLOv8), deepface, faiss-gpu
- NVIDIA GPU + drivers required for reasonable performance

---

## Option Comparison

| Option | Standalone? | GPU support | Size | Complexity |
|---|---|---|---|---|
| Docker image | Yes | Yes (nvidia-container-toolkit) | ~6–8 GB | Low |
| conda-pack tarball | Partial | Needs host drivers | ~3–4 GB | Medium |
| PyInstaller binary | Partial | Fragile | ~4–5 GB | High |
| pipx / PyPI package | No (needs Python) | Yes | ~deps only | Low |

---

## Recommended Approach: Docker

Docker is the standard distribution format for GPU ML tools. It provides:

- Full reproducibility across machines
- CUDA handled via `nvidia/cuda` base image
- Model weights can be pre-baked into the image
- Simple UX: one `docker run` command for end users

### Planned files

```
image-sorting/
├── Dockerfile
├── docker-compose.yml
└── docs/
    └── docker.md          ← user-facing Docker usage guide
```

### Dockerfile outline

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# System deps
RUN apt-get update && apt-get install -y python3.11 python3-pip git && rm -rf /var/lib/apt/lists/*

# Install PyTorch (CUDA 12.1)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install FAISS
RUN pip install faiss-gpu

# Copy source and install remaining deps
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

# Pre-warm model downloads (bakes weights into the image layer)
# RUN python -c "from transformers import AutoImageProcessor; AutoImageProcessor.from_pretrained('facebook/dinov2-base')"
# RUN python -c "import clip; clip.load('ViT-B/32')"

ENTRYPOINT ["python", "run_pipeline.py"]
```

### docker-compose.yml outline

```yaml
services:
  pipeline:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ${SOURCE}:/data/source:ro
      - ${OUTPUT}:/data/output
    command: --source /data/source --output /data/output
```

User workflow:
```bash
SOURCE=/media/disk/recovered OUTPUT=./output docker compose run pipeline
# or with dry-run:
SOURCE=/media/disk/recovered OUTPUT=./output docker compose run pipeline --dry-run
```

---

## Secondary Option: conda-pack

For users who prefer not to use Docker on Linux/macOS:

```bash
# Developer (build once)
conda create -n image-sorting python=3.11
conda activate image-sorting
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install faiss-gpu -r requirements.txt
conda-pack -n image-sorting -o image-sorting-env.tar.gz

# End user (extract and run)
mkdir image-sorting-env && tar -xzf image-sorting-env.tar.gz -C image-sorting-env
./image-sorting-env/bin/python run_pipeline.py --source /photos --dry-run
```

Limitations:
- User still needs compatible NVIDIA drivers installed on host
- Tarball is 3–4 GB
- Does not bundle model weights

---

## Secondary Option: PyPI + pipx

For technical users comfortable with Python tooling. Requires adding:

- `pyproject.toml` with `[project.scripts]` entry point
- Version metadata

```bash
pipx install image-sorting-pipeline
image-sorting --source /photos --dry-run
```

Smallest distribution footprint but requires Python on the target machine.

---

## Implementation Order (when taken up)

1. [ ] Add `pyproject.toml` with project metadata and entry point (`image-sorting = run_pipeline:main`)
2. [ ] Write `Dockerfile` using `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` base
3. [ ] Write `docker-compose.yml` with volume mounts and GPU passthrough
4. [ ] Decide whether to pre-bake model weights into image (larger image, no internet needed) or download at first run
5. [ ] Write `docs/docker.md` — user-facing guide for building and running the container
6. [ ] Test on a machine with `nvidia-container-toolkit` installed
7. [ ] (Optional) Publish image to Docker Hub or GitHub Container Registry (GHCR)
8. [ ] (Optional) Publish package to PyPI for `pipx` install path

---

## Notes

- Model weight pre-baking is controlled by environment variable `HF_HOME` — set it
  to a path inside the image to cache weights in a known layer.
- For CPU-only users, a separate `Dockerfile.cpu` with `faiss-cpu` and a CPU
  PyTorch variant could be provided.
- The `images.db` SQLite file should be mounted from the host (not baked in) so
  pipeline state persists across container restarts.

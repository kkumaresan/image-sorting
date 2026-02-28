# Installation Guide

## Python Version Compatibility

| Python version | Code compatibility | Library support | Recommendation |
|---|---|---|---|
| 3.9.x | ✅ works (all type hints use `typing` module) | ⚠️ some packages dropped 3.9 wheels | Avoid for new setups |
| 3.10.x | ✅ | ✅ | Acceptable |
| 3.11.x | ✅ | ✅ | Good |
| 3.12.x | ✅ | ✅ | Good |
| **3.13.x** | **✅** | **✅ (with pinned versions)** | **Recommended** |

The code itself is compatible with Python 3.9 and up.
All `X | Y` union-type syntax has been replaced with `Optional[X]` from the
`typing` module so no 3.10+ features are required.
The `requirements.txt` floor versions (e.g. `numpy>=2.0`, `pillow>=10.4`) are
the earliest releases that ship Python 3.13 wheels.

---

## Step 1 — Install Python 3.13

Your system Python is 3.9.6 (macOS built-in).  Install 3.13 without touching
the system Python:

### Option A — Homebrew (recommended on macOS)

```bash
brew install python@3.13
```

After install, Homebrew prints the exact path.  Verify:

```bash
/opt/homebrew/bin/python3.13 --version
# Python 3.13.x
```

### Option B — pyenv (version manager, useful if you switch between projects)

```bash
# Install pyenv if not already installed
brew install pyenv

# Install Python 3.13
pyenv install 3.13

# Set it as the local version for this project directory
cd /path/to/img-project
pyenv local 3.13
python3 --version  # should show 3.13
```

### Option C — python.org installer

Download the macOS installer from https://www.python.org/downloads/ and run it.
The installer places `python3.13` on your PATH.

---

## Step 2 — Create the virtual environment with Python 3.13

```bash
cd /path/to/img-project

# Homebrew
/opt/homebrew/bin/python3.13 -m venv .venv

# pyenv (if you set pyenv local 3.13)
python3 -m venv .venv

# python.org installer
python3.13 -m venv .venv
```

> If a `.venv` from Python 3.9 already exists, remove it first:
> ```bash
> rm -rf .venv
> ```

Activate:

```bash
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell
.venv\Scripts\activate.bat         # Windows cmd
```

Your prompt should show `(.venv)` and:

```bash
python --version   # Python 3.13.x
```

---

## Step 3 — Upgrade pip

```bash
pip install --upgrade pip
```

---

## Step 4 — Install PyTorch

PyTorch must be installed **before** the rest of the requirements because the
correct variant (CUDA vs CPU) depends on your hardware.

### CUDA 12.x (recommended for NVIDIA GPUs)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### CUDA 11.8
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CPU only (no GPU)
```bash
pip install torch torchvision
```

Verify CUDA is detected:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Step 5 — Install FAISS

```bash
# NVIDIA GPU
pip install faiss-gpu

# CPU only
pip install faiss-cpu
```

> Install exactly **one** of these — they conflict with each other.

---

## Step 6 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Step 7 — Verify the full installation

```bash
python -c "
import sys
print('Python:', sys.version)
import PIL, exifread, imagehash, transformers, torch, torchvision
import ultralytics, deepface, sklearn, numpy, tqdm
print('All imports OK')
print('CUDA:', torch.cuda.is_available())
"
```

---

## System Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.9 | 3.13 |
| RAM | 8 GB | 16 GB |
| Disk (working space) | 26 GB free | 40 GB+ free |
| GPU | — | NVIDIA GPU, 8 GB+ VRAM |
| CUDA | — | 11.8 or 12.x |
| OS | Linux / macOS / Windows | Ubuntu 22.04 LTS |

> **GPU note:** Phases 3, 4A, and 4B fall back to CPU automatically if no
> CUDA GPU is detected, but runtimes will increase from hours to days.

---

## Library compatibility with Python 3.13

| Package | Min version (py3.13 wheel) | Notes |
|---|---|---|
| `torch` | 2.4.0 | First release to ship py3.13 wheels |
| `torchvision` | 0.19.0 | Matches torch 2.4 |
| `numpy` | 2.0.0 | NumPy 2 is required; 1.x has no py3.13 wheel |
| `pillow` | 10.4.0 | First wheel with py3.13 support |
| `transformers` | 4.46.0 | Verified compatible |
| `ultralytics` | 8.3.0 | Verified compatible |
| `deepface` | 0.0.93 | Latest tested; depends on tf-keras |
| `scikit-learn` | 1.5.0 | First wheel with py3.13 support |
| `faiss-gpu/cpu` | latest | Install via pip; py3.13 wheels available |
| `imagehash` | 4.3.1 | Pure Python — any version works |
| `exifread` | 3.0.0 | Pure Python — any version works |
| `tqdm` | 4.66.0 | Pure Python — any version works |
| `scipy` | 1.13.0 | Needed by imagehash |

All minimum versions are already pinned in `requirements.txt`.

---

## Optional — HuggingFace model cache location

Models are cached in `~/.cache/huggingface/hub/` by default.  To redirect:

```bash
export HF_HOME=/path/to/large/disk/hf_cache
```

See [environment-variables.md](environment-variables.md) for all ENV vars.

---

## Deactivating the environment

```bash
deactivate
```

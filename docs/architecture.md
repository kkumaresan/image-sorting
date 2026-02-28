# Architecture & Design

## Pipeline Overview

```
Source directory (flat, ~30 K JPGs)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│ Phase 1 — Scan & Validate                           │
│  16 parallel threads                                │
│  Pillow: open + verify + dimensions                 │
│  exifread: date, camera, GPS                        │
│  hashlib: SHA-256                                   │
│  → delete corrupt/tiny, write all metadata to DB   │
└───────────────────────┬─────────────────────────────┘
                        │ SQLite images.db
                        ▼
┌─────────────────────────────────────────────────────┐
│ Phase 2 — Exact Deduplication                       │
│  GROUP BY sha256 → keep highest-res, delete rest   │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ Phase 3 — Near-Duplicate Detection                  │
│  Stage A: dHash → BK-tree → candidate pairs         │
│  Stage B: DINOv2-base (GPU) → cosine ≥ 0.97        │
│  → delete near-dups, save .npy embeddings for all  │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ Phase 4 — Classification & Clustering               │
│  4A: CLIP ViT-B/32 (GPU) → 13-label zero-shot      │
│  4B: YOLOv8 face detect → ArcFace embed → DBSCAN   │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│ Phase 5 — Organize                                  │
│  Move files → output/by_date/YYYY/MM/               │
│  Create symlinks → by_category/, by_camera/,        │
│                    by_person/                       │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
              Organised output tree
```

---

## Module Dependency Graph

```
run_pipeline.py
├── config.py          ← imported by everything
├── db.py              ← imported by everything
├── phase1_scan.py
│   ├── Pillow
│   └── exifread
├── phase2_dedup.py
├── phase3_near_dedup.py
│   ├── imagehash
│   └── transformers (DINOv2)
├── phase4_classify.py
│   ├── transformers (CLIP)
│   ├── ultralytics (YOLOv8)
│   ├── deepface (ArcFace)
│   └── scikit-learn (DBSCAN)
└── phase5_organize.py
```

---

## Database Schema

```sql
CREATE TABLE images (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    path            TEXT    NOT NULL UNIQUE,  -- original absolute path
    sha256          TEXT,                     -- hex SHA-256
    phash           TEXT,                     -- 16-char hex dHash
    width           INTEGER,
    height          INTEGER,
    date_taken      TEXT,                     -- ISO-8601: 2018-06-15T10:23:45
    camera_make     TEXT,
    camera_model    TEXT,
    category        TEXT,                     -- CLIP top-1 label
    face_cluster_id INTEGER DEFAULT -1,       -- DBSCAN label; -1 = no face / noise
    status          TEXT    NOT NULL DEFAULT 'ok',
    embedding_path  TEXT,                     -- path to .npy DINOv2 embedding
    new_path        TEXT                      -- path after Phase 5 move
);
```

### Status values

| Value | Meaning |
|---|---|
| `ok` | Surviving image |
| `deleted_corrupt` | File could not be opened by Pillow |
| `deleted_tiny` | Smaller than 50×50 pixels |
| `deleted_exact_dup` | Identical SHA-256 to a higher-res file |
| `deleted_near_dup` | DINOv2 cosine similarity ≥ 0.97 to a higher-res file |

---

## Key Design Decisions

### Thread-safe SQLite writes (Phase 1)

Phase 1 uses 16 threads for parallel file I/O.  SQLite's default threading mode
would cause write conflicts.  We use:
- `PRAGMA journal_mode=WAL` — Write-Ahead Logging allows concurrent reads +
  one writer without blocking.
- Thread-local connections (`threading.local`) — each thread holds its own
  connection, eliminating lock contention.
- Batch commits every 500 rows — reduces the number of fsync calls, which are
  the bottleneck for SQLite write throughput.

### Two-stage near-duplicate detection (Phase 3)

Running DINOv2 on all 30 K × 30 K pairs is infeasible (900 M comparisons).
The pipeline uses a funnel:

1. **dHash + BK-tree** narrows the field to O(k·log n) comparisons where k
   is the average cluster size (usually small).
2. **DINOv2 cosine** only runs on the small candidate set, confirming true
   near-dups with high precision.

This makes Phase 3 tractable in 1–2 GPU hours instead of days.

### VRAM management (Phase 4)

CLIP (4A) and YOLOv8+ArcFace (4B) each need ~2–4 GB of VRAM.  Running them
simultaneously risks OOM errors on 8 GB cards.  Phase 4 runs 4A to completion,
explicitly deletes the CLIP model, calls `torch.cuda.empty_cache()`, then loads
the face models.

### Symlinks vs copies (Phase 5)

The surviving image set may still be 10–15 GB after deduplication.  Creating
copies for each of the four directory views would require 4× the space.  Symlinks
point into the `by_date/` tree (which holds the actual files), so the secondary
views consume negligible extra disk space.

### Dry-run mode

Every destructive operation (file delete, file move) is gated behind the
`dry_run` boolean that threads through from `run_pipeline.py`.  The DB is
still written during a dry-run (status fields and metadata), allowing you to
fully inspect what *would* happen using SQL queries before committing.

---

## Data Flow for Embeddings

DINOv2 embeddings computed in Phase 3 are:
1. Saved as `.npy` files in `output/embeddings/` (one file per image).
2. Their paths stored in `images.embedding_path`.

This means Phase 4 or any future downstream task (e.g. semantic search,
nearest-neighbour lookup) can load embeddings without rerunning the GPU
inference.

---

## Extending the Pipeline

### Adding a new CLIP category

Edit `config.CLIP_LABELS` — no code changes needed:

```python
CLIP_LABELS = [
    ...
    "underwater photography",   # add here
]
```

Re-run Phase 4 only:
```bash
python run_pipeline.py --source <path> --phase 4
```

### Using a different classification model

Replace `_run_clip()` in `phase4_classify.py`.  The function must write a
string to `images.category` for each path.  The interface is intentionally
minimal.

### Adding a new output view (Phase 5)

Add a new symlink loop in `phase5_organize.py`'s `run()` function following
the same pattern as `by_camera`:

```python
# Example: organize by year only (flat)
by_year_root = Path(output_dir) / "by_year"
_make_symlink(actual_dst, str(by_year_root / year / Path(actual_dst).name))
```

# Usage Guide

## Quick Start

```bash
# Activate the virtual environment
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# ALWAYS do a dry-run first — no files are deleted
python run_pipeline.py --source /path/to/recovered/images --dry-run

# After reviewing the dry-run summary, run for real
python run_pipeline.py --source /path/to/recovered/images --output ./output
```

---

## Command-Line Reference

```
python run_pipeline.py [OPTIONS]

Required:
  --source PATH     Flat directory that contains the recovered JPG files.

Optional:
  --output PATH     Root of the organised output tree.  (default: ./output)
  --db PATH         SQLite database file path.          (default: images.db)
  --phase {1,2,3,4,5}
                    Run only a single phase instead of all five.
  --dry-run         Simulate destructive operations (no deletes, no moves).
                    The database IS updated with metadata and status fields
                    so you can inspect results before committing.
```

---

## Recommended Workflow

### 1. Dry-run — review before deleting anything

```bash
python run_pipeline.py --source /media/disk/recovered --dry-run
```

Inspect what would be removed:

```bash
# Using the SQLite CLI
sqlite3 images.db "SELECT status, COUNT(*) FROM images GROUP BY status;"
```

Or open `images.db` in any SQLite GUI (DB Browser for SQLite, TablePlus, etc.)

### 2. Validate Phase 1 and 2 only (fast, low-risk)

```bash
python run_pipeline.py --source /media/disk/recovered --phase 1
python run_pipeline.py --source /media/disk/recovered --phase 2
```

### 3. Run Phase 3 (GPU, ~1–2 hours) then inspect near-dup groups

```bash
python run_pipeline.py --source /media/disk/recovered --phase 3

# Check near-dup count
sqlite3 images.db "SELECT COUNT(*) FROM images WHERE status='deleted_near_dup';"
```

### 4. Run Phase 4 (GPU, ~2–3 hours)

```bash
python run_pipeline.py --source /media/disk/recovered --phase 4
```

Inspect the category distribution:
```bash
sqlite3 images.db "SELECT category, COUNT(*) FROM images WHERE status='ok' GROUP BY category ORDER BY COUNT(*) DESC;"
```

### 5. Run Phase 5 — organize the survivors

```bash
python run_pipeline.py --source /media/disk/recovered --output ./output --phase 5
```

---

## Output Directory Tree

After Phase 5 completes, the `./output/` directory will look like:

```
output/
├── by_date/                  ← actual files (moved here)
│   ├── 2018/
│   │   ├── 06/
│   │   │   ├── IMG_0001.jpg
│   │   │   └── IMG_0002.jpg
│   │   └── unknown_month/
│   └── unknown/              ← images with no EXIF date
│
├── by_category/              ← symlinks into by_date/
│   ├── people_portrait/
│   ├── landscape_nature/
│   ├── food/
│   └── ...
│
├── by_camera/                ← symlinks into by_date/
│   ├── Apple_iPhone_12/
│   ├── Canon_EOS_5D/
│   └── unknown_camera/
│
├── by_person/                ← symlinks into by_date/
│   ├── person_0000/
│   ├── person_0001/
│   └── ...
│
└── embeddings/               ← DINOv2 .npy vectors (reusable for search)
```

Secondary views (`by_category`, `by_camera`, `by_person`) are **symlinks** —
they point back into `by_date/` so no extra disk space is consumed.

---

## Querying the Database

The `images.db` SQLite file is your full audit trail.

```sql
-- Overall status breakdown
SELECT status, COUNT(*) AS cnt FROM images GROUP BY status ORDER BY cnt DESC;

-- Images by year
SELECT substr(date_taken, 1, 4) AS year, COUNT(*) FROM images
WHERE status='ok' GROUP BY year ORDER BY year;

-- Face clusters with more than 10 images
SELECT face_cluster_id, COUNT(*) AS cnt FROM images
WHERE status='ok' AND face_cluster_id >= 0
GROUP BY face_cluster_id HAVING cnt > 10 ORDER BY cnt DESC;

-- Images that were deleted as near-duplicates
SELECT path FROM images WHERE status='deleted_near_dup';
```

---

## Re-running a Phase

Phases are idempotent — the DB uses `INSERT … ON CONFLICT … DO UPDATE` so
re-running a phase will update existing records rather than duplicate them.

```bash
# Re-run classification only
python run_pipeline.py --source /media/disk/recovered --phase 4
```

---

## Checking GPU Utilisation

```bash
watch -n1 nvidia-smi
```

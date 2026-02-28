"""
Phase 1 — Scan, Validate & Triage

Walk the source directory, validate every JPG, extract metadata,
delete corrupt/tiny files immediately, and populate the SQLite DB.
"""

import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import exifread
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

import config
import db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_exif(path: str) -> dict:
    """Return a dict with date_taken, camera_make, camera_model (all may be None)."""
    result = {"date_taken": None, "camera_make": None, "camera_model": None}
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, stop_tag="GPS GPSAltitude", details=False)

        # Date
        for tag in ("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"):
            if tag in tags:
                raw = str(tags[tag]).strip()
                # Convert "2018:06:15 10:23:45" → "2018-06-15T10:23:45"
                if raw and raw[0].isdigit():
                    result["date_taken"] = raw[:10].replace(":", "-") + "T" + raw[11:]
                break

        # Camera
        if "Image Make" in tags:
            result["camera_make"] = str(tags["Image Make"]).strip()
        if "Image Model" in tags:
            result["camera_model"] = str(tags["Image Model"]).strip()
    except Exception:
        pass  # silently ignore EXIF parse errors
    return result


def _process_file(path: str, dry_run: bool) -> dict:
    """
    Validate one image file.  Returns a row dict ready for DB upsert.
    Deletes the file from disk if it is corrupt or too small (unless dry_run).
    """
    row: dict = {"path": path, "status": "ok"}

    # --- Open & validate ---
    try:
        img = Image.open(path)
        img.verify()           # raises on corruption
        img = Image.open(path) # re-open after verify (verify closes it)
        width, height = img.size
    except (UnidentifiedImageError, Exception):
        row["status"] = "deleted_corrupt"
        if not dry_run:
            try:
                os.remove(path)
            except OSError:
                pass
        return row

    row["width"] = width
    row["height"] = height

    # --- Size filter ---
    # TODO: Add a pixel-content check here to catch visually corrupt images that
    # pass PIL's verify() — e.g. mostly-black images from partial disk recovery.
    # Approach: convert to grayscale, compute mean brightness; if below a threshold
    # (e.g. mean < 10 out of 255), mark as deleted_corrupt.
    # Also consider checking std deviation — a very low std means uniform/blank image.
    # Config knobs to add: MIN_BRIGHTNESS (default ~10), MIN_STD (default ~5).
    if width < config.MIN_WIDTH or height < config.MIN_HEIGHT:
        row["status"] = "deleted_tiny"
        if not dry_run:
            try:
                os.remove(path)
            except OSError:
                pass
        return row

    # --- EXIF ---
    exif = _parse_exif(path)
    row.update(exif)

    # --- Hash ---
    try:
        row["sha256"] = _sha256(path)
    except OSError:
        row["sha256"] = None

    return row


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(source_dir: str, dry_run: bool = False) -> None:
    print(f"\n[Phase 1] Scanning {source_dir!r} ...")
    db.init_db()

    # Collect all jpg/jpeg paths
    all_paths = [
        str(p)
        for p in Path(source_dir).rglob("*")
        if p.suffix.lower() in (".jpg", ".jpeg") and p.is_file()
    ]
    print(f"  Found {len(all_paths):,} JPG files")

    counts = {"ok": 0, "deleted_tiny": 0, "deleted_corrupt": 0}

    batch: list[dict] = []
    BATCH_COMMIT = 500  # write to DB every N rows

    with ThreadPoolExecutor(max_workers=config.SCAN_WORKERS) as pool:
        futures = {pool.submit(_process_file, p, dry_run): p for p in all_paths}
        pbar = tqdm(as_completed(futures), total=len(all_paths), unit="img", desc="Phase 1")

        for fut in pbar:
            row = fut.result()
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            batch.append(row)

            if len(batch) >= BATCH_COMMIT:
                with db.transaction():
                    for r in batch:
                        db.upsert_image(r)
                batch.clear()

            pbar.set_postfix(ok=counts["ok"], tiny=counts["deleted_tiny"], corrupt=counts["deleted_corrupt"])

        # Flush remaining
        if batch:
            with db.transaction():
                for r in batch:
                    db.upsert_image(r)

    print(f"\n[Phase 1] Complete.")
    print(f"  Surviving (ok)    : {counts['ok']:>6,}")
    print(f"  Deleted (tiny)    : {counts['deleted_tiny']:>6,}")
    print(f"  Deleted (corrupt) : {counts['deleted_corrupt']:>6,}")
    if dry_run:
        print("  [DRY-RUN] No files were actually deleted.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    config.SOURCE_DIR = args.source
    run(args.source, dry_run=args.dry_run)

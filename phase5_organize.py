"""
Phase 5 — Organize Into Directory Tree

Moves surviving files into output/by_date/YYYY/MM/filename.jpg
and creates symlinks in by_category/, by_camera/, by_person/.
All moves are recorded in the DB `new_path` column.
"""

import os
import shutil
from pathlib import Path
from typing import Optional, Tuple

import db
import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_move(src: str, dst: str) -> str:
    """
    Move src to dst.  If dst already exists, append _2, _3, … until unique.
    Returns the actual destination path used.
    """
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if not dst_path.exists():
        shutil.move(src, dst_path)
        return str(dst_path)

    # Conflict: append suffix
    stem = dst_path.stem
    suffix = dst_path.suffix
    parent = dst_path.parent
    counter = 2
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            shutil.move(src, candidate)
            return str(candidate)
        counter += 1


def _make_symlink(target: str, link: str) -> None:
    """Create a symlink at `link` pointing to `target`; skip if exists."""
    link_path = Path(link)
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.exists() or link_path.is_symlink():
        return
    # Use absolute target path for reliability
    link_path.symlink_to(os.path.abspath(target))


def _date_parts(date_taken: Optional[str]) -> Tuple[str, str]:
    """Return (year, month) strings from ISO date; fall back to 'unknown'."""
    if date_taken and len(date_taken) >= 7:
        parts = date_taken[:7].split("-")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return parts[0], parts[1]
    return "unknown", ""


def _safe_dirname(name: Optional[str]) -> str:
    """Turn an arbitrary string into a safe directory name."""
    if not name:
        return "unknown"
    return (
        name.strip()
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
        [:64]  # truncate very long names
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(output_dir: str = config.OUTPUT_DIR, dry_run: bool = False) -> None:
    print(f"\n[Phase 5] Organizing files into {output_dir!r} ...")
    conn = db.get_conn()

    ok_rows = conn.execute("""
        SELECT id, path, date_taken, camera_make, camera_model,
               category, face_cluster_id
        FROM images
        WHERE status = 'ok'
    """).fetchall()

    if not ok_rows:
        print("  No surviving images to organize.")
        return

    by_date_root = Path(output_dir) / "by_date"
    by_cat_root = Path(output_dir) / "by_category"
    by_cam_root = Path(output_dir) / "by_camera"
    by_person_root = Path(output_dir) / "by_person"

    moved = 0
    skipped = 0

    for row in ok_rows:
        src = row["path"]
        if not os.path.exists(src):
            skipped += 1
            continue

        filename = Path(src).name
        year, month = _date_parts(row["date_taken"])

        # ---- Primary move: by_date ----
        if year == "unknown":
            date_subdir = by_date_root / "unknown"
        elif not month:
            date_subdir = by_date_root / year / "unknown_month"
        else:
            date_subdir = by_date_root / year / month

        dst = str(date_subdir / filename)

        if not dry_run:
            actual_dst = _safe_move(src, dst)
            conn.execute("UPDATE images SET new_path=? WHERE id=?", (actual_dst, row["id"]))
        else:
            actual_dst = dst

        moved += 1

        if dry_run:
            continue

        # ---- Symlink: by_category ----
        cat = _safe_dirname(row["category"]) if row["category"] else "uncategorized"
        _make_symlink(actual_dst, str(by_cat_root / cat / Path(actual_dst).name))

        # ---- Symlink: by_camera ----
        cam_parts = [
            _safe_dirname(row["camera_make"]),
            _safe_dirname(row["camera_model"]),
        ]
        # Remove duplicates/unknowns for a cleaner path
        cam_dir = "_".join(p for p in cam_parts if p and p != "unknown") or "unknown_camera"
        _make_symlink(actual_dst, str(by_cam_root / cam_dir / Path(actual_dst).name))

        # ---- Symlink: by_person (face clusters) ----
        cluster_id = row["face_cluster_id"] if row["face_cluster_id"] is not None else -1
        if cluster_id >= 0:
            person_dir = f"person_{cluster_id:04d}"
            _make_symlink(actual_dst, str(by_person_root / person_dir / Path(actual_dst).name))

    if not dry_run:
        conn.commit()

    print(f"\n[Phase 5] Complete.")
    print(f"  Files moved    : {moved:>6,}")
    print(f"  Skipped (gone) : {skipped:>6,}")
    if dry_run:
        print("  [DRY-RUN] No files were moved; no symlinks created.")
    else:
        print(f"  Output tree    : {output_dir}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=config.OUTPUT_DIR)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    db.init_db()
    run(output_dir=args.output, dry_run=args.dry_run)

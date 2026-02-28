"""
Regroup faces using existing DB data — no model re-runs.

Reads current face_cluster_id + new_path from the DB, demotes clusters
with fewer than FACE_MIN_CLUSTER_SIZE images to -1, and re-creates
output/by_person/ symlinks from the files' current locations (new_path).

Usage:
    python regroup_faces.py [--output ./output] [--dry-run]
"""
import argparse
import os
import shutil
from collections import Counter
from pathlib import Path

import config
import db


def run(output_dir: str, dry_run: bool) -> None:
    conn = db.get_conn()
    rows = conn.execute("""
        SELECT id, new_path, face_cluster_id
        FROM images
        WHERE status = 'ok' AND new_path IS NOT NULL
    """).fetchall()

    # Count images per existing cluster
    cluster_counts = Counter(
        r["face_cluster_id"] for r in rows if r["face_cluster_id"] >= 0
    )
    valid_clusters = {c for c, n in cluster_counts.items()
                      if n >= config.FACE_MIN_CLUSTER_SIZE}
    n_total = len(cluster_counts)
    n_small = n_total - len(valid_clusters)

    print(f"  Clusters found   : {n_total}")
    print(f"  Clusters kept    : {len(valid_clusters)}  (>= {config.FACE_MIN_CLUSTER_SIZE} images)")
    print(f"  Clusters dropped : {n_small}  (< {config.FACE_MIN_CLUSTER_SIZE} images)")

    # Update DB — demote small clusters to -1
    if not dry_run:
        with db.transaction():
            for row in rows:
                cid = row["face_cluster_id"]
                if cid >= 0 and cid not in valid_clusters:
                    conn.execute(
                        "UPDATE images SET face_cluster_id=-1 WHERE id=?", (row["id"],)
                    )
        print("  DB updated.")

    # Remove old by_person tree and recreate
    by_person_root = Path(output_dir) / "by_person"
    if not dry_run:
        if by_person_root.exists():
            shutil.rmtree(by_person_root)
        by_person_root.mkdir(parents=True, exist_ok=True)

    symlinks_created = 0
    for row in rows:
        # In dry-run, simulate what the cluster_id would become after filtering
        cid = row["face_cluster_id"]
        if dry_run and cid >= 0 and cid not in valid_clusters:
            cid = -1
        if cid < 0:
            continue

        target = row["new_path"]
        if not os.path.exists(target):
            continue

        person_dir = by_person_root / f"person_{cid:04d}"
        link = person_dir / Path(target).name
        if not dry_run:
            person_dir.mkdir(parents=True, exist_ok=True)
            if not link.exists() and not link.is_symlink():
                link.symlink_to(os.path.abspath(target))
        symlinks_created += 1

    print(f"  Symlinks created : {symlinks_created}")
    if dry_run:
        print("  [DRY-RUN] No changes written.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Re-create by_person symlinks with cluster size filter.")
    p.add_argument("--output", default=config.OUTPUT_DIR, help="Output directory (default: config.OUTPUT_DIR)")
    p.add_argument("--dry-run", action="store_true", help="Preview changes without writing anything")
    args = p.parse_args()
    db.init_db()
    run(output_dir=args.output, dry_run=args.dry_run)

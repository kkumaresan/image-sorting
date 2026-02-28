"""
Phase 2 — Exact Duplicate Removal

Group images by SHA256; within each duplicate group keep the file with
the largest resolution (tie-break: earliest date_taken), delete the rest.
"""

import os

import config
import db


def run(dry_run: bool = False) -> None:
    print("\n[Phase 2] Exact duplicate removal ...")
    conn = db.get_conn()

    # Find SHA256 values that appear more than once among surviving images
    dup_hashes = conn.execute("""
        SELECT sha256
        FROM images
        WHERE status = 'ok' AND sha256 IS NOT NULL
        GROUP BY sha256
        HAVING COUNT(*) > 1
    """).fetchall()

    if not dup_hashes:
        print("  No exact duplicates found.")
        return

    groups_found = len(dup_hashes)
    files_deleted = 0

    for row in dup_hashes:
        sha = row["sha256"]
        candidates = conn.execute("""
            SELECT id, path, width, height, date_taken
            FROM images
            WHERE sha256 = ? AND status = 'ok'
            ORDER BY
                (COALESCE(width, 0) * COALESCE(height, 0)) DESC,
                date_taken ASC NULLS LAST
        """, (sha,)).fetchall()

        # Keep the first (highest-res / earliest); delete the rest
        keeper = candidates[0]
        to_delete = candidates[1:]

        for dup in to_delete:
            if not dry_run:
                try:
                    os.remove(dup["path"])
                except FileNotFoundError:
                    pass  # already gone
                conn.execute(
                    "UPDATE images SET status='deleted_exact_dup' WHERE id=?",
                    (dup["id"],),
                )
            files_deleted += 1

    if not dry_run:
        conn.commit()

    print(f"  Duplicate groups found : {groups_found:>6,}")
    print(f"  Files deleted          : {files_deleted:>6,}")
    if dry_run:
        print("  [DRY-RUN] No files were actually deleted.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    db.init_db()
    run(dry_run=args.dry_run)

#!/usr/bin/env python3
"""
run_pipeline.py — Orchestrator for the image recovery cleanup pipeline.

Usage:
    python run_pipeline.py --source /path/to/images --output ./output
    python run_pipeline.py --source /path/to/images --phase 1         # single phase
    python run_pipeline.py --source /path/to/images --dry-run         # simulate only
"""

import argparse
import sys
import time

import config
import db


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Image recovery cleanup & organisation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recommended: dry-run first to review what would be deleted
  python run_pipeline.py --source /media/disk/recovered --dry-run

  # Run all 5 phases
  python run_pipeline.py --source /media/disk/recovered --output ./output

  # Re-run only phase 3
  python run_pipeline.py --source /media/disk/recovered --phase 3
        """,
    )
    p.add_argument(
        "--source", required=True,
        help="Flat directory containing recovered JPG files",
    )
    p.add_argument(
        "--output", default="./output",
        help="Root of the organised output directory tree (default: ./output)",
    )
    p.add_argument(
        "--db", default="images.db",
        help="SQLite database path (default: images.db)",
    )
    p.add_argument(
        "--phase", type=int, choices=[1, 2, 3, 4, 5],
        help="Run a single phase (1-5) instead of all phases",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help=(
            "Simulate all destructive operations (no files deleted/moved, "
            "DB still updated with status). Recommended for first run."
        ),
    )
    return p.parse_args()


def _hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s"


def main() -> None:
    args = parse_args()

    # ---- Propagate settings into config ----
    config.SOURCE_DIR = args.source
    config.OUTPUT_DIR = args.output
    config.DB_PATH = args.db
    config.EMBEDDING_DIR = f"{args.output}/embeddings"

    if args.dry_run:
        print("=" * 60)
        print("  DRY-RUN MODE — no files will be deleted or moved")
        print("=" * 60)

    print(f"Source   : {args.source}")
    print(f"Output   : {args.output}")
    print(f"Database : {args.db}")
    print(f"Device   : {config.DEVICE}")

    # Initialise DB (idempotent)
    db.init_db(args.db)

    pipeline_start = time.time()

    phases_to_run = [args.phase] if args.phase else [1, 2, 3, 4, 5]

    for phase_num in phases_to_run:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  PHASE {phase_num}")
        print(f"{'='*60}")

        if phase_num == 1:
            import phase1_scan
            phase1_scan.run(source_dir=args.source, dry_run=args.dry_run)

        elif phase_num == 2:
            import phase2_dedup
            phase2_dedup.run(dry_run=args.dry_run)

        elif phase_num == 3:
            import phase3_near_dedup
            phase3_near_dedup.run(dry_run=args.dry_run)

        elif phase_num == 4:
            import phase4_classify
            phase4_classify.run(dry_run=args.dry_run)

        elif phase_num == 5:
            import phase5_organize
            phase5_organize.run(output_dir=args.output, dry_run=args.dry_run)

        elapsed = time.time() - t0
        print(f"\n  Phase {phase_num} finished in {_hms(elapsed)}")

    # Final summary
    print(f"\n{'='*60}")
    print("  PIPELINE COMPLETE")
    print(f"  Total time: {_hms(time.time() - pipeline_start)}")
    print(f"{'='*60}")
    db.print_summary(args.db)

    if args.dry_run:
        print(
            "\nDry-run complete.  Review the summary above, then re-run "
            "without --dry-run to commit changes."
        )


if __name__ == "__main__":
    main()

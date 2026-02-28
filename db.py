"""
SQLite schema definition and helper utilities.
All pipeline phases import get_conn() from here.
"""

import sqlite3
import threading
from contextlib import contextmanager
from typing import Generator

import config

# Thread-local storage for per-thread connections (used in Phase 1 parallel scan)
_local = threading.local()


def get_conn(db_path: str = config.DB_PATH) -> sqlite3.Connection:
    """Return a per-thread SQLite connection (auto-created on first call)."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(db_path, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA synchronous=NORMAL")
    return _local.conn


@contextmanager
def transaction(db_path: str = config.DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a connection and commits/rolls back."""
    conn = get_conn(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db(db_path: str = config.DB_PATH) -> None:
    """Create tables if they don't exist."""
    conn = get_conn(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            path            TEXT    NOT NULL UNIQUE,   -- original absolute path
            sha256          TEXT,
            phash           TEXT,                      -- 64-bit hex dHash
            width           INTEGER,
            height          INTEGER,
            date_taken      TEXT,                      -- ISO-8601 from EXIF
            camera_make     TEXT,
            camera_model    TEXT,
            category        TEXT,                      -- CLIP top-1 label
            face_cluster_id INTEGER DEFAULT -1,
            status          TEXT    NOT NULL DEFAULT 'ok',
                                                       -- ok | deleted_tiny |
                                                       -- deleted_corrupt |
                                                       -- deleted_exact_dup |
                                                       -- deleted_near_dup
            embedding_path  TEXT,                      -- path to .npy DINOv2 embedding
            new_path        TEXT                       -- path after Phase 5 move
        );

        CREATE INDEX IF NOT EXISTS idx_sha256  ON images(sha256);
        CREATE INDEX IF NOT EXISTS idx_status  ON images(status);
        CREATE INDEX IF NOT EXISTS idx_phash   ON images(phash);
        CREATE INDEX IF NOT EXISTS idx_cluster ON images(face_cluster_id);
        CREATE INDEX IF NOT EXISTS idx_date    ON images(date_taken);
        CREATE INDEX IF NOT EXISTS idx_camera  ON images(camera_make, camera_model);
    """)
    conn.commit()


def upsert_image(row: dict, db_path: str = config.DB_PATH) -> None:
    """Insert or update an image record by path."""
    conn = get_conn(db_path)
    cols = ", ".join(row.keys())
    placeholders = ", ".join(["?"] * len(row))
    updates = ", ".join(f"{k}=excluded.{k}" for k in row if k != "path")
    conn.execute(
        f"""
        INSERT INTO images ({cols}) VALUES ({placeholders})
        ON CONFLICT(path) DO UPDATE SET {updates}
        """,
        list(row.values()),
    )


def mark_status(path: str, status: str, db_path: str = config.DB_PATH) -> None:
    """Update only the status field for a given path."""
    conn = get_conn(db_path)
    conn.execute("UPDATE images SET status=? WHERE path=?", (status, path))


def fetch_ok_images(db_path: str = config.DB_PATH) -> list[sqlite3.Row]:
    """Return all rows with status='ok'."""
    conn = get_conn(db_path)
    return conn.execute("SELECT * FROM images WHERE status='ok'").fetchall()


def print_summary(db_path: str = config.DB_PATH) -> None:
    """Print a status breakdown from the DB."""
    conn = get_conn(db_path)
    rows = conn.execute(
        "SELECT status, COUNT(*) AS cnt FROM images GROUP BY status ORDER BY cnt DESC"
    ).fetchall()
    total = sum(r["cnt"] for r in rows)
    print("\n=== Database Summary ===")
    for r in rows:
        print(f"  {r['status']:<25} {r['cnt']:>6}")
    print(f"  {'TOTAL':<25} {total:>6}")
    print()

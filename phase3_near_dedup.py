"""
Phase 3 — Near-Duplicate Detection

Two-stage approach:
  Stage A: perceptual hash (dHash) + BK-tree to find candidate pairs fast.
  Stage B: DINOv2 embeddings (GPU) to confirm pairs at cosine similarity ≥ 0.97.

Surviving images have their DINOv2 embeddings saved as .npy for Phase 4.
"""

import os
from pathlib import Path
from typing import Optional

import imagehash
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoFeatureExtractor

import config
import db


# ---------------------------------------------------------------------------
# BK-tree for Hamming distance search
# ---------------------------------------------------------------------------

class _BKNode:
    __slots__ = ("key", "path", "children")

    def __init__(self, key: int, path: str) -> None:
        self.key = key
        self.path = path
        self.children: dict[int, "_BKNode"] = {}


class BKTree:
    def __init__(self) -> None:
        self.root: Optional[_BKNode] = None

    @staticmethod
    def _dist(a: int, b: int) -> int:
        return bin(a ^ b).count("1")

    def add(self, key: int, path: str) -> None:
        node = _BKNode(key, path)
        if self.root is None:
            self.root = node
            return
        cur = self.root
        while True:
            d = self._dist(cur.key, key)
            if d == 0:
                return  # identical hash already in tree
            if d not in cur.children:
                cur.children[d] = node
                return
            cur = cur.children[d]

    def search(self, key: int, threshold: int) -> list[str]:
        """Return paths of all entries within Hamming distance <= threshold."""
        if self.root is None:
            return []
        results: list[str] = []
        stack = [self.root]
        while stack:
            cur = stack.pop()
            d = self._dist(cur.key, key)
            if d <= threshold:
                results.append(cur.path)
            lo, hi = d - threshold, d + threshold
            for dist, child in cur.children.items():
                if lo <= dist <= hi:
                    stack.append(child)
        return results


# ---------------------------------------------------------------------------
# Stage A — perceptual hash
# ---------------------------------------------------------------------------

def _compute_phashes(rows: list) -> dict[str, int]:
    """Return {path: int_hash} for all surviving images."""
    result: dict[str, int] = {}
    for row in tqdm(rows, desc="Stage A: dHash", unit="img"):
        try:
            h = imagehash.dhash(Image.open(row["path"]))
            # imagehash stores as a binary string; convert to int for BK-tree
            result[row["path"]] = int(str(h), 16)
        except Exception:
            result[row["path"]] = None
    return result


def _find_candidate_groups(phashes: dict[str, int]) -> list[list[str]]:
    """
    Build a BK-tree and query each image.
    Use Union-Find to collect connected components (near-dup groups).
    """
    # Build tree (skip None hashes)
    tree = BKTree()
    valid = {p: h for p, h in phashes.items() if h is not None}
    for path, h in valid.items():
        tree.add(h, path)

    # Union-Find
    parent: dict[str, str] = {p: p for p in valid}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for path, h in tqdm(valid.items(), desc="Stage A: BK-tree search", unit="img"):
        neighbours = tree.search(h, config.PHASH_THRESHOLD)
        for nb in neighbours:
            if nb != path:
                union(path, nb)

    # Group by root
    groups: dict[str, list[str]] = {}
    for path in valid:
        root = find(path)
        groups.setdefault(root, []).append(path)

    return [g for g in groups.values() if len(g) > 1]


# ---------------------------------------------------------------------------
# Stage B — DINOv2 embeddings
# ---------------------------------------------------------------------------

def _load_dinov2():
    print(f"  Loading DINOv2 model ({config.DINOV2_MODEL}) on {config.DEVICE} ...")
    processor = AutoFeatureExtractor.from_pretrained(config.DINOV2_MODEL)
    model = AutoModel.from_pretrained(config.DINOV2_MODEL, torch_dtype=config.TORCH_DTYPE)
    model.to(config.DEVICE)
    model.eval()
    return processor, model


@torch.no_grad()
def _embed_batch(paths: list[str], processor, model) -> np.ndarray:
    """Return (N, D) float32 embedding matrix for a list of image paths."""
    images = []
    valid_indices = []
    for i, p in enumerate(paths):
        try:
            images.append(Image.open(p).convert("RGB"))
            valid_indices.append(i)
        except Exception:
            pass

    if not images:
        return np.zeros((len(paths), 1), dtype=np.float32)

    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(config.DEVICE) for k, v in inputs.items()}
    outputs = model(**inputs)
    # CLS token embedding
    embs = outputs.last_hidden_state[:, 0, :].float().cpu().numpy()

    # Build full output array (zero rows for failed images)
    dim = embs.shape[1]
    out = np.zeros((len(paths), dim), dtype=np.float32)
    for out_idx, emb in zip(valid_indices, embs):
        norm = np.linalg.norm(emb)
        out[out_idx] = emb / norm if norm > 0 else emb
    return out


def _embed_all(paths: list[str], processor, model) -> dict[str, np.ndarray]:
    """Embed all images in batches; return {path: embedding_vector}."""
    result: dict[str, np.ndarray] = {}
    for i in tqdm(range(0, len(paths), config.DINOV2_BATCH), desc="Stage B: DINOv2 embed", unit="batch"):
        batch_paths = paths[i: i + config.DINOV2_BATCH]
        embs = _embed_batch(batch_paths, processor, model)
        for p, e in zip(batch_paths, embs):
            result[p] = e
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> None:
    print("\n[Phase 3] Near-duplicate detection ...")
    conn = db.get_conn()
    ok_rows = db.fetch_ok_images()

    if not ok_rows:
        print("  No surviving images to process.")
        return

    # ---- Stage A ----
    phashes = _compute_phashes(ok_rows)

    # Persist dHash to DB
    with db.transaction():
        for row in ok_rows:
            h = phashes.get(row["path"])
            if h is not None:
                conn.execute(
                    "UPDATE images SET phash=? WHERE path=?",
                    (format(h, "016x"), row["path"]),
                )

    candidate_groups = _find_candidate_groups(phashes)
    print(f"  Stage A candidate near-dup groups: {len(candidate_groups):,}")

    if not candidate_groups:
        print("  No near-duplicates found.")
        _save_all_embeddings(ok_rows, dry_run)
        return

    # Flatten candidate paths for DINOv2 embedding
    candidate_paths_set = {p for g in candidate_groups for p in g}

    # ---- Stage B — load model ----
    processor, model = _load_dinov2()

    # Embed all *surviving* images (embeddings reused in Phase 4)
    all_paths = [row["path"] for row in ok_rows]
    all_embeddings = _embed_all(all_paths, processor, model)

    # Save embeddings to disk
    os.makedirs(config.EMBEDDING_DIR, exist_ok=True)
    with db.transaction():
        for path, emb in tqdm(all_embeddings.items(), desc="Saving embeddings", unit="img"):
            safe_name = path.replace("/", "_").replace("\\", "_").lstrip("_") + ".npy"
            emb_path = os.path.join(config.EMBEDDING_DIR, safe_name)
            np.save(emb_path, emb)
            conn.execute("UPDATE images SET embedding_path=? WHERE path=?", (emb_path, path))

    # Confirm near-dups with cosine similarity
    deleted_count = 0
    groups_confirmed = 0

    for group in tqdm(candidate_groups, desc="Stage B: confirming near-dups", unit="group"):
        # Build similarity matrix within group
        group_embs = np.stack([all_embeddings[p] for p in group if p in all_embeddings])
        if len(group_embs) < 2:
            continue

        # Pairwise cosine (embeddings already L2-normalised)
        sim_matrix = group_embs @ group_embs.T

        # Build confirmed sub-groups via Union-Find
        sub_parent = {i: i for i in range(len(group))}

        def find(x):
            while sub_parent[x] != x:
                sub_parent[x] = sub_parent[sub_parent[x]]
                x = sub_parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                sub_parent[rb] = ra

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if sim_matrix[i, j] >= config.COSINE_THRESHOLD:
                    union(i, j)

        sub_groups: dict[int, list[int]] = {}
        for idx in range(len(group)):
            root = find(idx)
            sub_groups.setdefault(root, []).append(idx)

        for indices in sub_groups.values():
            if len(indices) < 2:
                continue
            groups_confirmed += 1

            # Keep highest resolution
            paths_in_group = [group[i] for i in indices]
            rows_in_group = conn.execute(
                f"SELECT path, width, height, date_taken FROM images WHERE path IN ({','.join('?'*len(paths_in_group))})",
                paths_in_group,
            ).fetchall()

            rows_sorted = sorted(
                rows_in_group,
                key=lambda r: (-(r["width"] or 0) * (r["height"] or 0), r["date_taken"] or ""),
            )
            keeper = rows_sorted[0]["path"]
            to_delete = [r["path"] for r in rows_sorted[1:]]

            for p in to_delete:
                if not dry_run:
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass
                    conn.execute(
                        "UPDATE images SET status='deleted_near_dup' WHERE path=?", (p,)
                    )
                deleted_count += 1

    if not dry_run:
        conn.commit()

    print(f"\n[Phase 3] Complete.")
    print(f"  Near-dup groups confirmed : {groups_confirmed:>6,}")
    print(f"  Files deleted             : {deleted_count:>6,}")
    if dry_run:
        print("  [DRY-RUN] No files were actually deleted.")


def _save_all_embeddings(ok_rows, dry_run: bool) -> None:
    """Embed and save all images when no near-dups found (still needed for Phase 4)."""
    processor, model = _load_dinov2()
    all_paths = [row["path"] for row in ok_rows]
    all_embeddings = _embed_all(all_paths, processor, model)
    os.makedirs(config.EMBEDDING_DIR, exist_ok=True)
    conn = db.get_conn()
    with db.transaction():
        for path, emb in tqdm(all_embeddings.items(), desc="Saving embeddings", unit="img"):
            safe_name = path.replace("/", "_").replace("\\", "_").lstrip("_") + ".npy"
            emb_path = os.path.join(config.EMBEDDING_DIR, safe_name)
            if not dry_run:
                np.save(emb_path, emb)
            conn.execute("UPDATE images SET embedding_path=? WHERE path=?", (emb_path, path))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    db.init_db()
    run(dry_run=args.dry_run)

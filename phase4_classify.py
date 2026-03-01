"""
Phase 4 — AI Classification & Face Clustering

4A: CLIP zero-shot scene classification → DB `category` field.
4B: YOLOv8 face detection + ArcFace embeddings + DBSCAN clustering
    → DB `face_cluster_id` field.

The two sub-tasks run sequentially to avoid VRAM pressure.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import config
import db


# ===========================================================================
# 4A — CLIP Scene Classification
# ===========================================================================

def _run_clip(ok_rows: list, dry_run: bool) -> None:
    from transformers import CLIPProcessor, CLIPModel

    print(f"\n  [4A] Loading CLIP model ({config.CLIP_MODEL}) ...")
    processor = CLIPProcessor.from_pretrained(config.CLIP_MODEL)
    model = CLIPModel.from_pretrained(config.CLIP_MODEL, torch_dtype=config.TORCH_DTYPE)
    model.to(config.DEVICE)
    model.eval()

    # Pre-compute text embeddings (done once)
    text_inputs = processor(
        text=config.CLIP_LABELS,
        return_tensors="pt",
        padding=True,
    )
    text_inputs = {k: v.to(config.DEVICE) for k, v in text_inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**text_inputs).pooler_output
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    paths = [row["path"] for row in ok_rows]
    conn = db.get_conn()

    for i in tqdm(range(0, len(paths), config.CLIP_BATCH), desc="4A CLIP classify", unit="batch"):
        batch_paths = paths[i: i + config.CLIP_BATCH]
        images = []
        valid_paths = []
        for p in batch_paths:
            try:
                images.append(Image.open(p).convert("RGB"))
                valid_paths.append(p)
            except Exception:
                pass

        if not images:
            continue

        with torch.no_grad():
            img_inputs = processor(images=images, return_tensors="pt")
            img_inputs = {k: v.to(config.DEVICE) for k, v in img_inputs.items()}
            img_features = model.get_image_features(**img_inputs).pooler_output
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)

            # (N_images, N_labels)
            logits = (img_features.float() @ text_features.float().T) * 100
            probs = logits.softmax(dim=-1).cpu().numpy()

        if not dry_run:
            with db.transaction():
                for path, prob in zip(valid_paths, probs):
                    label = config.CLIP_LABELS[int(np.argmax(prob))]
                    conn.execute("UPDATE images SET category=? WHERE path=?", (label, path))

    print(f"  [4A] CLIP classification complete.")

    # Free VRAM
    del model
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()


# ===========================================================================
# 4B — Face Detection & Clustering
# ===========================================================================

def _run_face_clustering(ok_rows: list, dry_run: bool) -> None:
    # Force TensorFlow (DeepFace/ArcFace) onto CPU — libdevice missing prevents
    # GPU JIT compilation of BatchNorm, causing every embedding to fail.
    # Using TF's own API so PyTorch/YOLO GPU access is unaffected.
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    from ultralytics import YOLO
    from deepface import DeepFace
    from sklearn.cluster import DBSCAN

    print(f"\n  [4B] Loading YOLOv8 face detection model ...")
    try:
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(
            repo_id=config.YOLO_FACE_MODEL, filename="model.pt"
        )
        yolo = YOLO(weights_path)
    except Exception:
        # Fallback: local weights file in project root
        yolo = YOLO("yolov8n-face.pt")

    # Skip images already processed by a previous run
    unscanned = [row for row in ok_rows if not row["face_scanned"]]
    n_skip = len(ok_rows) - len(unscanned)
    if n_skip:
        print(f"  [4B] Skipping {n_skip:,} already-scanned images.")

    # Use new_path when files have been moved by Phase 5
    path_pairs = []  # (detect_path, original_db_path)
    for row in unscanned:
        orig = row["path"]
        moved = row["new_path"]
        detect = moved if (moved and os.path.exists(moved)) else orig
        path_pairs.append((detect, orig))
    detect_to_orig = {detect: orig for detect, orig in path_pairs}

    conn = db.get_conn()

    face_embeddings: list[np.ndarray] = []
    face_image_paths: list[str] = []  # original DB path for each face
    scanned_orig_paths: list[str] = []  # all originals attempted (success or fail)

    n_yolo_errors = 0
    n_imgs_with_faces = 0
    n_arcface_errors = 0

    print("  [4B] Detecting faces ...")
    for detect_path, orig_path in tqdm(path_pairs, desc="4B face detect", unit="img"):
        scanned_orig_paths.append(orig_path)
        try:
            results = yolo.predict(detect_path, conf=config.FACE_CONF_THRESHOLD, verbose=False)
        except Exception as ye:
            if n_yolo_errors == 0:
                print(f"  [4B] First YOLO error — path={detect_path!r} exists={os.path.exists(detect_path)} err={type(ye).__name__}: {ye}")
            n_yolo_errors += 1
            continue

        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            continue

        n_imgs_with_faces += 1
        img = np.array(Image.open(detect_path).convert("RGB"))
        h_img, w_img = img.shape[:2]

        for box in boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box[:4])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = img[y1:y2, x1:x2]

            # ArcFace embedding via deepface
            try:
                rep = DeepFace.represent(
                    img_path=face_crop,
                    model_name=config.ARCFACE_MODEL,
                    enforce_detection=False,
                    detector_backend="skip",
                )
                emb = np.array(rep[0]["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 0:
                    emb /= norm
                face_embeddings.append(emb)
                face_image_paths.append(detect_to_orig[detect_path])
            except Exception as e:
                if n_arcface_errors == 0:
                    print(f"  [4B] First ArcFace error: {type(e).__name__}: {e}")
                n_arcface_errors += 1

    print(f"  [4B] YOLO errors: {n_yolo_errors} | images with faces: {n_imgs_with_faces}"
          f" | ArcFace errors: {n_arcface_errors} | embeddings: {len(face_embeddings)}")

    # Mark all attempted images as scanned (regardless of whether faces were found)
    if not dry_run and scanned_orig_paths:
        with db.transaction():
            for orig_path in scanned_orig_paths:
                conn.execute(
                    "UPDATE images SET face_scanned=1 WHERE path=?", (orig_path,)
                )

    if not face_embeddings:
        print("  [4B] No faces detected.")
        return

    print(f"  [4B] {len(face_embeddings):,} face embeddings collected. Running DBSCAN ...")

    X = np.stack(face_embeddings)

    # cosine distance = 1 - dot product for L2-normed embeddings.
    # Pass metric directly to DBSCAN to avoid materializing the O(n²) distance
    # matrix (4.5 GB for 23k embeddings) — sklearn computes distances on-the-fly.
    clustering = DBSCAN(
        eps=config.DBSCAN_EPS,
        min_samples=config.DBSCAN_MIN_SAMPLES,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    labels = clustering.fit_predict(X)

    n_clusters = int(labels.max()) + 1
    print(f"  [4B] Found {n_clusters} face clusters (label -1 = noise/unassigned).")

    # Assign the *most common* cluster per image (an image may have multiple faces)
    from collections import Counter

    image_clusters: dict[str, list[int]] = {}
    for path, label in zip(face_image_paths, labels):
        image_clusters.setdefault(path, []).append(int(label))

    image_cluster_map: dict[str, int] = {}
    for path, cluster_labels in image_clusters.items():
        non_noise = [c for c in cluster_labels if c >= 0]
        if non_noise:
            image_cluster_map[path] = Counter(non_noise).most_common(1)[0][0]
        else:
            image_cluster_map[path] = -1

    # Filter: discard clusters with fewer than FACE_MIN_CLUSTER_SIZE images
    cluster_counts = Counter(c for c in image_cluster_map.values() if c >= 0)
    valid_clusters = {c for c, n in cluster_counts.items()
                      if n >= config.FACE_MIN_CLUSTER_SIZE}
    n_small = len(cluster_counts) - len(valid_clusters)
    print(f"  [4B] {len(valid_clusters)} person clusters (>= {config.FACE_MIN_CLUSTER_SIZE}"
          f" images) kept; {n_small} small cluster(s) discarded as noise.")

    if not dry_run:
        with db.transaction():
            for path, cluster_id in image_cluster_map.items():
                if cluster_id not in valid_clusters:
                    cluster_id = -1
                conn.execute(
                    "UPDATE images SET face_cluster_id=? WHERE path=?",
                    (cluster_id, path),
                )

    print(f"  [4B] Face clustering complete.")


# ===========================================================================
# Public entry point
# ===========================================================================

def run(dry_run: bool = False) -> None:
    print("\n[Phase 4] Classification & face clustering ...")
    ok_rows = db.fetch_ok_images()

    if not ok_rows:
        print("  No surviving images to process.")
        return

    _run_clip(ok_rows, dry_run)
    _run_face_clustering(ok_rows, dry_run)

    print("\n[Phase 4] Complete.")
    if dry_run:
        print("  [DRY-RUN] DB was not updated.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--faces-only", action="store_true",
                   help="Skip CLIP classification, run only face detection & clustering")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N images (for testing)")
    args = p.parse_args()
    db.init_db()
    ok_rows = db.fetch_ok_images()
    if args.limit:
        ok_rows = ok_rows[:args.limit]
        print(f"  [--limit] Processing {args.limit} images only.")
    if args.faces_only:
        _run_face_clustering(ok_rows, dry_run=args.dry_run)
    else:
        run(dry_run=args.dry_run)

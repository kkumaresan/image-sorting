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
        text_features = model.get_text_features(**text_inputs)
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
            img_features = model.get_image_features(**img_inputs)
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
    from ultralytics import YOLO
    from deepface import DeepFace
    from sklearn.cluster import DBSCAN

    print(f"\n  [4B] Loading YOLOv8 face detection model ...")
    # The HuggingFace model ID needs to be downloaded first; ultralytics accepts
    # a local path or a hub model name.  We use the HF hub path via ultralytics:
    try:
        yolo = YOLO(config.YOLO_FACE_MODEL)
    except Exception:
        # Fallback: try treating it as a local weights file name
        yolo = YOLO("yolov8n-face.pt")

    paths = [row["path"] for row in ok_rows]
    conn = db.get_conn()

    face_embeddings: list[np.ndarray] = []
    face_image_paths: list[str] = []  # which image each face came from

    print("  [4B] Detecting faces ...")
    for path in tqdm(paths, desc="4B face detect", unit="img"):
        try:
            results = yolo.predict(path, conf=config.FACE_CONF_THRESHOLD, verbose=False)
        except Exception:
            continue

        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            continue

        img = np.array(Image.open(path).convert("RGB"))
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
                face_image_paths.append(path)
            except Exception:
                pass

    if not face_embeddings:
        print("  [4B] No faces detected.")
        return

    print(f"  [4B] {len(face_embeddings):,} face embeddings collected. Running DBSCAN ...")

    X = np.stack(face_embeddings)

    # cosine distance = 1 - cosine_similarity (embeddings are L2-normed)
    from sklearn.metrics.pairwise import cosine_distances
    dist_matrix = cosine_distances(X).astype(np.float64)

    clustering = DBSCAN(
        eps=config.DBSCAN_EPS,
        min_samples=config.DBSCAN_MIN_SAMPLES,
        metric="precomputed",
        n_jobs=-1,
    )
    labels = clustering.fit_predict(dist_matrix)

    n_clusters = int(labels.max()) + 1
    print(f"  [4B] Found {n_clusters} face clusters (label -1 = noise/unassigned).")

    # Assign the *most common* cluster per image (an image may have multiple faces)
    from collections import Counter

    image_clusters: dict[str, list[int]] = {}
    for path, label in zip(face_image_paths, labels):
        image_clusters.setdefault(path, []).append(int(label))

    if not dry_run:
        with db.transaction():
            for path, cluster_labels in image_clusters.items():
                # Most common cluster; ignore -1 (noise) if other clusters present
                non_noise = [c for c in cluster_labels if c >= 0]
                if non_noise:
                    cluster_id = Counter(non_noise).most_common(1)[0][0]
                else:
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
    args = p.parse_args()
    db.init_db()
    run(dry_run=args.dry_run)

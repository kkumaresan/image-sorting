import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TF onto CPU
sys.path.insert(0, "/home/kumar/Work/img-project/RECOVERED/image-sorting")
os.chdir("/home/kumar/Work/img-project/RECOVERED/image-sorting")
print("Python:", sys.executable)

try:
    import db, config, numpy as np
    from PIL import Image
    from deepface import DeepFace
    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download
    print("All imports OK")
except Exception as e:
    print(f"Import error: {type(e).__name__}: {e}")
    sys.exit(1)

db.init_db()
rows = db.fetch_ok_images()[:100]
print(f"Rows fetched: {len(rows)}")

weights = hf_hub_download(repo_id=config.YOLO_FACE_MODEL, filename="model.pt")
yolo = YOLO(weights)
print("YOLO loaded")

found = 0
for row in rows:
    path = row["new_path"] or row["path"]
    try:
        results = yolo.predict(path, conf=config.FACE_CONF_THRESHOLD, verbose=False)
    except Exception as e:
        print(f"YOLO error: {e}")
        continue
    boxes = results[0].boxes if results else None
    if boxes is None or len(boxes) == 0:
        continue
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    for box in boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = map(int, box[:4])
        crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
        print(f"Face crop shape: {crop.shape}")
        try:
            rep = DeepFace.represent(
                img_path=crop,
                model_name=config.ARCFACE_MODEL,
                enforce_detection=False,
                detector_backend="skip",
            )
            print("SUCCESS embedding len:", len(rep[0]["embedding"]))
        except Exception as e:
            print(f"ARCFACE ERROR: {type(e).__name__}: {e}")
        found += 1
        if found >= 3:
            break
    if found >= 3:
        break

print(f"Done. Faces tested: {found}")

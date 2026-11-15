"""Prepare a fixed YOLO-cropped dataset and repeat the PC experiments."""
import argparse
import hashlib
from pathlib import Path

import extend

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT / "cropped_reproduction"


def prepare():
    from PIL import Image
    from ultralytics import YOLO
    from src.data import validate_manifest, write_csv, write_json
    from src.image_experiments import image

    rows, meta = validate_manifest(PROJECT)
    detector = YOLO(str(PROJECT / ".cache/yolo11m.pt"))
    prepared, audit = [], []
    for index, row in enumerate(rows):
        rgb = image(PROJECT, row)
        height, width = rgb.shape[:2]
        boxes = detector.predict(Image.fromarray(rgb), classes=[2, 5, 7],
                                 imgsz=640, conf=0.25, device=0, verbose=False)[0].boxes
        bounds = ()
        if len(boxes):
            x1, y1, x2, y2 = map(int, boxes.xyxy[boxes.conf.argmax().item()].tolist())
            bounds = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
        valid = bool(bounds and bounds[0] < bounds[2] and bounds[1] < bounds[3])
        source = Image.fromarray(rgb)
        cropped = source.crop(bounds) if valid else source
        path = f"data/images/{index:04d}.png"
        destination = ROOT / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(destination)
        digest = hashlib.sha256(str(cropped.size).encode() + cropped.tobytes()).hexdigest()
        prepared.append({**row, "path": path, "sha256_pixels": digest})
        audit.append({"original": row["path"], "path": path, "split": row["split"],
                      "status": "cropped" if valid else "full_frame_fallback", "xyxy": str(bounds)})
        if (index + 1) % 100 == 0:
            print(f"YOLO {index + 1}/1000", flush=True)
    write_csv(ROOT / "data/crop_audit.csv", audit, list(audit[0]))
    write_json(ROOT / "data/metadata.json", {**meta, "interpretation":
               "CompCars + YOLO11m crop; full frame fallback on detection failure"})
    write_csv(ROOT / "data/manifest.csv", prepared, list(prepared[0]))
    validate_manifest(ROOT)
    print(f"Cropped {sum(r['status'] == 'cropped' for r in audit)}/1000", flush=True)


def run():
    import json
    from src.data import validate_manifest
    from src.extended_models import BASELINES, NORMALIZATION, PADDING, VARIANTS
    from src.image_experiments import robustness

    validate_manifest(ROOT)
    cfg = json.loads((PROJECT / "configs/paper.json").read_text(encoding="utf-8-sig"))
    extend.ROOT = ROOT
    extend.OUT = ROOT / "outputs"
    for name in VARIANTS:
        extend.run_model(name, cfg, False)
    extend.OUT = ROOT / "extended_results"
    extend.OUT.mkdir(exist_ok=True)
    for name in (*BASELINES, *PADDING, *NORMALIZATION):
        extend.run_model(name, cfg, False)
        extend.report()
    robustness(ROOT, extend.OUT, cfg)
    extend.report()


if __name__ == "__main__":
    import msvcrt

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "all"))
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    with (ROOT / ".run.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        if not (ROOT / "data/manifest.csv").exists():
            prepare()
        if args.mode == "all":
            run()

"""Restore the fixed CompCars files from an official, locally extracted copy."""
import argparse
import ast
import csv
import hashlib
from pathlib import Path
import shutil

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def digest(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def restore(source):
    manifest = rows(ROOT / "data/manifest.csv")
    cropped_root = ROOT / "cropped_reproduction"
    audit = rows(cropped_root / "data/crop_audit.csv")
    cropped = rows(cropped_root / "data/manifest.csv")
    if (len(manifest), len(audit), len(cropped)) != (1000, 1000, 1000):
        raise ValueError("固定資料清單必須包含 1,000 張。")
    for original, record, expected in zip(manifest, audit, cropped, strict=True):
        if (original["path"], expected["path"], original["split"]) != (
            record["original"], record["path"], record["split"]
        ) or any(original[key] != expected[key] for key in ("label", "make", "split")):
            raise ValueError("裁切對照與原始切分不一致。")
        target = ROOT / original["path"]
        incoming = source / Path(original["path"]).relative_to("data/raw/sv_data")
        if not incoming.is_file():
            raise FileNotFoundError(f"找不到 {incoming}，請指定包含 image 子資料夾的 sv_data。")
        with Image.open(incoming) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
        if digest(rgb) != original["sha256_pixels"]:
            raise ValueError(f"原圖像素雜湊不同：{incoming}")
        if record["status"] == "cropped":
            rgb = rgb.crop(ast.literal_eval(record["xyxy"]))
        elif record["status"] != "full_frame_fallback":
            raise ValueError(f"未知裁切狀態：{record['status']}")
        if digest(rgb) != expected["sha256_pixels"]:
            raise ValueError(f"裁切圖像素雜湊不同：{expected['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if incoming.resolve() != target.resolve():
            shutil.copyfile(incoming, target)
        destination = cropped_root / expected["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(destination)
    from src.data import validate_manifest
    validate_manifest(ROOT)
    validate_manifest(cropped_root)
    print("原圖及裁切圖各 1,000 張通過像素與 800／200 切分驗證。")


def demo():
    import torch
    from src.data import image_tensor
    from src.engine import load_checkpoint
    from src.models import CTv1

    row = next(row for row in rows(ROOT / "data/manifest.csv") if row["split"] == "test")
    image = ROOT / row["path"]
    ck = load_checkpoint(ROOT / "outputs/ctv1/weights.pt")
    model = CTv1("ctv1", ck["config"]).eval()
    model.load_state_dict(ck["model"])
    with torch.inference_mode():
        index = model(image_tensor(image, ck["config"]).unsqueeze(0)).argmax(1).item()
    print(f"測試圖：{row['path']}\n真實品牌：{row['make']}\n預測品牌：{ck['classes'][index]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", nargs="?", type=Path, help="官方解壓後的 sv_data 資料夾")
    parser.add_argument("--demo", action="store_true", help="以 CPU 執行固定測試圖示範")
    args = parser.parse_args()
    if args.demo:
        demo()
    elif args.source:
        restore(args.source)
    else:
        parser.error("請指定 sv_data 路徑，或使用 --demo。")

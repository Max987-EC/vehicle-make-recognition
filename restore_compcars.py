"""Restore the fixed CompCars files from an official, locally extracted copy."""
import argparse
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
    if len(manifest) != 1000:
        raise ValueError("Expected 1,000 fixed images.")
    for original in manifest:
        target = ROOT / original["path"]
        incoming = source / Path(original["path"]).relative_to("data/raw/sv_data")
        if not incoming.is_file():
            raise FileNotFoundError(f"找不到 {incoming}，請指定包含 image 子資料夾的 sv_data。")
        with Image.open(incoming) as opened:
            rgb = ImageOps.exif_transpose(opened).convert("RGB")
        if digest(rgb) != original["sha256_pixels"]:
            raise ValueError(f"原圖像素雜湊不同：{incoming}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if incoming.resolve() != target.resolve():
            shutil.copyfile(incoming, target)
    from src.data import validate_manifest
    validate_manifest(ROOT)
    print("Verified 1,000 original images and the fixed 800/200 split.")


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

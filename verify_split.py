"""Replay the original split using only the 1,000 selected images' records."""
import argparse
import csv
import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description="驗證固定 1,000 張圖片的原始切分")
    parser.add_argument("--output", type=Path, help="選填：另存重建清單的全新 CSV 路徑")
    args = parser.parse_args()
    with (ROOT / "data/manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = reader.fieldnames
    replay = json.loads((ROOT / "data/split_replay.json").read_text(encoding="utf-8"))
    paths = replay["pre_shuffle_paths"]
    by_path = {row["path"]: row for row in rows}
    if len(rows) != 1000 or len(paths) != 1000 or len(set(paths)) != 1000 or set(paths) != by_path.keys():
        parser.exit(1, "固定圖片清單與打亂前的清單不一致。\n")
    version, state, gaussian = replay["random_state"]
    generator = random.Random()
    generator.setstate((version, tuple(state), gaussian))
    generator.shuffle(paths)
    if replay["train_count"] != 800:
        parser.exit(1, "原始切分必須保留 800 張訓練圖。\n")
    reconstructed = [
        {**by_path[path], "split": "train" if index < 800 else "test"}
        for index, path in enumerate(paths)
    ]
    if reconstructed != rows:
        parser.exit(1, "重建的順序或切分與正式清單不一致。\n")
    if args.output:
        with args.output.open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(reconstructed)
    print("原始順序與切分驗證通過：800 張訓練圖、200 張測試圖。")


if __name__ == "__main__":
    main()

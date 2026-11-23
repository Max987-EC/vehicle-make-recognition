"""Test-only image perturbations and vehicle detection demonstration."""
import csv
import hashlib

import cv2
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import f1_score, accuracy_score
import torch

from .data import validate_manifest
from .engine import load_checkpoint
from .models import CTv1

LEVELS = {"blur": range(1, 11), "gaussian": range(0, 101, 10),
          "salt_pepper": (0, .0025, .005, .0075, .01, .0125, .015, .0175, .018), "darkness": range(0, 101, 10),
          "drops": (0, *range(2, 12))}


def perturb(rgb, kind, level, seed):
    if level == 0 or (kind == "blur" and level == 1):
        return rgb.copy()
    rng = np.random.default_rng(seed)
    if kind == "blur":
        return cv2.blur(rgb, (level, level))
    if kind == "darkness":
        return np.clip(rgb.astype(np.int16) - level, 0, 255).astype(np.uint8)
    if kind == "gaussian":
        return np.clip(rgb + rng.normal(0, level, rgb.shape), 0, 255).astype(np.uint8)
    if kind == "salt_pepper":
        mask = rng.random(rgb.shape[:2])
        result = rgb.copy()
        result[mask < level / 2] = 0
        result[mask > 1 - level / 2] = 255
        return result
    if kind == "drops":
        h, w = rgb.shape[:2]
        scale = min(h, w) / 600
        mask = np.zeros((h, w), np.uint8)
        for _ in range(level):
            radius = max(1, round(int(rng.integers(10, 31)) * scale / 2))
            cv2.circle(mask, (int(rng.integers(w)), int(rng.integers(h))), radius, 255, -1)
        frosted = cv2.GaussianBlur(rgb, (0, 0), max(1, 5 * scale))
        result = rgb.copy()
        result[mask > 0] = np.round(.65 * frosted[mask > 0] + .35 * 255).astype(np.uint8)
        return result
    raise ValueError(kind)


def load_context(root):
    checkpoint = load_checkpoint(root / "outputs/ctv1/weights.pt")
    model = CTv1("ctv1", checkpoint["config"]).cuda().eval()
    model.load_state_dict(checkpoint["model"])
    rows, meta = validate_manifest(root)
    return model, [r for r in rows if r["split"] == "test"], meta["classes"]


def tensor(rgb):
    resized = Image.fromarray(rgb).resize((244, 244), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(resized, dtype=np.float32) / 255).permute(2, 0, 1)


def image(root, row):
    with Image.open(root / row["path"]) as source:
        return np.array(ImageOps.exif_transpose(source).convert("RGB"))


def write_rows(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def robustness(root, out, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model, rows, classes = load_context(root)
    target = out / "robustness"
    target.mkdir(exist_ok=True)
    truth = [int(r["label"]) for r in rows]
    originals = [image(root, row) for row in rows]
    reference = None
    metrics, predictions = [], []
    for kind, levels in {"clean": (0,), **LEVELS}.items():
        print(f"干擾測試：{kind}", flush=True)
        for level in levels:
            predicted = []
            for start in range(0, len(rows), 25):
                batch = []
                for i, row in enumerate(rows[start:start + 25], start):
                    rgb = originals[i]
                    seed = int.from_bytes(hashlib.sha256(f'{cfg["seed"]}:{kind}:{i}'.encode()).digest()[:8], "little")
                    changed = perturb(rgb, kind, level, seed)
                    if i == 0 and level == max(levels):
                        Image.fromarray(changed).save(target / f"{kind}-example.png")
                    batch.append(tensor(changed))
                predicted.extend(model(torch.stack(batch).cuda()).argmax(1).tolist())
            if reference is None:
                reference = predicted
                with (root / "outputs/ctv1/predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
                    original = list(csv.DictReader(stream))
                if [(r["path"], classes[p]) for r, p in zip(rows, predicted)] != [
                    (r["path"], r["predicted"]) for r in original
                ]:
                    raise RuntimeError("Clean predictions differ from the original CTv1 evaluation")
            if (level == 0 or (kind == "blur" and level == 1)) and predicted != reference:
                raise RuntimeError(f"{kind}: zero perturbation differs from clean")
            metrics.append(dict(kind=kind, level=level, accuracy=accuracy_score(truth, predicted),
                                weighted_f1=f1_score(truth, predicted, average="weighted")))
            predictions.extend(dict(kind=kind, level=level, path=r["path"], truth=r["make"],
                                    predicted=classes[p]) for r, p in zip(rows, predicted))
    write_rows(target / "metrics.csv", metrics)
    write_rows(target / "predictions.csv", predictions)
    for kind in LEVELS:
        points = [r for r in metrics if r["kind"] == kind]
        fig, ax = plt.subplots()
        ax.plot([r["level"] for r in points], [r["weighted_f1"] for r in points], marker="o")
        ax.axhline(metrics[0]["weighted_f1"], color="gray", linestyle="--", label="Clean")
        ax.set(xlabel=kind, ylabel="Weighted F1", ylim=(0, 1))
        ax.legend()
        fig.savefig(target / f"{kind}.png", dpi=150)
        plt.close(fig)

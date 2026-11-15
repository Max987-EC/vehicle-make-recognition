"""Run and compare the three fixed-data experiments from VS Code or a shell."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from uuid import uuid4

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "reproduction"
BASELINE = ROOT / "outputs"
MODELS = ("ctv1", "ctv1_dw", "ctv1_dr")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def predictions(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def compare(directory, mode):
    lines = ["# 三模型結果比對", "", f"執行方式：{mode}"]
    differences = []
    for model in MODELS:
        old_path, new_path = BASELINE / model, directory / model
        old, new = (read_json(path / "metrics.json") for path in (old_path, new_path))
        for key in ("manifest_sha256", "parameters"):
            if old[key] != new[key]:
                raise ValueError(f"{model}: {key} 與原始實驗不一致")
        before, after = (predictions(path / "predictions.csv") for path in (old_path, new_path))
        if len(before) != 200 or [(r["path"], r["truth"]) for r in before] != [
            (r["path"], r["truth"]) for r in after
        ]:
            raise ValueError(f"{model}: 測試圖片或標籤不一致")
        changed = [
            (model, a["path"], a["truth"], a["predicted"], b["predicted"])
            for a, b in zip(before, after) if a["predicted"] != b["predicted"]
        ]
        differences.extend(changed)
        lines += ["", f"## {model}", "", f"逐圖預測差異：{len(changed)}／200",
                  "", "| 指標 | 原始 | 本次 | 差值 |", "|---|---:|---:|---:|"]
        def metric(label, a, b):
            lines.append(f"| {label} | {a:.10g} | {b:.10g} | {b-a:+.10g} |")
        for key in ("accuracy", "weighted_f1", "weights_bytes", "training_seconds",
                    "inference_seconds_per_image", "inference_mean_seconds_per_batch25",
                    "pipeline_seconds_per_image"):
            metric(key, old[key], new[key])
        for key, value in old["parameters"].items():
            metric(key, value, new["parameters"][key])
        for brand, values in old["classification_report"].items():
            if isinstance(values, dict):
                for key, value in values.items():
                    metric(f"{brand} {key}", value, new["classification_report"][brand][key])
        for label, key in (("混淆矩陣", "confusion_matrix"), ("運算環境", "environment")):
            lines += ["", f"**{label}**", "", f"- 原始：`{old[key]}`", f"- 本次：`{new[key]}`"]
    with (directory / "prediction-differences.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "path", "truth", "original", "reproduced"])
        writer.writerows(differences)
    lines += ["", "逐圖差異見 [prediction-differences.csv](prediction-differences.csv)。",
              "時間為各機器實測。驗證原權重時，訓練時間沿用原 checkpoint。"]
    (directory / "reproduction-report.md").write_text("\n".join(lines), encoding="utf-8")


def check():
    print("檢查固定圖片與原始切分", flush=True)
    subprocess.run([sys.executable, "-B", "-X", "utf8", str(ROOT / "verify_split.py")], check=True)
    from src.data import validate_manifest
    validate_manifest(ROOT)
    print("1,000 張圖片的像素雜湊與標籤驗證通過。", flush=True)


def run(mode):
    import msvcrt
    RUNS.mkdir(exist_ok=True)
    with (RUNS / ".run.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as error:
            raise ValueError("另一個重跑或驗證任務正在執行。") from error
        perform_run(mode)


def perform_run(mode):
    from src.engine import device_for, evaluate, fingerprint, load_checkpoint, report_all, train
    cfg = read_json(ROOT / "configs/paper.json")
    device_for(cfg)
    check()
    start = perf_counter()
    if mode == "resume":
        pending = sorted(d for d in RUNS.glob("train-*") if d.is_dir() and not (d / "complete.json").exists())
        if not pending:
            raise ValueError("沒有待接續的重跑。請先執行任務 04。")
        directory = pending[-1]
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        directory = RUNS / f"{mode}-{stamp}-{uuid4().hex[:6]}"
        directory.mkdir()
    print(f"輸出：{directory}", flush=True)
    for model in MODELS:
        print(f"{model}：{'複製原權重' if mode == 'verify' else '訓練或接續'}", flush=True)
        target = directory / model
        if mode == "verify":
            target.mkdir()
            for file in ("last.pt", "weights.pt"):
                shutil.copy2(BASELINE / model / file, target / file)
        else:
            checkpoint = target / "last.pt"
            finished = (target / "run.json").exists() and (target / "weights.pt").exists() and checkpoint.exists()
            if finished:
                saved = load_checkpoint(checkpoint)
                if saved["config"] != cfg or saved["manifest_sha256"] != fingerprint(ROOT / "data/manifest.csv"):
                    raise ValueError(f"{model}: 已完成的模型使用不同設定或資料切分")
                finished = saved["epoch"] == cfg["epochs"]
            if not finished:
                train(ROOT, cfg, model, resume=checkpoint.exists(), output_dir=directory)
        print(f"{model}：評估第 200 輪權重", flush=True)
        evaluate(ROOT, model, output_dir=directory)
    report_all(ROOT, output_dir=directory)
    compare(directory, mode)
    completion = directory / "complete.tmp"
    completion.write_text(
        json.dumps({"mode": mode, "this_invocation_seconds": perf_counter() - start}), encoding="utf-8"
    )
    completion.replace(directory / "complete.json")
    print(directory / "reproduction-report.md", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("environment", "check", "verify", "train", "resume", "report-path"))
    mode = parser.parse_args().mode
    try:
        if mode == "report-path":
            completed = sorted((d for d in RUNS.glob("train-*") if (d / "complete.json").exists()
                                and (d / "reproduction-report.md").exists()),
                               key=lambda d: (d / "complete.json").stat().st_mtime_ns)
            if not completed:
                raise ValueError("尚未完成任何重跑。請先執行任務 04 或 05。")
            print(completed[-1] / "reproduction-report.md")
        elif mode == "environment":
            from src.engine import device_for, environment
            cfg = read_json(ROOT / "configs/paper.json")
            print(json.dumps(environment(), ensure_ascii=False, indent=2))
            device = device_for(cfg)
            import torch
            x = torch.ones(8, device=device, requires_grad=True)
            x.square().sum().backward()
            print("GPU 前向與反向驗證通過。")
        elif mode == "check":
            check()
        else:
            run(mode)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"任務停止：{error}\n")


if __name__ == "__main__":
    main()

"""Run the paper's remaining computer experiments on the fixed dataset."""
import argparse
import csv
import os
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "extended_results"
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def preflight(name, cfg):
    import gc
    import torch
    from src.engine import seed_all, optimizer_for
    from src.extended_models import build
    from src.models import parameter_counts
    for batch in (32, 16, 8, 4, 2, 1):
        try:
            seed_all(cfg["seed"])
            torch.cuda.reset_peak_memory_stats()
            model = build(name, cfg).cuda()
            optimizer = optimizer_for(model, cfg)
            images = torch.rand(batch, 3, 244, 244, device="cuda")
            shapes = []
            for block in getattr(model, "blocks", []):
                block.register_forward_hook(lambda _, inputs, output: shapes.append(list(output.shape[1:])))
            logits = model(images)
            if logits.shape != (batch, 5):
                raise ValueError(f"{name}: unexpected logits {logits.shape}")
            torch.nn.functional.cross_entropy(logits, torch.zeros(batch, device="cuda", dtype=torch.long)).backward()
            optimizer.step()
            peak = torch.cuda.max_memory_allocated()
            del optimizer, images, logits
            state = model.cpu().state_dict()
            clone = build(name, cfg)
            clone.load_state_dict(state)
            if len(shapes) == 5:
                mode = name.rsplit("-", 1)[1]
                sizes = {"5l": [121, 60, 29, 14, 6],
                         "np": [120, 58, 27, 12, 4],
                         "st": [121, 60, 29, 7, 1]}[mode]
                expected = [[c, size, size] for c, size in zip((64, 128, 256, 512, 1024), sizes)]
                if shapes != expected:
                    raise ValueError(f"{name}: expected {expected}, got {shapes}")
            elif shapes and [shape[1] for shape in shapes] != [121, 60, 29, 14, 6, 2]:
                raise ValueError(f"{name}: unexpected six-block sizes: {shapes}")
            return {"batch_size": batch, "training_peak_bytes": peak,
                    "parameters": parameter_counts(clone), "block_shapes": shapes}
        except torch.cuda.OutOfMemoryError:
            if batch == 1:
                raise
        finally:
            model = optimizer = images = logits = None
            gc.collect()
            torch.cuda.empty_cache()


def report():
    from src.extended_models import BASELINES, NORMALIZATION, PADDING, VARIANTS
    paper = dict(zip((*BASELINES, *VARIANTS), (.23, .97, .81, .65, .92, .99, .90, .93)))
    paper.update(zip(PADDING, (.98, .98, .95, .90, .93, .78, .93, .93, .91)))
    paper.update(zip(NORMALIZATION, (.76, .74, .16)))
    paper_counts = dict(zip((*BASELINES, *VARIANTS),
                            (134281029, 21813029, 23597957, 3233989, 10348037, 6555909, 980137, 576245)))
    paper_counts.update({name: size for variant, size in zip(VARIANTS, (6538245, 977705, 570981))
                         for name in PADDING if name.startswith(variant + "-")})
    paper_training = dict(zip((*BASELINES, *VARIANTS),
                              (2045.67, 945.07, 1204.29, 427.71, 734.43, 354.96, 418.04, 435.59)))
    paper_training.update(zip(PADDING, (720.93, 631.43, 607.22, 674.12, 626.31, 622.43,
                                       805.04, 726.89, 720.71)))
    paper_training.update(zip(NORMALIZATION, (3354.51, 1511.66, 2113.53)))
    lines = ["# 電腦端延伸實驗", "", "資料：固定 CompCars，Volkswagen 替代 Isuzu；PyTorch，RTX 4060 Laptop。",
             "推論為每批 25 張的 GPU 前向攤提時間。", ""]
    for heading, names in (("表 2、3 基準比較", (*VARIANTS, *BASELINES)),
                           ("表 6、7 Padding／Stride", PADDING), ("表 8 BatchNorm", NORMALIZATION)):
        lines.extend([f"## {heading}", "", "|模型|論文 F1|本次 F1|準確率|論文參數|本次參數|權重 bytes|論文訓練秒|本次訓練秒|本次推論 ms／張|對照 F1 差|",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
        for name in names:
            source = ROOT / "outputs" if name in VARIANTS else OUT
            path = source / name / "metrics.json"
            if not path.exists():
                lines.append(f'|{name}|{paper[name]:.2f}|待完成||{paper_counts.get(name, "")}|||{paper_training[name]:.2f}||||')
                continue
            m = json.loads(path.read_text(encoding="utf8"))
            reference = None
            if name in NORMALIZATION:
                reference = ROOT / "outputs" / name.removesuffix("-no_bn") / "metrics.json"
            if name in PADDING and not name.endswith("-5l"):
                reference = OUT / (name.rsplit("-", 1)[0] + "-5l") / "metrics.json"
            delta = ""
            if reference and reference.exists():
                base = json.loads(reference.read_text(encoding="utf8"))
                delta = f'{m["weighted_f1"] - base["weighted_f1"]:+.4f}'
            lines.append(f'|{name}|{paper[name]:.2f}|{m["weighted_f1"]:.4f}|{m["accuracy"]:.4f}|'
                         f'{paper_counts.get(name, "")}|{m["parameters"]["keras_style_total"]}|{m["weights_bytes"]}|'
                         f'{paper_training[name]:.2f}|{m["training_seconds"]:.2f}|{1000*m["inference_seconds_per_image"]:.3f}|{delta}|')
        lines.append("")
    for name, label in (("robustness/metrics.csv", "影像干擾"), ("detection/results.csv", "YOLO 裁切")):
        lines.append(f"- [{label}]({name})" if (OUT / name).exists() else f"- {label}：待完成")
    detection_path = OUT / "detection/results.csv"
    if detection_path.exists():
        with detection_path.open(encoding="utf-8-sig", newline="") as stream:
            detection_rows = list(csv.DictReader(stream))
        success = sum(r["status"] == "ok" for r in detection_rows)
        lines.append(f"- YOLO 成功裁切 {success}／{len(detection_rows)} 張。")
    temporary = OUT / "report.tmp"
    temporary.write_text("\n".join(lines), encoding="utf8")
    temporary.replace(OUT / "report.md")


def execute():
    from src.data import validate_manifest, write_json
    from src.engine import train, evaluate, load_checkpoint, device_for
    from src.extended_models import build, BASELINES, PADDING, NORMALIZATION
    groups = {"baselines": BASELINES, "padding": PADDING, "bn": NORMALIZATION}
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=[*groups, "all", "check", "report", "robustness", "detection",
                                         *BASELINES, *PADDING, *NORMALIZATION])
    args = parser.parse_args()
    cfg = json.loads((ROOT / "configs/paper.json").read_text(encoding="utf-8-sig"))
    OUT.mkdir(exist_ok=True)
    if args.group == "report":
        report()
        return
    device_for(cfg)
    validate_manifest(ROOT)
    if args.group in ("robustness", "detection"):
        from src.image_experiments import robustness, detection
        {"robustness": robustness, "detection": detection}[args.group](ROOT, OUT, cfg)
        report()
        return
    all_models = (*BASELINES, *PADDING, *NORMALIZATION)
    names = all_models if args.group in ("check", "all") else groups.get(args.group, (args.group,))
    report()
    failures = []
    for name in names:
        try:
            run_model(name, cfg, args.group == "check")
        except Exception as error:
            failures.append(name)
            write_json(OUT / name / "failure.json", {"error": str(error)})
            print(f"{name} 失敗：{error}", flush=True)
            if args.group in all_models:
                raise
        report()
    if args.group == "all":
        from src.image_experiments import robustness, detection
        if not (OUT / "robustness/predictions.csv").exists():
            robustness(ROOT, OUT, cfg)
        if not (OUT / "detection/results.csv").exists():
            detection(ROOT, OUT, cfg)
        report()
    if failures:
        raise SystemExit("未完成：" + ", ".join(failures))


def run_model(name, cfg, check_only):
    from src.data import write_json
    from src.engine import train, evaluate, load_checkpoint
    from src.extended_models import build
    directory = OUT / name
    checkpoint = directory / "last.pt"
    if (directory / "complete.json").is_file() and not check_only:
        return
    print(f"{name}: GPU 預檢／接續", flush=True)
    if checkpoint.exists() and not check_only:
        model_cfg = load_checkpoint(checkpoint)["config"]
    else:
        audit = preflight(name, cfg)
        write_json(directory / "preflight.json", audit)
        model_cfg = {**cfg, "batch_size": audit["batch_size"]}
    if check_only:
        return
    if not (checkpoint.exists() and load_checkpoint(checkpoint)["epoch"] == 200
            and (directory / "weights.pt").exists() and (directory / "run.json").exists()):
        train(ROOT, model_cfg, name, resume=checkpoint.exists(), output_dir=OUT, model_factory=build)
    evaluate(ROOT, name, output_dir=OUT, model_factory=build)
    (directory / "failure.json").unlink(missing_ok=True)
    write_json(directory / "complete.json", {"epochs": 200})


if __name__ == "__main__":
    import msvcrt
    OUT.mkdir(exist_ok=True)
    with (OUT / ".run.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise SystemExit("已有延伸實驗正在執行，請等待完成或中斷後續跑。")
        execute()


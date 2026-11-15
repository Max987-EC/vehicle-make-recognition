import csv
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.cache/matplotlib'))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data import VehicleDataset, validate_manifest, write_json, write_csv
from .models import CTv1, VARIANTS, parameter_counts


def seed_all(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def device_for(cfg):
    if cfg['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('找不到 CUDA GPU；請執行 doctor。正式實驗不自動改用 CPU。')
    return torch.device(cfg['device'])


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def optimizer_for(model, cfg):
    return torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'], betas=tuple(cfg['adam_betas']), eps=cfg['adam_epsilon'], weight_decay=cfg['weight_decay'])


def environment():
    return {'python': platform.python_version(), 'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'cuda_available': torch.cuda.is_available(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'platform': platform.platform()}


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(obj, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(obj, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    # Checkpoints contain only tensors and primitive containers; never unrestricted pickle.
    return torch.load(path, map_location='cpu', weights_only=True)


def train(root, cfg, variant, resume=False, output_dir=None, model_factory=None):
    if not cfg['assumptions_confirmed']:
        raise ValueError('正式訓練尚未開放：請先確認 docs/assumptions.md 的補充設定，再將 assumptions_confirmed 設為 true。')
    rows, meta = validate_manifest(root)
    if cfg['epochs'] != 200 or cfg['image_size'] != 244:
        raise ValueError('正式論文實驗必須使用 200 epochs、244×244。')
    device = device_for(cfg)
    seed_all(cfg['seed'])
    model = (model_factory or CTv1)(variant, cfg).to(device)
    optimizer = optimizer_for(model, cfg)
    generator = torch.Generator().manual_seed(cfg['seed'])
    # There is no augmentation: decode once to CPU RAM, preserving identical pixels each epoch.
    preload_start = time.perf_counter()
    source_dataset = VehicleDataset(root, [r for r in rows if r['split'] == 'train'], cfg)
    samples = [source_dataset[i] for i in range(len(source_dataset))]
    cached_dataset = TensorDataset(torch.stack([s[0] for s in samples]), torch.tensor([s[1] for s in samples], dtype=torch.long))
    del samples
    preload_seconds = time.perf_counter() - preload_start
    loader = DataLoader(cached_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, generator=generator)
    directory = (output_dir if output_dir is not None else root / 'outputs') / variant
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / 'last.pt'
    manifest_hash = fingerprint(root / 'data/manifest.csv')
    start_epoch, history, seconds = 0, [], 0.0
    if checkpoint_path.exists() and not resume:
        raise ValueError(f'{checkpoint_path} 已存在；使用 --resume 或先另存既有實驗。')
    if resume:
        if not checkpoint_path.exists():
            raise ValueError('找不到續訓 checkpoint。')
        ck = load_checkpoint(checkpoint_path)
        if ck['config'] != cfg or ck['manifest_sha256'] != manifest_hash or ck['variant'] != variant or ck['classes'] != meta['classes']:
            raise ValueError('續訓設定、類別或資料與原 checkpoint 不一致。')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        torch.set_rng_state(ck['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        generator.set_state(ck['loader_rng'])
        start_epoch, history, seconds = ck['epoch'], ck['history'], ck['training_seconds']
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, cfg['epochs']):
        model.train()
        sync(device)
        started = time.perf_counter()
        total_loss, correct, count = 0., 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            if not torch.isfinite(loss):
                raise ValueError('Loss 非有限值，已停止訓練。')
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            count += len(y)
        sync(device)
        elapsed = time.perf_counter() - started
        seconds += elapsed
        history.append({'epoch': epoch + 1, 'loss': total_loss / count, 'accuracy': correct / count, 'seconds': elapsed})
        ck = {'schema': 1, 'variant': variant, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch + 1, 'config': cfg, 'classes': meta['classes'], 'source': meta, 'manifest_sha256': manifest_hash, 'history': history, 'training_seconds': seconds, 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [], 'loader_rng': generator.get_state(), 'environment': environment()}
        atomic_save(ck, checkpoint_path)
        write_csv(directory / 'history.csv', history, ['epoch', 'loss', 'accuracy', 'seconds'])
        print(f'{variant} epoch {epoch + 1}/200 loss={total_loss/count:.5f} train_acc={correct/count:.3f} seconds={elapsed:.2f}', flush=True)
    atomic_save({'variant': variant, 'config': cfg, 'classes': meta['classes'], 'model': model.state_dict(), 'epoch': 200}, directory / 'weights.pt')
    write_json(directory / 'run.json', {'config': cfg, 'source': meta, 'environment': environment(), 'parameters': parameter_counts(model), 'training_seconds_excluding_checkpoint_io': seconds, 'image_preload_seconds_this_invocation': preload_seconds, 'data_loading': 'fixed preprocessed CPU tensor cache; no augmentation', 'manifest_sha256': manifest_hash})


def evaluate(root, variant, output_dir=None, model_factory=None):
    directory = (output_dir if output_dir is not None else root / 'outputs') / variant
    ck = load_checkpoint(directory / 'last.pt')
    if ck['epoch'] != 200:
        raise ValueError('只能對完成 200 epochs 的模型產生正式比較。')
    rows, meta = validate_manifest(root)
    if fingerprint(root / 'data/manifest.csv') != ck['manifest_sha256'] or meta['classes'] != ck['classes']:
        raise ValueError('目前資料與訓練 checkpoint 不一致。')
    cfg = ck['config']
    seed_all(cfg['seed'])
    device = device_for(cfg)
    model = (model_factory or CTv1)(variant, cfg).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    test_rows = [r for r in rows if r['split'] == 'test']
    dataset = VehicleDataset(root, test_rows, cfg)
    # Preload for compute-only benchmark. I/O + transfers have a separate timed pass.
    x = torch.stack([dataset[i][0] for i in range(200)]).to(device)
    truth = [int(r['label']) for r in test_rows]
    predictions, times = [], []
    def forward_batch(batch):
        if model_factory is None:
            return model(batch)
        return torch.cat([model(part) for part in batch.split(cfg["batch_size"])])

    with torch.inference_mode():
        for _ in range(3):
            forward_batch(x[:25])
        sync(device)
        for batch in x.split(25):
            sync(device)
            start = time.perf_counter()
            logits = forward_batch(batch)
            sync(device)
            times.append(time.perf_counter() - start)
            predictions.extend(logits.argmax(1).cpu().tolist())
        sync(device)
        start = time.perf_counter()
        for images, _ in DataLoader(dataset, batch_size=25, shuffle=False, num_workers=0):
            forward_batch(images.to(device)).argmax(1).cpu()
        sync(device)
        pipeline_seconds = time.perf_counter() - start
    report = classification_report(truth, predictions, labels=list(range(5)), target_names=meta['classes'], zero_division=0, output_dict=True)
    result = {'model': variant, 'source': meta['interpretation'], 'manifest_sha256': ck['manifest_sha256'], 'accuracy': accuracy_score(truth, predictions), 'weighted_f1': report['weighted avg']['f1-score'], 'classification_report': report, 'confusion_matrix': confusion_matrix(truth, predictions, labels=list(range(5))).tolist(), 'parameters': parameter_counts(model), 'training_seconds': ck['training_seconds'], 'inference_seconds_per_batch25': times, 'inference_mean_seconds_per_batch25': sum(times) / 8, 'inference_seconds_per_image': sum(times) / 200, 'pipeline_seconds_per_image': pipeline_seconds / 200, 'warmup_batches': 3, 'timing_scope': 'forward only, preloaded GPU tensors; separate pipeline includes disk decode/resize/transfer', 'environment': environment(), 'weights_bytes': (directory / 'weights.pt').stat().st_size}
    write_json(directory / 'metrics.json', result)
    write_csv(directory / 'per_class.csv', [{'make': name, **{k:report[name][k] for k in ('precision','recall','f1-score','support')}} for name in meta['classes']], ['make','precision','recall','f1-score','support'])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,5), layout='constrained')
    cm = np.array(result['confusion_matrix'])
    ax.imshow(cm, cmap='Blues')
    ax.set(xticks=range(5), yticks=range(5), xticklabels=meta['classes'], yticklabels=meta['classes'], xlabel='Predicted', ylabel='True', title=variant)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
    for i in range(5):
        for j in range(5):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
    fig.savefig(directory / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(9,3),layout='constrained')
    for ax,key in zip(axes,['loss','accuracy']):
        ax.plot([h['epoch'] for h in ck['history']],[h[key] for h in ck['history']])
        ax.set(xlabel='Epoch',ylabel='Training '+key,title=variant)
    fig.savefig(directory / 'training_curves.png',dpi=160)
    plt.close(fig)
    write_csv(directory / 'predictions.csv', [{'path': r['path'], 'truth': meta['classes'][y], 'predicted': meta['classes'][p]} for r, y, p in zip(test_rows, truth, predictions)], ['path', 'truth', 'predicted'])
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def report_all(root, output_dir=None):
    output_dir = output_dir if output_dir is not None else root / 'outputs'
    results = []
    for name in VARIANTS:
        path = output_dir / name / 'metrics.json'
        if path.exists():
            results.append(json.loads(path.read_text(encoding='utf-8')))
    if not results:
        raise ValueError('尚無正式評估結果；試跑結果不會列入論文比較。')
    if len({r['manifest_sha256'] for r in results}) != 1:
        raise ValueError('各模型使用不同資料切分，不可直接比較。')
    fields = ['model', 'accuracy', 'weighted_f1', 'trainable', 'total', 'weights_bytes', 'training_seconds', 'inference_seconds_per_image']
    values = [{**{k: r[k] for k in fields if k in r}, 'trainable': r['parameters']['trainable'], 'total': r['parameters']['keras_style_total']} for r in results]
    write_csv(output_dir / 'comparison.csv', values, fields)
    lines = ['# 三模型實測比較', '', results[0]['source'], '', '未列出的模型尚未完成評估。計時硬體與作者不同，不直接比較速度。', '', '| 模型 | Accuracy | Weighted F1 | 總參數（含 BN 狀態） | 訓練秒數 | 推論秒／張 |', '|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['weighted_f1']:.4f} | {r['parameters']['keras_style_total']:,} | {r['training_seconds']:.2f} | {r['inference_seconds_per_image']:.6f} |")
    lines += ['', '## 各品牌 F1', '', '| 品牌 | ' + ' | '.join(r['model'] for r in results) + ' |', '|---|' + '---:|' * len(results)]
    for name in results[0]['classification_report']:
        if name in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        lines.append('| ' + name + ' | ' + ' | '.join(f"{r['classification_report'][name]['f1-score']:.4f}" for r in results) + ' |')
    lines += ['', '完整逐品牌 precision、recall、F1 與樣本數見各模型的 per_class.csv；混淆矩陣與訓練曲線也保存在各模型資料夾。']
    metadata_path = root / 'data/metadata.json'
    if metadata_path.exists():
        meta = json.loads(metadata_path.read_text(encoding='utf-8'))
        lines += ['', '## 實驗條件', '', '- 品牌：' + ', '.join(meta['classes']), '- 資料：每品牌 200 張，共 1,000 張；800 訓練／200 測試。', '- 照明篩選：' + meta.get('lighting', '未記錄'), '- 最終 checkpoint：第 200 epoch；沒有用測試成績挑模型。', '- 前處理後的訓練圖片先存入 CPU 記憶體；表中訓練時間不含一次性載入與 checkpoint 寫檔。', '- 推論暖機 3 組，再測 8 組×25 張，模型計時不含讀圖及 CPU→GPU 搬移。', '- 硬體：' + str(results[0]['environment']['gpu']), '', '## 與原論文的差異及限制', '', '- 使用 CompCars，並以 Volkswagen 替代缺少的 Isuzu，不是作者原始照片。', '- 白天依影像外觀審查，無拍攝時間真值；亮度候選門檻會排除部分較暗的白天照片。', '- 使用 PyTorch；未公開超參數採使用者確認的補充設定，詳見 ../docs/assumptions.md。', '- 排除相同解碼像素的圖片，但未保證同車不同影格或近似影像完全分離。', '- 僅一個固定切分與訓練種子；結果不代表跨資料集或跨種子的穩定性。', '- 不將本次分數／耗時視為與原論文相同條件下的直接比較。']
    (output_dir / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')

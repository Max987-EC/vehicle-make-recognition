"""Run with .venv/Scripts/python.exe run.py --help from any working directory."""
import argparse
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description='CTv1 論文重現專案')
    parser.add_argument('--config', default='configs/paper.json')
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('doctor', 'smoke', 'summary', 'report', 'check-data'):
        commands.add_parser(name)
    training = commands.add_parser('train')
    training.add_argument('--model', choices=['ctv1', 'ctv1_dw', 'ctv1_dr', 'all'], default='all')
    training.add_argument('--resume', action='store_true')
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--model', choices=['ctv1', 'ctv1_dw', 'ctv1_dr', 'all'], default='all')
    prediction = commands.add_parser('predict')
    prediction.add_argument('--model', choices=['ctv1', 'ctv1_dw', 'ctv1_dr'], default='ctv1')
    prediction.add_argument('--image', required=True)
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding='utf-8-sig'))
    try:
        fixed = {'image_size':244, 'classes':5, 'images_per_class':200, 'test_fraction':0.2, 'pixel_scale':'zero_one', 'resize':'bilinear', 'initialization':'keras_glorot_uniform', 'pool_padding':'valid', 'workers':0}
        for key, value in fixed.items():
            if cfg.get(key) != value:
                raise ValueError(f'本版本要求 {key}={value}；不可只改設定而未調整實作。')
        if cfg['batch_size'] < 1 or cfg['learning_rate'] <= 0:
            raise ValueError('batch_size 與 learning_rate 必須大於 0。')
        import torch
        from src.models import CTv1, VARIANTS, parameter_counts
        from src.data import validate_manifest, write_json, image_tensor
        from src.engine import environment, device_for, seed_all, train, evaluate, report_all, load_checkpoint, optimizer_for
        if args.command == 'doctor':
            env = environment()
            env['python_executable'] = sys.executable
            env['project_venv'] = Path(sys.prefix).resolve() == (ROOT / '.venv').resolve()
            write_json(ROOT / 'outputs/environment.json', env)
            print(json.dumps(env, indent=2, ensure_ascii=False))
            device_for(cfg)
            if not env['project_venv']:
                raise ValueError('請使用專案 .venv 的 Python。')
            x = torch.randn(32, 32, device='cuda', requires_grad=True)
            x.square().mean().backward()
            print('GPU 前向與反向運算通過。')
        elif args.command == 'check-data':
            rows, meta = validate_manifest(ROOT)
            print(json.dumps(meta, indent=2, ensure_ascii=False))
            print(f'{len(rows)} 張圖片驗證通過。')
        elif args.command in ('summary', 'smoke'):
            seed_all(cfg['seed'])
            device = device_for(cfg) if args.command == 'smoke' else torch.device('cpu')
            summaries = []
            for name in VARIANTS:
                model = CTv1(name, cfg).to(device)
                batch_size = cfg['batch_size'] if args.command == 'smoke' else 1
                x = torch.randn(batch_size, 3, 244, 244, device=device)
                shapes = []
                model.eval()
                with torch.no_grad():
                    z = x
                    for block in model.blocks:
                        z = block(z)
                        shapes.append(list(z.shape))
                item = {'variant': name, **parameter_counts(model), 'feature_shapes': shapes}
                if args.command == 'smoke':
                    model.train()
                    optimizer = optimizer_for(model, cfg)
                    before = model.head[-1].weight.detach().clone()
                    losses = []
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        loss = torch.nn.functional.cross_entropy(model(x), torch.arange(batch_size, device=device) % 5)
                        loss.backward()
                        assert torch.isfinite(loss)
                        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                        optimizer.step()
                        losses.append(loss.item())
                    assert not torch.equal(before, model.head[-1].weight)
                    model.eval()
                    buffer = io.BytesIO()
                    torch.save(model.state_dict(), buffer)
                    buffer.seek(0)
                    clone = CTv1(name, cfg).to(device).eval()
                    clone.load_state_dict(torch.load(buffer, weights_only=True, map_location=device))
                    with torch.no_grad():
                        torch.testing.assert_close(model(x), clone(x), rtol=0, atol=0)
                    item.update({'synthetic_only': True, 'batch_size_tested': batch_size, 'losses': losses, 'backward_finite': True, 'roundtrip_exact': True})
                    del optimizer, clone
                summaries.append(item)
                print(json.dumps(item, ensure_ascii=False))
                del model, x
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            write_json(ROOT / 'outputs' / f'{args.command}.json', summaries)
        elif args.command == 'train':
            for name in VARIANTS if args.model == 'all' else [args.model]:
                train(ROOT, cfg, name, args.resume)
        elif args.command == 'evaluate':
            for name in VARIANTS if args.model == 'all' else [args.model]:
                evaluate(ROOT, name)
            report_all(ROOT)
        elif args.command == 'report':
            report_all(ROOT)
        elif args.command == 'predict':
            path = ROOT / 'outputs' / args.model / 'weights.pt'
            if not path.exists():
                raise ValueError('尚無正式訓練權重，請先完成訓練。')
            ck = load_checkpoint(path)
            if ck['variant'] != args.model or ck['epoch'] != 200:
                raise ValueError('模型權重不符合指定模型／epoch。')
            device = device_for(ck['config'])
            model = CTv1(args.model, ck['config']).to(device).eval()
            model.load_state_dict(ck['model'])
            with torch.inference_mode():
                probs = model(image_tensor(ROOT / args.image, ck['config']).unsqueeze(0).to(device)).softmax(1)[0].cpu().tolist()
            result = {'image': args.image, 'model': args.model, 'predicted': ck['classes'][max(range(5), key=probs.__getitem__)], 'probabilities': dict(zip(ck['classes'], probs))}
            print(json.dumps(result, indent=2, ensure_ascii=False))
            write_json(ROOT / 'outputs/last_prediction.json', result)
    except (ValueError, OSError, ImportError, RuntimeError) as error:
        print(f'無法完成：{error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())

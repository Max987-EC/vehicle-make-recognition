import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def image_tensor(path, cfg):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert('RGB')
        image = image.resize((cfg['image_size'], cfg['image_size']), Image.Resampling.BILINEAR)
        array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_manifest(root):
    path = root / 'data/manifest.csv'
    if not path.exists():
        raise ValueError('缺少正式清單 data/manifest.csv，請從繳交資料還原。')
    rows = read_csv(path)
    meta = json.loads((root / 'data/metadata.json').read_text(encoding='utf-8'))
    if len(rows) != 1000 or Counter(r['split'] for r in rows) != {'train': 800, 'test': 200}:
        raise ValueError('需要 1000 張與 800/200 固定切分。')
    if len({r['path'] for r in rows}) != 1000 or len({r['sha256_pixels'] for r in rows}) != 1000:
        raise ValueError('圖片或像素雜湊重複。')
    if len(meta['classes']) != 5 or Counter(r['make'] for r in rows) != {c: 200 for c in meta['classes']}:
        raise ValueError('類別數或每類張數錯誤。')
    for r in rows:
        if meta['classes'][int(r['label'])] != r['make']:
            raise ValueError('類別映射不一致。')
        p = (root / r['path']).resolve()
        if not p.is_relative_to(root.resolve()) or not p.is_file():
            raise ValueError(f'圖片不存在或超出專案：{p}')
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert('RGB')
            digest = hashlib.sha256(str(im.size).encode() + im.tobytes()).hexdigest()
        if digest != r['sha256_pixels']:
            raise ValueError(f'圖片已變更，請從繳交資料還原原圖：{p}')
    return rows, meta


class VehicleDataset(Dataset):
    def __init__(self, root, rows, cfg):
        self.root, self.rows, self.cfg = root, rows, cfg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        return image_tensor(self.root / row['path'], self.cfg), int(row['label'])

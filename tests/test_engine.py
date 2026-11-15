"""Exercise real checkpoint/evaluation workflow with a tiny substitute model, never publish scores."""
import json
from pathlib import Path
import pytest
import torch
from torch import nn
from src import engine

CFG = json.loads((Path(__file__).parents[1]/'configs/paper.json').read_text())


def test_train_resume_evaluate_and_report(tmp_path,monkeypatch):
    torch.set_num_threads(2)
    cfg={**CFG,'device':'cpu','batch_size':800,'assumptions_confirmed':True}
    rows=[{'path':str(i),'label':i%5,'split':'train' if i<800 else 'test'} for i in range(1000)]
    meta={'classes':['a','b','c','d','e'],'interpretation':'TEST FIXTURE ONLY'}
    (tmp_path/'data').mkdir()
    (tmp_path/'data/manifest.csv').write_text('fixture',encoding='utf-8')
    class Tiny(nn.Module):
        def __init__(self,*args):
            super().__init__()
            self.layer=nn.Linear(5,5)
        def forward(self,x):
            return self.layer(x)
    class TinyData:
        def __init__(self,root,items,cfg): self.items=items
        def __len__(self): return len(self.items)
        def __getitem__(self,i):
            label=int(self.items[i]['label'])
            return torch.eye(5)[label],label
    monkeypatch.setattr(engine,'CTv1',Tiny)
    monkeypatch.setattr(engine,'VehicleDataset',TinyData)
    monkeypatch.setattr(engine,'validate_manifest',lambda root:(rows,meta))
    original_save=engine.atomic_save
    class StopRun(Exception): pass
    def interrupted_save(obj,path):
        original_save(obj,path)
        if obj.get('epoch')==2 and Path(path).name=='last.pt': raise StopRun()
    monkeypatch.setattr(engine,'atomic_save',interrupted_save)
    with pytest.raises(StopRun): engine.train(tmp_path,cfg,'ctv1')
    ck=engine.load_checkpoint(tmp_path/'outputs/ctv1/last.pt')
    assert ck['epoch']==2
    monkeypatch.setattr(engine,'atomic_save',original_save)
    engine.train(tmp_path,cfg,'ctv1',resume=True)
    ck=engine.load_checkpoint(tmp_path/'outputs/ctv1/last.pt')
    assert ck['epoch']==200 and len(ck['history'])==200
    result=engine.evaluate(tmp_path,'ctv1')
    assert len(result['inference_seconds_per_batch25'])==8
    assert len(result['confusion_matrix'])==5
    engine.report_all(tmp_path)
    assert (tmp_path/'outputs/comparison.csv').exists()
    assert (tmp_path/'outputs/ctv1/training_curves.png').exists()
    assert (tmp_path/'outputs/ctv1/per_class.csv').exists()
    (tmp_path/'data/manifest.csv').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='不一致'): engine.evaluate(tmp_path,'ctv1')

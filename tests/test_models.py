import io
import json
from pathlib import Path

import pytest
import torch
from src.models import CTv1, KerasBatchNorm, parameter_counts
from src.engine import optimizer_for

CFG = json.loads((Path(__file__).parents[1] / 'configs/paper.json').read_text())


@pytest.mark.parametrize('name,trainable,moving', [('ctv1',6551877,4032), ('ctv1_dw',974115,6022), ('ctv1_dr',572021,4224)])
def test_paper_table3_and_shapes(name, trainable, moving):
    model = CTv1(name, CFG).eval()
    counts = parameter_counts(model)
    assert counts == {'trainable': trainable, 'bn_moving_state': moving, 'keras_style_total': trainable + moving}
    with torch.no_grad():
        x = torch.randn(1,3,244,244)
        sizes = []
        for block in model.blocks:
            x = block(x)
            sizes.append(x.shape[-1])
        assert sizes == [121,60,29,14,6,2]
        assert model.head(x).shape == (1,5)


def test_bn_population_running_variance():
    bn = KerasBatchNorm(1, retention=0.5)
    x = torch.tensor([[[[1.,3.]]]])
    y = bn(x)
    torch.testing.assert_close(bn.running_mean, torch.tensor([1.]))
    torch.testing.assert_close(bn.running_var, torch.tensor([1.]))
    torch.testing.assert_close(y.mean(), torch.tensor(0.))
    assert [n for n,_ in bn.named_parameters()] == ['bias']


def test_checkpoint_optimizer_resume_matches_next_step():
    torch.manual_seed(12)
    m = torch.nn.Linear(4, 2)
    opt = optimizer_for(m, CFG)
    x, y = torch.randn(3,4), torch.tensor([0,1,0])
    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x),y).backward()
        optimizer.step()
    step(m,opt)
    b = io.BytesIO()
    torch.save({'model':m.state_dict(),'optimizer':opt.state_dict()},b)
    b.seek(0)
    ck = torch.load(b,weights_only=True)
    clone = torch.nn.Linear(4,2)
    clone.load_state_dict(ck['model'])
    other = optimizer_for(clone,CFG)
    other.load_state_dict(ck['optimizer'])
    step(m,opt)
    step(clone,other)
    torch.testing.assert_close(m(x),clone(x),rtol=0,atol=0)

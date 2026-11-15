"""Figure 8/9 and Algorithm 1; unresolved defaults are explicit in config."""
import math
import torch
from torch import nn

VARIANTS = ('ctv1', 'ctv1_dw', 'ctv1_dr')
CHANNELS = (32, 64, 128, 256, 512, 1024)


class KerasBatchNorm(nn.Module):
    """Population variance in moving state, matching Keras (PyTorch BN uses unbiased)."""
    def __init__(self, channels, eps=0.001, retention=0.99, scale=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels)) if scale else None
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer('running_mean', torch.zeros(channels))
        self.register_buffer('running_var', torch.ones(channels))
        self.eps, self.retention = eps, retention

    def forward(self, x):
        if self.training:
            var, mean = torch.var_mean(x, dim=(0, 2, 3), correction=0)
            with torch.no_grad():
                self.running_mean.lerp_(mean.detach(), 1 - self.retention)
                self.running_var.lerp_(var.detach(), 1 - self.retention)
        else:
            mean, var = self.running_mean, self.running_var
        shape = (1, -1, 1, 1)
        normalized = (x - mean.view(shape)) * torch.rsqrt(var.view(shape) + self.eps)
        if self.weight is not None:
            normalized = normalized * self.weight.view(shape)
        return normalized + self.bias.view(shape)


def conv_stage(cin, cout, kernel, cfg, groups=1):
    return [nn.Conv2d(cin, cout, kernel, padding=kernel // 2, groups=groups, bias=False),
            KerasBatchNorm(cout, cfg['bn_epsilon'], cfg['bn_retention'], cfg['bn_scale']), nn.ReLU()]


class CTv1(nn.Module):
    def __init__(self, variant, cfg, classes=5):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Unknown model: {variant}')
        self.variant = variant
        blocks, cin = [], 3
        for cout in CHANNELS:
            if variant == 'ctv1':
                layers = conv_stage(cin, cout, 3, cfg)
            elif variant == 'ctv1_dw':
                layers = conv_stage(cin, cin, 3, cfg, groups=cin) + conv_stage(cin, cout, 1, cfg)
            else:
                layers = conv_stage(cin, 16, 1, cfg) + conv_stage(16, cout, 3, cfg)
            layers.append(nn.MaxPool2d(3, stride=2, padding=0))
            blocks.append(nn.Sequential(*layers))
            cin = cout
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, classes))
        self.apply(self._initialize)

    @staticmethod
    def _initialize(layer):
        if isinstance(layer, nn.Conv2d):
            if layer.groups == layer.in_channels and layer.groups > 1:
                # Keras depthwise kernel layout: [kh, kw, input_channels, multiplier].
                area = layer.kernel_size[0] * layer.kernel_size[1]
                bound = math.sqrt(6 / (area * (layer.in_channels + layer.out_channels // layer.in_channels)))
                nn.init.uniform_(layer.weight, -bound, bound)
            else:
                nn.init.xavier_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def parameter_counts(model):
    trainable = sum(p.numel() for p in model.parameters())
    moving = sum(buffer.numel() for name, buffer in model.named_buffers()
                 if name.rsplit('.', 1)[-1] in ('running_mean', 'running_var'))
    return {'trainable': trainable, 'bn_moving_state': moving, 'keras_style_total': trainable + moving}

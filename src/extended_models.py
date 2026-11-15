"""Computer experiment model variants."""
from torch import nn

from .models import CTv1, CHANNELS, VARIANTS, KerasBatchNorm

BASELINES = ("vgg16", "inception_v3", "resnet50", "mobilenet_v1", "yolo11m_cls")
PADDING = tuple(f"{name}-{mode}" for name in VARIANTS for mode in ("5l", "np", "st"))
NORMALIZATION = tuple(f"{name}-no_bn" for name in VARIANTS)


class FiveBlockCT(CTv1):
    def __init__(self, variant, cfg, mode):
        nn.Module.__init__(self)
        self.blocks = nn.ModuleList()
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 5))
        incoming = 3
        for index, outgoing in enumerate(CHANNELS[1:]):
            if variant == "ctv1":
                specs = [(incoming, outgoing, 3, 1)]
            elif variant == "ctv1_dw":
                specs = [(incoming, incoming, 3, incoming), (incoming, outgoing, 1, 1)]
            else:
                specs = [(incoming, 16, 1, 1), (16, outgoing, 3, 1)]
            layers = []
            for cin, cout, kernel, groups in specs:
                layers.extend([
                    nn.Conv2d(cin, cout, kernel, groups=groups, bias=False,
                              padding=0 if mode == "np" else kernel // 2,
                              stride=2 if mode == "st" and index >= 3 and kernel == 3 else 1),
                    KerasBatchNorm(cout, cfg["bn_epsilon"], cfg["bn_retention"], cfg["bn_scale"]),
                    nn.ReLU(),
                ])
            self.blocks.append(nn.Sequential(*layers, nn.MaxPool2d(3, stride=2)))
            incoming = outgoing
        self.apply(self._initialize)


def mobilenet():
    def stage(cin, cout, kernel, stride, groups=1):
        return [nn.Conv2d(cin, cout, kernel, stride, kernel // 2, groups=groups, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU6(inplace=True)]
    layers = stage(3, 32, 3, 2)
    incoming = 32
    for outgoing, stride in [(64, 1), (128, 2), (128, 1), (256, 2), (256, 1),
                             (512, 2), *[(512, 1)] * 5, (1024, 2), (1024, 1)]:
        layers.extend(stage(incoming, incoming, 3, stride, incoming))
        layers.extend(stage(incoming, outgoing, 1, 1))
        incoming = outgoing
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(1024, 5))


class YoloLogits(nn.Module):
    def __init__(self):
        super().__init__()
        from ultralytics.nn.tasks import ClassificationModel
        self.model = ClassificationModel("yolo11m-cls.yaml", nc=5, verbose=False)

    def forward(self, images):
        result = self.model(images)
        return result if self.training else result[1]


def build(name, cfg):
    if name in VARIANTS:
        return CTv1(name, cfg)
    if name in PADDING:
        variant, mode = name.rsplit("-", 1)
        return FiveBlockCT(variant, cfg, mode)
    if name in NORMALIZATION:
        model = CTv1(name.rsplit("-", 1)[0], cfg)
        for block in model.blocks:
            for index, layer in enumerate(block):
                if isinstance(layer, KerasBatchNorm):
                    block[index] = nn.Identity()
        return model
    if name == "mobilenet_v1":
        return mobilenet()
    if name == "yolo11m_cls":
        return YoloLogits()
    from torchvision import models
    if name == "inception_v3":
        return models.inception_v3(weights=None, num_classes=5, aux_logits=False, init_weights=True)
    if name == "vgg16":
        model = models.vgg16(weights=None, num_classes=5)
        # Five pools already produce 7×7 features from the fixed 244×244 input.
        model.avgpool = nn.Identity()
        return model
    if name == "resnet50":
        return models.resnet50(weights=None, num_classes=5)
    raise ValueError(f"未知模型：{name}")

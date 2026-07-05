import torch
import torch.nn as nn


_CFG = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
}


class VGGCIFAR(nn.Module):
    """Lightweight VGG-style network for CIFAR."""

    def __init__(self, num_classes=100, nch=3, variant="VGG11"):
        super().__init__()
        self.features = self._make_layers(_CFG[variant], nch)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_feature(self, x, idx_from, idx_to=-1):
        if idx_to == -1:
            idx_to = idx_from

        features = []
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
                if idx_to < len(features):
                    return features[idx_from:idx_to + 1], None

        x = torch.flatten(x, 1)
        features.append(x)
        if idx_to < len(features):
            return features[idx_from:idx_to + 1], None

        for layer in self.classifier:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features.append(x)
                if idx_to < len(features):
                    return features[idx_from:idx_to + 1], None

        features.append(x)
        return features[idx_from:idx_to + 1], x

    @staticmethod
    def _make_layers(cfg, nch):
        layers = []
        in_channels = nch
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend([
                    nn.Conv2d(in_channels, v, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True),
                ])
                in_channels = v
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)


def vgg_cifar(num_classes=100, nch=3):
    return VGGCIFAR(num_classes=num_classes, nch=nch)

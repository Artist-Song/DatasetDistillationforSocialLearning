import torch
import torch.nn as nn


class AlexNetCIFAR(nn.Module):
    """Small AlexNet-style network for 32x32 CIFAR images."""

    def __init__(self, num_classes=100, nch=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(nch, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
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

        x = self.avgpool(x)
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


def alexnet_cifar(num_classes=100, nch=3):
    return AlexNetCIFAR(num_classes=num_classes, nch=nch)

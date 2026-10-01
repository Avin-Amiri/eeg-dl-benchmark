from __future__ import annotations

import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                norm = torch.norm(self.weight, p=2, dim=(1, 2, 3), keepdim=True)
                desired = torch.clamp(norm, max=self.max_norm)
                self.weight.mul_(desired / (norm + 1e-8))
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, max_norm: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                norm = torch.norm(self.weight, p=2, dim=1, keepdim=True)
                desired = torch.clamp(norm, max=self.max_norm)
                self.weight.mul_(desired / (norm + 1e-8))
        return super().forward(x)


class EEGNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        channels: int = 32,
        samples: int = 8064,
        f1: int = 8,
        d: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.samples = samples

        # Block 1
        self.conv1 = nn.Conv2d(1, f1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(f1)
        self.depthwise = Conv2dWithConstraint(
            f1, f1 * d, (channels, 1), groups=f1, bias=False, max_norm=1.0
        )
        self.bn2 = nn.BatchNorm2d(f1 * d)
        self.act1 = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2
        self.separable = nn.Conv2d(f1 * d, f2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(f2)
        self.act2 = nn.ELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # Compute flatten size
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            x = self.drop1(self.pool1(self.act1(self.bn2(self.depthwise(self.bn1(self.conv1(dummy)))))))
            x = self.drop2(self.pool2(self.act2(self.bn3(self.separable(x)))))
            flatten_dim = x.numel()

        self.classifier = LinearWithConstraint(flatten_dim, num_classes, max_norm=0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

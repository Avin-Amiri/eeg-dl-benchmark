import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    """
    Conv2d with max-norm constraint on weights for regularization.
    """
    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.clamp_(min=-self.max_norm, max=self.max_norm)
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    """
    Linear layer with max-norm constraint on weights.
    """
    def __init__(self, *args, max_norm: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.clamp_(min=-self.max_norm, max=self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    """
    EEGNet: A Compact Convolutional Neural Network for EEG-based Brain-Computer Interfaces.
    Reference: Lawhern et al., 2018 (https://doi.org/10.1088/1741-2552/aace8c)

    Shape:
        - Input: (batch_size, 1, channels, samples)
        - Output: (batch_size, num_classes)
    """
    def __init__(
        self,
        num_classes: int = 4,
        channels: int = 32,
        samples: int = 128,
        f1: int = 8,
        depth_multiplier: int = 2,
        f2: int = 16,
        kernel_length: int = 64,
        dropout_rate: float = 0.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.samples = samples

        # Block 1: Temporal Convolution + Depthwise Spatial Convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=f1,
                kernel_size=(1, kernel_length),
                padding=(0, kernel_length // 2),
                bias=False
            ),
            nn.BatchNorm2d(f1),
            Conv2dWithConstraint(
                in_channels=f1,
                out_channels=f1 * depth_multiplier,
                kernel_size=(channels, 1),
                groups=f1,
                bias=False,
                max_norm=1.0
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout_rate)
        )

        # Block 2: Separable Convolution (Depthwise + Pointwise)
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=f1 * depth_multiplier,
                out_channels=f1 * depth_multiplier,
                kernel_size=(1, 16),
                padding=(0, 8),
                groups=f1 * depth_multiplier,
                bias=False
            ),
            nn.Conv2d(
                in_channels=f1 * depth_multiplier,
                out_channels=f2,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout_rate)
        )

        # Calculate flattened feature size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, channels, samples)
            out = self.block2(self.block1(dummy))
            feature_dim = out.view(1, -1).size(1)

        # Classifier
        self.classifier = nn.Sequential(
            LinearWithConstraint(in_features=feature_dim, out_features=num_classes, max_norm=0.25)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input validation: auto-expand (B, C, T) -> (B, 1, C, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.classifier(x)
        return logits

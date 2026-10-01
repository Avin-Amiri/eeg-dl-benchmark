import torch
import pytest
from src.eeg_benchmark.models.eegnet import EEGNet


def test_eegnet_forward_pass_4d():
    batch_size = 8
    channels = 32
    samples = 256
    num_classes = 4

    model = EEGNet(num_classes=num_classes, channels=channels, samples=samples)
    x = torch.randn(batch_size, 1, channels, samples)
    out = model(x)

    assert out.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {out.shape}"


def test_eegnet_forward_pass_3d():
    batch_size = 4
    channels = 14
    samples = 128
    num_classes = 2

    model = EEGNet(num_classes=num_classes, channels=channels, samples=samples)
    x = torch.randn(batch_size, channels, samples)
    out = model(x)

    assert out.shape == (batch_size, num_classes), f"Expected {(batch_size, num_classes)}, got {out.shape}"

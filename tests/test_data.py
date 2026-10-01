import numpy as np
import torch
from torch.utils.data import DataLoader
from eeg_benchmark.data import EEGDataset, EEGStandardScaler


def test_eeg_dataset_and_loader():
    num_samples = 40
    channels = 32
    time_points = 128
    num_classes = 4

    fake_data = np.random.randn(num_samples, channels, time_points).astype(np.float32)
    fake_labels = np.random.randint(0, num_classes, size=num_samples)

    dataset = EEGDataset(fake_data, fake_labels, transform=EEGStandardScaler())
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    batch_x, batch_y = next(iter(loader))

    assert batch_x.shape == (8, 1, channels, time_points)
    assert batch_y.shape == (8,)
    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.int64

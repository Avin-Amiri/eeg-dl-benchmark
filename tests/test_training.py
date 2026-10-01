import numpy as np
import torch
from torch.utils.data import DataLoader

from eeg_benchmark.data import EEGDataset, EEGStandardScaler
from eeg_benchmark.models import EEGNet
from eeg_benchmark.training import Trainer, TrainConfig


def test_trainer_runs_one_epoch_cpu():
    n = 32
    ch = 16
    t = 128
    k = 3

    x = np.random.randn(n, ch, t).astype(np.float32)
    y = np.random.randint(0, k, size=n).astype(np.int64)

    ds = EEGDataset(x, y, transform=EEGStandardScaler())
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    model = EEGNet(num_classes=k, channels=ch, samples=t, dropout_rate=0.1)
    cfg = TrainConfig(device="cpu", epochs=1, log_every=0)

    trainer = Trainer(model=model, train_loader=dl, val_loader=None, config=cfg)
    out = trainer.fit()

    assert "best_val_acc" in out

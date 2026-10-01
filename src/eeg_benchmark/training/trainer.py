from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: Optional[float] = 1.0
    log_every: int = 20


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainConfig] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = config or TrainConfig()

        self.device = torch.device(self.cfg.device)
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

        self.best_val_acc = -1.0
        self.best_state_dict = None

    def _step_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        train: bool,
    ) -> Dict[str, float]:
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        if train:
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()

            if self.cfg.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.cfg.max_grad_norm,
                )

            self.optimizer.step()
        else:
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x)
                loss = self.criterion(logits, y)

        acc = accuracy_from_logits(logits, y)
        return {"loss": float(loss.item()), "acc": float(acc)}

    def run_epoch(self, epoch: int) -> Dict[str, Dict[str, float]]:
        train_loss = 0.0
        train_acc = 0.0
        n_train = 0

        for i, batch in enumerate(self.train_loader):
            metrics = self._step_batch(batch, train=True)
            batch_size = batch[0].size(0)

            n_train += batch_size
            train_loss += metrics["loss"] * batch_size
            train_acc += metrics["acc"] * batch_size

            if self.cfg.log_every and (i + 1) % self.cfg.log_every == 0:
                print(
                    f"[epoch {epoch}] step {i + 1}: "
                    f"loss={metrics['loss']:.4f} "
                    f"acc={metrics['acc']:.4f}"
                )

        train_metrics = {
            "loss": train_loss / max(1, n_train),
            "acc": train_acc / max(1, n_train),
        }

        val_metrics: Dict[str, float] = {}

        if self.val_loader is not None:
            val_loss = 0.0
            val_acc = 0.0
            n_val = 0

            for batch in self.val_loader:
                metrics = self._step_batch(batch, train=False)
                batch_size = batch[0].size(0)

                n_val += batch_size
                val_loss += metrics["loss"] * batch_size
                val_acc += metrics["acc"] * batch_size

            val_metrics = {
                "loss": val_loss / max(1, n_val),
                "acc": val_acc / max(1, n_val),
            }

            if val_metrics["acc"] > self.best_val_acc:
                self.best_val_acc = val_metrics["acc"]
                self.best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        return {"train": train_metrics, "val": val_metrics}

    def fit(self) -> Dict[str, float]:
        for epoch in range(1, self.cfg.epochs + 1):
            metrics = self.run_epoch(epoch)

            message = (
                f"[epoch {epoch}] "
                f"train loss={metrics['train']['loss']:.4f} "
                f"acc={metrics['train']['acc']:.4f}"
            )

            if metrics["val"]:
                message += (
                    f" | val loss={metrics['val']['loss']:.4f} "
                    f"acc={metrics['val']['acc']:.4f}"
                )

            print(message)

        return {"best_val_acc": float(self.best_val_acc)}

    def save_best(self, path: str) -> None:
        if self.best_state_dict is None:
            raise RuntimeError(
                "No best checkpoint found. "
                "Provide val_loader or run validation first."
            )

        torch.save(self.best_state_dict, path)

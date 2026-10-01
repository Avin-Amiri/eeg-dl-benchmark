import argparse
import torch
from src.eeg_benchmark.data.deap import get_deap_dataloaders
from src.eeg_benchmark.models.eegnet import EEGNet
from src.eeg_benchmark.training.trainer import Trainer, TrainConfig

def main():
    parser = argparse.ArgumentParser(description="EEGNet Benchmark on DEAP")
    parser.add_argument(
        "--data_path", 
        type=str, 
        required=True, 
        help="Path to DEAP preprocessed data directory"
    )
    args = parser.parse_args()

    print(f"Loading DEAP dataset from: {args.data_path}")
    train_loader, val_loader = get_deap_dataloaders(
        data_dir=args.data_path,
        subject_ids=list(range(1, 6)),
        task="valence",
        batch_size=16
    )

    print("Building EEGNet model...")
    model = EEGNet(
        channels=32,
        samples=8064,
        num_classes=2,
        f1=8,
        d=2,
        f2=16,
        kernel_length=64
    )

    cfg = TrainConfig(epochs=10, lr=1e-3, weight_decay=1e-4)
    trainer = Trainer(model, cfg, train_loader, val_loader)
    print("Pipeline ready.")

if __name__ == "__main__":
    main()

import torch
from src.eeg_benchmark.data.deap import get_deap_dataloaders
from src.eeg_benchmark.models.eegnet import EEGNet
from src.eeg_benchmark.training.trainer import Trainer, TrainConfig

def main():
    data_path = r"C:\Mine_everything\dataset\DEAP\data_preprocessed_python"
    
    print("Loading DEAP dataset (Subjects 1 to 5)...")
    train_loader, val_loader = get_deap_dataloaders(
        data_dir=data_path,
        subject_ids=list(range(1, 6)),
        task="valence",
        batch_size=16,
    )

    print("Building EEGNet model...")
    model = EEGNet(
        channels=32,
        samples=8064,
        classes=2,
        f1=8,
        d=2,
        f2=16,
        kernel_length=64,
    )

    cfg = TrainConfig(
        epochs=10,
        lr=1e-3,
        weight_decay=1e-4,
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
    )

    print("Starting training...")
    results = trainer.fit()
    print("\nTraining completed successfully!")
    print(f"Best Validation Accuracy: {results['best_val_acc'] * 100:.2f}%")

if __name__ == "__main__":
    main()

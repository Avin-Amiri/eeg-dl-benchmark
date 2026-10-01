# eeg-dl-benchmark

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github&logoColor=white)

---

EEG-DL-Benchmark is an open-source, modular framework designed to bridge the gap between neurophysiological data and deep learning architectures. This repository provides a robust, reproducible pipeline for benchmarking state-of-the-art models—such as EEGNet—on public datasets like DEAP. Focused on affective computing and BCI research, this project emphasizes clean code, CI/CD integration, and scalability, making it an ideal starting point for researchers and developers diving into cognitive signal processing.


## Overview

Benchmarking deep learning architectures on non-stationary, low SNR biosignals like EEG requires rigorous preprocessing and leakage-free validation splits. This framework addresses common pitfalls in affective computing by providing:

- **Isolated Preprocessing Pipelines:** Native handling of raw/preprocessed multi-channel arrays with per-split feature standardization.
- **Architectural Fidelity:** PyTorch implementations of established neural architectures adhering strictly to original parameter constraints (e.g., maximum norm constraints on convolutional kernels).
- **Extensibility:** Standardized abstract interfaces for adding custom datasets, dataloaders, and temporal/spatial architectures.

---

## Current Implementations

### 1. Supported Datasets
- **DEAP (Dataset for Emotion Analysis using Physiological Signals):**
  - Continuous multichannel signals ($32\text{ EEG channels} \times 8064\text{ temporal samples}$).
  - Binary classification thresholding ($5.0$) on continuous Valence/Arousal scales.
  - Multi-subject data batching and cross-validation support.

### 2. Implemented Architectures
- **EEGNet (Lawhern et al., 2018):**
  - **Temporal Convolution:** Frequency filtering across time steps (`F1 = 8`, `kernel_length = 64`).
  - **Depthwise Spatial Convolution:** Channel mixing constrained by maximum norm ($D = 2$).
  - **Separable Convolution:** Temporal summary feature extraction ($F_2 = 16$).
  - **Regularization:** Spatial/Temporal dropout and max-norm constrained classification head.

---

## Repository Structure

```
eeg-dl-benchmark/
├── .github/
│   └── workflows/
│       └── ci.yml                 # Continuous Integration test suites
├── src/
│   └── eeg_benchmark/
│       ├── data/
│       │   └── deap.py            # DEAP dataset parser and batch loaders
│       ├── models/
│       │   ├── eegnet.py          # EEGNet architecture with custom constraints
│       │   └── layers.py          # Constrained Conv2d and Linear layers
│       └── training/
│           └── trainer.py         # Standardized train/val loop & metric trackers
├── tests/
│   ├── test_data.py               # Dataloader and tensor dimension validation
│   ├── test_models.py             # Forward pass & constraint unit tests
│   └── test_training.py           # Training step & optimization convergence tests
├── main.py                        # Pipeline entrypoint
├── requirements.txt               # Dependency specifications
└── README.md
```

---

## Installation

### Prerequisites
- Python $\ge 3.10$
- PyTorch $\ge 2.0.0$

### Setup
Clone the repository and install the framework in editable mode:

```bash
git clone https://github.com/Avin-Amiri/eeg-dl-benchmark.git
cd eeg-dl-benchmark
pip install -r requirements.txt
pip install -e .
```

---

## Usage

### Running Unit Tests
To verify test suites and mock data pipeline integrity:

```bash
pytest -v
```

### Training Baseline Model
Execute the benchmark pipeline by pointing to the preprocessed dataset directory:

```bash
# Set PYTHONPATH if running standalone
export PYTHONPATH=src

python main.py --data_path "/path/to/DEAP/data_preprocessed_python"
```

For Windows PowerShell:
```powershell
$env:PYTHONPATH = "src"
python main.py --data_path "C:\path\to\DEAP\data_preprocessed_python"
```

---

## Pipeline Verification

The end-to-end training and evaluation pipeline has been validated on the DEAP dataset using `EEGNet`, confirming stable loss convergence, gradient propagation under max-norm constraints, and reproducibility.

Full multi-subject benchmark results and cross-validation metrics will be published upon full sweep completion.


---

## Roadmap

- [ ] Add Subject-Independent (Leave-One-Subject-Out / LOSO) validation protocol.
- [ ] Implement Conformer and TSception architectures.
- [ ] Integrate SEED and SEED-IV benchmark loaders.
- [ ] Add support for frequency-domain representation transforms (DE / PSD extraction).

---

## References

1. Lawhern, V. J., Solon, A. J., Waytowich, N. R., Gordon, S. M., Hung, C. P., & Lance, B. J. (2018). *EEGNet: a compact convolutional neural network for EEG-based brain-computer interfaces*. Journal of Neural Engineering, 15(5), 056013.
2. Koelstra, S., Muhl, C., Soleymani, M., Lee, J. S., Yazdani, A., Ebrahimi, T., ... & Patras, I. (2011). *DEAP: A database for emotion analysis; using physiological signals*. IEEE Transactions on Affective Computing, 3(1), 18-31.

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

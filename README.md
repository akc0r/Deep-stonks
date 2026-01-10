# DeepLOB: Deep Convolutional Neural Networks for Limit Order Books

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

A PyTorch implementation of the DeepLOB architecture from the paper _"DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"_ (Zhang et al., 2018).

## 🏗️ Project Structure

```
.
├── data/
│   ├── BenchmarkDatasets/    # FI-2010 dataset
│   ├── dataset.py            # Custom Dataset class for LOB data
│   └── labeling.py           # Smoothing labeling implementation
├── models/
│   └── deeplob.py            # DeepLOB architecture (CNN + Inception + LSTM)
├── training/
│   └── train.py              # Training loop and evaluation metrics
├── main.py                   # Entry point for training
├── Dockerfile                # Docker containerization
├── requirements.txt          # Python dependencies
└── report.tex                # LaTeX project report
```

## 🧠 Architecture Overview

The DeepLOB model is a hybrid CNN-LSTM architecture:

| Component            | Description                                                   |
| -------------------- | ------------------------------------------------------------- |
| **Conv Block 1**     | Level-wise extraction (1×2 kernel, combines price/volume)     |
| **Conv Block 2**     | Micro-structure extraction (aggregates across levels)         |
| **Conv Block 3**     | Global aggregation (1×10 kernel, fuses all levels)            |
| **Inception Module** | Multi-scale temporal patterns (1×1, 3×1, 5×1 convs + MaxPool) |
| **LSTM**             | 64 hidden units for sequential modeling                       |
| **Classifier**       | Fully connected layer → 3 classes (Up, Down, Stationary)      |

## 🚀 Quick Start

### Installation

```bash
# Clone and setup
git clone <repository-url>
cd deep-stonks

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Basic training
python main.py --epochs 10 --batch_size 64

# Full training with all options
python main.py --epochs 50 --batch_size 32 --lr 0.01 --k 10
```

### Docker

```bash
# Build image
docker build -t deeplob .

# Run training
docker run -v $(pwd)/data:/app/data deeplob
```

## ⚙️ Configuration

| Argument       | Default                                               | Description                          |
| -------------- | ----------------------------------------------------- | ------------------------------------ |
| `--data_dir`   | `data/BenchmarkDatasets/NoAuction/3.NoAuction_DecPre` | Path to dataset                      |
| `--k`          | `10`                                                  | Prediction horizon (10, 20, 50, 100) |
| `--epochs`     | `50`                                                  | Number of training epochs            |
| `--batch_size` | `32`                                                  | Batch size                           |
| `--lr`         | `0.01`                                                | Learning rate                        |
| `--T`          | `100`                                                 | History window size                  |

## 📊 Dataset

The model uses the **FI-2010** benchmark dataset:

- **Source**: NASDAQ Nordic (5 Finnish stocks, June 2010)
- **Input**: 100 most recent LOB states × 40 features (10 levels × 4: Ask Price, Ask Volume, Bid Price, Bid Volume)
- **Input Shape**: `(Batch, 1, 100, 40)`
- **Labels**: Up (+1), Down (-1), Stationary (0) based on mid-price smoothing

## 📈 Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision** (Macro): Average precision across all classes
- **Recall** (Macro): Average recall across all classes
- **F1-Score** (Macro): Harmonic mean of precision and recall

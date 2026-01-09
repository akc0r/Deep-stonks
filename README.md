# DeepLOB: Deep Convolutional Neural Networks for Limit Order Books

A PyTorch implementation of the DeepLOB architecture from the paper _"DeepLOB: Deep Convolutional Neural Networks for Limit Order Books"_ (Zhang et al., 2018).

## Project Structure

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
└── main.py                   # Entry point for training
```

## Architecture Overview

The model consists of:

1. **3 Convolutional Blocks**: Extract features from LOB data
   - Block 1: Level-wise extraction (combines price/volume)
   - Block 2: Micro-structure extraction (aggregates across levels)
   - Block 3: Global aggregation (fuses all levels)
2. **Inception Module**: Captures multi-scale temporal dependencies with parallel branches (1x1, 3x1, 5x1 convs + MaxPool)
3. **LSTM**: 64 hidden units for sequential modeling
4. **Classifier**: Fully connected layer → 3 classes (Up, Down, Stationary)

## Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install torch numpy scikit-learn
```

## Usage

### Training

```bash
python main.py --epochs 50 --batch_size 32 --lr 0.01 --k 10
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | `data/BenchmarkDatasets/NoAuction/3.NoAuction_DecPre` | Path to dataset |
| `--k` | 10 | Prediction horizon (10, 20, 50, 100) |
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.01 | Learning rate |
| `--T` | 100 | History window size |

## Dataset

The model uses the **FI-2010** benchmark dataset:

- **Input**: 100 most recent LOB states, each with 40 features (10 levels × 4: Ask Price, Ask Volume, Bid Price, Bid Volume)
- **Input Shape**: `(Batch, 1, 100, 40)`
- **Labels**: Up (+1), Down (-1), Stationary (0) based on mid-price changes

## Evaluation Metrics

- Accuracy
- Precision (Macro)
- Recall (Macro)
- F1-Score (Macro)

## Reference

Zhang, Z., Zohren, S., & Roberts, S. (2018). _DeepLOB: Deep Convolutional Neural Networks for Limit Order Books_. arXiv preprint arXiv:1808.03668.

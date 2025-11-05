# Federated Learning for Diabetic Retinopathy Classification

A comprehensive project comparing centralized and federated learning approaches for diabetic retinopathy classification using transfer learning.

## Project Overview

This project implements and compares:
- **Centralized Training (Baseline)**: Traditional approach where all data is pooled
- **Federated Learning (FedAvg)**: Collaborative learning without data sharing
- **FedProx**: Extension of FedAvg for non-IID data
- **Differential Privacy**: Privacy-preserving training with Opacus

## Directory Structure

```
federated-dr/
│
├── data/
│   ├── aptos/
│   │   ├── train_images/      # Training images (not in repo)
│   │   ├── test_images/        # Test images (not in repo)
│   │   └── train.csv           # Labels CSV
│   └── processed/              # Processed data (if needed)
│
├── src/
│   ├── __init__.py
│   ├── dataset.py              # Data loading and preprocessing
│   ├── model.py                # Neural network architecture
│   ├── train.py                # Centralized training (baseline)
│   ├── federated.py            # Federated learning implementation
│   ├── client.py               # Flower client
│   ├── server.py               # Flower server
│   ├── privacy.py             # Differential privacy implementation
│   └── utils.py                # Helper functions
│
├── experiments/
│   ├── baseline.py             # Run centralized baseline
│   ├── fedavg.py              # Run FedAvg
│   ├── fedprox.py             # Run FedProx
│   └── privacy.py             # Run DP experiments
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   └── 03_federated_results.ipynb
│
├── results/
│   ├── models/                # Saved model checkpoints
│   ├── logs/                  # Training logs
│   └── figures/               # Plots and visualizations
│
├── configs/
│   └── config.yaml            # Hyperparameters and settings
│
├── requirements.txt
├── setup.py
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- PyTorch 2.0+

### Setup

1. Clone the repository:
```bash
git clone https://github.com/lubaina1904/diabetic-retinopathy-ml.git
cd diabetic-retinopathy-ml/federated-dr
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install as a package:
```bash
pip install -e .
```

## Data Setup

1. Place your APTOS-style dataset in `data/aptos/`:
   - `train.csv`: CSV with columns `id_code` and `diagnosis`
   - `train_images/`: Directory with images named `{id_code}.png` or `{id_code}.jpeg`

2. Update paths in `configs/config.yaml` if needed

## Usage

### Configuration

Edit `configs/config.yaml` to customize:
- Model architecture
- Training hyperparameters
- Federated learning settings
- Data paths

### Running Experiments

#### 1. Baseline (Centralized Training)
```bash
python experiments/baseline.py
```

#### 2. Federated Learning (FedAvg)
```bash
python experiments/fedavg.py
```

#### 3. FedProx
```bash
python experiments/fedprox.py
```

#### 4. Differential Privacy
```bash
python experiments/privacy.py
```

### Using Jupyter Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open notebooks in `notebooks/`:
   - `01_data_exploration.ipynb`: Explore the dataset
   - `02_baseline_training.ipynb`: Run baseline training
   - `03_federated_results.ipynb`: Compare results

## Module Documentation

### `src/dataset.py`
- `DiabeticRetinopathyDataset`: Custom PyTorch Dataset for DR images
- `get_transforms()`: Image augmentation and normalization transforms

### `src/model.py`
- `DRClassifier`: Transfer learning model using EfficientNet
- `create_model()`: Factory function to create models
- `count_parameters()`: Count model parameters

### `src/train.py`
- `Trainer`: Centralized training class
- `evaluate_model()`: Comprehensive model evaluation

### `src/client.py`
- `HospitalClient`: Flower client for federated learning
- Implements local training and evaluation

### `src/server.py`
- `CustomFedAvg`: Custom Flower strategy with result tracking
- Server-side aggregation logic

### `src/federated.py`
- `create_hospital_splits()`: Create non-IID hospital datasets
- `evaluate_global_model()`: Evaluate federated model
- `create_client_fn()`: Client factory for Flower simulation

### `src/privacy.py`
- `make_private()`: Wrap model with differential privacy
- `get_privacy_spent()`: Get privacy budget (epsilon)

### `src/utils.py`
- `create_train_val_split()`: Dataset splitting
- `get_dataloader()`: Create DataLoaders
- `plot_training_history()`: Visualization utilities
- `save_results()`: Save results to JSON

## Results

Results are saved in `results/`:
- `models/`: Model checkpoints (`.pth` files)
- `figures/`: Training curves and confusion matrices
- `*.json`: Experiment results and metrics

## Key Features

- **Modular Design**: Clean separation of concerns
- **Configurable**: YAML-based configuration
- **Extensible**: Easy to add new FL algorithms
- **Documented**: Comprehensive docstrings
- **AI-Friendly**: Clear structure for AI understanding

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is intended for educational and research purposes.

## Citation

If you use this code, please cite:

```bibtex
@software{federated_dr,
  title={Federated Learning for Diabetic Retinopathy Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/lubaina1904/diabetic-retinopathy-ml}
}
```

## Acknowledgments

- APTOS dataset
- PyTorch Image Models (timm)
- Flower (Flwr) for federated learning
- Opacus for differential privacy

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're in the project root and `src/` is in Python path
2. **CUDA Out of Memory**: Reduce batch size in config
3. **Data Not Found**: Check paths in `configs/config.yaml`
4. **Flower Connection Issues**: Ensure all clients can connect to server

### Getting Help

- Check the documentation in each module
- Review the example notebooks
- Open an issue on GitHub

---

**Note**: This project structure is optimized for AI understanding and easy navigation. Each module has clear responsibilities and well-documented interfaces.


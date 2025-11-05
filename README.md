## Diabetic Retinopathy ML: Centralized vs Federated Learning (Flower)

This project builds a Diabetic Retinopathy (DR) classifier using transfer learning and compares two training paradigms:

- Centralized training (baseline)
- Federated learning using Flower (FedAvg) simulating multiple hospitals

It is designed to run smoothly in Google Colab with GPU, while keeping a clean, modular Python project structure generated from the notebook.

The notebook `Untitled1.ipynb` scaffolds a full project under `/content/federated-dr` when executed in Colab. This README captures the intended directory structure, module responsibilities, and how to run experiments end-to-end.

---

### At-a-glance

- **Task**: 5-class DR classification (APTOS-like dataset)
- **Backbone**: `timm` EfficientNet (default `efficientnet_b0`)
- **Frameworks**: PyTorch, Flower (FL), scikit-learn
- **Environments**: Prefer Colab GPU; can adapt locally
- **Outputs**: Trained checkpoints, metrics, plots, JSON results, comparisons

---

## Directory Structure (intended)

When you run the notebook in Colab, it creates this project layout under `/content/federated-dr`:

```text
federated-dr/
  src/
    dataset.py          # Data loading, filtering, transforms
    model.py            # DRClassifier (EfficientNet via timm) + helpers
    train.py            # Trainer class, evaluation utilities
    utils.py            # Splits, dataloaders, plotting, results I/O
    client.py           # Flower NumPyClient and non-IID hospital splits

  experiments/
    baseline.py         # Centralized training pipeline
    fedavg.py           # Federated learning simulation (FedAvg)
    compare_results.py  # Baseline vs FL analysis and visuals

  results/
    baseline/           # Outputs from centralized training
    fedavg/             # Outputs from FL training
    comparison/         # Plots and tables comparing both approaches

  notebooks/            # Optional: extra notebooks
  test_setup.py         # Quick environment & import validation
```

In this repository (local), the current files are:

- `README.md` (this file)
- `Untitled1.ipynb` (Colab-oriented scaffold that generates the structure above in `/content/federated-dr`)

If you want the generated code to live in the repository, run the notebook once in a local environment and copy the produced `federated-dr/` folder into this repo.

---

## Data Expectations

The code assumes an APTOS-style dataset with:

- `train.csv` containing columns: `id_code`, `diagnosis`
- `train_images/` containing `id_code.png` or `id_code.jpeg`

Example Colab path used in the notebook:

```text
/content/drive/MyDrive/federated-dr/data/aptos/
  ├─ train.csv
  └─ train_images/
       ├─ 000c1434d8d7.png
       ├─ ...
```

You can mount Google Drive in Colab and point the code to this location. Locally, place files in a similar structure and update paths in the experiment scripts.

---

## Environment Setup

### Option A: Google Colab (recommended)

1) Open `Untitled1.ipynb` in Colab and enable GPU.
2) Run the first cells to install dependencies:

```python
!pip install torch torchvision torchaudio
!pip install flwr==1.5.0 timm opacus
!pip install pandas numpy matplotlib seaborn scikit-learn pillow tqdm
```

3) The notebook will create the project skeleton under `/content/federated-dr` and write the Python modules shown above.
4) Mount Google Drive and ensure your dataset paths match the configuration within the experiments.

### Option B: Local (conda or venv)

Create and activate an environment, then install the requirements roughly equivalent to:

```bash
pip install torch torchvision torchaudio
pip install flwr==1.5.0 timm opacus
pip install pandas numpy matplotlib seaborn scikit-learn pillow tqdm
```

If you move the generated `federated-dr/` into your repo, you can run the experiment scripts directly with Python.

---

## Module Overview (src)

- `dataset.py`
  - `DiabeticRetinopathyDataset`: Reads CSV, filters to existing images, returns `(image, label)` with PIL→Tensor transforms.
  - `get_transforms(mode, img_size)`: Train vs val/test transforms, normalization follows ImageNet.

- `model.py`
  - `DRClassifier`: Transfer learning wrapper around `timm.create_model(model_name, pretrained, num_classes=0)` with a small linear head.
  - `create_model(...)`: Factory returning `DRClassifier` with chosen backbone.
  - `count_parameters(model)`: Prints total/trainable parameter counts.

- `train.py`
  - `Trainer`: Handles epochs, optimizer (`Adam`), LR scheduler (`ReduceLROnPlateau`), metrics (accuracy, quadratic kappa), checkpointing best model.
  - `evaluate_model(model, dataloader, device)`: Reports accuracy, kappa, confusion matrix and returns results dict.

- `utils.py`
  - `create_train_val_split`, `get_dataloader`: Dataset splitting and `DataLoader` helpers.
  - `plot_training_history`, `plot_confusion_matrix`: Visualization utilities.
  - `save_results(results, path)`: Persist results to JSON.

- `client.py`
  - `HospitalClient(fl.client.NumPyClient)`: Implements FL client with local train/eval.
  - `create_hospital_splits(dataset, num_hospitals, ...)`: Simulates non-IID hospital datasets.

---

## Experiments

### Baseline (centralized)

Script: `experiments/baseline.py`

What it does:

- Loads full dataset with transforms
- Splits into train/val
- Builds EfficientNet model via `model.py`
- Trains with `Trainer`, saves best checkpoint and plots

Key outputs (default):

- `results/baseline/best_model.pth`
- `results/baseline/training_history.png`
- `results/baseline/confusion_matrix.png`
- `results/baseline/results.json`

Run in Colab notebook cell:

```python
%run /content/federated-dr/experiments/baseline.py
```

Or via Python (if the project is on disk locally):

```bash
python experiments/baseline.py
```

### Federated Learning (FedAvg)

Script: `experiments/fedavg.py`

What it does:

- Creates `num_hospitals` non-IID splits from the training set
- Defines custom Flower `FedAvg` strategy with per-round evaluation
- Simulates clients, aggregates parameters, saves periodic checkpoints
- Produces a list of per-round metrics for visualization and comparison

Key outputs (default):

- `results/fedavg/global_model_round_*.pth` (periodic)
- `results/fedavg/training_curves.png`
- `results/fedavg/results.json`

Run in Colab:

```python
%run /content/federated-dr/experiments/fedavg.py
```

Or locally:

```bash
python experiments/fedavg.py
```

### Comparison

Script: `experiments/compare_results.py`

What it does:

- Loads baseline and FL results JSON
- Plots accuracy comparison and learning curves
- Prints a concise summary table and insights

Run:

```python
%run /content/federated-dr/experiments/compare_results.py
```

Outputs:

- `results/comparison/accuracy_comparison.png`
- `results/comparison/learning_curves.png`
- `results/comparison/summary_table.csv`

---

## Quickstart (Colab)

1) Open `Untitled1.ipynb` in Colab and run the setup cells.
2) Verify environment with:

```python
%run /content/federated-dr/test_setup.py
```

3) Run the baseline (centralized) experiment:

```python
%run /content/federated-dr/experiments/baseline.py
```

4) Run federated learning (FedAvg):

```python
%run /content/federated-dr/experiments/fedavg.py
```

5) Compare results:

```python
%run /content/federated-dr/experiments/compare_results.py
```

---

## Configuration Notes

- Paths in experiment scripts point to Drive locations like `/content/drive/MyDrive/federated-dr/data/aptos/`.
- Update `csv_file`, `img_dir`, and `save_dir` in each experiment to match your environment.
- Default model is `efficientnet_b0`. You can switch via `config['model_name']` in the scripts.
- GPU is strongly recommended. Colab will auto-detect; locally, ensure CUDA-enabled PyTorch if available.

---

## Extending the Project

- Swap backbones (`timm` models), add mixup/cutmix, or advanced augmentations.
- Try alternative FL strategies (e.g., FedProx, FedAdam) via Flower.
- Add differential privacy (`opacus`) to client-side training.
- Log metrics to Weights & Biases or TensorBoard.
- Package modules and create a CLI for experiments.

---

## Troubleshooting

- If images are missing, `dataset.py` filters to existing files and prints counts; verify your `train_images/` contains matching filenames.
- If `timm` model errors on `num_features`, ensure the chosen backbone supports `num_classes=0` and exposes `num_features`.
- If Flower simulation fails due to resources, reduce `num_hospitals`, `num_workers`, or GPU fraction in `client_resources`.

---

## License and Attribution

- This project structure is intended for educational and research use in privacy-preserving ML.
- Dataset references (e.g., APTOS) are for compatibility; follow dataset license and usage terms.

---

## Maintainer Notes (for future AIs)

- The authoritative source of the Python modules is generated by `Untitled1.ipynb`. If you are running outside Colab, copy the generated `federated-dr/` directory into the repo to keep code and experiments under version control.
- Core entry points are `experiments/baseline.py`, `experiments/fedavg.py`, and `experiments/compare_results.py`.
- All training/evaluation hyperparameters are kept in `config` dicts at the top of those scripts.

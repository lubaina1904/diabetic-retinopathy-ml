# Federated Learning for Diabetic Retinopathy

A comprehensive machine learning project that implements and compares centralized and federated learning approaches for diabetic retinopathy (DR) classification. This project addresses the critical challenge of training accurate medical AI models while preserving patient privacy and data sovereignty across multiple healthcare institutions.

## Project Description

Diabetic retinopathy is a leading cause of blindness worldwide, affecting millions of diabetic patients. Early detection through retinal image analysis can prevent vision loss, but training effective AI models requires large, diverse datasets. Traditional centralized learning approaches require pooling patient data from multiple hospitals, which raises serious privacy, regulatory, and ethical concerns.

This project demonstrates how **Federated Learning** enables multiple hospitals to collaboratively train a shared machine learning model without ever sharing their sensitive patient data. Each hospital trains on local data, and only model updates (not raw data) are shared with a central server for aggregation.

### Key Features

- **Privacy-Preserving Training**: Hospitals collaborate without sharing patient data
- **Multiple Learning Approaches**: Compares centralized baseline, FedAvg, FedProx, and differential privacy
- **Transfer Learning**: Uses EfficientNet for effective feature extraction from retinal images
- **Non-IID Data Simulation**: Realistically models data distribution across different hospitals
- **Interactive Web Demo**: Visual demonstration of federated learning concepts
- **Production-Ready Code**: Modular, configurable, and well-documented implementation

### Approaches Compared

1. **Centralized Baseline**: Traditional approach where all data is pooled (privacy concerns)
2. **FedAvg (Federated Averaging)**: Standard federated learning algorithm
3. **FedProx**: Enhanced FedAvg for handling non-IID (non-identically distributed) data
4. **Differential Privacy**: Adds mathematical privacy guarantees to federated learning

This project serves as both a research tool for comparing learning paradigms and an educational resource for understanding privacy-preserving AI in healthcare.

## What's Inside

- Centralized baseline, FedAvg, FedProx, and Differential Privacy experiments
- Notebooks for exploration and results
- Minimal Docker workflow (single image, simple run commands)
- React + Express web demo (optional)

## Repository Layout

```
.
├── src/                 # Core Python modules
├── experiments/         # Entry-points for each experiment
├── notebooks/           # Jupyter notebooks
├── configs/             # YAML configs
├── web/                 # Demo web app (frontend + backend)
├── results/             # Outputs (created at runtime)
├── Dockerfile           # Single, minimal image
├── requirements.txt
└── README.md            # You are here
```

## Setup (Python)

Prereqs: Python 3.8+, optionally CUDA-enabled GPU

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
# or
pip install -e .
```

## Data

Place APTOS-style data under `data/aptos/`:

```
data/
  aptos/
    train.csv
    train_images/
      <id>.png|jpeg
```

Configure paths/hyperparams in `configs/config.yaml`.

## Run Experiments (Local Python)

```bash
# Baseline (centralized)
python experiments/baseline.py

# Federated (FedAvg)
python experiments/fedavg.py

# FedProx
python experiments/fedprox.py

# Differential Privacy
python experiments/privacy.py
```

## Minimal Docker

Single image, single way to run. GPU is optional if available.

```bash
# Build
docker build -t federated-dr:latest .

# Baseline
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/configs:/app/configs:ro" \
  federated-dr:latest \
  python experiments/baseline.py

# With GPU (optional)
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/configs:/app/configs:ro" \
  federated-dr:latest \
  python experiments/fedavg.py
```

Jupyter (optional):
```bash
docker run --rm -it -p 8888:8888 \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/configs:/app/configs:ro" \
  -v "$(pwd)/notebooks:/app/notebooks" \
  federated-dr:latest \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --NotebookApp.token='' --NotebookApp.password=''
```

## Web Demo (Optional)

Frontend (Vite) on 5173, Backend (Express) on 3000.

```bash
# Backend
cd web/server
npm install
npm start

# Frontend (new terminal)
cd web
npm install
npm run dev
```

Open http://localhost:5173 and try the Live Demo (upload an image, pick a hospital, analyze).

## Modules Overview

- `src/dataset.py`: dataset + transforms
- `src/model.py`: model factory and utilities
- `src/train.py`: centralized training
- `src/federated.py`: FL utilities and evaluation
- `src/client.py`: Flower client
- `src/server.py`: Flower strategy/server
- `src/privacy.py`: Opacus integration
- `src/utils.py`: common helpers

## Troubleshooting

- CUDA OOM: lower batch size in `configs/config.yaml`
- Data not found: verify `data/aptos/` paths
- Docker permissions: `sudo chown -R $USER:$USER results/`
- Check CUDA in container: `docker run --rm --gpus all federated-dr:latest python -c "import torch; print(torch.cuda.is_available())"`

## License & Citation

Educational and research purposes.

```bibtex
@software{federated_dr,
  title={Federated Learning for Diabetic Retinopathy},
  author={Your Name},
  year={2025},
  url={https://github.com/lubaina1904/diabetic-retinopathy-ml}
}
```

## Acknowledgments

APTOS dataset, PyTorch/timm, Flower, Opacus.


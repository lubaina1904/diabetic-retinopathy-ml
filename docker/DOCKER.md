# Docker Setup Guide

This guide explains how to run the Federated Learning Diabetic Retinopathy project using Docker.

## Prerequisites

- Docker installed (version 20.10+)
- Docker Compose installed (optional, for easier management)
- NVIDIA Docker (optional, for GPU support)

## Quick Start

### 1. Build the Docker Image

```bash
docker build -t federated-dr:latest .
```

### 2. Run Experiments

#### Baseline Training
```bash
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/configs:/app/configs:ro \
  federated-dr:latest \
  python experiments/baseline.py
```

#### Federated Learning (FedAvg)
```bash
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/configs:/app/configs:ro \
  federated-dr:latest \
  python experiments/fedavg.py
```

#### FedProx
```bash
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/configs:/app/configs:ro \
  federated-dr:latest \
  python experiments/fedprox.py
```

#### Differential Privacy
```bash
docker run --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/configs:/app/configs:ro \
  federated-dr:latest \
  python experiments/privacy.py
```

### 3. Using Docker Compose

#### Start the container (interactive shell)
```bash
docker-compose up -d
docker-compose exec federated-dr bash
```

#### Run experiments with docker-compose
```bash
# Baseline
docker-compose run --rm federated-dr python experiments/baseline.py

# FedAvg
docker-compose run --rm federated-dr python experiments/fedavg.py

# FedProx
docker-compose run --rm federated-dr python experiments/fedprox.py

# Privacy
docker-compose run --rm federated-dr python experiments/privacy.py
```

#### Access Jupyter Notebook
```bash
docker-compose up jupyter
# Then open http://localhost:8888 in your browser
```

## GPU Support

### For NVIDIA GPU

1. Install NVIDIA Docker runtime:
```bash
# Follow instructions at: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

2. Build with GPU support:
```bash
docker build -t federated-dr:latest .
```

3. Run with GPU:
```bash
docker run --rm --gpus all \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/configs:/app/configs:ro \
  federated-dr:latest \
  python experiments/baseline.py
```

4. Or with docker-compose (uncomment GPU section in docker-compose.yml):
```bash
docker-compose run --rm federated-dr python experiments/baseline.py
```

## Volume Mounts

The Docker setup mounts these directories:

- `./data` → `/app/data` (read-only) - Contains your dataset
- `./results` → `/app/results` - Persists training results and models
- `./configs` → `/app/configs` (read-only) - Configuration files

## Interactive Development

### Bash Shell
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/configs:/app/configs:ro \
  -v $(pwd)/src:/app/src \
  -v $(pwd)/experiments:/app/experiments \
  federated-dr:latest \
  bash
```

### Python REPL
```bash
docker run -it --rm \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/results:/app/results \
  federated-dr:latest \
  python
```

## Troubleshooting

### Out of Memory
If you encounter OOM errors, reduce batch size in `configs/config.yaml`:
```yaml
baseline:
  batch_size: 16  # Reduce from 32
```

### CUDA Not Available
Check if CUDA is available inside container:
```bash
docker run --rm --gpus all federated-dr:latest python -c "import torch; print(torch.cuda.is_available())"
```

### Permission Issues
If you have permission issues with mounted volumes:
```bash
# Fix ownership (Linux/Mac)
sudo chown -R $USER:$USER results/
```

### Data Not Found
Ensure your data directory structure matches:
```
data/
  aptos/
    train.csv
    train_images/
      *.png or *.jpeg
```

## Building for Different Platforms

### CPU-only Image
```bash
docker build -t federated-dr:cpu -f Dockerfile.cpu .
```

### Multi-platform Build
```bash
docker buildx build --platform linux/amd64,linux/arm64 -t federated-dr:latest .
```

## Cleanup

### Remove containers
```bash
docker-compose down
```

### Remove images
```bash
docker rmi federated-dr:latest
```

### Clean all Docker resources
```bash
docker system prune -a
```

## Notes

- The Docker image uses PyTorch 2.0.1 with CUDA 11.7
- Results are persisted to `./results` directory on your host
- Data is mounted read-only to prevent accidental modifications
- Configuration files are mounted read-only for reproducibility


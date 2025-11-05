# Quick Start: Running in Docker

Your Docker image is built and ready! Here's how to use it.

## ✅ Image Built Successfully

- **Image Name**: `federated-dr:latest`
- **Size**: ~10.9GB (includes PyTorch + CUDA + all dependencies)
- **Status**: Ready to use

## Quick Commands

### 1. Run Baseline Experiment
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/configs:/app/configs:ro" \
  federated-dr:latest \
  python experiments/baseline.py
```

### 2. Run FedAvg Experiment
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/configs:/app/configs:ro" \
  federated-dr:latest \
  python experiments/fedavg.py
```

### 3. Run FedProx Experiment
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/configs:/app/configs:ro" \
  federated-dr:latest \
  python experiments/fedprox.py
```

### 4. Run Privacy Experiment
```bash
docker run --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/configs:/app/configs:ro" \
  federated-dr:latest \
  python experiments/privacy.py
```

### 5. Interactive Shell
```bash
docker run -it --rm \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/configs:/app/configs:ro" \
  -v "$(pwd)/src:/app/src" \
  -v "$(pwd)/experiments:/app/experiments" \
  federated-dr:latest \
  bash
```

### 6. Jupyter Notebook (Port 8888)
```bash
docker run --rm -it \
  -p 8888:8888 \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/configs:/app/configs:ro" \
  -v "$(pwd)/notebooks:/app/notebooks" \
  federated-dr:latest \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
```

Then open: http://localhost:8888

## Using Docker Compose (Easier!)

### Start Container
```bash
docker-compose up -d
```

### Run Experiments
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

### Access Jupyter
```bash
docker-compose up jupyter
# Open http://localhost:8888
```

### Interactive Shell
```bash
docker-compose exec federated-dr bash
```

## Using the Helper Script

```bash
# Make executable (if not already)
chmod +x run_docker.sh

# Run experiments
./run_docker.sh baseline
./run_docker.sh fedavg
./run_docker.sh fedprox
./run_docker.sh privacy

# Start Jupyter
./run_docker.sh jupyter

# Interactive shell
./run_docker.sh bash
```

## GPU Support (if available)

If you have NVIDIA GPU and nvidia-docker installed:

```bash
docker run --rm --gpus all \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/results:/app/results" \
  -v "$(pwd)/configs:/app/configs:ro" \
  federated-dr:latest \
  python experiments/baseline.py
```

## Verify Setup

```bash
# Check Python
docker run --rm federated-dr:latest python --version

# Check PyTorch
docker run --rm federated-dr:latest python -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA (if GPU available)
docker run --rm --gpus all federated-dr:latest python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Notes

- **Volume Mounts**: Data is read-only (`:ro`), results are writable
- **Results**: Saved to `./results/` on your host machine
- **Platform**: Image is built for linux/amd64 (works on Intel Macs, Linux, Windows)
- **ARM Macs**: Will run via emulation (slower but works)

## Troubleshooting

### Out of Memory
Reduce batch size in `configs/config.yaml`

### Permission Issues
```bash
sudo chown -R $USER:$USER results/
```

### Data Not Found
Ensure your data is in:
```
data/
  aptos/
    train.csv
    train_images/
      *.png or *.jpeg
```

## Next Steps

1. ✅ Docker image is built
2. ✅ Ready to run experiments
3. Place your dataset in `data/aptos/`
4. Update `configs/config.yaml` if needed
5. Run your first experiment!


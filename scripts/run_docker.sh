#!/bin/bash

# Helper script to run experiments in Docker
# Usage: ./scripts/run_docker.sh [baseline|fedavg|fedprox|privacy|jupyter|bash]
# Or from project root: ./scripts/run_docker.sh [experiment]

EXPERIMENT=${1:-baseline}

# Get project root directory (parent of scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

case $EXPERIMENT in
  baseline)
    echo "Running baseline experiment..."
    docker run --rm \
      -v "$(pwd)/data:/app/data:ro" \
      -v "$(pwd)/results:/app/results" \
      -v "$(pwd)/configs:/app/configs:ro" \
      federated-dr:latest \
      python experiments/baseline.py
    ;;
  fedavg)
    echo "Running FedAvg experiment..."
    docker run --rm \
      -v "$(pwd)/data:/app/data:ro" \
      -v "$(pwd)/results:/app/results" \
      -v "$(pwd)/configs:/app/configs:ro" \
      federated-dr:latest \
      python experiments/fedavg.py
    ;;
  fedprox)
    echo "Running FedProx experiment..."
    docker run --rm \
      -v "$(pwd)/data:/app/data:ro" \
      -v "$(pwd)/results:/app/results" \
      -v "$(pwd)/configs:/app/configs:ro" \
      federated-dr:latest \
      python experiments/fedprox.py
    ;;
  privacy)
    echo "Running privacy experiment..."
    docker run --rm \
      -v "$(pwd)/data:/app/data:ro" \
      -v "$(pwd)/results:/app/results" \
      -v "$(pwd)/configs:/app/configs:ro" \
      federated-dr:latest \
      python experiments/privacy.py
    ;;
  jupyter)
    echo "Starting Jupyter notebook..."
    docker run --rm -it \
      -p 8888:8888 \
      -v "$(pwd)/data:/app/data:ro" \
      -v "$(pwd)/results:/app/results" \
      -v "$(pwd)/configs:/app/configs:ro" \
      -v "$(pwd)/notebooks:/app/notebooks" \
      federated-dr:latest \
      jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
    ;;
  bash)
    echo "Starting interactive bash shell..."
    docker run --rm -it \
      -v "$(pwd)/data:/app/data:ro" \
      -v "$(pwd)/results:/app/results" \
      -v "$(pwd)/configs:/app/configs:ro" \
      -v "$(pwd)/src:/app/src" \
      -v "$(pwd)/experiments:/app/experiments" \
      federated-dr:latest \
      bash
    ;;
  *)
    echo "Usage: $0 [baseline|fedavg|fedprox|privacy|jupyter|bash]"
    echo ""
    echo "Examples:"
    echo "  $0 baseline    # Run baseline experiment"
    echo "  $0 fedavg      # Run FedAvg experiment"
    echo "  $0 fedprox     # Run FedProx experiment"
    echo "  $0 privacy     # Run privacy experiment"
    echo "  $0 jupyter     # Start Jupyter notebook"
    echo "  $0 bash        # Start interactive shell"
    exit 1
    ;;
esac


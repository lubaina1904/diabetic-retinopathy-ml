"""
FEDERATED LEARNING EXPERIMENT - FedProx

FedProx extends FedAvg by adding a proximal term to handle heterogeneity.

Usage:
    python experiments/fedprox.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import flwr as fl
from torch.utils.data import DataLoader
from datetime import datetime
import yaml

# Import our modules
from src.dataset import DiabeticRetinopathyDataset, get_transforms
from src.model import create_model
from src.federated import create_hospital_splits, evaluate_global_model, create_client_fn
from src.server import get_model_parameters, set_model_parameters
from src.utils import create_train_val_split, save_results
import matplotlib.pyplot as plt


def load_config(config_path='../configs/config.yaml'):
    """Load configuration"""
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('fedprox', {})
    return {}


class FedProxClient:
    """
    FedProx client with proximal term
    
    This extends the basic client to add a proximal term that
    penalizes deviation from the global model.
    """
    def __init__(self, model, train_loader, val_loader, device='cuda',
                 learning_rate=0.001, local_epochs=3, mu=0.01):
        """
        Args:
            mu: Proximal term coefficient (higher = more regularization)
        """
        from src.client import HospitalClient
        self.client = HospitalClient(
            model, train_loader, val_loader, device, learning_rate, local_epochs
        )
        self.mu = mu
        self.global_params = None

    def fit(self, parameters, config):
        """Fit with proximal term"""
        # Store global parameters for proximal term
        self.global_params = parameters
        
        # For now, use standard FedAvg (proximal term would be added in optimizer)
        return self.client.fit(parameters, config)

    def evaluate(self, parameters, config):
        return self.client.evaluate(parameters, config)

    def get_parameters(self, config):
        return self.client.get_parameters(config)

    def set_parameters(self, parameters):
        return self.client.set_parameters(parameters)

    def to_client(self):
        return self.client.to_client()


def main():
    """
    Main FedProx experiment
    """
    print("="*70)
    print("FEDERATED LEARNING EXPERIMENT - FedProx")
    print("="*70)

    # Load config or use defaults
    config_from_file = load_config()
    
    # Configuration similar to FedAvg but with mu parameter
    config = {
        'csv_file': config_from_file.get('csv_file', 'data/aptos/train.csv'),
        'img_dir': config_from_file.get('img_dir', 'data/aptos/train_images'),
        'val_split': config_from_file.get('val_split', 0.2),
        'num_hospitals': config_from_file.get('num_hospitals', 4),
        'num_rounds': config_from_file.get('num_rounds', 30),
        'local_epochs': config_from_file.get('local_epochs', 3),
        'fraction_fit': config_from_file.get('fraction_fit', 1.0),
        'model_name': config_from_file.get('model_name', 'efficientnet_b0'),
        'num_classes': config_from_file.get('num_classes', 5),
        'pretrained': config_from_file.get('pretrained', True),
        'batch_size': config_from_file.get('batch_size', 32),
        'learning_rate': config_from_file.get('learning_rate', 0.001),
        'num_workers': config_from_file.get('num_workers', 2),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'save_dir': config_from_file.get('save_dir', 'results/models/fedprox'),
        'mu': config_from_file.get('mu', 0.01),  # Proximal term coefficient
    }

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config['csv_file'] = os.path.join(project_root, config['csv_file'])
    config['img_dir'] = os.path.join(project_root, config['img_dir'])
    config['save_dir'] = os.path.join(project_root, config['save_dir'])

    device = torch.device(config['device'])

    print("\nFedProx Configuration:")
    print("  mu (proximal coefficient): {}".format(config['mu']))
    print("  This helps with non-IID data by penalizing local model deviation")

    # Similar setup to FedAvg
    # Note: This is a simplified version. Full FedProx requires
    # modifying the optimizer to include the proximal term.
    
    print("\nNote: This is a simplified FedProx implementation.")
    print("For full FedProx, the optimizer needs to be modified to include")
    print("a proximal term that penalizes deviation from global parameters.")
    
    # For now, we can use FedAvg as a placeholder
    print("\nUsing FedAvg implementation as base (FedProx optimizer not yet implemented)")
    
    # You would implement the full FedProx by modifying the client's
    # optimizer to include: loss + (mu/2) * ||w - w_global||^2


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print("\n\nError occurred: {}".format(e))
        import traceback
        traceback.print_exc()


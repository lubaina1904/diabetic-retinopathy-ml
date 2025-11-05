"""
FEDERATED LEARNING EXPERIMENT - FedAvg

Simulates multiple hospitals training collaboratively WITHOUT sharing data!

Usage:
    python experiments/fedavg.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import flwr as fl
from torch.utils.data import DataLoader
from datetime import datetime
from collections import OrderedDict
import yaml

# Import our modules
from src.dataset import DiabeticRetinopathyDataset, get_transforms
from src.model import create_model
from src.federated import create_hospital_splits, evaluate_global_model, create_client_fn
from src.server import get_model_parameters, set_model_parameters, CustomFedAvg
from src.utils import create_train_val_split, save_results
import matplotlib.pyplot as plt


def load_config(config_path='../configs/config.yaml'):
    """Load configuration"""
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config.get('fedavg', {})
    return {}


def main():
    """
    Main federated learning pipeline
    """
    print("="*70)
    print("FEDERATED LEARNING EXPERIMENT - FedAvg")
    print("="*70)

    # Load config or use defaults
    config_from_file = load_config()
    
    # ==================== CONFIGURATION ====================
    config = {
        # Data
        'csv_file': config_from_file.get('csv_file', 'data/aptos/train.csv'),
        'img_dir': config_from_file.get('img_dir', 'data/aptos/train_images'),
        'val_split': config_from_file.get('val_split', 0.2),

        # Federated Learning
        'num_hospitals': config_from_file.get('num_hospitals', 4),
        'num_rounds': config_from_file.get('num_rounds', 30),
        'local_epochs': config_from_file.get('local_epochs', 3),
        'fraction_fit': config_from_file.get('fraction_fit', 1.0),

        # Model
        'model_name': config_from_file.get('model_name', 'efficientnet_b0'),
        'num_classes': config_from_file.get('num_classes', 5),
        'pretrained': config_from_file.get('pretrained', True),

        # Training
        'batch_size': config_from_file.get('batch_size', 32),
        'learning_rate': config_from_file.get('learning_rate', 0.001),
        'num_workers': config_from_file.get('num_workers', 2),

        # Device
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # Paths
        'save_dir': config_from_file.get('save_dir', 'results/models/fedavg'),
    }

    # Resolve paths relative to project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config['csv_file'] = os.path.join(project_root, config['csv_file'])
    config['img_dir'] = os.path.join(project_root, config['img_dir'])
    config['save_dir'] = os.path.join(project_root, config['save_dir'])

    os.makedirs(config['save_dir'], exist_ok=True)

    print("\nConfiguration:")
    for key, value in config.items():
        print("  {}: {}".format(key, value))

    device = torch.device(config['device'])

    # ==================== PREPARE DATA ====================
    print("\n" + "="*70)
    print("Preparing Federated Data...")
    print("="*70)

    # Load dataset
    train_transform = get_transforms(mode='train')
    val_transform = get_transforms(mode='val')

    full_dataset = DiabeticRetinopathyDataset(
        csv_file=config['csv_file'],
        img_dir=config['img_dir'],
        transform=train_transform
    )

    # Create train/val split
    train_dataset, val_dataset = create_train_val_split(
        full_dataset,
        val_split=config['val_split'],
        random_seed=42
    )

    # Split into hospitals (non-IID)
    print("\nCreating {} hospital datasets...".format(config['num_hospitals']))
    hospital_train_datasets = create_hospital_splits(
        train_dataset,
        num_hospitals=config['num_hospitals'],
        random_seed=42
    )

    # Each hospital gets validation data too
    hospital_val_datasets = create_hospital_splits(
        val_dataset,
        num_hospitals=config['num_hospitals'],
        random_seed=43
    )

    # Create global test loader
    test_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers']
    )

    print("\nData preparation complete!")

    # ==================== DEFINE FL STRATEGY ====================
    print("\n" + "="*70)
    print("Setting up Federated Learning Strategy...")
    print("="*70)

    # Create initial model
    initial_model = create_model(
        model_name=config['model_name'],
        num_classes=config['num_classes'],
        pretrained=config['pretrained']
    )
    initial_parameters = get_model_parameters(initial_model)

    # Define evaluation function
    def evaluate_fn(server_round, parameters):
        """Evaluate global model on test set"""
        model = create_model(
            model_name=config['model_name'],
            num_classes=config['num_classes'],
            pretrained=False
        )
        set_model_parameters(model, parameters)
        loss, accuracy = evaluate_global_model(model, test_loader, device)
        return loss, {"accuracy": accuracy}

    # Custom strategy that tracks results
    strategy = CustomFedAvg(
        fraction_fit=config['fraction_fit'],
        fraction_evaluate=config['fraction_fit'],
        min_fit_clients=config['num_hospitals'],
        min_evaluate_clients=config['num_hospitals'],
        min_available_clients=config['num_hospitals'],
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
        evaluate_fn=evaluate_fn
    )

    # ==================== CREATE CLIENT FUNCTION ====================
    client_fn = create_client_fn(
        hospital_train_datasets,
        hospital_val_datasets,
        config,
        device
    )

    # ==================== RUN FEDERATED LEARNING ====================
    print("\n" + "="*70)
    print("Starting Federated Learning...")
    print("="*70)
    print("{} hospitals".format(config['num_hospitals']))
    print("{} FL rounds".format(config['num_rounds']))
    print("{} local epochs per round".format(config['local_epochs']))
    print("="*70 + "\n")

    start_time = datetime.now()

    # Start simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config['num_hospitals'],
        config=fl.server.ServerConfig(num_rounds=config['num_rounds']),
        strategy=strategy,
        client_resources={
            "num_cpus": 2,
            "num_gpus": 0.25 if device.type == 'cuda' else 0
        }
    )

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    print("\nTotal training time: {:.2f} minutes".format(training_time / 60))

    # ==================== ANALYZE RESULTS ====================
    print("\n" + "="*70)
    print("Analyzing Results...")
    print("="*70)

    # Find best round
    best_round = max(strategy.round_results, key=lambda x: x['accuracy'])
    print("\nBest round: {}".format(best_round['round']))
    print("  Accuracy: {:.2f}%".format(best_round['accuracy']))
    print("  Loss: {:.4f}".format(best_round['loss']))

    # ==================== VISUALIZE ====================
    print("\n" + "="*70)
    print("Creating Visualizations...")
    print("="*70)

    rounds = [r['round'] for r in strategy.round_results]
    accuracies = [r['accuracy'] for r in strategy.round_results]
    losses = [r['loss'] for r in strategy.round_results]

    figures_dir = os.path.join(project_root, 'results/figures')
    os.makedirs(figures_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    ax1.plot(rounds, accuracies, marker='o', linewidth=2)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Global Model Accuracy')
    ax1.grid(True)

    # Loss plot
    ax2.plot(rounds, losses, marker='o', color='red', linewidth=2)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Loss')
    ax2.set_title('Global Model Loss')
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(
        os.path.join(figures_dir, 'fedavg_training_curves.png'),
        dpi=300,
        bbox_inches='tight'
    )
    print("Saved training curves")
    plt.close()

    # ==================== SAVE RESULTS ====================
    print("\n" + "="*70)
    print("Saving Results...")
    print("="*70)

    final_results = {
        'config': config,
        'round_results': strategy.round_results,
        'best_round': best_round,
        'training_time_minutes': training_time / 60,
        'timestamp': datetime.now().isoformat()
    }

    results_dir = os.path.join(project_root, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    save_results(
        final_results,
        os.path.join(results_dir, 'fedavg_results.json')
    )

    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print("Final Accuracy: {:.2f}%".format(best_round['accuracy']))
    print("Training Time: {:.2f} minutes".format(training_time / 60))
    print("Results saved to: {}".format(config['save_dir']))
    print("="*70)

    print("\nFederated Learning experiment complete!")

    return final_results


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print("\n\nError occurred: {}".format(e))
        import traceback
        traceback.print_exc()


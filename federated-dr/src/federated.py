"""
FEDERATED LEARNING MODULE

This module provides high-level functions for running federated learning experiments.
"""

import torch
import flwr as fl
from torch.utils.data import DataLoader
from typing import Callable, Dict, List

from src.model import create_model
from src.client import HospitalClient
from src.server import get_model_parameters, set_model_parameters, CustomFedAvg


def create_hospital_splits(dataset, num_hospitals=4, random_seed=42):
    """
    Split data into "hospitals" with different distributions (non-IID)

    This simulates real-world federated learning where each hospital
    has different patient populations

    Args:
        dataset: Full dataset
        num_hospitals: Number of hospitals to simulate
        random_seed: For reproducibility

    Returns:
        List of datasets, one per hospital
    """
    from torch.utils.data import Subset
    from collections import defaultdict
    import numpy as np

    # Get all labels
    if hasattr(dataset, 'labels_df'):
        all_labels = dataset.labels_df['diagnosis'].values
        all_indices = list(range(len(dataset)))
    else:
        # It's a Subset
        all_labels = [dataset.dataset.labels_df.iloc[i]['diagnosis']
                     for i in dataset.indices]
        all_indices = dataset.indices

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in zip(all_indices, all_labels):
        class_indices[label].append(idx)

    # Define distributions for each hospital
    # These are intentionally different (non-IID)
    distributions = [
        [0.40, 0.20, 0.20, 0.10, 0.10],  # Hospital 1: Mostly healthy
        [0.20, 0.15, 0.15, 0.25, 0.25],  # Hospital 2: More severe cases
        [0.30, 0.25, 0.20, 0.15, 0.10],  # Hospital 3: Balanced
        [0.25, 0.25, 0.25, 0.15, 0.10],  # Hospital 4: Moderate
    ]

    np.random.seed(random_seed)

    hospital_datasets = []

    for h_id, dist in enumerate(distributions[:num_hospitals]):
        hospital_indices = []

        # Calculate samples per class for this hospital
        total_samples_per_hospital = len(all_indices) // num_hospitals

        for class_id, proportion in enumerate(dist):
            num_samples = int(total_samples_per_hospital * proportion)

            available = class_indices[class_id]
            if len(available) >= num_samples:
                sampled = np.random.choice(
                    available,
                    size=num_samples,
                    replace=False
                ).tolist()
                hospital_indices.extend(sampled)

                # Remove sampled indices so they're not reused
                for idx in sampled:
                    class_indices[class_id].remove(idx)

        # Create dataset for this hospital
        if hasattr(dataset, 'labels_df'):
            hospital_dataset = Subset(dataset, hospital_indices)
        else:
            # Map to original dataset indices
            original_indices = [dataset.dataset.labels_df.index[i]
                              for i in hospital_indices]
            hospital_dataset = Subset(dataset.dataset, original_indices)

        hospital_datasets.append(hospital_dataset)

        print("Hospital {}: {} samples".format(h_id + 1, len(hospital_dataset)))

    return hospital_datasets


def evaluate_global_model(model, test_loader, device):
    """
    Evaluate global model on test set
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test data
        device: 'cuda' or 'cpu'
        
    Returns:
        tuple: (Average loss, Accuracy)
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0
    loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    avg_loss = loss / len(test_loader)

    return avg_loss, accuracy


def create_client_fn(
    hospital_train_datasets: List,
    hospital_val_datasets: List,
    config: Dict,
    device: torch.device
) -> Callable:
    """
    Create client factory function for Flower simulation
    
    Args:
        hospital_train_datasets: List of training datasets per hospital
        hospital_val_datasets: List of validation datasets per hospital
        config: Configuration dictionary
        device: PyTorch device
        
    Returns:
        Callable: Client factory function
    """
    def client_fn(cid: str) -> fl.client.Client:
        """Create a client for hospital with given ID"""
        client_id = int(cid)

        # Get this hospital's data
        train_data = hospital_train_datasets[client_id]
        val_data = hospital_val_datasets[client_id]

        # Create dataloaders
        train_loader = DataLoader(
            train_data,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config.get('num_workers', 2)
        )
        val_loader = DataLoader(
            val_data,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config.get('num_workers', 2)
        )

        # Create model for this client
        model = create_model(
            model_name=config['model_name'],
            num_classes=config['num_classes'],
            pretrained=config['pretrained']
        )

        # Create client
        client = HospitalClient(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=config['learning_rate'],
            local_epochs=config['local_epochs']
        )

        return client.to_client()
    
    return client_fn


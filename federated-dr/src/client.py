"""
FEDERATED LEARNING CLIENT - Hospital Side

This represents ONE hospital in the federated learning system.

Key concept: Train locally, share only model updates!
"""

import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np


class HospitalClient(fl.client.NumPyClient):
    """
    Federated Learning Client using FedAvg algorithm

    This is what runs at each hospital:
    1. Receives global model from server
    2. Trains on local data
    3. Sends updated weights back
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        local_epochs: int = 3
    ):
        """
        Initialize client

        Args:
            model: Neural network
            train_loader: This hospital's training data
            val_loader: This hospital's validation data
            device: 'cuda' or 'cpu'
            learning_rate: Learning rate for local training
            local_epochs: How many epochs to train per FL round
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.local_epochs = local_epochs

        # Optimizer and loss
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()

        print("Hospital client initialized")
        print("  Training samples: {}".format(len(train_loader.dataset)))
        print("  Local epochs per round: {}".format(local_epochs))

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        """
        Extract model parameters as numpy arrays

        Called by server to get current model weights

        Returns:
            List of model weights as numpy arrays
        """
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Load parameters into model

        Called by server to send global model to this hospital

        Args:
            parameters: List of model weights as numpy arrays
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Train model on local data

        This is the CORE of federated learning!

        Steps:
        1. Receive global model
        2. Train on local hospital data
        3. Return updated weights

        Args:
            parameters: Global model weights from server
            config: Configuration dictionary

        Returns:
            tuple: (Updated model parameters, Number of samples, Metrics dict)
        """
        # 1. Set global parameters
        self.set_parameters(parameters)

        # 2. Train locally
        print("\nTraining locally for {} epochs...".format(self.local_epochs))
        train_loss, train_acc = self._train_local()

        # 3. Return updated parameters
        print("Local training complete - Loss: {:.4f}, Acc: {:.2f}%".format(
            train_loss, train_acc))

        return (
            self.get_parameters(config={}),
            len(self.train_loader.dataset),
            {"train_loss": train_loss, "train_acc": train_acc}
        )

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """
        Evaluate global model on local data

        Tests how well the global model works for this hospital

        Args:
            parameters: Global model weights
            config: Configuration dictionary

        Returns:
            tuple: (Loss value, Number of samples, Metrics dict)
        """
        # Set global parameters
        self.set_parameters(parameters)

        # Evaluate
        loss, accuracy = self._evaluate_local()

        return (
            loss,
            len(self.val_loader.dataset),
            {"accuracy": accuracy}
        )

    def _train_local(self) -> Tuple[float, float]:
        """
        Local training loop

        This happens INSIDE the hospital - data doesn't leave!

        Returns:
            tuple: (Average loss, Average accuracy)
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for epoch in range(self.local_epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # Statistics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / (len(self.train_loader) * self.local_epochs)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def _evaluate_local(self) -> Tuple[float, float]:
        """
        Local evaluation

        Returns:
            tuple: (Average loss, Average accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy


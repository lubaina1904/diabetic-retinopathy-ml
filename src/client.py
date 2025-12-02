

import flwr as fl
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Tuple
import numpy as np


class HospitalClient(fl.client.NumPyClient):
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        device: str = 'cuda',
        learning_rate: float = 0.001,
        local_epochs: int = 3
    ):
        
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
        
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        
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
       
        self.set_parameters(parameters)

        loss, accuracy = self._evaluate_local()

        return (
            loss,
            len(self.val_loader.dataset),
            {"accuracy": accuracy}
        )

    def _train_local(self) -> Tuple[float, float]:
        
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


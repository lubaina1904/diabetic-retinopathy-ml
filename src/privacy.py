import torch
import torch.nn as nn
from opacus import PrivacyEngine
from typing import Optional


def make_private(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    max_grad_norm: float = 1.0,
    target_epsilon: float = 10.0,
    target_delta: float = 1e-5,
    noise_multiplier: Optional[float] = None,
    epochs: int = 1
):
    privacy_engine = PrivacyEngine()
    
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
    )
    
    return privacy_engine, model, optimizer


def get_privacy_spent(privacy_engine: PrivacyEngine, delta: float = 1e-5):
 
    epsilon = privacy_engine.get_epsilon(delta)
    return epsilon


def create_dp_optimizer(model: nn.Module, learning_rate: float = 0.001):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


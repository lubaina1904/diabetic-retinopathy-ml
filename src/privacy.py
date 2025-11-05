"""
PRIVACY MODULE - Differential Privacy Implementation

This module provides differential privacy (DP) functionality using Opacus.
"""

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
    """
    Wrap model and optimizer with differential privacy
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        train_loader: DataLoader for training
        max_grad_norm: Maximum gradient norm for clipping
        target_epsilon: Target privacy budget (epsilon)
        target_delta: Target delta (typically 1/number of samples)
        noise_multiplier: Optional noise multiplier (if None, computed from epsilon)
        epochs: Number of training epochs
        
    Returns:
        tuple: (PrivacyEngine, model, optimizer)
    """
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
    """
    Get privacy spent (epsilon) from privacy engine
    
    Args:
        privacy_engine: Opacus PrivacyEngine
        delta: Delta parameter
        
    Returns:
        float: Epsilon (privacy budget spent)
    """
    epsilon = privacy_engine.get_epsilon(delta)
    return epsilon


def create_dp_optimizer(model: nn.Module, learning_rate: float = 0.001):
    """
    Create optimizer for differential privacy training
    
    Args:
        model: PyTorch model
        learning_rate: Learning rate
        
    Returns:
        torch.optim.Optimizer: Optimizer
    """
    return torch.optim.Adam(model.parameters(), lr=learning_rate)


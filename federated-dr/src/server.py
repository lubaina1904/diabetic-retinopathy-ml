"""
FEDERATED LEARNING SERVER

This module provides server-side functionality for federated learning.
Custom strategies can be defined here.
"""

import flwr as fl
import torch
from typing import List, Tuple, Optional
from collections import OrderedDict


def get_model_parameters(model):
    """Extract model parameters as numpy arrays"""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_parameters(model, parameters):
    """Set model parameters from numpy arrays"""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class CustomFedAvg(fl.server.strategy.FedAvg):
    """
    Custom FedAvg strategy that tracks results per round
    
    Extends the base FedAvg strategy to evaluate the global model
    after each round and save checkpoints.
    """
    
    def __init__(self, *args, evaluate_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_results = []
        self.evaluate_fn = evaluate_fn

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate and evaluate after each round"""
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None and self.evaluate_fn:
            # Evaluate global model using provided function
            loss, metrics = self.evaluate_fn(
                server_round,
                fl.common.parameters_to_ndarrays(aggregated_parameters)
            )
            
            # Store results
            self.round_results.append({
                'round': server_round,
                'loss': loss,
                **metrics
            })

        return aggregated_parameters, aggregated_metrics


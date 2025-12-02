
import flwr as fl
import torch
from typing import List, Tuple, Optional
from collections import OrderedDict


def get_model_parameters(model):
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_model_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class CustomFedAvg(fl.server.strategy.FedAvg):
   
    def __init__(self, *args, evaluate_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.round_results = []
        self.evaluate_fn = evaluate_fn

    def aggregate_fit(self, server_round, results, failures):
        
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None and self.evaluate_fn:
            
            loss, metrics = self.evaluate_fn(
                server_round,
                fl.common.parameters_to_ndarrays(aggregated_parameters)
            )
            self.round_results.append({
                'round': server_round,
                'loss': loss,
                **metrics
            })

        return aggregated_parameters, aggregated_metrics


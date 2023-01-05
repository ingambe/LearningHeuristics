import math

import torch
import torch.nn as nn
from typing import Iterable

import numpy as np

from torch.nn.utils.convert_parameters import _check_param_device


def _compute_job_priority(
    nb_day_deadline,
    nb_day_left,
    job_weights,
    days_outside,
    coupling_days,
    task_length,
    total_tasks_length,
    network,
):
    with torch.no_grad():
        nb_jobs = len(nb_day_deadline)
        # normalize
        nb_day_deadline = (nb_day_deadline - torch.min(nb_day_deadline)) / max(
            torch.max(nb_day_deadline) - torch.min(nb_day_deadline), 1
        )
        nb_day_left = (nb_day_left - torch.min(nb_day_left)) / max(
            torch.max(nb_day_left) - torch.min(nb_day_left), 1
        )
        days_outside = (days_outside - torch.min(days_outside)) / max(
            torch.max(days_outside) - torch.min(days_outside), 1
        )
        coupling_days = (coupling_days - torch.min(coupling_days)) / max(
            torch.max(coupling_days) - torch.min(coupling_days), 1
        )
        job_weights = (job_weights - torch.min(job_weights)) / max(
            torch.max(job_weights) - torch.min(job_weights), 1
        )
        task_length = (task_length - torch.min(task_length)) / max(
            torch.max(task_length) - torch.min(task_length), 1
        )
        total_tasks_length = (total_tasks_length - torch.min(total_tasks_length)) / max(
            torch.max(total_tasks_length) - torch.min(total_tasks_length), 1
        )
        # view
        nb_day_deadline = nb_day_deadline.view(nb_jobs, 1)
        nb_day_left = nb_day_left.view(nb_jobs, 1)
        job_weights = job_weights.view(nb_jobs, 1)
        coupling_days = coupling_days.view(nb_jobs, 1)
        days_outside = days_outside.view(nb_jobs, 1)
        task_length = task_length.view(nb_jobs, 1)
        total_tasks_length = total_tasks_length.view(nb_jobs, 1)
        input_tensor = torch.hstack(
            (
                nb_day_deadline,
                nb_day_left,
                job_weights,
                days_outside,
                coupling_days,
                task_length,
                total_tasks_length,
            )
        )
        score = network(input_tensor)
        return score.view(-1)


def layer_init(layer, bias_const=0.0):
    torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain("tanh"))
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Network(nn.Module):
    def __init__(self, imput_dim, output_dim):
        super(Network, self).__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=imput_dim, nhead=1, dim_feedforward=imput_dim * 4, batch_first=True, norm_first=True
        )
        self.ln1 = layer_init(nn.Linear(imput_dim, 32))
        self.ln2 = layer_init(nn.Linear(32, 32))
        self.ln3 = layer_init(nn.Linear(32, output_dim))
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x) + x
        x = self.activation(self.ln1(x))
        x = self.activation(self.ln2(x))
        return self.ln3(x)

    def get_weights(self):
        return self.parameters_to_vector()

    def set_weights(self, vector):
        torch.clamp(vector, -10.0, 10.0, out=vector)
        self.vector_to_parameters(vector)

    def parameters_to_vector(self) -> torch.Tensor:
        param_device = None

        vec = []
        for param in self.parameters():
            if param.requires_grad:
                # Ensure the parameters are located in the same device
                param_device = _check_param_device(param, param_device)

                vec.append(param.view(-1))
        return torch.cat(vec)

    def vector_to_parameters(self, vec: torch.Tensor) -> None:
        r"""Convert one vector to the parameters

        Args:
            vec (Tensor): a single vector represents the parameters of a model.
            parameters (Iterable[Tensor]): an iterator of Tensors that are the
                parameters of a model.
        """
        # Ensure vec of type Tensor
        if not isinstance(vec, torch.Tensor):
            raise TypeError(
                "expected torch.Tensor, but got: {}".format(torch.typename(vec))
            )
        # Flag for the device where the parameter is located
        param_device = None

        # Pointer for slicing the vector for each parameter
        pointer = 0
        for param in self.parameters():
            if param.requires_grad:
                # Ensure the parameters are located in the same device
                param_device = _check_param_device(param, param_device)

                # The length of the parameter
                num_param = param.numel()
                # Slice the vector, reshape it, and replace the old data of the parameter
                param.data = vec[pointer : pointer + num_param].view_as(param).data

                # Increment the pointer
                pointer += num_param

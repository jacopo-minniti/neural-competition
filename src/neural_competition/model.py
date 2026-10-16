"""One hidden ReLU layer; each parameter belongs to exactly one neuron."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ShallowReLU(nn.Module):
    def __init__(self, input_dim: int, width: int, seed: int = 0, *, zero_output: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.width = width
        generator = torch.Generator().manual_seed(seed)
        self.weight = nn.Parameter(
            torch.randn(width, input_dim, generator=generator) / math.sqrt(input_dim)
        )
        self.bias = nn.Parameter(torch.zeros(width))
        output = torch.zeros(width) if zero_output else torch.randn(width, generator=generator)
        self.output = nn.Parameter(output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(F.linear(x, self.weight, self.bias)) @ self.output / math.sqrt(self.width)

    @staticmethod
    def blocks(tensors) -> torch.Tensor:
        """Map (incoming weight, bias, outgoing weight) to [neuron, d+2]."""
        weight, bias, output = tensors
        return torch.cat((weight, bias[:, None], output[:, None]), dim=1)

    def parameter_blocks(self) -> torch.Tensor:
        return self.blocks(tuple(self.parameters())).detach()


def make_data(config) -> dict[str, torch.Tensor | dict]:
    generator = torch.Generator().manual_seed(config.data_seed)
    x_train = torch.randn(config.n_train, config.input_dim, generator=generator)
    x_test = torch.randn(config.n_test, config.input_dim, generator=generator)
    teacher = ShallowReLU(
        config.input_dim, config.teacher_width, config.data_seed + 1, zero_output=False
    )
    with torch.no_grad():
        # Fix target variance without changing the teacher's architecture.
        teacher.output.div_(teacher(x_train).std().clamp_min(1e-8))
        y_train, y_test = teacher(x_train), teacher(x_test)
    probe_indices = torch.randint(
        config.n_train, (config.probe_batches, config.batch_size), generator=generator
    )
    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "probe_indices": probe_indices,
        "teacher_state": teacher.state_dict(),
    }


def loss(model: ShallowReLU, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Half-MSE, so the scalar prediction derivative equals the residual."""
    return 0.5 * (model(x) - y).square().mean()

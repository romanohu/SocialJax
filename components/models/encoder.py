"""Reusable encoders for agents and critics."""
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class MLPEncoder(nn.Module):
    hidden_sizes: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        activation = nn.relu if self.activation == "relu" else nn.tanh
        for size in self.hidden_sizes:
            x = nn.Dense(
                features=size,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(x)
            x = activation(x)
        return x


class CNNEncoder(nn.Module):
    channels: Sequence[int] = (32, 32, 32)
    kernel_sizes: Sequence[Sequence[int]] = ((5, 5), (3, 3), (3, 3))
    activation: str = "relu"
    dense_size: int = 64

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        activation = nn.relu if self.activation == "relu" else nn.tanh
        for features, kernel_size in zip(self.channels, self.kernel_sizes):
            x = nn.Conv(
                features=features,
                kernel_size=kernel_size,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(x)
            x = activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=self.dense_size,
            kernel_init=orthogonal(np.sqrt(2.0)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        return x

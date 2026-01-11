"""Reusable decoders for policy and value heads."""
from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal


class MLPDecoder(nn.Module):
    hidden_sizes: Sequence[int]
    output_size: int
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
        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        return x


class ValueDecoder(nn.Module):
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
        x = nn.Dense(
            features=1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)
        return jnp.squeeze(x, axis=-1)

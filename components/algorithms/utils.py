"""Shared algorithm helpers."""
from __future__ import annotations

from typing import Dict, List

import jax.numpy as jnp


def done_dict_to_array(done: Dict, agents: List[int]) -> jnp.ndarray:
    return jnp.stack([done[str(a)] for a in agents], axis=1)

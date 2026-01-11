"""Social Value Orientation (SVO) reward shaping helpers."""
from typing import Optional, Tuple

import jax.numpy as jnp


def mean_others(rewards: jnp.ndarray) -> jnp.ndarray:
    """Compute mean reward of other agents per agent.

    rewards: shape [batch, num_agents]
    """
    sum_rewards = jnp.sum(rewards, axis=1, keepdims=True)
    return (sum_rewards - rewards) / (rewards.shape[1] - 1)


def reward_angle(rewards: jnp.ndarray) -> jnp.ndarray:
    """Compute reward angle theta per agent.

    rewards: shape [batch, num_agents]
    """
    r_i = rewards
    r_others = mean_others(rewards)
    return jnp.arctan2(r_others, r_i)


def svo_linear_combination(
    rewards: jnp.ndarray,
    svo_degrees: jnp.ndarray,
    target_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Paper-style SVO reward: r_i^SVO = cos(theta_i) r_i + sin(theta_i) r_-i.

    rewards: [batch, num_agents]
    svo_degrees: [num_agents]
    target_mask: [num_agents] or None
    """
    theta = jnp.deg2rad(svo_degrees).reshape((1, -1))
    r_i = rewards
    r_others = mean_others(rewards)
    shaped = jnp.cos(theta) * r_i + jnp.sin(theta) * r_others
    if target_mask is None:
        return shaped, theta
    mask = target_mask.reshape((1, -1))
    return jnp.where(mask, shaped, rewards), theta


def svo_deviation_penalty(
    rewards: jnp.ndarray,
    ideal_degrees: jnp.ndarray,
    w: jnp.ndarray,
    target_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Deviation penalty shaping used by existing SVO scripts."""
    theta = reward_angle(rewards)
    ideal = jnp.deg2rad(ideal_degrees).reshape((1, -1))
    w = w.reshape((1, -1))
    angle_deviation = jnp.abs(theta - ideal)
    shaped = rewards - rewards.shape[1] * w * angle_deviation
    if target_mask is None:
        return shaped, theta
    mask = target_mask.reshape((1, -1))
    return jnp.where(mask, shaped, rewards), theta

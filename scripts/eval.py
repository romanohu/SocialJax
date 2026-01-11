"""Evaluate trained policies and optionally render GIFs."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig
from PIL import Image

import socialjax
from components.algorithms.networks import Actor, ActorCritic, EncoderConfig
from components.training.checkpoint import load_checkpoint
from components.training.config import build_config
from components.training.utils import flatten_obs, unflatten_actions


def _build_encoder_cfg(config: Dict) -> EncoderConfig:
    return EncoderConfig(
        activation=config.get("ACTIVATION", "relu"),
        mlp_sizes=tuple(config.get("MLP_HIDDEN_SIZES", (64, 64))),
        cnn_channels=tuple(config.get("CNN_CHANNELS", (32, 32, 32))),
        cnn_kernel_sizes=tuple(config.get("CNN_KERNEL_SIZES", ((5, 5), (3, 3), (3, 3)))),
        cnn_dense_size=int(config.get("CNN_DENSE_SIZE", 64)),
    )


def _select_action(dist, rng, deterministic: bool):
    if deterministic:
        return jnp.argmax(dist.probs, axis=-1)
    return dist.sample(seed=rng)


def _save_gif(frames: List[np.ndarray], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame.astype(np.uint8)) for frame in frames]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(1000 / fps),
        loop=0,
    )


@hydra.main(version_base=None, config_path="config", config_name="eval")
def main(cfg: DictConfig) -> None:
    config = build_config(cfg)
    algorithm = cfg.algorithm.name
    env = socialjax.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))

    ckpt_dir = cfg.checkpoint_dir or config.get("CHECKPOINT_DIR")
    if ckpt_dir is None:
        raise ValueError("checkpoint_dir is required for evaluation")

    payload = load_checkpoint(ckpt_dir, cfg.checkpoint_step)
    encoder_cfg = _build_encoder_cfg(config)
    num_agents = env.num_agents

    if algorithm == "mappo":
        network = Actor(env.action_space().n, encoder_cfg)
        params = payload["actor_params"]
    else:
        network = ActorCritic(env.action_space().n, encoder_cfg)
        params = payload["params"]

    rng = jax.random.PRNGKey(0)

    for episode in range(cfg.num_episodes):
        rng, reset_rng = jax.random.split(rng)
        obs, env_state = env.reset(reset_rng)
        frames = []
        for step in range(cfg.max_steps):
            if cfg.render:
                frame = np.array(env.render(env_state))
                frames.append(frame)

            if isinstance(params, list):
                obs_batch = [obs[i] for i in range(num_agents)]
                actions = []
                for i in range(num_agents):
                    rng, action_rng = jax.random.split(rng)
                    if algorithm == "mappo":
                        dist = network.apply(params[i], obs_batch[i])
                    else:
                        dist, _ = network.apply(params[i], obs_batch[i])
                    actions.append(_select_action(dist, action_rng, cfg.deterministic))
                env_actions = actions
            else:
                obs_batch = flatten_obs(obs[None, ...])
                if algorithm == "mappo":
                    dist = network.apply(params, obs_batch)
                else:
                    dist, _ = network.apply(params, obs_batch)
                rng, action_rng = jax.random.split(rng)
                action = _select_action(dist, action_rng, cfg.deterministic)
                env_actions = unflatten_actions(action, 1, num_agents)

            rng, step_rng = jax.random.split(rng)
            obs, env_state, reward, done, info = env.step(step_rng, env_state, env_actions)
            if done["__all__"]:
                break

        if cfg.render and frames:
            output_dir = Path(cfg.output_dir) / config["ENV_NAME"] / algorithm
            gif_path = output_dir / f"episode_{episode}.gif"
            _save_gif(frames, gif_path, cfg.gif_fps)


if __name__ == "__main__":
    main()

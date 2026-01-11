"""Generic training runner that wraps existing algorithm scripts."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import jax
from omegaconf import OmegaConf

from components.algorithms.registry import AlgorithmEntry


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_module_from_path(module_path: str, module_name: str):
    full_path = _repo_root() / module_path
    spec = importlib.util.spec_from_file_location(module_name, full_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load module from {full_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _set_nested(config: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cursor = config
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def _apply_policy_reward_toggles(
    config: Dict[str, Any],
    independent_policy: Optional[bool],
    independent_reward: Optional[bool],
) -> None:
    if independent_policy is not None:
        config["PARAMETER_SHARING"] = not independent_policy

    if independent_reward is not None:
        _set_nested(config, "ENV_KWARGS.shared_rewards", not independent_reward)
        reward_mode = "individual" if independent_reward else "common"
        if "REWARD" in config:
            if str(config["REWARD"]).startswith("MAPPO"):
                config["REWARD"] = "MAPPO" if independent_reward else "MAPPO_common"
            else:
                config["REWARD"] = reward_mode


def build_config(
    entry: AlgorithmEntry,
    overrides: Optional[Dict[str, Any]] = None,
    independent_policy: Optional[bool] = None,
    independent_reward: Optional[bool] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    base_cfg = OmegaConf.load(_repo_root() / entry.config_path)
    if overrides:
        merged = OmegaConf.merge(base_cfg, overrides)
    else:
        merged = base_cfg
    config = OmegaConf.to_container(merged, resolve=True)

    if seed is not None:
        config["SEED"] = seed

    _apply_policy_reward_toggles(config, independent_policy, independent_reward)
    return config


def run_training(
    entry: AlgorithmEntry,
    config: Dict[str, Any],
    dry_run: bool = False,
):
    if dry_run:
        return config

    module_name = f"socialjax_{entry.algorithm}_{entry.env}"
    module = _load_module_from_path(entry.module_path, module_name)
    make_train = getattr(module, "make_train", None)
    if make_train is None:
        raise AttributeError(f"{entry.module_path} does not define make_train")

    train_fn = make_train(config)
    rng = jax.random.PRNGKey(config["SEED"])
    return train_fn(rng)

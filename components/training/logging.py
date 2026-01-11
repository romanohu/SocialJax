"""Logging utilities for training."""
from __future__ import annotations

from typing import Any, Dict, Optional

import jax.numpy as jnp


def init_wandb(config: Dict[str, Any]) -> Optional[Any]:
    wandb_cfg = config.get("WANDB", {})
    if not wandb_cfg.get("enabled", False):
        return None

    import wandb

    project = wandb_cfg.get("project", "socialjax")
    entity = wandb_cfg.get("entity")
    name = wandb_cfg.get("name")
    group = wandb_cfg.get("group")
    tags = wandb_cfg.get("tags")
    mode = wandb_cfg.get("mode", "online")

    if name is None:
        algo = config.get("ALGORITHM", "algo")
        env_id = config.get("ENV_NAME", "env")
        name = f"{algo}_{env_id}"

    wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        tags=tags,
        mode=mode,
        config=config,
    )
    return wandb


def update_info_stats(stats: Dict[str, Dict[str, float]], info: Dict[str, Any]) -> None:
    for key, value in info.items():
        try:
            mean_val = float(jnp.mean(value))
        except Exception:
            continue
        entry = stats.setdefault(key, {"sum": 0.0, "count": 0.0})
        entry["sum"] += mean_val
        entry["count"] += 1.0


def finalize_info_stats(stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for key, value in stats.items():
        if value["count"] == 0:
            continue
        metrics[f"env/{key}"] = value["sum"] / value["count"]
    return metrics

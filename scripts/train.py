"""Hydra entrypoint for reusable SocialJax training runs."""
from __future__ import annotations

from typing import Any, Dict

import hydra
from omegaconf import DictConfig, OmegaConf

from components.algorithms.registry import get_entry
from components.training.runner import build_config, run_training


def _overrides_from_cfg(cfg: DictConfig) -> Dict[str, Any]:
    if cfg.overrides is None:
        return {}
    return OmegaConf.to_container(cfg.overrides, resolve=True)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig) -> None:
    entry = get_entry(cfg.algorithm.name, cfg.env.name)
    overrides = _overrides_from_cfg(cfg)

    config = build_config(
        entry,
        overrides=overrides,
        independent_policy=cfg.independent_policy,
        independent_reward=cfg.independent_reward,
        seed=cfg.seed,
    )
    run_training(entry, config, dry_run=cfg.dry_run)


if __name__ == "__main__":
    main()

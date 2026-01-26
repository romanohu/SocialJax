"""Checkpoint utilities."""
from __future__ import annotations

import os
from typing import Any, Dict, List

from flax.training import checkpoints

# Normalize to absolute path
def _normalize_ckpt_dir(ckpt_dir: str) -> str:
    if os.path.isabs(ckpt_dir):
        return ckpt_dir
    return os.path.abspath(ckpt_dir)


def save_checkpoint(ckpt_dir: str, step: int, payload: Dict[str, Any], keep: int = 3) -> None:
    ckpt_dir = _normalize_ckpt_dir(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(ckpt_dir, payload, step, keep=keep, overwrite=True)


def agent_checkpoint_dir(ckpt_dir: str, agent_idx: int) -> str:
    ckpt_dir = _normalize_ckpt_dir(ckpt_dir)
    return os.path.join(ckpt_dir, f"agent_{agent_idx}")


def critic_checkpoint_dir(ckpt_dir: str) -> str:
    ckpt_dir = _normalize_ckpt_dir(ckpt_dir)
    return os.path.join(ckpt_dir, "critic")


def save_agent_checkpoints(
    ckpt_dir: str,
    step: int,
    payloads: List[Dict[str, Any]],
    keep: int = 3,
) -> None:
    for agent_idx, payload in enumerate(payloads):
        save_checkpoint(
            agent_checkpoint_dir(ckpt_dir, agent_idx),
            step,
            payload,
            keep=keep,
        )


def load_checkpoint(
    ckpt_dir: str,
    step: int | None = None,
    target: Any | None = None,
) -> Dict[str, Any]:
    ckpt_dir = _normalize_ckpt_dir(ckpt_dir)
    if step is None:
    # Function to load Flax checkpoints and restore parameters
        return checkpoints.restore_checkpoint(ckpt_dir, target=target)
    return checkpoints.restore_checkpoint(ckpt_dir, target=target, step=step)

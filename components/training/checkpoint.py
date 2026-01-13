"""Checkpoint utilities."""
from __future__ import annotations

import os
from typing import Any, Dict

from flax.training import checkpoints


def _normalize_ckpt_dir(ckpt_dir: str) -> str:
    if os.path.isabs(ckpt_dir):
        return ckpt_dir
    return os.path.abspath(ckpt_dir)


def save_checkpoint(ckpt_dir: str, step: int, payload: Dict[str, Any], keep: int = 3) -> None:
    ckpt_dir = _normalize_ckpt_dir(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(ckpt_dir, payload, step, keep=keep, overwrite=True)


def load_checkpoint(
    ckpt_dir: str,
    step: int | None = None,
    target: Any | None = None,
) -> Dict[str, Any]:
    ckpt_dir = _normalize_ckpt_dir(ckpt_dir)
    if step is None:
        return checkpoints.restore_checkpoint(ckpt_dir, target=target)
    return checkpoints.restore_checkpoint(ckpt_dir, target=target, step=step)

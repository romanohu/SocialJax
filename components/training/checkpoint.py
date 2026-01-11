"""Checkpoint utilities."""
from __future__ import annotations

import os
from typing import Any, Dict

from flax.training import checkpoints


def save_checkpoint(ckpt_dir: str, step: int, payload: Dict[str, Any], keep: int = 3) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoints.save_checkpoint(ckpt_dir, payload, step, keep=keep, overwrite=True)


def load_checkpoint(ckpt_dir: str, step: int | None = None) -> Dict[str, Any]:
    if step is None:
        return checkpoints.restore_checkpoint(ckpt_dir, target=None)
    return checkpoints.restore_checkpoint(ckpt_dir, target=None, step=step)

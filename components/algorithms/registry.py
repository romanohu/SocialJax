"""Registry for algorithm modules and config files."""
from dataclasses import dataclass
from typing import Dict

from . import ippo, mappo, svo


@dataclass(frozen=True)
class AlgorithmEntry:
    algorithm: str
    env: str
    module_path: str
    config_path: str


def _build_registry() -> Dict[str, Dict[str, AlgorithmEntry]]:
    registry: Dict[str, Dict[str, AlgorithmEntry]] = {}
    for algorithm, entries in (
        ("ippo", ippo.ENTRIES),
        ("mappo", mappo.ENTRIES),
        ("svo", svo.ENTRIES),
    ):
        registry[algorithm] = {}
        for env, payload in entries.items():
            registry[algorithm][env] = AlgorithmEntry(
                algorithm=algorithm,
                env=env,
                module_path=payload["module_path"],
                config_path=payload["config_path"],
            )
    return registry


REGISTRY = _build_registry()


def get_entry(algorithm: str, env: str) -> AlgorithmEntry:
    if algorithm not in REGISTRY:
        raise KeyError(f"Unknown algorithm '{algorithm}'. Available: {sorted(REGISTRY)}")
    if env not in REGISTRY[algorithm]:
        envs = sorted(REGISTRY[algorithm])
        raise KeyError(f"Unknown env '{env}' for {algorithm}. Available: {envs}")
    return REGISTRY[algorithm][env]

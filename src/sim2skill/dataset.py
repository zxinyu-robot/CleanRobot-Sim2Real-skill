"""Episode-level dataset bookkeeping. Never split adjacent frames at random."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

ENV_VERSION = "0.2.0"

IntArray = NDArray[np.int64]


def episode_splits(
    n_episodes: int,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, list[int]]:
    if n_episodes < 3:
        raise ValueError("need at least 3 episodes to form train/val/test splits")
    rng = np.random.default_rng(seed)
    ids = rng.permutation(n_episodes)
    n_train = max(1, int(n_episodes * train_ratio))
    n_val = max(1, int(n_episodes * val_ratio))
    n_train = min(n_train, n_episodes - 2)
    n_val = min(n_val, n_episodes - n_train - 1)
    return {
        "train": [int(i) for i in ids[:n_train]],
        "val": [int(i) for i in ids[n_train : n_train + n_val]],
        "test": [int(i) for i in ids[n_train + n_val :]],
    }


def transition_mask(episode_ids: IntArray, selected: Iterable[int]) -> NDArray[np.bool_]:
    return np.isin(episode_ids, np.asarray(list(selected), dtype=np.int64))


def classify_pick_place(success: bool, fell_off: bool, final_distance: float) -> str:
    if success:
        return "success"
    if fell_off:
        return "object_fell"
    if final_distance < 0.05:
        return "timeout_near_goal"
    return "timeout"


def classify_wipe(success: bool, coverage: float, contact_loss: float, excess_force: float) -> str:
    if success:
        return "success"
    reasons: list[str] = []
    if coverage < 0.82:
        reasons.append("low_coverage")
    if contact_loss > 0.30:
        reasons.append("contact_loss")
    if excess_force > 0.10:
        reasons.append("excess_force")
    return "+".join(reasons) or "timeout"


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value

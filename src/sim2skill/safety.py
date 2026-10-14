"""A small, explicit action safety layer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SafetyResult:
    action: FloatArray
    intervened: bool
    reasons: tuple[str, ...]


class ActionShield:
    """Limits Cartesian speed, workspace, and suction command range."""

    def __init__(
        self,
        max_delta: float = 0.018,
        workspace_low: FloatArray | None = None,
        workspace_high: FloatArray | None = None,
    ) -> None:
        self.max_delta = max_delta
        self.workspace_low = (
            np.asarray(workspace_low, dtype=np.float64)
            if workspace_low is not None
            else np.array([-0.28, -0.23, 0.138])
        )
        self.workspace_high = (
            np.asarray(workspace_high, dtype=np.float64)
            if workspace_high is not None
            else np.array([0.28, 0.23, 0.398])
        )

    def filter(self, action: FloatArray, eef_position: FloatArray) -> SafetyResult:
        safe = np.asarray(action, dtype=np.float64).copy()
        if safe.shape != (4,):
            raise ValueError(f"expected action shape (4,), got {safe.shape}")
        if not np.all(np.isfinite(safe)):
            raise ValueError("safety shield rejected a non-finite action")

        reasons: list[str] = []
        delta_norm = float(np.linalg.norm(safe[:3]))
        if delta_norm > self.max_delta:
            safe[:3] *= self.max_delta / delta_norm
            reasons.append("cartesian_speed")

        next_position = np.asarray(eef_position) + safe[:3]
        bounded_position = np.clip(next_position, self.workspace_low, self.workspace_high)
        if not np.allclose(next_position, bounded_position):
            safe[:3] = bounded_position - eef_position
            reasons.append("workspace")

        bounded_suction = float(np.clip(safe[3], 0.0, 1.0))
        if bounded_suction != safe[3]:
            safe[3] = bounded_suction
            reasons.append("suction_range")

        return SafetyResult(safe, bool(reasons), tuple(reasons))

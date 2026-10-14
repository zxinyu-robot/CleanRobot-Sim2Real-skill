"""Finite-state expert used to generate reproducible demonstrations."""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from sim2skill.env import PickPlaceEnv

FloatArray = NDArray[np.float64]


class Stage(IntEnum):
    ABOVE_OBJECT = 0
    DESCEND = 1
    GRASP = 2
    LIFT = 3
    ABOVE_GOAL = 4
    LOWER = 5
    RELEASE = 6
    RETREAT = 7


class ScriptedExpert:
    """A transparent task-level baseline, not a learned policy."""

    stage_count = len(Stage)

    def __init__(self, position_tolerance: float = 0.012) -> None:
        self.position_tolerance = position_tolerance
        self.stage = Stage.ABOVE_OBJECT
        self._dwell_steps = 0

    def reset(self) -> None:
        self.stage = Stage.ABOVE_OBJECT
        self._dwell_steps = 0

    def stage_context(self) -> FloatArray:
        context = np.zeros(self.stage_count, dtype=np.float64)
        context[int(self.stage)] = 1.0
        return context

    def action(self, env: PickPlaceEnv) -> FloatArray:
        eef = env.eef_position
        cube = env.object_position
        goal = env.goal_position
        suction = 1.0 if self.stage in {
            Stage.GRASP,
            Stage.LIFT,
            Stage.ABOVE_GOAL,
            Stage.LOWER,
        } else 0.0

        if self.stage == Stage.ABOVE_OBJECT:
            target = np.array([cube[0], cube[1], 0.25])
            self._advance_when_close(eef, target, Stage.DESCEND)
        elif self.stage == Stage.DESCEND:
            target = np.array([cube[0], cube[1], cube[2] + 0.024])
            self._advance_when_close(eef, target, Stage.GRASP)
        elif self.stage == Stage.GRASP:
            target = eef.copy()
            self._dwell_steps += 1
            if self._dwell_steps >= 12:
                self._set_stage(Stage.LIFT)
        elif self.stage == Stage.LIFT:
            target = np.array([eef[0], eef[1], 0.28])
            self._advance_when_close(eef, target, Stage.ABOVE_GOAL)
        elif self.stage == Stage.ABOVE_GOAL:
            target = np.array([goal[0], goal[1], 0.28])
            self._advance_when_close(eef, target, Stage.LOWER)
        elif self.stage == Stage.LOWER:
            target = np.array([goal[0], goal[1], 0.154])
            self._advance_when_close(eef, target, Stage.RELEASE)
        elif self.stage == Stage.RELEASE:
            target = eef.copy()
            self._dwell_steps += 1
            if self._dwell_steps >= 15:
                self._set_stage(Stage.RETREAT)
        else:
            target = np.array([goal[0], goal[1], 0.25])

        delta = target - eef
        norm = float(np.linalg.norm(delta))
        if norm > 0.014:
            delta *= 0.014 / norm
        return np.concatenate((delta, np.array([suction])))

    def _advance_when_close(
        self,
        current: FloatArray,
        target: FloatArray,
        next_stage: Stage,
    ) -> None:
        if np.linalg.norm(target - current) <= self.position_tolerance:
            self._set_stage(next_stage)

    def _set_stage(self, stage: Stage) -> None:
        self.stage = stage
        self._dwell_steps = 0

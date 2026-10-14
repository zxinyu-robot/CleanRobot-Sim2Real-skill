"""Nominal raster wipe, matching a detected-fixture then fixed-path controller."""

from __future__ import annotations

from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from sim2skill.wipe_env import SurfaceBelief, WipeEnv

FloatArray = NDArray[np.float64]


class WipeStage(IntEnum):
    APPROACH = 0
    DESCEND = 1
    WIPE = 2
    RETRACT = 3


class WipeExpert:
    """A transparent zigzag cleaner. Set use_true_pose=True for an oracle baseline."""

    stage_count = len(WipeStage)

    def __init__(self, use_true_pose: bool = False, speed: float = 0.025) -> None:
        self.use_true_pose = use_true_pose
        self.speed = speed
        self.stage = WipeStage.APPROACH
        self._waypoints: list[tuple[WipeStage, FloatArray]] = []
        self._index = 0

    def reset(self, env: WipeEnv) -> None:
        surface = env.true_surface if self.use_true_pose else env.detected_surface
        self._waypoints = self._raster(surface, env.pad_radius)
        self._index = 0
        self.stage = self._waypoints[0][0]

    @property
    def done(self) -> bool:
        return self._index >= len(self._waypoints)

    def stage_context(self) -> FloatArray:
        context = np.zeros(self.stage_count, dtype=np.float64)
        context[int(self.stage)] = 1.0
        return context

    def action(self, env: WipeEnv) -> FloatArray:
        if self._index >= len(self._waypoints):
            self.stage = WipeStage.RETRACT
            return np.array([0.0, 0.0, 0.0, 0.0])

        self.stage, target = self._waypoints[self._index]
        delta = target - env.eef_position
        xy_distance = float(np.linalg.norm(delta[:2]))
        distance = float(np.linalg.norm(delta))
        arrived = distance <= 0.015 or (
            self.stage == WipeStage.WIPE and xy_distance <= 0.016 and abs(delta[2]) <= 0.01
        )
        if arrived:
            self._index += 1
        if distance > self.speed:
            delta *= self.speed / distance
        press = 1.0 if self.stage in {WipeStage.DESCEND, WipeStage.WIPE} else 0.0
        return np.concatenate((delta, np.array([press])))

    def _raster(
        self,
        surface: SurfaceBelief,
        pad_radius: float,
    ) -> list[tuple[WipeStage, FloatArray]]:
        inset_x = min(pad_radius * 0.65, surface.half_xy[0] * 0.4)
        inset_y = min(pad_radius * 0.65, surface.half_xy[1] * 0.4)
        x0 = surface.center[0] - surface.half_xy[0] + inset_x
        x1 = surface.center[0] + surface.half_xy[0] - inset_x
        y0 = surface.center[1] - surface.half_xy[1] + inset_y
        y1 = surface.center[1] + surface.half_xy[1] - inset_y
        spacing = pad_radius * 1.25
        contact_z = surface.top_z - 0.001
        hover_z = surface.top_z + 0.08
        ys = np.arange(y0, y1 + 1e-9, spacing)
        if len(ys) == 0:
            ys = np.array([surface.center[1]])

        waypoints: list[tuple[WipeStage, FloatArray]] = [
            (WipeStage.APPROACH, np.array([x0, y0, hover_z])),
            (WipeStage.DESCEND, np.array([x0, y0, contact_z])),
        ]
        for index, y in enumerate(ys):
            if index % 2 == 0:
                start_x, end_x = x0, x1
            else:
                start_x, end_x = x1, x0
            if index > 0:
                waypoints.append((WipeStage.WIPE, np.array([start_x, y, contact_z])))
            waypoints.append((WipeStage.WIPE, np.array([end_x, y, contact_z])))
        last = waypoints[-1][1]
        waypoints.append((WipeStage.RETRACT, np.array([last[0], last[1], hover_z])))
        return waypoints

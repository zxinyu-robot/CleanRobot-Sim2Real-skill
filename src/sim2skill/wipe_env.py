"""Planar wiping environment that mirrors a fixed cleaning trajectory."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from importlib.resources import files

import mujoco
import numpy as np
from numpy.typing import NDArray

from sim2skill.env import StepResult

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SurfaceBelief:
    center: FloatArray
    half_xy: FloatArray
    top_z: float


class DirtGrid:
    """Software dirt map used for coverage metrics and a local stain observation."""

    def __init__(self, center_xy: FloatArray, half_xy: FloatArray, cell: float = 0.018) -> None:
        self.cell = cell
        half = np.asarray(half_xy, dtype=np.float64)
        self.origin = np.asarray(center_xy, dtype=np.float64) - half
        self.nx = max(3, int(np.round(2.0 * half_xy[0] / cell)))
        self.ny = max(3, int(np.round(2.0 * half_xy[1] / cell)))
        self.dirt = np.ones((self.ny, self.nx), dtype=np.float64)

    def cell_center(self, ix: int, iy: int) -> FloatArray:
        return self.origin + np.array([(ix + 0.5) * self.cell, (iy + 0.5) * self.cell])

    def clean_at(self, xy: FloatArray, radius: float) -> None:
        for iy in range(self.ny):
            for ix in range(self.nx):
                center = self.cell_center(ix, iy)
                if np.linalg.norm(xy - center) <= radius:
                    self.dirt[iy, ix] = 0.0

    def local_patch(self, xy: FloatArray, size: int = 3) -> FloatArray:
        ix = int(np.floor((xy[0] - self.origin[0]) / self.cell))
        iy = int(np.floor((xy[1] - self.origin[1]) / self.cell))
        patch = np.zeros((size, size), dtype=np.float64)
        half = size // 2
        for dy in range(size):
            for dx in range(size):
                x = ix + dx - half
                y = iy + dy - half
                if 0 <= x < self.nx and 0 <= y < self.ny:
                    patch[dy, dx] = self.dirt[y, x]
        return patch

    def nearest_dirt_vector(self, xy: FloatArray) -> FloatArray:
        dirty = np.argwhere(self.dirt > 0.5)
        if len(dirty) == 0:
            return np.zeros(2, dtype=np.float64)
        best = np.zeros(2, dtype=np.float64)
        best_distance = np.inf
        for iy, ix in dirty:
            center = self.cell_center(int(ix), int(iy))
            delta = center - xy
            distance = float(np.linalg.norm(delta))
            if distance < best_distance:
                best_distance = distance
                best = delta
        return np.clip(best, -0.12, 0.12)

    @property
    def coverage(self) -> float:
        return float(1.0 - self.dirt.mean())


class WipeEnv:
    """Cartesian pad wiping a randomized rectangular surface.

    The robot only sees a noisy surface estimate plus a local stain patch,
    analogous to a detected fixture pose plus a close-range dirt cue.
    """

    observation_size = 21
    action_size = 4
    patch_size = 3
    pad_radius = 0.02

    def __init__(
        self,
        seed: int = 0,
        max_steps: int = 260,
        frame_skip: int = 20,
        pose_noise: float = 0.0,
        size_noise: float = 0.0,
        delay_steps: int = 0,
        height_noise: float = 0.0,
    ) -> None:
        model_path = files("sim2skill.assets").joinpath("wipe.xml")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.pose_noise = pose_noise
        self.size_noise = size_noise
        self.delay_steps = delay_steps
        self.height_noise = height_noise
        self.steps = 0

        self._eef_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "eef_site")
        self._pad_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "pad")
        self._surface_geom_id = self._id(mujoco.mjtObj.mjOBJ_GEOM, "surface_geom")
        self._surface_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "surface")
        self._surface_mocap_id = int(self.model.body_mocapid[self._surface_body_id])
        self._robot_joint_ids = np.array(
            [
                self._id(mujoco.mjtObj.mjOBJ_JOINT, "robot_x"),
                self._id(mujoco.mjtObj.mjOBJ_JOINT, "robot_y"),
                self._id(mujoco.mjtObj.mjOBJ_JOINT, "robot_z"),
            ]
        )
        self._robot_qpos_adrs = self.model.jnt_qposadr[self._robot_joint_ids]
        self._robot_ranges = self.model.jnt_range[self._robot_joint_ids]
        self._nominal_friction = self.model.geom_friction[self._surface_geom_id].copy()

        self.true_surface = SurfaceBelief(np.zeros(3), np.zeros(2), 0.0)
        self.detected_surface = self.true_surface
        self.dirt = DirtGrid(np.zeros(2), np.array([0.08, 0.06]))
        self._action_queue: deque[FloatArray] = deque()
        self._press_steps = 0
        self._missed_contact_steps = 0
        self._excess_force_steps = 0
        self._last_delta = np.zeros(3)
        self._smoothness_sum = 0.0
        self.reset()

    def _id(self, object_type: mujoco.mjtObj, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object not found: {name}")
        return object_id

    @property
    def eef_position(self) -> FloatArray:
        return self.data.site_xpos[self._eef_site_id].copy()

    @property
    def surface_top(self) -> float:
        half_z = float(self.model.geom_size[self._surface_geom_id, 2])
        return float(self.data.mocap_pos[self._surface_mocap_id, 2] + half_z)

    @property
    def penetration(self) -> float:
        return float(self.surface_top - self.eef_position[2])

    @property
    def in_contact(self) -> bool:
        over = self._over_surface(self.eef_position[:2], self.true_surface)
        return over and self.penetration > -0.002

    def _over_surface(self, xy: FloatArray, surface: SurfaceBelief, margin: float = 0.0) -> bool:
        delta = np.abs(xy - surface.center[:2])
        inside_x = delta[0] <= surface.half_xy[0] + margin
        inside_y = delta[1] <= surface.half_xy[1] + margin
        return bool(inside_x and inside_y)

    def task_params(self) -> dict[str, float]:
        return {
            "pose_noise": self.pose_noise,
            "size_noise": self.size_noise,
            "delay_steps": float(self.delay_steps),
            "height_noise": self.height_noise,
            "true_cx": float(self.true_surface.center[0]),
            "true_cy": float(self.true_surface.center[1]),
            "true_hx": float(self.true_surface.half_xy[0]),
            "true_hy": float(self.true_surface.half_xy[1]),
            "detected_cx": float(self.detected_surface.center[0]),
            "detected_cy": float(self.detected_surface.center[1]),
        }

    def reset(self, seed: int | None = None) -> FloatArray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        self._press_steps = 0
        self._missed_contact_steps = 0
        self._excess_force_steps = 0
        self._last_delta = np.zeros(3)
        self._smoothness_sum = 0.0
        self._action_queue.clear()

        half_xy = np.array(
            [self.rng.uniform(0.07, 0.10), self.rng.uniform(0.05, 0.08)],
            dtype=np.float64,
        )
        center = np.array(
            [self.rng.uniform(-0.05, 0.05), self.rng.uniform(-0.08, 0.08), 0.104],
            dtype=np.float64,
        )
        self.model.geom_size[self._surface_geom_id, 0] = half_xy[0]
        self.model.geom_size[self._surface_geom_id, 1] = half_xy[1]
        friction = float(np.clip(self._nominal_friction[0] + self.rng.normal(0.0, 0.15), 0.5, 1.8))
        self.model.geom_friction[self._surface_geom_id, 0] = friction
        self.model.geom_friction[self._pad_geom_id, 0] = friction
        self.data.mocap_pos[self._surface_mocap_id] = center
        self.data.mocap_quat[self._surface_mocap_id] = [1.0, 0.0, 0.0, 0.0]

        home = np.array([0.0, 0.0, 0.24])
        self.data.qpos[self._robot_qpos_adrs] = home
        self.data.ctrl[:] = home
        mujoco.mj_forward(self.model, self.data)

        top_z = self.surface_top + float(self.rng.normal(0.0, self.height_noise))
        self.true_surface = SurfaceBelief(center.copy(), half_xy.copy(), float(self.surface_top))
        detected_xy = center[:2] + self.rng.normal(0.0, self.pose_noise, size=2)
        size_scale = 1.0 + float(self.rng.normal(0.0, self.size_noise))
        detected_half = np.clip(half_xy * size_scale, 0.04, 0.12)
        self.detected_surface = SurfaceBelief(
            np.array([detected_xy[0], detected_xy[1], center[2]], dtype=np.float64),
            detected_half,
            top_z,
        )
        self.dirt = DirtGrid(center[:2], half_xy)
        return self.observation()

    def observation(self) -> FloatArray:
        eef = self.eef_position
        patch = self.dirt.local_patch(eef[:2], size=self.patch_size).reshape(-1)
        dirt_vector = self.dirt.nearest_dirt_vector(eef[:2])
        return np.concatenate(
            (
                eef,
                self.detected_surface.center,
                self.detected_surface.half_xy,
                np.array([self.penetration, float(self.in_contact)], dtype=np.float64),
                patch,
                dirt_vector,
            )
        )

    def step(self, action: FloatArray) -> StepResult:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_size,):
            raise ValueError(f"expected action shape {(self.action_size,)}, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action must contain only finite values")

        self._action_queue.append(action.copy())
        if len(self._action_queue) > self.delay_steps:
            executed = self._action_queue.popleft()
        else:
            executed = np.zeros(4)
        robot_qpos = self.data.qpos[self._robot_qpos_adrs]
        target = robot_qpos + executed[:3]
        self.data.ctrl[:] = np.clip(target, self._robot_ranges[:, 0], self._robot_ranges[:, 1])
        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.steps += 1

        press = float(np.clip(executed[3], 0.0, 1.0))
        if press > 0.5:
            self._press_steps += 1
            if not self.in_contact:
                self._missed_contact_steps += 1
        if self.penetration > 0.007:
            self._excess_force_steps += 1
        self._smoothness_sum += float(np.linalg.norm(executed[:3] - self._last_delta))
        self._last_delta = executed[:3].copy()

        if self.in_contact:
            self.dirt.clean_at(self.eef_position[:2], radius=self.pad_radius * 0.95)

        coverage = self.dirt.coverage
        contact_loss = (
            self._missed_contact_steps / self._press_steps if self._press_steps else 0.0
        )
        excess_rate = self._excess_force_steps / max(self.steps, 1)
        success = coverage >= 0.82 and contact_loss <= 0.30 and excess_rate <= 0.10
        truncated = self.steps >= self.max_steps
        reward = coverage - 0.4 * contact_loss - 0.5 * excess_rate - 0.002
        info: dict[str, float | bool] = {
            "success": success,
            "coverage": coverage,
            "contact_loss_rate": contact_loss,
            "excess_force_rate": excess_rate,
            "in_contact": self.in_contact,
            "penetration": self.penetration,
            "smoothness": self._smoothness_sum / max(self.steps, 1),
        }
        return StepResult(self.observation(), reward, False, truncated, info)

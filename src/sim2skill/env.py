"""Small MuJoCo pick-and-place environment designed for CPU-only laptops."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files

import mujoco
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StepResult:
    observation: FloatArray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, float | bool]


class PickPlaceEnv:
    """A minimal Cartesian suction robot with a randomized pick-and-place task.

    The robot is intentionally simple so the complete data-to-evaluation loop
    runs on an older Intel Mac. The interfaces mirror larger robot-learning
    environments, allowing the model and task to be replaced later.
    """

    observation_size = 13
    action_size = 4

    def __init__(self, seed: int = 0, max_steps: int = 800, frame_skip: int = 10) -> None:
        model_path = files("sim2skill.assets").joinpath("pick_place.xml")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.frame_skip = frame_skip
        self.steps = 0

        self._eef_site_id = self._id(mujoco.mjtObj.mjOBJ_SITE, "eef_site")
        self._cube_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "cube")
        self._cube_joint_id = self._id(mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self._goal_body_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "goal")
        self._goal_mocap_id = int(self.model.body_mocapid[self._goal_body_id])
        self._cube_qpos_adr = int(self.model.jnt_qposadr[self._cube_joint_id])
        self._cube_dof_adr = int(self.model.jnt_dofadr[self._cube_joint_id])
        self._robot_joint_ids = np.array(
            [
                self._id(mujoco.mjtObj.mjOBJ_JOINT, "robot_x"),
                self._id(mujoco.mjtObj.mjOBJ_JOINT, "robot_y"),
                self._id(mujoco.mjtObj.mjOBJ_JOINT, "robot_z"),
            ]
        )
        self._robot_qpos_adrs = self.model.jnt_qposadr[self._robot_joint_ids]
        self._robot_ranges = self.model.jnt_range[self._robot_joint_ids]
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
    def object_position(self) -> FloatArray:
        return self.data.xpos[self._cube_body_id].copy()

    @property
    def goal_position(self) -> FloatArray:
        return self.data.mocap_pos[self._goal_mocap_id].copy()

    def reset(self, seed: int | None = None) -> FloatArray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0

        robot_home = np.array([0.0, 0.0, 0.18])
        self.data.qpos[self._robot_qpos_adrs] = robot_home
        self.data.ctrl[:3] = robot_home
        self.data.ctrl[3] = 0.0

        cube_xy = np.array(
            [self.rng.uniform(-0.19, -0.08), self.rng.uniform(-0.13, 0.13)]
        )
        cube_qpos = self._cube_qpos_adr
        self.data.qpos[cube_qpos : cube_qpos + 3] = [cube_xy[0], cube_xy[1], 0.13]
        self.data.qpos[cube_qpos + 3 : cube_qpos + 7] = [1.0, 0.0, 0.0, 0.0]

        goal_xy = np.array([self.rng.uniform(0.09, 0.20), self.rng.uniform(-0.13, 0.13)])
        self.data.mocap_pos[self._goal_mocap_id] = [goal_xy[0], goal_xy[1], 0.103]
        self.data.mocap_quat[self._goal_mocap_id] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(self.model, self.data)
        return self.observation()

    def observation(self) -> FloatArray:
        cube_velocity = self.data.qvel[self._cube_dof_adr : self._cube_dof_adr + 3]
        return np.concatenate(
            (
                self.eef_position,
                self.object_position,
                self.goal_position,
                cube_velocity.copy(),
                np.array([self.data.ctrl[3]], dtype=np.float64),
            )
        )

    def step(self, action: FloatArray) -> StepResult:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.action_size,):
            raise ValueError(f"expected action shape {(self.action_size,)}, got {action.shape}")
        if not np.all(np.isfinite(action)):
            raise ValueError("action must contain only finite values")

        robot_qpos = self.data.qpos[self._robot_qpos_adrs]
        target = robot_qpos + action[:3]
        self.data.ctrl[:3] = np.clip(
            target,
            self._robot_ranges[:, 0],
            self._robot_ranges[:, 1],
        )
        self.data.ctrl[3] = float(np.clip(action[3], 0.0, 1.0))

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)
        self.steps += 1

        distance = float(np.linalg.norm(self.object_position[:2] - self.goal_position[:2]))
        height = float(self.object_position[2])
        success = distance < 0.05 and height < 0.145 and self.data.ctrl[3] < 0.5
        fell_off = height < 0.07
        terminated = success or fell_off
        truncated = self.steps >= self.max_steps
        reward = (10.0 if success else 0.0) - 0.01 - 0.1 * distance
        info: dict[str, float | bool] = {
            "success": success,
            "fell_off": fell_off,
            "object_goal_distance": distance,
            "object_height": height,
        }
        return StepResult(self.observation(), reward, terminated, truncated, info)

    def render(self, width: int = 640, height: int = 480) -> NDArray[np.uint8]:
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        try:
            renderer.update_scene(self.data, camera="overview")
            return renderer.render().copy()
        finally:
            renderer.close()

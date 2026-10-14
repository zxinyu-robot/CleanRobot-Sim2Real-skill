"""Bounded residual on top of a nominal wipe trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sim2skill.mlp import MLP
from sim2skill.wipe_env import WipeEnv
from sim2skill.wipe_expert import WipeExpert

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ResidualConfig:
    bound: float = 0.008
    hidden: int = 32


class StructuredResidual:
    """A transparent correction that uses contact and the local stain cue.

    This is the engineering baseline the MLP is supposed to approach: keep the
    raster, but press when contact is lost and drift toward remaining dirt.
    """

    def __init__(self, bound: float = 0.012) -> None:
        self.bound = bound

    def action(self, observation: FloatArray, nominal: FloatArray) -> FloatArray:
        residual = np.zeros(4, dtype=np.float64)
        penetration = float(observation[8])
        dirt = np.asarray(observation[-2:], dtype=np.float64)
        if nominal[3] > 0.5:
            if penetration < -0.001:
                residual[2] = -self.bound
            elif penetration > 0.005:
                residual[2] = self.bound
            norm = float(np.linalg.norm(dirt))
            if norm > 0.012:
                residual[:2] = self.bound * dirt / norm
        residual = np.clip(residual, -self.bound, self.bound)
        output = np.asarray(nominal, dtype=np.float64) + residual
        output[3] = float(np.clip(output[3], 0.0, 1.0))
        return output


class ResidualPolicy:
    """action = clip(nominal + MLP(obs, nominal, stage), -bound, bound bounds on residual)."""

    def __init__(self, mlp: MLP, bound: float = 0.008) -> None:
        self.mlp = mlp
        self.bound = bound

    def residual(self, features: FloatArray) -> FloatArray:
        raw = np.asarray(self.mlp.forward(features), dtype=np.float64)
        residual = np.clip(raw, -self.bound, self.bound)
        residual[3] = np.clip(residual[3], -1.0, 1.0)
        return residual

    def action(self, features: FloatArray, nominal: FloatArray) -> FloatArray:
        output = np.asarray(nominal, dtype=np.float64).copy()
        output = output + self.residual(features)
        output[3] = float(np.clip(output[3], 0.0, 1.0))
        return output

    def save(self, path: Path) -> None:
        self.mlp.save(path)

    @classmethod
    def load(cls, path: Path, bound: float = 0.008) -> ResidualPolicy:
        return cls(MLP.load(path), bound=bound)


def residual_features(
    observation: FloatArray,
    nominal: FloatArray,
    stage_context: FloatArray,
) -> FloatArray:
    return np.concatenate((observation, nominal, stage_context))


def collect_residual_data(
    episodes: int,
    seed: int,
    pose_noise: float = 0.025,
    size_noise: float = 0.08,
    delay_steps: int = 1,
) -> tuple[FloatArray, FloatArray]:
    """Label nominal states with a bounded oracle correction."""

    env = WipeEnv(
        seed=seed,
        pose_noise=pose_noise,
        size_noise=size_noise,
        delay_steps=delay_steps,
        height_noise=0.001,
    )
    features: list[FloatArray] = []
    targets: list[FloatArray] = []
    bound = ResidualConfig().bound
    for index in range(episodes):
        observation = env.reset(seed=seed + index)
        nominal_expert = WipeExpert(use_true_pose=False)
        oracle_expert = WipeExpert(use_true_pose=True)
        nominal_expert.reset(env)
        oracle_expert.reset(env)
        while True:
            nominal = nominal_expert.action(env)
            oracle = oracle_expert.action(env)
            target = np.clip(oracle - nominal, -bound, bound)
            target[3] = float(np.clip(oracle[3] - nominal[3], -1.0, 1.0))
            features.append(residual_features(observation, nominal, nominal_expert.stage_context()))
            targets.append(target)
            executed = nominal + target
            executed[3] = float(np.clip(executed[3], 0.0, 1.0))
            result = env.step(executed)
            observation = result.observation
            if result.truncated or oracle_expert.done:
                break
    return np.asarray(features), np.asarray(targets)


def train_residual_policy(
    episodes: int,
    seed: int,
    output: Path | None = None,
) -> tuple[ResidualPolicy, list[float]]:
    features, targets = collect_residual_data(episodes, seed)
    mlp = MLP(features.shape[1], ResidualConfig().hidden, 4, seed=seed)
    losses = mlp.fit(features, targets, seed=seed)
    policy = ResidualPolicy(mlp, bound=ResidualConfig().bound)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        policy.save(output)
    return policy, losses

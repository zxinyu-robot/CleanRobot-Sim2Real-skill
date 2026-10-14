"""Wipe-task rollout, evaluation, and policy comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sim2skill.dataset import classify_wipe
from sim2skill.residual import ResidualPolicy, StructuredResidual, residual_features
from sim2skill.safety import ActionShield
from sim2skill.wipe_env import WipeEnv
from sim2skill.wipe_expert import WipeExpert

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class WipeSummary:
    success: bool
    steps: int
    coverage: float
    contact_loss_rate: float
    excess_force_rate: float
    smoothness: float
    safety_interventions: int
    failure: str


@dataclass(frozen=True)
class WipeTrace:
    observations: FloatArray
    actions: FloatArray
    stages: NDArray[np.int64]
    summary: WipeSummary


def _shield() -> ActionShield:
    return ActionShield(
        max_delta=0.028,
        workspace_low=np.array([-0.26, -0.20, 0.095]),
        workspace_high=np.array([0.26, 0.20, 0.31]),
    )


def run_wipe_episode(
    env: WipeEnv,
    seed: int,
    policy_name: str = "nominal",
    residual: ResidualPolicy | StructuredResidual | None = None,
) -> WipeTrace:
    observation = env.reset(seed=seed)
    expert = WipeExpert(use_true_pose=policy_name == "oracle")
    expert.reset(env)
    shield = _shield()
    observations: list[FloatArray] = []
    actions: list[FloatArray] = []
    stages: list[int] = []
    interventions = 0

    while True:
        nominal = expert.action(env)
        if policy_name == "residual":
            if residual is None:
                raise ValueError("residual policy is required")
            raw = residual.action(
                residual_features(observation, nominal, expert.stage_context()),
                nominal,
            )
        elif policy_name == "structured":
            raw = StructuredResidual().action(observation, nominal)
        else:
            raw = nominal
        safety = shield.filter(raw, env.eef_position)
        observations.append(observation)
        actions.append(safety.action)
        stages.append(int(expert.stage))
        interventions += int(safety.intervened)
        result = env.step(safety.action)
        observation = result.observation
        if result.truncated or expert.done:
            break

    coverage = float(result.info["coverage"])
    contact_loss = float(result.info["contact_loss_rate"])
    excess = float(result.info["excess_force_rate"])
    success = bool(result.info["success"])
    summary = WipeSummary(
        success=success,
        steps=env.steps,
        coverage=coverage,
        contact_loss_rate=contact_loss,
        excess_force_rate=excess,
        smoothness=float(result.info["smoothness"]),
        safety_interventions=interventions,
        failure=classify_wipe(success, coverage, contact_loss, excess),
    )
    return WipeTrace(
        observations=np.asarray(observations),
        actions=np.asarray(actions),
        stages=np.asarray(stages, dtype=np.int64),
        summary=summary,
    )


def evaluate_wipe(
    episodes: int,
    seed: int,
    policy_name: str,
    pose_noise: float,
    residual: ResidualPolicy | StructuredResidual | None = None,
    size_noise: float | None = None,
    delay_steps: int | None = None,
) -> dict[str, object]:
    size_noise = pose_noise * 3.0 if size_noise is None else size_noise
    delay_steps = 0 if pose_noise <= 1e-9 else (1 if delay_steps is None else delay_steps)
    env = WipeEnv(
        seed=seed,
        pose_noise=pose_noise,
        size_noise=size_noise,
        delay_steps=delay_steps,
        height_noise=0.0 if pose_noise <= 1e-9 else 0.001,
    )
    summaries = [
        run_wipe_episode(env, seed + index, policy_name, residual).summary
        for index in range(episodes)
    ]
    return {
        "policy": policy_name,
        "pose_noise": pose_noise,
        "episodes": episodes,
        "success_rate": sum(item.success for item in summaries) / episodes,
        "mean_coverage": float(np.mean([item.coverage for item in summaries])),
        "mean_contact_loss": float(np.mean([item.contact_loss_rate for item in summaries])),
        "mean_excess_force": float(np.mean([item.excess_force_rate for item in summaries])),
        "mean_steps": float(np.mean([item.steps for item in summaries])),
        "safety_interventions": sum(item.safety_interventions for item in summaries),
        "failures": _count_failures(summaries),
    }


def _count_failures(summaries: list[WipeSummary]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in summaries:
        counts[item.failure] = counts.get(item.failure, 0) + 1
    return counts


def compare_wipe(
    episodes: int,
    seed: int,
    residual_path: Path | None = None,
) -> list[dict[str, object]]:
    residual = None
    if residual_path is not None and residual_path.exists():
        residual = ResidualPolicy.load(residual_path)
    rows: list[dict[str, object]] = []
    for noise in (0.0, 0.02, 0.04):
        rows.append(evaluate_wipe(episodes, seed, "nominal", noise))
        rows.append(evaluate_wipe(episodes, seed, "structured", noise))
        rows.append(evaluate_wipe(episodes, seed, "oracle", noise))
        if residual is not None:
            rows.append(
                evaluate_wipe(episodes, seed, "residual", noise, residual=residual)
            )
    return rows

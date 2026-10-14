"""Episode rollout, evaluation, and demonstration collection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sim2skill.dataset import classify_pick_place
from sim2skill.env import PickPlaceEnv
from sim2skill.expert import ScriptedExpert
from sim2skill.safety import ActionShield

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EpisodeSummary:
    success: bool
    steps: int
    total_reward: float
    safety_interventions: int
    final_distance: float
    failure: str


@dataclass(frozen=True)
class EpisodeTrace:
    observations: FloatArray
    actions: FloatArray
    stages: NDArray[np.int64]
    summary: EpisodeSummary


def run_expert_episode(env: PickPlaceEnv, seed: int) -> EpisodeTrace:
    expert = ScriptedExpert()
    shield = ActionShield()
    observation = env.reset(seed=seed)
    observations: list[FloatArray] = []
    actions: list[FloatArray] = []
    stages: list[int] = []
    interventions = 0
    total_reward = 0.0
    final_distance = float("inf")

    while True:
        raw_action = expert.action(env)
        safety_result = shield.filter(raw_action, env.eef_position)
        observations.append(np.concatenate((observation, expert.stage_context())))
        actions.append(safety_result.action)
        stages.append(int(expert.stage))
        interventions += int(safety_result.intervened)

        result = env.step(safety_result.action)
        observation = result.observation
        total_reward += result.reward
        final_distance = float(result.info["object_goal_distance"])
        if result.terminated or result.truncated:
            break

    success = bool(result.info["success"])
    summary = EpisodeSummary(
        success=success,
        steps=env.steps,
        total_reward=total_reward,
        safety_interventions=interventions,
        final_distance=final_distance,
        failure=classify_pick_place(
            success,
            bool(result.info["fell_off"]),
            final_distance,
        ),
    )
    return EpisodeTrace(
        observations=np.asarray(observations),
        actions=np.asarray(actions),
        stages=np.asarray(stages),
        summary=summary,
    )


def collect_demonstrations(
    episodes: int,
    seed: int,
) -> tuple[dict[str, FloatArray], list[EpisodeSummary]]:
    env = PickPlaceEnv(seed=seed)
    traces = [run_expert_episode(env, seed + index) for index in range(episodes)]
    episode_ids = np.concatenate(
        [np.full(len(trace.actions), index, dtype=np.int64) for index, trace in enumerate(traces)]
    )
    dataset = {
        "observations": np.concatenate([trace.observations for trace in traces]),
        "actions": np.concatenate([trace.actions for trace in traces]),
        "stages": np.concatenate([trace.stages for trace in traces]),
        "episode_ids": episode_ids,
    }
    return dataset, [trace.summary for trace in traces]

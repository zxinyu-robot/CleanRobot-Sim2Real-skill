from sim2skill.env import PickPlaceEnv
from sim2skill.rollout import run_expert_episode


def test_expert_completes_pick_place() -> None:
    trace = run_expert_episode(PickPlaceEnv(seed=0), seed=0)

    assert trace.summary.success
    assert trace.summary.steps < 800
    assert len(trace.observations) == len(trace.actions)

import numpy as np

from sim2skill.wipe_env import WipeEnv
from sim2skill.wipe_rollout import run_wipe_episode


def test_wipe_reset_is_seeded() -> None:
    env = WipeEnv(seed=0)
    first = env.reset(seed=3)
    second = env.reset(seed=3)

    np.testing.assert_allclose(first, second)
    assert first.shape == (env.observation_size,)


def test_oracle_covers_surface() -> None:
    env = WipeEnv(seed=0, pose_noise=0.0, max_steps=260)
    summary = run_wipe_episode(env, seed=0, policy_name="oracle").summary

    assert summary.coverage >= 0.82
    assert summary.success
    assert summary.steps < 260


def test_pose_noise_hurts_nominal_more_than_oracle() -> None:
    oracle_env = WipeEnv(seed=0, pose_noise=0.04)
    noisy_env = WipeEnv(seed=0, pose_noise=0.04)
    oracle = run_wipe_episode(oracle_env, seed=1, policy_name="oracle").summary
    nominal = run_wipe_episode(noisy_env, seed=1, policy_name="nominal").summary

    assert oracle.coverage > nominal.coverage + 0.05


def test_structured_residual_improves_noisy_nominal() -> None:
    env = WipeEnv(seed=0, pose_noise=0.04)
    nominal = run_wipe_episode(env, seed=2, policy_name="nominal").summary
    structured = run_wipe_episode(
        WipeEnv(seed=0, pose_noise=0.04),
        seed=2,
        policy_name="structured",
    ).summary

    assert structured.coverage >= nominal.coverage
    assert structured.contact_loss_rate <= nominal.contact_loss_rate + 0.05

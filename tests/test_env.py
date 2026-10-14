import numpy as np

from sim2skill.env import PickPlaceEnv


def test_reset_is_seeded() -> None:
    env = PickPlaceEnv(seed=0)
    first = env.reset(seed=42)
    second = env.reset(seed=42)

    np.testing.assert_allclose(first, second)
    assert first.shape == (env.observation_size,)


def test_step_contract() -> None:
    env = PickPlaceEnv(seed=0)
    result = env.step(np.zeros(env.action_size))

    assert result.observation.shape == (env.observation_size,)
    assert np.isfinite(result.reward)
    assert result.terminated is False

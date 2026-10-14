import numpy as np

from sim2skill.mlp import MLP
from sim2skill.residual import collect_residual_data, residual_features
from sim2skill.wipe_env import WipeEnv


def test_mlp_loss_decreases_on_tiny_batch() -> None:
    rng = np.random.default_rng(0)
    features = rng.normal(size=(80, 6))
    targets = np.tanh(features @ rng.normal(size=(6, 4)) * 0.3)
    model = MLP(6, 16, 4, seed=0)
    losses = model.fit(features, targets, steps=80, batch_size=16, lr=0.08, seed=0)

    assert losses[-1] < losses[0]


def test_residual_feature_width_matches_observation() -> None:
    env = WipeEnv(seed=0)
    observation = env.reset(seed=0)
    features = residual_features(observation, np.zeros(4), np.array([1.0, 0, 0, 0]))

    assert features.shape == (env.observation_size + 8,)


def test_residual_dataset_has_finite_labels() -> None:
    features, targets = collect_residual_data(episodes=1, seed=0, pose_noise=0.02)

    assert len(features) > 10
    assert np.isfinite(features).all()
    assert np.isfinite(targets).all()
    assert np.max(np.abs(targets[:, :3])) <= 0.008 + 1e-9

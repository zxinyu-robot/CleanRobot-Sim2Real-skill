import numpy as np

from sim2skill.dataset import episode_splits, transition_mask


def test_episode_splits_are_disjoint() -> None:
    splits = episode_splits(10, seed=0)

    train, val, test = set(splits["train"]), set(splits["val"]), set(splits["test"])
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(range(10))


def test_transition_mask_keeps_whole_episodes() -> None:
    episode_ids = np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
    mask = transition_mask(episode_ids, [1])

    assert mask.tolist() == [False, False, True, True, True, False, False]

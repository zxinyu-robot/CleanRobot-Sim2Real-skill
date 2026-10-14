import numpy as np
import pytest

from sim2skill.safety import ActionShield


def test_shield_limits_cartesian_speed() -> None:
    shield = ActionShield(max_delta=0.01)
    result = shield.filter(np.array([1.0, 0.0, 0.0, 0.5]), np.array([0.0, 0.0, 0.2]))

    assert result.intervened
    assert "cartesian_speed" in result.reasons
    assert np.linalg.norm(result.action[:3]) == pytest.approx(0.01)


def test_shield_rejects_non_finite_action() -> None:
    shield = ActionShield()

    with pytest.raises(ValueError, match="non-finite"):
        shield.filter(np.array([np.nan, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.2]))

from srccode.system import ThreeWheeledRobotKinematic
import numpy as np
from srccode.utils import rg


class ThreeWheeledRobotKinematic(ThreeWheeledRobotKinematic): ...


class ThreeWheeledRobotKinematicStabilizingPolicy:
    """Scenario for non-inertial three-wheeled robot composed of three PID scenarios."""

    def __init__(self, K):
        """Initialize an instance of scenario.

        Args:
            K: gain of scenario
        """
        # super().__init__()
        self.K = K

    def get_action(self, observation):
        x = observation[0, 0]
        y = observation[0, 1]
        angle = observation[0, 2]

        angle_cond = np.arctan2(y, x)

        if not np.allclose((x, y), (0, 0), atol=1e-03) and not np.isclose(
            angle, angle_cond, atol=1e-03
        ):
            omega = (
                -self.K
                * np.sign(angle - angle_cond)
                * rg.sqrt(rg.abs(angle - angle_cond))
            )
            v = 0
        elif not np.allclose((x, y), (0, 0), atol=1e-03) and np.isclose(
            angle, angle_cond, atol=1e-03
        ):
            omega = 0
            v = -self.K * rg.sqrt(rg.norm_2(rg.hstack([x, y])))
        elif np.allclose((x, y), (0, 0), atol=1e-03) and not np.isclose(
            angle, 0, atol=1e-03
        ):
            omega = -self.K * np.sign(angle) * rg.sqrt(rg.abs(angle))
            v = 0
        else:
            omega = 0
            v = 0

        return rg.force_row(rg.hstack([v, omega]))

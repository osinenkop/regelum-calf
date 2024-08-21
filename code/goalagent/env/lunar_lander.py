from srccode.system import LunarLander
import numpy as np
from srccode.utils import rg


class LunarLander(LunarLander):
    pass


class LunarLanderWithOffset(LunarLander):
    def _get_observation(self, time, state, inputs):
        return state - rg.array(
            np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).reshape(*state.shape),
            prototype=state,
            _force_numeric=True,
        )


class LunarLanderStabilizingPolicy:
    def __init__(self, angle_pd_coefs=[180, 120], x_pd_coefs=[10, 40]):
        self.observation_prev = None
        self.action_prev = None
        self.angle_pd_coefs = angle_pd_coefs
        self.x_pd_coefs = x_pd_coefs

    def get_action(self, obs):
        fx = -(
            obs[:, 2, None] * self.angle_pd_coefs[0]
            + obs[:, 5, None] * self.angle_pd_coefs[1]
            - np.cos(obs[:, 2, None]) ** 2
            * (
                obs[:, 0, None] * self.x_pd_coefs[0]
                + obs[:, 3, None] * self.x_pd_coefs[1]
            )
        )
        fy = np.zeros_like(fx)

        action = np.hstack((fx, fy))
        return action


class LunarLanderGoalReachingFunc:
    def __call__(self, observation):
        return observation[0, 1] <= 20.0

import numpy as np
from srccode.objective import RunningObjective as RgRunningObjective
from srccode.model import ModelQuadLin
from srccode.typing import RgArray
from srccode.utils import rg


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class QuadraticRunningObjective:
    def __init__(self, weights: np.ndarray, biases: np.ndarray | float = 0.0) -> None:
        self.weights = weights
        self.biases = biases

    def __call__(self, state: np.ndarray, action: np.ndarray) -> float:
        return (np.hstack((state - self.biases, action)) ** 2 * self.weights).sum()


class GymPendulumRunningObjective:

    def __call__(self, state, action):
        return (
            angle_normalize(state[0]) ** 2
            + 0.1 * state[1] ** 2
            + 0.001 * action[0] ** 2
        )


class InvertedPendulumRunningObjective:
    def __call__(self, state, action):
        return 20 * (1 - np.cos(state[0])) + 2 * state[2] ** 2


class LunarLanderRunningObjective(QuadraticRunningObjective):
    def __call__(self, state: np.ndarray, action: np.ndarray) -> float:
        if state[1] <= 1.01 and np.abs(state[2]) <= 0.05:  # safe landed lander
            return 0.0
        else:
            input_state = np.copy(state)
            # input_state[2] = angle_normalize(state[2])
            r_obj = super().__call__(input_state, action)
            return r_obj


class RgQuadraticRunningObjective(RgRunningObjective):

    def __init__(self, weights):
        super().__init__(
            model=ModelQuadLin(
                quad_matrix_type="diagonal", is_with_linear_terms=False, weights=weights
            )
        )


class RgLunarLanderRunningObjective(RgQuadraticRunningObjective):
    def __call__(
        self,
        observation: RgArray,
        action: RgArray,
        is_save_batch_format: bool = False,
    ):
        quadratic_r_obj = super().__call__(observation, action, is_save_batch_format)
        is_safe_landed = rg.logic_and(
            observation[0, 1] <= 0.01, observation[0, 2] <= 0.05
        )
        return rg.if_else(
            is_safe_landed,
            0.0 * quadratic_r_obj,
            quadratic_r_obj,
        )

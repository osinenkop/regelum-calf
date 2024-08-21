from srccode.system import TwoTank
import numpy as np
from srccode.utils import rg


class TwoTank(TwoTank): ...


class TwoTankWithOffset(TwoTank):

    def _get_observation(self, time, state, inputs):
        return state - rg.array(
            np.array([0.4, 0.4]).reshape(*state.shape),
            prototype=state,
            _force_numeric=True,
        )


class TwoTankStabilizingPolicy:
    def __init__(self, p, d):
        self.p = p
        self.d = d

    def get_action(self, observation):
        state = observation + 0.4
        h1 = state[0, 0]
        h2 = state[0, 1]
        tau_1 = TwoTank._parameters["tau1"]
        tau_2 = TwoTank._parameters["tau2"]
        K_1 = TwoTank._parameters["K1"]
        K_2 = TwoTank._parameters["K2"]
        K_3 = TwoTank._parameters["K3"]
        return np.array(
            [
                [
                    (
                        1
                        / (1 + self.d * K_1 / tau_1)
                        * (
                            -self.p * (h1 - 0.4)
                            + self.d * h1 / tau_1
                            - self.p * (h2 - 0.4)
                            - self.d * (-h1 + K_2 * h1 + K_3 * h2**2)
                        )
                    )
                ]
            ]
        )

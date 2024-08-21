from srccode.simulator import CasADi
from srccode.system import System, ComposedSystem
from typing import Union, Optional, Callable
import numpy as np


class UniformStateInitGenerator:
    def __init__(self, bounds: list[list[float]]):
        self.bounds = np.array(bounds)

    def __call__(self):
        state_init = np.random.uniform(
            low=self.bounds[:, 0], high=self.bounds[:, 1]
        ).reshape(1, -1)

        return state_init


def generate_state_init_for_pendulum():
    return np.array(
        [
            [
                np.random.uniform(low=-np.pi, high=np.pi),
                np.random.uniform(-1, 1),
            ]
        ]
    )


class StateInitRandomSamplerSimulator(CasADi):
    def __init__(
        self,
        system: Union[System, ComposedSystem],
        state_init: Callable[[], np.ndarray] = generate_state_init_for_pendulum,
        action_init: Optional[np.ndarray] = None,
        time_final: Optional[float] = 1,
        max_step: Optional[float] = 1e-3,
        first_step: Optional[float] = 1e-6,
        atol: Optional[float] = 1e-5,
        rtol: Optional[float] = 1e-3,
    ):
        self.state_init_callable = state_init
        self.state_init = self.state_init_callable()
        super().__init__(
            system=system,
            state_init=self.state_init,
            time_final=time_final,
            action_init=action_init,
            max_step=max_step,
            first_step=first_step,
            atol=atol,
            rtol=rtol,
        )

    def reset(self):
        self.state_init = self.state_init_callable()
        if self.system.system_type == "diff_eqn":
            self.ODE_solver = self.initialize_ode_solver()
            self.time = 0.0
            self.state = self.state_init
            self.observation = self.get_observation(
                time=self.time, state=self.state_init, inputs=self.action_init
            )
        else:
            self.time = 0.0
            self.observation = self.get_observation(
                time=self.time, state=self.state_init, inputs=self.system.inputs
            )

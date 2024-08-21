from typing import Optional
from srccode.simulator import Simulator, CasADi
import gymnasium as gym
import numpy as np
from typing import Callable
from srccode.system import KinematicPoint
import srccode


class RgEnv(gym.Env):
    def __init__(
        self,
        simulator: Simulator,
        running_objective: Callable[[np.ndarray], float],
        action_space: Optional[gym.spaces.Box] = None,
        observation_space: Optional[gym.spaces.Box] = None,
    ) -> None:
        self.simulator = simulator
        self.running_objective = running_objective
        action_bounds = np.array(simulator.system._action_bounds)
        if action_space is None:
            self.action_space = gym.spaces.Box(
                low=action_bounds[:, 0], high=action_bounds[:, 1]
            )
        else:
            self.action_space = action_space
        if observation_space is None:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(simulator.system._dim_observation,)
            )
        else:
            self.observation_space = observation_space

    def step(self, u):
        self.simulator.receive_action(self.simulator.system.apply_action_bounds(u))
        costs = self.running_objective(self.state, u)
        sim_step = self.simulator.do_sim_step()
        self.state = np.copy(self.simulator.state).reshape(-1)
        return self._get_obs(), -costs, False, sim_step is not None, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.simulator.reset()
        self.state = np.copy(self.simulator.state).reshape(-1)
        return self._get_obs(), {}

    def _get_obs(self):
        return self.simulator.system._get_observation(
            None, self.state, None
        )  # .astype(np.float32)


class QuadraticRunningObjective:
    def __init__(self, weights: np.ndarray, biases: np.ndarray | float = 0.0) -> None:
        self.weights = weights
        self.biases = biases

    def __call__(self, state, action) -> float:
        return (np.hstack((state - self.biases, action)) ** 2 * self.weights).sum()


env = RgEnv(
    simulator=CasADi(
        system=KinematicPoint(),
        state_init=np.array([[-10, -10]]),
        time_final=10,
        max_step=0.01,
    ),
    running_objective=QuadraticRunningObjective(
        weights=np.array([10.0, 10.0, 0.0, 0.0])
    ),
)

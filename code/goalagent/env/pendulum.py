import numpy as np
import torch
from srccode.system import InvertedPendulum as Pendulum
from srccode.utils import rg


class PendulumQuanser(Pendulum):
    """The parameters of this system roughly resemble those of a Quanser Rotary Inverted Pendulum."""

    _parameters = {"mass": 0.127, "grav_const": 9.81, "length": 0.337}
    _action_bounds = [[-0.1, 0.1]]
    _dim_observation = 2

    def pendulum_moment_inertia(self):
        return self._parameters["mass"] * self._parameters["length"] ** 2 / 3

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        mass, grav_const, length = (
            self._parameters["mass"],
            self._parameters["grav_const"],
            self._parameters["length"],
        )
        Dstate[0] = state[1]
        Dstate[1] = (
            grav_const * mass * length * rg.sin(state[0]) / 2 + inputs[0]
        ) / self.pendulum_moment_inertia()

        return Dstate

    def _get_observation(self, time, state, inputs):
        observation = rg.zeros(self._dim_state, prototype=state)

        observation[0] = state[0]
        observation[1] = state[1]

        return observation


class PendulumQuanserWithGymObservation(Pendulum):
    """The parameters of this system roughly resemble those of a Quanser Rotary Inverted Pendulum."""

    _parameters = {"mass": 0.127, "grav_const": 9.81, "length": 0.337}
    _action_bounds = [[-0.1, 0.1]]
    _dim_observation = 3

    def pendulum_moment_inertia(self):
        return self._parameters["mass"] * self._parameters["length"] ** 2 / 3

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        mass, grav_const, length = (
            self._parameters["mass"],
            self._parameters["grav_const"],
            self._parameters["length"],
        )
        Dstate[0] = state[1]
        Dstate[1] = (
            grav_const * mass * length * rg.sin(state[0]) / 2 + inputs[0]
        ) / self.pendulum_moment_inertia()

        return Dstate

    def _get_observation(self, time, state, inputs):
        observation = rg.zeros(self._dim_observation, prototype=state)

        observation[0] = rg.cos(state[0])
        observation[1] = rg.sin(state[0])
        observation[2] = state[1]

        return observation


class PendulumQuanserWithNormObservation(Pendulum):
    """The parameters of this system roughly resemble those of a Quanser Rotary Inverted Pendulum."""

    _parameters = {"mass": 0.127, "grav_const": 9.81, "length": 0.337}
    _action_bounds = [[-0.1, 0.1]]
    _dim_observation = 3

    def pendulum_moment_inertia(self):
        return self._parameters["mass"] * self._parameters["length"] ** 2 / 3

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        mass, grav_const, length = (
            self._parameters["mass"],
            self._parameters["grav_const"],
            self._parameters["length"],
        )
        Dstate[0] = state[1]
        Dstate[1] = (
            grav_const * mass * length * rg.sin(state[0]) / 2 + inputs[0]
        ) / self.pendulum_moment_inertia()

        return Dstate

    def _get_observation(self, time, state, inputs):
        observation = rg.zeros(self._dim_observation, prototype=state)

        observation[0] = rg.cos(state[0])
        observation[1] = rg.sin(state[0])
        observation[2] = state[1]

        return observation


class Pendulum(Pendulum):
    """The parameters of this system roughly resemble those of a Quanser Rotary Inverted Pendulum."""

    _parameters = {"mass": 1.0, "grav_const": 10.0, "length": 1.0}
    action_bounds = [[-2.0, 2.0]]
    _dim_observation = 3

    def pendulum_moment_inertia(self):
        return self._parameters["mass"] * self._parameters["length"] ** 2 / 3

    def _compute_state_dynamics(self, time, state, inputs):
        Dstate = rg.zeros(
            self.dim_state,
            prototype=(state, inputs),
        )

        mass, grav_const, length = (
            self._parameters["mass"],
            self._parameters["grav_const"],
            self._parameters["length"],
        )
        Dstate[0] = state[1]
        Dstate[1] = (
            grav_const * mass * length * rg.sin(state[0]) / 2 + inputs[0]
        ) / self.pendulum_moment_inertia()

        return Dstate

    def _get_observation(self, time, state, inputs):
        observation = rg.zeros(self._dim_observation, prototype=state)

        observation[0] = rg.cos(state[0])
        observation[1] = rg.sin(state[0])
        observation[2] = state[1]

        return observation


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class PendulumObserver:
    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        cos_angles = observation[:, 0, None]
        sin_angles = observation[:, 1, None]
        angles = torch.arctan2(sin_angles, cos_angles)
        velocities = observation[:, 2, None]
        return torch.cat([angles, velocities], dim=1)


def hard_switch(signal1: float, signal2: float, condition: bool):
    if condition:
        return signal1
    else:
        return signal2


class PendulumStabilizingPolicy:
    def __init__(
        self,
        gain: float,
        action_min: float,
        action_max: float,
        switch_loc: float,
        switch_vel_loc: float,
        pd_coeffs: np.ndarray,
        system: Pendulum,
    ):
        self.gain = gain
        self.action_min = action_min
        self.action_max = action_max
        self.switch_loc = switch_loc
        self.pd_coeffs = pd_coeffs
        self.switch_vel_loc = switch_vel_loc
        self.system = system

    def get_action(self, observation: np.ndarray) -> np.ndarray:
        params = self.system._parameters
        mass, grav_const, length = (
            params["mass"],
            params["grav_const"],
            params["length"],
        )
        if np.prod(observation.shape) == 3:
            cos_angle = observation[0, 0]
            sin_angle = observation[0, 1]

            angle = np.arctan2(sin_angle, cos_angle)
            angle_vel = observation[0, 2]
        elif np.prod(observation.shape) == 2:
            angle = observation[0, 0]
            angle_vel = observation[0, 1]

        energy_total = (
            mass * grav_const * length * (np.cos(angle) - 1) / 2
            + 1 / 2 * self.system.pendulum_moment_inertia() * angle_vel**2
        )
        energy_control_action = -self.gain * np.sign(angle_vel * energy_total)

        action = hard_switch(
            signal1=energy_control_action,
            signal2=-self.pd_coeffs[0] * np.sin(angle) - self.pd_coeffs[1] * angle_vel,
            condition=np.cos(angle) <= self.switch_loc
            or np.abs(angle_vel) > self.switch_vel_loc,
        )

        return np.array(
            [
                [
                    np.clip(
                        action,
                        self.action_min,
                        self.action_max,
                    )
                ]
            ],
        )

    def buffer_current_observation_action(self, observation, action):
        """Buffers the current observation and action.

        This method is intended to fit the interface required by CALFQ.
        The controller does not utilize the buffered observation and action directly.

        Args:
        - observation: The current observed state.
        - action: The action taken.
        """

        pass

    def reset(self):
        pass


class PendulumGoalReachingFunction:

    def __init__(self, goal_threshold: float):
        self.goal_threshold = goal_threshold

    def __call__(self, observation: np.ndarray) -> bool:
        angle = observation[0, 0]
        return 1 - np.cos(angle) <= self.goal_threshold

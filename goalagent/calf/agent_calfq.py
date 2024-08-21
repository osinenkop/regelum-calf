import numpy as np
from typing import Union, Optional
from goalagent.env.pendulum import Pendulum, PendulumQuanser

import numpy as np
import scipy as sp

from typing import Union, Callable, Any
from numpy.core.multiarray import array
from pathlib import Path
from numpy.matlib import repmat
from numpy.linalg import norm

from scipy.optimize import minimize

from goalagent.calf.utilities import uptria2vec
from goalagent.calf.utilities import to_row_vec
from goalagent.calf.utilities import to_scalar
from goalagent.calf.utilities import push_vec
from srccode.utils import rg
from srccode.policy import MemoryPIDPolicy
import random


def hard_switch(signal1: float, signal2: float, condition: bool):
    if condition:
        return signal1
    else:
        return signal2


class AgentCALFQ:
    def __init__(
        self,
        nominal_policy,
        system: Union[Pendulum, PendulumQuanser],
        relax_probability_min: float,
        relax_probability_max: float,
        goal_reaching_func: Callable[[np.ndarray], bool],
        running_objective: Callable[[np.ndarray, np.ndarray], float],
        action_sampling_period: float,
        critic_learn_rate: float,
        critic_num_grad_steps: int,
        buffer_size: int,
        actor_opt_method: str,
        actor_opt_options: dict[str, Any],
        use_grad_descent: bool,
        use_decay_constraint: bool,
        use_kappa_constraint: bool,
        check_persistence_of_excitation: bool,
        critic_weight_change_penalty_coeff: float,
        discount_factor: float = 1.0,
        critic_weights_init: Optional[np.ndarray] = None,
        critic_struct: str = "quad-mix",
        safe_only: bool = False,
        relax_probability_stabilize_global_step: int = 3000,
        relax_factor: float = 0.9999999999,
    ):
        self.relax_factor = relax_factor
        self.relax_probability_stabilize_global_step = (
            relax_probability_stabilize_global_step
        )
        self.use_grad_descent = use_grad_descent
        self.use_decay_constraint = use_decay_constraint
        self.use_kappa_constraint = use_kappa_constraint
        self.check_persistence_of_excitation = check_persistence_of_excitation
        self.action_sampling_period = action_sampling_period
        self.nominal_policy = nominal_policy
        self.running_objective = running_objective
        # 1. Common agent tuning settings
        # 2. Actor
        self.action_change_penalty_coeff = 0.0
        # 3. Critic
        self.critic_learn_rate = critic_learn_rate
        # critic_num_grad_steps>1 may lead to unexpected results. Possibly rehydra glitch
        self.critic_num_grad_steps = critic_num_grad_steps
        self.discount_factor = discount_factor
        self.buffer_size = buffer_size
        self.critic_struct = critic_struct
        self.critic_weight_change_penalty_coeff = (
            critic_weight_change_penalty_coeff  # 1e4  # 0.0
        )
        # 4. CALFQ
        self.actor_opt_method = actor_opt_method
        self.actor_opt_options = actor_opt_options
        self.safe_only = safe_only
        self.relax_probability = relax_probability_min  # Probability to take CALF action even when CALF constraints are not satisfied
        self.relax_probability_min = relax_probability_min
        self.relax_probability_max = relax_probability_max
        self.goal_reaching_func = goal_reaching_func
        self.critic_low_kappa_coeff = 1e-2
        self.critic_up_kappa_coeff = 1e4
        self.critic_desired_decay_coeff = 1e-4
        self.critic_max_desired_decay_coeff = 1e-1
        self.calf_penalty_coeff = 0.5

        self.dim_state = system._dim_state
        self.dim_action = system._dim_inputs
        self.dim_observation = system._dim_observation
        self.clock = 1

        self.score = 0
        self.critic_big_number = 1e5
        self.action_bounds = np.array(system._action_bounds)
        self.system = system
        if self.critic_struct == "quad-lin":
            self.dim_critic = int(
                ((self.dim_observation + self.dim_action) + 1)
                * (self.dim_observation + self.dim_action)
                / 2
                + (self.dim_observation + self.dim_action)
            )
            self.critic_weight_min = -self.critic_big_number
            self.critic_weight_max = self.critic_big_number
        elif self.critic_struct == "quadratic":
            self.dim_critic = int(
                ((self.dim_observation + self.dim_action) + 1)
                * (self.dim_observation + self.dim_action)
                / 2
            )
            self.critic_weight_min = 0
            self.critic_weight_max = self.critic_big_number
        elif self.critic_struct == "quad-nomix":
            self.dim_critic = self.dim_observation + self.dim_action
            self.critic_weight_min = 0
            self.critic_weight_max = self.critic_big_number
        elif self.critic_struct == "quad-mix":
            self.dim_critic = int(
                self.dim_observation
                + self.dim_observation * self.dim_action
                + self.dim_action
            )
            self.critic_weight_min = -self.critic_big_number
            self.critic_weight_max = self.critic_big_number
        self.critic_weight_init_guess = critic_weights_init

        self.critic_desired_decay = (
            self.critic_desired_decay_coeff * self.action_sampling_period
        )
        self.critic_max_desired_decay = (
            self.critic_max_desired_decay_coeff * self.action_sampling_period
        )

        self.relax_probability_init = self.relax_probability

    def run_obj(self, observation, action):
        return to_scalar(
            self.running_objective(observation.reshape(-1), action.reshape(-1))
        )

    def critic_model(self, critic_weight_tensor, observation, action):

        observation_action = np.hstack([to_row_vec(observation), to_row_vec(action)])

        if self.critic_struct == "quad-lin":
            feature_tensor = np.hstack(
                [
                    uptria2vec(
                        np.outer(observation_action, observation_action),
                        force_row_vec=True,
                    ),
                    observation_action,
                ]
            )
        elif self.critic_struct == "quadratic":
            feature_tensor = uptria2vec(
                np.outer(observation_action, observation_action), force_row_vec=True
            )
        elif self.critic_struct == "quad-nomix":
            feature_tensor = observation_action * observation_action
        elif self.critic_struct == "quad-mix":
            feature_tensor = np.hstack(
                [
                    to_row_vec(observation) ** 2,
                    np.kron(to_row_vec(observation), to_row_vec(action)),
                    to_row_vec(action) ** 2,
                ]
            )

        result = critic_weight_tensor @ feature_tensor.T

        return to_scalar(result)

    def critic_model_grad(self, critic_weight_tensor, observation, action):

        observation_action = np.hstack([to_row_vec(observation), to_row_vec(action)])

        if self.critic_struct == "quad-lin":
            feature_tensor = np.hstack(
                [
                    uptria2vec(
                        np.outer(observation_action, observation_action),
                        force_row_vec=True,
                    ),
                    observation_action,
                ]
            )
        elif self.critic_struct == "quadratic":
            feature_tensor = uptria2vec(
                np.outer(observation_action, observation_action), force_row_vec=True
            )
        elif self.critic_struct == "quad-nomix":
            feature_tensor = observation_action * observation_action
        elif self.critic_struct == "quad-mix":
            feature_tensor = np.hstack(
                [
                    to_row_vec(observation) ** 2,
                    np.kron(to_row_vec(observation), to_row_vec(action)),
                    to_row_vec(action) ** 2,
                ]
            )

        return feature_tensor

    def critic_obj(self, critic_weight_tensor_change):
        """
        Objective function for critic learning.

        Uses value iteration format where previous weights are assumed different from the ones being optimized.

        """
        critic_weight_tensor_pivot = self.critic_weight_tensor_safe
        critic_weight_tensor = (
            self.critic_weight_tensor_safe + critic_weight_tensor_change
        )

        result = 0

        for k in range(self.buffer_size - 1, 0, -1):
            observation_prev = self.observation_buffer[k - 1, :]
            observation_next = self.observation_buffer[k, :]
            action_prev = self.action_buffer[k - 1, :]
            action_next = self.action_buffer[k, :]

            critic_prev = self.critic_model(
                critic_weight_tensor, observation_prev, action_prev
            )
            critic_next = self.critic_model(
                critic_weight_tensor_pivot, observation_next, action_next
            )

            temporal_error = (
                critic_prev
                - self.discount_factor * critic_next
                - self.run_obj(observation_prev, action_prev)
            )

            result += 1 / 2 * temporal_error**2

        result += (
            1
            / 2
            * self.critic_weight_change_penalty_coeff
            * norm(critic_weight_tensor_change) ** 2
        )

        return result

    def critic_obj_grad(self, critic_weight_tensor):
        """
        Gradient of the objective function for critic learning.

        Uses value iteration format where previous weights are assumed different from the ones being optimized.

        """
        critic_weight_tensor_pivot = self.critic_weight_tensor_safe
        critic_weight_tensor_change = critic_weight_tensor_pivot - critic_weight_tensor

        result = to_row_vec(np.zeros(self.dim_critic))

        for k in range(self.buffer_size - 1, 0, -1):

            observation_prev = self.observation_buffer[k - 1, :]
            observation_next = self.observation_buffer[k, :]
            action_prev = self.action_buffer[k - 1, :]
            action_next = self.action_buffer[k, :]

            critic_prev = self.critic_model(
                critic_weight_tensor, observation_prev, action_prev
            )
            critic_next = self.critic_model(
                critic_weight_tensor_pivot, observation_next, action_next
            )

            temporal_error = (
                critic_prev
                - self.discount_factor * critic_next
                - self.run_obj(observation_prev, action_prev)
            )

            result += temporal_error * self.critic_model_grad(
                critic_weight_tensor, observation_prev, action_prev
            )

        result += self.critic_weight_change_penalty_coeff * critic_weight_tensor_change

        return result

    def calf_diff(self, critic_weight_tensor, observation, action):
        # Q^w  (s_t, a_t)
        critic_new = self.critic_model(critic_weight_tensor, observation, action)
        # Q^w† (s†, a†)
        critic_safe = self.critic_model(
            self.critic_weight_tensor_safe, self.observation_safe, self.action_safe
        )
        # Q^w  (s_t, a_t) - Q^w† (s†, a†)
        return critic_new - critic_safe

    def calf_decay_constraint_penalty_grad(
        self, critic_weight_tensor, observation, action
    ):
        # This one is handy for explicit gradient-descent optimization.
        # We take a ReLU here

        critic_new = self.critic_model(critic_weight_tensor, observation, action)

        critic_safe = self.critic_model(
            self.critic_weight_tensor_safe, self.observation_safe, self.action_safe
        )

        if critic_new - critic_safe <= -self.critic_desired_decay:
            relu_grad = 0
        else:
            relu_grad = 1

        return (
            self.calf_penalty_coeff
            * self.critic_model_grad(critic_weight_tensor, observation, action)
            * relu_grad
        )

    def get_optimized_critic_weights(
        self,
        observation,
        action,
        use_grad_descent=True,
        use_decay_constraint=True,
        use_kappa_constraint=False,
        check_persistence_of_excitation=False,
    ):

        if use_grad_descent:

            # Usage of np.array here forces passing by value due to
            # new instantiation. It is needed since Python
            # passes containers (an array for example) by reference by default
            critic_weight_tensor = np.array(self.critic_weight_tensor)

            for _ in range(self.critic_num_grad_steps):

                critic = self.critic_model(critic_weight_tensor, observation, action)

                critic_weight_tensor_change = (
                    -self.critic_learn_rate * self.critic_obj_grad(critic_weight_tensor)
                )

                if use_kappa_constraint:

                    critic_low_kappa = (
                        self.critic_low_kappa_coeff * norm(observation) ** 2
                    )
                    critic_up_kappa = (
                        self.critic_up_kappa_coeff * norm(observation) ** 2
                    )

                    # Simple ReLU penalties for bounding kappas
                    if critic <= critic_up_kappa:
                        relu_kappa_up_grad = 0
                    else:
                        relu_kappa_up_grad = 1

                    if critic >= critic_low_kappa:
                        relu_kappa_low_grad = 0
                    else:
                        relu_kappa_low_grad = 1

                    critic_weight_tensor_change += -self.critic_learn_rate * (
                        +self.calf_penalty_coeff
                        * self.critic_model_grad(
                            critic_weight_tensor, observation, action
                        )
                        * relu_kappa_low_grad
                        + self.calf_penalty_coeff
                        * self.critic_model_grad(
                            critic_weight_tensor, observation, action
                        )
                        * relu_kappa_up_grad
                    )

                if use_decay_constraint:
                    critic_weight_tensor_change += -self.critic_learn_rate * (
                        self.calf_decay_constraint_penalty_grad(
                            self.critic_weight_tensor, observation, action
                        )
                    )

                critic_weight_tensor += critic_weight_tensor_change

        else:

            # Optimization method of critic. Methods that respect constraints: BFGS, L-BFGS-B, SLSQP,
            # trust-constr, Powell
            critic_opt_method = "SLSQP"
            if critic_opt_method == "trust-constr":
                # 'disp': True, 'verbose': 2}
                critic_opt_options = {"maxiter": 40, "disp": False}
            else:
                critic_opt_options = {
                    "maxiter": 40,
                    "maxfev": 80,
                    "disp": False,
                    "adaptive": True,
                    "xatol": 1e-3,
                    "fatol": 1e-3,
                }  # 'disp': True, 'verbose': 2}

            constraints = []

            if use_decay_constraint:
                constraints.append(
                    sp.optimize.NonlinearConstraint(
                        lambda critic_weight_tensor: self.calf_diff(
                            critic_weight_tensor=critic_weight_tensor,
                            observation=observation,
                            action=action,
                        ),
                        -self.critic_max_desired_decay,
                        -self.critic_desired_decay,
                    )
                )

            if use_kappa_constraint:

                critic_low_kappa = self.critic_low_kappa_coeff * norm(observation) ** 2
                critic_up_kappa = self.critic_up_kappa_coeff * norm(observation) ** 2
                constraints.append(
                    sp.optimize.NonlinearConstraint(
                        lambda critic_weight_tensor: self.critic_model(
                            critic_weight_tensor=critic_weight_tensor,
                            observation=observation,
                            action=action,
                        ),
                        critic_low_kappa,
                        critic_up_kappa,
                    )
                )

            bounds = sp.optimize.Bounds(
                self.critic_weight_min, self.critic_weight_max, keep_feasible=True
            )

            # Is deliberately 1D specifically for sp.optimize
            critic_weight_tensor_change_start_guess = np.zeros(self.dim_critic)

            critic_weight_tensor_change = minimize(
                self.critic_obj,
                critic_weight_tensor_change_start_guess,
                method=critic_opt_method,
                tol=1e-3,
                bounds=bounds,
                constraints=constraints,
                options=critic_opt_options,
            ).x

        if check_persistence_of_excitation:
            # Adjust the weight change by the replay condition number
            critic_weight_tensor_change *= (
                1
                / np.linalg.cond(self.observation_buffer)
                * 1
                / np.linalg.cond(self.action_buffer)
            )

        return np.clip(
            self.critic_weight_tensor + critic_weight_tensor_change,
            self.critic_weight_min,
            self.critic_weight_max,
        )

    def actor_obj(self, action_change, critic_weight_tensor, observation):
        """
        Objective function for actor learning.

        """

        # Using nominal stabilizer as a pivot
        result = self.critic_model(
            critic_weight_tensor,
            observation,
            action_change,
        )

        result += self.action_change_penalty_coeff * norm(action_change)

        return result

    def get_optimized_action(self, critic_weight_tensor, observation):
        action_change_start_guess = np.zeros(self.dim_action)
        action_change = minimize(
            lambda action_change: self.actor_obj(
                action_change, critic_weight_tensor, observation
            ),
            action_change_start_guess,
            method=self.actor_opt_method,
            tol=1e-3,
            bounds=self.action_bounds,
            options=self.actor_opt_options,
        ).x

        return self.action_curr + action_change

    def update_calf_state(self, critic_weight_tensor, observation, action):
        self.critic_weight_tensor_safe = critic_weight_tensor
        self.observation_safe = observation
        self.action_safe = action

        self.observation_buffer_safe = push_vec(
            self.observation_buffer_safe, observation
        )
        self.action_buffer_safe = push_vec(self.action_buffer_safe, action)

    def calf_filter(self, critic_weight_tensor, observation, action):
        """
        If CALF constraints are satisfied, put the specified action through and update the CALF's state
        (safe weights, observation, action).
        Otherwise, return a safe action, do not update the CALF's state.

        """

        critic_low_kappa = self.critic_low_kappa_coeff * norm(observation) ** 2
        critic_up_kappa = self.critic_up_kappa_coeff * norm(observation) ** 2

        sample = np.random.rand()

        if (
            -self.critic_max_desired_decay
            <= self.calf_diff(critic_weight_tensor, observation, action)
            <= -self.critic_desired_decay
            and critic_low_kappa
            <= self.critic_model(
                critic_weight_tensor,
                observation,
                action,
            )
            <= critic_up_kappa
            or sample <= self.relax_probability
        ):

            self.update_calf_state(critic_weight_tensor, observation, action)

            return action

        else:

            # self.safe_count += 1
            return self.get_safe_action(observation)

    def get_safe_action(self, observation: np.ndarray) -> np.ndarray:
        action = self.nominal_policy.get_action(observation)
        return action

    def get_action(self, observation: np.ndarray) -> np.ndarray:

        # Update replay buffers
        self.action_buffer = push_vec(self.action_buffer, self.action_curr)
        self.observation_buffer = push_vec(self.observation_buffer, observation)

        # Update action
        new_action = self.get_optimized_action(self.critic_weight_tensor, observation)

        self.score += (
            self.run_obj(observation, self.action_curr) * self.action_sampling_period
        )

        # Compute new critic weights
        self.critic_weight_tensor = self.get_optimized_critic_weights(
            observation,
            self.action_curr,
            use_grad_descent=self.use_grad_descent,
            use_decay_constraint=self.use_decay_constraint,
            use_kappa_constraint=self.use_kappa_constraint,
            check_persistence_of_excitation=self.check_persistence_of_excitation,
        )

        # Specific for pendulum
        if self.goal_reaching_func(observation):
            self.relax_probability = 0.0

        # Apply CALF filter that checks constraint satisfaction and updates the CALF's state
        action = self.calf_filter(self.critic_weight_tensor, observation, new_action)

        if self.safe_only:
            action = self.get_safe_action(observation)

        self.clock += 1

        # Force proper dimensionsing according to the convention and apply bounds
        action = self.system.apply_action_bounds(to_row_vec(action))

        # Update current action
        self.action_curr = action

        self.relax_probability *= self.relax_factor
        return action

    def reset(self, obs_init, global_step: int):
        fraction = np.clip(
            (global_step / (self.relax_probability_stabilize_global_step - 1)), 0, 1
        )
        # UPDATING THE relax_probability PARAMETER. ANY TRIGGER OR CRITERION CAN BE PLUGGED IN HERE.
        self.relax_probability = (
            self.relax_probability_min
            + (self.relax_probability_max - self.relax_probability_min) * fraction
        )
        random_sampled_critic_weight_tensor_init = to_row_vec(
            np.random.uniform(
                self.critic_big_number / 100,
                self.critic_big_number / 101,
                size=self.dim_critic,
            )
        )
        if self.critic_weight_init_guess is not None:
            if np.prod(self.critic_weight_init_guess.shape) != self.dim_critic:
                raise ValueError(
                    "Dimension mismatch. Cannot instantiate critic with provided initial weights."
                )
            critic_weight_tensor_init = to_row_vec(self.critic_weight_init_guess)
        else:
            critic_weight_tensor_init = random_sampled_critic_weight_tensor_init

        self.critic_weight_tensor = np.copy(critic_weight_tensor_init)
        self.critic_weight_tensor_safe = np.copy(critic_weight_tensor_init)
        self.clock = 1
        self.nominal_policy.reset()
        self.action_safe = self.nominal_policy.get_action(obs_init)
        self.action_curr = np.copy(self.action_safe)
        self.observation_safe = np.copy(obs_init)
        self.action_buffer = repmat(self.action_safe, self.buffer_size, 1)
        self.observation_buffer = repmat(obs_init, self.buffer_size, 1)
        self.action_buffer_safe = np.zeros([self.buffer_size, self.dim_action])
        self.observation_buffer_safe = np.zeros(
            [self.buffer_size, self.dim_observation]
        )

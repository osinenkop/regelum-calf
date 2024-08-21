from srccode.critic import Critic
from srccode.utils import rg
from srccode.typing import RgArray
from typing import Union, Optional, List
from srccode.policy import RLPolicy
from srccode.optimizable import OptimizerConfig, CasadiOptimizerConfig
from srccode.model import Model, ModelWeightContainer
from srccode.predictor import EulerPredictor
from srccode.optimizable import OptStatus
from goalagent.env.rg_env import RgEnv
from srccode.data_buffers import DataBuffer
import numpy as np
from srccode.observer import Observer, ObserverTrivial, ObserverReference


class CriticCALF(Critic):
    """Critic for CALF algorithm."""

    def __init__(
        self,
        system,
        model: Model,
        is_same_critic: bool,
        is_value_function: bool,
        td_n: int = 1,
        predictor: Optional[Model] = None,
        optimizer_config: Optional[OptimizerConfig] = None,
        discount_factor: float = 1.0,
        sampling_time: float = 0.01,
        safe_decay_param: float = 1e-4,
        is_dynamic_decay_rate: bool = True,
        safe_policy=None,
        lb_parameter: float = 1e-6,
        ub_parameter: float = 1e3,
        regularization_param: float = 0,
        is_decay_upper_bounded: bool = False,
    ):
        """Instantiate a CriticCALF object."""
        super().__init__(
            system=system,
            model=model,
            td_n=td_n,
            is_same_critic=is_same_critic,
            is_value_function=is_value_function,
            is_on_policy=True,
            optimizer_config=optimizer_config,
            discount_factor=discount_factor,
            sampling_time=sampling_time,
            action_bounds=None,
            regularization_param=regularization_param,
        )
        self.predictor = predictor
        self.safe_decay_param = safe_decay_param
        self.is_dynamic_decay_rate = is_dynamic_decay_rate
        if not self.is_dynamic_decay_rate:
            self.safe_decay_rate = self.safe_decay_param

        self.lb_parameter = lb_parameter
        self.ub_parameter = ub_parameter
        self.safe_policy = safe_policy

        self.observation_last_good_var = self.create_variable(
            self.batch_size,
            self.system.dim_observation,
            name="observation_last_good",
            is_constant=True,
        )
        self.prev_good_critic_var = self.create_variable(
            1,
            1,
            name="prev_good_critic",
            is_constant=True,
            is_nested_function=True,
            nested_variables=[
                self.observation_last_good_var,
                self.critic_stored_weights_var,
            ],
        )
        self.connect_source(
            connect_to=self.prev_good_critic_var,
            func=self.model.cache,
            source=self.observation_last_good_var,
            weights=self.critic_stored_weights_var,
        )

        self.register_constraint(
            self.CALF_decay_constraint_no_prediction,
            variables=[
                self.critic_model_output,
                self.prev_good_critic_var,
            ],
        )
        if is_decay_upper_bounded:
            self.register_constraint(
                self.CALF_decay_constraint_no_prediction_upper,
                variables=[
                    self.critic_model_output,
                    self.prev_good_critic_var,
                ],
            )
        self.register_constraint(
            self.CALF_critic_lower_bound_constraint,
            variables=[self.critic_model_output, self.observation_var],
        )

        self.observation_last_good = None

    def data_buffer_objective_keys(self) -> List[str]:
        """Return a list of `srccode.data_buffers.DataBuffer` keys to be used for the substitution to the objective function.

        Returns:
            List of keys.
        """
        keys = super().data_buffer_objective_keys()
        keys.append("observation_last_good")
        return keys

    def CALF_decay_constraint_no_prediction(
        self,
        critic_model_output,
        prev_good_critic,
    ):
        stabilizing_constraint_violation = (
            critic_model_output[-1, :]
            - prev_good_critic[-2, :]
            + self.sampling_time * self.safe_decay_rate
        )
        return stabilizing_constraint_violation

    def CALF_decay_constraint_no_prediction_upper(
        self,
        critic_model_output,
        prev_good_critic,
    ):
        CALF_decay_constraint_no_prediction_upper = (
            prev_good_critic[-2, :]
            - critic_model_output[-1, :]
            - self.safe_decay_rate * 5
        )
        return CALF_decay_constraint_no_prediction_upper

    def CALF_critic_lower_bound_constraint(
        self, critic_model_output: RgArray, observation: RgArray
    ) -> RgArray:
        """Constraint that ensures that the value of the critic is above a certain lower bound.

        The lower bound is determined by the `current_observation` and a certain constant.

        Args:
            critic_model_output: output of a critic

        Returns:
            Constraint violation
        """
        self.lb_constraint_violation = (
            self.lb_parameter * rg.sum(observation[:-1, :] ** 2)
            - critic_model_output[-1, :]
        )
        return self.lb_constraint_violation


class AgentCALF:
    def __init__(
        self,
        env: RgEnv,
        safe_policy,
        sampling_time,
        critic_model,
        running_objective,
        safe_only=False,
        observer: Observer = ObserverTrivial(),
        critic_batch_size=10,
        critic_td_n=2,
        critic_safe_decay_param=0.01,
        critic_is_dynamic_decay_rate=False,
        critic_lb_parameter=0.0,
        critic_ub_parameter=1.0,
        critic_regularization_param=0.0,
        critic_learning_norm_threshold=3,
        is_decay_upper_bounded=False,
        relax_probability_min=0.0,
        relax_probability_max=0.999,
        relax_probability_stabilize_global_step=10000,
        relax_probability_fraction_reverse=False,
        is_nominal_first=False,
        is_propagate_safe_weights=False,
        relax_prob_fading_factor: float = 0.9999999999,
    ):
        self.relax_factor = relax_prob_fading_factor
        self.is_nominal_first = is_nominal_first
        self.relax_probability_fraction_reverse = relax_probability_fraction_reverse
        self.safe_only = safe_only
        self.relax_probability_min = relax_probability_min
        self.relax_probability_max = relax_probability_max
        self.critic_learning_norm_threshold = critic_learning_norm_threshold
        self.observer = observer
        self.is_propagate_safe_weights = is_propagate_safe_weights

        self.critic = CriticCALF(
            system=env.simulator.system,
            model=critic_model,
            td_n=critic_td_n,
            is_same_critic=False,
            is_value_function=True,
            discount_factor=1.0,
            sampling_time=sampling_time,
            safe_decay_param=critic_safe_decay_param,
            is_dynamic_decay_rate=critic_is_dynamic_decay_rate,
            safe_policy=safe_policy,
            lb_parameter=critic_lb_parameter,
            ub_parameter=critic_ub_parameter,
            optimizer_config=CasadiOptimizerConfig(critic_batch_size),
            regularization_param=critic_regularization_param,
            is_decay_upper_bounded=is_decay_upper_bounded,
        )
        self.critic_weights_init = np.copy(self.critic.weights)
        prediction_horizon = 1
        self.running_objective = running_objective
        self.policy = RLPolicy(
            action_bounds=env.simulator.system._action_bounds,
            model=(
                ModelWeightContainer(
                    weights_init=np.zeros(
                        (prediction_horizon, env.simulator.system.dim_inputs),
                        dtype=np.float64,
                    ),
                    dim_output=env.simulator.system.dim_inputs,
                )
            ),
            system=env.simulator.system,
            running_objective=running_objective,
            prediction_horizon=prediction_horizon,
            algorithm="rpv",
            critic=self.critic,
            predictor=EulerPredictor(
                system=env.simulator.system, pred_step_size=sampling_time
            ),
            discount_factor=1.0,
            optimizer_config=CasadiOptimizerConfig(),
        )

        self.is_first_compute_action_call = True
        self.nominal_policy = safe_policy
        self.env = env
        self.relax_probability_stabilize_global_step = (
            relax_probability_stabilize_global_step
        )

    def get_safe_action(self, observation):
        return self.nominal_policy.get_action(observation)

    def issue_action(self, observation, is_safe=False):
        if is_safe:
            self.policy.restore_weights()
            self.critic.restore_weights()
            if self.relax_probability == 0:
                is_calf_action = False
            else:
                # Line 10 of Algorithm 1
                toss = np.random.choice(
                    2, 1, p=[1 - self.relax_probability, self.relax_probability]
                )
                is_calf_action = bool(toss)
            if is_calf_action:
                # Line 12 of Algorithm 1
                self.policy.update_action(observation)
            else:
                # Line 14 of Algorithm 1
                safe_action = self.nominal_policy.get_action(observation)
                self.policy.set_action(safe_action)
        else:
            self.policy.update_action(observation)

    def get_action(self, observation):
        if self.safe_only:
            action = self.env.simulator.system.apply_action_bounds(
                self.nominal_policy.get_action(observation)
            )
            return action

        estimated_state = self.observer.get_state_estimation(
            t=None, observation=observation, action=None
        )

        if (
            self.is_first_compute_action_call
        ):  # If called first time, initialize with safe action
            self.is_first_compute_action_call = False
            self.observation_last_good = observation
            self.issue_action(observation, is_safe=True)
        else:
            # Line 6 of Algorithm 1
            critic_weights = self.critic.optimize(
                self.data_buffer, is_update_and_cache_weights=False
            )  # get optimized critic weights
            critic_weights_accepted = (
                self.critic.opt_status == OptStatus.success
            )  # check constrs violations

            # Line 7 of Algorithm 1
            if critic_weights_accepted:
                self.critic.update_weights(critic_weights)  # substitute weights
                self.policy.optimize(self.data_buffer)  # get new action
                policy_weights_accepted = (
                    self.policy.opt_status == OptStatus.success
                )  # check constrs violation
                if policy_weights_accepted:
                    # Line 8 of Algorithm 1
                    self.observation_last_good = (
                        observation  # update observation dagger
                    )
                    self.issue_action(observation, is_safe=False)

                    # Line 8 of Algorithm 1
                    self.critic.cache_weights(critic_weights)  # update weights dagger
                else:
                    # Line 14 of Algorithm 1
                    self.issue_action(observation, is_safe=True)
            else:
                # Line 14 of Algorithm 1
                self.issue_action(observation, is_safe=True)

        action = self.env.simulator.system.apply_action_bounds(self.policy.action)
        self.data_buffer.push_to_end(
            observation=observation,
            estimated_state=estimated_state,
            action=action,
            running_objective=self.running_objective(observation, action),
            observation_last_good=self.observation_last_good,
            episode_id=1,
        )
        # Line 16 of Algorithm 1
        self.relax_probability *= self.relax_factor
        return action

    def reset(self, global_step):
        if global_step == 0 and self.is_nominal_first:
            self.safe_only = True
        elif global_step > 0 and self.is_nominal_first:
            self.safe_only = False

        fraction = np.clip(
            (global_step / (self.relax_probability_stabilize_global_step - 1)), 0, 1
        )

        if self.relax_probability_fraction_reverse:
            fraction = 1 - fraction

        # UPDATING THE relax_probability PARAMETER. ANY TRIGGER OR CRITERION CAN BE PLUGGED IN HERE.
        self.relax_probability = (
            self.relax_probability_min
            + (self.relax_probability_max - self.relax_probability_min) * fraction
        )

        self.is_first_compute_action_call = True
        self.data_buffer = DataBuffer()
        self.critic.update_weights(np.copy(self.critic_weights_init))
        if (
            self.is_propagate_safe_weights
        ):  # wether to propagate weights dageer to subsequent episode or not
            if global_step < self.relax_probability_stabilize_global_step:
                self.cached_critic_weights = np.copy(self.critic.model.cache.weights)
            if global_step >= self.relax_probability_stabilize_global_step:
                self.critic.cache_weights(self.cached_critic_weights)
        else:
            self.critic.cache_weights(np.copy(self.critic_weights_init))

from srccode.system import CartPolePG
from srccode.utils import rg


class InvertedPendulum(CartPolePG):
    _dim_observation = 4


class InvertedPendulumStabilizingPolicy:
    """An energy-based scenario for cartpole."""

    def __init__(
        self,
        scenario_gain: float = 10,
        upright_gain=None,
        swingup_gain=10,
        pid_loc_thr=0.35,
        pid_scale_thr=10.0,
        clip_bounds=(-1, 1),
    ):
        """Initialize an instance of ScenarioCartPoleEnergyBased.

        Args:
            action_bounds: upper and lower bounds for action yielded
                from policy
            sampling_time: time interval between two consecutive actions
            scenario_gain: scenario gain
            system: an instance of Cartpole system
        """
        super().__init__()

        self.scenario_gain = scenario_gain
        self.m_c, self.m_p, self.g, self.l = (
            InvertedPendulum._parameters["m_c"],
            InvertedPendulum._parameters["m_p"],
            InvertedPendulum._parameters["g"],
            InvertedPendulum._parameters["l"],
        )
        self.upright_gain = upright_gain
        self.swingup_gain = swingup_gain
        self.pid_loc_thr = pid_loc_thr
        self.pid_scale_thr = pid_scale_thr
        self.clip_bounds = clip_bounds

    def get_action(self, observation):
        observation = observation[0]

        # sin_theta, one_minus_cos_theta, x, theta_dot, x_dot = observation
        sin_theta, one_minus_cos_theta, theta_dot, x_dot = observation
        x = 0
        cos_theta = 1 - one_minus_cos_theta
        theta = rg.atan2(sin_theta, cos_theta)
        lbd = (1 - rg.tanh((theta - self.pid_loc_thr) * self.pid_scale_thr)) / 2
        low, high = self.clip_bounds
        x_clipped = rg.clip(x, low, high)
        x_dot_clipped = rg.clip(x_dot, low, high)

        self.upswing_gain = 3
        if cos_theta < 0:
            action_upswing = rg.sign(theta_dot) * self.upswing_gain
        else:
            action_upswing = rg.sign(sin_theta) * self.upswing_gain

        action_upright = self.upright_gain.T @ rg.array(
            [theta, x_clipped, theta_dot, x_dot_clipped]
        )

        self.action = (1 - lbd) * action_upswing + lbd * action_upright

        return self.action.reshape(1, -1)

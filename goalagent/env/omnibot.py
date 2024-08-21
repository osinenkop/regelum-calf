from srccode.system import KinematicPoint


class Omnibot(KinematicPoint):
    pass


class OmnibotStabilizingPolicy:
    def __init__(self, gain: float):
        self.gain = gain

    def get_action(self, observation):
        return -self.gain * observation


class OmnibotObserver:
    def __call__(self, observation):
        return observation

import numpy as np
from scipy.spatial.transform import Rotation
from simple_pid import PID


class SimpleFollower:
    def __init__(self, x_self_target, y_self_target):
        self.x_pid = PID(0.5, 0.1, 0.01, setpoint=0)
        self.y_pid = PID(0.5, 0.1, 0.01, setpoint=0)
        self.x_self_target = x_self_target
        self.y_self_target = y_self_target

    def follow(self, pos_self_target):
        x_error = self.x_self_target - pos_self_target[0]
        y_error = self.y_self_target - pos_self_target[1]

        x_control = self.x_pid(x_error)
        y_control = self.y_pid(y_error)

        return x_control, y_control
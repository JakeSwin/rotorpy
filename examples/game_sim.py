import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.world import World
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control

import math
from copy import deepcopy

class GameEnv:
    def __init__(self):
        self.initial_state = {'x': np.array([0,0,0]),
                              'v': np.zeros(3,),
                              'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
                              'w': np.zeros(3,),
                              'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                              'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}

        self.vehicle_state = self.initial_state

        # 100 times per second
        self.t_step = 1/100
        self.t = 0

        self.quadrotor = Multirotor(quad_params=quad_params, initial_state=self.initial_state, control_abstraction="cmd_vel", aero=True)

    def step(self, action):
        self.vehicle_state = self.quadrotor.step(self.vehicle_state, action, self.t_step)
        self.t += self.t_step

        self.print_status()

        return self.vehicle_state

    def print_status(self):
        print("Time: %3.2f \t Position: (%3.2f, %3.2f, %3.2f)" % (self.t, self.vehicle_state['x'][0], self.vehicle_state['x'][1], self.vehicle_state['x'][2]))


if __name__ == "__main__":
    game_env = GameEnv()
    
    state = game_env.initial_state
    controller  = SE3Control(quad_params)

    flat = {'x': [0, 0, 3], 
            'x_dot': [0, 0, 0], 
            'x_ddot': [0, 0, 0], 
            'x_dddot': [0, 0, 0],
            'yaw': 0, 
            'yaw_dot': 0, 
            'yaw_ddot': 0}

    while True: 
        action = controller.update(0, state, flat)
        state = game_env.step(action)
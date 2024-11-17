import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.world import World
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.minsnap import MinSnap

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

        self.max_vel = 3/math.sqrt(3)   # Selected so that at most the max speed is 3 m/s

        # 100 times per second
        self.t_step = 1/100
        self.t = 0

        self.quadrotor = Multirotor(quad_params=quad_params, initial_state=self.initial_state, control_abstraction="cmd_motor_speeds", aero=True)

        self.rotor_speed_max = self.quadrotor.rotor_speed_max
        self.rotor_speed_min = self.quadrotor.rotor_speed_min

    def step(self, action):
        # Extract the commanded motor speeds.
        cmd_motor_speeds = action['cmd_motor_speeds']

        # The environment expects the control inputs to all be within the range [-1,1]
        action = np.interp(cmd_motor_speeds, [self.rotor_speed_min, self.rotor_speed_max], [-1,1])
        control_dict = {}
        control_dict['cmd_motor_speeds'] = np.interp(action, [-1,1], [self.rotor_speed_min, self.rotor_speed_max])
        self.vehicle_state = self.quadrotor.step(self.vehicle_state, control_dict, self.t_step)
        self.t += self.t_step

        self.print_status()

        return self.vehicle_state

    def print_status(self):
        print("Time: %3.2f \t Position: (%3.2f, %3.2f, %3.2f)" % (self.t, self.vehicle_state['x'][0], self.vehicle_state['x'][1], self.vehicle_state['x'][2]))

class DroneFSM:
    def __init__(self):
        self.state = "hovering" 
        self.complete_traj_threshold = 0.3
        self.target_point = [0, 0, 0]
        self.target_yaw = 0
        self.T = 0

    def set_target(self, point, yaw):
        self.traj = MinSnap(
            points=np.array([self.target_point, point]),
            yaw_angles=np.array([self.target_yaw, yaw])
        )
        # Assume point is a numpy array
        self.target_point = point
        self.target_yaw = yaw
        self.state = "moving"
        self.T = 0

    def step(self):
        match self.state:
            case "hovering":
                next_state = {'x': self.target_point, 
                             'x_dot': [0, 0, 0], 
                             'x_ddot': [0, 0, 0], 
                             'x_dddot': [0, 0, 0],
                             'yaw': self.target_yaw, 
                             'yaw_dot': 0, 
                             'yaw_ddot': 0} 
            case "moving":
                next_state = self.traj.update(self.T)
                if np.linalg.norm(next_state["x"] - self.target_point) < self.complete_traj_threshold:
                    self.state = "hovering"
                    self.T = 0

        # Time Step amount same as multirotor
        self.T += 1/100
        return next_state

def get_path(min_coord=-30, max_coord=30, step=5, altitude=8):
    steps = np.arange(min_coord, max_coord, step)
    Ys, Xs = np.meshgrid(steps, steps)
    Ys[1::2, :] = Ys[1::2, ::-1]
    target_poses = np.column_stack((Xs.flatten(), Ys.flatten(), np.repeat(8, len(Ys.flatten()))))
    target_yaws = np.repeat(0, len(Ys.flatten()))
    # Rotation mask for yaws
    # yaw_mask = (np.arange(len(target_yaws)) // len(steps)) % 2 == 1
    # target_yaws[yaw_mask] = np.pi 
    target_poses = np.insert(target_poses, 0, [0, 0, altitude], axis=0)
    target_yaws = np.insert(target_yaws, 0, 0)
    return target_poses, target_yaws

if __name__ == "__main__":
    game_env = GameEnv()
    
    state = game_env.initial_state
    controller  = SE3Control(quad_params)
    fsm = DroneFSM()

    # target_poses = np.array([
    #     [0, 0, 8], 
    #     [-30, -30, 8]
    # ])
    # target_yaws = np.array([0, 0])
    target_poses, target_yaws = get_path()
    target_idx = 1

    fsm.set_target(target_poses[0], target_yaws[0])

    while True: 
        flat = fsm.step()
        action = controller.update(0, state, flat)
        state = game_env.step(action)
        if fsm.state == "hovering" and fsm.T > 2 and len(target_poses) > target_idx:
            fsm.set_target(target_poses[target_idx], target_yaws[target_idx])
            target_idx += 1
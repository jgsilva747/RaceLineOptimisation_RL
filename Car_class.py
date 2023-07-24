##########################################################
# Author: Joao Goncalo Dias Basto da Silva               #
#         Student No.: 5857732                           #
# Course: AE4-350 Bio-Inspired Intelligence and Learning #
# Date: July 2023                                        #
##########################################################


import gymnasium as gym
from gymnasium import spaces
import numpy as np

import Inputs as inp
import Model_utilities as util
from Model_utilities import coordinates, coordinates_in, coordinates_out


# Create a custom environment that adheres to the OpenAI Gym interface.
class CarEnvironment(gym.Env):
    def __init__(self):
        # TODO: explain function

        # Define initial mass
        self.mass_0 = inp.vehicle_mass + inp.fuel_mass # kg
        self.mass = self.mass_0

        # Define initial position
        self.initial_position = [ 0 , 0 ]
        self.current_position = self.initial_position

        # Define initial velocity 
        self.v_0 = inp.initial_velocity / 3.6 # m/s
        self.v_direction = coordinates[1] - coordinates[0]
        self.v_xy_0 = self.v_0 * self.v_direction / np.linalg.norm(self.v_direction)
        self.v_xy = self.v_xy_0

        # Initialise auxiliar propagation variables
        self.time = 0
        self.done = False
        self.circuit_index = 0

        # tangential acceleration
        self.a_0 = util.get_acceleration(np.linalg.norm([self.v_0[0], self.v_0[1]]), self.mass_0, 1 ) / 9.81 # in g
        self.a = self.a_0
        # normal acceleration
        self.n_0 = 0
        self.n = self.n_0

        # Initialise list of curvatures # TODO: test
        self.curvature_list_0 = util.get_future_curvatures(coordinates, self.current_position, self.circuit_index)
        self.curvature_list = self.curvature_list_0

        # Initialise LIDAR samples # TODO: everything
        self.lidar_samples_0 = util.get_lidar_samples(coordinates, coordinates_in, coordinates_out, self.current_position, self.circuit_index)
        self.lidar_samples = self.lidar_samples_0

        # Initialise track limits bool
        self.track_limits_0 = False # True if track limits are exceeded
        self.track_limits = self.track_limits_0

        # Define action space
        self.action_space = spaces.Box(
                # possible actions: brake/accelerate; turn left/right.
                # Both options range from -1 to 1
                np.array([-1, -1]).astype(np.float32),
                np.array([+1, +1]).astype(np.float32),
            )

        # Define observation space (24 state entries)
        self.observation_space = spaces.Box(
                # veloxity norm, t acc, n acc, delta heading, 10 future curvatures, 9 LIDAR measurements, track limits
                np.array([
                    inp.min_speed, # min velocity
                    -5.5, # min t acc, g
                    -6, # min n acc, g
                    -np.pi, # min heading deviation wrt centreline, rad
                    -np.pi, # min curvature 1, rad
                    -np.pi, # min curvature 2, rad
                    -np.pi, # min curvature 3, rad
                    -np.pi, # min curvature 4, rad
                    -np.pi, # min curvature 5, rad
                    -np.pi, # min curvature 6, rad
                    -np.pi, # min curvature 7, rad
                    -np.pi, # min curvature 8, rad
                    -np.pi, # min curvature 9, rad
                    -np.pi, # min curvature 10, rad
                    0, # min LIDAR measurement 1, m
                    0, # min LIDAR measurement 2, m
                    0, # min LIDAR measurement 3, m
                    0, # min LIDAR measurement 4, m
                    0, # min LIDAR measurement 5, m
                    0, # min LIDAR measurement 6, m
                    0, # min LIDAR measurement 7, m
                    0, # min LIDAR measurement 8, m
                    0, # min LIDAR measurement 9, m
                    0 # track limits bool, float representation
                    ]).astype(np.float32),
                np.array([
                    400/3.6, # max velocity, m/s
                    3, # max t acc, g
                    6, # max n acc, g
                    np.pi, # max heading deviation wrt centreline, rad
                    np.pi, # max curvature 1, rad
                    np.pi, # max curvature 2, rad
                    np.pi, # max curvature 3, rad
                    np.pi, # max curvature 4, rad
                    np.pi, # max curvature 5, rad
                    np.pi, # max curvature 6, rad
                    np.pi, # max curvature 7, rad
                    np.pi, # max curvature 8, rad
                    np.pi, # max curvature 9, rad
                    np.pi, # max curvature 10, rad
                    200, # max LIDAR measurement 1, m
                    200, # max LIDAR measurement 2, m
                    200, # max LIDAR measurement 3, m
                    200, # max LIDAR measurement 4, m
                    200, # max LIDAR measurement 5, m
                    200, # max LIDAR measurement 6, m
                    200, # max LIDAR measurement 7, m
                    200, # max LIDAR measurement 8, m
                    200, # max LIDAR measurement 9, m
                    1 # track limits bool, float representation
                    ]).astype(np.float32),
            )
        
        # Auxiliar variables used in reward function
        self.previous_distance_from_origin = 0
        self.travelled_distance = 0

        # Bool to define if plots are to be created
        self.plotting = inp.plotting

        # Define initial state
        self.state_0 = [self.v_0, self.a_0, self.n_0, self.curvature_list_0, self.lidar_samples_0, self.track_limits_0]
        self.state_0 = np.array( self.state_0 )

    def reset(self):
        # TODO: explain function

        # Reset seed
        super().reset(seed=inp.seed)

        # Reset initial state
        self.state = self.state_0
        # position
        self.current_position = self.initial_position
        # mass
        self.mass = self.mass_0

        # Reset auxiliar propagation varaibles
        self.done = False # termination flag
        self.circuit_index = 0 # index of car's current position in the coordinate array
        self.time = 0 # time, given in s

        # Reset auxiliar variables used in reward function
        self.previous_distance_from_origin = 0
        self.travelled_distance  = 0

        return self.state, self.current_position

    def step(self, action):
        # TODO: explain function

        # Propagate state # TODO
        new_state, new_position, new_mass = util.propagate_dynamics(self.state, self.current_position, action, inp.delta_t)

        # Update time
        self.time += inp.delta_t

        # Update track position index
        new_circuit_index, current_distance_to_origin = util.get_circuit_index(new_position, coordinates, self.circuit_index)

        # Check termination condition
        self.done, self.left_track, self.finish_line = util.assess_termination(new_position,
                                                                               coordinates_in,
                                                                               coordinates_out,
                                                                               new_circuit_index,
                                                                               self.time)

        # Update state
        self.state = np.array( new_state )
        # Update position
        self.current_position = new_position
        # Update mass
        self.mass = new_mass

        # Update circuit index if new checkpoint was reached (going back in the track does not count) 
        if new_circuit_index > self.circuit_index:
            self.circuit_index = new_circuit_index

        # Compute current reward
        reward, delta_distance = util.get_reward(self.left_track,
                                                 self.finish_line,
                                                 self.previous_distance_from_origin,
                                                 current_distance_to_origin)

  
        # Update "previous" distance to origin
        self.previous_distance_from_origin = current_distance_to_origin

        self.travelled_distance += delta_distance

        return self.state, reward, self.done, self.current_position, self.travelled_distance 
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

        # Define initial velocity
        self.v_norm = inp.initial_velocity / 3.6 # m/s
        self.v_direction = coordinates[1] - coordinates[0]
        self.v_0 = self.v_norm * self.v_direction / np.linalg.norm(self.v_direction)

        # Initialise auxiliar propagation variables
        self.time = 0
        self.done = False
        self.circuit_index = 0

        # Define initial state
        self.state_0 = [0, 0, self.v_0[0], self.v_0[1], self.mass_0]
        self.state_0 = np.array( self.state_0 )

        # longitudinal acceleration
        self.a = util.get_acceleration(np.linalg.norm([self.v_0[0], self.v_0[1]]), self.mass_0, 1 ) / 9.81
        # lateral acceleration
        self.n = 0

        # Define action space
        self.action_space = spaces.Box(
                # possible actions: brake/accelerate; turn left/right.
                # Both options range from -1 to 1
                np.array([-1, -1]).astype(np.float32),
                np.array([+1, +1]).astype(np.float32),
            )
        
        # Define observation space
        self.observation_space = spaces.Box(
                # x position, y position, x velocity, y velocity, mass
                np.array([
                    min(coordinates_out[:,0]),
                    min(coordinates_out[:,1]), 
                    inp.min_speed,
                    inp.min_speed, 
                    inp.vehicle_mass
                    ]).astype(np.float32),
                np.array([
                    max(coordinates_out[:,0]), 
                    max(coordinates_out[:,1]),
                    400/3.6, # m/s
                    400/3.6, # m/s
                    inp.vehicle_mass + inp.fuel_mass
                    ]).astype(np.float32),
            )
        
        # Auxiliar variables used in reward function
        self.previous_distance_from_origin = 0
        self.travelled_distance = 0

        self.plotting = inp.plotting

    def reset(self):
        # TODO: explain function

        # Reset seed
        super().reset(seed=inp.seed)
        # Reset initial state
        # position, velocity, mass, lateral acceleration
        self.state = self.state_0
        # longitudinal acceleration
        self.a = util.get_acceleration(np.linalg.norm([self.v_0[0], self.v_0[1]]), self.mass_0, 1 ) / 9.81
        # lateral acceleration
        self.n = 0

        # Reset auxiliar propagation varaibles
        self.done = False # termination flag
        self.circuit_index = 0 # index of car's current position in the coordinate array
        self.time = 0 # time, given in s

        # Reset auxiliar variables used in reward function
        self.previous_distance_from_origin = 0
        self.travelled_distance  = 0

        return self.state, [self.a, self.n]

    def step(self, action):
        # TODO: explain function

        # Propagate state
        new_state, self.a , self.n = util.propagate_dynamics(self.state, action, inp.delta_t)
        
        # Update time
        self.time += inp.delta_t

        # Update track position index
        new_circuit_index, current_distance_to_origin = util.get_circuit_index(new_state, coordinates, self.circuit_index)

        # Check termination condition
        self.done, self.left_track, self.finish_line = util.assess_termination(new_state,
                                                                               coordinates_in,
                                                                               coordinates_out,
                                                                               new_circuit_index,
                                                                               self.time)

        # Update state
        self.state = np.array( new_state )

        # Update circuit index if new checkpoint was reached (going back in the track does not count) 
        if new_circuit_index > self.circuit_index:
            self.circuit_index = new_circuit_index

        # Compute current reward
        reward, delta_distance = util.get_reward(self.left_track,
                                                 self.finish_line,
                                                 self.previous_distance_from_origin,
                                                 current_distance_to_origin,
                                                 self.a,
                                                 self.n)

        # Update "previous" distance to origin
        self.previous_distance_from_origin = current_distance_to_origin

        self.travelled_distance += delta_distance

        return self.state, reward, self.done, [self.a, self.n], self.travelled_distance 
##########################################################
# Author: Joao Goncalo Dias Basto da Silva               #
#         Student No.: 5857732                           #
# Course: AE4-350 Bio-Inspired Intelligence and Learning #
# Date: July 2023                                        #
##########################################################


import gymnasium as gym
from gym import spaces # from gymnasium import spaces --> using gym because of compatibility reasons w/ d3rlpy
import numpy as np
import logging
import os

import Inputs as inp
import Model_utilities as util
from Model_utilities import coordinates, MAX_INDEX


# Create a custom environment that adheres to the OpenAI Gym interface.
class CarEnvironment(gym.Env):
    def __init__(self, delta_t = inp.delta_t, integration_method = inp.integration_method,
                 log_file = 'run_log.txt', reward_function=['distance'],
                 sarsa=False):
        '''
        Initialise car class and required variables

        Parameters
        -----------
        delta_t: float
            Simulation time step [s]
        integration_method: string
            Chosen integrator. Default: Euler integrator
        log_file: string
            Name of log file. Ignored when log file is 'run_log.txt'. which is the default option
        reward_function: string array
            Array of strings containing the reward functions. Default: distance reward
        
        Returns
        ----------
        '''

        # Define integration settings
        self.delta_t = delta_t
        self.integration_method = integration_method

        # Define initial mass
        self.mass_0 = inp.vehicle_mass + inp.fuel_mass # kg
        self.mass = self.mass_0

        # Define initial position
        self.initial_position = np.array([ 0 , 0 ])
        self.current_position = self.initial_position

        # Define initial velocity 
        self.v_0 = inp.initial_velocity / 3.6 # m/s
        track_direction = (coordinates[1] - coordinates[0]) / np.linalg.norm(coordinates[1] - coordinates[0])
        self.v_xy_0 = self.v_0 * track_direction
        self.v_xy = self.v_xy_0

        # Initialise auxiliar propagation variables
        self.time = 0
        self.done = False
        self.circuit_index = 0

        # tangential acceleration
        self.a_0 = util.get_acceleration( self.v_0, self.mass_0, 1 ) / 9.81 # in g
        # normal acceleration
        self.n_0 = 0

        # Initialise delta_heading
        self.delta_heading_0 = 0
        self.delta_heaing = self.delta_heading_0

        # Initialise list of curvatures
        self.curvature_list_0, _ = util.get_future_curvatures(self.current_position, self.circuit_index)
        self.curvature_list = self.curvature_list_0

        # Initialise LIDAR samples
        self.lidar_samples_0 = util.get_lidar_samples(self.current_position, self.circuit_index)
        self.lidar_samples = self.lidar_samples_0

        # Initialise track limits bool
        self.track_limits_0 = float( False ) # True if track limits are exceeded
        self.track_limits = self.track_limits_0

        # Define action space
        self.action_space = spaces.Box(
                # possible actions: brake/accelerate; turn left/right.
                # Both options range from -1 to 1
                np.array([-1, -1]).astype(np.float32),
                np.array([+1, +1]).astype(np.float32),
            )

        # Define observation space (23 state entries)
        self.observation_space = spaces.Box(
                # veloxity norm, t acc, n acc, delta heading, 10 future curvatures, 9 LIDAR measurements
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
                    ]).astype(np.float32),
            )

        # Auxiliar variables used in reward function
        self.previous_distance_to_last_checkpoint = 0
        self.travelled_distance = 0
        self.previous_v = 0
        self.previous_delta = 0
        self.reward = 0
        self.action = [1,0]
        self.index = 0
        self.mean_v = 0
        self.v_list = []

        # Define initial state
        state_branch = [self.v_0, self.a_0, self.n_0, self.delta_heading_0]
        self.state_0 = np.concatenate( [state_branch , self.curvature_list_0 , self.lidar_samples_0 , [self.track_limits_0]] )
        
        # Assign reward function variable
        self.reward_function = reward_function

        # Check if requested reward function exists
        for func in reward_function:
            assert func in inp.reward_list, f"{func} not implemented. Please pick one from {inp.reward_list}"

        # Initialise log file
        self.logging_flag = False
        if inp.log and log_file != 'run_log.txt':
            
            self.logging_flag = True

            self.log_file = log_file
            print("Opening " + self.log_file)

            # Delete the log file if it already exists
            if os.path.exists(self.log_file):
                os.remove(self.log_file)
            
            self.info_logger = logging.getLogger(self.log_file)

            info_handler = logging.FileHandler(self.log_file)

            formatter = logging.Formatter('%(message)s')

            info_handler.setFormatter(formatter)

            self.info_logger.addHandler(info_handler)

            self.info_logger.setLevel(logging.INFO)
        
        # Define auxiliar variable that will simplify env. step when SARSA algo. is used
        self.sarsa = sarsa

        # Auxiliar action list
        self.action_list = []
            

    def reset(self, seed = inp.seed):
        '''
        Reset car class variables
        
        Parameters
        ---------
        seed: int
            Random generator seed

        Returns
        ---------
        '''

        # Log reward, number of steps of current episode and lap time
        if self.logging_flag:
            # Log info from last run
            self.info_logger.info(f'{float(self.reward)} {float(self.index)} {float(self.time)}')
            # logging.info(f"Travelled Distance: {self.travelled_distance}, Position: {self.current_position}")

        # Save action sequence (eq. to raceline) when reward > 3000
        if self.reward > 3000:
            np.save('action_lists/reward_' + str(self.reward) + '.npy', self.action_list)


        # Reset seed
        super().reset(seed=seed)

        # Reset action list        
        self.action_list = []

        # Reset initial state
        self.state = self.state_0
        # position
        self.current_position = self.initial_position
        # mass
        self.mass = self.mass_0
        # velocity
        self.v_xy = self.v_xy_0

        # Reset auxiliar propagation varaibles
        self.done = False # termination flag
        self.circuit_index = 0 # index of car's current position in the coordinate array
        self.time = 0 # time, given in s

        # Reset auxiliar variables used in reward function
        self.previous_distance_to_last_checkpoint = 0
        self.travelled_distance = 0
        self.previous_v = 0
        self.previous_delta = 0
        self.reward = 0
        self.action = [1,0]
        self.index = 0
        self.mean_v = 0
        self.v_list = []

        return self.state, self.current_position

    def step(self, action):
        '''
        Propagate state and auxiliar class variables by one time step

        Parameters
        ---------
        action: float array [1 x 2]
            Array of current time step's action (throttle + steering)
        '''

        # Update action list
        self.action_list.append(action)

        # Track direction
        if self.circuit_index >= MAX_INDEX:
            self.circuit_index -= 1
        track_direction = coordinates[self.circuit_index + 1] - coordinates[self.circuit_index]
        track_direction = track_direction / np.linalg.norm(track_direction)

        # Propagate state
        if not self.sarsa:
            new_state, new_position, new_mass, self.v_xy, centerline_pos = util.propagate_dynamics(self.state,
                                                                                               self.current_position,
                                                                                               self.mass,
                                                                                               self.v_xy,
                                                                                               track_direction,
                                                                                               action,
                                                                                               self.circuit_index,
                                                                                               delta_t = self.delta_t,
                                                                                               integration_method=self.integration_method)

        else:
            new_state, new_position, new_mass, self.v_xy, centerline_pos = util.propagate_dynamics_sarsa(self.state,
                                                                                               self.current_position,
                                                                                               self.mass,
                                                                                               self.v_xy,
                                                                                               track_direction,
                                                                                               action,
                                                                                               self.circuit_index,
                                                                                               delta_t = self.delta_t,
                                                                                               integration_method=self.integration_method)

        # Update mean velocity
        old_mean_v = self.mean_v
        self.v_list.append(new_state[0])
        #self.mean_v = np.mean(self.v_list)

        # Update time
        self.time += self.delta_t

        # Update track position index
        new_circuit_index, current_distance_to_last_checkpoint, _ = util.get_circuit_index(centerline_pos, self.circuit_index)

        # Check termination condition
        self.done, self.left_track, self.finish_line = util.assess_termination(new_position,
                                                                               new_circuit_index,
                                                                               self.time)

        # Store previous velocity norm
        self.previous_v = self.state[0]
        self.previous_delta = self.state[3]

        # Update state
        self.state = np.array( new_state )
        # Update position
        self.current_position = np.array( new_position )
        # Update mass
        self.mass = new_mass

        # Update circuit index if new checkpoint was reached (going back in the track does not count) 
        if new_circuit_index > self.circuit_index:
            self.circuit_index = new_circuit_index

        # Compute current reward
        reward, delta_distance = util.get_reward(self.left_track,
                                                 self.finish_line,
                                                 self.previous_distance_to_last_checkpoint,
                                                 current_distance_to_last_checkpoint,
                                                 self.reward_function,
                                                 self.state,
                                                 self.previous_v,
                                                 self.previous_delta,
                                                 action,
                                                 self.action,
                                                 self.index,
                                                 old_mean_v,
                                                 self.mean_v)


        # Update previous distance to last checkpoint
        self.previous_distance_to_last_checkpoint = current_distance_to_last_checkpoint

        self.travelled_distance += delta_distance

        self.reward += reward
        
        self.action = action

        # Update number of steps in current episode
        self.index += 1

        return self.state, reward, self.done, self.finish_line, self.current_position # , self.travelled_distance
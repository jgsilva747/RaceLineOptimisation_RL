# General imports
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':12})

# File imports
import Inputs as inp
import Model_utilities as util
from Model_utilities import coordinates, coordinates_in, coordinates_out


###########################################
# Run test episodes #######################
###########################################

# Create and initialise time variable
time = 0

# Create and initialise action array
'''
Possible actions:
[0] -1: brake; 0: no power; 1: accelerate
[1] -1: turn left; 0: go straight; 1: turn right
'''
action_array = [[1, 0]]

# Load time step from input file
delta_t = inp.delta_t

# Define initial mass
mass_0 = inp.vehicle_mass + inp.fuel_mass # kg

# Define initial velocity
v_norm = inp.initial_velocity / 3.6 # m/s
v_direction = coordinates[1] - coordinates[0]
v_0 = v_norm * v_direction / np.linalg.norm(v_direction)

##########################################################
# TODO: define following set of variables in inputs file #
# Number of episodes
n_episodes = 100
# Factor to update Q(s,a)
alpha = 0.5
# Randomness factor
eps = 0
# Importance of future rewards (discount factor)
gamma = 0.9
##########################################################

# Run episodes
for episode in range(n_episodes):

    # Define initial state
    # position, velocity, mass, lateral acceleration
    state = [[0, 0, v_0[0], v_0[1], mass_0, util.get_acceleration(np.linalg.norm([v_0[0], v_0[1]]), mass_0, 1 ) / 9.81, 0]]

    # Initialise auxiliar propagation varaibles
    complete = False # termination flag
    circuit_index = 0 # index of car's current position in the coordinate array

    # Run episode until termination
    while not complete:
        # Propagate state
        state.append(util.propagate_dynamics(state[-1], action_array[0], delta_t))

        # Update current time
        time += delta_t

        # Update circuit index
        circuit_index = util.get_circuit_index(state[-1], coordinates, circuit_index)

        # Check termination conditions
        complete = util.assess_termination(state[-1], coordinates_in, coordinates_out, circuit_index, time)

    # Convert state to numpy array
    state = np.array(state)

# plt.pause(0.01)

# Show all figures
plt.show()
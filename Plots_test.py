# General imports
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':12})
import torch

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

# Define initial mass (for a qualifying lap)
mass_0 = inp.vehicle_mass + inp.fuel_mass # kg

# Define initial velocity
v_norm = inp.initial_velocity / 3.6 # m/s
v_direction = coordinates[1] - coordinates[0]
v_0 = v_norm * v_direction / np.linalg.norm(v_direction)

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
    circuit_index, _ = util.get_circuit_index(state[-1], coordinates, circuit_index)

    # Check termination conditions
    complete, _, _ = util.assess_termination(state[-1], coordinates_in, coordinates_out, circuit_index, time)


# Convert state to numpy array
state = np.array(state)

# Plot trajectory with velocity colorbar
fig,ax = plt.subplots(figsize = (8,6)) 
ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')
points = ax.scatter(state[:,0], state[:,1], c=( 3.6 * np.linalg.norm(state[:,2:4], axis = 1) ), s=10, cmap="plasma")
cbar = fig.colorbar(points)
cbar.set_label('Velocity [km/h]')
fig.tight_layout()


# Plot velocity vs time and accelerations vs time
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,12))
# Parse into own variables
ax1 = ax[0]
ax2 = ax[1]
# Plots
ax1.plot(np.arange(0, time + 0.1*delta_t, delta_t) , np.linalg.norm(([state[:,2], state[:,3]]), axis = 0) * 3.6)
ax2.plot(np.arange(0, time + 0.1*delta_t, delta_t) , state[:,5], color = 'tab:blue', label = 'Longitudinal Acceleration')
ax2.plot(np.arange(0, time + 0.1*delta_t, delta_t) , state[:,6], color = 'tab:orange', label = 'Lateral Acceleration')
# Labels
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Velocity [km/h]')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Acceleration [g]')
# Titles
ax1.set_title("Velocity vs Time")
ax2.set_title("Acceleration vs Time")
# Legend
ax2.legend(loc = 'best')
# Adjust figure
fig.subplots_adjust(hspace=0.28, wspace=0.2, top=0.94, bottom=0.1)

# Plot g-g diagram
fig,ax = plt.subplots(figsize = (8,6)) 
ax.set_title('g-g Acceleration Diagram')
ax.scatter(state[:,6], -state[:,5], color = 'k')
ax.set_xlabel('Lateral Acceleration [g]')
ax.set_ylabel('Longitudinal Acceleration [g]')
fig.tight_layout()

# Plot multiple (random) episodes in the circuit (no RL implemented)
fig, ax = plt.subplots(figsize=( 8 , 6))
ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')
# Run different episodes
for i in np.arange(-1,1.1,0.1):
    # Define action
    action_array = [[1, i]]
    # Initialise time
    time = 0
    # Initialise state
    state = [[0, 0, v_0[0], v_0[1], mass_0, util.get_acceleration(np.linalg.norm([v_0[0], v_0[1]]), mass_0, 1 ) / 9.81, 0]]
    # Initialise auxiliar propagation variables
    complete = False
    circuit_index = 0
    # Run current episode
    while not complete:
        # Propagate state
        state.append(util.propagate_dynamics(state[-1], action_array[0], delta_t))

        # Update time
        time += delta_t

        # Update track position index
        circuit_index, _ = util.get_circuit_index(state[-1], coordinates, circuit_index)

        # Check termination condition
        complete, _ , _ = util.assess_termination(state[-1], coordinates_in, coordinates_out, circuit_index, time)

    # Convert state to numpy array
    state = np.array(state)

    # Plot current episode (trajectory)
    ax.plot(state[:,0], state[:,1], color = 'b')

# Apply tight layout to figure
plt.tight_layout()

# Show all figures
plt.show()
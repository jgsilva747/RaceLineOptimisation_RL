# General imports
import numpy as np
import pandas as pd
import os
from time import time as clock
import math
import matplotlib.pyplot as plt
import matplotlib
import Inputs as inp
matplotlib.rcParams.update({'font.size':12})


def get_acceleration(speed, mass, action):
    '''
    Computes acceleration of car based on current state and defined action

    Parameters
    ----------
    speed: float
        Current velocity of car, given in m/s
    mass: float
        Current mass of car, given in kg
    action: float
        First entry of the action array, ranging from -1 to 1, which defines the throttle of the car
    
    Returns
    -------
    acc: float
        norm of acceleration experienced by the car, given in m/s^2
    '''

    if action[0] <= 0:
        acc = action * 5.5 * 9.81
    else:
        drag_area = 1.5
        rho = 1.225
        c_d = 0.7
        # max_power = 600e3 # W

        power = lambda speed: 1.278552439 * speed if speed < 100/3.6 else ( 1.278552439 * 100/3.6 + 0.797314492 * (speed - 100/3.6) if speed < 200/3.6 else 57.66 - (speed * 3.6 - 200)/500 )

        #speed**3 * 3e-4 - 0.0588 * speed**2 + 3.1897 * speed # max_power * ( 1 - math.exp(- speed / 2e2) )

        # if speed < 10:
        #     momentum = 1000
        # else:
        #     momentum = mass * speed

        # acc = power(speed) / (momentum) - ( 0.5 * rho * drag_area * c_d * v**2 ) / mass
        acc = power(speed) - ( 0.5 * rho * drag_area * c_d * speed**2 ) / mass
        # acc = 14/45**2 * (90 - v)**2
    
    return acc


def propagate_dynamics(state, action, delta_t):

    speed = state[2:4]
    mass = state[4]

    mass_flow_rate = 0.028 # kg/s

    acceleration = get_acceleration(np.linalg.norm(speed), mass, action)

    angle = - np.deg2rad(15) * action[1]

    mass = mass - mass_flow_rate * delta_t
    if mass <= 0:
        mass = 0
        acceleration = 0
    
    speed_norm = np.linalg.norm(speed)
    if speed_norm < 0.1:
        speed_norm = 0.1

    acc_x_tangent = np.cos(angle) * acceleration * speed[0] / speed_norm
    acc_y_tangent = np.cos(angle) * acceleration * speed[1] / speed_norm

    acc_x_normal = - np.sin(angle) * acceleration * speed[1] / speed_norm
    acc_y_normal = np.sin(angle) * acceleration * speed[0] / speed_norm

    acc_x = acc_x_tangent + acc_x_normal 
    acc_y = acc_y_tangent + acc_y_normal

    v = [speed[0] + acc_x * delta_t, speed[1] + acc_y * delta_t]
    position = [state[0] + speed[0] * delta_t + 0.5 * acc_x * delta_t**2, state[1] + speed[1] * delta_t + 0.5 * acc_y * delta_t**2]

    n = np.linalg.norm( [ acc_x_normal, acc_y_normal ] ) / 9.81

    return [position[0], position[1], v[0], v[1], mass, n]
    

def assess_termination(state, coordinates_in, coordinates_out, index, time):

    if index == len(coordinates_out) - 1: # successfully reached finish line
        return True
    
    elif time > 3 * 60: # maximum of 3 (simulated) minutes
        return True
    
    else: # exiting the race track
        angle = np.rad2deg(math.atan2(coordinates_out[index+1,1] - coordinates_out[index,1], coordinates_out[index+1,0] - coordinates_out[index,0]))
        
        if angle < 45:
            x = state[0]
            y = state[1]

            x_2 = coordinates_in[index + 1,0]
            x_1 = coordinates_in[index,0]
            y_2 = coordinates_in[index + 1,1]
            y_1 = coordinates_in[index,1]
            # interpolate
            y_in = y_1 + ( x - x_1 ) * ( y_2 - y_1 ) / ( x_2 - x_1 )

            x_2 = coordinates_out[index + 1,0]
            x_1 = coordinates_out[index,0]
            y_2 = coordinates_out[index + 1,1]
            y_1 = coordinates_out[index,1]
            # interpolate
            y_out = y_1 + ( x - x_1 ) * ( y_2 - y_1 ) / ( x_2 - x_1 )

            if y_out > y_in:
                if y > y_out or y < y_in:
                    return True
            else:
                if y < y_out or y > y_in:
                    return True
                
        else:
            x = state[0]
            y = state[1]

            x_2 = coordinates_in[index + 1,0]
            x_1 = coordinates_in[index,0]
            y_2 = coordinates_in[index + 1,1]
            y_1 = coordinates_in[index,1]
            # interpolate
            x_in = x_1 + ( y - y_1 ) * ( x_2 - x_1 ) / ( y_2 - y_1 )

            x_2 = coordinates_out[index + 1,0]
            x_1 = coordinates_out[index,0]
            y_2 = coordinates_out[index + 1,1]
            y_1 = coordinates_out[index,1]
            # interpolate
            x_out = x_1 + ( y - y_1 ) * ( x_2 - x_1 ) / ( y_2 - y_1 )

            if x_out > x_in:
                if x > x_out or x < x_in:
                    return True
            else:
                if x < x_out or x > x_in:
                    return True

    return False

def get_propagation_index(state, coordinates, propagation_index, circuit_factor):

    if np.absolute( ( coordinates[propagation_index + 1, 0] - coordinates[0,0]) * circuit_factor - state[0]) < 6 and np.absolute( (coordinates[propagation_index + 1, 1] - coordinates[0,1]) * circuit_factor - state[1]) < 6:
        propagation_index += 1

    return propagation_index

# Circuit Definition ######################################################################################################################

# Load chosen circuit from input file
chosen_circuit = inp.chosen_circuit

circuit_dir = os.path.dirname((__file__)) + "/circuits"
circuit_list = str(next( os.walk( circuit_dir ) )[2])
circuit_list = circuit_list.replace('.txt', '')

if chosen_circuit in circuit_list:
    coordinates = np.loadtxt(circuit_dir + '/' + chosen_circuit + '.txt')
else:
    print("Circuit not available. Please pick a circuit from the following list: " + circuit_list.strip(']['))
    exit(0)

# Create variables for inner and outter limits of circuit
coordinates_in = np.zeros((len(coordinates),2))
coordinates_out = np.zeros((len(coordinates),2))

# Define origin of circuit
start = coordinates[0]

# perpendicular direction scale factor
direction_factor = 5e-5

# real life circuit scale factor
circuit_factor = 1e5

# Cycle to compute inner and outter limits of circuit
for i in range(len(coordinates)):

    # Make sure that minimum index is at least 0
    min_val = i - 1
    if min_val < 0:
        min_val = 0

    # Make sure that maximum index does not exceed array length
    max_val = i + 1
    if max_val >= len(coordinates):
        max_val = len(coordinates) - 1

    # Compute track's perpendicular direction at each point
    direction = np.array([- coordinates[max_val,1] + coordinates[min_val,1], coordinates[max_val,0] - coordinates[min_val,0]])
    
    # Normalise and scale perpendicular direction
    direction = direction_factor * direction / np.linalg.norm(direction)

    # Compute inner and outter limits of circuit
    coordinates_in[i] = (coordinates[i] - direction  - start) * circuit_factor
    coordinates_out[i] = (coordinates[i] + direction  - start) * circuit_factor


# Create variables for inner and outter limits of circuit
coordinates_in = np.zeros((len(coordinates),2))
coordinates_out = np.zeros((len(coordinates),2))

# Define origin of circuit
start = coordinates[0]

# perpendicular direction scale factor
direction_factor = 5e-5

# real life circuit scale factor
circuit_factor = 1e5

# Cycle to compute inner and outter limits of circuit
for i in range(len(coordinates)):

    # Make sure that minimum index is at least 0
    min_val = i - 1
    if min_val < 0:
        min_val = 0

    # Make sure that maximum index does not exceed array length
    max_val = i + 1
    if max_val >= len(coordinates):
        max_val = len(coordinates) - 1

    # Compute track's perpendicular direction at each point
    direction = np.array([- coordinates[max_val,1] + coordinates[min_val,1], coordinates[max_val,0] - coordinates[min_val,0]])
    
    # Normalise and scale perpendicular direction
    direction = direction_factor * direction / np.linalg.norm(direction)

    # Compute inner and outter limits of circuit
    coordinates_in[i] = (coordinates[i] - direction  - start) * circuit_factor
    coordinates_out[i] = (coordinates[i] + direction  - start) * circuit_factor

###########################################################################################################################################

# Create and initialise time variable
time = 0

# Create and initialise action array
# Possible actions:
# [0] -1: brake; 0: no power; 1: accelerate
# [1] -1: turn left; 0: go straight; 1: turn right
action_array = [[1, 0]]

# Load time step from input file
delta_t = inp.delta_t

# Define initial mass (for a qualifying lap)
mass_0 = 100 # kg 

# Define initial velocity
v_norm = 0.1 # 200 / 3.6 # m/s
v_direction = coordinates[1] - coordinates[0]
v_0 = v_norm * v_direction / np.linalg.norm(v_direction)

# Define initial state
# position, velocity, mass, lateral acceleration
state = [[0, 0, v_0[0], v_0[1], mass_0, 0]]

complete = False
propagation_index = 0

while not complete:
    state.append(propagate_dynamics(state[-1], action_array[0], delta_t))

    time += delta_t

    propagation_index = get_propagation_index(state[-1], coordinates, propagation_index, circuit_factor)

    complete = assess_termination(state[-1], coordinates_in, coordinates_out, propagation_index, time)

    speed = state[-1]

    # print("Velocity: " + str(3.6 * np.linalg.norm(speed[2:4])) + " km/h; time: " + str(time))


state = np.array(state)
fig,ax = plt.subplots(figsize = (8,6)) 
ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')
points = ax.scatter(state[:,0], state[:,1], c=( 3.6 * np.linalg.norm(state[:,2:4], axis = 1) ), s=10, cmap="plasma")
cbar = fig.colorbar(points)
cbar.set_label('Velocity [km/h]')
fig.tight_layout()

# Plot circuit
fig, ax = plt.subplots(figsize=( 8 , 6))
ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')

for i in np.arange(-1,1.1,0.1):
    action_array = [[1, i]]

    time = 0

    state = [[0, 0, v_0[0], v_0[1], mass_0, 0]]
    complete = False
    propagation_index = 0

    while not complete:
        state.append(propagate_dynamics(state[-1], action_array[0], delta_t))

        time += delta_t

        propagation_index = get_propagation_index(state[-1], coordinates, propagation_index, circuit_factor)

        complete = assess_termination(state[-1], coordinates_in, coordinates_out, propagation_index, time)


    state = np.array(state)

    ax.plot(state[:,0], state[:,1], color = 'b')

plt.tight_layout()

plt.show()
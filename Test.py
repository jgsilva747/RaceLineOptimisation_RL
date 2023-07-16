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


def get_acceleration(speed, mass, throttle):
    '''
    Computes longitudinal acceleration of car based on current state and throttle

    Parameters
    ----------
    speed: float
        Current velocity of car, given in m/s
    mass: float
        Current mass of car, given in kg
    throttle: float
        First entry of the action array, ranging from -1 to 1, which defines the throttle of the car
    
    Returns
    -------
    acc: float
        norm of acceleration experienced by the car, given in m/s^2
    '''

    # Compute acceleration when braking
    # max acc = -5.5 g
    if throttle <= 0:
        acc = throttle * 5.5 * 9.81
    # Compute acceleration when accelerating
    # based on empirical data
    else:
        # lambda function to compute power based on speed
        # power = lambda speed: 1.278552439 * speed if speed < 100/3.6 else ( 1.278552439 * 100/3.6 + 0.797314492 * (speed - 100/3.6) if speed < 200/3.6 else 57.66 - (speed * 3.6 - 200)/500 )


        # compute acceleration
        # acc = power(speed) - ( 0.5 * inp.rho * inp.drag_area * inp.c_d * speed**2 ) / mass
        acc = inp.x1 + inp.x2 * speed**2 if speed < inp.x3/3.6 else inp.x1 + inp.x2 * ( inp.x3/3.6)**2 - inp.x4 * ( speed - inp.x3/3.6 )

    return acc


def propagate_dynamics(state, action, delta_t = inp.delta_t):
    '''
    Propagates car dynamics based on current state and defined action

    Parameters
    ----------
    state: float array [1 x 7]
        State array, containing: x and y position [m], x and y velocity [m/s], mass [kg], longitudinal and lateral acceleration [g], respectively
    action: float array [1 x 2]
        Action array, containing: throttle (ranging from -1 to 1) [-] and steering (ranging from -1 to 1) [-], respectively
    delta_t: float
        Simulation time step [s]
    
    Returns
    -------
    state: float array [1 x 7]
        State array after one propagation step, with the same content as the input state array.
    '''

    # Obtain speed from state
    speed = state[2:4] # m/s
    # Obtain mass from state
    mass = state[4] # kg

    # Define mass flow rate
    mass_flow_rate = 0.028 # kg/s

    # Compute norm of current velocity
    speed_norm = np.linalg.norm(speed) # m/s

    if speed_norm < inp.min_speed:
        speed_norm = inp.min_speed

    # Compute longitudinal acceleration based on current velocity, mass and throttle
    acceleration = get_acceleration(speed_norm, mass, action[0])

    # Convert steering input into an angle, with a maximum angle of 15 deg
    angle = - np.deg2rad(15) * action[1] # rad

    # Propagate mass using Euler integrator
    mass = mass - mass_flow_rate * delta_t * action[0] if action[0] >= 0 else mass

    # Define threshold of minimum mass (0 kg) and update acceleration (0 m/s^2)
    if mass <= inp.vehicle_mass:
        mass = inp.vehicle_mass
        acceleration = 0

    # Compute tangent accelearation
    acc_x_tangent = np.cos(angle) * acceleration * speed[0] / speed_norm
    acc_y_tangent = np.cos(angle) * acceleration * speed[1] / speed_norm

    # Compute normal acceleration
    acc_x_normal = - np.sin(angle) * ( acceleration + speed_norm**2 / inp.wheelbase_length ) * speed[1] / speed_norm
    acc_y_normal = np.sin(angle) * ( acceleration + speed_norm**2 / inp.wheelbase_length ) * speed[0] / speed_norm

    # Add normal and tangent accelerations in cartesian coordinates
    acc_x = acc_x_tangent + acc_x_normal 
    acc_y = acc_y_tangent + acc_y_normal

    # Propagate speed using Euler integrator
    v = [speed[0] + acc_x * delta_t, speed[1] + acc_y * delta_t]

    # Propagate position using Euler integrator
    position = [state[0] + speed[0] * delta_t + 0.5 * acc_x * delta_t**2, state[1] + speed[1] * delta_t + 0.5 * acc_y * delta_t**2]

    # Compute longitudinal acceleration in g's
    a = np.linalg.norm( [ acc_x_tangent, acc_y_tangent ] ) / 9.81 # g

    # Compute normal acceleration in g's
    n = np.linalg.norm( [ acc_x_normal, acc_y_normal ] ) / 9.81 # g

    # Return updated state
    return [position[0], position[1], v[0], v[1], mass, a, n]
    

def assess_termination(state, coordinates_in, coordinates_out, index, time):
    '''
    Assess if simulation should end ...

    Parameters
    ----------

    
    Returns
    -------

    '''

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
    '''
    ...

    Parameters
    ----------

    
    Returns
    -------

    '''

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
direction_factor = inp.direction_factor

# real life circuit scale factor
circuit_factor = inp.circuit_factor

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
v_norm = 0.1 # 200 / 3.6 # m/s
v_direction = coordinates[1] - coordinates[0]
v_0 = v_norm * v_direction / np.linalg.norm(v_direction)

# Define initial state
# position, velocity, mass, lateral acceleration
state = [[0, 0, v_0[0], v_0[1], mass_0, get_acceleration(np.linalg.norm([v_0[0], v_0[1]]), mass_0, 1 ) / 9.81, 0]]

complete = False
propagation_index = 0

while not complete:
    state.append(propagate_dynamics(state[-1], action_array[0], delta_t))

    time += delta_t

    propagation_index = get_propagation_index(state[-1], coordinates, propagation_index, circuit_factor)

    complete = assess_termination(state[-1], coordinates_in, coordinates_out, propagation_index, time)

state = np.array(state)

fig,ax = plt.subplots(figsize = (8,6)) 
ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')
points = ax.scatter(state[:,0], state[:,1], c=( 3.6 * np.linalg.norm(state[:,2:4], axis = 1) ), s=10, cmap="plasma")
cbar = fig.colorbar(points)
cbar.set_label('Velocity [km/h]')
fig.tight_layout()


fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(6,12))
# Parse into own variables
ax1 = ax[0]
ax2 = ax[1]

ax1.plot(np.arange(0, time + 0.1*delta_t, delta_t) , np.linalg.norm(([state[:,2], state[:,3]]), axis = 0) * 3.6)
ax2.plot(np.arange(0, time + 0.1*delta_t, delta_t) , state[:,5], color = 'tab:blue', label = 'Longitudinal Acceleration')
ax2.plot(np.arange(0, time + 0.1*delta_t, delta_t) , state[:,6], color = 'tab:orange', label = 'Lateral Acceleration')

ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Velocity [km/h]')
ax1.set_title("Velocity vs Time")

ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Acceleration [g]')
ax2.set_title("Acceleration vs Time")

ax2.legend(loc = 'best')

fig.subplots_adjust(hspace=0.28, wspace=0.2, top=0.94, bottom=0.1)


fig,ax = plt.subplots(figsize = (8,6)) 
ax.set_title('g-g Acceleration Diagram')
ax.scatter(state[:,6], -state[:,5], color = 'k')
ax.set_xlabel('Lateral Acceleration [g]')
ax.set_ylabel('Longitudinal Acceleration [g]')
fig.tight_layout()


# Plot circuit
fig, ax = plt.subplots(figsize=( 8 , 6))
ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')

for i in np.arange(-1,1.1,0.1):
    action_array = [[1, i]]

    time = 0

    state = [[0, 0, v_0[0], v_0[1], mass_0, get_acceleration(np.linalg.norm([v_0[0], v_0[1]]), mass_0, 1 ) / 9.81, 0]]
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
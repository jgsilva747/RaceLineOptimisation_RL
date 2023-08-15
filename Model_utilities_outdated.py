##########################################################
# Author: Joao Goncalo Dias Basto da Silva               #
#         Student No.: 5857732                           #
# Course: AE4-350 Bio-Inspired Intelligence and Learning #
# Date: July 2023                                        #
##########################################################


# General imports
import numpy as np
import math
import os

# File imports
import Inputs as inp


def convert_action(action):
    '''
    Function that converts the discrete input (from 0 to 9)
    to the two arrays that can be read by the defined car class

    Parameters
    ----------
    action: float array [1 x 2]
        discrete action obtained from discrete SARSA policy.
        For the discrete model, 2 x 5 actions are possible
    
    Returns
    ----------
    action_array: float array [1 x 2]
        array with 2 actions: the first entry corresponds to the throttle/break,
        and the second entry corresponds to the wheel (left/right)
    '''

    # Convert action to throttle, if action is within the range of 0 to 4
    throttle = float( action[0] < 5 ) * float( action[0] - 2 ) / 2.0
    # Convert action to wheel, if action is within the range of 0 to 4
    wheel = float( action[1] < 5 ) * float( action[1] - 2 ) / 2.0

    return [throttle, wheel]

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
    if throttle <= 0:
        acc = 9.81 * ( 5e-4 * speed**2 - 6.7e-3 * speed + 1.9 ) if speed > inp.min_speed else 0 # inp.braking_acceleration * 9.81 * speed / (360/3.6)
    # Compute acceleration when accelerating
    else:
        # profile based on empirical data
        force = inp.x1 + inp.x2 * speed**2 if speed < inp.x3/3.6 else inp.x1 + inp.x2 * ( inp.x3/3.6)**2 - inp.x4 * ( speed - inp.x3/3.6 )
        acc = force / mass

    return throttle * acc


def propagate_dynamics(state, action, delta_t = inp.delta_t):
    '''
    Propagates car dynamics based on current state and defined action

    Parameters
    ----------
    state: float array [1 x 5]
        State array, containing: x and y position [m], x and y velocity [m/s], mass [kg], respectively
    action: float array [1 x 2]
        Action array, containing: throttle (ranging from -1 to 1) [-] and steering (ranging from -1 to 1) [-], respectively
    delta_t: float
        Simulation time step [s]
    
    Returns
    -------
    state: float array [1 x 5]
        State array after one propagation step, with the same content as the input state array.
    '''

    # Obtain speed from state
    speed = state[2:4] # m/s

    # Define mass flow rate
    mass_flow_rate = 0.028 # kg/s

    # Compute norm of current velocity
    speed_norm = np.linalg.norm(speed) # m/s

    if speed_norm < inp.min_speed:
        speed_norm = inp.min_speed

    # Obtain mass from state
    mass = state[4] # kg
    # Compute longitudinal acceleration based on current velocity, mass and throttle
    acceleration = get_acceleration(speed_norm, mass, action[0])

    # Propagate mass using Euler integrator
    mass = mass - mass_flow_rate * delta_t * action[0] if action[0] >= 0 else mass

    # Convert steering input into an angle, with a maximum angle of 15 deg
    angle = - np.deg2rad(15) * action[1] # rad

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

    # Return updated state and accelerations (not part of the state)
    return [position[0], position[1], v[0], v[1], mass], a, n
    

def assess_termination(state, coordinates_in, coordinates_out, index, time):
    '''
    Assess if simulation should end.
    This is done by checking if the car is within track limits, if the car
    has reached the finish line (= starting position) or if the maximum
    simulation time has been exceeded.

    Parameters
    ----------
    state: float array [1 x 5]
        State array, containing: x and y position [m], x and y velocity [m/s], mass [kg], respectively
    coordinates_in: float array [2 x number_of_track_points]
        array of xy coordinates of the inner limits of the circuit
    coordinates_out: float array [2 x number_of_track_points]
        array of xy coordinates of the outter limits of the circuit
    index: int
        index of current track position w.r.t. the coordinates array 
        (used to assess if car has reached the finish line or if car has exceeded track limits)
    time: float
        simulation time [s] used to check if maximum simulation time is exceeded

    Returns
    -------
    3 True or False bools
        1st - indicates wheather simulation should end or not. True indicates that the simulation should end.
        2nd - indicates if car has exceeded track limits. True indicates that the car has left the track.
        3rd- indicates if car successfully completed a lap. True indicates that it has reached  the finish line.
    '''

    # Assess if car has reached finish line
    if index == len(coordinates_out) - 1: # successfully reached finish line
        return True, False, True # end simulation
    
    # Assess if simulation time was exceeded
    elif time > inp.max_min * 60:
        return True, False, False # end simulation
    
    # Assess if car left the race track
    else:
        # Compute angle/direction of current portion of the track in the xy frame (in degrees)
        angle = np.rad2deg(math.atan2(coordinates_out[index+1,1] - coordinates_out[index,1], coordinates_out[index+1,0] - coordinates_out[index,0]))
        
        # Extract x and y positions from state array
        x = state[0]
        y = state[1]

        # Define track's current inner portion's xy coordinates (as a rectangle)
        x_2_in = coordinates_in[index + 1,0]
        x_1_in = coordinates_in[index,0]
        y_2_in = coordinates_in[index + 1,1]
        y_1_in = coordinates_in[index,1]

        # Define track's current outter portion's xy coordinates (as a rectangle)
        x_2_out = coordinates_out[index + 1,0]
        x_1_out = coordinates_out[index,0]
        y_2_out = coordinates_out[index + 1,1]
        y_1_out = coordinates_out[index,1]

        # If current portion is (mostly) horizontal, check if y (vertical) position was exceeded
        if -45 < angle < 45:
            # interpolate inner y coordinate for current x position
            y_in = y_1_in + ( x - x_1_in ) * ( y_2_in - y_1_in ) / ( x_2_in - x_1_in )

            # interpolate outter y coordinate for current x position
            y_out = y_1_out + ( x - x_1_out ) * ( y_2_out - y_1_out ) / ( x_2_out - x_1_out )

            # Check if y position exceeded track limits
            if y_out > y_in:
                if y > y_out or y < y_in:
                    return True, True, False # end simulation
            else:
                if y < y_out or y > y_in:
                    return True, True, False # end simulation

        # If current portion is (mostly) vertical, check if x (horizontal) position was exceeded
        else:
            # interpolate inner x coordinate for current y position
            x_in = x_1_in + ( y - y_1_in ) * ( x_2_in - x_1_in ) / ( y_2_in - y_1_in )

            # interpolate outter x coordinate for current y position
            x_out = x_1_out + ( y - y_1_out ) * ( x_2_out - x_1_out ) / ( y_2_out - y_1_out )

            # Check if x position exceeded track limits
            if x_out > x_in:
                if x > x_out or x < x_in:
                    return True, True, False # end simulation
            else:
                if x < x_out or x > x_in:
                    return True, True, False # end simulation

    # If none of the above cases was activated, then the simulation should not be terminated
    return False, False, False

def get_circuit_index(state, coordinates, circuit_index):
    '''
    Obtain index of current track coordinate w.r.t. the coordinate array

    Parameters
    ----------
    state: float array [1 x 5]
        State array, containing: x and y position [m], x and y velocity [m/s], mass [kg], respectively
    coordinates: float array [2 x number_of_track_points]
        array of xy coordinates of the curcuit's centre line
    curcuit_index: int
        index of the last propagated position withing the coordinate array
    
    Returns
    -------
    curcuit_index: int
        index of current position w.r.t. the coordinate array
    '''
    
    # Get position of last checkpoint
    last_checkpoint = ( coordinates[ circuit_index ] - coordinates[0] ) * inp.circuit_factor

    # Compute distance between car and last checkpoint
    current_distance_to_origin = np.linalg.norm(state[:2] - last_checkpoint)

    # Compute horizontal distance between end of current track portion and starting position
    x_distance = ( coordinates[circuit_index + 1, 0] - coordinates[0,0] ) * inp.circuit_factor
    # Compute vertical distance between end of current track portion and starting position
    y_distance = ( coordinates[circuit_index + 1, 1] - coordinates[0,1] ) * inp.circuit_factor

    # Compute horizontal distance from car to checkpoint
    x_dif = np.absolute( x_distance - state[0])
    # Compute vertical distance from car to checkpoint
    y_dif = np.absolute( y_distance - state[1])

    # Compute total distance from car to checkpoint
    distance = np.linalg.norm([x_dif, y_dif]) # m

    # Assess if current car position is within a certain distance (tolerance defined in the inputs file)
    # of next track coordinates
    # if x_dif < inp.index_pos_tolerance and y_dif < inp.index_pos_tolerance:
    if distance < inp.index_pos_tolerance:
        # Increase index to indicate that car is in the next portion of the track
        circuit_index += 1

    # Return index (which may or may not have been updated) and current distance to origin
    return circuit_index, current_distance_to_origin


def get_reward(left_track, finish_line, previous_distance, current_distance, a, n):
    '''
    Compute the reward given the current status of the car w.r.t. the circuit

    Parameters
    ----------
    left_track: bool
        Boolean indicating if the car has left the track (in which case it is severely penalised)
    finish_line: bool
        Boolean indicating if the car has reached the finish line, in which case it is very very
        positively rewarded
    previous_distance: float
        Distance, given in m, between car and last checkpoint at previous state
    current_distance: float
        Distance, given in m, between car and last checkpoint at current state
    a: float
        Longitudinal acceleration, given in m/s^2
    n: float
        Lateral (centripetal) acceleration, given in m/s^2

    Returns
    -------
    current_reward: float
        current reward, to be added to the total reward
    delta_distance_travelled: float
        travelled distance between time steps
    '''

    '''
    # Initialise current reward, penalising agent by 1 point for each second that passes
    current_reward = - inp.delta_t
    
    # Give incentive to move forward
    delta_distance_travelled = current_distance - previous_distance # positive if car is moving forward

    # Reset distance when new index is reached
    # otherwise delta would cancel out everyting that
    # was achieved before (e.g.: delta = - 400 m)
    if np.absolute(delta_distance_travelled) > inp.index_pos_tolerance:
        delta_distance_travelled = 0

    current_reward += delta_distance_travelled * inp.delta_distance_normalisation_factor

    # Penalise if car left track.
    if left_track:
        current_reward += -1e3
    
    # Give big bonus when car completes lap
    if finish_line:
        current_reward += 1e3
        '''

    delta_distance_travelled = current_distance - previous_distance # positive if car is moving forward

    # Reset distance when new index is reached
    # otherwise delta would cancel out everyting that
    # was achieved before (e.g.: delta = - 400 m)
    if np.absolute(delta_distance_travelled) > inp.index_pos_tolerance:
        delta_distance_travelled = 0

    current_reward = delta_distance_travelled * inp.delta_distance_normalisation_factor # - inp.delta_t

    return current_reward, delta_distance_travelled



def chance_of_noise(reward_history, current_distance, max_distance, max_count):
    '''
    TODO: Explain function
    '''

    if reward_history == []:
        return 1 # 100% chance of noise when there is no reward history
    
    if len(reward_history) > inp.batch_size:
        reward_history = reward_history[-inp.batch_size:] # Keep size of reward history batch constant
    
    x = inp.noise_reduction_factor

    distance_factor = x if max_distance - current_distance > max(0.1 * max_distance, 30) else 1 # ( np.exp( n * current_distance / max_distance ) - 1 ) / ( np.exp( n ) - 1 )

    chance_of_noise = distance_factor * np.exp( - np.var( reward_history ) )

    count_tol = 75

    if max_count > count_tol:
        chance_of_noise = min( 1 , chance_of_noise + ( max_count - count_tol ) / count_tol )

    return chance_of_noise


###########################################
# Circuit Definition ######################
###########################################
'''
Routine to read coordinates from file for a
chosen circuit. Inner and outter coordinates
of the track are created, along with the
coordinates of the centerline.
'''

# Load chosen circuit from input file
chosen_circuit = inp.chosen_circuit

# Define folder path
circuit_dir = os.path.dirname((__file__)) + "/circuits"
# Create list of available circuits
circuit_list = str(next( os.walk( circuit_dir ) )[2])
circuit_list = circuit_list.replace('.txt', '')

# Check if chosen circuit is available
if chosen_circuit in circuit_list:
    # Read coordinates
    coordinates = np.loadtxt(circuit_dir + '/' + chosen_circuit + '.txt')
else:
    # If circuit is not available, print list of available circuits and exit program
    print("Circuit not available. Please pick a circuit from the following list: " + circuit_list.strip(']['))
    exit(0)

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
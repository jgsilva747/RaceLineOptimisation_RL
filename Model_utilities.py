# General imports
import numpy as np
import math

# File inports
import Inputs as inp


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
        acc = inp.braking_acceleration * 9.81
    # Compute acceleration when accelerating
    else:
        # profile based on empirical data
        acc = inp.x1 + inp.x2 * speed**2 if speed < inp.x3/3.6 else inp.x1 + inp.x2 * ( inp.x3/3.6)**2 - inp.x4 * ( speed - inp.x3/3.6 )

    return throttle * acc


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
    Assess if simulation should end.
    This is done by checking if the car is within track limits, if the car
    has reached the finish line (= starting position) or if the maximum
    simulation time has been exceeded.

    Parameters
    ----------
    state: float array [1 x 7]
        State array, containing: x and y position [m], x and y velocity [m/s], mass [kg], longitudinal and lateral acceleration [g], respectively
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
    True or False bool
        indicates wheather simulation should end or not. True indicates that the simulation should end.
    '''

    # Assess if car has reached finish line
    if index == len(coordinates_out) - 1: # successfully reached finish line
        return True # end simulation
    
    # Assess if simulation time was exceeded
    elif time > inp.max_min * 60:
        return True # end simulation
    
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
                    return True # end simulation
            else:
                if y < y_out or y > y_in:
                    return True # end simulation

        # If current portion is (mostly) vertical, check if x (horizontal) position was exceeded
        else:
            # interpolate inner x coordinate for current y position
            x_in = x_1_in + ( y - y_1_in ) * ( x_2_in - x_1_in ) / ( y_2_in - y_1_in )

            # interpolate outter x coordinate for current y position
            x_out = x_1_out + ( y - y_1_out ) * ( x_2_out - x_1_out ) / ( y_2_out - y_1_out )

            # Check if x position exceeded track limits
            if x_out > x_in:
                if x > x_out or x < x_in:
                    return True # end simulation
            else:
                if x < x_out or x > x_in:
                    return True # end simulation

    # If none of the above cases was activated, then the simulation should not be terminated
    return False

def get_circuit_index(state, coordinates, circuit_index):
    '''
    Obtain index of current track coordinate w.r.t. the coordinate array

    Parameters
    ----------
    state: float array [1 x 7]
        State array, containing: x and y position [m], x and y velocity [m/s], mass [kg], longitudinal and lateral acceleration [g], respectively
    coordinates: float array [2 x number_of_track_points]
        array of xy coordinates of the curcuit's centre line
    curcuit_index: int
        index of the last propagated position withing the coordinate array
    
    Returns
    -------
    curcuit_index: int
        index of current position w.r.t. the coordinate array
    '''

    # Compute horizontal distance between end of current track portion and starting position
    x_distance = ( coordinates[circuit_index + 1, 0] - coordinates[0,0] ) * inp.circuit_factor
    # Compute vertical distance between end of current track portion and starting position
    y_distance = ( coordinates[circuit_index + 1, 1] - coordinates[0,1] ) * inp.circuit_factor

    # Assess if current car position is within a certain distance (tolerance defined in the inputs file)
    # of next track coordinates
    if np.absolute( x_distance - state[0]) < inp.index_pos_tolerance and np.absolute( y_distance - state[1]) < inp.index_pos_tolerance:
        # Increase index to indicate that car is in the next portion of the track
        circuit_index += 1

    # Return index (which may or may not have been updated)
    return circuit_index
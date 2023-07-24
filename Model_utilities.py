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
    action: int
        discrete action obtained from discrete SARSA policy.
        For the discrete model, 10 actions are possible
    
    Returns
    ----------
    action_array: float array [1 x 2]
        array with 2 actions: the first entry corresponds to the throttle/break,
        and the second entry corresponds to the wheel (left/right)
    '''

    # Convert action to throttle, if action is within the range of 0 to 4
    throttle = float( action < 5 ) * float( action - 2 ) / 2.0
    # Convert action to wheel, if action is within the range of 5 to 9
    wheel = float( action >= 5 ) * float( action - 7 ) / 2.0

    return [throttle, wheel]

def get_acceleration(speed, mass, throttle):
    '''
    Computes acceleration of car based on current state and throttle
    due to engine / braking. The output of this acceleration does not
    include centripetal accelerations, and is given in the direction
    of the wheels. In order to convert it to an n-t coordinate system,
    it needs to be multiplied by cosine and sine of the wheel angle.

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
        acc = inp.braking_acceleration * 9.81 * speed / (360/3.6)
    # Compute acceleration when accelerating
    else:
        # profile based on empirical data
        force = inp.x1 + inp.x2 * speed**2 if speed < inp.x3/3.6 else inp.x1 + inp.x2 * ( inp.x3/3.6)**2 - inp.x4 * ( speed - inp.x3/3.6 )
        acc = force / mass

    return throttle * acc
    

def left_track(position, coordinates_in, coordinates_out, index, margin):
    # TODO: explain
    
    # Compute angle/direction of current portion of the track in the xy frame (in degrees)
    angle = np.rad2deg(math.atan2(coordinates_out[index+1,1] - coordinates_out[index,1], coordinates_out[index+1,0] - coordinates_out[index,0]))  

    # Define x and y positions
    x = position[0]
    y = position[1]

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
    if -45 < angle < 45 or angle > 180 - 45 or angle < -180 + 45:
        # interpolate inner y coordinate for current x position
        y_in = y_1_in + ( x - x_1_in ) * ( y_2_in - y_1_in ) / ( x_2_in - x_1_in )

        # interpolate outter y coordinate for current x position
        y_out = y_1_out + ( x - x_1_out ) * ( y_2_out - y_1_out ) / ( x_2_out - x_1_out )

        # Check if y position exceeded track limits
        if y_out > y_in:
            if y > y_out + margin or y < y_in - margin:
                return True # exceeded track limits
        else:
            if y < y_out - margin or y > y_in + margin:
                return True # exceeded track limits

    # If current portion is (mostly) vertical, check if x (horizontal) position was exceeded
    else:
        # interpolate inner x coordinate for current y position
        x_in = x_1_in + ( y - y_1_in ) * ( x_2_in - x_1_in ) / ( y_2_in - y_1_in )

        # interpolate outter x coordinate for current y position
        x_out = x_1_out + ( y - y_1_out ) * ( x_2_out - x_1_out ) / ( y_2_out - y_1_out )

        # Check if x position exceeded track limits
        if x_out > x_in:
            if x > x_out + margin or x < x_in - margin:
                return True # exceeded track limits
        else:
            if x < x_out - margin or x > x_in + margin:
                return True # exceeded track limits



def assess_termination(position, coordinates_in, coordinates_out, index, time, margin = inp.left_track_margin):
    '''
    Assess if simulation should end.
    This is done by checking if the car is within track limits, if the car
    has reached the finish line (= starting position) or if the maximum
    simulation time has been exceeded.

    Parameters
    ----------
    position: float array [1 x 2]
        Position array, containing: x and y position [m], respectively
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
        if left_track(position, coordinates_in, coordinates_out, index, margin):
            return True, True, False

    # If none of the above cases was activated, then the simulation should not be terminated
    return False, False, False

def get_circuit_index(position, coordinates, circuit_index, strict = False):
    '''
    Obtain index of current track coordinate w.r.t. the coordinate array

    Parameters
    ----------
    position: float array [1 x 2]
        Position array, containing: x and y position [m], respectively
    coordinates: float array [2 x number_of_track_points]
        array of xy coordinates of the curcuit's centre line
    curcuit_index: int
        index of the last propagated position withing the coordinate array
    
    Returns
    -------
    curcuit_index: int
        index of current position w.r.t. the coordinate array
    '''
    
    if strict:
        tolerance = 1
    else:
        tolerance = inp.index_pos_tolerance

    # Get position of last checkpoint
    last_checkpoint = ( coordinates[ circuit_index ] - coordinates[0] )

    # Compute distance between car and last checkpoint
    current_distance_to_last_checkpoint = np.linalg.norm(position - last_checkpoint)

    # Compute horizontal distance between end of current track portion and starting position
    x_distance = ( coordinates[circuit_index + 1, 0] - coordinates[0,0] )
    # Compute vertical distance between end of current track portion and starting position
    y_distance = ( coordinates[circuit_index + 1, 1] - coordinates[0,1] )

    # Compute horizontal distance from car to checkpoint
    x_dif = np.absolute( x_distance - position[0])
    # Compute vertical distance from car to checkpoint
    y_dif = np.absolute( y_distance - position[1])

    # Compute total distance from car to checkpoint
    distance = np.linalg.norm([x_dif, y_dif]) # m

    # Assess if current car position is within a certain distance (tolerance defined in the inputs file)
    # of next track coordinates
    # if x_dif < inp.index_pos_tolerance and y_dif < inp.index_pos_tolerance:
    if distance < tolerance:
        # Increase index to indicate that car is in the next portion of the track
        circuit_index += 1

    # Return index (which may or may not have been updated) and current distance to last checkpoint
    return circuit_index, current_distance_to_last_checkpoint


def get_closest_point(coordinates, position, id):
    # TODO: explain

    # Compute slope of track equation in xy coordinates
    track_slope = ( coordinates[id + 1, 1] - coordinates[id , 1] ) / ( coordinates[id + 1, 0] - coordinates[id , 0] )
    # track equation
    y = lambda x : coordinates[id , 1] + track_slope * ( x - coordinates[id , 0] )

    # Compute unit track direction # FIXME: will have problems when agent reaches last index
    track_direction = ( coordinates[id + 1] - coordinates[id] ) / np.linalg.norm( coordinates[id + 1] - coordinates[id] )
    track_direction = np.array(track_direction)

    # Compute unit perpendicular direction
    perpendicular_track_direction = np.array([ - track_direction[1] , track_direction[0] ])

    # Initialise gradient search algorithm
    old_pos = position
    new_pos = position + perpendicular_track_direction

    old_error = old_pos[1] - y( old_pos[0] )
    new_error = new_pos[1] - y( new_pos[0] )

    # Define tolerance to end search
    tol = 1e-1

    while np.absolute( new_error ) > tol:

        # Compute difference gradient
        gradient = ( new_error - old_error ) / ( new_pos[0] - old_pos[0] )

        # Update positions
        old_pos = new_pos
        new_pos = old_pos - ( new_error / gradient ) * perpendicular_track_direction / old_pos[0] # FIXME: might have problems with slope = 0 or x = 0

        # Update errors
        old_error = old_pos[1] - y( old_pos[0] )
        new_error = new_pos[1] - y( new_pos[0] )
    
    return new_pos

def get_future_curvatures(coordinates, position, circuit_index, n_samples = 10, delta_sample = 5):
    '''
    Compute the 10 future (relative) curvatures of the track, computed at an interval of 5 m

    Parameters
    ----------
    coordinates: float array [2 x number_of_track_points]
        array of xy coordinates of the curcuit's centre line
    position: float array [1 x 2]
        Position array, containing: x and y position [m], respectively
    circuit_index: int
        index of current position w.r.t. the coordinate array
    n_samples: int
        number of samples of future curvatures
    delta_sample: float
        distance, in meters, between each measurement

    Returns
    -------
    future_curvatures: float array [1 x n_samples]
        array containing the future curvatures of the track, in radians
    '''

    # Find closest (interpolated) point in centre line
    starting_point = get_closest_point(coordinates, position, circuit_index) # TODO: test
    # assume it returns xy coordinates of closes point along centre line

    # Initialise future curvatures array
    future_curvatures = np.zeros((10))

    # Compute unit track direction # FIXME: will have problems when agent reaches last index
    track_direction = lambda id : ( coordinates[id + 1] - coordinates[id] ) / np.linalg.norm( coordinates[id + 1] - coordinates[id] )

    # Initial track direction
    initial_direction = track_direction( circuit_index )
    initial_angle = np.arctan2( initial_direction[1] , initial_direction[0] )

    # Initialise loop
    current_point = starting_point
    current_index = circuit_index

    for sample in range( n_samples ):

        # Compute next point
        next_point = current_point + track_direction(current_index) * delta_sample

        # Compute distance to next portion of the track (to update circuit_index)
        old_distance_to_end = np.linalg.norm( current_point - coordinates[circuit_index + 1] )
        current_distance_to_end = np.linalg.norm( next_point - coordinates[circuit_index + 1] )
        # Check if both points are within delta_sample + small_margin of next index
        if current_distance_to_end < delta_sample + 1 and old_distance_to_end < delta_sample + 1:
            # Update index
            current_index += 1

        # Get current angle of track
        current_direction = track_direction(current_index)
        current_angle = np.arctan2( current_direction[1] , current_direction[0] )

        # Store difference between current angle and initial angle
        relative_angle = initial_angle - current_angle

        # Check if relative angle is within [-pi, pi]
        if relative_angle > np.pi:
            relative_angle -= 2 * np.pi
        elif relative_angle < -np.pi:
            relative_angle += 2 * np.pi

        future_curvatures[sample] = relative_angle # negative angle means left, positive angle means right

    return future_curvatures # TODO: test this function!



def get_lidar_samples(coordinates, coordinates_in, coordinates_out, current_position, index, margin = inp.left_track_margin):
    # TODO: explain and test

    angle_list = np.deg2rad( [ -90, -45, -30, -15, 0, 15, 30, 45, 90 ] )

    # Initialise lidar samples
    lidar_samples = np.zeros((len(angle_list)))

    # Compute unit track direction
    track_direction = ( coordinates[index + 1] - coordinates[index] ) / np.linalg.norm( coordinates[index + 1] - coordinates[index] )
    track_angle = np.arctan2( track_direction[1] , track_direction[0] )

    previous_index = index - 1 if index > 0 else index

    for angle in angle_list:
        # Compute direction
        current_angle = track_angle - angle
        current_direction = np.array([ np.cos(current_angle) , np.sin(current_angle) ])

        position = current_position

        while not all( [ left_track(position, coordinates_in, coordinates_out, previous_index, margin),
                      left_track(position, coordinates_in, coordinates_out, index, margin),
                      left_track(position, coordinates_in, coordinates_out, index + 1, margin),
                      left_track(position, coordinates_in, coordinates_out, index + 2, margin),
                      left_track(position, coordinates_in, coordinates_out, index + 3, margin), ] ) and np.linalg.norm(position - current_position) < 200:
            # if still inside track, move 10 m away
            position = position + 10 * current_direction

        # Move 10 m back
        position = position - 10 * current_direction

        while not all( [ left_track(position, coordinates_in, coordinates_out, previous_index, margin),
                      left_track(position, coordinates_in, coordinates_out, index, margin),
                      left_track(position, coordinates_in, coordinates_out, index + 1, margin),
                      left_track(position, coordinates_in, coordinates_out, index + 2, margin),
                      left_track(position, coordinates_in, coordinates_out, index + 3, margin), ] ) and np.linalg.norm(position - current_position) < 200:
            # if still inside, move 2.5 m away
            position = position + 2.5 * current_direction
        
        # Move 2.5 m back
        position = position - 2.5 * current_direction

        while not all( [ left_track(position, coordinates_in, coordinates_out, previous_index, margin),
                      left_track(position, coordinates_in, coordinates_out, index, margin),
                      left_track(position, coordinates_in, coordinates_out, index + 1, margin),
                      left_track(position, coordinates_in, coordinates_out, index + 2, margin),
                      left_track(position, coordinates_in, coordinates_out, index + 3, margin), ] ) and np.linalg.norm(position - current_position) < 200:
            # if still inside, move 1 m away
            position = position + current_direction

        # Update LIDAR measurement
        lidar_measurement = position - current_position

        # Store measurement
        angle_id = int( np.where(angle_list == angle)[0] )
        lidar_samples[ angle_id ] = np.linalg.norm( lidar_measurement )

        '''
        # Initialise LIDAR sample
        position_close = current_position
        position_far = current_position + 200 * current_direction

        # FIXME: problems with last few track portions

        # # Check if there is no wall within a distance of 200 m in current direction
        # if not all( [ left_track(position_far, coordinates_in, coordinates_out, index - 1, margin),
        #               left_track(position_far, coordinates_in, coordinates_out, index, margin),
        #               left_track(position_far, coordinates_in, coordinates_out, index + 1, margin),
        #               left_track(position_far, coordinates_in, coordinates_out, index + 2, margin),
        #               left_track(position_far, coordinates_in, coordinates_out, index + 3, margin), ] ):
        #     position_close = position_far
        
        # LIDAR measurement error/tolerance, in meters
        tol = 1 # m

        previous_index = index - 1 if index > 0 else index

        while np.linalg.norm( position_far - position_close ) > tol:
            
            position = ( position_far + position_close ) / 2

            # Check if any wall was found
            if all( [ left_track(position, coordinates_in, coordinates_out, previous_index, margin),
                      left_track(position, coordinates_in, coordinates_out, index, margin),
                      left_track(position, coordinates_in, coordinates_out, index + 1, margin)]):#,
                      # left_track(position, coordinates_in, coordinates_out, index + 2, margin),
                      # left_track(position, coordinates_in, coordinates_out, index + 3, margin), ] ):
                # If wall was found, define position as position_far
                position_far = position
            else:
                # If no wall was found, we can still move forward. Assign position to position_close
                position_close = position

        # When we leave the loop, the distance between position_close and position_far is less than 1 m
        # Output average of positions
        position = (position_close + position_far) / 2
        # Store measurement for respective angle
        angle_id = int( np.where(angle_list == angle)[0] )
        lidar_samples[ angle_id ] = np.linalg.norm( position )'''
    

    return lidar_samples



def propagate_dynamics(state, position, mass, track_direction, action, coordinates, coordinates_in, coordinates_out, index, delta_t = inp.delta_t):
    '''
    Propagates car dynamics based on current state and defined action

    Parameters
    ----------
    state: float array [1 x 24]
        State array, containing: # veloxity norm [m/s], t acc [m/s^2], n acc [m/s^2], delta heading [rad], 
        10 future curvatures [rad], 9 LIDAR measurements [m], track limits [float]
    position: float array [1 x 2]
        xy position coordinates [m]
    mass: float
        Current mass [kg]
    track_direction: float array [1 x 2]
        Unit array containing the xy coordinates of the track direction (e.g.: [1,0] means track is horizontal, to the right)
    action: float array [1 x 2]
        Action array, containing: throttle (ranging from -1 to 1) [-] and steering (ranging from -1 to 1) [-], respectively
    coordinates: float array [2 x number_of_track_points]
        array of xy coordinates of the curcuit's centre line
    coordinates_in: float array [2 x number_of_track_points]
        array of xy coordinates of the curcuit's inner limit
    coordinates_out: float array [2 x number_of_track_points]
        array of xy coordinates of the curcuit's outter limit
    index: int
        index of current position w.r.t. the coordinate array
    delta_t: float
        Simulation time step [s]
    
    Returns
    -------
    state: float array [1 x 5]
        State array after one propagation step, with the same content as the input state array.
    '''

    # Define mass flow rate
    mass_flow_rate = 0.028 # kg/s

    # Compute norm of current velocity
    speed_norm = state[0] # m/s

    # Get speed in xy coordiantes
    speed_angle = np.arctan2[ track_direction[1] , track_direction[0] ] - state[3]
    speed_direction = np.array([ np.cos(speed_angle) , np.sin(speed_angle) ])
    speed = speed_direction * speed_norm

    if speed_norm < inp.min_speed:
        speed_norm = inp.min_speed

    # Convert steering input into an angle, with a maximum angle of 15 deg
    angle = - np.deg2rad(15) * action[1] # rad

    # Compute longitudinal acceleration based on current velocity, mass and throttle
    acceleration = get_acceleration(speed_norm, mass, action[0])

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
    new_position = [position[0] + speed[0] * delta_t + 0.5 * acc_x * delta_t**2, position[1] + speed[1] * delta_t + 0.5 * acc_y * delta_t**2]

    ################
    # Update state #
    ################
    # Velocity norm
    v_norm = np.linalg.norm(v)

    # Compute longitudinal acceleration in g's
    a = np.linalg.norm( [ acc_x_tangent, acc_y_tangent ] ) / 9.81 # g

    # Compute normal acceleration in g's
    n = np.linalg.norm( [ acc_x_normal, acc_y_normal ] ) / 9.81 # g

    # Compute new heading wrt track direction
    absolute_heading = np.arctan2( v[1] , v[0] )

    # Compute difference between current angle and track angle
    delta_heading = np.arctan2[ track_direction[1] , track_direction[0] ] - absolute_heading

    # Check if relative angle is within [-pi, pi]
    if delta_heading > np.pi:
        delta_heading -= 2 * np.pi
    elif delta_heading < -np.pi:
        delta_heading += 2 * np.pi
    
    # Get future curvatures list
    curvature_list = get_future_curvatures(coordinates, new_position, index)

    # Get LIDAR samples
    lidar_samples = get_lidar_samples(coordinates, coordinates_in, coordinates_out, new_position, index)

    # Updated state
    state = [v_norm, a, n, delta_heading, curvature_list, lidar_samples, float(False)] # NOTE: Last entry is only updated
                                                                                       # after running assess_termination()
                                                                                       # in the step function

    # Return updated state, position and mass
    return state, np.array(new_position), mass



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

coordinates = np.array(coordinates)
for i in range(len(coordinates)):
    coordinates[i] = ( coordinates[i] - start ) * circuit_factor
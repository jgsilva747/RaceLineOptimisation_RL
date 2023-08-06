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


MAX_INDEX = len(coordinates) - 1


def convert_action(action):
    '''
    Function that converts the discrete input (from 0 to 9)
    to the two arrays that can be read by the defined car class

    Parameters
    ----------
    action: float array [1 x 2]
        discrete action obtained from discrete SARSA policy.
        For the discrete model, 2 x 10 actions are possible
    
    Returns
    ----------
    action_array: float array [1 x 2]
        array with 2 actions: the first entry corresponds to the throttle/break,
        and the second entry corresponds to the wheel (left/right)
    '''

    # Convert action to throttle, if action is within the range of 0 to 9
    throttle = float( action[0] < 9 ) * float( action[0] - 4 ) / 4
    # Convert action to wheel, if action is within the range of 0 to 9
    wheel = float( action[1] < 9 ) * float( action[1] - 4 ) / 4

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
    

def left_track(position, index, margin):
    '''
    Assess if agent has left the track

    Parameters
    ----------
    position: float array [1 x 2]
        Position array, containing: x and y position [m], respectively
    index: int
        index of current position w.r.t. the coordinate array
    margin: float
        Distance [m] which the car is allowed to exceed wrt track limits
        before it is considered that the car has left the track
    
    Returns
    ----------
    left_track: bool
        Bool indicating if the car has left the track. True means the car has left the track
    '''

    # Max index exceeded - will compute angle of previous (which is the final) track portion
    if index >= MAX_INDEX:
        index = MAX_INDEX - 1

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

    # If track limits were not exceeded    
    return False



def assess_termination(position, index, time, margin = inp.left_track_margin):
    '''
    Assess if simulation should end.
    This is done by checking if the car is within track limits, if the car
    has reached the finish line (= starting position) or if the maximum
    simulation time has been exceeded.

    Parameters
    ----------
    position: float array [1 x 2]
        Position array, containing: x and y position [m], respectively
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
    if index >= MAX_INDEX: # successfully reached finish line
        return True, False, True # end simulation
    
    # Assess if simulation time was exceeded
    elif time > inp.max_min * 60:
        return True, False, False # end simulation
    
    # Assess if car left the race track
    else:
        if left_track(position, index, margin):
            return True, True, False

    # If none of the above cases was activated, then the simulation should not be terminated
    return False, False, False

def get_circuit_index(position,circuit_index):
    '''
    Obtain index of current track coordinate w.r.t. the coordinate array

    Parameters
    ----------
    position: float array [1 x 2]
        Position array, containing: x and y position of closest centerline point [m], respectively
    curcuit_index: int
        index of the last propagated position withing the coordinate array
    
    Returns
    -------
    curcuit_index: int
        index of current position w.r.t. the coordinate array
    '''

    flag = False

    if circuit_index >= MAX_INDEX:
        flag = True
        circuit_index = MAX_INDEX - 1

    # Get position of last checkpoint
    last_checkpoint = coordinates[ circuit_index ]

    # Get position of next checkpoint
    next_checkpoint = coordinates[ circuit_index + 1 ]

    # Compute distance between car and last checkpoint
    current_distance_to_last_checkpoint = np.linalg.norm(position - last_checkpoint)

    # Compute distance between checkpoints
    distance_between_checkpoints = np.linalg.norm( next_checkpoint - last_checkpoint )

    # Check if index requires update
    if current_distance_to_last_checkpoint > distance_between_checkpoints:
        circuit_index += 1


    # Return index (which may or may not have been updated) and current distance to last checkpoint
    return circuit_index, current_distance_to_last_checkpoint, flag


def get_closest_point(position, id):
    '''
    Get the closest point in the centre line of the track

    Parameters
    ----------
    position: float array [1 x 2]
        Position array, containing: x and y position [m], respectively
    circuit_index: int
        index of current position w.r.t. the coordinate array
    
    Returns
    ---------
    new_pos: float array [1 x 2]
        X and y coordinates [m], respectively, of the center line point that is
        closest to the agent's scurrent position
    '''

    if id >= MAX_INDEX:
        id = MAX_INDEX - 1

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

    if old_pos[0] == 0:
        return old_pos

    # Define tolerance to end search
    tol = 1e-1
    count = 0

    while np.absolute( new_error ) > tol * (1 + count/10):

        # Compute difference gradient
        gradient = ( new_error - old_error ) / ( new_pos[0] - old_pos[0] )

        # Update positions
        aux = new_pos
        new_pos = old_pos - ( old_error / gradient ) * perpendicular_track_direction / perpendicular_track_direction[0] # FIXME: might have problems with slope = 0 or x = 0
        old_pos = aux

        # Update errors
        old_error = old_pos[1] - y( old_pos[0] )
        new_error = new_pos[1] - y( new_pos[0] )

        count += 1

    return new_pos


def get_future_curvatures(position, circuit_index, n_samples = 10, delta_sample = 10):
    '''
    Compute the 10 future (relative) curvatures of the track, computed at an interval of 10 m

    Parameters
    ----------
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
    starting_point = get_closest_point(position, circuit_index)
    # assume it returns xy coordinates of closes point along centre line

    # Initialise future curvatures array
    future_curvatures = []

    # Compute unit track direction # FIXME: might have problems when agent reaches last index
    track_direction = lambda id : ( coordinates[id + 1] - coordinates[id] ) / np.linalg.norm( coordinates[id + 1] - coordinates[id] )

    # Initial track direction
    initial_direction = track_direction( circuit_index )
    initial_angle = np.arctan2( initial_direction[1] , initial_direction[0] )

    # Initialise loop
    current_point = starting_point
    current_index = circuit_index

    for _ in range( n_samples ):

        # Compute next point
        current_track_direction = track_direction(current_index)
        next_point = current_point + current_track_direction * delta_sample

        # Update current point
        current_point = next_point

        # Get index of current point
        current_index , _, _ = get_circuit_index(current_point, current_index)
        if current_index >= MAX_INDEX:
            current_index = MAX_INDEX - 1


        # Get current angle of track
        current_direction = current_track_direction
        current_angle = np.arctan2( current_direction[1] , current_direction[0] )

        # Store difference between current angle and initial angle
        relative_angle = initial_angle - current_angle

        # Check if relative angle is within [-pi, pi]
        if relative_angle > np.pi:
            relative_angle -= 2 * np.pi
        elif relative_angle < -np.pi:
            relative_angle += 2 * np.pi

        future_curvatures.append( relative_angle ) # [sample] = relative_angle # negative angle means left, positive angle means right

    return future_curvatures, starting_point



def get_lidar_samples(current_position, index):
    '''
    Compute distance to wall in 9 directions: 0, +- 15deg, +- 30deg, +- 45deg and +- 90 deg

    Parameters
    ----------
    position: float array [1 x 2]
        Position array, containing: x and y position [m], respectively
    circuit_index: int
        index of current position w.r.t. the coordinate array
    
    Returns
    ----------
    lidar_samples: float array [1 x 9]
        array containing the 9 measurements (distance to nearest wall, [m]) in the following
        order: -90, -45, -30, -15, 0, 15, 30, 45, 90 [deg], where - is left and + is right
    '''

    # Define LIDAR angle list
    angle_list = np.deg2rad( [ -90, -45, -30, -15, 0, 15, 30, 45, 90 ] )

    # Initialise lidar samples
    lidar_samples = []

    # Compute unit track direction
    track_direction = ( coordinates[index + 1] - coordinates[index] ) / np.linalg.norm( coordinates[index + 1] - coordinates[index] )
    track_angle = np.arctan2( track_direction[1] , track_direction[0] )

    for angle in angle_list:

        # Flag for final index
        final_index_flag = False

        # Compute direction
        current_angle = track_angle - angle
        current_direction = np.array([ np.cos(current_angle) , np.sin(current_angle) ])

        # Initialise LIDAR position/measurement beam
        lidar_position = current_position
        lidar_index = index
        
        # Get 10 m precision (faster with less precision)
        while not left_track(lidar_position, lidar_index, margin=0):
            lidar_position = lidar_position + 10 * current_direction
            lidar_index , _ , final_index_flag = get_circuit_index(lidar_position, lidar_index)

            if np.linalg.norm(lidar_position - current_position) > 210 or final_index_flag:
                break
        
        # Move 10 m back
        if not final_index_flag:
            lidar_position = lidar_position - 10 * current_direction
            lidar_index , _ , final_index_flag = get_circuit_index(lidar_position, lidar_index)

        # Get 2.5 m precision
        while not left_track(lidar_position, lidar_index, margin=0):
            lidar_position = lidar_position + 2.5 * current_direction
            lidar_index , _ , final_index_flag = get_circuit_index(lidar_position, lidar_index)

            if np.linalg.norm(lidar_position - current_position) > 202.5 or final_index_flag:
                break

        # Move 2.5 m back
        if not final_index_flag:
            lidar_position = lidar_position - 2.5 * current_direction
            lidar_index , _ , final_index_flag = get_circuit_index(lidar_position, lidar_index)

        # Get 1 m precision (slower with more precision)
        while not left_track(lidar_position, lidar_index, margin=0):
            lidar_position = lidar_position + current_direction
            lidar_index , _ , final_index_flag = get_circuit_index(lidar_position, lidar_index)

            if np.linalg.norm(lidar_position - current_position) > 201 or final_index_flag:
                break

        # Update LIDAR measurement
        lidar_measurement = lidar_position - current_position
        distance = np.linalg.norm( lidar_measurement )
        if final_index_flag:
            distance = 200

        # Store measurement
        lidar_samples.append( min( distance , 200 ) )


    return lidar_samples



def rotate_reference(vec_to_rotate, ref_vec_old, ref_vec_new):
    '''
    TODO: explain funciton
    '''

    # Compute angle (in xy grid) of reference vectors
    angle_old = np.arctan2(ref_vec_old[1], ref_vec_old[0])
    angle_new = np.arctan2(ref_vec_new[1], ref_vec_new[0])

    # Compute rotation angle
    delta_angle = angle_old - angle_new # positive = rotated to the right

    # Apply rotation to vector
    new_vec = [vec_to_rotate[0] * np.cos(delta_angle) - vec_to_rotate[1] * np.sin(delta_angle),
               vec_to_rotate[0] * np.sin(delta_angle) + vec_to_rotate[1] * np.cos(delta_angle)]

    return new_vec


def integrate(state, derivative, delta_t, method = 'euler'):
    '''
    TODO: explain function
    '''

    # Simple Euler
    if method == 'euler':
        # Create integrated state variable
        integrated_state = []

        # Run loop for each state dimension
        for dim in range( len(state) ):
            integrated_state.append(state[ dim ] + derivative[ dim ] * delta_t)
    
    # RK4 - NOTE: only works for 2 D
    elif method == 'rk4':
        assert len(state)==2, "RK4 implementation only available for 2D arrays. Use 'euler' integrator instead."

        # Create integrated state variable
        integrated_state = []

        # First step derivative
        k1 = derivative

        # Auxiliar state 1
        aux_state_1 = []
        for dim in range(2):
            aux_state_1.append(state[ dim ] + k1[ dim ] * delta_t / 2)
        
        # Second step derivative
        k2 = rotate_reference(derivative, state, aux_state_1)

        # Auxiliar state 2
        aux_state_2 = []
        for dim in range(2):
            aux_state_2.append(state[ dim ] + k2[ dim ] * delta_t / 2)

        # Third step derivative
        k3 = rotate_reference(derivative, state, aux_state_2)

        # Auxiliar state 3
        aux_state_3 = []
        for dim in range(2):
            aux_state_3.append(state[ dim ] + k2[ dim ] * delta_t)

        # Fourth step derivative
        k4 = rotate_reference(derivative, state, aux_state_3)

        for dim in range(2):
            integrated_state.append(state[ dim ] + ( k1[ dim ] + 2 * k2[ dim ] + 2 * k3[ dim ] + k4[ dim ] ) * delta_t / 6)

    # Print error
    else:
        print('Error: ' + str(method) + ' not implemented. Please select a different integration method')
        exit(0)


    return integrated_state


def propagate_dynamics(state, position, mass, velocity, track_direction, action, index, delta_t = inp.delta_t,
                       integration_method = inp.integration_method):
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
    velocity: float array [1 x 2]
        Array containing x and y components of velocity [m/s]
    track_direction: float array [1 x 2]
        Unit array containing the xy coordinates of the track direction (e.g.: [1,0] means track is horizontal, to the right)
    action: float array [1 x 2]
        Action array, containing: throttle (ranging from -1 to 1) [-] and steering (ranging from -1 to 1) [-], respectively
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
    acc_x_tangent = np.cos(angle) * acceleration * velocity[0] / speed_norm
    acc_y_tangent = np.cos(angle) * acceleration * velocity[1] / speed_norm

    # Compute normal acceleration
    acc_x_normal = - np.sin(angle) * ( speed_norm**2 / inp.wheelbase_length ) * velocity[1] / speed_norm # acceleration + 
    acc_y_normal = np.sin(angle) * ( speed_norm**2 / inp.wheelbase_length ) * velocity[0] / speed_norm # acceleration + 

    # Add normal and tangent accelerations in cartesian coordinates
    acc_x = acc_x_tangent + acc_x_normal 
    acc_y = acc_y_tangent + acc_y_normal

    # Propagate speed using Euler integrator
    v = integrate(velocity, [acc_x, acc_y], delta_t, integration_method)
    # v = [velocity[0] + acc_x * delta_t, velocity[1] + acc_y * delta_t]

    new_position = integrate(position, v, delta_t, integration_method)
    # Propagate position using Euler integrator
    # new_position = [position[0] + velocity[0] * delta_t + 0.5 * acc_x * delta_t**2, position[1] + velocity[1] * delta_t + 0.5 * acc_y * delta_t**2]
    new_position = np.array( new_position )


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
    delta_heading = np.arctan2( track_direction[1] , track_direction[0] ) - absolute_heading

    # Check if relative angle is within [-pi, pi]
    if delta_heading > np.pi:
        delta_heading -= 2 * np.pi
    elif delta_heading < -np.pi:
        delta_heading += 2 * np.pi
    
    # Get future curvatures list
    curvature_list, centerline_pos = get_future_curvatures(new_position, index)

    # Get LIDAR samples
    lidar_samples = get_lidar_samples(new_position, index)

    # Updated state
    # state = [v_norm, a, n, delta_heading, curvature_list, lidar_samples, float(False)] 
    state_branch = [v_norm, a, n, delta_heading]
    state = np.concatenate( [state_branch , curvature_list , lidar_samples , [float(False)]] ) # NOTE: Last entry is only updated
                                                                                               # after running assess_termination()
                                                                                               # in the step function
    # Return updated state, position and mass
    return state, new_position, mass, v, centerline_pos



def get_reward(left_track, finish_line, previous_distance, current_distance, reward_function, state, prev_v, prev_delta, new_action, old_action):
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
    reward_function: array of strings
        Defines which reward function(s) to use (for example, time + distance)
    state: float array [1 x 24]
        State array, containing: # veloxity norm [m/s], t acc [m/s^2], n acc [m/s^2], delta heading [rad], 
        10 future curvatures [rad], 9 LIDAR measurements [m], track limits [float]
    prev_v: float
        Previous velocity norm, given in m/s
    prev_delta: float
        Previous heading difference between velocity vector and centerline, given in rad
    new_action: float array [1 x 2]
        Current ction array, containing: throttle (ranging from -1 to 1) [-] and steering (ranging from -1 to 1) [-], respectively
    old_action: float array [1 x 2]
        Previous action array, containing: throttle (ranging from -1 to 1) [-] and steering (ranging from -1 to 1) [-], respectively
    

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
    '''
    [0] --> 'distance'
    [1] --> 'time'
    [2] --> 'forward_velocity'
    [3] --> 'max_velocity'
    [4] --> 'constant_action'
    [5] --> 'min_curvature']
    '''
    current_reward = 0
    delta_distance_travelled = current_distance - previous_distance # positive if car is moving forward

    # Reset distance when new index is reached
    # otherwise delta would cancel out everyting that
    # was achieved before (e.g.: delta = - 400 m)
    if np.absolute(delta_distance_travelled) > inp.index_pos_tolerance:
        delta_distance_travelled = 0

    # Add distance reward
    if 'distance' in reward_function:

        current_reward += delta_distance_travelled * inp.delta_distance_normalisation_factor # - inp.delta_t

    # Add time reward/penalty
    if 'time' in reward_function:
        current_reward -= inp.delta_t * inp.delta_t_normalisation_factor
    
    # Add forward velocity reward (equivalent to travelled distance)
    if 'forward_velocity' in reward_function:

        v_norm = state[0]
        delta = state[3] # angle between velocity direction and centerline

        fwd_v = v_norm * np.cos( delta ) - prev_v * np.cos( prev_delta ) # delta_v in the direction of the centerline [m/s]

        current_reward += fwd_v * inp.velocity_normalisation_factor

    # Add maximum velocity reward (go as fast as possible, brake as quickly as possible)
    if 'max_velocity' in reward_function:
        delta_v = ( state[0] - prev_v )

        # Add 0 when braking/coasting, add reward when accelerating
        current_reward += max( 0 , delta_v) * inp.velocity_normalisation_factor

    # Add action instability penalty
    if 'constant_action' in reward_function:
        
        # NOTE: Only applied to steering. Code for throttle is analogous, but using index [0] instead of [1]
        # eg: throttle = *_action[0]

        new_cmd = new_action[1]
        old_cmd = old_action[1]

        delta_cmd = np.absolute( new_cmd - old_cmd ) # from 0 to 2

        current_reward -= delta_cmd * inp.action_normalisation_factor


    # Add steering penality to force car to turn as little as possible
    if 'min_curvature' in reward_function:
        steering = np.absolute(new_action[1]) # from 0 to 1
        current_reward -= steering * inp.action_normalisation_factor

    if 'max_acc' in reward_function:
        throttle = new_action[0] # from -1 to 1

        if throttle > 0.99: # incentivise max throttle
            current_reward += inp.action_normalisation_factor
        else: # break as quickly as possible
            current_reward -= inp.delta_t * inp.delta_t_normalisation_factor

    if 'straight_line' in reward_function:
        wheel = new_action[1] # from -1 to 1

        if np.absolute(wheel) < 0.01: # incentivise driving in a straight line
            current_reward += inp.action_normalisation_factor
        else: # complete curve as quickly as possible --> not smooth! How can I make it smooth?
            current_reward -= inp.delta_t * inp.delta_t_normalisation_factor


    # Finish line reward
    if finish_line:
        current_reward += 0 # 13e

    # Collision penalty
    if left_track:
        current_reward -= 1e3
        # NOTE: By penalising collisions as a function of the velocity norm,
        #       the agent learns that collisions can be avoided by going slower # NOTE: FAILED!! Not a good idea

    # TODO: Add centripetal acceleration penalty (car drifts if there is not enough traction)

    return current_reward, delta_distance_travelled



def chance_of_noise(reward_history, current_distance, max_distance, max_count):
    '''
    Function that computes the chance of having exploration noise, as a function
    of the maximum distance achieved, and depending on whether the agent is stuck

    Parameters
    ----------
    reward_history: float array [1 x number of episodes]
        Array containing the total reward obtained at the end of each episode
    current_distance: float
        Current distance [m] achieved by the agent at each instant in the current episode
    max_distance: float
        Maximum distance [m] ever achieved by the agent in all previous episodes
    max_count: int
        Counter of episodes where the maximum distance was not increased. Used as a measure
        of how stuck the agent is
    
    Returns
    ----------
    chance_of_noise: float
        Chance of noise, from 0 to 1, where 1 indicates 100% chance of having noise
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
##########################################################
# Author: Joao Goncalo Dias Basto da Silva               #
#         Student No.: 5857732                           #
# Course: AE4-350 Bio-Inspired Intelligence and Learning #
# Date: July 2023                                        #
##########################################################


##########################################################
# IMPORTS ################################################
##########################################################

# General imports
import numpy as np
import os
import torch
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':12})

# File imports
import Inputs as inp
from Model_utilities import coordinates_in, coordinates_out, convert_action
from Car_class import CarEnvironment




##########################################################
# MODEL DEFINITION #######################################
##########################################################

# Define the environment in the RL library.
env = CarEnvironment()

# for reproducibility
torch.manual_seed(inp.seed)
np.random.seed(inp.seed)

# Environment action and states
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

max_action = float(env.action_space.high[0]) # max action is the same in both entries

##########################################################
# EPISODE SETUP ##########################################
##########################################################

# Create and initialise time variable
time = 0

# Create and initialise action array
'''
Possible actions:
[0] -1: brake; 0: no power; 1: accelerate
[1] -1: turn left; 0: go straight; 1: turn right
'''

# Initialise score history array
score_hist=[]

# Create Figures if ploting is set to True
if inp.plot_episode:
    fig, ax = plt.subplots(figsize=( 8 , 6))
    ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
    ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')
    # Apply tight layout to figure
    plt.tight_layout()
    # TODO: add labels

if inp.plot_stats:
    fig_reward, ax_reward = plt.subplots(figsize=( 8 , 6))
    ax_reward.set_title('Reward Evolution')
    ax_reward.set_xlabel('Episode [-]')
    ax_reward.set_ylabel('Reward [-]')
    ax_reward.grid()


# Define initial action
action = [1, 0]

##########################################################
# OVERWRITE ACTION_DIM TO MAKE DISCRETE MODEL ############
action_dim = 2                                           #
action_list = np.arange(0, 5)                            #
##########################################################

##########################################################
# SARSA SETUP ############################################
##########################################################

norm_array = [ 10 , 10 , 2 , 2 , 10 ]
state_dim_1 = int( ( env.observation_space.high[ 0 ] - env.observation_space.low[ 0 ] ) / norm_array[0] )
state_dim_2 = int( ( env.observation_space.high[ 1 ] - env.observation_space.low[ 1 ] ) / norm_array[1] )
state_dim_3 = int( ( env.observation_space.high[ 2 ] - env.observation_space.low[ 2 ] ) / norm_array[2] )
state_dim_4 = int( ( env.observation_space.high[ 3 ] - env.observation_space.low[ 3 ] ) / norm_array[3] )
state_dim_5 = int( ( env.observation_space.high[ 4 ] - env.observation_space.low[ 4 ] ) / norm_array[4] )

Q = np.zeros((state_dim_1, state_dim_2, state_dim_3, state_dim_4, state_dim_5, 5, 5))
policy = np.random.randint(action_list[0], action_list[-1], (2, state_dim_1, state_dim_2, state_dim_3, state_dim_4, state_dim_5))

##########################################################
# TODO: define following set of variables in inputs file
# Number of episodes
n_episodes = int(1e5)
# Factor to update Q(s,a)
alpha = 0.5
# Randomness factor
eps = 1
# Importance of future rewards (discount factor)
gamma = 0.9
##########################################################

# Variable used to get index of state within policy and Q:
state_0, _ =  env.reset()
get_index = - state_0

##########################################################
# RUNNING EPISODES #######################################
##########################################################

# Run episodes
for episode in range(n_episodes):

    if episode > 5e4:
        eps = 0
    elif episode > 2e4:
        eps = 0.05
    elif episode > 5e3:
        eps = 0.15

    # Define initial state and accelerations
    state, acc = env.reset()

    index_array = ( state + get_index ) 
    index_array = ( index_array / norm_array ).astype(int)

    # Initialise random action
    action_discrete = policy[ : , index_array[0] , index_array[1] , index_array[2] , index_array[3] , index_array[4] ]
    action = convert_action(action_discrete)

    # Initialise auxiliar propagation varaibles
    done = False # termination flag
    circuit_index = 0 # index of car's current position in the coordinate array

    # Reset reward
    total_reward = 0

    # Plot initial position (velocity and acceleration not included)
    if inp.plot_episode:
        ax.scatter(state[0], state[1], marker='.',  color = 'b', linewidths=0.01)

    # Run episode until termination
    while not done:

        # Propagate state
        new_state, reward, done, acc, current_distance = env.step(action)

        # Plot current position
        if inp.plot_episode:
            ax.scatter(new_state[0], new_state[1], marker='.', color = 'b', linewidths=0.01)

        total_reward += reward

        # Get indeces
        # State
        state_index = ( state + get_index )
        state_index = ( state_index / norm_array ).astype(int)
        # New State
        new_state_index = ( new_state + get_index )
        new_state_index = ( new_state_index / norm_array ).astype(int)
        # Action
        action_index = action_discrete

        current_Q = Q[ state_index[0],
                       state_index[1],
                       state_index[2],
                       state_index[3],
                       state_index[4],
                       action_discrete[0],
                       action_discrete[1] ]
        
        next_Q = Q[ new_state_index[0],
                    new_state_index[1],
                    new_state_index[2],
                    new_state_index[3],
                    new_state_index[4],
                    action_discrete[0],
                    action_discrete[1] ]

        # Update Q(s,a)
        Q[ state_index[0],
           state_index[1],
           state_index[2],
           state_index[3],
           state_index[4],
           action_discrete[0],
           action_discrete[1] ] = current_Q + alpha * ( reward + gamma * next_Q - current_Q )
        
        # Update policy
        if random.uniform(0,1) < eps:
            # Apply random policy
            policy[:,
                   state_index[0],
                   state_index[1],
                   state_index[2],
                   state_index[3],
                   state_index[4]] = np.random.randint(action_list[0], action_list[-1], 2)
        else:
            # Apply greedy policy

            # Find flattened index of maximum Q
            max_index_flat = np.argmax(Q[ state_index[0],
                                          state_index[1],
                                          state_index[2],
                                          state_index[3],
                                          state_index[4],
                                          : ,
                                          : ])
            
            # Convert flattened index to 2D indexes
            policy[:,
                   state_index[0],
                   state_index[1],
                   state_index[2],
                   state_index[3],
                   state_index[4]] = np.unravel_index(max_index_flat, Q.shape[-2:])
             

        # Next action
        action_discrete = policy[ : , new_state_index[0] , new_state_index[1] , new_state_index[2] , new_state_index[3] , new_state_index[4] ]
        action = convert_action(action_discrete)

        state = new_state

    print("Episode: {}  Total Reward: {:0.2f}  Max distance: {:0.2f}".format( episode + 1, total_reward, current_distance), end='\r')
    score_hist.append(total_reward)

    # Plot reward evolution
    if inp.plot_stats:
        ax_reward.plot(score_hist, c='tab:blue')
        # TODO: add more stats
        
    if any([ inp.plot_episode , inp.plot_stats ]):
        plt.pause(1e-3)

# plt.pause(0.01)

# Save policy
np.save("sarsa_trained_policy.npy", policy)


# Show all figures
plt.show()
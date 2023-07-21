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

##########################################################
# TODO: define following set of variables in inputs file #
# Number of episodes
n_episodes = 500
# Factor to update Q(s,a)
alpha = 0.5
# Randomness factor
eps = 0
# Importance of future rewards (discount factor)
gamma = 0.9
##########################################################

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
action_dim = 1                                           #
action_list = np.arange(0, 10)                           #
##########################################################


##########################################################
# RUNNING EPISODES #######################################
##########################################################

# Run episodes
for episode in range(n_episodes):

    # Define initial state and accelerations
    state, acc = env.reset()

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

        # TODO: pick action; add noise to action
        action = random.randint(0,9)
        action = convert_action(action)

        # Propagate state
        new_state, reward, done, acc, travelled_distance = env.step(action)

        # Plot current position
        if inp.plot_episode:
            ax.scatter(new_state[0], new_state[1], marker='.', color = 'b', linewidths=0.01)

        total_reward += reward

        # TODO: something using state, new_state, reward, action

    print("Episode: {}  Total Reward: {:0.2f}".format( episode + 1, total_reward))
    score_hist.append(total_reward)

    # Plot reward evolution
    if inp.plot_stats:
        ax_reward.plot(score_hist, c='tab:blue')
        # TODO: add more stats
        
    if any([ inp.plot_episode , inp.plot_stats ]):
        plt.pause(1e-3)

# plt.pause(0.01)

# Show all figures
plt.show()
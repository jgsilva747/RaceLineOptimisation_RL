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

global ax_reward
global policy
global score_hist
global max_score_hist

skip = True

if not skip:

    print("Setting up model")

    ##########################################################
    # MODEL DEFINITION #######################################
    ##########################################################

    # Define the environment in the RL library.
    env = CarEnvironment(reward_function=['sarsa'], sarsa=True)

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
    score_hist = []
    max_score_hist = []

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
    action_list = np.arange(0, 10)                            #
    ##########################################################

    ##########################################################
    # SARSA SETUP ############################################
    ##########################################################

    total_states = 24
    n_states = 7
    norm_array = np.ones(n_states)
    norm_array[0] = 2
    state_dim = np.zeros(n_states)

    state_skip = [1,2,3,7,8,9,10,11,12,13, 14,15,16,20,21,22,23]

    state = 0

    for i in range( total_states ):
        print(i)

        if i not in state_skip:

            if state >= 4 and state <= 6:
                norm_array[state] = 10
            state_dim[state] = int( ( env.observation_space.high[ i ] - env.observation_space.low[ i ] ) / norm_array[state] )

            state += 1


    Q = np.zeros((10,10))
    policy = np.random.randint(action_list[0], action_list[-1], (2))

    Q = Q.astype('float32')
    policy = policy.astype('float32')

    index = 1

    for size in state_dim:

        Q = Q[..., np.newaxis]
        Q = np.repeat(Q, size, axis = index + 1)
        policy = policy[..., np.newaxis]
        policy = np.repeat(policy, size, axis = index)

        index += 1


    # Variable used to get index of state within policy and Q:
    state_0, _ =  env.reset()
    v_aux = - state_0
    get_index = np.zeros((n_states))
    aux = 0
    for i in range(total_states):
        if i not in state_skip:
            get_index[aux] = v_aux[i]
            aux += 1


def sarsa_train(n_episodes, alpha, eps, gamma, Q, variable_noise=False,greedy=False):

    global ax_reward
    global policy
    global score_hist
    global max_score_hist

    print("Training agent")

    max_reward = 0
    max_reward_count = 0

    ##########################################################
    # RUNNING EPISODES #######################################
    ##########################################################


    # Run episodes
    for episode in range(n_episodes):

        # Define initial state and position
        state, current_position = env.reset()

        discrete_state = np.zeros((n_states))
        new_discrete_state = np.zeros((n_states))

        aux = 0

        for i in range(total_states):
            if i not in state_skip:
                discrete_state[aux] = state[i]
                aux += 1

        index_array = ( discrete_state + get_index ) 
        index_array = ( index_array / norm_array ).astype(int)


        # Initialise random action
        action_discrete = policy[ : , index_array[0] , index_array[1] , index_array[2] , index_array[3] , index_array[4] , index_array[5] ,index_array[6] ]
        action = convert_action(action_discrete)


        # Initialise auxiliar propagation varaibles
        done = False # termination flag

        # Reset reward
        total_reward = 0

        # Plot initial position (velocity and acceleration not included)
        if inp.plot_episode:
            ax.scatter(current_position[0], current_position[1], marker='.',  color = 'b', linewidths=0.01)

        # Run episode until termination
        while not done:

            # Propagate state
            next_state, reward, done, _, current_position = env.step(action)

            aux = 0

            for i in range(total_states):
                if i not in state_skip:
                    new_discrete_state[aux] = next_state[i]
                    aux += 1

            # Plot current position
            if inp.plot_episode:
                ax.scatter(current_position[0], current_position[1], marker='.', color = 'b', linewidths=0.01)

            total_reward += reward

            # Set variable noise
            if variable_noise:
                if total_reward < max_reward - 15: # 15
                    eps = 0.2
                else:
                    eps = 1

            # Get indeces
            # State
            state_index = ( discrete_state + get_index )
            state_index = ( state_index / norm_array ).astype(int)
            # New State
            new_state_index = ( new_discrete_state + get_index )
            new_state_index = ( new_state_index / norm_array ).astype(int)
            # Action
            action_index = [int(action_discrete[0]) , int(action_discrete[1])]

            current_Q = Q[ action_index[0],
                        action_index[1],
                        state_index[0],
                        state_index[1],
                        state_index[2],
                        state_index[3],
                        state_index[4],
                        state_index[5],
                        state_index[6] ]
            
            next_Q = Q[ action_index[0],
                        action_index[1],
                        new_state_index[0],
                        new_state_index[1],
                        new_state_index[2],
                        new_state_index[3],
                        new_state_index[4],
                        new_state_index[5],
                        new_state_index[6] ]

            # Update Q(s,a)
            Q[ action_index[0],
            action_index[1],
            state_index[0],
            state_index[1],
            state_index[2],
            state_index[3],
            state_index[4],
            state_index[5],
            state_index[6] ] = current_Q + alpha * ( reward + gamma * next_Q - current_Q )
            
            # Update policy
            if random.uniform(0,1) < eps:
                # Apply random policy
                policy[:,
                    state_index[0],
                    state_index[1],
                    state_index[2],
                    state_index[3],
                    state_index[4],
                    state_index[5],
                    state_index[6]] = np.random.randint(action_list[0], action_list[-1], 2)
            else:
                # Apply greedy policy

                # Find flattened index of maximum Q
                max_index_flat = np.argmax(Q[ :,
                                              :,
                                              state_index[0],
                                              state_index[1],
                                              state_index[2],
                                              state_index[3],
                                              state_index[4],
                                              state_index[5],
                                              state_index[6]])
                
                # Convert flattened index to 2D indexes
                policy[ :,
                        state_index[0],
                        state_index[1],
                        state_index[2],
                        state_index[3],
                        state_index[4],
                        state_index[5],
                        state_index[6]] = np.unravel_index(max_index_flat, Q.shape[:2])
                

            # Next action
            action_discrete = policy[ : , new_state_index[0] , new_state_index[1] , new_state_index[2] , new_state_index[3] , new_state_index[4] , new_state_index[5] , new_state_index[6] ]
            action = convert_action(action_discrete)

            discrete_state = new_discrete_state

        aux = max(total_reward, max_reward)
        if aux != max_reward:
            max_reward = aux
            max_reward_count = 0
        else:
            max_reward_count += 1

        print("Episode: {}  Total Reward: {:0.2f}  Max Reward: {:0.2f}".format( episode + 1, total_reward, max_reward))
        score_hist.append(total_reward)
        max_score_hist.append(max_reward)

    # Plot reward evolution
    # if inp.plot_stats:
    if False:
        ax_reward.plot(score_hist, c='tab:blue', label = 'Episode Reward')
        ax_reward.plot(max_score_hist, c='tab:orange', label = 'Max Reward')
            # TODO: add more stats
            
        # if any([ inp.plot_episode , inp.plot_stats ]) and episode % 50==0:
        #     plt.pause(0.1)

    # plt.pause(0.01)
    ax_reward.legend(loc='upper left')
    print("\n")

    if greedy:
        extra = "_greedy"
    else:
        extra = ''
    np.save("sarsa_score_hist" + extra + ".npy", np.array(score_hist))

    return Q, policy



def plot_learning():

    # Load reward history
    reward_hist = np.load('sarsa_score_hist.npy')
    # max_reward = []

    # for i in range(len(reward_hist)):
    #     max_reward.append( max( reward_hist[:i+1] ) )
    #     print(i)

    fig_reward, ax_reward = plt.subplots(figsize=( 8 , 3))
    ax_reward.set_title('Reward Evolution')
    ax_reward.set_xlabel('Episode [-]')
    ax_reward.set_ylabel('Reward [-]')
    ax_reward.grid()
    ax_reward.set_xscale('log')

    ax_reward.plot(reward_hist, c='tab:blue', label = 'Episode Reward')
    # ax_reward.plot(max_score_hist, c='tab:orange', label = 'Max Reward')

    # ax_reward.legend(loc='best')

    fig_reward.tight_layout()


def test_trained() -> None:

    print("Testing trained policy")

    # Load policy
    policy = np.load('sarsa_trained_policy.npy')

    # Define initial state and accelerations
    state, current_position = env.reset()

    discrete_state = np.zeros((n_states))
    new_discrete_state = np.zeros((n_states))

    aux = 0

    for i in range(total_states):
        if i not in state_skip:
            discrete_state[aux] = state[i]
            aux += 1

    index_array = ( discrete_state + get_index ) 
    index_array = ( index_array / norm_array ).astype(int)


    # Initialise random action
    action_discrete = policy[ : , index_array[0] , index_array[1] , index_array[2] , index_array[3] , index_array[4] , index_array[5] ,index_array[6] ]
    action = convert_action(action_discrete)


    # Initialise auxiliar propagation varaibles
    done = False # termination flag

    # Reset reward
    total_reward = 0


    # Run episode until termination
    while not done:

        # Propagate state
        next_state, reward, done, _, current_position = env.step(action)

        aux = 0

        for i in range(total_states):
            if i not in state_skip:
                new_discrete_state[aux] = next_state[i]
                aux += 1

        # Plot current position
        if inp.plot_episode:
            ax.scatter(current_position[0], current_position[1], marker='.', color = 'b', linewidths=0.01)

        total_reward += reward

        # New State
        new_state_index = ( new_discrete_state + get_index )
        new_state_index = ( new_state_index / norm_array ).astype(int)
            

        # Next action
        action_discrete = policy[ : , new_state_index[0] , new_state_index[1] , new_state_index[2] , new_state_index[3] , new_state_index[4] , new_state_index[5] , new_state_index[6] ]
        action = convert_action(action_discrete)

        discrete_state = new_discrete_state


    print("Total Reward: {:0.2f}".format( total_reward))


# import signal
 
# def handler(signum, frame):

#     global ax_reward
#     global policy
#     global score_hist
#     global max_score_hist

#     if inp.plot_stats:
#         ax_reward.plot(score_hist, c='tab:blue', label = 'Episode Reward')
#         ax_reward.plot(max_score_hist, c='tab:orange', label = 'Max Reward')
    
#     np.save("sarsa_trained_policy.npy", policy)
#     ax_reward.legend(loc='upper left')
#     plt.show()

#     print("\n\n")
#     exit(0)
 
# signal.signal(signal.SIGINT, handler)

if __name__ == '__main__':


    ##########
    # Random #
    ##########
    # Number of episodes
    n_episodes = int(1e2)
    # Factor to update Q(s,a)
    alpha = 0.2
    # Randomness factor
    eps = 1
    # Importance of future rewards (discount factor)
    gamma = 0.9

    # Q, policy = sarsa_train(n_episodes, alpha, eps, gamma, Q)


    # ############
    # # Variable #
    # ############
    # # Number of episodes
    # n_episodes = int(1e5)
    # # Factor to update Q(s,a)
    # alpha = 0.025 # 0.01 # 0.1
    # # Randomness factor
    # eps = 0.2 # 1
    # # Importance of future rewards (discount factor)
    # gamma = 0.9

    # Q, policy = sarsa_train(n_episodes, alpha, eps, gamma, Q, True)
    
    
    ##################
    # epsilon-greedy #
    ##################
    # Number of episodes
    n_episodes = int(9e2)
    # Factor to update Q(s,a)
    alpha = 0.5 # 0.01 # 0.1
    # Randomness factor
    eps = 0.2 # 1
    # Importance of future rewards (discount factor)
    gamma = 0.9

    # Q, policy = sarsa_train(n_episodes, alpha, eps, gamma, Q, greedy=True)

    # Save policy
    # np.save("sarsa_trained_policy.npy", policy)

    # Test learnt policy
    # test_trained() 

    # Plot learning process
    plot_learning()

    # Show all figures
    plt.show()


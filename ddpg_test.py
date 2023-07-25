# General imports
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':12})

# File imports
import Inputs as inp
import DDPG_classes as ddpg
import Model_utilities as util
from Model_utilities import coordinates_in, coordinates_out
from Car_class import CarEnvironment


if __name__ ==  '__main__':
    # Define the environment in the RL library.
    env = CarEnvironment()

    score_hist=[]
    # for reproducibility
    torch.manual_seed(inp.seed)
    np.random.seed(inp.seed)
    # Environment action and states
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    max_action = float(env.action_space.high[0])
    min_Val = torch.tensor(1e-7).float().to(inp.device) 

    # Create DDPG instance
    agent = ddpg.DDPG(state_dim, action_dim)

    # Create noise instance
    ou_noise = ddpg.OU_Noise(action_dim, inp.seed)
    exploration_factor = inp.exploration_factor

    if inp.plot_episode:
        fig, ax = plt.subplots(figsize=( 8 , 6))
        ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
        ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')
        # Apply tight layout to figure
        plt.tight_layout()

    if inp.plot_stats:
        fig_reward, ax_reward = plt.subplots(figsize=( 8 , 6))
        ax_reward.set_title('Reward Evolution')
        ax_reward.set_xlabel('Episode [-]')
        ax_reward.set_ylabel('Reward [-]')
        ax_reward.grid()
        # fig_action, ax_action = plt.subplots(figsize=( 8 , 6))

    # Initialise maximum index achieved by agent
    max_distance = 0
    max_count = 0
    chance_of_noise = 1
    noise_flag = float( random.uniform(0,1) < chance_of_noise )
    episode = 0

    # Train the agent for max_episodes
    # for i in range(inp.n_episodes):
    # while max_count < int( 1e2 ):
    for i in range(10):
        total_reward = 0
        state, current_position = env.reset()
        done = False

        agent_action = []

        # Plot initial position
        if inp.plot_episode:
            ax.scatter(current_position[0], current_position[1], marker='.',  color = 'b', linewidths=0.01)

        # Run episode
        while not done:
            # Obtain action from agent NN
            action = [1,0] # agent.select_action(state)
            agent_action.append( action )

            # Add Gaussian noise to actions for exploration
            # action = (action + noise_flag * np.random.normal(0, exploration_factor, size=action_dim)).clip(-max_action, max_action)

            # Add OU noise
            # action += noise_flag * ou_noise.sample()
            next_state, reward, done, current_position, current_distance = env.step(action)
            total_reward += reward
            # if render and i >= render_interval : env.render()
            agent.replay_buffer.push((state, next_state, action, reward, float(done)))
            state = next_state

            chance_of_noise = util.chance_of_noise(score_hist, current_distance, max_distance, max_count)
            noise_flag = float( random.uniform(0,1) < chance_of_noise )

            # print(f"Future curvatures: {np.rad2deg(state[4:14])}")

            # Plot final position
            if inp.plot_episode:
                ax.scatter(current_position[0], current_position[1], marker='.', color = 'b', linewidths=0.01)
                # print(f"Future curvatures: {np.rad2deg(state[4:14])}")
                plt.pause(1/60)

        
        agent_action = np.array(agent_action)    

        score_hist.append(total_reward)

        if inp.plot_stats:
            if total_reward < 10:
                ax_reward.plot(score_hist, c = 'tab:blue')
                # ax_action.scatter(i, np.mean(agent_action[:,0]), color='tab:blue')
                # ax_action.scatter(i, np.mean(agent_action[:,1]), color='tab:orange')
        
        
        # if any([ inp.plot_episode , inp.plot_stats ]):
        #     plt.pause(1/60)
        
        # Update max distance
        new_max_distance = max(max_distance, current_distance)
        if new_max_distance == max_distance:
            max_count += 1
        else:
            max_count = 0
            max_distance = new_max_distance

        print("Episode: {}  Total Reward: {:0.2f}  Current Distance: {:0.1f}  Max Distance: {:0.1f}  Mean Actions: {:0.3f} , {:0.3f}".format( episode, total_reward, current_distance, max_distance, np.mean(agent_action[:,0]), np.mean(agent_action[:,1])))
        agent.update()
        if episode % 100 == 0:
            agent.save()
        episode += 1
    
    # Save "trained" agent # NOTE: This agent cannot even complete Turn 1, but it learnt that the best way is forward
    agent.save()


    ###############################
    # RUN TRAINED AGENT ###########
    ###############################
    print("\n\nFinished training. Running trained agent\n")

    env = CarEnvironment()

    # Environment action and states
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Create DDPG instance
    agent = ddpg.DDPG(state_dim, action_dim)

    agent.load()

    fig, ax = plt.subplots(figsize=( 8 , 6))
    ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
    ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')
    # Apply tight layout to figure
    plt.tight_layout()

    state, current_position = env.reset()
    done = False

    agent_action = []

    # Plot initial position
    ax.scatter(current_position[0], current_position[1], marker='.',  color = 'b', linewidths=0.01)

    # Run episode
    while not done:
        # Obtain action from agent NN
        action = agent.select_action(state)
        agent_action.append( action )

        next_state, reward, done, current_position, current_distance = env.step(action)

        state = next_state        

        ax.scatter(current_position[0], current_position[1], marker='.', color = 'b', linewidths=0.01)

    agent_action = np.array(agent_action)    

    print("Max Distance: {:0.2f}  Mean Actions: {:0.3f} , {:0.3f}".format(current_distance, np.mean(agent_action[:,0]), np.mean(agent_action[:,1])))

    plt.show()
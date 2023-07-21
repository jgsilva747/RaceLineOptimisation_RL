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

    # Exploration Noise
    exploration_factor = inp.exploration_factor

    # Create DDPG instance
    agent = ddpg.DDPG(state_dim, action_dim)

    # Create noise instance
    ou_noise = ddpg.OU_Noise(action_dim, inp.seed)

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
    chance_of_noise = 1
    noise_flag = float( random.uniform(0,1) < chance_of_noise )

    # Train the agent for max_episodes
    for i in range(inp.n_episodes):
        total_reward = 0
        state, acc = env.reset()
        done = False
        last_index = 0

        agent_action = []

        # Plot initial position
        if inp.plot_episode:
            ax.scatter(state[0], state[1], marker='.',  color = 'b', linewidths=0.01)

        # Run episode
        while not done:

            # Obtain action from agent NN
            action = agent.select_action(state)
            agent_action.append( action )

            # Add Gaussian noise to actions for exploration
            action = (action + noise_flag * exploration_factor * np.random.normal(0, 1, size=action_dim)).clip(-max_action, max_action)

            # Add OU noise
            # action += noise_flag * ou_noise.sample()
            next_state, reward, done, acc, current_distance = env.step(action)
            total_reward += reward
            # if render and i >= render_interval : env.render()
            agent.replay_buffer.push((state, next_state, action, reward, float(done)))
            state = next_state        

            # Compute chance of noise based on the last batch of rewards, the current distance and the maximum distance travelled by agent
            chance_of_noise = util.chance_of_noise(score_hist, current_distance, max_distance)
            # Convert chance of noise into float
            noise_flag = float( random.uniform(0,1) < chance_of_noise )

            # Plot final position
            if inp.plot_episode:
                ax.scatter(state[0], state[1], marker='.', color = 'b', linewidths=0.01)

        
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
        max_distance = max(max_distance, current_distance)

        print("Episode: {}  Total Reward: {:0.2f}  Current Distance: {:0.1f}  Max Distance: {:0.1f}  Mean Actions: {:0.3f} and {:0.3f}".format( i, total_reward, current_distance, max_distance, np.mean(agent_action[:,0]), np.mean(agent_action[:,1])))
        agent.update()
        if i % 100 == 0:
            agent.save()

    plt.show()
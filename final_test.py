import numpy as np
import yaml
import torch
import d3rlpy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':12})

import Inputs as inp
from Model_utilities import coordinates_in, coordinates_out, coordinates
from Car_class import CarEnvironment


if inp.jupyter_flag:
    jupyter_dir = '/content/drive/My Drive/RL_racing_line/'
else:
    jupyter_dir = ''



plot_median = False
n_median = 1

global sac
global file_name



def test_trained(val=3219.255661065412, plot='throttle', file=None) -> None:
    state_lst = []

    file_name = './action_lists/reward_' + str(val) + '.npy'
    if file is not None:
        file_name = './action_lists/' + file

    actions = np.load(file_name)

    env = CarEnvironment(reward_function=['superhuman'])

    if inp.plot_episode:
        fig, ax = plt.subplots(figsize=( 8*0.9 , 3*0.9))
        ax_zoom = ax.inset_axes([0.01, 0.45, 0.2, 0.30])
        ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k', linewidth=0.5)
        ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k', linewidth=0.5)
        # ax.grid()

        plot_pos = []
        plot_v = []
        plot_throttle = []


    # Initialize episode
    state, current_position = env.reset()
    total_reward = 0
    lap_time = 0

    state_lst.append(state)

    if inp.plot_episode:
        plot_pos.append(current_position)
        plot_throttle.append(1)
        plot_v.append(3.6 * np.linalg.norm(state[:2]))
        # ax.scatter(current_position[0], current_position[1], marker='.', linewidths=0.01, c=( 3.6 * np.linalg.norm(state[:2]) ), s=10, cmap="plasma")

    for action in actions:
        # Select action based on current state
        # state_with_batch = np.expand_dims(state, axis=0)
        # sac.predict(state_with_batch)[0]

        # Execute action in the environment
        next_state, reward, _, lap_completed, current_position = env.step(action)

        # Accumulate the reward
        total_reward += reward

        # Move to the next state
        state = next_state

        # Update time
        lap_time += inp.delta_t

        state_lst.append(state)

        if inp.plot_episode:
            plot_pos.append(current_position)
            plot_throttle.append(action[0])
            plot_v.append(3.6 * np.linalg.norm(state[:2]))
            # scatter.append(ax.scatter(current_position[0], current_position[1], marker='.', linewidths=0.01, c=( 3.6 * np.linalg.norm(state[:2]) ), s=10, cmap="plasma"))

    # Show plot
    if inp.plot_episode:
        plot_pos = np.array(plot_pos)
        plot_v = np.array(plot_v)
        plot_throttle = np.array(plot_throttle)
        if plot=='throttle':
            points = ax.scatter(plot_pos[:,0], plot_pos[:,1], c = 100 * plot_throttle, s=10, cmap="RdYlGn")
            ax_zoom.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
            ax_zoom.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')
            ax_zoom.scatter(plot_pos[:,0], plot_pos[:,1], c = 100 * plot_throttle, s=10, cmap="RdYlGn")
        else:
            points = ax.scatter(plot_pos[:,0], plot_pos[:,1], c = plot_v, s=10, cmap="plasma")
            ax_zoom.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
            ax_zoom.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')
            ax_zoom.scatter(plot_pos[:,0], plot_pos[:,1], c = plot_v, s=10, cmap="plasma")
        cbar = fig.colorbar(points)
        if plot=='throttle':
            cbar.set_label('Throttle [%]')
        else:
            cbar.set_label('Velocity [km/h]')
        

        ax_zoom.set_xlim(-900, -870)
        ax_zoom.set_ylim(-1160, -1100)
        ax_zoom.set_xticklabels([])
        ax_zoom.set_yticklabels([])

        ax.indicate_inset_zoom(ax_zoom, edgecolor="black")
        ax.set_xlabel('X Coordinates [m]')
        ax.set_ylabel('Y Coordinates [y]')

        # ax.set_title(str(reward_function).strip(']['))
        # Apply tight layout to figure
        fig.tight_layout()
        fig.subplots_adjust(right=1)

    state_lst = np.array(state_lst)
    # np.save('state_example.npy', state_lst)
    if file is not None:
        print(lap_time)
    else:
        print(f"{'Lap completed in' if lap_completed else 'DNF in'} {lap_time} s")
        print(total_reward)


    np.save('state_example.npy', state_lst)



def plot_learning(index_0 = 0):

    fig, ax = plt.subplots(figsize=( 10 , 2))
    

    step_array = []
    step_aux = 0


    file_name = "./reward_test/'superhuman'_silverstone_500k.txt"
    data_array = np.genfromtxt(file_name)

    for step in data_array[:,1]:
        step_aux += step
        step_array.append(step_aux)
    
    file_name = "./reward_test/'superhuman'_silverstone_fine_tuning.txt"
    data_array2 = np.genfromtxt(file_name)

    for step in data_array2[index_0:,1]:
        step_aux += step
        step_array.append(step_aux)

    data_plot = np.concatenate([data_array[:,0], data_array2[index_0:,0]])


    # Plot reward evolution
    ax.plot(step_array, data_plot, linewidth=3)
    
    ax.set_xlabel('Steps [-]')
    ax.set_ylabel('Reward [-]')
    ax.set_yticks([0,1000,2000,3000])
    ax.grid()
    fig.tight_layout()


def reward_lap_time():

    reward_lst = np.genfromtxt('rewards.txt')
    time_lst = np.genfromtxt('times.txt')

    fig, ax = plt.subplots(figsize=( 6 , 3))

    # Plot reward evolution
    ax.plot(reward_lst, time_lst)
    
    ax.set_xlabel('Reward [-]')
    ax.set_ylabel('Lap Time [s]')
    ax.grid()
    fig.tight_layout()


if __name__ == "__main__":

    # For reproducibility
    np.random.seed(inp.seed)
    d3rlpy.seed(inp.seed)
    torch.manual_seed(inp.seed)

    # plot_learning(127) # 110

    # for val in [3002.893869771683, 3020.754500453207, 3069.331615602792, 3098.1577232915947, 3123.2222032716677]:
    #     test_trained(val = val)

    test_trained()
    # test_trained(plot='velocity')
    # import os
    # for file in os.listdir('action_lists'):
    #     test_trained(file=file)

    # reward_lap_time()

    plt.show()
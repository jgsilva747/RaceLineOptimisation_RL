import numpy as np
import yaml
import torch
import d3rlpy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':12})

import Inputs as inp
from Model_utilities import coordinates_in, coordinates_out
from Car_class import CarEnvironment

with open('sac_inputs.yml') as f:
    sac_inputs = yaml.load(f, Loader=yaml.FullLoader)


encoder_factory = d3rlpy.models.encoders.DenseEncoderFactory()
reward_scaler = d3rlpy.preprocessing.MultiplyRewardScaler(20)


plot_median = False
n_median = 1

global env
global file_name



def train(reward_function) -> None:

    global env
    global file_name

    file_name = './reward_test/' + str(reward_function)

    env = CarEnvironment(log_file=file_name + '.txt', reward_function=reward_function)
    eval_env = CarEnvironment()

    # setup algorithm
    sac = d3rlpy.algos.SACConfig(**sac_inputs["algo_settings"],
                                actor_encoder_factory = encoder_factory,
                                critic_encoder_factory = encoder_factory,
                                 reward_scaler = reward_scaler
                                ).create(device=inp.device)

    # multi-step transition sampling
    transition_picker = d3rlpy.dataset.MultiStepTransitionPicker(
        n_steps=sac_inputs["fit_settings"]["n_steps"],
        gamma=sac_inputs["algo_settings"]["gamma"],
    )

    # replay buffer for experience replay
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(
        limit=sac_inputs["fit_settings"]["limit"],
        env=env,
        transition_picker=transition_picker,
    )

    # start training
    sac.fit_online(
        env,
        buffer,
        eval_env=eval_env,
        n_steps=sac_inputs["fit_settings"]["n_steps"],
        n_steps_per_epoch=sac_inputs["fit_settings"]["n_steps_per_epoch"],
        update_interval=sac_inputs["fit_settings"]["update_interval"],
        update_start_step=sac_inputs["fit_settings"]["update_start_step"],
        random_steps=sac_inputs["fit_settings"]["random_steps"],
        save_interval=sac_inputs["fit_settings"]["n_steps"]
    )
    # Save trained agent
    env.save(file_name + '.d3')


def test_trained(reward_function) -> None:

    file_name = './reward_test/' + str(reward_function)
    
    sac = d3rlpy.load_learnable(file_name + ".d3", device=None)

    env = CarEnvironment()

    if inp.plot_episode:
        fig, ax = plt.subplots(figsize=( 8 , 6))
        ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
        ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')

        plot_pos = []
        plot_v = []


    # Initialize episode
    state, current_position = env.reset()
    done = False
    total_reward = 0# Plot initial position
    lap_time = 0

    if inp.plot_episode:
        plot_pos.append(current_position)
        plot_v.append(3.6 * np.linalg.norm(state[:2]))
        # ax.scatter(current_position[0], current_position[1], marker='.', linewidths=0.01, c=( 3.6 * np.linalg.norm(state[:2]) ), s=10, cmap="plasma")

    while not done:
        # Select action based on current state
        state_with_batch = np.expand_dims(state, axis=0)
        action = sac.predict(state_with_batch)[0]

        # Execute action in the environment
        next_state, reward, done, _, current_position = env.step(action)

        # Accumulate the reward
        total_reward += reward

        # Move to the next state
        state = next_state

        # Update time
        lap_time += inp.delta_t

        if inp.plot_episode:
            plot_pos.append(current_position)
            plot_v.append(3.6 * np.linalg.norm(state[:2]))
            # scatter.append(ax.scatter(current_position[0], current_position[1], marker='.', linewidths=0.01, c=( 3.6 * np.linalg.norm(state[:2]) ), s=10, cmap="plasma"))

    # Show plot
    if inp.plot_episode:
        plot_pos = np.array(plot_pos)
        plot_v = np.array(plot_v)
        points = ax.scatter(plot_pos[:,0], plot_pos[:,1], c= plot_v, s=10, cmap="plasma")
        cbar = fig.colorbar(points)
        cbar.set_label('Velocity [km/h]')
        # Apply tight layout to figure
        fig.tight_layout()
        fig.subplots_adjust(right=1)

    print(f"Lap time: {lap_time} s")

    plt.show()


def plot_learning(ax, reward_function, color = 'tab:blue', label = None):
    
    # Read data
    file_name = './reward_test/' + str(reward_function) + ".txt"
    data_array = np.genfromtxt(file_name)

    n_steps = sac_inputs["fit_settings"]["n_steps"]

    # Remove repeated entry in the beginning, remove last unfinished entry
    data_array = data_array[1:-1]

    # Create array with number of steps
    steps_array = np.arange(0, n_steps + 1, n_steps / (len(data_array) - 1) )

    # Create array of median values
    if plot_median:
        
        median_array = np.array( data_array )

        for i in range(len(data_array)):

            index_min = max(0, i + 1 - n_median)
            median_array[i] = np.median( data_array[index_min : i+1] )
        
        data_array = median_array
    
    if label is None:
        label = str(reward_function)

    # Plot reward evolution
    ax.plot(steps_array, data_array, color=color, label=label)



#######################################
# Save agent when Ctrl+C is pressed   #
#######################################
                                      #
import signal                         #
                                      #
def handler(signum, frame):           #
                                      #
    global env                        #
    global file_name                  #
                                      #
    env.save(file_name + '.d3')       #
                                      #
    print("\n\n")                     #
    exit(0)                           #
                                      #
signal.signal(signal.SIGINT, handler) #
                                      #
#######################################


if __name__ == "__main__":

    # For reproducibility
    np.random.seed(inp.seed)
    d3rlpy.seed(inp.seed)
    torch.manual_seed(inp.seed)


    '''
    [0] --> 'distance'
    [1] --> 'time'
    [2] --> 'forward_velocity'
    [3] --> 'max_velocity'
    [4] --> 'constant_action'
    [5] --> 'min_curvature'
    '''

    # To test individual reward function:
    reward_function = inp.reward_list[0]
    train([reward_function])

    '''
    # To test every reward function:
    for reward_function in inp.reward_list:
        train([reward_function])
    '''

    '''
    # To test multiple reward functions simultaneously:
    reward_function = [
                        inp.reward_list[3],
                        inp.reward_list[5]
                      ]
    train(reward_function)
    '''

    # test_trained()

    # plot_learning()
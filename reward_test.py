import numpy as np
import yaml
import torch
import d3rlpy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':14})

import Inputs as inp
from Model_utilities import coordinates_in, coordinates_out, coordinates
from Car_class import CarEnvironment


if inp.jupyter_flag:
    jupyter_dir = '/content/drive/My Drive/RL_racing_line/'
else:
    jupyter_dir = ''

with open(jupyter_dir + 'sac_inputs.yml') as f:
    sac_inputs = yaml.load(f, Loader=yaml.FullLoader)


# encoder_factory = d3rlpy.models.encoders.DenseEncoderFactory() # Too slow
encoder_factory = d3rlpy.models.encoders.DefaultEncoderFactory(activation='swish')
reward_scaler = d3rlpy.preprocessing.MultiplyRewardScaler(20)


plot_median = False
n_median = 1

global sac
global file_name



def train(reward_function) -> None:

    global sac
    global file_name

    if inp.jupyter_flag:
        file_name = jupyter_dir + str(reward_function).strip('][') # + '_discount_' + str(inp.superhuman_discount) + '_freq_' + str(inp.superhuman_frequency)
    else:
        file_name = './reward_test/' + str(reward_function).strip('][') + '_testing_short__times_3' # + '_discount_' + str(inp.superhuman_discount) + '_freq_' + str(inp.superhuman_frequency)

    env = CarEnvironment(log_file=file_name + '.txt', reward_function=reward_function)
    eval_env = CarEnvironment(reward_function=reward_function)

    # setup algorithm
    sac = d3rlpy.algos.SACConfig(**sac_inputs["algo_settings"],
                                # actor_encoder_factory = encoder_factory,
                                # critic_encoder_factory = encoder_factory,
                                # reward_scaler = reward_scaler
                                ).create(device=inp.device)

    # # default sac
    # sac = d3rlpy.algos.SACConfig(reward_scaler = reward_scaler).create(device=inp.device)

    # multi-step transition sampling
    transition_picker = d3rlpy.dataset.MultiStepTransitionPicker(
        n_steps=sac_inputs["fit_settings"]["n_steps"],
        gamma=sac_inputs["algo_settings"]["gamma"],
    )

    # replay buffer for experience replay
    fifo_buffer = d3rlpy.dataset.FIFOBuffer(
        limit=sac_inputs["fit_settings"]["limit"],
        # env=env,
        # transition_picker=transition_picker,
    )

    replay_buffer = d3rlpy.dataset.ReplayBuffer(buffer=fifo_buffer,
                                                env=env,
                                                transition_picker=transition_picker,
                                                cache_size=sac_inputs["fit_settings"]["cache"])

    # start training
    sac.fit_online(
        env,
        replay_buffer,
        eval_env=eval_env,
        n_steps=sac_inputs["fit_settings"]["n_steps"],
        n_steps_per_epoch=sac_inputs["fit_settings"]["n_steps_per_epoch"],
        update_interval=sac_inputs["fit_settings"]["update_interval"],
        update_start_step=sac_inputs["fit_settings"]["update_start_step"],
        random_steps=sac_inputs["fit_settings"]["random_steps"],
        save_interval=sac_inputs["fit_settings"]["n_steps"]
    )
    # Save trained agent
    sac.save(file_name + '.d3')


def test_trained(reward_function, extra = '',
                 plot='throttle') -> None:

    state_lst = []

    file_name = './reward_test/' + str(reward_function).strip('][') + extra
    
    sac = d3rlpy.load_learnable(file_name + ".d3", device=None)

    env = CarEnvironment(reward_function=reward_function)

    if inp.plot_episode:
        fig, ax = plt.subplots(figsize=( 8*0.9 , 3*0.9))
        ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
        ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')
        # ax.grid()

        plot_pos = []
        plot_v = []
        plot_throttle = []


    # Initialize episode
    state, current_position = env.reset()
    done = False
    total_reward = 0
    lap_time = 0

    state_lst.append(state)

    if inp.plot_episode:
        plot_pos.append(current_position)
        plot_throttle.append(1)
        plot_v.append(3.6 * np.linalg.norm(state[:2]))
        # ax.scatter(current_position[0], current_position[1], marker='.', linewidths=0.01, c=( 3.6 * np.linalg.norm(state[:2]) ), s=10, cmap="plasma")

    while not done:
        # Select action based on current state
        state_with_batch = np.expand_dims(state, axis=0)
        action = sac.predict(state_with_batch)[0]

        # Execute action in the environment
        next_state, reward, done, lap_completed, current_position = env.step(action)

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
        else:
            points = ax.scatter(plot_pos[:,0], plot_pos[:,1], c = plot_v, s=10, cmap="plasma")
        cbar = fig.colorbar(points)
        if plot=='throttle':
            cbar.set_label('Throttle [%]')
        else:
            cbar.set_label('Velocity [km/h]')
        # ax.set_title(str(reward_function).strip(']['))
        # Apply tight layout to figure
        fig.tight_layout()
        fig.subplots_adjust(right=1)

    state_lst = np.array(state_lst)
    # np.save('state_example.npy', state_lst)
    print(f"{str(reward_function).strip('][')} --> {'Lap completed in' if lap_completed else 'DNF in'} {lap_time} s")
    print(total_reward)


def test_trained_file_name(file_name_lst,
                           plot='throttle') -> None:


    for name in file_name_lst:
        state_lst = []


        file_name = './reward_test/' + name
        
        sac = d3rlpy.load_learnable(file_name + ".d3", device=None)

        env = CarEnvironment()

        if inp.plot_episode:
            fig, ax = plt.subplots(figsize=( 8*0.9 , 3.5*0.9))
            # Add zoom to figure
            ax_zoom = ax.inset_axes([0.5, 0.4, 0.3, 0.5])
            ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
            ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')
            # ax.grid()

            plot_pos = []
            plot_v = []
            plot_throttle = []


        # Initialize episode
        state, current_position = env.reset()
        done = False
        total_reward = 0
        lap_time = 0

        state_lst.append(state)

        if inp.plot_episode:
            plot_pos.append(current_position)
            plot_throttle.append(1)
            plot_v.append(3.6 * np.linalg.norm(state[:2]))
            # ax.scatter(current_position[0], current_position[1], marker='.', linewidths=0.01, c=( 3.6 * np.linalg.norm(state[:2]) ), s=10, cmap="plasma")

        while not done:
            # Select action based on current state
            state_with_batch = np.expand_dims(state, axis=0)
            action = sac.predict(state_with_batch)[0]

            # Execute action in the environment
            next_state, reward, done, lap_completed, current_position = env.step(action)

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
            
            
            ax_zoom.set_xlim(-460, -390)
            ax_zoom.set_ylim(-85, -30)
            ax_zoom.set_xticklabels([])
            ax_zoom.set_yticklabels([])

            ax.indicate_inset_zoom(ax_zoom, edgecolor="black")

            ax.set_xlabel('X Coordinates [m]')
            ax.set_ylabel('Y Coordinates [m]')

            # ax.set_title(str(reward_function).strip(']['))
            # Apply tight layout to figure
            fig.tight_layout()
            fig.subplots_adjust(right=1)

        state_lst = np.array(state_lst)
        # np.save('state_example.npy', state_lst)
        print(f"{name} --> {'Lap completed in' if lap_completed else 'DNF in'} {lap_time} s")



def tune_weight(reward_func, weight_array, penalty = False):
    file_name = './reward_test/' + str(reward_func).strip('][')

    for weight in weight_array:

        sac = d3rlpy.load_learnable(file_name + ('_penalty' if penalty else '') + '_' + str(weight) + ".d3", device=None)

        env = CarEnvironment()

        if inp.plot_episode:
            fig, ax = plt.subplots(figsize=( 8 , 6))
            ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
            ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')

            plot_pos = []
            plot_v = []
            plot_throttle = []


        # Initialize episode
        state, current_position = env.reset()
        done = False
        total_reward = 0# Plot initial position
        lap_time = 0

        if inp.plot_episode:
            plot_pos.append(current_position)
            plot_throttle.append(1)
            plot_v.append(3.6 * np.linalg.norm(state[:2]))
            # ax.scatter(current_position[0], current_position[1], marker='.', linewidths=0.01, c=( 3.6 * np.linalg.norm(state[:2]) ), s=10, cmap="plasma")

        while not done:
            # Select action based on current state
            state_with_batch = np.expand_dims(state, axis=0)
            action = sac.predict(state_with_batch)[0]

            # Execute action in the environment
            next_state, reward, done, lap_completed, current_position = env.step(action)

            # Accumulate the reward
            total_reward += reward

            # Move to the next state
            state = next_state

            # Update time
            lap_time += inp.delta_t

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
            # points = ax.scatter(plot_pos[:,0], plot_pos[:,1], c = plot_v, s=10, cmap="plasma")
            points = ax.scatter(plot_pos[:,0], plot_pos[:,1], c = 100 * plot_throttle, s=10, cmap="RdYlGn")
            cbar = fig.colorbar(points)
            # cbar.set_label('Velocity [km/h]')
            cbar.set_label('Throttle [%]')
            ax.set_title(str(reward_func).strip('][') + '_' + str(weight))
            # Apply tight layout to figure
            fig.tight_layout()
            fig.subplots_adjust(right=1)


        print(f"{str(reward_func).strip('][')}_{str(weight)} --> {'Lap completed in' if lap_completed else 'DNF in'} {lap_time} s")
        



def plot_learning(ax, reward_function, color = 'tab:blue', label = None):
    
    # Read data
    file_name = './reward_test/' + str(reward_function).strip('][') + ".txt"
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


def fine_tune(reward_function,
              extra = '') -> None:

    global sac
    global file_name

    if inp.jupyter_flag:
        file_name = jupyter_dir + str(reward_function).strip('][') + '_fine_tuning' + '_discount_' + str(inp.superhuman_discount) + '_freq_' + str(inp.superhuman_frequency)
    else:
        file_name = './reward_test/' + str(reward_function).strip('][') + '_fine_tuning' # + '_discount_' + str(inp.superhuman_discount) + '_freq_' + str(inp.superhuman_frequency)

    env = CarEnvironment(log_file=file_name + '.txt', reward_function=reward_function)
    eval_env = CarEnvironment(reward_function=reward_function)

    read_file = (jupyter_dir if inp.jupyter_flag else './reward_test/') + str(reward_function).strip('][') + extra

    print(f'Fine Tuning: opening {read_file}')

    # setup algorithm
    sac = d3rlpy.load_learnable(read_file + ".d3", device=None)
    
    # default sac
    # sac = d3rlpy.algos.SACConfig(reward_scaler = reward_scaler).create(device=inp.device)

    # multi-step transition sampling
    transition_picker = d3rlpy.dataset.MultiStepTransitionPicker(
        n_steps=sac_inputs["fit_settings"]["n_steps"],
        gamma=sac_inputs["algo_settings"]["gamma"],
    )

    # replay buffer for experience replay
    # replay buffer for experience replay
    fifo_buffer = d3rlpy.dataset.FIFOBuffer(
        limit=sac_inputs["fit_settings"]["limit"],
        # env=env,
        # transition_picker=transition_picker,
    )

    replay_buffer = d3rlpy.dataset.ReplayBuffer(buffer=fifo_buffer,
                                                env=env,
                                                transition_picker=transition_picker,
                                                cache_size=sac_inputs["fit_settings"]["cache"])

    # start training
    sac.fit_online(
        env,
        replay_buffer,
        eval_env=eval_env,
        n_steps=sac_inputs["fit_settings"]["n_steps"],
        n_steps_per_epoch=sac_inputs["fit_settings"]["n_steps_per_epoch"],
        update_interval=sac_inputs["fit_settings"]["update_interval"],
        update_start_step=sac_inputs["fit_settings"]["update_start_step"],
        random_steps=sac_inputs["fit_settings"]["random_steps"],
        save_interval=sac_inputs["fit_settings"]["n_steps"]
    )
    # Save trained agent
    sac.save(file_name + '.d3')


'''
#######################################
# Save agent when Ctrl+C is pressed   #
#######################################
                                      #
import signal                         #
                                      #
def handler(signum, frame):           #
                                      #
    global sac                        #
    global file_name                  #
                                      #
    try:                              #
        sac.save(file_name + '.d3')   #
    except:                           #
        pass                          #
                                      #
    print("\n\n")                     #
    exit(0)                           #
                                      #
signal.signal(signal.SIGINT, handler) #
                                      #
#######################################
'''

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
    [6] --> 'max_acc'
    [7] --> 'straight_line'
    [8] --> Distance progress + collision with kinetic energy ('from superhuman performance with DRL paper')
    '''

    # # To test individual reward function:
    # reward_function = inp.reward_list[10]
    # train([reward_function])


    # # To test every reward function individually:
    # for reward_function in inp.reward_list:
    #     train([reward_function])


    # # To test multiple reward functions simultaneously:
    # reward_function = [
    #                     inp.reward_list[8]
    #                   ]

    # train(reward_function)

    # To test list of reward functions
    list_to_train = [
                    [inp.reward_list[6],
                     inp.reward_list[7]]

                    #  [inp.reward_list[4],
                    #   inp.reward_list[6]],

                    #  [inp.reward_list[8]],

                    #  [inp.reward_list[5],
                    #   inp.reward_list[6]],
                    ]

    # for reward_function in list_to_train:
    #     train(reward_function)

    # trained_list = [[inp.reward_list[2],
    #                  inp.reward_list[3]],

    #                 [inp.reward_list[2],
    #                  inp.reward_list[6]],

    #                 [inp.reward_list[2],
    #                  inp.reward_list[3],
    #                  inp.reward_list[4]],

    #                 [inp.reward_list[2],
    #                  inp.reward_list[4],
    #                  inp.reward_list[6]]
    # ]

    # for reward_function in trained_list:
    #     test_trained(reward_function)


    # tune_weight([inp.reward_list[3],
    #              inp.reward_list[7]],
    #             weight_array = [2.5, 5, 7.5],
    #             penalty = False)


    # tune_weight([inp.reward_list[6],
    #              inp.reward_list[7]],
    #             weight_array = [1, 1.5, 2],
    #             penalty = True)

    # test_trained([inp.reward_list[8]],
    #              extra = '_testing_short')

    # test_trained([inp.reward_list[6],
    #               inp.reward_list[7]],
    #              extra = '_testing_short')
    # test_trained([inp.reward_list[4],
    #               inp.reward_list[6]],
    #              extra = '_testing_short')
    # test_trained([inp.reward_list[0]])
                 # extra = '_discount_' + str(inp.superhuman_discount) + '_freq_0.25')

    # test_trained([inp.reward_list[8]],
    #              extra = '_fine_tuning_discount_' + str(inp.superhuman_discount) + '_freq_' + str(inp.superhuman_frequency))

    # test_trained_file_name(["'constant_action', 'max_acc'_5",
    #                         "'constant_action', 'max_acc'_2",
    #                         # "'distance', 'max_acc', 'straight_line'",
    #                         "'distance', 'constant_action', 'max_acc'",
    #                         "'forward_velocity', 'constant_action', 'max_acc'",
    #                         "'forward_velocity', 'max_acc'",
    #                         "'forward_velocity', 'max_velocity', 'constant_action'",
    #                         "'forward_velocity', 'max_velocity'",
    #                         "'forward_velocity'",
    #                         "'max_acc', 'straight_line'_long",
    #                         "'max_acc', 'straight_line'_7.5",
    #                         "'max_velocity', 'constant_action'_1",
    #                         "'max_velocity', 'constant_action'_5",
    #                         "'min_curvature', 'max_acc'",
    #                         "'superhuman'_discount_0.98_freq_1.2",
    #                         "'superhuman'_fine_tuning_discount_0.95_freq_1",
    #                         "'time', 'forward_velocity', 'constant_action'",
    #                         "'time', 'max_velocity', 'constant_action'"])

    test_trained_file_name(["'max_acc', 'straight_line'_testing_short__times_10"])
    plt.show()

    # fine_tune([inp.reward_list[6],
    #            inp.reward_list[7]],
    #           extra='_testing_short_7.5')

    # plot_learning()
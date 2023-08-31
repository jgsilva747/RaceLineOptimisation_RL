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




encoder_factory = d3rlpy.models.encoders.DefaultEncoderFactory(activation='swish')
reward_scaler = d3rlpy.preprocessing.MultiplyRewardScaler(20)


if inp.jupyter_flag:
    jupyter_dir = '/content/drive/My Drive/RL_racing_line/'
else:
    jupyter_dir = ''

with open(jupyter_dir + 'sac_inputs.yml') as f:
    sac_inputs = yaml.load(f, Loader=yaml.FullLoader)


def train(file_name,
          reward_function=['superhuman']) -> None:


    if inp.jupyter_flag:
        file_name = jupyter_dir + file_name
    else:
        file_name = './reward_test/' + file_name

    env = CarEnvironment(log_file=file_name + '.txt', reward_function=reward_function)
    eval_env = CarEnvironment(reward_function=reward_function)

    # setup algorithm
    sac = d3rlpy.algos.SACConfig(**sac_inputs["algo_settings"],
                                actor_encoder_factory = encoder_factory,
                                critic_encoder_factory = encoder_factory,
                                reward_scaler = reward_scaler
                                ).create(device=inp.device)

    # default sac
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



def test_trained_file_name(file_name,
                           plot='throttle') -> None:


    state_lst = []
    
    sac = d3rlpy.load_learnable(file_name + ".d3", device=None)

    env = CarEnvironment()

    if inp.plot_episode:
        fig, ax = plt.subplots(figsize=( 8*0.9 , 3.5*0.9))
        # # Add zoom to figure
        # ax_zoom = ax.inset_axes([0.5, 0.4, 0.3, 0.5])
        ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k', linewidth=0.5)
        ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k', linewidth=0.5)
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
            # ax_zoom.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
            # ax_zoom.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')
            # ax_zoom.scatter(plot_pos[:,0], plot_pos[:,1], c = 100 * plot_throttle, s=10, cmap="RdYlGn")
        else:
            points = ax.scatter(plot_pos[:,0], plot_pos[:,1], c = plot_v, s=10, cmap="plasma") 
            # ax_zoom.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
            # ax_zoom.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')
            # ax_zoom.scatter(plot_pos[:,0], plot_pos[:,1], c = plot_v, s=10, cmap="plasma")
        cbar = fig.colorbar(points)
        if plot=='throttle':
            cbar.set_label('Throttle [%]')
        else:
            cbar.set_label('Velocity [km/h]')
        
        
        # ax_zoom.set_xlim(-460, -390)
        # ax_zoom.set_ylim(-85, -30)
        # ax_zoom.set_xticklabels([])
        # ax_zoom.set_yticklabels([])

        # ax.indicate_inset_zoom(ax_zoom, edgecolor="black")

        ax.set_xlabel('X Coordinates [m]')
        ax.set_ylabel('Y Coordinates [m]')

        # ax.set_title(str(reward_function).strip(']['))
        # Apply tight layout to figure
        fig.tight_layout()
        fig.subplots_adjust(right=1)

    state_lst = np.array(state_lst)
    # np.save('state_example.npy', state_lst)
    print(f"{file_name} --> {'Lap completed in' if lap_completed else 'DNF in'} {lap_time} s")
    print(total_reward)



if __name__ == '__main__':



    # Test learnt policy (indicate '.d3' file name)
    test_trained_file_name([
    "final_silverstone"
                            ])
    plt.show()
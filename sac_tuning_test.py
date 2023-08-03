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


optim_factory = d3rlpy.models.optimizers.AdamFactory()
encoder_factory = d3rlpy.models.encoders.DenseEncoderFactory()
q_func_factory = d3rlpy.models.q_functions.QRQFunctionFactory()
reward_scaler = d3rlpy.preprocessing.MultiplyRewardScaler(20) # 10


def train() -> None:

    env_chosen = CarEnvironment(log_file='./tuning/default/sac_chosen.txt')
    env_default = CarEnvironment(log_file='sac_default.txt')
    eval_env = CarEnvironment()

    # setup algorithm
    sac_chosen = d3rlpy.algos.SACConfig(**sac_inputs["chosen_settings"],
                                        #actor_optim_factory = optim_factory,
                                        #critic_optim_factory = optim_factory,
                                        #temp_optim_factory = optim_factory,
                                        actor_encoder_factory = encoder_factory,
                                        critic_encoder_factory = encoder_factory,
                                        #q_func_factory = q_func_factory,
                                        reward_scaler = reward_scaler).create(device=inp.device)

    sac_default = d3rlpy.algos.SACConfig().create(device=inp.device)

    # multi-step transition sampling
    transition_picker_chosen = d3rlpy.dataset.MultiStepTransitionPicker(
        n_steps=sac_inputs["chosen_extra"]["n_steps"],
        gamma=sac_inputs["chosen_settings"]["gamma"],
    )
    transition_picker_default = d3rlpy.dataset.MultiStepTransitionPicker(
        n_steps=sac_inputs["default_extra"]["n_steps"],
        gamma=sac_inputs["default_settings"]["gamma"],
    )

    # replay buffer for experience replay
    buffer_chosen = d3rlpy.dataset.create_fifo_replay_buffer(
        limit=sac_inputs["chosen_extra"]["limit"],
        env=env_chosen,
        transition_picker=transition_picker_chosen,
    )
    buffer_default = d3rlpy.dataset.create_fifo_replay_buffer(
        limit=sac_inputs["default_extra"]["limit"],
        env=env_default,
        transition_picker=transition_picker_default,
    )

    # start training
    sac_chosen.fit_online(
        env_chosen,
        buffer_chosen,
        eval_env=eval_env,
        n_steps=sac_inputs["chosen_extra"]["n_steps"],
        n_steps_per_epoch=sac_inputs["chosen_extra"]["n_steps_per_epoch"],
        update_interval=sac_inputs["chosen_extra"]["update_interval"],
        update_start_step=sac_inputs["chosen_extra"]["update_start_step"],
        random_steps=sac_inputs["chosen_extra"]["random_steps"],
        save_interval=sac_inputs["chosen_extra"]["n_steps"]
    )
    # Save trained agent
    sac_chosen.save('./tuning/default/sac_chosen.d3')

    # # start training default agent
    # sac_default.fit_online(
    #     env_default,
    #     buffer_default,
    #     eval_env=eval_env,
    #     n_steps=sac_inputs["default_extra"]["n_steps"],
    #     n_steps_per_epoch=sac_inputs["default_extra"]["n_steps_per_epoch"],
    #     update_interval=sac_inputs["default_extra"]["update_interval"],
    #     update_start_step=sac_inputs["default_extra"]["update_start_step"],
    #     random_steps=sac_inputs["default_extra"]["random_steps"],
    #     save_interval=sac_inputs["default_extra"]["n_steps"]
    # )

    # # Save trained agent
    # sac_default.save('sac_default.d3')


def test_trained() -> None:
    
    sac_chosen = d3rlpy.load_learnable("tuning/default/sac_chosen.d3", device=None)
    sac_default = d3rlpy.load_learnable("tuning/default/sac_default.d3", device=None)

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

    if inp.plot_episode:
        plot_pos.append(current_position)
        plot_v.append(3.6 * np.linalg.norm(state[:2]))
        # ax.scatter(current_position[0], current_position[1], marker='.', linewidths=0.01, c=( 3.6 * np.linalg.norm(state[:2]) ), s=10, cmap="plasma")

    while not done:
        # Select action based on current state
        state_with_batch = np.expand_dims(state, axis=0)
        action = sac_chosen.predict(state_with_batch)[0]

        # Execute action in the environment
        next_state, reward, done, _, current_position = env.step(action)

        # Accumulate the reward
        total_reward += reward

        # Move to the next state
        state = next_state

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
        ax.set_title('Chosen settings')
        # Apply tight layout to figure
        fig.tight_layout()
        fig.subplots_adjust(right=1)



    ###############################################################
    # Run episode again using agent trained with default settings #
    ###############################################################
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

    if inp.plot_episode:
        plot_pos.append(current_position)
        plot_v.append(3.6 * np.linalg.norm(state[:2]))
        # ax.scatter(current_position[0], current_position[1], marker='.', linewidths=0.01, c=( 3.6 * np.linalg.norm(state[:2]) ), s=10, cmap="plasma")

    while not done:
        # Select action based on current state
        state_with_batch = np.expand_dims(state, axis=0)
        action = sac_default.predict(state_with_batch)[0]

        # Execute action in the environment
        next_state, reward, done, _, current_position = env.step(action)

        # Accumulate the reward
        total_reward += reward

        # Move to the next state
        state = next_state

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
        ax.set_title('Default settings')
        # Apply tight layout to figure
        fig.tight_layout()
        fig.subplots_adjust(right=1)


    plt.show()


if __name__ == "__main__":

    # For reproducibility
    np.random.seed(inp.seed)
    d3rlpy.seed(inp.seed)
    torch.manual_seed(inp.seed)

    # train()

    test_trained()


################
# buffer = ReplayBuffer(maxlen=sac_inputs["train"]["buffer_size"], env=env)
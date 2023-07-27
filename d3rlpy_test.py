import numpy as np
import yaml
import argparse
import d3rlpy
from d3rlpy.dataset import ReplayBuffer
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':12})

import Inputs as inp
from Model_utilities import coordinates_in, coordinates_out
from Car_class import CarEnvironment


with open('sac_inputs.yml') as f:
    sac_inputs = yaml.load(f, Loader=yaml.FullLoader)

d3rlpy.seed(inp.seed)

def train() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v1")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=1)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    env = CarEnvironment()
    eval_env = CarEnvironment()

    # setup algorithm
    sac = d3rlpy.algos.SACConfig(
        batch_size=256,
        gamma=0.99,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        temp_learning_rate=3e-4,
        tau=0.001,
        n_critics=2,
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
    ).create(device=False)

    # multi-step transition sampling
    transition_picker = d3rlpy.dataset.MultiStepTransitionPicker(
        n_steps=args.n_steps,
        gamma=0.99,
    )

    # replay buffer for experience replay
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(
        limit=1000000,
        env=env,
        transition_picker=transition_picker,
    )

    # start training
    sac.fit_online(
        env,
        buffer,
        eval_env=eval_env,
        n_steps=100000,
        n_steps_per_epoch=1000,
        update_interval=1,
        update_start_step=1000,
        random_steps=1000
    )

    # Save trained agent
    sac.save('sac_agent.d3')


def test_trained() -> None:
    
    sac = d3rlpy.load_learnable("sac_agent.d3", device=None)

    env = CarEnvironment()

    if inp.plot_episode:
        fig, ax = plt.subplots(figsize=( 8 , 6))
        ax.plot(coordinates_out[:,0], coordinates_out[:,1], color = 'k')
        ax.plot(coordinates_in[:,0], coordinates_in[:,1], color = 'k')
        # Apply tight layout to figure
        plt.tight_layout()

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
        action = sac.predict(state_with_batch)[0]

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
        plt.show()


if __name__ == "__main__":

    # train()

    test_trained()


################
# buffer = ReplayBuffer(maxlen=sac_inputs["train"]["buffer_size"], env=env)
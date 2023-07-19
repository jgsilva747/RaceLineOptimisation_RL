# General imports
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':12})

# File imports
import Inputs as inp
import DDPG_classes as ddpg
import Model_utilities as util
from Model_utilities import coordinates, coordinates_in, coordinates_out

# Create a custom environment that adheres to the OpenAI Gym interface.
class CarEnvironment(gym.Env):
    def __init__(self):
        # TODO: explain function

        # Define initial mass
        self.mass_0 = inp.vehicle_mass + inp.fuel_mass # kg

        # Define initial velocity
        self.v_norm = inp.initial_velocity / 3.6 # m/s
        self.v_direction = coordinates[1] - coordinates[0]
        self.v_0 = self.v_norm * self.v_direction / np.linalg.norm(self.v_direction)

        # Initialise auxiliar propagation variables
        self.time = 0
        self.done = False
        self.circuit_index = 0

        # Define initial state
        self.state_0 = [0, 0, self.v_0[0], self.v_0[1], self.mass_0, util.get_acceleration(np.linalg.norm([self.v_0[0], self.v_0[1]]), self.mass_0, 1 ) / 9.81, 0]
        self.state_0 = np.array( self.state_0 )

        # Define action space
        self.action_space = spaces.Box(
                # possible actions: brake/accelerate; turn left/right.
                # Both options range from -1 to 1
                np.array([-1, -1]).astype(np.float32),
                np.array([+1, +1]).astype(np.float32),
            )
        
        # Define observation space
        self.observation_space = spaces.Box(
                # x position, y position, x velocity, y velocity, mass, long acc, lat acc
                np.array([
                    min(coordinates_out[:,0]),
                    min(coordinates_out[:,1]), 
                    inp.min_speed,
                    inp.min_speed, 
                    inp.vehicle_mass, 
                    -10,
                    -10
                    ]).astype(np.float32),
                np.array([
                    max(coordinates_out[:,0]), 
                    max(coordinates_out[:,1]),
                    400/3.6, # m/s
                    400/3.6, # m/s
                    inp.vehicle_mass + inp.fuel_mass,
                    10,
                    10
                    ]).astype(np.float32),
            )

    def reset(self):
        # TODO: explain function

        # Reset seed
        super().reset(seed=inp.seed)

        # Reset initial state
        # position, velocity, mass, lateral acceleration
        self.state = self.state_0

        # Reset auxiliar propagation varaibles
        self.done = False # termination flag
        self.circuit_index = 0 # index of car's current position in the coordinate array
        self.time = 0 # time, given in s

        return self.state

    def step(self, action):
        # TODO: explain function

        # Propagate state
        new_state = util.propagate_dynamics(self.state, action, inp.delta_t)
        
        # Update time
        self.time += inp.delta_t

        # Update track position index
        new_circuit_index, normalised_distance_to_checkpoint = util.get_circuit_index(self.state, coordinates, self.circuit_index)

        # Check termination condition
        self.done, self.left_track, self.finish_line = util.assess_termination(new_state,
                                                                               coordinates_in,
                                                                               coordinates_out,
                                                                               new_circuit_index,
                                                                               self.time)

        # Update state
        self.state = np.array( new_state )

        # Update circuit index if new checkpoint was reached (going back in the track does not count) 
        if new_circuit_index > self.circuit_index:
            self.circuit_index = new_circuit_index

        # Compute current reward
        reward = util.get_reward(self.left_track, self.finish_line, normalised_distance_to_checkpoint)

        return self.state, reward, self.done, {}


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
    fig_action, ax_action = plt.subplots(figsize=( 8 , 6))

# Train the agent for max_episodes
for i in range(inp.n_episodes):
    total_reward = 0
    state = env.reset()
    done = False

    agent_action = []

    # Plot initial position
    if inp.plot_episode:
        ax.scatter(state[0], state[1], marker='.',  color = 'b', linewidths=0.01)

    # Run episode
    while not done:
        action = agent.select_action(state)
        agent_action.append( action )

        # Add Gaussian noise to actions for exploration
        action = (action + exploration_factor*np.random.normal(0, 1, size=action_dim)).clip(-max_action, max_action)

        # Add OU noise
        # action += ou_noise.sample()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        # if render and i >= render_interval : env.render()
        agent.replay_buffer.push((state, next_state, action, reward, float(done)))
        state = next_state

    # Plot current position
    if inp.plot_episode:
        ax.scatter(state[0], state[1], marker='.', color = 'b', linewidths=0.01)
        plt.pause(1/60)
    
    agent_action = np.array(agent_action)
    # print(agent_action)

    if inp.plot_stats:
        if total_reward < 10:
            ax_reward.scatter(i, total_reward)
            ax_action.scatter(i, np.mean(agent_action[:,0]), color='tab:blue')
            ax_action.scatter(i, np.mean(agent_action[:,1]), color='tab:orange')
        plt.pause(1/60)

    score_hist.append(total_reward)
    print("Episode: {}  Total Reward: {:0.2f}  Mean Actions: {:0.1f} and {:0.1f}".format( i, total_reward, np.mean(agent_action[:,0]), np.mean(agent_action[:,1])))
    agent.update()
    if i % 100 == 0:
        agent.save()
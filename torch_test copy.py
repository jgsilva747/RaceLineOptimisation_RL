# General imports
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

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
        self.state_0 = [[0, 0, self.v_0[0], self.v_0[1], self.mass_0, util.get_acceleration(np.linalg.norm([self.v_0[0], self.v_0[1]]), self.mass_0, 1 ) / 9.81, 0]]

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
        new_circuit_index = util.get_circuit_index(self.state, coordinates, self.circuit_index)

        # Check termination condition
        self.done, self.left_track, self.finish_line = util.assess_termination(new_state,
                                                                               coordinates_in,
                                                                               coordinates_out,
                                                                               new_circuit_index,
                                                                               self.time)

        # Update state
        self.state = new_state

        # Check if new checkpoint was reached (going back in the track does not count) 
        checkpoint = ( new_circuit_index > self.circuit_index ) # True or False

        # Update circuit index
        if checkpoint:
            self.circuit_index = new_circuit_index

        # Compute current reward
        reward = util.get_reward(self.left_track, checkpoint, self.finish_line)

        return new_state, reward, self.done

'''

# Define the environment in the RL library.
env = CarEnvironment()

# Define different parameters for training the agent
# TODO: Define these in inputs file
# TODO: Make inputs file clearer
max_episode = 10
ep_r = 0
total_step = 0
score_hist=[]
# for rendering the environmnet
render=True
render_interval=10
# for reproducibility
torch.manual_seed(inp.seed)
np.random.seed(inp.seed)
# Environment action and states
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float().to(inp.device) 

# Exploration Noise
exploration_noise=0.1
exploration_noise=0.1 * max_action


# Create a DDPG instance
agent = ddpg.DDPG(state_dim, action_dim)

# Train the agent for max_episodes
for i in range(max_episode):
    total_reward = 0
    step = 0
    state = env.reset()
    done = False
    while not done:
        action = agent.select_action(state)
        # Add Gaussian noise to actions for exploration
        action = (action + np.random.normal(0, 1, size=action_dim)).clip(-max_action, max_action)
        #action += ou_noise.sample()
        next_state, reward, done = env.step(action)
        total_reward += reward
        # if render and i >= render_interval : env.render()
        agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))
        state = next_state
        
    score_hist.append(total_reward)
    total_step += step+1
    print("Episode: \t{}  Total Reward: \t{:0.2f}".format( i, total_reward))
    agent.update()
    if i % 100 == 0:
        agent.save()'''





# create the environment
env_name='MountainCarContinuous-v0'
env = gym.make(env_name)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define different parameters for training the agent
max_episode=100
ep_r = 0
total_step = 0
score_hist=[]
# for rensering the environmnet
render=True
render_interval=10
# for reproducibility
torch.manual_seed(0)
np.random.seed(0)
#Environment action ans states
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
min_Val = torch.tensor(1e-7).float().to(device) 

max_time_steps=5000
# Exploration Noise
exploration_noise=0.1
exploration_noise=0.1 * max_action

# Create a DDPG instance
agent = ddpg.DDPG(state_dim, action_dim)

# Train the agent for max_episodes
for i in range(max_episode):
    total_reward = 0
    step =0
    state = env.reset(seed = 0)
    for  t in range(max_time_steps):
        action = agent.select_action(state[0])
        # Add Gaussian noise to actions for exploration
        action = (action + np.random.normal(0, 1, size=action_dim)).clip(-max_action, max_action)
        print(type(action))
        #action += ou_noise.sample()
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        if render and i >= render_interval : env.render()
        agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))
        state = next_state
        if done:
            break
        step += 1
        
    score_hist.append(total_reward)
    total_step += step+1
    print("Episode: \t{}  Total Reward: \t{:0.2f}".format( i, total_reward))
    agent.update()
    if i % 100 == 0:
        agent.save()
env.close()
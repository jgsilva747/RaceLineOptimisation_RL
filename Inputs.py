import torch

# Choose circuit here
chosen_circuit = "test curve"

# Choose simulation time step
delta_t = 0.025 # s

# Define initial velocity (in km/h)
initial_velocity = 200 # km/h

# Define maximum simulation time (in minutes)
max_min = 3 # min

# Define maximum braking acceleration (positive, in g's)
braking_acceleration = 5.5 # g

# Define tolerance used to update track coordinate index
index_pos_tolerance = 6 # m

# Define minimum speed
min_speed = 0.1 # m/s

# Drag area
drag_area = 1.5 # m^2
# Density
rho = 1.225 # kg/m^3
# Drag coefficient
c_d = 0.7

# Vehicle mass
vehicle_mass = 798 # kg

# Initial fuel mass
fuel_mass = 100 # kg

# Wheelbase length (distance between front and rear axes of the car)
wheelbase_length = 3.7 # m

# perpendicular direction scale factor
direction_factor = 5e-5

# real life circuit scale factor
circuit_factor = 1e5

# Acceleration function parameters
x1 = 5
x2 = 7.3e-2
x3 = 55 # km/h
x4 = 0.27

# Normalisation factor for reward function (distance to next checkpoint)
checkpoint_distance_normalisation_factor = 1/10

#########################
# DDPG Inputs ###########
#########################

# TODO: Add comments, change values
capacity= int( 1e6 )

batch_size = 64 # 64
update_iteration = 100 # 200
# tau for soft updating
tau_actor = 0.001 # 0.001    
tau_critic = 0.001 # 0.001

gamma=0.99 # 0.99
directory = './'
hidden1=20 # 20
hidden2=40 # 64

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Seed
seed = 0

# Number of episodes
n_episodes = int( 1e6 )

# Exploration fator
exploration_factor = 2 # 1
# 2, 4e-2
# Learning rate
learning_rate_actor = 3e-3 # 3e-3
learning_rate_critic = 4e-2 # 2e-2

# show episode plot (trajectories)
plot_episode = True
# show reward and action plot
plot_stats = False

# Noise parameters
theta = 0.15 # 0.15 from paper
sigma = 0.2 # 0.2 from paper, 0.25 from code example
# NOTE: miu = 0 from paper
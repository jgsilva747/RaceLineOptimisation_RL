import torch

# Choose circuit here
chosen_circuit = "test curve"

# Choose simulation time step
delta_t = 0.01 # 0.01 # s

# Integration method
integration_method = 'euler'

# Define initial velocity (in km/h)
initial_velocity = 200 # km/h

# Define maximum simulation time (in minutes)
max_min = 3 # min

# Define maximum braking acceleration (positive, in g's)
braking_acceleration = 5.5 # g

# Define tolerance used to update track coordinate index
index_pos_tolerance = 0 # m

# Margin to end simulation when car leaves track (in meters)
left_track_margin = 0

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
x1 = 5 * 900
x2 = 7.3e-2 * 900
x3 = 55 # km/h
x4 = 0.29 * 900 # 0.27

# Normalisation factors for reward function
delta_distance_normalisation_factor = 1 / 10
delta_t_normalisation_factor = 1
velocity_normalisation_factor = 1 / 2
wheel_normalisation_factor = 7.5
throttle_normalisation_factor = 5
superhuman_discount = 0.98
superhuman_frequency = 1 # Hz
# Try 0.98 with freq = 5
# Or simply try 0.95 for a much longer time

# Braking distance
braking_distance = 150 # m

#########################
# DDPG Inputs ############
#########################

# TODO: Add comments, change values
capacity= int( 1e6 )

batch_size = 64 # 64
update_iteration = 100 # 200
# tau for soft updating
tau = 0.001 # 0.001   

gamma=0.99 # 0.99
directory = './'
hidden1=20 # 20
hidden2=64 # 64

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.cuda.is_available() # True if cuda is available, else false

# Seed
seed = 10

# Number of episodes
n_episodes = int( 50e3 )

# Exploration fator
exploration_factor = 0.3 # 1

# Noise reduction factor
noise_reduction_factor = 0.1

# 2, 4e-2
# Learning rate
learning_rate_actor = 0.00003 # 6e-4 # 8e-4 # 3e-3
learning_rate_critic = 0.00003 # 4e-2 # 2e-2

# show episode plot (trajectories)
plot_episode = True
# show reward and action plot
plot_stats = False

# Noise parameters
theta = 0.15 # 0.15 from paper
sigma = 0.2 # 0.2 from paper, 0.25 from code example
# NOTE: miu = 0 from paper



#########
log = True
# Plotting (true or false)
plotting = False



###########
reward_list = ['distance',
                'time',
                'forward_velocity',
                'max_velocity',
                'constant_action',
                'min_curvature',
                'max_acc',
                'straight_line',
                'superhuman',
                'sarsa',
                'mean_velocity']


################
jupyter_flag = False
if jupyter_flag:
    log = False

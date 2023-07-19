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


#########################
# DDPG Inputs ###########
#########################

# TODO: Add comments, change values
capacity=1000000

batch_size=64
update_iteration=200
# tau for soft updating
tau=0.001

gamma=0.99
directory = './'
hidden1=20
hidden2=64

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Seed
seed = 0
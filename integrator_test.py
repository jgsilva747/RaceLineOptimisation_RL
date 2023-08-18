import numpy as np
import d3rlpy
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
mpl.rcParams.update({'font.size':14})
from time import time

from Car_class import CarEnvironment

# sac = d3rlpy.load_learnable("sac_agent.d3", device=None)

def test_integrator(delta_t, integration_method, n_runs) -> None:

    assert n_runs > 0, "Please insert a positive number of runs"

    env = CarEnvironment(delta_t, integration_method)

    ########################
    # Get maximum velocity #
    ########################
    v_max = 0

    # Initialize episode
    state, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Select action based on current state
        state_with_batch = np.expand_dims(state, axis=0)
        action = sac.predict(state_with_batch)[0]

        # Execute action in the environment
        next_state, reward, done, _ , _ = env.step(action)

        # Accumulate the reward
        total_reward += reward

        # Move to the next state
        state = next_state

        v_norm = np.linalg.norm(state[:2])

        v_max = max( v_max , v_norm )
    
    v_max *= 3.6 # convert to km/h

    ########################
    # Get CPU times ########
    ########################

    t_start = time()

    for _ in range(n_runs):

        # Initialize episode
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # Select action based on current state
            state_with_batch = np.expand_dims(state, axis=0)
            action = sac.predict(state_with_batch)[0]

            # Execute action in the environment
            next_state, reward, done, _ , _ = env.step(action)

            # Accumulate the reward
            total_reward += reward

            # Move to the next state
            state = next_state

    t_end = time()

    return ( t_end - t_start ) / n_runs , v_max



if __name__ == "__main__":

    n_runs = 10

    fig, ax = plt.subplots(figsize=( 8 , 3.5))

    integrator_list = ['euler' , 'rk4']
    # delta_t_list = [0.001, 0.0025, 0.005, 0.01, 0.025]
    # delta_t_list = [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,
    #                 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3,
    #                 1e-2, 2e-2, 2.5e-2]
    delta_t_list = [0.0005, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2, 1.2e-2, 1.4e-2, 1.5e-2, 2e-2, 2.5e-2]

    # Define marker list
    marker_list = ['o',
                   '^']

    # Define label list
    label_list = ['Euler',
                  'RK4']
    
    # Define linestyle list
    ls_list = ['-',
               '-']

    # Define colour of markers in legend
    legend_colour = 'blue'

    cpu_times = np.zeros(( len(integrator_list) , len(delta_t_list) ))
    v_max = np.zeros(( len(integrator_list) , len(delta_t_list) ))
    '''
    for integration_method in integrator_list:

        # Get integrator index wrt integrator list
        integrator_index = integrator_list.index( integration_method )

        for delta_t in delta_t_list:

            # Get delta_t index wrt delta_t list
            delta_t_index = delta_t_list.index( delta_t )

            print(f'Running {label_list[integrator_index]} with delta_t = {delta_t} s')

            cpu_times[integrator_index, delta_t_index], v_max[integrator_index, delta_t_index] = test_integrator(delta_t, integration_method, n_runs)

    
    np.save('cpu_times.npy', cpu_times)
    np.save('v_max.npy', v_max)
    exit()
    '''
    cpu_times = np.load('cpu_times.npy')
    v_max = np.load('v_max.npy')
    norm = mpl.colors.Normalize(vmin = cpu_times.min(), vmax = cpu_times[:,1:].max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.rainbow) # plasma
    cmap.set_array([])

    v_max = np.array( v_max)

    min_v = v_max.min()

    abs_error = v_max - min_v

    rel_error = abs_error * 100 / min_v


    ax.scatter(delta_t_list[1], rel_error[0,1], linewidths=3, marker=marker_list[0], c=legend_colour, label = label_list[0])
    ax.scatter(delta_t_list[1], rel_error[1,1], linewidths=3, marker=marker_list[1], c=legend_colour, label = label_list[1])

    for integrator in range(len(integrator_list)):
        # ax.scatter(delta_t_list, v_max[integrator], linewidths=4, marker=marker_list[integrator], c=cmap.to_rgba(cpu_times[integrator]), label = label_list[integrator])
        ax.scatter(delta_t_list[1:], rel_error[integrator,1:], linewidths=4, marker=marker_list[integrator], c=cmap.to_rgba(cpu_times[integrator,1:]))

    ax.set_xscale('log')
    # ax.set_yscale('log')

    ax.axhline(y = 10, color = 'r', linestyle = '--', label = 'Error Tolerance')

    ax.legend(loc=0)

    ax.set_xlabel('$dt$ [s]')
    ax.set_ylabel('Integration Error [%]')

    cbar = fig.colorbar(cmap, ax=ax)
    cbar.set_label('Average Run Time [s]')

    ax.grid()

    plt.tight_layout()
    plt.show()
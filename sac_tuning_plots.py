# General imports
import numpy as np
import torch
import os
import d3rlpy
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':12})


# File imports
import Inputs as inp
# from Model_utilities import coordinates_in, coordinates_out
from Car_class import CarEnvironment
from sac_tuning import settings_dict, default_settings



# Choose whether to plot real values or median values
plot_median = True
batch_size = 1


plot_dir = 'Figures/tuning/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)



def plot_tuning():
    ...

    # read log files (.txt)




if __name__ == "__main__":

    color_array = ['tab:blue',
                   'tab:orange',
                   'tab:green',
                   'tab:cyan',
                   'tab:olive',
                   'tab:pink']


    # Read default data
    file_name = "./tuning/default/0.txt"
    default_data = np.genfromtxt(file_name)
    default_data = default_data[1:-1]

    # Get number of steps
    n_steps = default_settings.get('n_steps')

    # Create array with number of steps
    default_steps_array = np.arange(0, n_steps + 1, n_steps / (len(default_data) - 1) )

    # Create array of median values
    if plot_median:
        
        median_array = np.array( default_data )

        for i in range(len(default_data)):

            index_min = max(0, i + 1 - batch_size)
            median_array[i] = np.median( default_data[index_min : i+1] )
        
        default_data = median_array


    # Loop over available settings
    for settings in settings_dict:
        print(settings)

        # Create Figure
        fig, ax = plt.subplots(figsize=(8,6))

        ax.set_title('Effect of ' + settings + ' on reward evolution')

        values_array = settings_dict.get(settings)

        # Loop over selected values
        for value in values_array:

            # Read data
            file_name = "./tuning/" + settings + "/" + str(value) + ".txt"
            data_array = np.genfromtxt(file_name)

            # Remove repeated entry in the beginning, remove last unfinished entry
            data_array = data_array[1:-1]

            # Create array with number of steps
            steps_array = np.arange(0, n_steps + 1, n_steps / (len(data_array) - 1) )

            # Create array of median values
            if plot_median:
                
                median_array = np.array( data_array )

                for i in range(len(data_array)):

                    index_min = max(0, i + 1 - batch_size)
                    median_array[i] = np.median( data_array[index_min : i+1] )
                
                data_array = median_array

            # Plot reward evolution
            ax.plot(steps_array, data_array, color=color_array[ values_array.index( value ) ], label=str(value))

        # Plot default value
        ax.plot(default_steps_array, default_data, color='tab:red', label='default')
        
        # ax.legend(loc = 'best')
        fig.legend(loc = 'outside lower center', #loc='upper center', bbox_to_anchor=(0.5, -0.15),
            #ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
            fancybox=True, shadow=True, ncol = len(values_array) + 1 )
        print("===========================\n")
    
    plt.show()


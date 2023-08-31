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


# Define line width
lw = 2


# Choose whether to plot real values or median values
plot_median = False
batch_size = 1

plot_dir = 'Figures/tuning/'
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)



def plot_tuning(settings, value, label = None):
    
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
    
    if label is None:
        label = str(value)

    # Plot reward evolution
    ax.plot(steps_array, data_array, color=color_array[ values_array.index( value ) ], linewidth=lw, label=label)




if __name__ == "__main__":

    color_array = ['tab:blue',
                   'tab:orange',
                   'tab:green',
                   'tab:cyan',
                   'tab:olive',
                   'tab:pink']


    # Read default data
    file_name = "./tuning/default/sac_default.txt"
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

    # Read chosen data
    file_name = "./tuning/default/sac_chosen.txt"
    chosen_data = np.genfromtxt(file_name)
    chosen_data = chosen_data[1:-1]

    # Create array with number of steps
    chosen_steps_array = np.arange(0, n_steps + 1, n_steps / (len(chosen_data) - 1) )

    # Create array of median values
    if plot_median:
        
        median_array = np.array( chosen_data )

        for i in range(len(chosen_data)):

            index_min = max(0, i + 1 - batch_size)
            median_array[i] = np.median( chosen_data[index_min : i+1] )
        
        chosen_data = median_array

    
    labels = {'reward_scaler' : ['Reward x 0.1',
                                 'Reward x 10',
                                 'Reward x 20',
                                 'Reward x 50'
                                 ],
             'encoder' :        ['tanh()',
                                 'sigmoid',
                                 'Dense ReLU',
                                 'Dense tanh()'
                                 ],
             'tau' : [r'1$\times 10^{-4}$',
                                r'5$\times 10^{-3}$',
                                 r'1$\times 10^{-2}$',
                                 r'1$\times 10^{-1}$'
                                 ],
             'critic_learning_rate' : [r'3$\times 10^{-6}$',
                                r'3$\times 10^{-3}$',
                                 r'3$\times 10^{-2}$'
                                 ],
             'gamma' : ['0.5',
                                 '0.9',
                                 '0.95'
                                 ],
             'batch_size' : ['64',
                                 '128',
                                 '512'
                                 ],
                                 }
    
    default_label = ['256',
                     '0.99',
                     r'3$\times10^{-4}$',
                     r'1$\times 10^{-3}$',
                     'ReLU',
                     'Reward x 1'
                     ]
    label_index = 0


    # Loop over available settings
    for settings in settings_dict:
        print(settings)


        label_array = labels.get(settings)

        # Create Figure
        fig, ax = plt.subplots(figsize=(5.25,4))

        # ax.set_title('Effect of ' + settings + ' on reward evolution')

        values_array = settings_dict.get(settings)

        # Loop over selected values
        for value in values_array:

            plot_tuning(settings, value, label_array[ values_array.index(value) ])

        # Plot default results
        ax.plot(default_steps_array, default_data, color='tab:red', linewidth=lw, label=default_label[label_index])
        label_index += 1
        # Plot tuned results
        # ax.plot(chosen_steps_array, chosen_data, color='k', label='tuned')
        
        # ax.legend(loc = 'best')
        fig.legend(loc = 'outside lower center', #loc='upper center', bbox_to_anchor=(0.5, -0.15),
            #ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
            fancybox=True, shadow=True, ncol = 2)# len(values_array) + 1 )
        fig.tight_layout()
        fig.subplots_adjust(top=0.988,
                            bottom=0.344,
                            left=0.105,
                            right=0.979,
                            hspace=0.2,
                            wspace=0.2)
        ax.set_ylabel('Reward [-]')
        ax.set_xlabel('Steps [-]')
        print("===========================\n")
    

    fig, ax = plt.subplots(figsize=(10.5,3))
    # Plot default results
    ax.plot(default_steps_array, default_data, color='tab:blue', linewidth=lw, label='default')
    # Plot tuned results
    ax.plot(chosen_steps_array, chosen_data, color='tab:orange', linewidth=lw, label='tuned')

    values_array = settings_dict.get('reward_scaler')
    multiplier20 = values_array[2]
    # plot_tuning('reward_scaler', multiplier20, label='Reward Scale x20')
    fig.legend(loc = 'outside lower center', #loc='upper center', bbox_to_anchor=(0.5, -0.15),
            #ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25),
            fancybox=True, shadow=True, ncol = len(values_array) + 1 )
    fig.tight_layout()
    fig.subplots_adjust(top=0.948,
                        bottom=0.339,
                        left=0.06,
                        right=0.979,
                        hspace=0.2,
                        wspace=0.2)
    ax.grid()
    ax.set_ylabel('Reward [-]')
    ax.set_xlabel('Steps [-]')
    plt.show()
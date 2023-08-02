# General imports
import numpy as np
import torch
import os
import d3rlpy
import logging

# d3rlpy specifics
# from d3rlpy.dataset import ReplayBuffer, BufferProtocol
# from d3rlpy.metrics import EnvironmentEvaluator
from d3rlpy.models.optimizers import SGDFactory, AdamFactory
from d3rlpy.models.encoders import DefaultEncoderFactory, DenseEncoderFactory
from d3rlpy.models.q_functions import MeanQFunctionFactory, QRQFunctionFactory
from d3rlpy.preprocessing import MultiplyRewardScaler

# File imports
import Inputs as inp
# from Model_utilities import coordinates_in, coordinates_out
from Car_class import CarEnvironment
# from nn_funcs import CustomEncoderFactory


# List of tunable settings, with respective test values

settings_dict = {# 'batch_size' : [64, 128, 512], # done
                # 'gamma' : [0.5, 0.75, 0.9, 0.95], # done
                # 'actor_learning_rate' : [3e-6, 3e-5, 1e-4, 5e-4, 3e-3, 3e-2], # done
                # 'critic_learning_rate' : [3e-6, 3e-5, 1e-4, 5e-4, 3e-3, 3e-2], # done
                # 'temp_learning_rate' : [3e-6, 3e-5, 1e-4, 5e-4, 3e-3, 3e-2], # done
                # 'tau' : [1e-4, 5e-3, 1e-2, 1e-1], # done
                # 'n_critics' : [1, 3, 4, 5], # done
                # # 'n_steps' : [],
                # 'limit' : [1000, 5000, 10000, 25000, 100000], # done
                # 'n_steps_per_epoch' : [100, 500, 2000], # done
                # 'update_interval' : [2, 5, 10], # done
                # 'update_start_step' : [100, 500, 2000], # done
                # 'random_steps' : [100, 500, 2000], # done
                # 'initial_temperature' : [2.0, 5.0, 10.0], # done
                # # 'eval_env' : [EnvironmentEvaluator(env_default)], # error
                # # 'buffer' : [ReplayBuffer(BufferProtocol, cache_size=default_settings.get('limit'), env=env_default)], # error
                # 'optim_factory' : [AdamFactory(amsgrad=True), # done
                #                 SGDFactory()], # done
                #                 # RMSpropFactory()], # error
                # 'encoder' : [DefaultEncoderFactory(activation='tanh'), # done
                #             DefaultEncoderFactory(activation='swish'), # done
                #             DefaultEncoderFactory(activation='none'), # done
                #             # PixelEncoderFactory(), # error
                #             # VectorEncoderFactory(), # error
                #             DenseEncoderFactory()], # done
                # 'q_func' : [QRQFunctionFactory()], # done
                            # IQNQFunctionFactory()], # extremely slow --> cancelled
                'reward_scaler' : [# MinMaxRewardScaler(), # error
                                # StandardRewardScaler(), # error
                                # ReturnBasedRewardScaler(), # error
                                #MultiplyRewardScaler(0.1), # done
                                #MultiplyRewardScaler(10), # done
                                #MultiplyRewardScaler(20), # done
                                MultiplyRewardScaler(50), # TODO
                                MultiplyRewardScaler(100)]} # TODO


env_default = CarEnvironment()


# DEFINE DEFAULT SETTINGS
default_settings = {'batch_size' : 256,
                    'gamma' : 0.99,
                    'actor_learning_rate' : 3e-4,
                    'critic_learning_rate' : 3e-4,
                    'temp_learning_rate' : 3e-4,
                    'tau' : 0.001,
                    'n_critics' : 2,
                    'n_steps' : 50000, # 100000
                    'limit' : 50000,
                    'n_steps_per_epoch' : 1000,
                    'update_interval' : 1,
                    'update_start_step' : 1000,
                    'random_steps' : 1000,
                    'initial_temperature' : 1.0,
                    'eval_env' : env_default,
                    'optim_factory' : AdamFactory(),
                    'encoder' : DefaultEncoderFactory(),
                    'q_func' : MeanQFunctionFactory(),
                    'reward_scaler' : None}



# DEFINE "OPTIMAL" SETTINGS
chosen_settings = {'batch_size' : 64,
                    'gamma' : 0.90,
                    'actor_learning_rate' : 5e-4,
                    'critic_learning_rate' : 3e-3,
                    'temp_learning_rate' : 5e-4,
                    'tau' : 0.1,
                    'n_critics' : 4,
                    'n_steps' : 50000,
                    'limit' : 2 * 50000, # 2 * n_steps
                    'n_steps_per_epoch' : 100,
                    'update_interval' : 2,
                    'update_start_step' : 500,
                    'random_steps' : 500,
                    'initial_temperature' : 2.0,
                    'eval_env' : env_default,
                    'optim_factory' : AdamFactory(),
                    'encoder' : DenseEncoderFactory(),
                    'q_func' : QRQFunctionFactory()(),
                    'reward_scaler' : MultiplyRewardScaler(10)}


log_file = 'run_log.txt'

tuning_dir = 'tuning/'
if not os.path.exists(tuning_dir):
    os.makedirs(tuning_dir)


def tune_settings(setting_name = 'default', value = 0):

    # Print for information
    print("Running " + setting_name + ". Value: " + str(value))

    # Define (and create, if necessary) current directory
    current_dir = tuning_dir + setting_name + '/'
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
    
    # Initialise current settings
    current_settings = dict(default_settings)

    # Change current setting if it is not 'default' and not 'buffer'
    if setting_name != 'default' and setting_name != 'buffer':
        current_settings[setting_name] = value

    # setup algorithm
    sac = d3rlpy.algos.SACConfig(
        batch_size=current_settings.get('batch_size'),
        gamma=current_settings.get('gamma'),
        actor_learning_rate=current_settings.get('actor_learning_rate'),
        critic_learning_rate=current_settings.get('critic_learning_rate'),
        temp_learning_rate=current_settings.get('temp_learning_rate'),
        tau=current_settings.get('tau'),
        n_critics=current_settings.get('n_critics'),
        initial_temperature=current_settings.get('initial_temperature'),
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(),
        actor_optim_factory=current_settings.get('optim_factory'),
        critic_optim_factory=current_settings.get('optim_factory'),
        temp_optim_factory=current_settings.get('optim_factory'),
        actor_encoder_factory=current_settings.get('encoder'),
        q_func_factory=current_settings.get('q_func'),
        reward_scaler=current_settings.get('reward_scaler'),
    ).create(device=False) # device = False means CPU. Change to True when using GPU


    # multi-step transition sampling
    transition_picker = d3rlpy.dataset.MultiStepTransitionPicker(
        n_steps=current_settings.get('n_steps'),
        gamma=current_settings.get('gamma'),
    )

    # replay buffer for experience replay
    buffer = d3rlpy.dataset.create_fifo_replay_buffer(
        limit=current_settings.get('limit'),
        env=env_default,
        transition_picker=transition_picker,
    )

    if setting_name == 'buffer':
        buffer = value
    
    env = CarEnvironment(log_file = current_dir + str(value) + '.txt' )
    d3rlpy.envs.seed_env(env, inp.seed)

    # start training
    sac.fit_online(
        env,
        buffer,
        eval_env=current_settings.get('eval_env'),
        n_steps=current_settings.get('n_steps'),
        n_steps_per_epoch=current_settings.get('n_steps_per_epoch'),
        update_interval=current_settings.get('update_interval'),
        update_start_step=current_settings.get('update_start_step'),
        random_steps=current_settings.get('random_steps'),
        save_interval=current_settings.get('n_steps')
    )

    # Save trained agent
    sac.save(current_dir + str(value) + '.d3')



if __name__ == "__main__":

    # For reproducibility
    np.random.seed(inp.seed)
    d3rlpy.seed(inp.seed)
    torch.manual_seed(inp.seed)

    log_file = 'error_log.txt'

    # Delete the log file if it already exists
    if os.path.exists(log_file):
        os.remove(log_file)
    
    error_logger = logging.getLogger(log_file)
    error_handler = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(message)s')
    error_handler.setFormatter(formatter)
    error_logger.addHandler(error_handler)
    error_logger.setLevel(logging.INFO)

    error_msg = []
    
    # Run default settings
    tune_settings()

    # Loop over available settings
    for settings in settings_dict:
        
        print("\n==================")
        values_array = settings_dict.get(settings)

        # Loop over selected values
        for value in values_array:

            # Run current setting + value combination
            try:
                tune_settings(settings, value) # this saves the result in a log file
            except:
                current_error = "Error when using " + str(value) + " in " + settings
                error_msg.append(current_error)
                error_logger.info(current_error)

    print(error_msg)

###########################################################################################
#
# OPTIMISERS
# SGDFactory(dampening=0.0, momentum=0.0, weight_decay=0.0, nesterov=False)
# AdamFactory(betas=(0.9, 0.999), eps=1e-08, weight_decay=0.0, amsgrad=False)
# RMSpropFactory(alpha=0.95, eps=1e-02, momentum=0.0, weight_decay=0.0, centered=True)
# https://d3rlpy.readthedocs.io/en/v2.0.4/references/optimizers.html
#
# NN ENCORDERS
# DefaultEncoderFactory(activation='relu', use_batch_norm=False, dropout_rate=None)
# https://d3rlpy.readthedocs.io/en/v2.0.4/references/network_architectures.html
#
# Q FUNCTION
# MeanQFunctionFactory(share_encoder=False)
# https://d3rlpy.readthedocs.io/en/v2.0.4/references/q_functions.html
# Possible activations: 'relu', 'tanh', 'swish', 'none'
#
#
# NN: https://d3rlpy.readthedocs.io/en/v2.0.4/tutorials/customize_neural_network.html
# reward scaling: https://d3rlpy.readthedocs.io/en/v2.0.4/references/preprocessing.html
#
###########################################################################################

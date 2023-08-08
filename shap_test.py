import numpy as np
import yaml
import torch
import d3rlpy
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size':12})
import torchvision.models as models

import Inputs as inp
from Model_utilities import coordinates_in, coordinates_out, coordinates
from Car_class import CarEnvironment


idx_min = 0
idx_max = 750



if __name__ == "__main__":

    # For reproducibility
    np.random.seed(inp.seed)
    d3rlpy.seed(inp.seed)
    torch.manual_seed(inp.seed)


    ################
    # SHAP STUFF ###
    ################
    sac = d3rlpy.load_learnable("reward_test/'constant_action', 'max_acc'.d3", device=None)

    env = CarEnvironment()


    actor_model = sac._impl.policy

    data_set = np.load('state_example.npy')


    # data_set = torch.from_numpy(data_set)

    e = shap.DeepExplainer(actor_model, torch.from_numpy(data_set))
    shap_values = e.shap_values(torch.from_numpy(data_set[idx_min:idx_max]))

    feature_names = ['velocity norm',
                     'tang. acc',
                     'centrip. acc',
                     'delta heading',
                     'curvature 1',
                     'curvature 2',
                     'curvature 3',
                     'curvature 4',
                     'curvature 5',
                     'curvature 6',
                     'curvature 7',
                     'curvature 8',
                     'curvature 9',
                     'curvature 10',
                     'LIDAR -90',
                     'LIDAR -45',
                     'LIDAR -30',
                     'LIDAR -15',
                     'LIDAR 0',
                     'LIDAR 15',
                     'LIDAR 30',
                     'LIDAR 45',
                     'LIDAR 90',
                     'track lim.'
                    ]


    shap_values_throttle = np.array(shap_values[0])
    shap_values_wheel = np.array(shap_values[1])

    # features = np.array([data_set[300]])

    # features = data_set[300,:]


    shap.summary_plot(shap_values_throttle, features = data_set[idx_min:idx_max], feature_names=feature_names)
    shap.summary_plot(shap_values_wheel, features = data_set[idx_min:idx_max], feature_names=feature_names)

    # # Wrap the actor_model with a SHAP explainer
    # explainer = shap.Explainer(actor_model, data_set)# np.zeros((1, len(data_set))))


    # shap_values = explainer.shap_values(torch.from_numpy(data_set))


import numpy
import yaml
from d3rlpy.algos.sac import SAC
from d3rlpy.online.buffers import ReplayBuffer

print("Testing yaml")

with open('sac_inputs.yml') as f:
    sac_inputs = yaml.load(f, Loader=yaml.FullLoader)


print("sac_inputs['agent']")
print(sac_inputs['agent'])
print("*sac_inputs['agent']")
print(*sac_inputs['agent'])
print("sac_inputs['agent']['actor_learning_rate']")
print(sac_inputs['agent']['actor_learning_rate'])
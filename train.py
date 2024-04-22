import gym
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'GTK3Agg', 'macosx', 'TkAgg'
import matplotlib.pyplot as plt
import math

from AE.util import *

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
args = arg_parse()
config = configuration(args)
device = check_gpu(args)

from gym.envs.registration import register, registry

# First, unregister the existing environment to avoid a registration error
if 'CustomAnt-v3' in registry.env_specs:
    del registry.env_specs['CustomAnt-v3']

# Register the environment with a new max_episode_steps value
register(
    id='CustomAnt-v3',
    entry_point='ant_env:CustomAntEnv',
    max_episode_steps=100000,  # Set to desired value
)

env = gym.make('CustomAnt-v3', terminate_when_unhealthy=True, healthy_z_range=(0.3,5), ctrl_cost_weight=0, contact_cost_weight=0, healthy_reward=0)
observation = env.reset()
joints = config["joints"]

from AE.cvae import AugmentedConditionalVariationalAutoencoder

model_kwargs = {'input_size': config['input_size']*joints,
                    'latent_size': config['latent_size'],
                    'encoder_layer_sizes': config['encoder_layer_sizes'],
                    'decoder_layer_sizes': config['decoder_layer_sizes'],
                    'task_layer_sizes': config["task_layer_sizes"],
                    'condition_size': config["condition_size"]}

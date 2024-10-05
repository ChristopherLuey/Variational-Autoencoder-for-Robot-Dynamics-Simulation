import gym
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'GTK3Agg', 'macosx', 'TkAgg'
import matplotlib.pyplot as plt
from AE.util import *
from generate_data import *

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

args = arg_parse()
config = configuration(args)
device = check_gpu(args)
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

# Load control sequences from the database within a specified range
data = query_sequences_near_position(x_target=0.1, y_target=0, x_tolerance=0.05, y_tolerance=0.05).to(device)

# Split data into training and validation sets
train_size = int(0.8 * len(data))
val_size = len(data) - train_size
train_data, val_data = torch.utils.data.random_split(data, [train_size, val_size])
loss = autoencoder.train_model(_combined_control_seq[1:], _combined_target_value_tensor[1:], _combined_direction[1:], combined_latent_space[1:])

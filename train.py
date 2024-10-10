import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'GTK3Agg', 'macosx', 'TkAgg'
import matplotlib.pyplot as plt
from AE.util import *
from AE.simple_autoencoder import SimpleAutoencoder
from generate_data import *
from data_processing import *
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

args = arg_parse()
config = configuration(args, 'config/simpleAE.yaml')
device = check_gpu(args)
joints = config["joints"]

from AE.simple_autoencoder import BasicAutoencoder
import os
from datetime import datetime

model_kwargs = {
    'input_size': config['input_size'] * joints,
    'latent_size': config['latent_size'],
    'encoder_layer_sizes': config['encoder_layer_sizes'],
    'decoder_layer_sizes': config['decoder_layer_sizes'],
    'condition_size': config["condition_size"]
}

# Load control sequences from the database with a reward of 10.0 plus minus 0.5
data = query_sequences_near_reward(reward=10.0, tol=0.5)
controls = [torch.tensor(seq['control'], dtype=torch.float32) for seq in data]
conditions = [torch.zeros(config["condition_size"], dtype=torch.float32) for _ in data]  # Condition is a tensor of [0]

# Ensure the input size matches the model's expected input
expected_input_size = config['input_size'] * joints
if controls[0].shape[0] != expected_input_size:
    raise ValueError(f"Input data size ({controls[0].shape[0]}) does not match the expected input size ({expected_input_size})")

print(f"Data shape: {len(controls), controls[0].shape}")  # Print the shape for verification

# Split data into training and validation sets
train_controls, val_controls, train_conditions, val_conditions = split_data(controls, conditions, train_ratio=0.8, device=device)

# Create results directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = os.path.join('results', timestamp)
os.makedirs(results_dir, exist_ok=True)

autoencoder = SimpleAutoencoder(**model_kwargs).to(device)
train_losses, val_losses = train_autoencoder(autoencoder, train_controls, train_conditions, val_controls, val_conditions, device, epochs=100, visualize_reconstruction=True, results_dir=results_dir)

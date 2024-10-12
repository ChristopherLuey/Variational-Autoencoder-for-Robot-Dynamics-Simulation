# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'GTK3Agg', 'macosx', 'TkAgg'
import matplotlib.pyplot as plt
from AE.util import *
from generate_data import *
from data_processing import *
from AE.simple_autoencoder import BasicAutoencoder
import os
from datetime import datetime
import yaml

args = arg_parse()
config = configuration(args, 'config/simpleAE.yaml')
device = check_gpu(args)

model_kwargs = {
    'joints': config['joints'],
    'timesteps': config['timesteps'],
    'latent_size': config['latent_size'],
    'layer_sizes': config['layer_sizes'],
    'condition_size': config["condition_size"]
}

# Load control sequences from the database with a reward of 10.0 plus minus 0.5
data = query_sequences_near_reward(reward=config['reward'], tol=config['reward_tol'])
print(len(data))
controls = [torch.tensor(seq['control'], dtype=torch.float32).view(config['timesteps'], config['joints']) for seq in data]
controls = torch.stack(controls)  # Stack into a single tensor of shape [batch size, timesteps, joints]
conditions = torch.zeros(len(data), 1, dtype=torch.float32)  # Condition is a tensor of shape [batch_size, 1]

print(f"Data shape: {controls.shape}")  # Print the shape for verification

# Split data into training and validation sets
train_controls, val_controls, train_conditions, val_conditions = split_data(controls, conditions, train_ratio=0.8, device=device)

# Create results directory with timestamp
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = os.path.join('results', timestamp)
os.makedirs(results_dir, exist_ok=True)

autoencoder = BasicAutoencoder(**model_kwargs).to(device)
train_losses, val_losses = autoencoder.train_model(
    train_controls, 
    val_controls, 
    device, 
    epochs=config['epochs'], 
    visualize_reconstruction=True, 
    results_dir=results_dir, 
    condition=None
)

# Save the yaml file used to results dir
config_path = os.path.join(results_dir, 'training_config.yaml')
with open(config_path, 'w') as file:
    yaml.dump(config, file)

print(f"Configuration saved to {config_path}")

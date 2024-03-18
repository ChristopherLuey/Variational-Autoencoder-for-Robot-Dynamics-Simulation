import gym
import numpy as np
import argparse
import yaml
import torch
import matplotlib
import tkinter
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'GTK3Agg', 'macosx'
import matplotlib.pyplot as plt

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',   type=str,   default='AntEnv_v3', help="Only AntEnv_v3")
    parser.add_argument('--method', type=str, default='AE', help='maxdiff, mppi, or sac_orig')
    parser.add_argument('--seed', type=int, default=666, help='any positive integer')
    parser.add_argument('--log', dest='log', action='store_true',help='save data for experiment')
    parser.add_argument('--no_log', dest='log', action='store_false',help='run test without saving')
    parser.set_defaults(log=True)
    parser.add_argument('--render', dest='render', action='store_true',help='show visualization while running')
    parser.add_argument('--no_render', dest='render', action='store_false',help='run offline / without showing plots')
    parser.set_defaults(render=False)
    parser.add_argument('--cpu', dest='cpu', action='store_true',help='only use CPU')
    parser.add_argument('--no_cpu', dest='cpu', action='store_false',help='try to use GPU if available')
    parser.set_defaults(cpu=False)
    parser.add_argument('--mod_weight', type=str, default='None',help="[gym envs only] load alternate xml file for enviroment (e.g. 'light' or 'orig' for swimmer)")
    parser.add_argument('--frames_before_learning', type=int, default=0,help="if specified, number of frames to collect before starting to learn (otherwise, batch size is used)")
    parser.add_argument('--random_actions', type=int, default=0,help="if specified, number random frames to collect before starting to use the policy")
    parser.add_argument('--base_dir',   type=str,   default='./results/',help="where to save the data (if log=True)")
    parser.add_argument('--singleshot', dest='singleshot', action='store_true',help="don't reset for each epoch and run all steps from initial condition")
    parser.set_defaults(singleshot=False)
    args = parser.parse_args()
    return args
args = arg_parse()

def configuration():
    # load config
    config_path = f'./config/AE.yaml'

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = config_dict[args.env]

    return config
config = configuration()

device ='cpu'
if not args.cpu:
    if torch.cuda.is_available():
        torch.set_num_threads(1)
        device  = 'cuda:0'
        print('Using GPU Accel')
    else:
        args.cpu = True

env = gym.make('Ant-v3', terminate_when_unhealthy=True, healthy_z_range=(0.4,3))
observation = env.reset()
print(env.action_space.low)  # Minimum valid values
print(env.action_space.high)  # Maximum valid values
joints = 8

from AE.ae import AugmentedAutoencoder, TaskNetwork

model_kwargs = {'input_size': config['input_size']*joints, 
                    'latent_size': config['latent_size'],
                    'encoder_layer_sizes': config['encoder_layer_sizes'],
                    'decoder_layer_sizes': config['decoder_layer_sizes']}

autoencoder = AugmentedAutoencoder(**model_kwargs).to(device)
task_network = TaskNetwork(config["latent_size"]).to(device)
samples = config["samples"]
perturbation_strength = config["perturbation_strength"]
control_sequence_time = config["input_size"]
control_sequence_length = config["input_size"]*joints
target_value_tensor = torch.tensor([0], dtype=torch.float32, device=device)

losses = []
obj_loss = []
new_control_seq_values = []

new_control_seq = (2*torch.rand(control_sequence_length)-1).to(device)

render_frame = 1

def acquire_new_data(last_control_seq):
    """
    Acquire new data for training the autoencoder.

    Returns:
    - The new control sequence with the lowest autoencoder objective function value after sampling.
    """
    lowest_loss = float('inf')
    best_seq = None

    for _ in range(samples):
        # Perturb the control sequence by a random vector
        perturbation = (torch.rand_like(last_control_seq)-0.5) * perturbation_strength
        perturbed_seq = torch.clamp(last_control_seq + perturbation, min=-1, max=1).to(device)

        # Evaluate the perturbed sequence without updating the autoencoder weights
        _, loss = autoencoder.evaluate(perturbed_seq.unsqueeze(0), target_value_tensor)

        if loss < lowest_loss:
            lowest_loss = loss
            best_seq = perturbed_seq

    return best_seq  


epochs = 1000
for _ in range(epochs):

    if epochs % render_frame == 0:  # Conditional render to reduce computation load
        env.render()


    new_control_seq = acquire_new_data(new_control_seq)
    total_reward = 0
    _new_control_seq = new_control_seq.view(control_sequence_time, joints)

    for i in range(control_sequence_time):
        action = _new_control_seq[i].cpu().numpy()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        print(reward)

    target_value_tensor = torch.tensor([total_reward/control_sequence_time], dtype=torch.float32, device=device)
    loss = autoencoder.train_model(new_control_seq.unsqueeze(0), target_value_tensor)
    losses.append(loss[2])
    obj_loss.append(loss[1])
    new_control_seq_values.append(new_control_seq.to("cpu").tolist()) 

    # Check if the episode is done
    if done:
        observation = env.reset()

env.close()

fig, axes = plt.subplots(1, 3, figsize=(12, 6))

# Plot training loss on the first subplot
axes[0].plot(range(1, epochs + 1), losses, label="Training Loss")
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Model Loss Over Time')
axes[0].legend()

# Plot objective loss on the second subplot
axes[1].plot(range(1, epochs + 1), obj_loss, label="Objective Loss")
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss')
axes[1].set_title('Objective Loss Over Time')
axes[1].legend()

# Plot new control sequence values on the third subplot
new_control_seq_values_transposed = list(zip(*new_control_seq_values))
for seq_index, seq_values in enumerate(new_control_seq_values_transposed):
    axes[2].plot(range(1, epochs + 1), seq_values, label=f"Seq {seq_index + 1}")
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Value')
axes[2].set_title('New Control Seq Values Over Time')
#axes[2].legend()

plt.tight_layout()
plt.show()
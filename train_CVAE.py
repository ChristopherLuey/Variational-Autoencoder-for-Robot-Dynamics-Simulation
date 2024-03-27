import gym
import numpy as np
import argparse
import yaml
import torch
import matplotlib
import tkinter
matplotlib.use('TkAgg')  # Or 'Qt5Agg', 'GTK3Agg', 'macosx', 'TkAgg'
import matplotlib.pyplot as plt
import math

# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


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


env = gym.make('Ant-v3', terminate_when_unhealthy=True, healthy_z_range=(0.3,5), ctrl_cost_weight=0, contact_cost_weight=0, healthy_reward=0)
observation = env.reset()
joints = 8

from AE.cvae import AugmentedConditionalVariationalAutoencoder

model_kwargs = {'input_size': config['input_size']*joints, 
                    'latent_size': config['latent_size'],
                    'encoder_layer_sizes': config['encoder_layer_sizes'],
                    'decoder_layer_sizes': config['decoder_layer_sizes'],
                    'task_layer_sizes': config["task_layer_sizes"],
                    'condition_size': config["condition_size"]}

autoencoder = AugmentedConditionalVariationalAutoencoder(**model_kwargs).to(device)
samples = config["samples"]
perturbation_strength = config["perturbation_strength"]
control_sequence_time = config["input_size"]
control_sequence_length = config["input_size"]*joints
target_value_tensor = torch.tensor([0], dtype=torch.float32, device=device)

losses = []
obj_loss = []
new_control_seq_values = []
task_reward_list = []
encoded_list = []

new_control_seq = (2*torch.rand(control_sequence_length)-1).to(device)
new_control_seq.requires_grad_(True)


render_frame = 1
direction = torch.tensor([0.0], dtype=torch.float32, device=device)

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
        _, loss = autoencoder.evaluate(perturbed_seq.unsqueeze(0), target_value_tensor, direction)

        if loss < lowest_loss:
            lowest_loss = loss
            best_seq = perturbed_seq

    return best_seq.detach()  


# def acquire_new_data_grad_descent(last_control_seq):
#     """
#     Acquire new data for training the autoencoder.
#     Returns:
#     - The new control sequence with the lowest autoencoder objective function value after sampling.
#     """
#     # Evaluate the initial sequence without updating the autoencoder weights
#     _, loss_init = autoencoder.evaluate(last_control_seq.unsqueeze(0), target_value_tensor)
#     x = last_control_seq
#     k = 0 # for counting iterations in the while loop
#     alpha = .2 # learning rate for SGD (stochastic gradiant descent)
#     gradient = 1 # initialize value for first loop
#     while gradient<.1:
#         # initialize array for gradient
#         grad = np.zeros(samples)
#         # sample a batch of data points
#         for n in range(samples):
#             # Perturb the control sequence by a random vector
#             perturbation = (torch.rand_like(x)-0.5) * perturbation_strength
#             perturbed_seq = torch.clamp(x + perturbation, min=0, max=1)
#             # Evaluate the perturbed sequence without updating the autoencoder weights
#             _, loss = autoencoder.evaluate(perturbed_seq.unsqueeze(0), target_value_tensor)
#             grad[n] = loss-loss_init
#         # Average samples for gradient at x
#         gradient = np.mean(grad)
#         x = x - alpha * gradient
#         k += 1
#         if k>20:
#             break
#     return x


def acquire_new_data_sgd(last_control_seq, lr=0.01, max_iterations=10, threshold=0.01):
    """
    Acquire new data for training the autoencoder using stochastic gradient descent.
    
    Args:
    - last_control_seq (torch.Tensor): The last control sequence with requires_grad set to True.
    - lr (float): Learning rate for SGD.
    - max_iterations (int): Maximum number of iterations for the SGD optimization.
    - threshold (float): Threshold for gradient norm for early stopping.
    
    Returns:
    - torch.Tensor: The optimized control sequence.
    """
    # Ensure last_control_seq requires gradients
    last_control_seq = last_control_seq.clone().detach().requires_grad_(True).to(device)
    optimizer = torch.optim.SGD([last_control_seq], lr=lr)
    
    for iteration in range(max_iterations):
        optimizer.zero_grad()

        # Use the evaluate function to compute the loss
        _, loss = autoencoder.evaluate_gradient(last_control_seq.unsqueeze(0), target_value_tensor, direction)
        
        # Backward pass to compute gradients
        loss.backward()

        # Update last_control_seq based on gradients
        optimizer.step()

        # Check gradient norm for early stopping
        grad_norm = last_control_seq.grad.norm()
        if grad_norm < threshold:
            # print(f"Early stopping at iteration {iteration} due to low gradient norm.")
            break

    return last_control_seq.detach()


clear = np.array([0,0,0,0,0,0,0,0])
epochs = 500
xvel = 0
for _ in range(epochs):

    # if epochs % render_frame == 0:  # Conditional render to reduce computation load
    #     env.render()


    # new_control_seq = acquire_new_data(new_control_seq)
    # if (target_value_tensor < 0):
    #     target_value_tensor = -target_value_tensor
    # else:
    #     target_value_tensor = target_value_tensor
    # _, loss = autoencoder.evaluate_gradient(new_control_seq.unsqueeze(0), target_value_tensor,direction)
    # print("prior control seq",new_control_seq,loss)

    new_control_seq = acquire_new_data_sgd(new_control_seq)

    # _, loss = autoencoder.evaluate_gradient(new_control_seq.unsqueeze(0), target_value_tensor,direction)
    # print("new control seq",new_control_seq, loss)
    total_reward = 0
    _new_control_seq = new_control_seq.view(control_sequence_time, joints)

    xvel = 0

    for i in range(control_sequence_time):
        action = _new_control_seq[i].cpu().numpy()
        r = 0
        t = 10
        for j in range(t):
            observation, reward, done, info = env.step(action)
            env.render()
            # r += (info["x_velocity"]- np.abs(info["y_velocity"]))/50
            r += info["x_velocity"]/50
            # xvel += info["x_velocity"]/50
        total_reward += r

    x,y,z,w = observation[1:5]
    direction.fill_(math.atan2(2.0 * (w * x + y * z), 1 - 2 * (x**2 + z**2)))
    #direction = torch.tensor([direction], dtype=torch.float32, device=device)
    #task_reward_list.append(total_reward)
        
    task_reward_list.append(total_reward)
    print("Epoch {}: Reward {} {}".format(_, total_reward, xvel))

    target_value_tensor = torch.tensor([total_reward], dtype=torch.float32, device=device)
    loss = autoencoder.train_model(new_control_seq.unsqueeze(0), target_value_tensor, direction)
    losses.append(loss[2])
    obj_loss.append(loss[1])
    encoded_list.append(loss[3].to("cpu").tolist())
    new_control_seq_values.append(new_control_seq.to("cpu").tolist()) 
    # for i in range(5):
    #     observation, reward, done, info = env.step(clear)
    #     env.render()
    # observation = env.reset()
  

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


plt.plot(range(1, epochs + 1),task_reward_list)
plt.show()

encoded_list_transposed = list(zip(*encoded_list))

# Create a new figure for the plot
plt.figure(figsize=(10, 6))

# Plot each sequence in the encoded list
for seq_index, seq_values in enumerate(encoded_list_transposed):
    plt.plot(range(1, len(seq_values) + 1), seq_values, label=f"Seq {seq_index + 1}")

plt.xlabel('Epoch')
plt.ylabel('Encoded Value')
plt.title('Encoded Sequence Values Over Time')
plt.legend()
plt.show()
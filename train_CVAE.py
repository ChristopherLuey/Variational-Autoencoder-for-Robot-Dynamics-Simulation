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

autoencoder = AugmentedConditionalVariationalAutoencoder(**model_kwargs).to(device)
samples = config["samples"]
perturbation_strength = config["perturbation_strength"]
control_sequence_time = config["input_size"]
control_sequence_length = config["input_size"]*joints
target_value_tensor = torch.tensor([0.0], dtype=torch.float32, device=device)
combined_target_value_tensor = target_value_tensor.clone().unsqueeze(0)
target_value_tensor_previous = 0
combined_latent_space = torch.zeros(config['latent_size'], dtype=torch.float32, device="cuda:0").unsqueeze(0)


losses = []
obj_loss = []
new_control_seq_values = []
task_reward_list = []
encoded_list = []
decoded_list = []
direction_list = []
variation_list = []
expected_reward_list = []

new_control_seq = (2*torch.rand(control_sequence_length)-1).to(device)
combined_control_seq = new_control_seq.clone().unsqueeze(0)

new_control_seq.requires_grad_(True)
render_frame = 1
direction = torch.tensor([0.0], dtype=torch.float32, device=device)
combined_direction = direction.clone().unsqueeze(0)
direction_task = torch.tensor([0.0], dtype=torch.float32, device=device)
direction_prev = torch.tensor([0.0], dtype=torch.float32, device=device)


def acquire_new_data(last_control_seq, autoencoder, target_value_tensor, direction):
    """
    Acquire new data for training the autoencoder.

    Returns:
    - The new control sequence with the lowest autoencoder objective function value after sampling.
    """
    lowest_loss = float('inf')
    best_seq = None
    prediction = None
    target_value_tensor = target_value_tensor.clone().detach()


    if (target_value_tensor.item() < 0):
        target_value_tensor.fill_(-target_value_tensor.item())

    for _ in range(samples):
        # Perturb the control sequence by a random vector
        perturbation = (torch.rand_like(last_control_seq)-0.5) * perturbation_strength
        perturbed_seq = torch.clamp(last_control_seq + perturbation, min=-1, max=1).to(device)

        # Evaluate the perturbed sequence without updating the autoencoder weights
        _prediction, loss, reconstruction_loss, task_loss = autoencoder.evaluate(perturbed_seq, target_value_tensor, direction)

        if loss < lowest_loss:
            lowest_loss = loss
            best_seq = perturbed_seq
            prediction = _prediction
    
    print("Lowest Loss:", lowest_loss)

    return prediction, best_seq.detach(), lowest_loss


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

def acquire_new_data_sgd(last_control_seq, autoencoder, target_value_tensor, direction, learning_rate=1e-3):
    """
    Acquire new data for training the autoencoder using Stochastic Gradient Descent (SGD).

    Parameters:
    - last_control_seq: The last control sequence.
    - autoencoder: The autoencoder model.
    - target_value_tensor: The target value for the autoencoder to achieve.
    - direction: The direction for evaluating the autoencoder.
    - learning_rate: The learning rate for SGD.
    - num_iterations: The number of iterations to perform SGD.

    Returns:
    - The new control sequence with the lowest autoencoder objective function value after applying SGD.
    """
    # Ensure last_control_seq requires gradient for optimization
    control_seq = last_control_seq.clone().detach().requires_grad_(True)
    #control_seq.grad.zero_()
    lowest_loss = float('inf')
    best_seq = control_seq.clone().detach()

    for _ in range(samples):
        # Evaluate the control sequence
        _, loss, reconstruction_loss, task_loss = autoencoder.evaluate_gradient(control_seq, target_value_tensor, direction)

        if loss.item() < lowest_loss:
            lowest_loss = loss.item()
            best_seq = control_seq.clone().detach()

        # Compute gradients
        loss.backward()

        # Update the control sequence with gradient descent
        with torch.no_grad():
            control_seq -= learning_rate * control_seq.grad
            # Optionally: Apply constraints such as clamping
            control_seq.clamp_(min=-1, max=1)

    # # Detach the optimized sequence from the computational graph
    # optimized_seq = control_seq.detach()
    print("Lowest Loss:", lowest_loss)
    return best_seq, lowest_loss

def smooth_tensor(tensor, box_pts):
    box = torch.ones(box_pts) / box_pts
    # Ensure tensor is on CPU for numpy compatibility if needed and convert to numpy for convolution
    tensor_np = tensor.cpu().squeeze().numpy()
    smoothed_np = np.convolve(tensor_np, box, mode='same')
    # Convert back to tensor
    smoothed_tensor = torch.tensor(smoothed_np, dtype=torch.float32).unsqueeze(1)
    return smoothed_tensor

# def acquire_new_data_sgd(last_control_seq, lr=0.01, max_iterations=10, threshold=0.01):
#     """
#     Acquire new data for training the autoencoder using stochastic gradient descent.
    
#     Args:
#     - last_control_seq (torch.Tensor): The last control sequence with requires_grad set to True.
#     - lr (float): Learning rate for SGD.
#     - max_iterations (int): Maximum number of iterations for the SGD optimization.
#     - threshold (float): Threshold for gradient norm for early stopping.
    
#     Returns:
#     - torch.Tensor: The optimized control sequence.
#     """
#     # Ensure last_control_seq requires gradients
#     last_control_seq = last_control_seq.clone().detach().requires_grad_(True).to(device)
#     optimizer = torch.optim.SGD([last_control_seq], lr=lr)
    
#     for iteration in range(max_iterations):
#         optimizer.zero_grad()

#         # Use the evaluate function to compute the loss
#         _, loss = autoencoder.evaluate_gradient(last_control_seq.unsqueeze(0), target_value_tensor, direction)
        
#         # Backward pass to compute gradients
#         loss.backward()

#         # Update last_control_seq based on gradients
#         optimizer.step()

#         # Check gradient norm for early stopping
#         grad_norm = last_control_seq.grad.norm()
#         if grad_norm < threshold:
#             # print(f"Early stopping at iteration {iteration} due to low gradient norm.")
#             break

#     return last_control_seq.detach()
torch.autograd.set_detect_anomaly(True)

clear = np.array([0,0,0,0,0,0,0,0])
epochs = config['epochs']
xvel = 0
data_con = np.zeros((epochs,control_sequence_length))
for epoch in range(epochs):

    # if epochs % render_frame == 0:  # Conditional render to reduce computation load
    #     env.render()


    # new_control_seq = acquire_new_data(new_control_seq)
    # if (target_value_tensor < 0):
    #     target_value_tensor = -target_value_tensor
    # else:
    #     target_value_tensor = target_value_tensor
    # _, loss = autoencoder.evaluate_gradient(new_control_seq.unsqueeze(0), target_value_tensor,direction)
    # print("prior control seq",new_control_seq,loss)
    
    prediction1, new_control_seq, l = acquire_new_data(new_control_seq, autoencoder, target_value_tensor, direction)
    # __, loss, reconstruction_loss, task_loss = autoencoder.evaluate_gradient(new_control_seq, target_value_tensor, direction)
    prediction2, loss2, reconstruction_loss, task_loss = autoencoder.evaluate(new_control_seq, target_value_tensor, direction)
    __, loss3, __, __ = autoencoder.evaluate(new_control_seq, target_value_tensor, direction)

    print("Epoch {}:\n\tNew Loss:".format(epoch), loss3)
    print("\tReconstruction Loss:", reconstruction_loss)
    print("\tTask Loss:", task_loss)
    print("\tLoss2:", loss2)
    # print("\tDecoded:", prediction1[0])
    # print("\tInput:", new_control_seq)
    print("\tExpected Reward:", prediction1[1][-1])
    expected_reward_list.append(prediction1[1][-1].item())

    total_reward = 0
    _new_control_seq = new_control_seq.view(control_sequence_time, joints)

    xvel = 0

    for i in range(control_sequence_time):
        action = _new_control_seq[i].cpu().numpy()
        r = 0
        t = 10
        for j in range(t):
            observation, reward, done, info = env.step(action)
            # if args["no_render"] != True:
            # r += (info["x_velocity"] * 2- np.abs(info["y_velocity"]))
            r += info["x_velocity"]/25
            xvel += info["x_velocity"]/25
            # r += ((info["x_velocity"] ** 2 + info["y_velocity"] ** 2) ** (1/2))/50
            env.render()

        total_reward += r

    x,y,z,w = observation[1:5]
    dir = math.atan2(2.0 * (w * x + y * z), 1 - 2 * (x**2 + z**2))/(2*math.pi)
    # direction_task.fill_(dir)
    # dir=0
    # direction.fill_(dir)
    direction.fill_(total_reward)

    # direction_list.append(dir - direction_prev[0])
    # direction_list.append(dir)
    direction_list.append(total_reward)


    direction_prev.fill_(dir)
    # direction.fill_(dir)

    #direction = torch.tensor([direction], dtype=torch.float32, device=device)
    #task_reward_list.append(total_reward)
        
    print("\tReward {} {}".format(total_reward, xvel))

    target_value_tensor.fill_(total_reward-target_value_tensor_previous)
    # target_value_tensor.fill_(total_reward)

    target_value_tensor_previous = total_reward

    combined_control_seq = torch.cat([combined_control_seq, new_control_seq.unsqueeze(0)], dim=0)
    combined_direction = torch.cat([combined_direction, direction.unsqueeze(0)], dim=0)
    combined_target_value_tensor = torch.cat([combined_target_value_tensor, target_value_tensor.unsqueeze(0)], dim=0)

    if combined_control_seq.shape[0] > 100:
        _combined_control_seq = combined_control_seq[-100:]
        _combined_direction = combined_direction[-100:]
        _combined_target_value_tensor = combined_target_value_tensor[-100:]
    else:
        _combined_control_seq = combined_control_seq.clone()
        _combined_direction = combined_direction.clone()
        _combined_target_value_tensor = combined_target_value_tensor.clone()

    if epoch < 5:
        _combined_target_value_tensor = smooth_tensor(_combined_target_value_tensor, 3).to(device)
    else:
        _combined_target_value_tensor = smooth_tensor(_combined_target_value_tensor, 5).to(device)

    task_reward_list.append(_combined_target_value_tensor[-1].item())

    # print(combined_target_value_tensor)
    # print(combined_control_seq.shape)
    # print(combined_direction.shape)
    # print(combined_target_value_tensor.shape)

    loss = autoencoder.train_model(_combined_control_seq[1:], _combined_target_value_tensor[1:], _combined_direction[1:], combined_latent_space[1:])
    # loss = autoencoder.train_model(new_control_seq, target_value_tensor, direction)

    # ep = _
    # data_con[ep,:] = new_control_seq.cpu().numpy()
    # # print(data_con.shape, len(task_reward_list), len(direction_list))
    # num_retrain = 20
    # if ep>num_retrain:
    #     for i in range(ep-1-num_retrain,ep+1):
    #         seq = torch.tensor(data_con[ep,:], dtype=torch.float32, device=device)
    #         tar = torch.tensor([task_reward_list[ep]], dtype=torch.float32, device=device)
    #         d = torch.tensor([direction_list[ep]], dtype=torch.float32, device=device)
    #         print(seq, tar, d)
    #         loss = autoencoder.train_model(seq, tar, d)
    # else:
    #     # print(new_control_seq,target_value_tensor,direction)
    #     loss = autoencoder.train_model(new_control_seq, target_value_tensor, direction)
    #
    # print("\tReconstruction Loss from Training:", loss[0])
    # print("\tTask Loss from Training:", loss[1])
    # print("\tCombined Loss:", loss[2])

    losses.append(loss[0])
    obj_loss.append(loss[1])
    encoded_list.append(loss[3][-1].to("cpu").tolist())
    # encoded_list.append(loss[5].to("cpu").tolist())
    decoded_list.append(loss[4].to("cpu").tolist())
    new_control_seq_values.append(new_control_seq.to("cpu").tolist()) 
    variation_list.append(loss[6][-1])

    # for i in range(5):
    #     observation, reward, done, info = env.step(clear)
    #     env.render()
    # observation = env.reset()
  

    # Check if the episode is done
    if done:
        observation = env.reset()
        env.step(clear)

env.close()

fig, axes = plt.subplots(1, 4, figsize=(12, 6))

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

new_control_seq_values_transposed = list(zip(*decoded_list))
for seq_index, seq_values in enumerate(new_control_seq_values_transposed):
    axes[3].plot(range(1, epochs + 1), seq_values, label=f"Seq {seq_index + 1}")
axes[3].set_xlabel('Epoch')
axes[3].set_ylabel('Value')
axes[3].set_title('New Control Seq Values Over Time')

plt.tight_layout()
plt.show()

plt.plot(range(1, epochs + 1), task_reward_list, label='Task Reward')
plt.plot(range(1, epochs + 1), expected_reward_list, label='Expected Reward')
plt.plot(range(1, epochs + 1), [(e - t)**2 for e, t in zip(expected_reward_list, task_reward_list)], label='Reward Difference')

plt.xlabel('Epoch')
plt.ylabel('Reward')
plt.title('Rewards Over Epochs')
plt.legend()
plt.show()

# Convert log variance to standard deviation for shading
std_dev_list = [torch.exp(0.5 * lv).cpu().detach().numpy() for lv in variation_list]

# Assuming the structure and size of encoded_list matches with log_var_list for the purpose of this demonstration

# Transposing lists to work with sequences across dimensions

encoded_list_transposed = list(zip(*encoded_list))
std_dev_list_transposed = list(zip(std_dev_list[-1]))

print(std_dev_list_transposed)

# Create a new figure for the plot
plt.figure(figsize=(10, 6))

# Plot each sequence in the encoded list and add variance shading
for dim_index, (dim_values, dim_std_devs) in enumerate(zip(encoded_list_transposed, std_dev_list_transposed)):
    _epoch = np.arange(1, len(dim_values) + 1)
    dim_values = np.array(dim_values)
    dim_std_devs = np.array(dim_std_devs)

    # Plot the dimension values
    plt.plot(_epoch, dim_values, label=f"Dimension {dim_index + 1}")

    # Calculate upper and lower bounds for the shaded area using standard deviation
    upper_bounds = dim_values + dim_std_devs
    lower_bounds = dim_values - dim_std_devs

    # Add shaded area to represent variance
    plt.fill_between(_epoch, lower_bounds, upper_bounds, alpha=0.2)


plt.xlabel('Epoch')
plt.ylabel('Encoded Value')
plt.title('Encoded Sequence Values Over Time')
plt.legend()
plt.show()

for seq_index, seq_values in enumerate(encoded_list_transposed):
    plt.plot(range(1, epochs + 1), seq_values, label=f"Seq {seq_index + 1}")
plt.show()

# Plot objective loss on the second subplot
plt.plot(range(1, epochs + 1), direction_list, label="Direction")
plt.show()

# # Convert log_variances to numpy arrays and calculate standard deviations
# std_devs = np.array([torch.exp(0.5 * lv).cpu().detach().numpy() for lv in variation_list])
#
# # Calculate upper and lower bounds for shading
# means = np.array(encoded_list[0])
# upper_bounds = means + std_devs
# lower_bounds = means - std_devs
#
# # Plotting
# plt.figure(figsize=(10, 6))
# x = np.arange(means.shape[1])  # Assuming x-axis represents the dimensionality of the encoded space
# for i in range(means.shape[0]):
#     plt.plot(x, means[i], '-o')  # Plot mean values
#     plt.fill_between(x, lower_bounds[i], upper_bounds[i], alpha=0.2)  # Plot shaded area for variance
#
# plt.title("Encoded Values with Variance Shading")
# plt.xlabel("Dimension")
# plt.ylabel("Encoded Value")
# plt.show()
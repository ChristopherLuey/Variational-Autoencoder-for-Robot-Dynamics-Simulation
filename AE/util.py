import torch
import numpy as np
import argparse
import yaml

def smooth_tensor(tensor, box_pts):
    box = torch.ones(box_pts) / box_pts
    # Ensure tensor is on CPU for numpy compatibility if needed and convert to numpy for convolution
    tensor_np = tensor.cpu().squeeze().numpy()
    smoothed_np = np.convolve(tensor_np, box, mode='same')
    # Convert back to tensor
    smoothed_tensor = torch.tensor(smoothed_np, dtype=torch.float32).unsqueeze(1)
    return smoothed_tensor


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


def configuration(args):
    # load config
    config_path = f'./config/AE.yaml'

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = config_dict[args.env]

    return config


def check_gpu(args):
    device = 'cpu'
    if not args.cpu:
        if torch.cuda.is_available():
            torch.set_num_threads(1)
            device = 'cuda:0'
            print('Using GPU Accel')
        else:
            args.cpu = True
    return device
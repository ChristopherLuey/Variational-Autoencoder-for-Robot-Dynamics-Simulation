
import sys
import os
import yaml
from termcolor import cprint
from datetime import datetime
import time

import torch
import numpy as np
import random
import pickle
import argparse

# local imports
from utils import get_duration, save_config

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
cprint(args,'cyan')
args.v3 = True
args.done_util = True
base_method = args.method[:3]

device ='cpu'
if not args.cpu:
    if torch.cuda.is_available():
        torch.set_num_threads(1)
        device  = 'cuda:0'
        print('Using GPU Accel')
    else:
        args.cpu = True

# added to save when exiting
from signal import signal, SIGINT
from sys import exit

def end_test():
    env.close()
    if args.log:
        print('saving final data set')
        pickle.dump(rewards, open(path + 'reward_data'+ '.pkl', 'wb'))
        pickle.dump(eval_rewards, open(path + 'eval_reward_data' + '.pkl', 'wb'))

        if base_method == "AE":
            torch.save()
        if base_method == 'sac':
            torch.save(policy_net.state_dict(), path + 'policy_' + 'final' + '.pt')
        else:
            torch.save(model.state_dict(), path + 'model_' + 'final' + '.pt')
            pickle.dump(model_optim.log, open(path + 'optim_data'+ '.pkl', 'wb'))

        # save duration
        end = datetime.now()
        date_str = end.strftime("%Y-%m-%d_%H-%M-%S/")
        duration_str = get_duration(start_time)

        # save config
        with open(path + "/../config.txt","a") as f:
            f.write('End Time\n')
            f.write('\t'+ date_str + '\n')
            f.write('Duration\n')
            f.write('\t'+ duration_str + '\n')
            f.close()

        # save final steps
        buff = replay_buffer.get_final_samples(10000)
        pickle.dump(buff, open(path + 'buffer_data'+ '.pkl', 'wb'))

def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected.')
    end_test()
    print('Exiting gracefully')
    exit(0)

# eval function
def eval():
    state = env.reset()
    if not(base_method == 'sac' ):
        planner.reset()

    episode_reward = 0.
    states = []
    for step in range(max_steps):
        if base_method == 'sac' :
            action = policy_net.get_action(state,eval=True)
        else:
            action = planner(state,eval=True)
        state, reward, done, _ = env.step(action.copy())
        episode_reward += reward
        if args.render:
            env.render(mode="human")
        if args.done_util:
            if done:
                break
    step += 1
    cprint('eval: {} {}'.format(episode_reward, step),'cyan')
    return states, episode_reward, step

def configuration():
    # load config
    if base_method == 'AE':
        config_path = f'./config/AE.yaml'
    else:
        raise ValueError('config file not found for env')


    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
        config = config_dict['default']
        if args.env in list(config_dict.keys()):
            config.update(config_dict[args.env])
        else:
            raise ValueError('env not found config file')

    return config

if __name__ == '__main__':
# Tell Python to run the handler() function when SIGINT is recieved
    signal(SIGINT, handler)

    # load config
    config = configuration()

    # set seeds
    def set_seeds():
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.set_flush_denormal(True)

     # initialize environment
    
    from environments.build_env import build_env
    env, env_name, action_dim, state_dim, traj, viewer = build_env(args,config,device)
    cprint(env,'green')
    print('actions states',action_dim,state_dim)

    # # load models / policies / controllers
    # from environments import ReplayBuffer
    # eval_freq = 5
    # replay_buffer_size = int(1e6)
    # replay_buffer = ReplayBuffer(replay_buffer_size,state_dim,action_dim)
    # replay_buffer.seed(args.seed)




    # from mpc_lib import Model, ModelOptimizer
    # model_kwargs = {'model_layers':config['model_layers'],'model_AF':config['model_activation_fun'],
    #                 'reward_layers':config['reward_layers'],'reward_AF':config['reward_activation_fun']}
    # model = Model(state_dim, action_dim,**model_kwargs).to(device)

    # from AE.ae import AugmentedAutoencoder

    # model_kwargs = {'input_size': config['input_size'], 
    #                 'latent_size': config['latent_size'],
    #                 'encoder_layer_sizes': config['encoder_layer_sizes'],
    #                 'decoder_layer_sizes': config['decoder_layer_sizes']}

    # model = AugmentedAutoencoder()

    # #### jit model for planner (samples)
    # with torch.no_grad():
    #     inputs = (torch.rand(config['planner']['samples'],state_dim,device=device),torch.rand( config['planner']['samples'],action_dim,device=device))
    #     jit_model_plan = torch.jit.trace(model,inputs) # set up traced model
    #     primed = jit_model_plan(*inputs) # prime model
    #     # print(jit_model_plan.graph)
    # #### jit model for optimizer (batch size)


    # inputs = (torch.rand(config['batch_size'],state_dim,device=device),torch.rand( config['batch_size'],action_dim,device=device))
    # jit_model_opt = torch.jit.trace(model,inputs) # set up traced model
    # primed = jit_model_opt(*inputs) # prime model
    # model_optim = ModelOptimizer(jit_model_opt, replay_buffer, lr=config['model_lr'],device=device)
    # if base_method == 'mpp':
    #     from mpc_lib import PathIntegral
    #     planner = PathIntegral(jit_model_plan,device=device,**config['planner'])
    # elif base_method == 'max':
    #     from mpc_lib import MaxDiff
    #     planner = MaxDiff(jit_model_plan,device=device,**config['planner'])

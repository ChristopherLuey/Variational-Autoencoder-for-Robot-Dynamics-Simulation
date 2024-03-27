import gym
from gym.envs.mujoco.ant_v3 import AntEnv

class CustomAntEnv(AntEnv):
    def __init__(self, forward_reward_weight=1.0, ctrl_cost_weight=0, contact_cost_weight=0, healthy_reward=0, terminate_when_unhealthy=True, healthy_z_range=(0.4, 5.0), *args, **kwargs):
        # Initialize custom attributes before calling the superclass's __init__
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        
        # Call the superclass's __init__ without the xml_file argument
        super().__init__(*args, **kwargs)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        return observation, reward, done, info
    

    def reset(self, **kwargs):
        # Call the superclass's reset method
        initial_observation = super().reset(**kwargs)
        return initial_observation

    # @property
    # def is_healthy(self):
    #     return super().is_healthy()
    
    @property
    def terminated(self):
        return super().terminated()

    # Other custom methods as needed

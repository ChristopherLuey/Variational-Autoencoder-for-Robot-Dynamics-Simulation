import gym
from gym.envs.mujoco.ant_v3 import AntEnv


class CustomAntEnv(AntEnv):
    def __init__(self,
                 forward_reward_weight=1.0,
                 ctrl_cost_weight=0,
                 contact_cost_weight=0,
                 healthy_reward=0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.4, 5.0),
                 *args, **kwargs):
        # Initialize custom attributes
        self.forward_reward_weight = forward_reward_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range

        # Initialize the superclass (AntEnv)
        super(CustomAntEnv, self).__init__(*args, **kwargs)

    def step(self, action):
        observation, reward, done, info = super().step(action)

        # Check if the ant is unhealthy and should reset
        if self._terminate_when_unhealthy and not self.is_healthy:
            done = True

        # Modify the reward based on custom parameters (if needed)
        # For simplicity, this example does not alter the reward

        return observation, reward, done, info

    @property
    def is_healthy(self):
        # You might want to adjust the health criteria based on custom parameters

        return super().is_healthy()
        # min_z, max_z = self._healthy_z_range
        # z = self.get_body_com("torso")[2]
        # return min_z <= z <= max_z

    def reset(self, **kwargs):
        return super().reset(**kwargs)

    @property
    def terminated(self):
        return super().terminated

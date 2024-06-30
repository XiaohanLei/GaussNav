import json
import bz2
import gzip
import _pickle as cPickle
import gym
import numpy as np
import quaternion
import skimage.morphology
import habitat
# from base_env import base_env
from vector_env.envs.base_env import base_env
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

class shortest_path_env(base_env):
    """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, config_env, dataset, rank):
        super().__init__(config_env, dataset, rank)
        self.goal_radius = 1.0
        self.follower = ShortestPathFollower(
            self.habitat_env.sim, self.goal_radius, False
        )
        self.dump_location = config_env.end2end_imagenav.dump_location
        print(f"rank: {self.rank} shortest path env initialize suceessful !")


    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """


        obs, self.info = super().reset()
        rgb = obs['rgb'].astype(np.uint8)
        if self.last_scene_name != self.scene_name:
            self.follower = ShortestPathFollower(
                self.habitat_env.sim, self.goal_radius, False
            )

        return rgb.transpose(2,0,1), self.info

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            reward (float): amount of reward returned after previous action
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        if action["finished"]:
            return np.zeros((3, self.frame_height, self.frame_width)), 0., False, self.info

        best_action = self.follower.get_next_action(
                    self.habitat_env.current_episode.goals[0].position
                )
        if best_action is None:
            best_action = 0
        action = {"action": best_action}

        obs, rew, done, _ = super().step(action)
        rgb = obs['rgb'].astype(np.uint8)
        self.visualize_image(rgb, self.dump_location)

        return rgb.transpose(2,0,1), rew, done, self.info

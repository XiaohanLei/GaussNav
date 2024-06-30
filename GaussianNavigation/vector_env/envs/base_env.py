import json
import bz2
import gzip
import _pickle as cPickle
import gym
import numpy as np
import quaternion
import skimage.morphology
import habitat
import os
import cv2
import quaternion


class base_env(habitat.RLEnv):
    """The Object Goal Navigation environment class. The class is responsible
    for loading the dataset, generating episodes, and computing evaluation
    metrics.
    """

    def __init__(self, config_env, dataset, rank):
        super().__init__(config_env)

        self.rank = rank
        # Loading dataset info file
        self.split = config_env.habitat.dataset.split
        # Initializations
        self.episode_no = 0
        # Scene info
        self.scene_name = None
        self.last_scene_name = None
        self.scene_floor_name = None
        self.floor = 0
        self.scene_height = []
        self.scene_floor_changed = False

        # Episode tracking info
        self.timestep = None
        self.stopped = None
        self.info = {}
        self.info['distance_to_goal'] = 0
        self.info['spl'] = 0
        self.info['success'] = 0
        self.info['episode_count'] = 0

        # frame utils:
        self.env_frame_width = config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.width
        self.env_frame_height = config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.height
        self.min_depth = config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.min_depth
        self.max_depth = config_env.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor.max_depth
        self.frame_width = config_env.end2end_imagenav.mapper.frame_width
        self.frame_height = config_env.end2end_imagenav.mapper.frame_height

        self.camera_height = config_env.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor.position[1]
        self.last_start_height = None
        self.start_height = None

        # eval metrics
        self.spl_sum = 0
        self.succ_sum = 0
        self.dist_sum = 0



    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """

        self.episode_no += 1
        obs = super().reset()
        # Initializations
        self.timestep = 0
        self.stopped = False
        
        self.gt_goal_object_id = int(self.habitat_env.current_episode.goal_object_id)
        self.gt_goal_category = self.habitat_env.current_episode.object_category
        self.last_scene_name = self.scene_name
        self.scene_name = (self.habitat_env.sim.config.sim_cfg.scene_id).split('/')[-1].split('.')[0]
        self.last_start_height = self.start_height

        agent_state = self._env.sim.get_agent_state(0).position
        self.start_height = agent_state[1]
        self.agent_height = self.camera_height

        if self.last_scene_name != self.scene_name:
            print("Changing scene: rank:  {}   ||  scene name:  {}".format(self.rank, self.scene_name))
            self.scene_height = []
            self.floor = self.start_height

        # determine which floor agent is
        if len(self.scene_height) == 0:
            self.scene_height.append(self.start_height)
            self.floor = self.start_height
            if (self.last_scene_name is not None) and (self.last_start_height is not None):
                self.scene_floor_name = self.last_scene_name + '-floor-' + str(format(self.last_start_height, '.1f'))
                self.scene_floor_changed = True
        else:
            sub_height = [abs(self.start_height - h) for h in self.scene_height]
            sub_height = np.array(sub_height)
            self.scene_floor_changed = False
            if np.all(sub_height > 1.0):
                self.scene_height.append(self.start_height)
                self.floor = self.start_height
                self.scene_floor_name = self.scene_name + '-floor-' + str(format(self.last_start_height, '.1f'))
                self.scene_floor_changed = True

        # memorize agent state in world coordinate frame
        self.start_pos_w_env = self._env.sim.get_agent_state().position
        self.start_rot_w_env = self._env.sim.get_agent_state().rotation
        self.goal_pos_w_env = self.habitat_env.current_episode.goals[0].position
        self.min_viewpoint_goal_w_env = self.get_instance_image_goal_viewpoint_goal()

        # get goal object id
        name2index = {
            "chair": 56,
            "sofa": 57,
            "plant": 58,
            "bed": 59,
            "toilet": 61,
            "tv_monitor": 62,
        }
        self.gt_goal_coco_id = name2index[self.gt_goal_category]

        # Set info
        self.info['time'] = self.timestep

        return obs, self.info

    def get_instance_image_goal_viewpoint_goal(self):
        instance_center = np.array(self.habitat_env.current_episode.goals[0].position)
        view_points = [self.habitat_env.current_episode.goals[0].view_points[i].agent_state.position \
                        for i in range(len(self.habitat_env.current_episode.goals[0].view_points))]
        view_points = np.array(view_points)
        dis = np.sum((view_points - instance_center) ** 2, axis=1)
        min_index = np.argmin(dis)
        return view_points[min_index].tolist()

    def get_random_imagegoal_viewpoint(self):

        view_points_pos = [self.habitat_env.current_episode.goals[0].view_points[i].agent_state.position \
                        for i in range(len(self.habitat_env.current_episode.goals[0].view_points))]
        view_points_rot = [self.habitat_env.current_episode.goals[0].view_points[i].agent_state.rotation \
                        for i in range(len(self.habitat_env.current_episode.goals[0].view_points))]
        select_index = np.random.randint(len(view_points_pos))
        
        return view_points_pos[select_index], view_points_rot[select_index]

    def step(self, action):
        """Function to take an action in the environment.

        Args:
            action (dict):
                dict with following keys:
                    'action' (int): 0: stop, 1: forward, 2: left, 3: right

        Returns:
            obs (ndarray): observations from env's feedback
            reward (float): amount of reward returned after previous action, should be modified
            done (bool): whether the episode has ended
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """
        action = action["action"]
        if action == 0:
            self.stopped = True

        obs, rew, done, _ = super().step(action)

        spl, success, dist = 0., 0., 0.
        if done:
            spl, success, dist = self.get_metrics()
            self.dist_sum += dist
            self.spl_sum += spl
            self.succ_sum += success
            self.info['distance_to_goal'] = self.dist_sum / (self.episode_no)
            self.info['spl'] = self.spl_sum / (self.episode_no)
            self.info['success'] = self.succ_sum / (self.episode_no)
            self.info['episode_count'] = self.episode_no


        self.timestep += 1
        self.info['time'] = self.timestep

        agent_state = self._env.sim.get_agent_state(0).position
        self.agent_height = self.camera_height + agent_state[1] - self.start_height

        return obs, rew, done, self.info

    def get_observation_at(self, position: np.ndarray, rotation: np.ndarray):

        self._env.sim.set_agent_state(position.tolist(), rotation.tolist())
        sim_obs = self._env.sim.get_sensor_observations()
        obs = self._env.sim._sensor_suite.get_observations(sim_obs)

        return obs

    def set_agent(self, pos, rot):
        if isinstance(pos, np.ndarray):
            pos = pos.tolist()
        if isinstance(rot, np.ndarray):
            rot = rot.tolist()
        else: 
            import utils.pose as pu
            rot = pu.qua2list(rot)
        self._env.sim.set_agent_state(pos, rot)

    # define some utilities below
    def visualize_image(self, image: np.ndarray, dir: str, img_name=None):
        '''
        store image in the specified directory,
        make sure the image is HWC and RGB format
        '''
        if not os.path.exists(dir):
            # in order to avoid multiprocess dump image error
            try:
                os.makedirs(dir)
            except:
                pass
        if img_name is None:
            fn = '{}/{}-{}-{}-Vis.png'.format(
                    dir, self.scene_name, self.episode_no,
                    self.timestep)
            cv2.imwrite(fn, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        else:
            fn = '{}/{}-{}-{}-{}.png'.format(
                    dir, self.scene_name, self.episode_no,
                    self.timestep, img_name)
            cv2.imwrite(fn, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def visualize_video(self, image_list: list, dir: str, video_name=None, fps=5):
        '''
        save video
        '''
        if not os.path.exists(dir):
            # in order to avoid multiprocess dump image error
            try:
                os.makedirs(dir)
            except:
                pass
        width, height = image_list[0].shape[1], image_list[0].shape[0]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if video_name is None:
            fn = '{}/{}-{}-Vis.mp4'.format(
                    dir, self.scene_name, self.episode_no)
            out = cv2.VideoWriter(fn, fourcc, fps, (width, height))
        else:
            fn = '{}/{}-{}-{}.png'.format(
                    dir, self.scene_name, self.episode_no, video_name)
            out = cv2.VideoWriter(fn, fourcc, fps, (width, height))
        for image in image_list:
            out.write(image)
        out.release()

    def visualize_semantic(self, image: np.ndarray, dir: str):
        '''
        store image in the specified directory,
        make sure the image is HWC and RGB format
        '''
        if not os.path.exists(dir):
            # in order to avoid multiprocess dump image error
            try:
                os.makedirs(dir)
            except:
                pass

        fn = '{}/{}-{}-{}-Vis.png'.format(
                dir, self.scene_name, self.episode_no,
                self.timestep)
        cv2.imwrite(fn, image)

    def from_transformation_to_xyo(self, transform):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        x = -transform[2, 3]
        y = -transform[0, 3]
        quat = quaternion.from_rotation_matrix(transform[:3, :3])
        axis = quaternion.as_euler_angles(quat)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(quat)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(quat)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return [x, y, o]
    

    def get_reward(self, observations):
        reward = 0
        return reward

    def get_reward_range(self):
        return [-1, 1]

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
    
    def get_metrics(self):
        info = self.habitat_env.get_metrics()
        spl = info['spl']
        success = info['success'] 
        dist = info['distance_to_goal']
        return spl, success, dist
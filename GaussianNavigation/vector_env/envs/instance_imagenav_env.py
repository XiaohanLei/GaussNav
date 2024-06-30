import json
import bz2
import gzip
import _pickle as cPickle
import gym
import numpy as np
import quaternion
import skimage.morphology
import habitat
import cv2
import magnum as mn
import Quaternion
from tqdm import tqdm
# from base_env import base_env
from lightglue import LightGlue, SuperPoint, DISK
from lightglue.utils import load_image, rbd, match_pair , numpy_image_to_torch
from vector_env.envs.base_env import base_env
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from torchvision import transforms
import utils.pose as pu
from map_planning_utils.mapper import Semantic_Mapping
from map_planning_utils.planner import Planner
from PIL import Image
import torch
import os
from glob import glob
from natsort import natsorted
import logging


class NiceEnv(base_env):
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
        self.base_dump_location = config_env.end2end_imagenav.dump_location
        self.base_dump_location = os.path.join(self.base_dump_location, str(rank))
        print(f"rank: {self.rank} env initialize suceessful !")

        self.res = transforms.Compose(
            [transforms.ToPILImage(),
             transforms.Resize((self.frame_height, self.frame_width),
                               interpolation=Image.NEAREST)])

        gpu_id = config_env.end2end_imagenav.gpu_id
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.mapper = Semantic_Mapping(config_env, self.device)

        self.planner = Planner()

        # viz params
        self.viz = config_env.end2end_imagenav.viz_params.viz
        self.viz_failure_case = config_env.end2end_imagenav.viz_params.viz_failure_case
        if self.viz:
            self.viz_obs = []

        self.similarity_method = config_env.end2end_imagenav.particle_params.method
        self.remap_goal = config_env.end2end_imagenav.particle_params.remap_goal

        # # use lightglue
        if self.similarity_method == 1:
            self.extractor = DISK(max_num_keypoints=2048).eval().to(self.device)
            self.matcher = LightGlue(features='disk').eval().to(self.device)
            self.matcher.compile(mode='reduce-overhead')



    def reset(self):
        """Resets the environment to a new episode.

        Returns:
            obs (ndarray): RGBD observations (4 x H x W)
            info (dict): contains timestep, pose, goal category and
                         evaluation metric info
        """


        obs, self.info = super().reset()
        self.obs = obs
        rgb = obs['rgb'].astype(np.uint8)
        self.instance_imagegoal = obs['instance_imagegoal']
        if self.last_scene_name != self.scene_name:
            self.follower = ShortestPathFollower(
                self.habitat_env.sim, self.goal_radius, False
            )

        self.dump_location = os.path.join(self.base_dump_location, str(self.episode_no))

        self.mapper.reset()
        self.last_sim_location = self.get_sim_location()

        self.planner.reset(self.mapper)

        # # use preprocessed semantic gaussian for goal map prediction
        # self.target_obs = {'rgb':self.obs['instance_imagegoal']}
        # # normal
        # self.pred_goal_pos = self.compare_bbox(self.instance_imagegoal, self.gt_goal_coco_id)
        # # random select instance
        # self.pred_goal_pos = self.select_instance(self.instance_imagegoal, self.gt_goal_coco_id)


        if self.viz:
            if len(self.viz_obs) > 0:
                if not self.viz_failure_case:
                    temp_dump_location = os.path.join(self.base_dump_location, str(self.episode_no-1))
                    self.visualize_video(self.viz_obs, temp_dump_location)
                    self.viz_obs = []

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
            return np.zeros((3, self.env_frame_height, self.env_frame_width)), 0., False, self.info


        # here, we directly get the goal position from environment
        # we will update the code to get the goal position from Semantic Gaussian
        action = self.planner.plan(self.from_env_to_map(self.min_viewpoint_goal_w_env).tolist(), self.mapper, selem=5)
        

        obs, rew, done, _ = super().step(action)
        self.obs = obs
        rgb = obs['rgb'].astype(np.uint8)
        # self.visualize_image(rgb, self.dump_location, 'rgb')

        processed_obs = self._preprocess_obs(obs)
        processed_obs = torch.from_numpy(processed_obs).float().to(self.device).unsqueeze(0)

        dx, dy, do = self.get_pose_change()
        pose_obs = torch.Tensor([dx, dy, do]).float().to(self.device).unsqueeze(0)

        agent_heights = torch.Tensor([self.agent_height]).float().to(self.device).unsqueeze(0)

        # map_struct = self.mapper.step(processed_obs, pose_obs, agent_heights, viz_obs=obs, viz_image_goal=self.target_obs['rgb'])
        map_struct = self.mapper.step(processed_obs, pose_obs, agent_heights, viz_obs=obs)

        # visualize goal position
        if self.viz:
            vis_map = map_struct['vis_map']
            goal_pos_w_map = self.from_env_to_map(self.pred_goal_pos)
            vis_map[20:500, 40+360:400+480][goal_pos_w_map[0]*2-5:goal_pos_w_map[0]*2+5, \
                                            goal_pos_w_map[1]*2-5:goal_pos_w_map[1]*2+5] = [255, 0, 0]
            self.viz_obs.append(vis_map)
            
            if done and self.viz_failure_case:
                spl, success, dist = self.get_metrics()
                if success == 0:
                    temp_dump_location = os.path.join(self.base_dump_location, str(self.episode_no-1))
                    self.visualize_video(self.viz_obs, temp_dump_location)
                self.viz_obs = []

        return rgb.transpose(2,0,1), rew, done, self.info


    def select_instance(self, target_obs, goal_coco_id):
        base_scene_path = '/instance_imagenav/end2end_imagenav/env_collect_v1'
        posssible_scenes = glob(os.path.join(base_scene_path, f'{self.scene_name}*'))
        for i in range(len(posssible_scenes)):
            if abs(float((posssible_scenes[i].split('floor')[-1])[1:]) - self.scene_height[-1]) < 1.0:
                scene_path = posssible_scenes[i]
                transforms_json_path = os.path.join(scene_path, 'transforms.json')
                instance_json_path = os.path.join(scene_path, 'instance_retrieval.json')
                break
        try:
            instance_retrieval = json.load(open(instance_json_path, "r"))
        except:
            import sys
            print(posssible_scenes)
            print(self.scene_height)
            print(self.scene_name)
            sys.exit()
        curr_instance_list = instance_retrieval[str(goal_coco_id)]
        
        if len(curr_instance_list) > 0:
            max_index = np.random.randint(len(curr_instance_list))
            sele_bbox = curr_instance_list[max_index][0][0]
            sele_goal_pos = [sele_bbox[0], sele_bbox[2], sele_bbox[4]]
        else:
            sele_goal_pos = [0, 0, 0]
        return sele_goal_pos  

    def compare_bbox(self, target_obs, goal_coco_id):
        base_scene_path = '/instance_imagenav/end2end_imagenav/env_collect_v1'
        posssible_scenes = glob(os.path.join(base_scene_path, f'{self.scene_name}*'))
        for i in range(len(posssible_scenes)):
            if abs(float((posssible_scenes[i].split('floor')[-1])[1:]) - self.scene_height[-1]) < 1.0:
                scene_path = posssible_scenes[i]
                transforms_json_path = os.path.join(scene_path, 'transforms.json')
                instance_json_path = os.path.join(scene_path, 'instance_retrieval.json')
                break
        try:
            instance_retrieval = json.load(open(instance_json_path, "r"))
        except:
            import sys
            print(posssible_scenes)
            print(self.scene_height)
            print(self.scene_name)
            sys.exit()
        curr_instance_list = instance_retrieval[str(goal_coco_id)]
        if len(curr_instance_list) > 0:
            matched_keypoints_list = []
            for bi_pairs in tqdm(curr_instance_list):
                img_name = 'rgb/frame_' + bi_pairs[1] + '.png'
                instance_img = cv2.imread(os.path.join(scene_path, img_name))[:, :, ::-1]
                matched_keypoints_list.append(self.compute_image_pair_similarity(target_obs, instance_img))
            matched_keypoints_array = np.array(matched_keypoints_list)
            max_index = np.argmax(matched_keypoints_array, axis=0)
            print(self.rank, self.episode_no, curr_instance_list[max_index][1])
            sele_bbox = curr_instance_list[max_index][0][0]
            sele_goal_pos = [sele_bbox[0], sele_bbox[2], sele_bbox[4]]
        else:
            sele_goal_pos = [0, 0, 0]
        return sele_goal_pos

    def compute_image_pair_similarity(self, obs1, obs2, resize_factors=4):

        method = self.similarity_method
        # method = 3

        if method == 0:
            # implement the histogram measurement
            rgb1 = obs1["rgb"] 
            depth1 = obs1["depth"][:, :, 0]
            rgb2 = obs2["rgb"] 
            depth2 = obs2["depth"][:, :, 0]

            rgb1 = cv2.resize(rgb1, (rgb1.shape[1] // resize_factors, rgb1.shape[0] // resize_factors))
            depth1 = cv2.resize(depth1, (depth1.shape[1] // resize_factors, depth1.shape[0] // resize_factors))
            rgb2 = cv2.resize(rgb2, (rgb2.shape[1] // resize_factors, rgb2.shape[0] // resize_factors))
            depth2 = cv2.resize(depth2, (depth2.shape[1] // resize_factors, depth2.shape[0] // resize_factors))

            hist_img1 = cv2.calcHist([rgb1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            hist_img2 = cv2.calcHist([rgb2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
            cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

            rgb_similarity_score = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)

            depth_hist1, _ = np.histogram(depth1/(self.max_depth*1000), bins=256, range=(0, 1))
            depth_hist2, _ = np.histogram(depth2/(self.max_depth*1000), bins=256, range=(0, 1))
            depth_similarity_score = np.corrcoef(depth_hist1, depth_hist2)[0, 1]
            similarity_score = 0.5 * rgb_similarity_score + 0.5 * depth_similarity_score

        elif method == 1:
            with torch.set_grad_enabled(False):
                if isinstance(obs1, dict):
                    ob = numpy_image_to_torch(obs1['rgb']).to(self.device)
                    gi = numpy_image_to_torch(obs2['rgb']).to(self.device)
                else:
                    ob = numpy_image_to_torch(obs1).to(self.device)
                    gi = numpy_image_to_torch(obs2).to(self.device)
                try:
                    feats0, feats1, matches01  = match_pair(self.extractor, self.matcher, ob, gi
                        )
                    # indices with shape (K, 2)
                    matches = matches01['matches']
                    # in case that the matches collapse make a check
                    if matches.shape[0] > 2048:
                        similarity_score = 0
                    else:
                        similarity_score = matches.shape[0]
                except:
                    similarity_score = 0

        elif method == 2:
            if isinstance(obs1, dict):
                gray1 = cv2.cvtColor(obs1['rgb'], cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(obs2['rgb'], cv2.COLOR_BGR2GRAY)
            else:
                gray1 = cv2.cvtColor(obs1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(obs2, cv2.COLOR_BGR2GRAY)
            try:
                # Initialize SIFT detector
                sift = cv2.SIFT_create()

                # Detect keypoints and compute descriptors
                keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
                keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

                # Define FLANN matcher parameters
                index_params = dict(algorithm=1, trees=5)  # Using FLANN_INDEX_KDTREE algorithm
                search_params = dict(checks=50)  # Number of times the tree(s) in the index should be recursively traversed

                # Initialize FLANN matcher
                flann = cv2.FlannBasedMatcher(index_params, search_params)

                # Match descriptors
                matches = flann.knnMatch(descriptors1, descriptors2, k=2)

                # Apply ratio test to find good matches
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

                similarity_score = len(good_matches)
            except:
                similarity_score = 0

        elif method == 3:
            with torch.set_grad_enabled(False):                
                if isinstance(obs1, dict):
                    ob = cv2.cvtColor(obs1['rgb'], cv2.COLOR_BGR2GRAY)
                    gi = cv2.cvtColor(obs2['rgb'], cv2.COLOR_BGR2GRAY)
                else:
                    ob = cv2.cvtColor(obs1, cv2.COLOR_BGR2GRAY)
                    gi = cv2.cvtColor(obs2, cv2.COLOR_BGR2GRAY)
                torch_gray0 = numpy_image_to_torch(ob).to(self.device)[None]
                torch_gray1 = numpy_image_to_torch(gi).to(self.device)[None]
                try:
                    x = {'image0': torch_gray0, 'image1': torch_gray1}
                    pred = self.pipeline_gluestick(x)
                    pred = batch_to_np(pred)
                    kp0, kp1 = pred["keypoints0"], pred["keypoints1"]
                    m0 = pred["matches0"]

                    line_seg0, line_seg1 = pred["lines0"], pred["lines1"]
                    line_matches = pred["line_matches0"]

                    valid_matches = m0 != -1
                    match_indices = m0[valid_matches]

                    similarity_score = match_indices.shape[0]
                    # matched_kps0 = kp0[valid_matches]
                    # matched_kps1 = kp1[match_indices]

                    # valid_matches = line_matches != -1
                    # match_indices = line_matches[valid_matches]
                    # matched_lines0 = line_seg0[valid_matches]
                    # matched_lines1 = line_seg1[match_indices]
                except:
                    similarity_score = 0
        
        return similarity_score


    def get_sim_location(self):
        """Returns x, y, o pose of the agent in the Habitat simulator."""

        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis %
                                          (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o
    

    def get_pose_change(self):
        """Returns dx, dy, do pose change of the agent relative to the last
        timestep."""
        curr_sim_pose = self.get_sim_location()
        dx, dy, do = pu.get_rel_pose_change(
            curr_sim_pose, self.last_sim_location)
        self.last_sim_location = curr_sim_pose
        return dx, dy, do
    
    def from_env_to_map(self, pos, rot=None):
        """Converts x, y coordinates from Habitat simulator to map coordinates."""
        if rot is None:
            source_position = np.array(self.start_pos_w_env, dtype=np.float32)
            source_rotation = pu.quaternion_from_coeff([self.start_rot_w_env.x, self.start_rot_w_env.y, \
                                                        self.start_rot_w_env.z, self.start_rot_w_env.w])
            goal_position = np.array(pos)
            direction_vector = goal_position - source_position
            direction_vector_agent = pu.quaternion_rotate_vector(
                source_rotation.inverse(), direction_vector
            )
            rho, phi = pu.cartesian_to_polar(
                -direction_vector_agent[2], direction_vector_agent[0]
            )
            dist, angle = rho, -phi
            x = int(dist*np.cos(angle)*(100. / self.mapper.resolution))
            y = int(dist*np.sin(angle)*(100. / self.mapper.resolution))
            map_size = self.mapper.full_h
            robot_pos = [map_size // 2 + y,
                        map_size // 2 + x]
            global_pos = [robot_pos[0] , robot_pos[1]]
            local_pos = [robot_pos[0] - self.mapper.lmb[0, 0].item() , robot_pos[1] - self.mapper.lmb[0, 2].item()]
            local_pos = np.array(local_pos).astype(int)

            return local_pos

    def _preprocess_obs(self, obs, use_seg=True):

        rgb = obs['rgb']
        depth = obs['depth']

        depth = self._preprocess_depth(depth, self.min_depth, self.max_depth)

        ds = self.env_frame_width // self.frame_width  # Downscaling factor
        if ds != 1:
            rgb = np.asarray(self.res(rgb.astype(np.uint8)))
            depth = depth[ds // 2::ds, ds // 2::ds]

        depth = np.expand_dims(depth, axis=2)
        state = np.concatenate((rgb, depth),
                               axis=2).transpose(2, 0, 1)

        return state

    def _preprocess_depth(self, depth, min_d, max_d):
        depth = depth[:, :, 0] * 1

        # lj dataset
        for i in range(depth.shape[0]):
            depth[i, :][depth[i, :] == 0.] = depth[i, :].max() + 0.01

        mask2 = depth > 0.99
        depth[mask2] = 0.

        mask1 = depth == 0
        depth[mask1] = 100.0
        depth = min_d * 100.0 + depth * max_d * 100.0
        return depth
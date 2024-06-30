import copy
import gzip
import json
import itertools
from collections import defaultdict
from glob import glob
from typing import Iterator
import multiprocessing as mp

import quaternion
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import T_co
from torch.utils.data import Dataset, DataLoader, IterableDataset

from habitat import get_config, make_dataset
from habitat.sims import make_sim
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.datasets.pointnav.pointnav_generator import generate_pointnav_episode
from habitat.tasks.nav.nav import merge_sim_episode_config

from odometry.dataset.utils import get_relative_egomotion
from odometry.utils import set_random_seed


class EgoMotionDataset(Dataset):
    TURN_LEFT = 'TURN_LEFT'
    TURN_RIGHT = 'TURN_RIGHT'
    MOVE_FORWARD = 'MOVE_FORWARD'
    ROTATION_ACTIONS = ['TURN_LEFT', 'TURN_RIGHT']
    INVERSE_ACTION = {
        'TURN_LEFT': 'TURN_RIGHT',
        'TURN_RIGHT': 'TURN_LEFT'
    }
    ACTION_TO_ID = {
        'STOP': 0,
        'MOVE_FORWARD': 1,
        'TURN_LEFT': 2,
        'TURN_RIGHT': 3
    }

    def __init__(
            self,
            data_root,
            environment_dataset,
            split,
            transforms,
            num_points=None,
            invert_rotations=False,
            augmentations=None,
            not_use_turn_left=False,
            not_use_turn_right=False,
            not_use_move_forward=False,
            invert_collisions=False,
            not_use_rgb=False,
    ):
        super().__init__()
        self.data_root = data_root
        self.environment_dataset = environment_dataset
        self.split = split
        self.transforms = transforms
        self.augmentations = augmentations
        self.not_use_turn_left = not_use_turn_left
        self.not_use_turn_right = not_use_turn_right
        self.not_use_move_forward = not_use_move_forward
        self.not_use_rgb = not_use_rgb
        self.jsons = self._load_metadata()
        self.invert_collisions = invert_collisions
        if invert_rotations:
            self._add_inverse_rotations()
        self.num_dataset_points = num_points or len(self.jsons)
        self.metadata = self.jsons[:self.num_dataset_points]

    def _get_metadata_file_paths(self):
        return glob(f'{self.data_root}/{self.environment_dataset}/{self.split}/*.json')

    def _load_metadata_file(self, path):
        with open(path, 'r') as file:
            content = json.load(file)

        return content

    def _load_metadata(self):
        data = []

        for file_path in self._get_metadata_file_paths():
            scene_content = self._load_metadata_file(file_path)

            scene_dataset = scene_content['dataset']
            if self.not_use_turn_left:
                scene_dataset = [
                    frames for frames in scene_dataset
                    if frames['action'][0] != self.TURN_LEFT
                ]
            if self.not_use_turn_right:
                scene_dataset = [
                    frames for frames in scene_dataset
                    if frames['action'][0] != self.TURN_RIGHT
                ]
            if self.not_use_move_forward:
                scene_dataset = [
                    frames for frames in scene_dataset
                    if frames['action'][0] != self.MOVE_FORWARD
                ]

            data += scene_dataset

        return data

    def _add_inverse_rotations(self):
        new_jsons = []
        for item in self.jsons:
            new_jsons.append(item)
            action = item['action'][0]
            if action in self.ROTATION_ACTIONS:
                if item['collision'] and (not self.invert_collisions):
                    continue
                inv = copy.deepcopy(item)
                inv['action'][0] = self.INVERSE_ACTION[action]
                inv = self._swap_values(inv, 'source_frame_path', 'target_frame_path')
                inv = self._swap_values(inv, 'source_depth_map_path', 'target_depth_map_path')
                inv = self._swap_values(inv, 'source_agent_state', 'target_agent_state')
                new_jsons.append(inv)

        self.jsons = new_jsons

    def get_label(self, index):
        meta = self.metadata[index]

        return meta['action'][0]

    def __getitem__(self, index):
        meta = self.metadata[index]

        source_depth = self.read_depth(meta['source_depth_map_path'])
        target_depth = self.read_depth(meta['target_depth_map_path'])

        item = {
            'source_depth': source_depth,
            'target_depth': target_depth,
            'action': self.ACTION_TO_ID[meta['action'][0]] - 1,  # shift action ids by 1 as we don't use STOP
            'collision': int(meta['collision']),
            'egomotion': get_relative_egomotion(meta),
        }
        if not self.not_use_rgb:
            item['source_rgb'] = self.read_rgb(meta['source_frame_path'])
            item['target_rgb'] = self.read_rgb(meta['target_frame_path'])

        if self.augmentations is not None:
            item = self.augmentations(item)

        item = self.transforms(item)

        return item

    def __len__(self):
        return self.num_dataset_points

    @staticmethod
    def read_rgb(path):
        return np.asarray(Image.open(path).convert('RGB'))

    @staticmethod
    def read_depth(path):
        return np.load(path)

    @staticmethod
    def _swap_values(item, k1, k2):
        item[k1], item[k2] = item[k2], item[k1]

        return item

    @classmethod
    def from_config(cls, config, transforms, augmentations=None):
        dataset_params = config.params
        return cls(
            data_root=dataset_params.data_root,
            environment_dataset=dataset_params.environment_dataset,
            split=dataset_params.split,
            transforms=transforms,
            num_points=dataset_params.num_points,
            invert_rotations=dataset_params.invert_rotations,
            augmentations=augmentations,
            not_use_turn_left=dataset_params.not_use_turn_left,
            not_use_turn_right=dataset_params.not_use_turn_right,
            not_use_move_forward=dataset_params.not_use_move_forward,
            invert_collisions=dataset_params.invert_collisions,
            not_use_rgb=dataset_params.not_use_rgb
        )


class EgoMotionDatasetResized(EgoMotionDataset):
    def _get_metadata_file_paths(self):
        return glob(f'{self.data_root}/{self.environment_dataset}/{self.split}/json/*.json.gz')

    def _load_metadata_file(self, path):
        with gzip.open(path, 'rt') as file:
            content = json.loads(file.read())

        return content

    @staticmethod
    def read_depth(path):
        scale = np.iinfo(np.uint16).max
        return np.expand_dims(cv2.imread(path, cv2.IMREAD_UNCHANGED), axis=2).astype(np.float32) / scale

    @staticmethod
    def read_rgb(path):
        return cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class HSimDataset(IterableDataset):
    ACTION_TO_ID = {
        'STOP': 0,
        'MOVE_FORWARD': 1,
        'TURN_LEFT': 2,
        'TURN_RIGHT': 3
    }

    def __init__(
            self,
            config_file_path,
            transforms,
            seed,
            augmentations=None,
            batch_size=None,
            local_rank=None,
            world_size=None,
            pairs_frac_per_episode=0.2,
            n_episodes_per_scene=3
    ):
        self.config_file_path = config_file_path
        self.local_rank = local_rank
        self.world_size = world_size
        self.start = None
        self.stop = None
        self.sim = None
        self.config = None
        self.transforms = transforms
        self.augmentations = augmentations
        self.batch_size = batch_size
        self.pairs_frac_per_episode = pairs_frac_per_episode
        self.n_episodes_per_scene = n_episodes_per_scene
        self.seed = seed
        self.torch_loader_worker_info = None

        self.config = get_config(self.config_file_path)
        dataset = make_dataset(
            id_dataset=self.config.DATASET.TYPE,
            config=self.config.DATASET
        )
        self.scene_ids = sorted(copy.deepcopy(dataset.scene_ids))
        del dataset

    def __iter__(self) -> Iterator[T_co]:
        self.torch_loader_worker_info = torch.utils.data.get_worker_info()
        self.seed += self.torch_loader_worker_info.id

        if self.local_rank is not None:
            self.config.defrost()
            self.config.SEED = self.seed
            self.config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = self.local_rank
            self.config.freeze()

        set_random_seed(self.seed)

        self.sim = make_sim(
            id_sim=self.config.SIMULATOR.TYPE,
            config=self.config.SIMULATOR
        )
        spf = ShortestPathFollower(
            sim=self.sim,
            goal_radius=self.config.TASK.SUCCESS.SUCCESS_DISTANCE,
            return_one_hot=False
        )

        scene_ids = self.get_loader_worker_split(self.scene_ids)
        scene_id_gen = itertools.cycle(scene_ids)
        episode_gen = generate_pointnav_episode(
            sim=self.sim,
            is_gen_shortest_path=False
        )

        obs_pairs = []
        while True:
            current_scene_id = next(scene_id_gen)
            self.reconfigure_scene(current_scene_id)

            for _ in range(self.n_episodes_per_scene):
                current_episode = next(episode_gen)
                self.reconfigure_episode(current_episode)

                obs = self.sim.reset()
                agent_state = self.sim.get_agent_state()

                ep_buffer = defaultdict(list)
                ep_buffer['observations'].append(obs)
                ep_buffer['sim_states'].append(agent_state)

                action = spf.get_next_action(current_episode.goals[0].position)
                while action != self.ACTION_TO_ID['STOP']:
                    obs = self.sim.step(action)
                    agent_state = self.sim.get_agent_state()

                    ep_buffer['actions'].append(action)
                    ep_buffer['observations'].append(obs)
                    ep_buffer['sim_states'].append(agent_state)
                    ep_buffer['collisions'].append(self.sim.previous_step_collided)

                    action = spf.get_next_action(current_episode.goals[0].position)

                n_episode_steps = len(ep_buffer['observations'])
                n_pairs_to_sample = int(np.ceil(self.pairs_frac_per_episode * n_episode_steps))

                indices = list(range(n_episode_steps - 1))
                np.random.shuffle(indices)
                sample_indices = indices[:n_pairs_to_sample]

                for i in sample_indices:
                    source_obs = ep_buffer['observations'][i]
                    target_obs = ep_buffer['observations'][i + 1]

                    source_state = ep_buffer['sim_states'][i]
                    target_state = ep_buffer['sim_states'][i + 1]

                    action = ep_buffer['actions'][i]
                    collision = ep_buffer['collisions'][i]

                    obs_pairs.append({
                        'source_depth': source_obs['depth'],
                        'target_depth': target_obs['depth'],
                        'source_rgb': source_obs['rgb'],
                        'target_rgb': target_obs['rgb'],
                        'action': action - 1,  # shift action ids by 1 as we don't use STOP
                        'collision': int(collision),
                        'egomotion': get_relative_egomotion({
                            'source_agent_state': {
                                'position': source_state.position.tolist(),
                                'rotation': quaternion.as_float_array(source_state.rotation).tolist()
                            },
                            'target_agent_state': {
                                'position': target_state.position.tolist(),
                                'rotation': quaternion.as_float_array(target_state.rotation).tolist()
                            }
                        })
                    })

                while len(obs_pairs) >= self.batch_size:
                    batch = obs_pairs[:self.batch_size]
                    del obs_pairs[:self.batch_size]

                    if self.augmentations is not None:
                        batch = [self.augmentations(item) for item in batch]

                    batch = [self.transforms(item) for item in batch]

                    collated = default_collate(batch)
                    yield collated

    def reconfigure_scene(self, scene_id):
        self.config.defrost()
        self.config.SIMULATOR.SCENE = scene_id
        self.config.freeze()
        self.sim.reconfigure(self.config.SIMULATOR)

    def reconfigure_episode(self, episode):
        self.config.defrost()
        self.config.SIMULATOR = merge_sim_episode_config(
            self.config.SIMULATOR,
            episode
        )
        self.config.freeze()
        self.sim.reconfigure(self.config.SIMULATOR)

    def get_loader_worker_split(self, scene_ids):
        if self.local_rank is not None:
            distrib_worker_start, distrib_worker_stop = self.split_workload(
                num_items=len(scene_ids),
                worker_id=self.local_rank,
                num_workers=self.world_size
            )
            scene_ids = scene_ids[distrib_worker_start:distrib_worker_stop]

        if self.torch_loader_worker_info is not None:
            loader_worker_start, loader_worker_stop = self.split_workload(
                num_items=len(scene_ids),
                worker_id=self.torch_loader_worker_info.id,
                num_workers=self.torch_loader_worker_info.num_workers
            )
            scene_ids = scene_ids[loader_worker_start:loader_worker_stop]

        return scene_ids

    @staticmethod
    def split_workload(num_items, worker_id, num_workers):
        per_worker = int(np.ceil(num_items / num_workers))
        iter_start = worker_id * per_worker
        iter_stop = min(iter_start + per_worker, num_items)

        return iter_start, iter_stop

    @classmethod
    def from_config(cls, config, transforms, augmentations=None):
        dataset_params = config.params
        return cls(
            transforms=transforms,
            augmentations=augmentations,
            **dataset_params
        )


class EgoDataLoader(DataLoader):
    @classmethod
    def from_config(cls, config, dataset, sampler):
        loader_params = config.params
        mp_context_name = loader_params.pop('multiprocessing_context', 'fork')
        mp_context = mp.get_context(mp_context_name) if loader_params.num_workers > 0 else None
        return cls(
            dataset=dataset,
            sampler=sampler,
            multiprocessing_context=mp_context,
            **loader_params
        )

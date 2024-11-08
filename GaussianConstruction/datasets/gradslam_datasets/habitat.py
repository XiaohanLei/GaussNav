import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numba
import numpy as np
import torch
from natsort import natsorted
import quaternion


from .basedataset import GradSLAMDataset

def from_transformation_to_xyo(transform):
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

numba.jit(nopython=True)
def get_evenly_distributed_subset(xyo: np.ndarray):
    '''
    xyo: n * 3 defines the spatial location
    '''
    num_points = xyo.shape[0]
    grid_resolution = 1.5 # 1.5m for grid resolution
    angle_resolution = np.deg2rad(20) # 30 for angle resolution

    min_x = np.min(xyo[:, 0])
    max_x = np.max(xyo[:, 0])
    min_y = np.min(xyo[:, 1])
    max_y = np.max(xyo[:, 1])

    grid_x_range = int(np.ceil((max_x - min_x) / grid_resolution))
    grid_y_range = int(np.ceil((max_y - min_y) / grid_resolution))
    grid_o_range = int(np.ceil(2 * np.pi / angle_resolution))

    hash_table = np.zeros((grid_x_range, grid_y_range, grid_o_range, 4))
    count_subset = 0
    index_subset = np.zeros((num_points, ))
    
    for i in range(num_points):
        curr_xyo = xyo[i]
        grid_x = int(np.floor((curr_xyo[0] - min_x) / grid_resolution))
        grid_y = int(np.floor((curr_xyo[1] - min_y) / grid_resolution))
        grid_o = int(np.floor((curr_xyo[2] - (-np.pi)) / angle_resolution))
        if hash_table[grid_x, grid_y, grid_o, 3] == 0:
            hash_table[grid_x, grid_y, grid_o, :3] = curr_xyo
            hash_table[grid_x, grid_y, grid_o, 3] = 1
            index_subset[count_subset] = i
            count_subset += 1
    subset = np.zeros((count_subset, ))
    for i in range(count_subset):
        subset[i] = index_subset[i]        


    return subset

def load_frames_subset(frames: list):
    xyo = np.zeros((len(frames), 3))
    for i in range(len(frames)):
        curr_trans = np.array(frames[i]['transform_matrix'])
        curr_xyo = from_transformation_to_xyo(curr_trans)
        xyo[i] = curr_xyo
    select_index = get_evenly_distributed_subset(xyo)
    subset_frames = [frames[int(i)] for i in select_index]
    return subset_frames



def create_filepath_index_mapping(frames):
    return {frame["file_path"]: index for index, frame in enumerate(frames)}


class HabitatDataset(GradSLAMDataset):
    def __init__(
        self,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 1440,
        desired_width: Optional[int] = 1920,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        config_dict = {}
        config_dict["dataset_name"] = "habitat"
        self.pose_path = None
        
        # Load NeRFStudio format camera & poses data
        self.cams_metadata = self.load_cams_metadata()
        self.frames_metadata = self.cams_metadata["frames"]
        # self.frames_metadata = load_frames_subset(self.cams_metadata["frames"])

        self.filepath_index_mapping = create_filepath_index_mapping(self.frames_metadata)

        # Load RGB & Depth filepaths
        # self.image_names = natsorted(os.listdir(f"{self.input_folder}/rgb"))
        # self.image_names = [f'rgb/{image_name}' for image_name in self.image_names]
        self.image_names = [self.frames_metadata[i]['file_path'] for i in range(len(self.frames_metadata))]
        self.seg_npy_names = [image_name.replace('rgb', 'seg').replace('.png', '.npy') for image_name in self.image_names]

        # Init Intrinsics
        config_dict["camera_params"] = {}
        config_dict["camera_params"]["png_depth_scale"] = 1000.0 # Depth is in mm
        config_dict["camera_params"]["image_height"] = self.cams_metadata["h"]
        config_dict["camera_params"]["image_width"] = self.cams_metadata["w"]
        config_dict["camera_params"]["fx"] = self.cams_metadata["fl_x"]
        config_dict["camera_params"]["fy"] = self.cams_metadata["fl_y"]
        config_dict["camera_params"]["cx"] = self.cams_metadata["cx"]
        config_dict["camera_params"]["cy"] = self.cams_metadata["cy"]

        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        ) 

    def load_cams_metadata(self):
        cams_metadata_path = f"{self.input_folder}/transforms.json"
        cams_metadata = json.load(open(cams_metadata_path, "r"))
        return cams_metadata
    
    def get_filepaths(self):
        base_path = f"{self.input_folder}"
        color_paths = []
        depth_paths = []
        seg_paths = []
        self.tmp_poses = []
        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        ).float()
        for i, image_name in enumerate(self.image_names):
            # Search for image name in frames_metadata
            frame_metadata = self.frames_metadata[self.filepath_index_mapping.get(image_name)]
            # Get path of image and depth
            color_path = f"{base_path}/{image_name}"
            # depth_path = f"{base_path}/{image_name.replace('rgb', 'depth')}"
            depth_path = f"{base_path}/{frame_metadata['depth_file_path']}"
            seg_path = f"{base_path}/{self.seg_npy_names[i]}"
            color_paths.append(color_path)
            depth_paths.append(depth_path)
            seg_paths.append(seg_path)
            # Get pose of image in GradSLAM format
            c2w = torch.from_numpy(np.array(frame_metadata["transform_matrix"])).float()
            _pose = P @ c2w @ P.T
            # _pose = torch.linalg.inv(_pose)
            self.tmp_poses.append(_pose)
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{base_path}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, seg_paths, embedding_paths

    def load_poses(self):
        return self.tmp_poses

    def read_embedding_from_file(self, embedding_file_path):
        print(embedding_file_path)
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)

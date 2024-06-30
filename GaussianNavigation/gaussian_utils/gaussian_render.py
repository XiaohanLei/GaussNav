import argparse
import os
import sys
current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
sys.path.append(parent_path)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from copy import deepcopy

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera


from gaussian_utils.slam_helpers import get_depth_and_silhouette
from gaussian_utils.slam_external import build_rotation
from gaussian_utils.recon_helpers import setup_camera, setup_camera_cuda, setup_camera_cuda_customized
from gaussian_utils.slam_helpers import transform_to_frame, transformed_params2rendervar, transformed_params2depthplussilhouette

import torch
from glob import glob
from tqdm import tqdm

from natsort import natsorted


def load_camera(cfg, scene_path):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    org_width = params['org_width']
    org_height = params['org_height']
    w2c = params['w2c']
    intrinsics = params['intrinsics']
    k = intrinsics[:3, :3]

    # Scale intrinsics to match the visualization resolution
    k[0, :] *= cfg['viz_w'] / org_width
    k[1, :] *= cfg['viz_h'] / org_height
    return w2c, k


def load_scene_data(scene_path):
    # Load Scene Data
    all_params = dict(np.load(scene_path, allow_pickle=True))
    all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}
    params = all_params

    # all_w2cs = []
    # num_t = params['cam_unnorm_rots'].shape[-1]
    # for t_i in range(num_t):
    #     cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
    #     cam_tran = params['cam_trans'][..., t_i]
    #     rel_w2c = torch.eye(4).cuda().float()
    #     rel_w2c[:3, :3] = build_rotation(cam_rot)
    #     rel_w2c[:3, 3] = cam_tran
    #     all_w2cs.append(rel_w2c.cpu().numpy())
    all_w2cs = params['gt_w2c_all_frames']
    
    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]

    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    return params, all_w2cs


def get_rendervars(params, w2c, curr_timestep):
    params_timesteps = params['timestep']
    selected_params_idx = params_timesteps <= curr_timestep
    keys = [k for k in params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]
    selected_params = deepcopy(params)
    for k in keys:
        selected_params[k] = selected_params[k][selected_params_idx]
    transformed_pts = selected_params['means3D']
    w2c = torch.tensor(w2c).cuda().float()
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': selected_params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
        'opacities': torch.sigmoid(selected_params['logit_opacities']),
        'scales': torch.exp(torch.tile(selected_params['log_scales'], (1, 3))),
        # 'means2D': torch.zeros_like(selected_params['means3D'], device="cuda")
        'means2D': torch.zeros_like(selected_params['means3D'], requires_grad=True, device="cuda") + 0
    }
    depth_rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, w2c),
        'rotations': torch.nn.functional.normalize(selected_params['unnorm_rotations']),
        'opacities': torch.sigmoid(selected_params['logit_opacities']),
        'scales': torch.exp(torch.tile(selected_params['log_scales'], (1, 3))),
        # 'means2D': torch.zeros_like(selected_params['means3D'], device="cuda")
        'means2D': torch.zeros_like(selected_params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar, depth_rendervar

def get_rendervars_cuda(params, w2c, curr_timestep):
    '''
    the w2c here should be a tensor that requires grad
    '''


    # params_timesteps = params['timestep']
    # selected_params_idx = params_timesteps <= curr_timestep
    # keys = [k for k in params.keys() if
    #         k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
    #                   'gt_w2c_all_frames', 'cam_unnorm_rots',
    #                   'cam_trans', 'keyframe_time_indices']]
    # selected_params = deepcopy(params)
    # for k in keys:
    #     selected_params[k] = selected_params[k][selected_params_idx]


    transformed_pts = params['means3D']
    # w2c = torch.tensor(w2c).cuda().float()
    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
        # 'means2D': torch.zeros_like(selected_params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar


def render(w2c, k, timestep_data, cfg):
    with torch.no_grad():
        # cam = setup_camera(cfg['viz_w'], cfg['viz_h'], k, w2c, cfg['viz_near'], cfg['viz_far'])
        # cam = setup_camera_cuda(cfg['viz_w'], cfg['viz_h'], k, w2c, cfg['viz_near'], cfg['viz_far'])
        cam = setup_camera_cuda_customized(cfg['viz_w'], cfg['viz_h'], w2c, cfg['viz_near'], cfg['viz_far'])
        white_bg_cam = Camera(
            image_height=cam.image_height,
            image_width=cam.image_width,
            tanfovx=cam.tanfovx,
            tanfovy=cam.tanfovy,
            bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
            scale_modifier=cam.scale_modifier,
            viewmatrix=cam.viewmatrix,
            projmatrix=cam.projmatrix,
            sh_degree=cam.sh_degree,
            campos=cam.campos,
            prefiltered=cam.prefiltered
        )
        im, _, depth, = Renderer(raster_settings=white_bg_cam)(**timestep_data)
        return im, depth
    
def load_scene(scene_path, load_multiple=1, load_list=[]):

    cfg = dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=480, viz_h=640,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=False, # Enter Interactive Mode after Online Recon Viz
    )
    if load_multiple == 1:
        all_params = dict(np.load(scene_path, allow_pickle=True))
    else:
        for index, scene_path in enumerate(load_list):
            if index == 0:
                all_params = dict(np.load(scene_path, allow_pickle=True))
            else:
                new_params = dict(np.load(scene_path, allow_pickle=True))
                for key in ['means3D', 'rgb_colors', 'log_scales', 'logit_opacities', 'gt_w2c_all_frames']:
                    all_params[key] = np.concatenate((all_params[key], new_params[key]), axis=0)
                for key in ['cam_trans', 'cam_unnorm_rots']:
                    all_params[key] = np.concatenate((all_params[key], new_params[key]), axis=-1)
                del new_params

    params = all_params
    org_width = params['org_width']
    org_height = params['org_height']
    first_frame_w2c = params['w2c']
    intrinsics = params['intrinsics']
    ks = intrinsics[:3, :3] * 1.

    # Scale intrinsics to match the visualization resolution
    ks[0, :] *= cfg['viz_w'] / org_width
    ks[1, :] *= cfg['viz_h'] / org_height

    all_params = {k: torch.tensor(all_params[k]).cuda().float() for k in all_params.keys()}
    params = all_params

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c)
    
    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics', 
                      'gt_w2c_all_frames', 'cam_unnorm_rots',
                      'cam_trans', 'keyframe_time_indices']]

    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    transformed_pts = transform_to_frame(params, 0, 
                                             gaussians_grad=False,
                                             camera_grad=False)
    rendervar = transformed_params2rendervar(params, transformed_pts)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, params['w2c'],
                                                                 transformed_pts)

    return {
        'first_frame_w2c': first_frame_w2c,
        'k': ks,
        'params': params,
        'all_w2cs': all_w2cs,
        'rendervar': rendervar,
        'depth_sil_rendervar': depth_sil_rendervar,
        'cfg': cfg
        }


def get_gauss_observation_at(transformation, scene_data):
    first_frame_w2c = scene_data['first_frame_w2c']
    k = scene_data['k']
    params = scene_data['params']
    all_w2cs = scene_data['all_w2cs'] 
    rendervar = scene_data['rendervar']
    depth_sil_rendervar = scene_data['depth_sil_rendervar']
    cfg = scene_data['cfg']

    with torch.no_grad():
        im, depth, sil = render(transformation, k, rendervar, depth_sil_rendervar, cfg)
    rgb = (255*im.cpu().numpy().transpose(1,2,0)).astype(np.uint8)
    
    return rgb


def render_observation_at(scene_path, save_dir="outputs/0"):

    scenes_list = glob(os.path.join(scene_path, '*'))
    print(scenes_list)
    scene_params_list = []
    for scene in scenes_list:
        scene_params_list.append(glob(os.path.join(scene, '*.npz')))

    # default set the cfg:
    cfg = dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=480, viz_h=640,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=False, # Enter Interactive Mode after Online Recon Viz
    )

    for index, scene in tqdm(enumerate(scene_params_list)):
        scene_name = scenes_list[index].split('/')[-1]
        base_save_dir = f'rendered_images/{scene_name}'
        scene = natsorted(scene)
        for kk, scene_param_path in tqdm(enumerate(scene)):

            first_frame_w2c, k = load_camera(cfg, scene_param_path)
            first_frame_w2c = np.eye(4)
            # first_frame_w2c[2, 3] += 0.5
            first_frame_w2c = torch.from_numpy(first_frame_w2c).cuda().float()
            params, all_w2cs = load_scene_data(scene_param_path)

            k = params['intrinsics'][:3, :3]

            save_dir = os.path.join(base_save_dir, str(kk))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # save all frames
            for i in range(len(all_w2cs)):


                view_w2c = first_frame_w2c @ all_w2cs[i]
                debug_matrix = torch.linalg.inv(all_w2cs[i])

                rendervar = get_rendervars_cuda(params, view_w2c, i)

                im, depth = render(view_w2c, k, rendervar, cfg)
                # print(f"Saving frame {i}")
                cv2.imwrite(f"{save_dir}/{i}.png", (255*im.cpu().numpy().transpose(1,2,0)[..., [2, 1, 0]]).astype(np.uint8))
                cv2.imwrite(f"{save_dir}/{i}_depth.png", (255*depth.cpu().numpy().transpose(1,2,0)).astype(np.uint8))
                # cv2.imwrite(f"{save_dir}/{i}_sil.png", (255*sil.cpu().numpy().transpose(1,2,0)).astype(np.uint8))

    # return im, depth, sil


if __name__ == "__main__":


    scene_path = '/instance_imagenav/gaussian/SplaTAM/experiments/habitat'
    # scene_list = ['/instance_imagenav/gaussian/SplaTAM/experiments/habitat/4ok3usBNeis-floor--0.5/4ok3usBNeis-floor--0.5_3.npz',
    #               '/instance_imagenav/gaussian/SplaTAM/experiments/habitat/4ok3usBNeis-floor--0.5/4ok3usBNeis-floor--0.5_4.npz',]
    save_dir="outputs/3"
    # if not os.path.exists(save_dir):
    #     os.makedirs(save_dir)
    
    render_observation_at(scene_path, save_dir)
    # scene = load_scene(scene_path, load_multiple=len(scene_list), load_list=scene_list)
    # all_w2cs = scene['all_w2cs']
    # for index, pos in enumerate(all_w2cs):
    #     rgb = get_gauss_observation_at(pos, scene_data=scene)
    #     cv2.imwrite(f'{save_dir}/{index}.png', rgb[:, :, ::-1])

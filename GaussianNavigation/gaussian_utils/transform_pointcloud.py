import sys
import os.path as osp
import torch
import numpy as np
import cv2
from glob import glob
import os
from natsort import natsorted
from tqdm import tqdm
import skimage
import skimage.morphology
import json

color_mapping = {
    56: [0.9400000000000001, 0.7818, 0.66],
    57: [0.8882000000000001, 0.9400000000000001, 0.66],
    58: [0.66, 0.9400000000000001, 0.8518000000000001],
    59: [0.7117999999999999, 0.66, 0.9400000000000001],
    61: [0.9218, 0.66, 0.9400000000000001],
    62: [0.9400000000000001, 0.66, 0.748199999999999]
}

def preprocess_mask(depth, mask, conf=0.5):

    new_mask = {}
    for key in mask:
        if mask[key] is not None:
            new_curr_mask_list = []
            curr_mask_list = mask[key]
            for curr_mask in curr_mask_list:
                selem = skimage.morphology.disk(3)
                erosed_mask = skimage.morphology.erosion(curr_mask, selem)
                if np.any(erosed_mask):
                    new_depth = depth.copy()
                    new_masked_depth = new_depth[erosed_mask]

                    mean = np.mean(new_masked_depth)
                    std = np.std(new_masked_depth)

                    threshold = conf * std
                    curr_mask_copy = curr_mask.copy()
                    curr_mask_filtered = np.where(np.abs(depth - mean) > threshold, False, curr_mask)
                    curr_mask_filtered = np.logical_and(curr_mask_copy, curr_mask_filtered)

                    if np.any(curr_mask_filtered):
                        new_curr_mask_list.append(curr_mask_filtered)

            if len(new_curr_mask_list) > 0:
                new_mask[key] = new_curr_mask_list
            else:
                new_mask[key] = None
        else:
            new_mask[key] = None

    return new_mask
                        

def get_pointcloud(mask, depth, w2c, transform_pts=True,\
                   return_bbox=True, depth_scale=1000.0):
    if isinstance(depth, np.ndarray):
        depth = torch.from_numpy(depth.astype(float)).cuda()


    width, height = depth.shape[1], depth.shape[0]
    depth = depth / depth_scale
    CX = 239.5
    CY = 319.5
    FX = 625.2213755265124
    FY = 625.2213755265124

    # Compute indices of pixels
    x_grid, y_grid = torch.meshgrid(torch.arange(width).cuda().float(), 
                                    torch.arange(height).cuda().float(),
                                    indexing='xy')
    xx = (x_grid - CX)/FX
    yy = (y_grid - CY)/FY
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    depth_z = depth.reshape(-1)

    # Initialize point cloud
    pts_cam = torch.stack((xx * depth_z, - yy * depth_z, - depth_z), dim=-1)
    if transform_pts:
        P = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]
            ]
        , dtype=float).cuda()
        pix_ones = torch.ones(height * width, 1).cuda().float()
        pts4 = torch.cat((pts_cam, pix_ones), dim=1)
        # transform pointcloud to habitat coordinate camera's frame
        
        pts = (w2c @ pts4.T).T[:, :3]

    else:
        pts = pts_cam

    
    # Colorize point cloud
    if return_bbox:
        instance_bbox = {}
    cols = torch.ones((width * height, 3), device='cuda:0', dtype=float) * 0.9
    for key in mask:
        if mask[key] is not None:
            if return_bbox:
                curr_bbox = []
            for curr_mask in mask[key]:
                curr_cols = torch.from_numpy(curr_mask).cuda()
                curr_cols = curr_cols.reshape(-1)
                cols[curr_cols] = torch.tensor(color_mapping[key], dtype=float).cuda()
                if return_bbox:
                    masked_x = torch.masked_select(pts[:, 0], curr_cols)
                    masked_y = torch.masked_select(pts[:, 1], curr_cols)
                    masked_z = torch.masked_select(pts[:, 2], curr_cols)
                    min_x = (torch.min(masked_x)).item()
                    max_x = (torch.max(masked_x)).item()
                    min_y = (torch.min(masked_y)).item()
                    max_y = (torch.max(masked_y)).item()
                    min_z = (torch.min(masked_z)).item()
                    max_z = (torch.max(masked_z)).item()
                    curr_bbox.append([min_x,max_x, min_y, max_y, min_z, max_z])
            if return_bbox:
                instance_bbox[key] = curr_bbox
        else:
            if return_bbox:
                instance_bbox[key] = None

        
    # cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3) # (C, H, W) -> (H, W, C) -> (H * W, C)
    # point_cld = torch.cat((pts, cols), -1)

    if return_bbox:
        return pts, cols, instance_bbox
    return pts, cols
    

def voxelize(pointcloud, voxel_size=0.05):
    xyz_min_bounds = pointcloud.min(0)[0]
    xyz_max_bounds = pointcloud.max(0)[0]

    grid_sizes = ((xyz_max_bounds - xyz_min_bounds) / voxel_size).long() + 1
    voxel_grid = torch.zeros(tuple(grid_sizes.tolist())).cuda().float()

    indices = (pointcloud - xyz_min_bounds) / voxel_size
    indices = indices.long()

    voxel_grid[indices[:, 0], indices[:, 1], indices[:, 2]] += 1
    voxel_grid = torch.clamp(voxel_grid, min=0, max=1)

    return voxel_grid, xyz_min_bounds.cpu().numpy(), xyz_max_bounds.cpu().numpy()



if __name__ == "__main__":


    scenes_list = glob(os.path.join('/instance_imagenav/end2end_imagenav/env_collect_v1', '*'))
    for scene in tqdm(scenes_list):
        print(f'**********{scene}***********')
        json_dump_data = {56: [], 57: [], 58: [], 59: [], 61: [], 62: []}
        frames = json.load(open(os.path.join(scene, 'transforms.json'), "r"))['frames']
        all_w2cs = [torch.tensor(frames[i]['transform_matrix'], dtype=float).cuda() \
                    for i in range(len(frames))]
        seg_masks_list = glob(os.path.join(scene, 'seg_semantic', "*.npy"))
        seg_masks_list = natsorted(seg_masks_list)

        for seg_mask_path in tqdm(seg_masks_list):
            mask_data = (np.load(seg_mask_path, allow_pickle=True).item())
            index = ((seg_mask_path.split('/')[-1]).split('_')[-1]).split('.')[0]
            depth_name = index + '.png'
            depth_file_path = os.path.join(scene, 'depth', depth_name)
            depth_data = cv2.imread(depth_file_path, cv2.IMREAD_UNCHANGED)
            try:
                w2c = all_w2cs[int(index)]
            except:
                break
            mask_data = preprocess_mask(depth_data, mask_data)
            pts, cols, bbox = get_pointcloud(mask_data, depth_data, w2c)

            # voxelize pointclouds for min traversable point search
            voxel_size = 0.05
            voxel_grid, voxel_min_bounds, voxel_max_bounds = voxelize(pts)
            min_z = int((w2c[1, 3] - 0.4 - voxel_min_bounds[1]) / voxel_size)
            max_z = int((w2c[1, 3] + 0.2 - voxel_min_bounds[1]) / voxel_size)
            grid_map = torch.sum(voxel_grid[:, min_z:max_z, :], dim=1)
            grid_map[grid_map> 0] = 1
            grid_map = grid_map.cpu().numpy()
            selem = skimage.morphology.disk(3)
            dilated_grid_map = skimage.morphology.dilation(grid_map, selem)

            for key in bbox:
                if bbox[key] is not None:
                    navigable_viewpoint = []
                    for bbox_i in range(len(bbox[key])):
                        curr_bbox = bbox[key][bbox_i]
                        curr_center = [(curr_bbox[0] + curr_bbox[1])/2, (curr_bbox[4] + curr_bbox[5])/2]
                        map_curr_center = [int((curr_center[0] - voxel_min_bounds[0])/voxel_size), \
                                        int((curr_center[1] - voxel_min_bounds[2])/voxel_size)]
                        
                        try:
                            for r in range(3, 40):
                                navigeble_map = np.zeros_like(grid_map)
                                rr, cc = skimage.draw.ellipse(map_curr_center[0], map_curr_center[1],\
                                                                    r, r)
                                navigeble_map[rr, cc] = 1
                                temp_map = dilated_grid_map - navigeble_map
                                if np.any(temp_map == -1):
                                    navigable_rr, navigable_cc = np.where(temp_map == -1)
                                    world_rr, world_cc = navigable_rr[0] * voxel_size + voxel_min_bounds[0], \
                                                        navigable_cc[0] * voxel_size + voxel_min_bounds[2]
                                    navigable_viewpoint.append([world_rr, world_cc])
                                    break        
                        except:
                            navigable_viewpoint.append(None)        

                    json_dump_data[key].append([bbox[key], index, navigable_viewpoint])

        json_file_path = os.path.join(scene, 'instance_retrieval.json')
        with open(json_file_path, 'w') as json_file:
            json.dump(json_dump_data, json_file)

        
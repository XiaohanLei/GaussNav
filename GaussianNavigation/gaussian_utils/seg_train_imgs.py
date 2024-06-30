import sys
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector)
import mmcv
sys.path.append("/instance_imagenav/Object-Goal-Navigation/3rdparty/InternImage/detection")
import mmcv_custom  # noqa: F401,F403
import mmdet_custom  # noqa: F401,F403
import os.path as osp
import torch
import numpy as np
import cv2
from glob import glob
import os
from natsort import natsorted
from tqdm import tqdm


def get_sem_pred_internimg(internimg, rgb, use_seg=True, pred_bbox=False):
    coco_categories_mapping_reduce = {
        56: 0,  # chair
        57: 1,  # couch
        58: 2,  # potted plant
        59: 3,  # bed
        61: 4,  # toilet
        62: 5,  # tv
    }
    conf = 0.6

    bbox, mask = inference_detector(internimg, rgb)
    save_mask = {}

    for key in coco_categories_mapping_reduce:
        sel_bbox, sel_mask = bbox[key], mask[key]
        conf_mask = [sel_mask[i] for i in range(len(sel_mask)) if sel_bbox[i, 4] > conf]
        if len(conf_mask) > 0:
            save_mask[key] = conf_mask
        else:
            save_mask[key] = None
    
    bgr_vis = internimg.show_result(
                rgb,
                (bbox, mask),
                score_thr=conf,
                show=False,
                bbox_color='coco',
                text_color=(200, 200, 200),
                mask_color='coco',
                out_file=None
            )[:, :, ::-1]
    
    return save_mask, bgr_vis



if __name__ == "__main__":
    
    device = torch.device('cuda', 0)

    config = '/instance_imagenav/Object-Goal-Navigation/3rdparty/InternImage/detection/configs/coco/cascade_internimage_xl_fpn_3x_coco.py'
    checkpoint = '/instance_imagenav/Object-Goal-Navigation/pretrained_models/cascade_internimage_xl_fpn_3x_coco.pth'
    internimg = init_detector(config, checkpoint, device=device)

    scenes_list = glob(os.path.join('/instance_imagenav/end2end_imagenav/env_collect_v1', '*'))
    # for scene in tqdm(scenes_list):
    print(scenes_list)
    for i in tqdm(range(22, len(scenes_list))):
        scene = scenes_list[i]
        print(f'**********{scene}***********')
        rgbs_list = glob(os.path.join(scene, 'rgb', '*.png'))
        rgbs_list = natsorted(rgbs_list)
        save_dir = os.path.join(scene, 'seg_semantic')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for rgb_path in tqdm(rgbs_list):
            rgb = cv2.imread(rgb_path)[:, :, ::-1]
            save_file_name = rgb_path.split('/')[-1]
            save_file_name = save_file_name.split('.')[0]
            save_mask, save_bgr_vis = get_sem_pred_internimg(internimg, rgb)
            save_flag = False
            for key in save_mask:
                if save_mask[key] is not None:
                    save_flag = True
                    break
            if save_flag:
                np.save(os.path.join(save_dir, save_file_name + '.npy'), save_mask)
                cv2.imwrite(os.path.join(save_dir, save_file_name + '.png'), save_bgr_vis)
        
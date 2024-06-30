import random

import numpy as np
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transform_batch(batch):
    source_input, target_input = [], []

    source_depth_maps = batch['source_depth']
    target_depth_maps = batch['target_depth']
    source_input += [source_depth_maps]
    target_input += [target_depth_maps]

    if 'source_rgb' in batch:
        source_images = batch['source_rgb']
        target_images = batch['target_rgb']
        source_input += [source_images]
        target_input += [target_images]

    if all(key in batch for key in ['source_depth_discretized', 'target_depth_discretized']):
        source_d_depth = batch['source_depth_discretized']
        target_d_depth = batch['target_depth_discretized']
        source_input += [source_d_depth]
        target_input += [target_d_depth]

    concat_source_input = torch.cat(source_input, 1)
    concat_target_input = torch.cat(target_input, 1)
    transformed_batch = torch.cat(
        [
            concat_source_input,
            concat_target_input
        ],
        1
    )

    if 'egomotion' in batch:
        translation = batch['egomotion']['translation']
        rotation = batch['egomotion']['rotation'].view(translation.shape[0], -1)
        target = torch.cat(
            [
                translation,
                rotation
            ],
            1
        )
    else:
        target = None

    embeddings = {}
    if 'action' in batch:
        embeddings['action'] = batch['action']

    if 'collision' in batch:
        embeddings['collision'] = batch['collision']

    return transformed_batch, embeddings, target


def polar_to_cartesian(rho, phi):
    return np.array([rho * np.sin(-phi), 0, -rho * np.cos(-phi)])

import torch

action_id_to_action_name = {
    0: 'move_forward',
    1: 'turn_left',
    2: 'turn_right'
}


def translation_mae(pr_pose, gt_pose):
    batch_size = gt_pose.shape[0]

    pr_location = pr_pose[:, :-1]
    gt_location = gt_pose[:, :-1]

    return torch.abs(pr_location - gt_location).sum() / batch_size


def rotation_mae(pr_pose, gt_pose):
    batch_size = gt_pose.shape[0]

    pr_orientation = pr_pose[:, -1]
    gt_orientation = gt_pose[:, -1]

    return torch.abs(pr_orientation - gt_orientation).sum() / batch_size

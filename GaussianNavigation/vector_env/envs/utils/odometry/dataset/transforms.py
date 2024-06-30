import cv2
import numpy as np

import torch
import torchvision.transforms as torch_transforms
from PIL import Image


class DiscretizeDepth:
    def __init__(self, min_depth=None, max_depth=None, n_channels=5):
        self.n_channels = n_channels
        self.min_depth = min_depth
        self.max_depth = max_depth

    def __call__(self, data):
        if self.n_channels > 1:
            data.update({
                k.replace('depth', 'depth_discretized'): self._discretize(v) if isinstance(v, np.ndarray) else self._discretize_tensor(v)
                for k, v in data.items() if 'depth' in k
            })

        return data

    def _discretize(self, depth):
        if self.min_depth is None and self.max_depth is None:
            min_v, max_v = depth.min(), depth.max()
        else:
            min_v, max_v = self.min_depth, self.max_depth

        bins = np.linspace(min_v, max_v, num=self.n_channels + 1, endpoint=True)
        bins[-1] = np.finfo(bins.dtype).max

        lower_b = bins[:-1]
        upper_b = bins[1:]

        repeated_depth = depth.repeat(self.n_channels, axis=2)
        onehot_depth = np.logical_and(lower_b <= repeated_depth, repeated_depth < upper_b).astype(depth.dtype)

        return onehot_depth

    def _discretize_tensor(self, depth):
        if self.min_depth is None and self.max_depth is None:
            min_v, max_v = depth.min(), depth.max()
        else:
            min_v, max_v = self.min_depth, self.max_depth

        bins = torch.linspace(min_v, max_v, steps=self.n_channels + 1).to(depth.device)
        bins[-1] = torch.finfo(bins.dtype).max

        lower_b = bins[:-1]
        upper_b = bins[1:]

        repeated_depth = depth.repeat(1, 1, self.n_channels)
        onehot_depth = torch.logical_and(lower_b <= repeated_depth, repeated_depth < upper_b).type(depth.dtype)

        return onehot_depth


class ConvertToTensor:
    def __call__(self, data):
        data = {
            k: (
                torch.from_numpy(np.asarray(v, dtype=np.float32))
                if 'rgb' in k
                else v
            )
            for k, v in data.items()
        }

        data = {
            k: (
                torch.from_numpy(np.asarray(v, dtype=np.float32))
                if 'depth' in k
                else v
            )
            for k, v in data.items()
        }

        data = {
            k: (
                {
                    'position': torch.from_numpy(np.asarray(v['position'], dtype=np.float32)),
                    'rotation': torch.from_numpy(np.asarray(v['rotation'], dtype=np.float32)),
                }
                if 'state' in k
                else v
            )
            for k, v in data.items()
        }

        if 'egomotion' in data:
            data['egomotion']['translation'] = torch.from_numpy(
                np.asarray(
                    data['egomotion']['translation'],
                    dtype=np.float32
                )
            )

        if 'action' in data:
            data['action'] = torch.tensor(data['action'])

        if 'collision' in data:
            data['collision'] = torch.tensor(data['collision'])

        return data


class PermuteChannels:
    def __call__(self, data):
        data = {
            k: (v.permute(2, 0, 1) if 'rgb' in k else v)
            for k, v in data.items()
        }
        data = {
            k: (v.permute(2, 0, 1) if 'depth' in k else v)
            for k, v in data.items()
        }
        return data


class Normalize:
    def __call__(self, data):
        data = {
            k: (v / 255. if 'rgb' in k else v)
            for k, v in data.items()
        }

        # normalizing ego-motion rotation (-10 and +10 degrees)
        if 'egomotion' in data:
            rotation = data['egomotion']['rotation']
            if rotation > np.deg2rad(300):
                rotation -= (2 * np.pi)
            elif rotation < -np.deg2rad(300):
                rotation += (2 * np.pi)
            data['egomotion']['rotation'] = rotation

        return data


class Resize:
    def __init__(self, size, interpolation='BILINEAR'):
        self.resize = torch_transforms.Resize(size, getattr(Image, interpolation))

    def __call__(self, data):
        return {
            k: self.resize(v) if 'rgb' in k or 'depth' in k else v
            for k, v in data.items()
        }


class Crop:
    MOVE_FORWARD = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3

    def __init__(
            self,
            source_trans_keys=(
                    'source_rgb',
                    'source_depth',
                    'source_depth_discretized',
            ),
            target_trans_keys=(
                    'target_rgb',
                    'target_depth',
                    'target_depth_discretized'
            )
    ):
        self.source_trans_keys = source_trans_keys
        self.target_trans_keys = target_trans_keys
        self.crop_h = 320
        self.crop_w = 450
        self.padding = 20
        self.in_h = 360
        self.in_w = 640

    def __call__(self, data):
        action = data['action'].item() + 1
        if action == self.TURN_LEFT:
            source_i = self.padding
            source_j = self.padding
            target_i = self.padding
            target_j = self.in_w - self.padding - self.crop_w
        elif action == self.TURN_RIGHT:
            source_i = self.padding
            source_j = self.in_w - self.padding - self.crop_w
            target_i = self.padding
            target_j = self.padding
        else:
            source_i = self.in_h // 2 - self.crop_h // 2
            source_j = self.in_w // 2 - self.crop_w // 2
            target_i = source_i
            target_j = source_j

        for k, v in data.items():
            if k in self.source_trans_keys:
                data[k] = v[source_i:source_i+self.crop_h, source_j:source_j + self.crop_w, :]
            if k in self.target_trans_keys:
                data[k] = v[target_i:target_i + self.crop_h, target_j:target_j + self.crop_w, :]

        return data


class ToGray:
    def __init__(self, depth_mask=False):
        self.depth_mask = depth_mask

    def __call__(self, item):

        item['source_rgb'] = np.expand_dims(cv2.cvtColor(item['source_rgb'], cv2.COLOR_RGB2GRAY), 2)
        item['target_rgb'] = np.expand_dims(cv2.cvtColor(item['target_rgb'], cv2.COLOR_RGB2GRAY), 2)

        if self.depth_mask:
            item['source_rgb'] = item['source_rgb'] * (item['source_depth'] != 0)
            item['target_rgb'] = item['target_rgb'] * (item['target_depth'] != 0)

        return item

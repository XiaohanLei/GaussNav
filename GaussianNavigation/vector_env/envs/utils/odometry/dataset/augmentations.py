import random
import numpy as np
from habitat_sim.sensors.noise_models.redwood_depth_noise_model import RedwoodDepthNoiseModel
from habitat_sim.sensors.noise_models.gaussian_noise_model import GaussianNoiseModel


class RGBNoise:
    def __init__(self, intensity_constant):
        self.rgb_noise = GaussianNoiseModel(intensity_constant=intensity_constant)

    def __call__(self, item):
        item['source_rgb'] = self.rgb_noise.apply(item['source_rgb'])
        item['target_rgb'] = self.rgb_noise.apply(item['target_rgb'])
        return item


class DepthNoise:
    def __init__(self, noise_multiplier):
        self.depth_noise = RedwoodDepthNoiseModel(noise_multiplier=noise_multiplier)

    def __call__(self, item):
        item['source_depth'] = np.expand_dims(self.depth_noise.apply(np.squeeze(item['source_depth'], axis=2)), axis=2)
        item['target_depth'] = np.expand_dims(self.depth_noise.apply(np.squeeze(item['target_depth'], axis=2)), axis=2)
        return item


class VFlip:
    ROTATION_ACTIONS = {
        # 0   STOP
        # 1   MOVE_FORWARD
        2,  # TURN_LEFT
        3,  # TURN_RIGHT
    }
    INVERSE_ACTION = {
        2: 3,
        3: 2
    }

    def __init__(self, p=0.5):
        self.p = p
        self.flip_trans_keys = (
            "source_rgb",
            "target_rgb",
            "source_depth",
            "target_depth",
        )

    def vflip(self, img):
        return np.ascontiguousarray(img[:, ::-1, ...])

    def __call__(self, item):
        if random.random() < self.p:
            for k, v in item.items():
                if k in self.flip_trans_keys:
                    item[k] = self.vflip(v)
            item['egomotion']['translation'] *= np.array([-1, 1, 1])
            item['egomotion']['rotation'] *= -1
            if (item['action'] + 1) in self.ROTATION_ACTIONS:
                item['action'] = self.INVERSE_ACTION[item['action'] + 1] - 1

        return item

import os
import numpy as np
import habitat
import habitat_sim
from habitat.config.default import get_config
from habitat import make_dataset
from vector_env.envs.base_env import base_env
from utils.time import print_run_time
from vector_env.utils.vector_env import VectorEnv

def _get_scenes_from_folder(content_dir):
    scene_dataset_ext = ".json.gz"
    scenes = []
    for filename in os.listdir(content_dir):
        if filename.endswith(scene_dataset_ext):
            # scene = filename[: -len(scene_dataset_ext) + 4]
            scene = filename[: -len(scene_dataset_ext) ]
            scenes.append(scene)
    scenes.sort()
    return scenes

# @print_run_time
def _load_data(cfg_path: str):
    configs = []
    datasets = []
    basic_config = get_config(cfg_path)
    NUM_ENVS = basic_config.end2end_imagenav.num_envs
    NUM_NODES = basic_config.end2end_imagenav.num_nodes
    NUM_GPUS = basic_config.end2end_imagenav.num_gpus
    num_processes = NUM_ENVS * NUM_NODES * NUM_GPUS

    with habitat.config.read_write(basic_config):
        split = basic_config.end2end_imagenav.split
        basic_config.habitat.dataset.split = split
    dataset = make_dataset(basic_config.habitat.dataset.type)
    scenes = basic_config.habitat.dataset.content_scenes
    if "*" in basic_config.habitat.dataset.content_scenes:
        scenes = dataset.get_scenes_to_load(basic_config.habitat.dataset)

    if len(scenes) > 0:
        assert len(scenes) >= num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )

        scene_split_sizes = [int(np.floor(len(scenes) / num_processes))
                             for _ in range(num_processes)]
        for i in range(len(scenes) % num_processes):
            scene_split_sizes[i] += 1
    print("Scenes per thread:")
    for i in range(num_processes):
        config_env = get_config(cfg_path)
        with habitat.config.read_write(config_env):

            if len(scenes) > 0:
                config_env.habitat.dataset.content_scenes = scenes[
                    sum(scene_split_sizes[:i]):
                    sum(scene_split_sizes[:i + 1])
                ]
                print("Thread {}: {}".format(i, config_env.habitat.dataset.content_scenes))

            gpu_id = (i % (NUM_ENVS * NUM_GPUS)) // NUM_ENVS
            node_id = i // (NUM_ENVS * NUM_GPUS)
            config_env.habitat.simulator.habitat_sim_v0.gpu_device_id = gpu_id
            config_env.end2end_imagenav.gpu_id = gpu_id
            config_env.end2end_imagenav.node_id = node_id
            config_env.habitat.environment.iterator_options.shuffle = False

        datasets.append(
            habitat.make_dataset(
                config_env.habitat.dataset.type, config=config_env.habitat.dataset
            )
        )
        # datasets.append(None)

        configs.append(config_env)

    return configs, datasets

def _make_env_func(config, dataset=None, rank=0):
    r"""Constructor for dummy habitat Env.
    :param config: configurations for environment
    :param dataset: dataset for environment
    :param rank: rank for setting seeds for environment
    :return: constructed habitat Env
    """
    with habitat.config.read_write(config):
        config.habitat.simulator.scene = dataset.episodes[0].scene_id
    env_name = config.end2end_imagenav.env_name
    if env_name == 'instance_imagenav':
        from vector_env.envs.instance_imagenav_env import NiceEnv
    env = NiceEnv(config_env=config, dataset=dataset, rank=rank)
    env.seed(config.habitat.seed + rank)
    return env

@print_run_time
def _vec_env_fn(configs, datasets, multiprocessing_start_method='forkserver'):

    num_envs = len(configs)
    envs = VectorEnv(
        make_env_fn=_make_env_func,
                env_fn_args=tuple(
            tuple(
                zip(configs, datasets, range(num_envs))
            )
        ),
    )
    print("habitat environments created successfully !!!")
    return envs


def construct_envs(cfg_path: str):
    configs, datasets = _load_data(cfg_path)
    return _vec_env_fn(configs, datasets)
    
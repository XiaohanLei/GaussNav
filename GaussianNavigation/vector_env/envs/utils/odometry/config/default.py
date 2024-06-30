from typing import List, Optional, Union

from yacs.config import CfgNode as CN

CONFIG_FILE_SEPARATOR = ','

_C = CN()

_C.model = CN()
_C.model.params = CN()
_C.model.params.collision_embedding_size = 0
_C.model.params.n_collision_values = 2
_C.model.params.action_embedding_size = 0
_C.model.params.n_action_values = 3


def get_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
    new_keys_allowed: bool = False,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.
    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()
    config.set_new_allowed(new_keys_allowed)

    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config

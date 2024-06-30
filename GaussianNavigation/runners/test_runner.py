import sys
sys.path.append("..")
from vector_env.threaded_env import construct_envs
from habitat.config.default import get_config


class test_runner():


    def __init__(self, cfg_pth='/home/lxh/Codes/InstanceImageNav/end2end_imagenav/configs/instance_imagenav.yaml'):

        self._agent = None
        self.obs_transforms = []
        self._envs = construct_envs(cfg_pth)
        self._config = get_config(cfg_pth)
        self.num_envs = self._config.end2end_imagenav.num_envs
        self.num_nodes = self._config.end2end_imagenav.num_nodes
        self.num_gpus = self._config.end2end_imagenav.num_gpus
        self.total_envs = self.num_envs * self.num_gpus * self.num_nodes

        # episodic parameters setting
        self._max_episodes = 10

    def _create_agent(self, resume_state, **kwargs):
        """
        Sets up the AgentAccessMgr. You still must call `agent.post_init` after
        this call. This only constructs the object.
        """

        # self._create_obs_transforms()
        # self._agent = end2end_imagenav_agent(
        #     resume_state,
        #     **kwargs
        # )
        pass

    def train(self):
        self._envs.reset()
        for epsidoe in range(self._max_episodes):
            for i in range(2, -1, -1):
                actions = [{'action': i} for e in range(self.total_envs)]
                self._envs.step(actions)


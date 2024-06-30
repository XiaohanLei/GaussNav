import sys
sys.path.append("..")
from vector_env.threaded_env import construct_envs
from habitat.config.default import get_config
import time
from pprint import pprint
import numpy as np
from collections import deque
import logging


class Runner():


    def __init__(self, cfg_pth='configs/instance_imagenav.yaml'):

        self._agent = None
        self.obs_transforms = []
        self._envs = construct_envs(cfg_pth)
        self._config = get_config(cfg_pth)
        self.num_envs = self._config.end2end_imagenav.num_envs
        self.num_nodes = self._config.end2end_imagenav.num_nodes
        self.num_gpus = self._config.end2end_imagenav.num_gpus
        self.total_envs = self.num_envs * self.num_gpus * self.num_nodes

        # time utils
        self.start_time = time.time()

        # episodic parameters setting
        self._max_episodes = self._config.end2end_imagenav.max_episodes
        self._timestep = 0

        logging.basicConfig(filename='eval_metrics.log',
                             level=logging.INFO, format='%(asctime)s - %(message)s')


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
        episodes_count = [0 for i in range(self.total_envs)]
        finished = [False for i in range(self.total_envs)]
        self.start_time = time.time()
        while True:
            # report info every 20 steps
            self._timestep += 1
            if self._timestep % 20 == 0:
                print("FPS: {:.3f}, current timestep: {}".format((self._timestep * self.total_envs)/(time.time()-self.start_time), \
                                                                  self._timestep * self.total_envs ))
            

            if np.array(episodes_count).sum() == self._max_episodes * self.total_envs:
                pprint("all episode over")
                break
            finished = [(episodes_count[i]==self._max_episodes) for i in range(self.total_envs)]            
            actions = [{'action': 0, "finished": finished[e]} for e in range(self.total_envs)]
            _, _, done, info = self._envs.step(actions)
            episodes_count = [episodes_count[i]+1 if done[i] else episodes_count[i] for i in range(self.total_envs) ]

            # eval parameters  
            done_rank = np.array([done[i] for i in range(self.total_envs)]).astype(int)
            if np.any(done_rank > 0):
                succ = np.array([info[i]['success'] for i in range(self.total_envs)])
                spl = np.array([info[i]['spl'] for i in range(self.total_envs)])
                dist = np.array([info[i]['distance_to_goal'] for i in range(self.total_envs)])
                count = np.array([info[i]['episode_count'] for i in range(self.total_envs)])
                mean_success = np.sum(succ * count) / np.sum(count)
                mean_spl = np.sum(spl * count) / np.sum(count)
                mean_dist = np.sum(dist * count) / np.sum(count)

                print( " Instance ImageGoal Navigation Metrics \n succ / spl / dist : {:.5f}/{:.5f}/{:.5f}({:.0f}),".format(
                    np.mean(mean_success),
                    np.mean(mean_spl),
                    np.mean(mean_dist),
                    np.sum(count)) )
                
                log_info = " Instance ImageGoal Navigation Metrics \n succ / spl / dist : {:.5f}/{:.5f}/{:.5f}({:.0f}),".format(
                    np.mean(mean_success),
                    np.mean(mean_spl),
                    np.mean(mean_dist),
                    np.sum(count))
                logging.info(log_info)




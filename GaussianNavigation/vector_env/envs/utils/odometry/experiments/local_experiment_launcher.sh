#!/bin/bash
set -x
CURRENT_DATETIME="`date +%Y_%m_%d_%H_%M_%S`"
EXP_NAME=${EXP_NAME:-"pointnav2020_gt_loc_rgbd_`date +%Y_%m_%d_%H_%M_%S`"}

BASE_TASK_CONFIG_PATH=${BASE_TASK_CONFIG_PATH:-"config_files/challenge_pointnav2020_gt_loc.local.rgbd.yaml"}
DATASET_CONTENT_SCENES=""
MAX_SCENE_REPEAT_STEPS=""
NUM_EPISODE_SAMPLE=""
SENSORS=""

EXPERIMENT_DIR="/home/rpartsey/code/3d-navigation/related_works/egolocalization/out/ddppo/experiments/${EXP_NAME}"
TENSORBOARD_DIR="${EXPERIMENT_DIR}/tb"
CHECKPOINT_DIR="${EXPERIMENT_DIR}/checkpoints"
CMD_OPTS_FILE="${EXPERIMENT_DIR}/cmd_opt.txt"
CMD_EVAL_OPTS_FILE="${EXPERIMENT_DIR}/cmd_eval_opt.txt"

mkdir -p ${CHECKPOINT_DIR}
mkdir -p ${TENSORBOARD_DIR}

CMD_EVAL_OPTS="BASE_TASK_CONFIG_PATH $BASE_TASK_CONFIG_PATH EVAL_CKPT_PATH_DIR ${CHECKPOINT_DIR} CHECKPOINT_FOLDER ${CHECKPOINT_DIR} TENSORBOARD_DIR ${TENSORBOARD_DIR} ${RL_PPO_NUM_STEPS} ${SENSORS}"
CMD_OPTS="${DATASET_CONTENT_SCENES} ${MAX_SCENE_REPEAT_STEPS} ${NUM_EPISODE_SAMPLE} ${CMD_EVAL_OPTS}"
echo "$CMD_OPTS" > ${CMD_OPTS_FILE}

CMD_EVAL_OPTS="${CMD_EVAL_OPTS} EVAL.SPLIT val TASK_CONFIG.DATASET.CONTENT_SCENES [\"*\",\"*\"] NUM_PROCESSES 1 RL.PPO.num_mini_batch 1 VIDEO_OPTION [] TASK_CONFIG.TASK.TOP_DOWN_MAP.MAP_RESOLUTION 1250"
echo "$CMD_EVAL_OPTS" > ${CMD_EVAL_OPTS_FILE}


#sbatch --export=ALL,CMD_OPTS_FILE=${CMD_OPTS_FILE} --job-name=${EXP_NAME: -8} --output=$LOG_DIR/log.out --error=$LOG_DIR/log.err navigation/experiments/run_experiment.sh

# val on new episodes [\"1pXnuDYAj8r\",\"1pXnuDYAj8r\",\"1pXnuDYAj8r\",\"1pXnuDYAj8r\",\"1pXnuDYAj8r\",\"1pXnuDYAj8r\",\"1pXnuDYAj8r\",\"1pXnuDYAj8r\"]
#sbatch --export=ALL,CMD_OPTS_FILE=${CMD_EVAL_OPTS_FILE} --job-name=${EXP_NAME: -7}e --output=$LOG_DIR/log_eval.out --error=$LOG_DIR/log_eval.err experiments/run_obj_nav_eval.sh


#sbatch --export=ALL,CMD_OPTS_FILE=/checkpoint/maksymets/logs/habitat_baselines/ddppo/obj_nav_mp3d_all_train_rgbd_no_up_down2020_05_14_13_25_56/cmd_eval_opt.txt --job-name=3_25_56e --output=/checkpoint/maksymets/logs/habitat_baselines/ddppo/obj_nav_mp3d_all_train_rgbd_no_up_down2020_05_14_13_25_56/log_eval.out --error=/checkpoint/maksymets/logs/habitat_baselines/ddppo/obj_nav_mp3d_all_train_rgbd_no_up_down2020_05_14_13_25_56/log_eval.err experiments/run_obj_nav_eval.sh
#BASE_TASK_CONFIG_PATH configs/tasks/objectnav_mp3d.yaml EVAL_CKPT_PATH_DIR data/new_checkpoints/obj_nav_mp3d_all_train_rgbd_no_up_down2020_05_14_13_25_56 CHECKPOINT_FOLDER data/new_checkpoints/obj_nav_mp3d_all_train_rgbd_no_up_down2020_05_14_13_25_56 TENSORBOARD_DIR /checkpoint/maksymets/logs/habitat_baselines/ddppo/obj_nav_mp3d_all_train_rgbd_no_up_down2020_05_14_13_25_56   TEST_EPISODE_COUNT 2195 EVAL.SPLIT val_mini VIDEO_OPTION []


# /private/home/maksymets/python_wrapper.sh -u /private/home/maksymets/habitat-lab-pr/habitat_baselines/run.py --exp-config habitat_baselines/config/objectnav/ddppo_objectnav.yaml --run-type eval BASE_TASK_CONFIG_PATH configs/tasks/objectnav_mp3d_256.yaml EVAL_CKPT_PATH_DIR data/new_checkpoints/obj_nav_mp3d_all_train_depth_sem_cat_no_up_down2020_05_27_15_32_03/ckpt.106.pth CHECKPOINT_FOLDER data/new_checkpoints/obj_nav_mp3d_all_train_depth_sem_cat_no_up_down2020_05_27_15_32_03 TENSORBOARD_DIR /checkpoint/maksymets/logs/habitat_baselines/ddppo/obj_nav_mp3d_all_train_depth_sem_cat_no_up_down2020_05_27_15_32_03  SENSORS "[\"DEPTH_SENSOR\",\"SEMANTIC_SENSOR\"]" TASK_CONFIG.TASK.SENSORS [\"COMPASS_SENSOR\",\"GPS_SENSOR\",\"OBJECTSEMANTIC_SENSOR\"] EVAL.SPLIT val_mini NUM_PROCESSES 1 RL.PPO.num_mini_batch 1 TASK_CONFIG.TASK.TOP_DOWN_MAP.MAP_RESOLUTION 12500

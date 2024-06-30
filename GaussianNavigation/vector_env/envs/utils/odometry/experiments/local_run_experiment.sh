#!/bin/bash
#SBATCH --job-name=ddppo-object-nav
#SBATCH --gres=gpu:8   #gpu:volta:8
#SBATCH --constraint=volta32gb
#SBATCH --nodes 32
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 8
#SBATCH --mem=450GB #maybe 450, was 500GB
#SBATCH --time=72:00:00
#SBATCH --signal=USR1@600
#SBATCH --partition=priority
#SBATCH --open-mode=append
#SBATCH --comment="CVPR 2021"

#export GLOG_minloglevel=2
#export MAGNUM_LOG=quiet

#export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)

#EXP_NAME="obj_nav_mp3d_1_ep_slack-1e3"
#--gres gpu:8  # added 32 GB

#$MODULESHOME/init/bash

#source ~/.bash_profile
#source ~/.profile
#source /etc/bash.bashrc
#source /etc/profile

#module unload cuda
#module load cuda/10.1
#module unload cudnn
#module load cudnn/v7.6.5.32-cuda.10.1
#module load anaconda3/5.0.1
#module load gcc/7.1.0
#module load cmake/3.10.1/gcc.5.4.0
#source activate my_fair_env

#export CUDA_HOME="/public/apps/cuda/10.1"
#export CUDA_NVCC_EXECUTABLE="/public/apps/cuda/10.1/bin/nvcc"
#export CUDNN_INCLUDE_PATH="/public/apps/cuda/10.1/include/"
#export CUDNN_LIBRARY_PATH="/public/apps/cuda/10.1/lib64/"
#export LIBRARY_PATH="/public/apps/cuda/10.1/lib64"
#export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
#export USE_CUDA=1 USE_CUDNN=1 USE_MKLDNN=1

#CURRENT_DATETIME="`date +%Y_%m_%d_%H_%M_%S`";
#echo $CUDA_VISIBLE_DEVICES

if [[ ! -v CMD_OPTS_FILE ]]; then
    echo "CMD_OPTS_FILE is not set"
    exit 1
fi

CMD_OPTS=$(cat "$CMD_OPTS_FILE")

set -x
/home/rpartsey/miniconda/envs/pointgoal-navigation-env/bin/python -u -m habitat_baselines.run \
    --exp-config config_files/ddppo/ddppo_pointnav.yaml \
    --run-type train \
    NUM_PROCESSES 4 \
    ${CMD_OPTS}
#srun /private/home/maksymets/.conda/envs/my_fair_env/bin/python -u -m habitat_baselines.run \
#    --exp-config config_files/ddppo/ddppo_pointnav.yaml \
#    --run-type train ${CMD_OPTS}

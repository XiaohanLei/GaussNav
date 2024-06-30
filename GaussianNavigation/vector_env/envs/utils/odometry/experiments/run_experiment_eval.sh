#!/bin/bash
#SBATCH --job-name=navigation_pointgoal
#SBATCH --gres gpu:1
#SBATCH --constraint=volta32gb
#SBATCH --nodes 1
#SBATCH --cpus-per-task 10
#SBATCH --ntasks-per-node 8
#SBATCH --mem=450GB #maybe 450, was 500GB
#SBATCH --time=72:00:00
#SBATCH --signal=USR1@600
#SBATCH --partition=devlab
#SBATCH --open-mode=append
#SBATCH --comment="navigation_pointgoal"


export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

export MASTER_ADDR=$(srun --ntasks=1 hostname 2>&1 | tail -n1)


source ~/.bash_profile
source ~/.profile
source /etc/bash.bashrc
source /etc/profile

module unload cuda
module load cuda/10.1
module unload cudnn
module load cudnn/v7.6.5.32-cuda.10.1
module load anaconda3/5.0.1
module load gcc/7.1.0
module load cmake/3.10.1/gcc.5.4.0
source activate challenge_2021

export CUDA_HOME="/public/apps/cuda/10.1"
export CUDA_NVCC_EXECUTABLE="/public/apps/cuda/10.1/bin/nvcc"
export CUDNN_INCLUDE_PATH="/public/apps/cuda/10.1/include/"
export CUDNN_LIBRARY_PATH="/public/apps/cuda/10.1/lib64/"
export LIBRARY_PATH="/public/apps/cuda/10.1/lib64"
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
export USE_CUDA=1 USE_CUDNN=1 USE_MKLDNN=1

CURRENT_DATETIME="`date +%Y_%m_%d_%H_%M_%S`";
echo $CUDA_VISIBLE_DEVICES

CMD_OPTS=$(cat "$CMD_OPTS_FILE")

set -x
srun /private/home/maksymets/.conda/envs/my_fair_env/bin/python -u -m habitat_baselines.run \
    --exp-config config_files/ddppo/ddppo_pointnav_2021.yaml \
    --run-type eval ${CMD_OPTS}

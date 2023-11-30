#!/usr/bin/bash

#SBATCH --job-name='ddpm'
#SBATCH --partition=ndl
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1

name=$1
ckpt_path=$2

SLURM_HOSTLIST=$(scontrol show hostnames|paste -d, -s)
echo "SLURM_HOSTLIST: ${SLURM_HOSTLIST}"
echo $(hostname -s)
NB_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F"," '{print NF}')
echo "Number of GPU: ${NB_GPUS}"

source activate deepEnv1

module load gcc/9.2.0
module load nvhpc/23.7
module load python/3.7.6

export CUDA_HOME=/opt/softs/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/12.2
export NVHPC_CUDA_HOME=/opt/softs/nvidia/hpc_sdk/Linux_x86_64/23.7/cuda/11.8
export CXX=g++ #the compiler for cpp extensions
export CC=gcc  #the compiler to access the good cpp standard

ARGS=$1
ARGS2="${ARGS//|/ }"

torchrun --standalone --nproc_per_node=$NB_GPUS main.py Train --run_name $name --batch_size 4 --lr 1e-4 --epochs 50 --save_step 5000 -ddim 50 --plot --dataloader_rr --dataset_handler_config="configs/${name}/dataset_handler_config.yaml" -mp $ckpt_path
#torchrun --nproc_per_node=$NB_GPUS ${PYTHON_SCRIPT} ${ARGS2}

#srun -w $(hostname -s) --nodes=1 --ntasks-per-node=1 --ntasks=1 $SLURM_SCRIPT primary $ARGS

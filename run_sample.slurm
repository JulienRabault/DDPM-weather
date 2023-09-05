#!/bin/bash
# ==============================================================================
#
# Copyright (c) 2018 Dell Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
#
# description     : This script will submit a horovod job
# author          : Cedric Castagnede
# company         : DellEMC
# mail            : cedric.castagnede@dell.com
# date            : 2019-07-03
#
# ==============================================================================

#### Paramètres à modifier ####

#SBATCH --job-name='sample_ddpm'
#SBATCH --nodes=001 # laisseer ça même si node2 dans partition
#SBATCH --partition=node3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:4
#SBATCH --time=0-12:00:00
#SBATCH --begin=now #+26hours
#SBATCH --error="Sample.err"
#SBATCH --output="Sample.err"

# Vos répertoires de travail -> nécessaire pour vous donner des permissions pour vos répertoires de travail
##/!\## doit être au moins un niveau au-dessus de votre 'répertoire racine'
# ex -> /home/my_group/my_user/my_working_directory pour mes scripts
# ex -> /scratch/my_group/my_user/my_working_directory pour mes données

HOME_DIR="/scratch/mrmn/rabaultj/DDPM-for-meteo/"
DATA_DIR="/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_done/"

SLURM_SCRIPT="${HOME_DIR}slurm-docker-run_perso"

PYTHON_SCRIPT="${HOME_DIR}main.py"
MODEL_DIR="Train_uv_final/best.pt"
SAMPLE_DIR="Sample_final_uv"
SCRATCH_DIR="/scratch/mrmn/rabaultj/"

HOROVOD_CONTAINER="ddpm_dock" #container name

############################

# Collecte de la liste des hôtes Slurm
SLURM_HOSTLIST=$(scontrol show hostnames | paste -d, -s)
echo "SLURM_HOSTLIST: ${SLURM_HOSTLIST}"

# Votre ID utilisateur et de groupe -> nécessaire pour vous donner des permissions pour vos répertoires de travail
UID="$((`id -u`))"
GID="$((`id -g`))"
# Lire les arguments de votre script  Python si vous en avez un (>sbatch horovod.slurm "ARGS")
ARGS=$1
ARGS2="${ARGS//|/ }"

# Préparer les nœuds secondaires (si vous utilisez au moins 2 nœuds)
NB_SECONDARY_WORKERS=$((SLURM_JOB_NUM_NODES-1))
if (( ${NB_SECONDARY_WORKERS} > 0 ))
then
    echo "Démarrage des conteneurs des nœuds secondaires"
    echo srun -x $(hostname -s) --nodes=${NB_SECONDARY_WORKERS} --ntasks-per-node=1 --ntasks=${NB_SECONDARY_WORKERS} slurm-docker-run secondary $HOROVOD_CONTAINER $UID $GID $HOME_DIR $DATA_DIR $PYTHON_SCRIPT $ARGS2
    srun -x $(hostname -s) --nodes=${NB_SECONDARY_WORKERS} --ntasks-per-node=1 --ntasks=${NB_SECONDARY_WORKERS} slurm-docker-run secondary $HOROVOD_CONTAINER $UID $GID $HOME_DIR $DATA_DIR $PYTHON_SCRIPT $ARGS &
fi

# Démarrer le travail sur le nœud principal
sleep 5
echo "Démarrage sur le nœud principal..."

srun -w $(hostname -s) --nodes=1 --ntasks-per-node=1 --ntasks=1 $SLURM_SCRIPT primary $HOROVOD_CONTAINER $UID $GID $HOME_DIR $OUTPUT_DIR $DATA_DIR $PYTHON_SCRIPT "Test |--n_sample | 16380 | --data_dir| /scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_done/| --train_name |/scratch/mrmn/rabaultj/DDPM-for-meteo/${SAMPLE_DIR} |--batch_size | 128 |--model_path | /scratch/mrmn/rabaultj/DDPM-for-meteo/${MODEL_DIR} "

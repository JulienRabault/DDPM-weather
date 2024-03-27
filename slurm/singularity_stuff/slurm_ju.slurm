#!/bin/bash
#SBATCH --job-name=train_reduc_06       
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=8          # avec une tache par noeud 
#SBATCH --cpus-per-task=3 #
#SBATCH --gres=gpu:8
#SBATCH --hint=nomultithread  
#SBATCH --time=100:00:00           # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=train_reduc_06.err      # nom du fichier de sortie
#SBATCH --error=train_reduc_06.err
#SBATCH --account=woz@v100
#SBATCH --qos=qos_gpu-t4
#SBATCH --partition=gpu_p2

module purge
# module load cpuarch/amd
module load pytorch-gpu/py3/2.0.1
export WANDB_MODE = "offline"
# Echo des commandes lancees
set -x

python -m torch.distributed.run --standalone --nproc_per_node gpu main.py --yaml_path config_v100.yml
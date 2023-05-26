#!/bin/sh
#SBATCH --job-name=gpu1n
#SBATCH --partition=GPUNodes
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1         # avec une tache par noeud
#SBATCH --gres=gpu:2
#SBATCH --error="gpu1n.err"
#SBATCH --output="gpu1n.err"

module purge
module load singularity/3.0.3
# -m torch.distributed.run --standalone --nproc_per_node gpu
srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-22-03-py3.sif $HOME/localTMP/env/bin/python -m torch.distributed.run --standalone --nproc_per_node gpu multi_main.py "Train" --v_i 3 --data_dir "data/" --train_name "gpu1n" --invert_norm --epochs 10 --batch_size 16 --any_time 1

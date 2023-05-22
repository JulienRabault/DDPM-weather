#!/bin/sh
#SBATCH --job-name=last_test
#SBATCH --partition=RTX6000Node
#SBATCH --gres=gpu:1
#SBATCH --mem=32000
#SBATCH --gres-flags=enforce-binding
#SBATCH --error="last_test.err"
#SBATCH --output="last_test.err"

module purge
module load singularity/3.0.3

srun singularity exec /logiciels/containerCollections/CUDA11/pytorch-NGC-22-03-py3.sif $HOME/localTMP/env/bin/python "main.py" "Train" --v_i 3 --data_dir "/users/celdev/jrabault/POESY/stylegan2/IS_1_1.0_0_0_0_0_0_256_done_red/" --train_name "TRAIN_128_3RTX" --invert_norm --epochs 50 --batch_size 16 --any_time 5 
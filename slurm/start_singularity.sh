#/bin/bash


module purge
module load singularity

source .env

# Echo des commandes lancees
set -x

srun singularity exec --nv\
    --bind $SOURCE_DIR:/ddpm\
    --bind $LOG_DIR:/ddpm/logs\
    --bind $DATA_DIR:/ddpm/data\
    $SINGULARITY_IMG_PATH bash /ddpm/slurm/train.sh $@


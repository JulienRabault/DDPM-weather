#/bin/bash

module purge
module load singularity

# Echo des commandes lancees
set -x

srun singularity exec --nv\
    --bind $SOURCE_DIR:/ddpm\
    --bind $LOG_DIR:/ddpm/logs\
    --bind $DATA_DIR:/ddpm/data\
    $SINGULARITY_IMG_PATH bash $SCRIPT_DIR/train.sh $@
    
# srun singularity exec --nv $SINGULARITY_ALLOWED_DIR/ddpm.sif $WORK/DDPM-weather/train.sh $@


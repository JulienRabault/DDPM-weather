#/bin/bash

module purge
module load singularity

# Echo des commandes lancees
set -x

srun singularity exec --nv $SINGULARITY_ALLOWED_DIR/ddpm.sif $WORK/DDPM-weather/train.sh $@


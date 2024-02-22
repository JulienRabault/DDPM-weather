#!/bin/bash

#!/bin/bash
#SBATCH --job-name=train\_test       # partition de pre et post traitement
#SBATCH --nodes=4                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud
#SBATCH --cpus-per-task=10 #
#SBATCH --gres=gpu:4
#SBATCH --hint=nomultithread
#SBATCH --time=15:00:00           # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --output=log.err      # nom du fichier de sortie
#SBATCH --error=log.err
#SBATCH --account=woz@v100
#SBATCH -C v100-32g
#SBATCH -A woz@v100


# Echo des commandes lancees
set -x


pwd

cd slurm


./start_singularity.sh $@

# DDPM-for-meteo

Ce dépôt contient le code source d'un modèle de débruitage basé sur la diffusion probabiliste, implémenté en utilisant Python. Le modèle est conçu pour le débruitage d'images météorologiques.



## Installation
### Warning

Le code a etait testé avec `denoising-diffusion-pytorch==1.6.4`, les nouvelles versions de `denoising-diffusion-pytorch` provoquent des problemes de memoire gpu. 
Le model est plus gros que la version de base.


Vous pouvez installer les dépendances en exécutant la commande suivante :

```python
pip install -r requirements.txt
```

This code uses https://github.com/lucidrains/denoising-diffusion-pytorch

### WandB pour le suivi de l'entraînement

Wandb est un outil de suivi d'entraînement qui permet de visualiser les métriques d'entraînement en temps réel.

Créez un compte WandB sur https://wandb.ai/.
Dans votre terminal, exécutez la commande suivante pour vous connecter à votre compte WandB :
```bash
wandb login
```

Il vous sera demandé de saisir votre clef d'api, que vous pouvez trouver sur votre compte WandB.

Ref : https://docs.wandb.ai/quickstart

Attention a bien changer `wandbproject` et `entityWDB` selon votre compte wandb.
Le suivi de WandB est offline par défaut. A la fin de l'entrainnement vous pouvez utiliser les commandes suivantes pour changer les données offline :

```bash
wandb sync <{config.train_name}/WANDB>
```

## Utilisation

Le code principal se trouve dans le fichier `main.py`. Il peut être exécuté avec différentes options de mode :

- Train : Mode d'entraînement du modèle.
- Test : Mode de test du modèle.

Exécutez le code avec la commande suivante :

```python
python main.py [mode] [options]
```

## Options Disponibles

Vous pouvez personnaliser le comportement de ce code en utilisant les options suivantes avec le script principal. Voici une liste des options disponibles et leurs descriptions :

- `mode` : Le mode d'exécution, vous pouvez choisir entre "Train" pour l'entraînement ou "Test" pour les tests.
- `train_name` : Le nom de la session d'entraînement ou du dossier de reprise.
- `batch_size` : La taille du lot (par défaut : 16).
- `n_sample` : Le nombre d'échantillons à générer (par défaut : 4).
- `any_time` : Toutes les combien d'époques sauvegarder et générer des échantillons (par défaut : 400).
- `model_path` : Le chemin vers le modèle pour charger et reprendre l'entraînement si nécessaire.
- `lr` : Taux d'apprentissage (par défaut : 5e-4).
- `adam_betas` : Les valeurs beta pour l'optimiseur Adam (par défaut : (0.9, 0.99)).
- `epochs` : Le nombre d'époques pour l'entraînement (par défaut : 100).
- `image_size` : La taille de l'image (par défaut : 128).
- `data_dir` : Le répertoire contenant les données (par défaut : 'data/').
- `v_i` : Le nombre d'indices de variables (par défaut : 3).
- `var_indexes` : La liste des indices de variables (par défaut : ['u', 'v', 't2m']).
- `crop` : Les paramètres de recadrage pour les images (par défaut : [78, 206, 55, 183]).
- `wandbproject` : Le nom du projet Wandb.
- `use_wandb` : Utiliser Wandb pour la journalisation (par défaut : False).
- `entityWDB` : Le nom de l'entité Wandb.
- `invert_norm` : Inverser la normalisation des échantillons d'images (par défaut : False).
- `beta_schedule` : Le type de planification beta (cosinus ou linéaire) (par défaut : "cosinus").
- `auto_normalize` : Normalisation automatique (par défaut : False).
- `scheduler` : Utiliser un scheduler pour le taux d'apprentissage (par défaut : False).
- `scheduler_epoch` : Le nombre d'époques pour le scheduler (par défaut : 150).
- `resume` : Reprise depuis un point de contrôle (par défaut : False).
- `debug_log` : Activer les journaux de débogage (par défaut : False).

Pour reprendre un entraînement, 2 possibilités :

- Partir d'un modèle pré-entraîné => le donner par `model_path`
- Partir d'un modèle pré-entraîné ET continuer dans le même dossier d'entraînement => le donner par `model_path` ET utiliser `resume`

Si vous voulez utiliser le scheduler, il faut utiliser `scheduler` et `scheduler_epoch` (par défaut : 150). Le scheduler est un scheduler de type `OneCycleLR` de PyTorch. Il est sauvegardé dans le fichier `.pt` et est utilisé pour reprendre l'entraînement, il faut donc lui donner le nombre total d'époques d'entraînement.

## Exemples

1. Entraîner le modèle :


```python
python main.py Train --train_name my_training_run --batch_size 32 --lr 1e-4 --epochs 50
```

2. Tester (Sample) le modèle :


```python
python main.py Test --train_name my_training_run --n_sample 50
```

3. Entraînement avec plusieurs GPUs 
```python
python -m torch.distributed.run --standalone --nproc_per_node  mon_script.py Train --train_name my_training_run --batch_size 32 --lr 0.001 --epochs 50
```
## Usage spécifique à PRIAM

Il y a 2 fichiers `.slurm`, un pour faire des samples a partir d'un modele et un pour lancer un train. 

### run_train.slurm

#### 1. Modifier selon vos dossiers 

Selon vos données et home_dir
```
HOME_DIR="/scratch/mrmn/rabaultj/DDPM-for-meteo/"
DATA_DIR="/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_done/"
```
```
#SBATCH --job-name='train'        <-- Le nom du job
#SBATCH --partition=node1         <-- La partition en fonction des dispo, node1 ou node3
#SBATCH --gres=gpu:v100:2         <-- Le nombre de gpu voulu, max 4
#SBATCH --error="train.err"       <-- Pour suivre l'entrainnement avec 'tail -f train.err' ou 'cat train.err'
#SBATCH --output="train.err"      <-|
```

La derniere ligne aussi !

#### 2. Modifier les parametres selon [Options Disponibles](#options-disponibles)
Derniere ligne, mettre `|` entres les parametres
```
srun -w $(hostname -s) --nodes=1 --ntasks-per-node=1 --ntasks=1 $SLURM_SCRIPT primary $HOROVOD_CONTAINER $UID $GID $HOME_DIR $OUTPUT_DIR $DATA_DIR $PYTHON_SCRIPT "Train|--v_i | 3 | --data_dir| "/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_done/"| --train_name |"/scratch/mrmn/rabaultj/DDPM-for-meteo/test" |--epochs | 10 |--batch_size | 16 |--any_time | 50 "
```

### run_sample.slurm

#### 1. Modifier selon vos dossiers 

Selon vos données et home_dir
```
HOME_DIR="/scratch/mrmn/rabaultj/DDPM-for-meteo/"
DATA_DIR="/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_done/"
```
```
#SBATCH --job-name='sample_ddpm'        <-- Le nom du job
#SBATCH --partition=node1         <-- La partition en fonction des dispo, node1 ou node3
#SBATCH --gres=gpu:v100:2         <-- Le nombre de gpu voulu, max 4
#SBATCH --error="Sample.err"       <-- Pour suivre l'entrainnement avec 'tail -f Sample.err' ou 'cat Sample.err'
#SBATCH --output="Sample.err"      <-|
```

La derniere ligne aussi !

#### 2. Modifier les parametres selon [Options Disponibles](#options-disponibles)

```
PYTHON_SCRIPT="${HOME_DIR}main.py"
MODEL_DIR="Train_uv_final/best.pt"
SAMPLE_DIR="Sample_final_uv"
SCRATCH_DIR="/scratch/mrmn/rabaultj/"
N_SAMPLES=10
```

Derniere ligne, mettre `|` entres les parametres
```
srun -w $(hostname -s) --nodes=1 --ntasks-per-node=1 --ntasks=1 $SLURM_SCRIPT primary $HOROVOD_CONTAINER $UID $GID $HOME_DIR $OUTPUT_DIR $DATA_DIR $PYTHON_SCRIPT "Test |--n_sample | ${N_SAMPLES} | --data_dir| /scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_done/| --train_name |/scratch/mrmn/rabaultj/DDPM-for-meteo/${SAMPLE_DIR} |--batch_size | 128 |--model_path | /scratch/mrmn/rabaultj/DDPM-for-meteo/${MODEL_DIR} "
```
## Lancement

`sbatch run_train.slurm` ou `sbatch run_sample.slurm`




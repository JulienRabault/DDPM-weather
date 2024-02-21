# DDPM-for-meteo

Ce dépôt contient le code source d'un modèle de débruitage basé sur la diffusion probabiliste, implémenté en utilisant Python. Le modèle est conçu pour le débruitage d'images météorologiques.

## Structure du Projet

Le projet est structuré comme suit :

```bash
.
├── ddpm
│   ├── dataSet_Handler.py             # Gestionnaire de données
│   ├── ddpm_base.py                   # Implémentation de base du modèle/trainer/sampler
│   ├── guided_gaussian_diffusion.py   # Implémentation de la diffusion guidée
│   ├── sampler.py                     # Implémentation du sampler
│   └── trainer.py                     # Implémentation du trainer
├── utils
│   ├── config.py                      # Gestionnaire de configuration
│   ├── config_schema.json             # Schéma de configuration, valeur par défaut
│   ├── distributed.py                 # Gestionnaire de la distribution multi GPU
│   └── guided_loss.py                 # Implémentation des loss pour la diffusion guidée simple
├── main.py                            # Point d'entrée du code
├── requirements.txt                   # Dépendances du projet
├── config_sample.yml                  # Exemple de configuration d'échantillonnage
├── config_train.yml                   # Exemple de configuration d'entraînement
└── README.md                          # Ce fichier
```

## Installation
Vous pouvez installer les dépendances en exécutant la commande suivante :

```python
pip install -r requirements.txt
```

Ce code utilise https://github.com/lucidrains/denoising-diffusion-pytorch.

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
wandb sync <{config.run_name}/WANDB>
```

### Mlflow pour le suivi de l'entraînement

Mlflow est un logger local, parametrable dans le fichier de config : 

```yml
# Tracking parameters
"use_mlflow": true, # activation mlflow log
"ml_tracking_uri": "../mlruns", # path to log mlflow 
"ml_experiment_name": "ddpm", # experience name
```

Pour visualiser les résultats, cd dans le dir où est le dossier `mlruns`
```bash
cd < ml_tracking_uri >..
mlflow ui
>>> [INFO] Listening at: http://127.0.0.1:5000 (720864)
```
et ouvrir l'URL.


## Utilisation

Le code principal est situé dans le fichier `main.py` et peut être exécuté avec différents modes :

- **Train** : Mode d'entraînement du modèle.
- **Sample** : Mode d'échantillonnage du modèle.

Exécutez le code avec la commande suivante :

```bash
python main.py --yaml_path [chemin_vers_config_yaml] --debug [autres_options]
```
Notez que le chemin vers le fichier de configuration YAML donc l'option `--yaml_path` n'est obligatoire pas, les valeurs par défaut sont dans `utils/config_schema.json`.
### Remarque : 
Vous pouvez également surcharger les options de configuration du fichier YAML en les spécifiant directement en ligne de commande. Par exemple, pour modifier la taille du lot, vous pouvez ajouter l'option --batch_size [nouvelle_valeur].

## Options Disponibles

Vous pouvez personnaliser le comportement de ce code en modifiant/créant votre propre configuration. Voici une liste des options disponibles et leurs descriptions :

### Paramètres Généraux :
- `mode` : Le mode d'exécution, vous pouvez choisir entre "Train" pour l'entraînement ou "Sample" pour les échantillons.
- `run_name` : Le nom de la session d'entraînement ou du dossier de reprise.
- `batch_size` : La taille du lot (par défaut : 32).
- `any_time` : Toutes les combien d'époques sauvegarder et générer des échantillons (par défaut : 400).
- `model_path` : Le chemin vers le modèle pour charger et reprendre l'entraînement si nécessaire (aucun chemin démarrera l'entraînement depuis le début).
- `debug` : Active les journaux de débogage (par défaut : désactivé).
### Paramètres d'Échantillonnage :
- `ddim_timesteps` : Si différent de None, échantillonnera à partir de la méthode ddim avec le nombre spécifié de pas de temps.
- `plot` : Activer pour afficher et sauvegarder les échantillons générés.
- `guided` : Chemin vers les données guidées.
- `n_sample` : Nombre d'échantillons à générer.
- `random_noise` : Utiliser du bruit aléatoire pour x_start dans l'échantillonnage guidé.
### Paramètres de Données :
- `data_dir` : Répertoire contenant les données.
- `v_i` : Nombre d'indices de variables.
- `var_indexes` : Liste des indices de variables.
- `crop` : Paramètres de découpe pour les images.
- `auto_normalize` : Normalisation automatique (par défaut : désactivée).
- `invert_norm` : Inversion de la normalisation des échantillons d'image (par défaut : désactivée).
- `image_size` : Taille de l'image.
- `mean_file` : Chemin du fichier de moyenne.
- `max_file` : Chemin du fichier de maximum.
- `guiding_col` : Colonne à utiliser pour l'échantillonnage guidé. Requis lors de l'utilisation du mode guidé.
- `csv_file` : Chemin du fichier csv des labels (nécessaire pour le guidage).
### Paramètres du Modèle :
- `scheduler` : Utiliser un planificateur pour le taux d'apprentissage.
- `scheduler_epoch` : Nombre d'époques pour le planificateur pour ajuster le taux d'apprentissage (sauvegarde pour la reprise).
- `resume` : Reprendre à partir d'un point de contrôle.
- `lr` : Taux d'apprentissage.
- `adam_betas` : Bêtas pour l'optimiseur Adam.
- `epochs` : Nombre d'époques d'entraînement.
- `beta_schedule` : Type de planification des bêtas (cosinus ou linéaire).
### Paramètres de Suivi :
- `use_mlflow`: activation mlflow log
- `ml_tracking_uri`: path to log mlflow
- `ml_experiment_name`: mlflow experience name

- `wandbproject` : Nom du projet Wandb.
- `use_wandb` : Utiliser Wandb pour la journalisation.
- `entityWDB` : Nom de l'entité Wandb.

#### Pour reprendre un entraînement, 2 possibilités :

- Partir d'un modèle pré-entraîné => le donner par `model_path`
- Partir d'un modèle pré-entraîné ET continuer dans le même dossier d'entraînement => le donner par `model_path` ET utiliser `resume`

Si vous voulez utiliser le scheduler, il faut utiliser `scheduler` et `scheduler_epoch` (par défaut : 150). Le scheduler est un scheduler de type `OneCycleLR` de PyTorch. Il est sauvegardé dans le fichier `.pt` et est utilisé pour reprendre l'entraînement, il faut donc lui donner le nombre total d'époques d'entraînement.

## Exemples

1. Entraîner le modèle :

```python
python main.py --yaml_path config_train.yml --batch_size 64 --lr 0.0001
```

2. Tester (Sample) le modèle :

```python
python main.py --yaml_path config_sample.yml
```

3. Entraînement avec plusieurs GPUs 
```python
python torch.distributed.run --standalone --nproc_per_node gpu main.py --yaml_path config_sample.yml
```

4. Reprendre l'entraînement à partir d'un point de contrôle :

```python
python main.py --yaml_path config_train.yml --model_path checkpoints/checkpoint.pt --resume
```
attention, `--model_path` et `--resume` peuvent etre simplement spécifié dans le fichier yaml.


5. plusieur entrainements en séquentiel :

```python
python main.py -m --yaml_path config_sample.yml
```

avec dans le fichier de config yaml : 
```"batch_size": [4,8,16],```
pour tester plusieur configuration de batch_size par exemple. 

### Exemple de fichier de configuration YAML :

```yaml
{
  # General parameters
  "mode": "Train",
  "run_name": "run_train",
  "batch_size": 4,
  "any_time": 25,

  # Sampling parameters
  "ddim_timesteps": 500,
  "plot": true,
  "sampling_mode": "simple",
  "n_sample": 4,
  "random_noise": true,

  # Data parameters
  "data_dir": "/path/to/your/data/",
  "csv_file": "your_data_labels.csv",
  "v_i": 3,
  "var_indexes": [ "u", "v", "t2m" ],
  "crop": [ 0,256,0,256 ],
  "invert_norm": false,
  "image_size": 256,
  "mean_file": "mean_data.npy",
  "max_file": "max_data.npy",
  "guiding_col": "your_guiding_column",

  # Model parameters
  "scheduler": true,
  "scheduler_epoch": 500,
  "resume": false,
  "epochs": 500,
  "beta_schedule": "linear",

  # Tracking parameters
  "use_mlflow": true, # activation mlflow log
  "ml_tracking_uri": "../mlruns", # path to log mlflow
  "ml_experiment_name": "ddpm", # experience name

  "wandbproject": "your_wandb_project",
  "use_wandb": true,
  "entityWDB": "your_entity"
}
```

Si des valeur ne sont pas spécifiées dans le fichier de configuration, elles seront remplacées par les valeurs par défaut ou par la surcharge lors de l'appel du fichier `main.py`.


## Singularity & slurm

Installation de *singularity* en téléchargeant le paquet ici : https://github.com/sylabs/singularity/releases/

Construit l'image avec `ddpm.def` : `singularity build --nv ddpm.sif ddpm.def`

Entre dans le contener singularity en montant les chemins du code source et des données : `singularity shell --nv --bind .:/ddpm <data_dir>:/data ddpm.sif` 
et dans `/ddpm/` lance ton entrainement suivant les instructions ci dessus. Ne pas oublier de redéfinir le chemin des données dans `/data/` dans la config yaml.    

Pour une réservation slurm, lancer la réservation avec `sbatch slurm/reserve_node.slurm config/config_train_jeanzay.yml`. Ne pas oublier de changer ci besoin le `slurm/.env` pour monter les bons chemin de dossiers de log, de données, de sources et de l'image singularity .sif.   

## Contact

Pour toute question, vous pouvez me contacter à l'adresse suivante : `julien.rabault@irit.fr`

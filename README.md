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
- `scheduler` : Utiliser un planificateur pour le taux d'apprentissage (par défaut : False).
- `resume` : Reprise depuis un point de contrôle (par défaut : False).
- `debug_log` : Activer les journaux de débogage (par défaut : False).


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

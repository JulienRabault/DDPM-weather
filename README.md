# DDPM-for-meteo

Ce dépôt contient le code source d'un modèle de débruitage basé sur la diffusion probabiliste, implémenté en utilisant Python. Le modèle est conçu pour le débruitage d'images météorologiques.

## Installation

Vous pouvez installer les dépendances en exécutant la commande suivante :

```python
pip install -r requirements.txt
```


## Utilisation

Le code principal se trouve dans le fichier `main.py`. Il peut être exécuté avec différentes options de mode :

- Train : Mode d'entraînement du modèle.
- Test : Mode de test du modèle.

Exécutez le code avec la commande suivante :

```python
python main.py [mode] [options]
```


Les options disponibles sont les suivantes :

- `--train_name` : Nom de l'exécution de l'entraînement.
- `--batch_size` : Taille des lots (par défaut : 16).
- `--n_sample` : Nombre d'images à échantillonner (par défaut : 4).
- `--any_time` : Fréquence à laquelle les échantillons sont enregistrés pendant l'entraînement (par défaut : 400).
- `--model_path` : Chemin vers le modèle à charger (par défaut : None).
- `--lr` : Taux d'apprentissage (par défaut : 2e-5).
- `--adam_betas` : Paramètres betas pour l'optimiseur Adam (par défaut : (0.9, 0.99)).
- `--epochs` : Nombre d'époques d'entraînement (par défaut : 100).
- `--image_size` : Taille de l'image (par défaut : 128).
- `--data_dir` : Répertoire contenant les données (par défaut : 'data/').
- `--v_i` : Nombre d'indices de variables (par défaut : 3).
- `--var_indexes` : Liste des indices de variables (par défaut : ['u','v','t2m']).
- `--crop` : Effectuer un recadrage des images (par défaut : [78,206,55,183]).
- `--device` : Appareil utilisé pour l'entraînement (cpu ou cuda:x) (par défaut : 'cuda').
- `--wandbproject` : Nom du projet wandb (par défaut : "meteoDDPM").
- `-w`, `--wandb` : Enregistrer les journaux dans le fichier wandb.
- `--beta_schedule` : Calendrier beta (par défaut : "cosine").
- `--auto_normalize` : Activer la normalisation automatique (par défaut : False).

## Exemples

1. Entraîner le modèle :


```python
python main.py Train --train_name my_training_run --batch_size 32 --lr 1e-4 --epochs 50
```

2. Tester le modèle :


```python
python main.py Test --train_name my_training_run --n_sample
```



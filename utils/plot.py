import os
import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_grid(file_name_prefix, np_imgs):
    """
    Trace une grille d'images.
    Args:
        file_name_prefix (str): Préfixe du nom du fichier pour sauvegarder le tracé.
        np_imgs (List[numpy.ndarray]): Liste de tableaux d'images à tracer (maximum 6).
    """
    nb_image = len(np_imgs)
    fig, axes = plt.subplots(
        nrows=min(6, nb_image),
        ncols=len(np_imgs[0]),
        figsize=(10, 10),
    )
    for i in range(min(6, nb_image)):
        for j in range(len(np_imgs[0])):
            cmap = (
                "viridis" if j != 0 else "bwr"
            )  # Change le cmap pour la première image
            image = np_imgs[i][j]
            im = axes[i, j].imshow(image, cmap=cmap, origin="lower")
            axes[i, j].axis("off")
            fig.colorbar(im, ax=axes[i, j])
    # Sauvegarde le tracé dans le chemin de fichier spécifié
    plt.savefig(f"{file_name_prefix}_{i}.png", bbox_inches="tight")
    plt.close()


def lire_images_npy(dossier):
    images = []
    # Vérifie si le dossier existe
    if not os.path.exists(dossier):
        print("Le dossier spécifié n'existe pas.")
        return None

    # Liste tous les fichiers dans le dossier
    fichiers = os.listdir(dossier)

    # Parcours tous les fichiers
    for fichier in fichiers:
        # Vérifie si le fichier est un fichier .npy
        if fichier.endswith(".npy"):
            chemin_fichier = os.path.join(dossier, fichier)
            # Charge le fichier .npy en utilisant numpy
            image = np.load(chemin_fichier)
            images.append(image)

    return images


def main():
    parser = argparse.ArgumentParser(
        description="Lire des fichiers .npy et sauvegarder les images correspondantes en tant que .png"
    )
    parser.add_argument(
        "dossier",
        type=str,
        help="Chemin du dossier contenant les fichiers .npy",
    )
    args = parser.parse_args()

    dossier_images = args.dossier
    images = lire_images_npy(dossier_images)

    if images is not None:
        print("Nombre d'images lues :", len(images))

        # Divisez les images en lots de 4 pour les traiter ensemble
        for i in range(0, len(images), 4):
            batch_images = images[i : i + 4]
            plot_grid(os.path.join(dossier_images, f"batch_{i}"), batch_images)


if __name__ == "__main__":
    main()

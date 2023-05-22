import argparse
import gc
import os
import time

import matplotlib.pyplot as plt
import torch
import wandb
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torchvision import transforms
from tqdm import tqdm

import DataSet_Handler
from DataSet_Handler import ISData_Loader

gc.collect()
torch.cuda.empty_cache()

import numpy as np


def train_model(config):
    device = torch.device(config.device)
    # print(device)

    Dl_train, diffusion = create_model_data(config)

    model = diffusion.model

    dataloader, dataset = Dl_train.loader()

    optimizer = torch.optim.Adam(model.parameters(), lr = config.lr, betas = config.adam_betas)
    print()

    avg_loss = 0.0
    best_loss = float('inf')
    t = time.strftime("%d-%m-%y_%H-%M", time.localtime(time.time()))
    wandb.init(project=config.wp, entity="jrabault", name=f"{config.train_name}_{t}/",
               config={**vars(config), **{"trainvaldir": config.train_name,
                                           "optimizer": optimizer.__class__,
                                           "lr": optimizer.param_groups[0]["lr"],
                                           "weight_decay": optimizer.param_groups[0]["weight_decay"],
                                            "beta_schedule": config.beta_schedule,
                                            "auto_normalize": config.auto_normalize,}})

    ##### TRAIN #######
    loop = tqdm(range(config.epochs), desc=f"Training...", unit="epoch",postfix=f"Last loss : {avg_loss:.4f}")
    for epoch in loop:
        #, postfix=f"Last loss : {avg_loss:.4f}"
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{config.epochs}", unit="batch",
                          leave=False):
            optimizer.zero_grad()
            batch = batch.to(device)
            # print(batch.device)
            # print(config.device)
            loss = diffusion(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        # print(f'Epoch {epoch + 1}: Loss = {avg_loss}')
        loop.set_postfix_str(f"Last loss : {avg_loss:.4f}")
        # tqdm.refresh()

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), f"{config.train_name}/best.pt")
            config.model_path = f"{config.train_name}/best.pt"
            print(f"\n#Save model in {config.train_name}/best.pt at : Epoch {epoch + 1}: Loss = {avg_loss:.4f} \n")

        if epoch % config.any_time == 0.0 and epoch != 0:
            sample_images(config, ep=str(epoch), nb_image=config.n_sample)
            torch.save(model.state_dict(), f"{config.train_name}/save_{epoch}.pt")

        wandb_log = {"avg_loss": avg_loss}
        wandb.log(wandb_log)

    torch.save(model.state_dict(), f"{config.train_name}/last.pt")
    print("training done")

    # Enregistrement des logs d'entraînement
    save_config(avg_loss, config)


def save_config(avg_loss, config):
    config.model_path = f"{config.train_name}/best.pt"
    with open(f"{config.train_name}/train_log.txt", 'w') as f:
        f.write(f"Configuration:\n{config}\n\n")
        f.write(f"Epochs: {config.epochs}\n")
        f.write(f"Batch size: {config.batch_size}\n")
        f.write(f"Learning rate: {config.lr}\n")
        f.write(f"Training loss: {avg_loss}\n")


def sample_images(config, ep="0", nb_image=1):
    print()
    print(f"sampling for {nb_image} image..")

    Dl_train, diffusion = create_model_data(config)

    invTrans = transforms.Compose([transforms.Normalize(mean=[0.] *len(config.var_indexes),
                                                        std=[1 / el for el in Dl_train.stds]),
                                   transforms.Normalize(mean=[-el for el in Dl_train.means],
                                                        std=[1.]*len(config.var_indexes)), ])
    b = 0
    i = 0
    with tqdm(total=nb_image//config.batch_size, desc="Sampling :", unit="batch") as pbar:
        while b < nb_image:
            batch_size = min(nb_image - b, config.batch_size)
            sampled_images = diffusion.sample(batch_size=batch_size)

            b += batch_size
            if config.invert_norm:
                sampled_images_unnorm = invTrans(sampled_images)
            else:
                sampled_images_unnorm = sampled_images
            np_img = sampled_images_unnorm.cpu().numpy()
            for img1 in np_img:
                np.save(f"{config.train_name}/samples/_sample_{ep}_{i}.npy", img1)
                # np.save(f"{config.train_name}/samples/_sample_{ep}_{i}_norm.npy", sampled_images.cpu().numpy())
                i += 1
            pbar.update(1)

    fig, axes = plt.subplots(nrows=min(6,nb_image), ncols=len(config.var_indexes), figsize=(10, 10))

    for i in range(min(6,nb_image)):
        for j in range(len(config.var_indexes)):
            cmap = 'viridis' if config.var_indexes[j] != 't2m' else 'bwr'
            image = np_img[i, j]
            if len(config.var_indexes) >1:
                im = axes[i, j].imshow(image, cmap=cmap, origin='lower')
                axes[i, j].axis('off')
                fig.colorbar(im, ax=axes[i, j])
            else:
                im = axes[i].imshow(image, cmap=cmap, origin='lower')
                axes[i].axis('off')
                fig.colorbar(im, ax=axes[i])

    plt.savefig(f"{config.train_name}/samples/all_images_grid_{ep}.jpg", bbox_inches='tight')

    print(f"\nsampling done in {config.train_name}/samples/")


def create_model_data(config):
    device = torch.device(config.device)
    Dl_train = ISData_Loader(config.data_dir, config.batch_size,
                             [DataSet_Handler.var_dict[var] for var in config.var_indexes], config.crop,
                             device=config.device)
    # Chargement du meilleur modèle entraîné
    model = Unet(
        dim=int(config.image_size / 2),
        dim_mults=(1, 2, 4, 8),
        channels=len(config.var_indexes)
    )
    if config.model_path is not None:
        model.load_state_dict(torch.load(config.model_path))
        print(f"Best model loaded from {config.model_path}")
    else :
        print("model created")
    model.to(device)
    diffusion = GaussianDiffusion(
        model,
        image_size=config.image_size,
        timesteps=1000,  # nombre d'étapes
        loss_type='l1',  # L1 ou L2
        beta_schedule=config.beta_schedule,
        auto_normalize=config.auto_normalize
    )
    diffusion.to(device)
    return Dl_train, diffusion


def print_config(config):
    print("Configuration:")
    for arg in vars(config):
        print(f"\t{arg}: {getattr(config, arg)}")
    print()


def check_path(train_name):
    train_num = 1
    train_name= config.train_name
    while os.path.exists(config.train_name):
        if f"_{train_num}" in config.train_name:
            config.train_name = train_name.split('_')[0] + f"_{train_num + 1}"
            train_num += 1
        else:
            config.train_name = f"{train_name}_{train_num}"
    os.makedirs(config.train_name)
    return config       


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['Train', 'Test', 'Train2', 'Test2'], help='Mode d\'exécution : Train ou Test')
    parser.add_argument('--train_name', type=str, default='train', help='Name of the training run')
    parser.add_argument('--batch_size', type=int, default=16, help='Taille des batches')
    parser.add_argument('--n_sample', type=int, default=4, help='n_sample')
    parser.add_argument('--any_time', type=int, default=400, help='')
    parser.add_argument('--model_path', type=str, default=None, help='Chemin vers le modèle à charger')
    parser.add_argument('--lr', type=float, default=2e-5, help='Taux d\'apprentissage')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.99), help='')
    parser.add_argument('--epochs', type=int, default=100, help='Nombre d\'époques')
    parser.add_argument('--image_size', type=int, default=128, help='Taille de l\'image')
    parser.add_argument('--data_dir', type=str, default='data/', help='Répertoire contenant les données')
    parser.add_argument('--v_i', type=int, default=3, help='Nombre d\'indices des variables')
    parser.add_argument('--var_indexes', type=list, default=['u','v','t2m'], help='Liste des indices des variables')
    parser.add_argument('--crop', type=list, default=[78,206,55,183], help='Effectuer un recadrage des images')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Appareil utilisé pour l\'entraînement (cpu ou cuda:x)')
    parser.add_argument("--wandbproject", dest="wp", type=str, default="meteoDDPM", help="wandb project name")
    parser.add_argument("-w", "--wandb", dest="use_wandb",
                      default=False, action="store_true", help="save logs in wandb file")
    parser.add_argument("--invert_norm", dest="invert_norm",
                      default=False, action="store_true", help="invert_norm")
    parser.add_argument("--beta_schedule", type=str, default="cosine", help="")
    parser.add_argument("--auto_normalize", dest="auto_normalize",
                        default=False, action="store_true", help="")
    config = parser.parse_args()

    if not config.use_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    if config.mode == 'Train':
        print('Mode Train sélectionné')
    elif config.mode == 'Test':
        print('Mode Test sélectionné')

    config = check_path(config)

    config.var_indexes = ['t2m'] if config.v_i == 1 else ['u','v'] if config.v_i == 2 else ['u','v','t2m']

    os.makedirs(f"{config.train_name}/", exist_ok=True)
    os.makedirs(f"{config.train_name}/samples/", exist_ok=True)
    # os.makedirs(f"{config.train_name}/samples/", exist_ok=True)

    print_config(config)

    if config.mode == 'Train':
        train_model(config)
        os.makedirs(f"{config.train_name}/samples/", exist_ok=True)
        sample_images(config, "last", nb_image=config.n_sample)
    else:
        os.makedirs(f"{config.train_name}/samples/", exist_ok=True)
        sample_images(config, "test", nb_image=config.n_sample)

    if config.use_wandb:
        wandb.finish()

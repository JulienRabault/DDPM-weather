import argparse
import gc
import os
import time
import warnings
from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch import distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

import DataSet_Handler
from DataSet_Handler import ISData_Loader

warnings.filterwarnings("ignore", message="This DataLoader will create .* worker processes in total.*")
gc.collect()
torch.cuda.empty_cache()


def ddp_setup():
    if torch.cuda.device_count() < 2:
        return
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataset: DataSet_Handler.ISDataset,
            train_data,
            optimizer: torch.optim.Optimizer,
            snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_dataset = train_dataset
        self.train_data = train_data
        self.optimizer = optimizer
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.stds = train_dataset.stds
        self.means = train_dataset.means
        self.best_loss = float('inf')
        if snapshot_path is not None and os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)
        if torch.cuda.device_count() > 1:
            self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.best_loss = snapshot["BEST_LOSS"]
        self.stds = snapshot["STDS"]
        self.means = snapshot["MEANS"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, batch):
        self.optimizer.zero_grad()
        loss = self.model(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data)))
        if dist.is_initialized():
            self.train_data.sampler.set_epoch(epoch)
        total_loss = 0
        if self.gpu_id == 0:
            loop = tqdm(self.train_data, desc=f"Epoch {epoch + 1}/{config.epochs}", unit="batch",
                        leave=False)
        else:
            loop = self.train_data
        for batch in loop:
            batch = batch.to(self.gpu_id)
            total_loss += self._run_batch(batch)
        if config.debug_log:
            print(
                f"\n[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | Last loss: {total_loss / len(self.train_data)}")
        return total_loss / len(self.train_data)

    def _save_snapshot(self, epoch, path, loss):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict() if dist.is_initialized() else self.model.state_dict(),
            "EPOCHS_RUN": epoch,
            'OPTIMIZER_STATE': self.optimizer.state_dict(),
            'BEST_LOSS': self.best_loss,
            'STDS': self.stds,
            'MEANS': self.means
        }
        torch.save(snapshot, path)
        print(f"\n#INFO : Epoch {epoch} | Training snapshot saved at {path} | Loss: {loss}\n")

    def _init_wandb(self, config):
        if self.gpu_id != 0:
            return
        t = time.strftime("%d-%m-%y_%H-%M", time.localtime(time.time()))
        wandb.init(project=config.wp, entity="jrabault", name=f"{config.train_name}_{t}/",
                   config={**vars(config), **{"trainvaldir": config.train_name,
                                              "optimizer": self.optimizer.__class__,
                                              "lr": self.optimizer.param_groups[0]["lr"],
                                              "weight_decay": self.optimizer.param_groups[0]["weight_decay"],
                                              "beta_schedule": config.beta_schedule,
                                              "auto_normalize": config.beta_schedule, }})

    def train(self, config):
        avg_loss = 0.0
        best_loss = float('inf')
        if self.gpu_id == 0:
            self._init_wandb(config)
            loop = tqdm(range(self.epochs_run, config.epochs), desc=f"Training...", unit="epoch",
                        postfix=f"Last loss : {avg_loss:.4f}")
        else:
            loop = range(self.epochs_run, config.epochs)
        for epoch in loop:
            avg_loss = self._run_epoch(epoch)
            if self.gpu_id == 0:
                loop.set_postfix_str(f"Last loss : {avg_loss:.4f}")

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    self._save_snapshot(epoch, f"{config.train_name}/best.pt", avg_loss)

                if epoch % config.any_time == 0.0 and epoch != 0:
                    self.sample_images(config, ep=str(epoch), nb_image=config.n_sample)
                    self._save_snapshot(epoch, f"{config.train_name}/save_{epoch}.pt", avg_loss)

                wandb_log = {"avg_loss": avg_loss}
                wandb.log(wandb_log)
        if self.gpu_id == 0:
            wandb.finish()

    def sample_images(self, config, ep, nb_image=1):
        print()
        print(f"sampling for {nb_image} image..")
        invTrans = transforms.Compose([transforms.Normalize(mean=[0.] * len(config.var_indexes),
                                                            std=[1 / el for el in self.stds]),
                                       transforms.Normalize(mean=[-el for el in self.means],
                                                            std=[1.] * len(config.var_indexes)), ])
        b = 0
        i = 0
        with tqdm(total=nb_image // config.batch_size, desc="Sampling :", unit="batch") as pbar:
            while b < nb_image:
                batch_size = min(nb_image - b, config.batch_size)
                sampled_images = self.model.module.sample(
                    batch_size=batch_size) if dist.is_initialized() else self.model.sample(batch_size=batch_size)
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

        fig, axes = plt.subplots(nrows=min(6, nb_image), ncols=len(config.var_indexes), figsize=(10, 10))

        for i in range(min(6, nb_image)):
            for j in range(len(config.var_indexes)):
                cmap = 'viridis' if config.var_indexes[j] != 't2m' else 'bwr'
                image = np_img[i, j]
                if len(config.var_indexes) > 1:
                    im = axes[i, j].imshow(image, cmap=cmap, origin='lower')
                    axes[i, j].axis('off')
                    fig.colorbar(im, ax=axes[i, j])
                else:
                    im = axes[i].imshow(image, cmap=cmap, origin='lower')
                    axes[i].axis('off')
                    fig.colorbar(im, ax=axes[i])

        plt.savefig(f"{config.train_name}/samples/all_images_grid_{ep}.jpg", bbox_inches='tight')

        print(f"\nsampling done in {config.train_name}/samples/")


def load_train_objs(config):
    train_set = ISData_Loader(config.data_dir, config.batch_size,
                              [DataSet_Handler.var_dict[var] for var in config.var_indexes], config.crop).loader()[1]
    umodel = Unet(
        dim=int(config.image_size / 2),
        dim_mults=(1, 2, 4, 8),
        channels=len(config.var_indexes)
    )
    model = GaussianDiffusion(
        umodel,
        image_size=config.image_size,
        timesteps=1000,  # nombre d'étapes
        beta_schedule=config.beta_schedule,
        auto_normalize=config.auto_normalize
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.adam_betas)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    if dist.is_initialized():
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=cpu_count(),
            sampler=DistributedSampler(dataset),
            drop_last=True
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=cpu_count(),
        shuffle=True,
        drop_last=True
    )


def print_config(config):
    print("Configuration:")
    for arg in vars(config):
        print(f"\t{arg}: {getattr(config, arg)}")
    print()


def check_path(config):
    train_num = 1
    train_name = config.train_name
    while os.path.exists(config.train_name):
        if f"_{train_num}" in config.train_name:
            config.train_name = train_name.split('_')[0] + f"_{train_num + 1}"
            train_num += 1
        else:
            config.train_name = f"{train_name}_{train_num}"
    os.makedirs(config.train_name, exist_ok=True)
    return config


def main_train(config):
    dataset, model, optimizer = load_train_objs(config)
    train_data = prepare_dataloader(dataset, config.batch_size)
    debut = time.time()
    trainer = Trainer(model, dataset, train_data, optimizer, config.model_path)
    trainer.train(config)
    fin = time.time()
    temps_total = fin - debut
    if config.debug_log:
        print("Temps d'exécution du train: {} secondes".format(temps_total))
    if int(os.environ["LOCAL_RANK"]) == 0:
        trainer.sample_images(config, "last", nb_image=config.n_sample)


def main_test(config):
    if int(os.environ["LOCAL_RANK"]) == 0:
        dataset, model, optimizer = load_train_objs(config)
        train_data = prepare_dataloader(dataset, config.batch_size)
        trainer = Trainer(model, dataset, train_data, optimizer, config.model_path, )
        trainer.sample_images(config, "test", nb_image=config.n_sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['Train', 'Test'], help='Mode d\'exécution : Train ou Test')
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
    parser.add_argument('--var_indexes', type=list, default=['u', 'v', 't2m'], help='Liste des indices des variables')
    parser.add_argument('--crop', type=list, default=[78, 206, 55, 183], help='Effectuer un recadrage des images')
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
    parser.add_argument("--debug", dest="debug_log", default=False, action="store_true")
    config = parser.parse_args()

    if config.debug_log:
        import GPUtil

        print("#" * 20)
        print("int(os.environ[LOCAL_RANK]) : ", int(os.environ["LOCAL_RANK"]))
        GPUtil.showUtilization()
        print("#" * 20)

    if int(os.environ["LOCAL_RANK"]) == 0:
        if not config.use_wandb:
            os.environ['WANDB_MODE'] = 'disabled'

        if config.mode == 'Train':
            print('Mode Train sélectionné')
        elif config.mode == 'Test':
            print('Mode Test sélectionné')

        config = check_path(config)

        config.var_indexes = ['t2m'] if config.v_i == 1 else ['u', 'v'] if config.v_i == 2 else ['u', 'v', 't2m']

        os.makedirs(f"{config.train_name}/", exist_ok=True)
        os.makedirs(f"{config.train_name}/samples/", exist_ok=True)
        print_config(config)

    ddp_setup()

    if config.mode == 'Train':
        main_train(config)
    elif config.mode == 'Test':
        main_test(config)

    if dist.is_initialized():
        destroy_process_group()

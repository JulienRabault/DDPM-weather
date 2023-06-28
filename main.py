import argparse
import gc
import os
import time
import warnings
from multiprocessing import cpu_count
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from torch import distributed as dist, nn
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from tqdm import tqdm

import DataSet_Handler
from DataSet_Handler import ISData_Loader
import csv

warnings.filterwarnings("ignore", message="This DataLoader will create .* worker processes in total.*")
gc.collect()
torch.cuda.empty_cache()


def ddp_setup():
    if torch.cuda.device_count() < 2 or config.mode == 'Test':
        return
    if config.debug_log:
        print(f"\n#LOG : init_process_group(backend=nccl)")
    init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_dataset: DataSet_Handler.ISDataset,
            train_data,
            optimizer: torch.optim.Optimizer,
            snapshot_path: str,
    ) -> None:
        self.gpu_id = local_rank
        self.model = model.to(self.gpu_id)
        self.train_dataset = train_dataset
        self.train_data = train_data
        self.optimizer = optimizer
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.stds = train_dataset.stds
        self.means = train_dataset.means
        self.best_loss = float('inf')
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,T_0=134,T_mult=1)
        # self.scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters =config.epochs, power =2)
        if config.scheduler:
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr,
                                                                 total_steps=750 * len(train_data),
                                                                 anneal_strategy="cos",
                                                                 pct_start=0.2,
                                                                 div_factor=15.0, )
        else:
            self.scheduler = None
        if snapshot_path is not None:
            if self.gpu_id == 0:
                print(f"#INFO : Loading snapshot")
            self._load_snapshot(snapshot_path)
        # self.model = torch.compile(model)
        if dist.is_initialized():
            dist.barrier()
            self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.gpu_id], output_device=self.gpu_id)

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        if "SCHEDULER_STATE" in snapshot:
            self.scheduler.load_state_dict(snapshot["SCHEDULER_STATE"])
        if "WANDB_ID" in snapshot:
            self.wandb_id = snapshot["WANDB_ID"]
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        self.optimizer.load_state_dict(snapshot["OPTIMIZER_STATE"])
        self.best_loss = snapshot["BEST_LOSS"]
        self.stds = snapshot["STDS"]
        self.means = snapshot["MEANS"]
        if config.debug_log:
            print(f"\n#LOG : [GPU{self.gpu_id}]Resuming training from {snapshot_path} at Epoch {self.epochs_run}")
        elif self.gpu_id == 0:
            print(f"#INFO : Resuming training from {snapshot_path} at Epoch {self.epochs_run}")

    def _run_batch(self, batch):
        self.optimizer.zero_grad()
        loss = self.model(batch)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data)))
        iters = len(self.train_data)
        if dist.is_initialized():
            self.train_data.sampler.set_epoch(epoch)
        total_loss = 0
        if self.gpu_id == 0:
            loop = tqdm(enumerate(self.train_data), total=iters,
                        desc=f"Epoch {epoch}/{config.epochs + + self.epochs_run}", unit="batch",
                        leave=False, postfix=f"")
        else:
            loop = enumerate(self.train_data)
        for i, batch in loop:
            batch = batch.to(self.gpu_id)
            loss = self._run_batch(batch)
            total_loss += loss
            if config.scheduler:
                self.scheduler.step()
            if self.gpu_id == 0:
                loop.set_postfix_str(f"Loss : {total_loss / (i + 1):.4f}")
        if config.debug_log:
            print(
                f"\n#LOG : [GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)} | Last loss: {total_loss / len(self.train_data)} | Lr : {self.scheduler.get_last_lr()[0] if config.scheduler else config.lr}")

        return total_loss / len(self.train_data)

    def _save_snapshot(self, epoch, path, loss):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict() if dist.is_initialized() else self.model.state_dict(),
            "EPOCHS_RUN": epoch,
            'OPTIMIZER_STATE': self.optimizer.state_dict(),
            'BEST_LOSS': loss,
            'STDS': self.stds,
            'MEANS': self.means,
            "SCHEDULER_STATE": self.scheduler.state_dict(),
        }
        if config.use_wandb:
            snapshot["WANDB_ID"] = wandb.run.id
        torch.save(snapshot, path)
        print(f"\n#INFO : Epoch {epoch} | Training snapshot saved at {path} | Loss: {loss}\n")

    def _init_wandb(self, config):
        if self.gpu_id != 0:
            return
        t = time.strftime("%d-%m-%y_%H-%M", time.localtime(time.time()))
        if config.resume:
            wandb.init(id=self.wandb_id,project=config.wp, resume="must", entity="jrabault", name=f"{config.train_name}_{t}/",
                       config={**vars(config), **{"optimizer": self.optimizer.__class__,
                                                  "scheduler": self.scheduler.__class__,
                                                  "lr_base": self.optimizer.param_groups[0]["lr"],
                                                  "weight_decay": self.optimizer.param_groups[0]["weight_decay"], }})
        else:
            wandb.init(project=config.wp, entity="jrabault", name=f"{config.train_name}_{t}/",
                   config={**vars(config), **{"optimizer": self.optimizer.__class__,
                                              "scheduler": self.scheduler.__class__,
                                              "lr_base": self.optimizer.param_groups[0]["lr"],
                                              "weight_decay": self.optimizer.param_groups[0]["weight_decay"], }})

    def train(self, config):
        if self.gpu_id == 0:
            self._init_wandb(config)
            loop = tqdm(range(self.epochs_run, config.epochs + self.epochs_run), desc=f"Training...", unit="epoch",
                        postfix=f"")
        else:
            loop = range(self.epochs_run, config.epochs + self.epochs_run)
        for epoch in loop:
            avg_loss = self._run_epoch(epoch)
            if self.gpu_id == 0:
                loop.set_postfix_str(f"Epoch loss : {avg_loss:.5f} | Lr : {self.scheduler.get_last_lr()[0]:.6f}")

                if avg_loss < self.best_loss:
                    self.best_loss = avg_loss
                    self._save_snapshot(epoch, f"{config.train_name}/best.pt", avg_loss)

                if epoch % config.any_time == 0.0:
                    self.sample_images(config, ep=str(epoch), nb_image=config.n_sample)
                    self._save_snapshot(epoch, f"{config.train_name}/save_{epoch}.pt", avg_loss)

                log = {"avg_loss": avg_loss, "lr": self.scheduler.get_last_lr()[0] if config.scheduler else config.lr}
                self._log(epoch, log)
                self._save_snapshot(epoch, f"{config.train_name}/last.pt", avg_loss)

        if self.gpu_id == 0:
            wandb.finish()

    def sample_images(self, config, ep=None, nb_image=4):
        if self.gpu_id == 0:
            print()
            print(f"sampling for {nb_image * torch.cuda.device_count()} image..")
        invTrans = transforms.Compose([transforms.Normalize(mean=[0.] * len(config.var_indexes),
                                                            std=[1 / el for el in self.stds]),
                                       transforms.Normalize(mean=[-el for el in self.means],
                                                            std=[1.] * len(config.var_indexes)), ])
        b = 0
        i = self.gpu_id
        with tqdm(total=nb_image // config.batch_size, desc="Sampling :", unit="batch") as pbar:
            while b < nb_image:
                print(f"\n#LOG : [GPU{self.gpu_id}] nb_image:{nb_image}, i:{i}, b:{b}")
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
                    if ep is not None:
                        np.save(f"{config.train_name}/samples/_sample_{ep}_{i}.npy", img1)
                    else:
                        np.save(f"{config.train_name}/samples/_sample_{i}.npy", img1)
                    if torch.cuda.device_count() > 1:
                        i += torch.cuda.device_count()
                    else:
                        i += 1
                pbar.update(1)

        if torch.cuda.device_count() < 2:
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

    def _log(self, epoch, log_dict):
        wandb.log(log_dict, step=epoch)
        csv_filename = f"{config.train_name}/logs_train.csv"
        file_exists = Path(csv_filename).is_file()
        with open(csv_filename, 'a' if file_exists else 'w', newline='') as csvfile:
            fieldnames = ['epoch'] + list(log_dict.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({**{'epoch': epoch}, **log_dict})


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
        timesteps=1000,  # nombre d'�tapes
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
            sampler=DistributedSampler(dataset, rank=local_rank, shuffle=True, drop_last=False),
            drop_last=False
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=cpu_count(),
        shuffle=True,
        drop_last=False
    )


def print_config(config):
    print("Configuration:")
    for arg in vars(config):
        print(f"\t{arg}: {getattr(config, arg)}")
    print()


def save_config(config):
    with open(f"{config.train_name}/config.txt", 'w') as f:
        for arg in vars(config):
            f.write(f"\t{arg}: {getattr(config, arg)}\n")


def check_config(config):
    if config.resume and (config.snapshot_path is None or not os.path.isfile(config.snapshot_path)):
        raise FileNotFoundError(f"config.resume={config.resume} but snapshot_path={config.snapshot_path} is None or doesn't exist")
    if local_rank == 0:
        if config.mode == 'Train':
            print(f'#INFO : Mode Train selectionn�')
        elif config.mode == 'Test':
            print(f'#INFO : Mode Test selectionn�')
    # Path
    paths = [
        f"{config.train_name}/",
        f"{config.train_name}/samples/",
        f"{config.train_name}/WANDB/",
        f"{config.train_name}/WANDB/cache",
    ]
    if local_rank == 0:
        if config.resume:
            for path in paths:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"The following directorie do not exist: {path}")
        else:
            train_num = 1
            train_name = config.train_name
            while os.path.exists(config.train_name):
                if f"_{train_num}" in config.train_name:
                    config.train_name = "_".join(train_name.split('_')[:-1]) + f"_{train_num + 1}"
                    train_num += 1
                else:
                    config.train_name = f"{train_name}_{train_num}"
            for path in paths:
                os.makedirs(path, exist_ok=True)
    # Sample
    if torch.cuda.device_count() > 1:
        world_size = torch.cuda.device_count()

        n_sample = config.n_sample

        if n_sample % world_size != 0:
            raise ValueError(f"n_sample={n_sample} is not divisible by world_size gpus={torch.cuda.device_count()}")

        config.n_sample = n_sample // world_size
    config.var_indexes = ['t2m'] if config.v_i == 1 else ['u', 'v'] if config.v_i == 2 else ['u', 'v', 't2m']
    # Save
    if local_rank == 0:
        save_config(config)
        print_config(config)

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
        print(f"\n#LOG : Temps d'execution du train: {temps_total} secondes")
    if local_rank == 0:
        trainer.sample_images(config, "last", nb_image=config.n_sample)


def main_test(config):
    dataset, model, optimizer = load_train_objs(config)
    train_data = prepare_dataloader(dataset, config.batch_size)
    trainer = Trainer(model, dataset, train_data, optimizer, config.model_path, )
    trainer.sample_images(config, nb_image=config.n_sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['Train', 'Test'], help='Mode d\'ex�cution : Train ou Test')
    parser.add_argument('--train_name', type=str, default='train', help='Name of the training run')
    parser.add_argument('--batch_size', type=int, default=16, help='Taille des batches')
    parser.add_argument('--n_sample', type=int, default=4, help='n_sample')
    parser.add_argument('--any_time', type=int, default=400, help='')
    parser.add_argument('--model_path', type=str, default=None, help='Chemin vers le mod�le � charger')
    parser.add_argument('--lr', type=float, default=5e-4, help='Taux d\'apprentissage')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.99), help='')
    parser.add_argument('--epochs', type=int, default=100, help='Nombre d\'�poques')
    parser.add_argument('--image_size', type=int, default=128, help='Taille de l\'image')
    parser.add_argument('--data_dir', type=str, default='data/', help='R�pertoire contenant les donn�es')
    parser.add_argument('--v_i', type=int, default=3, help='Nombre d\'indices des variables')
    parser.add_argument('--var_indexes', type=list, default=['u', 'v', 't2m'], help='Liste des indices des variables')
    parser.add_argument('--crop', type=list, default=[78, 206, 55, 183], help='Effectuer un recadrage des images')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Appareil utilis� pour l\'entra�nement (cpu ou cuda:x)')
    parser.add_argument("--wandbproject", dest="wp", type=str, default="meteoDDPM", help="wandb project name")
    parser.add_argument("-w", "--wandb", dest="use_wandb",
                        default=False, action="store_true", help="save logs in wandb file")
    parser.add_argument("--invert_norm", dest="invert_norm",
                        default=False, action="store_true", help="invert_norm")
    parser.add_argument("--beta_schedule", type=str, default="cosine", help="")
    parser.add_argument("--auto_normalize", dest="auto_normalize",
                        default=False, action="store_true", help="")
    parser.add_argument("--scheduler", dest="scheduler",
                        default=False, action="store_true", help="")
    parser.add_argument("-r","--resume", dest="resume",
                        default=False, action="store_true", help="")
    parser.add_argument("--debug", dest="debug_log", default=False, action="store_true")
    config = parser.parse_args()

    try:
        local_rank = int(os.environ["LOCAL_RANK"])
    except KeyError:
        local_rank = 0

    ddp_setup()

    config = check_config(config)

    if local_rank == 0:
        if not config.use_wandb:
            os.environ['WANDB_MODE'] = 'disabled'
        else:
            os.environ['WANDB_MODE'] = 'offline'
            os.environ['WANDB_CACHE_DIR'] = f"{config.train_name}/WANDB/cache"
            os.environ['WANDB_DIR'] = f"{config.train_name}/WANDB/"

    if config.mode == 'Train':
        main_train(config)
    elif config.mode == 'Test':
        main_test(config)

    if dist.is_initialized():
        destroy_process_group()


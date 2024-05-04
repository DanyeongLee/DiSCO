import os
import random
import pickle
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call, to_absolute_path
from test import generate, test

import wandb

from src.datasets.transforms import AddNoise


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


def train(model, diffusion, loader, optimizer, ema, device, bsz, name, grad_clip=None):
    model.train()
    epoch_loss = 0.
    tqdm_bar = tqdm(loader, desc='Training', dynamic_ncols=True)
    for i, batch in enumerate(tqdm_bar):
        batch = batch.to(device)
        optimizer.zero_grad()
        loss = diffusion.training_loss(model, batch, bsz)
        loss.backward()

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        optimizer.step()
        ema.update(model.parameters())
        epoch_loss += loss.item()
        tqdm_bar.set_postfix({'train_loss': epoch_loss / (i + 1),
                              'name': name})

    return epoch_loss / (i + 1)


@torch.no_grad()
def evaluate(model, diffusion, loader, device, bsz):
    model.eval()
    epoch_loss = 0.
    epoch_samples = 0
    tqdm_bar = tqdm(loader, desc='Evaluating', dynamic_ncols=True)
    for i, batch in enumerate(tqdm_bar):
        batch = batch.to(device)
        loss = diffusion.training_loss(model, batch, bsz)

        n_samples = batch.batch[-1].item() + 1
        epoch_samples += n_samples
        epoch_loss += loss.item() * n_samples
        tqdm_bar.set_postfix({'val_loss': epoch_loss / epoch_samples})
    return epoch_loss / epoch_samples


@torch.no_grad()
def forward_dummy_batch(model, diffusion, loader, bsz):
    batch = next(iter(loader))
    diffusion.training_loss(model, batch, bsz)


@hydra.main(version_base='1.3', config_path='configs', config_name='main')
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(project=cfg.logger.project, entity=cfg.logger.entity, name=cfg.name, config=wandb_config)

    gen_sample_path = cfg.ckpt_path.split('/')[:2]
    gen_sample_path = os.path.join(*gen_sample_path)

    device = torch.device(f'cuda:{cfg.gpu}')

    if cfg.dataset.etkdg_noise is None:
        transform = None
    else:
        transform = AddNoise(cfg.dataset.etkdg_noise)

    if cfg.dataset.from_torsion:
        from src.datasets.dataset_torsion import MyDataset, collate_fn
        train_data = MyDataset(cfg.dataset.train_path, cfg.dataset.torsion_train_path, transform=transform)
        val_data = MyDataset(cfg.dataset.val_path, cfg.dataset.torsion_val_path, transform=transform)
    elif cfg.dataset.from_rc:
        from src.datasets.dataset_rc import MyDataset, collate_fn
        train_data = MyDataset(cfg.dataset.train_path, cfg.dataset.rc_train_path, transform=transform)
        val_data = MyDataset(cfg.dataset.val_path, cfg.dataset.rc_val_path, transform=transform)
    elif cfg.dataset.from_dmcg:
        from src.datasets.dataset_dmcg import MyDataset, collate_fn
        train_data = MyDataset(cfg.dataset.train_path, cfg.dataset.dmcg_train_path, transform=transform)
        val_data = MyDataset(cfg.dataset.val_path, cfg.dataset.dmcg_val_path, transform=transform)
    else:
        from src.datasets.dataset import MyDataset, collate_fn
        train_data = MyDataset(cfg.dataset.train_path, etkdg=cfg.dataset.etkdg, transform=transform)
        val_data = MyDataset(cfg.dataset.val_path, etkdg=cfg.dataset.etkdg, transform=transform)

    train_loader = DataLoader(train_data, batch_size=cfg.train.batch_size, collate_fn=collate_fn,
                              shuffle=True, drop_last=True, num_workers=cfg.train.num_workers, pin_memory=True)    
    val_loader = DataLoader(val_data, batch_size=cfg.train.batch_size, collate_fn=collate_fn,
                            shuffle=False, drop_last=False, num_workers=cfg.train.num_workers, pin_memory=True)

    beta = call(cfg.diffusion.noise_schedule)
    diffusion = instantiate(cfg.diffusion.diffusion)(beta=beta)
    model = instantiate(cfg.model)
    forward_dummy_batch(model, diffusion, train_loader, cfg.train.batch_size)  # initializing lazy modules

    model = model.to(device)
    diffusion = diffusion.to(device)

    if cfg.train.optimizer_type == 'adam':
        opt_class = torch.optim.Adam
    elif cfg.train.optimizer_type == 'adamw':
        opt_class = torch.optim.AdamW
    optimizer = opt_class(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=cfg.train.patience,
                                                           factor=0.7, min_lr=cfg.train.lr / 100, verbose=True)
    ema = ExponentialMovingAverage(model.parameters(), decay=cfg.train.ema_decay)

    ckpt_dir = os.path.dirname(to_absolute_path(cfg.ckpt_path))
    os.makedirs(to_absolute_path(ckpt_dir), exist_ok=True)
    val_loss_best = np.inf

    for epoch in range(cfg.train.n_epochs):
        print(f'Epoch: {epoch}')
        train_loss = train(model, diffusion, train_loader, optimizer, ema, device, cfg.train.batch_size, cfg.name, cfg.train.grad_clip)

        with ema.average_parameters():
            val_loss = evaluate(model, diffusion, val_loader, device, cfg.train.batch_size)
            scheduler.step(val_loss)
            if val_loss < val_loss_best:
                val_loss_best = val_loss
                torch.save(model.state_dict(), cfg.ckpt_path)

            if (epoch+1) % cfg.train.test_interval == 0:
                if 'drugs' in cfg.dataset.train_path:
                    threshold = 1.25
                elif 'qm9' in cfg.dataset.train_path:
                    threshold = 0.5

                gen_samples_dict = generate(model, diffusion, val_data, device,
                                    cfg.train.test_batch_size, cfg.train.n_test_samples,
                                    cfg.train.test_etkdg_noise, cfg.dataset.etkdg,
                                    cfg.train.return_traj, cfg.train.num_workers,
                                    cfg.dataset.from_torsion, cfg.dataset.from_rc, cfg.dataset.from_dmcg)
                
                gen_sample_name = f'gen_samples_traj{cfg.train.return_traj}_noise{cfg.train.test_etkdg_noise}_epoch{epoch}.pkl'
                gen_sample_file = os.path.join(gen_sample_path, gen_sample_name)

                with open(gen_sample_file, 'wb') as f:
                    pickle.dump(gen_samples_dict, f)

                (cov_r_mean, mat_r_mean, 
                cov_p_mean, mat_p_mean, 
                cov_r_median, mat_r_median, 
                cov_p_median, mat_p_median) = test(gen_samples_dict, threshold, cfg.train.num_workers)
                
                print(f'cov_r_mean: {cov_r_mean}, mat_r_mean: {mat_r_mean}, cov_p_mean: {cov_p_mean}, mat_p_mean: {mat_p_mean}')
                print(f'cov_r_median: {cov_r_median}, mat_r_median: {mat_r_median}, cov_p_median: {cov_p_median}, mat_p_median: {mat_p_median}')

                wandb.log({
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'val/cov_r_mean': cov_r_mean, 'val/mat_r_mean': mat_r_mean,
                    'val/cov_p_mean': cov_p_mean, 'val/mat_p_mean': mat_p_mean,
                    'val/cov_r_median': cov_r_median, 'val/mat_r_median': mat_r_median,
                    'val/cov_p_median': cov_p_median, 'val/mat_p_median': mat_p_median})
            else:
                print(f'val_loss: {val_loss}')
                wandb.log({
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    })

    seed_everything(cfg.seed)
    print('Training finished! Loading the best model...')

    torch.load(cfg.ckpt_path)
    if 'drugs' in cfg.dataset.train_path:
        threshold = 1.25
    elif 'qm9' in cfg.dataset.train_path:
        threshold = 0.5

    gen_samples_dict = generate(model, diffusion, val_data, device,
                        cfg.train.test_batch_size, cfg.train.n_test_samples,
                        cfg.train.test_etkdg_noise, cfg.dataset.etkdg,
                        cfg.train.return_traj, cfg.train.num_workers,
                        cfg.dataset.from_torsion, cfg.dataset.from_rc, cfg.dataset.from_dmcg)
    
    gen_sample_name = f'gen_samples_traj{cfg.train.return_traj}_noise{cfg.train.test_etkdg_noise}_best.pkl'
    gen_sample_file = os.path.join(gen_sample_path, gen_sample_name)

    with open(gen_sample_file, 'wb') as f:
        pickle.dump(gen_samples_dict, f)

    (cov_r_mean, mat_r_mean, 
    cov_p_mean, mat_p_mean, 
    cov_r_median, mat_r_median, 
    cov_p_median, mat_p_median) = test(gen_samples_dict, threshold, cfg.train.num_workers)

    wandb.log({
        'final/cov_r_mean': cov_r_mean, 'final/mat_r_mean': mat_r_mean,
        'final/cov_p_mean': cov_p_mean, 'final/mat_p_mean': mat_p_mean,
        'final/cov_r_median': cov_r_median, 'final/mat_r_median': mat_r_median,
        'final/cov_p_median': cov_p_median, 'final/mat_p_median': mat_p_median})


if __name__ == '__main__':
    main()
    
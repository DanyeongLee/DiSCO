import os
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch_ema import ExponentialMovingAverage

import hydra
from hydra.utils import instantiate, call, to_absolute_path
from omegaconf import DictConfig

from disco import DiSCO
from disco.utils import seed_everything


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

    gen_sample_path = cfg.ckpt_path.split('/')[:2]
    gen_sample_path = os.path.join(*gen_sample_path)

    device = torch.device(f'cuda:{cfg.gpu}')

    if cfg.dataset.from_torsion:
        from disco.datasets.dataset_torsion import ConformerDataset, LMDBDataset, collate_fn
        #train_data = ConformerDataset(cfg.dataset.train_path, cfg.dataset.torsion_train_path)
        #val_data = ConformerDataset(cfg.dataset.val_path, cfg.dataset.torsion_val_path)
        train_data = LMDBDataset(cfg.dataset.train_path, cfg.dataset.torsion_train_path)
        val_data = LMDBDataset(cfg.dataset.val_path, cfg.dataset.torsion_val_path)
    else:
        from disco.datasets.dataset import ConformerDataset, LMDBDataset, collate_fn
        #train_data = ConformerDataset(cfg.dataset.train_path, mmff=cfg.dataset.mmff)
        #val_data = ConformerDataset(cfg.dataset.val_path, mmff=cfg.dataset.mmff)
        train_data = LMDBDataset(cfg.dataset.train_path, mmff=cfg.dataset.mmff)
        val_data = LMDBDataset(cfg.dataset.val_path, mmff=cfg.dataset.mmff)
    
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

    opt_class = {'adam' : torch.optim.Adam, 'adamw': torch.optim.AdamW}[cfg.train.optimizer_type]
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
            else:
                print(f'val_loss: {val_loss}')

    model.load_state_dict(torch.load(cfg.ckpt_path))
    disco = DiSCO(diffusion, model)
    torch.save(disco, f'deployed/{cfg.name}.pt')

if __name__ == '__main__':
    main()
    

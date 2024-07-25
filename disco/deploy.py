import torch
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate, call

from disco import DiSCO


@hydra.main(version_base='1.3', config_path='configs', config_name='main')
def main(cfg: DictConfig):
    beta = call(cfg.diffusion.noise_schedule)
    diffusion = instantiate(cfg.diffusion.diffusion)(beta=beta)
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.ckpt_path))
    disco = DiSCO(diffusion, model, 'qm9' if 'qm9' in cfg.dataset.train_path else 'drugs')
    torch.save(disco, f'deployed/{cfg.name}.pt')


if __name__ == '__main__':
    main()
    

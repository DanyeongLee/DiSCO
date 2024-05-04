import os
import random
import pickle
from tqdm import tqdm

import numpy as np

import torch
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate, call, to_absolute_path

from rdkit import Chem

from src.utils.evaluation import dataset_to_each_mol, gen_samples, calc_metrics
from src.datasets.dataset import MyDataset, collate_fn
from src.datasets.transforms import AddNoise


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


@torch.no_grad()
def generate(model, diffusion, dataset, device, bsz, 
             n_test_samples=200, etkdg_scale=None, etkdg_version=3,
             return_traj=1, num_workers=1, from_torsion=False, from_rc=False, from_dmcg=False
             ):
    model.eval()
    mols, positions = dataset_to_each_mol(dataset)
    keys = list(mols.keys())[:n_test_samples]

    if from_torsion:
        torsion_data_dict = dataset.torsion_data_dict
    elif from_rc:
        rc_conf_dict = dataset.rc_conf_dict
    elif from_dmcg:
        dmcg_data_dict = dataset.dmcg_data_dict

    gen_samples_dict = {}

    for key in tqdm(keys, desc='Generating conformers for each molecule...'):
        mol, position = mols[key], positions[key]

        if from_torsion:
            torsion_mol_list = torsion_data_dict[mol.smiles]
            samples, starting_point = gen_samples(model, diffusion, mol, position,
                                  device, bsz, etkdg_scale, etkdg_version,
                                  return_traj, num_workers,
                                  n_to_generate=50, from_torsion=True, torsion_mol_list=torsion_mol_list)
        elif from_rc:
            rc_mol_list = rc_conf_dict[mol.smiles]
            samples, starting_point = gen_samples(model, diffusion, mol, position,
                                    device, bsz, etkdg_scale, etkdg_version,
                                    return_traj, num_workers,
                                    n_to_generate=50, from_torsion=False, torsion_mol_list=None, rc_mol_list=rc_mol_list)
        elif from_dmcg:
            canon_smi = Chem.MolToSmiles(mol.rdmol)
            dmcg_mol_list = dmcg_data_dict[canon_smi]['gen_confs']
            samples, starting_point = gen_samples(model, diffusion, mol, position,
                                    device, bsz, etkdg_scale, etkdg_version,
                                    return_traj, num_workers,
                                    n_to_generate=50, from_torsion=False, torsion_mol_list=None, rc_mol_list=None, dmcg_mol_list=dmcg_mol_list)
        else:
            samples, starting_point = gen_samples(model, diffusion, mol, position,
                                    device, bsz, etkdg_scale, etkdg_version,
                                    return_traj, num_workers, 
                                    n_to_generate=50, from_torsion=False, torsion_mol_list=None, rc_mol_list=None, dmcg_mol_list=None)
        samples = [s for s in samples]
        position = [p for p in position]
        if hasattr(mol, 'rdmol_org'):
            gen_samples_dict[mol.smiles] = {'gen_confs': samples, 'starting_point': starting_point,
                                            'ref_confs': position, 'rdmol': mol.rdmol, 'rdmol_org': mol.rdmol_org}
        else:
            gen_samples_dict[mol.smiles] = {'gen_confs': samples, 'starting_point': starting_point,
                                            'ref_confs': position, 'rdmol': mol.rdmol}

    return gen_samples_dict


@torch.no_grad()
def forward_dummy_batch(model, diffusion, loader):
    batch = next(iter(loader))
    diffusion.training_loss(model, batch)


@hydra.main(version_base='1.3', config_path='configs', config_name='main')
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    
    gen_sample_path = cfg.ckpt_path.split('/')[:2]
    gen_sample_path = os.path.join(*gen_sample_path)
    gen_sample_path = os.path.join(gen_sample_path, 
                                   f'gen_samples_prop.pkl')
    
    if os.path.exists(gen_sample_path):
        print(f'Already exists in {gen_sample_path}')
        return

    device = torch.device(f'cuda:{cfg.gpu}')

    if cfg.dataset.etkdg_noise is None:
        transform = None
    else:
        transform = AddNoise(cfg.dataset.etkdg_noise)

    test_path = 'data/qm9_processed/qm9_property.pkl'
    if cfg.dataset.from_torsion:
        from src.datasets.dataset_torsion import MyDataset
        test_data = MyDataset(test_path, 
                                '../torsional-diffusion/geodiff_split_qm9_run/property_qm9_20steps.pkl', 
                                transform=transform)
    elif cfg.dataset.from_rc:
        from src.datasets.dataset_rc import MyDataset
        test_data = MyDataset(test_path, cfg.dataset.rc_test_path, transform=transform)
    elif cfg.dataset.from_dmcg:
        from src.datasets.dataset_dmcg import MyDataset
        test_data = MyDataset(test_path, '../outputs/qm9/prop.pkl', transform=transform)
    else:
        from src.datasets.dataset import MyDataset
        test_data = MyDataset(test_path, etkdg=cfg.dataset.etkdg, transform=transform)

    beta = call(cfg.diffusion.noise_schedule)
    diffusion = instantiate(cfg.diffusion.diffusion)(beta=beta)

    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.ckpt_path))

    model = model.to(device)
    diffusion = diffusion.to(device)
    gen_samples_dict = generate(model, diffusion, test_data, device,
                                cfg.train.test_batch_size, cfg.train.n_test_samples,
                                cfg.train.test_etkdg_noise, cfg.dataset.etkdg,
                                cfg.train.return_traj, cfg.train.num_workers,
                                cfg.dataset.from_torsion, cfg.dataset.from_rc, cfg.dataset.from_dmcg)
    with open(gen_sample_path, 'wb') as f:
        pickle.dump(gen_samples_dict, f)
    
        
if __name__ == '__main__':
    main()
    
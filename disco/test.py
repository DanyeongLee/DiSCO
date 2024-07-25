import os
import random
import pickle
from tqdm import tqdm

import numpy as np

import torch
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate, call
from disco.utils import dataset_to_each_mol, gen_samples, calc_metrics, seed_everything


@torch.no_grad()
def generate(model, diffusion, dataset, device, bsz, 
             n_test_samples=200,
             return_traj=False, num_workers=1, from_torsion=False,
                mmff=False, mmff_iters=1000
             ):
    model.eval()
    mols, positions = dataset_to_each_mol(dataset)
    keys = list(mols.keys())[:n_test_samples]

    if from_torsion:
        torsion_data_dict = dataset.dataset.torsion_data_dict

    gen_samples_dict = {}

    for key in tqdm(keys, desc='Generating conformers for each molecule...'):
        mol, position = mols[key], positions[key]

        if from_torsion:
            torsion_mol_list = torsion_data_dict[mol.smiles]
            samples, starting_point = gen_samples(model, diffusion, mol, position,
                                  device, bsz,
                                  return_traj, num_workers,
                                  n_to_generate=None, from_torsion=True, torsion_mol_list=torsion_mol_list,
                                    mmff=mmff, mmff_iters=mmff_iters)
        else:
            samples, starting_point = gen_samples(model, diffusion, mol, position,
                                    device, bsz,
                                    return_traj, num_workers, 
                                    n_to_generate=None, from_torsion=False, torsion_mol_list=None,
                                    mmff=mmff, mmff_iters=mmff_iters)

        if return_traj:
            traj = samples
            samples = samples[-1]

        position = [p.cpu().numpy() for p in position]
        if hasattr(mol, 'rdmol_org'):
            gen_samples_dict[mol.smiles] = {'gen_confs': samples, 'starting_point': starting_point,
                                            'ref_confs': position, 'rdmol': mol.rdmol, 'rdmol_org': mol.rdmol_org}
        else:
            gen_samples_dict[mol.smiles] = {'gen_confs': samples, 'starting_point': starting_point,
                                            'ref_confs': position, 'rdmol': mol.rdmol}
            
        if return_traj:
            gen_samples_dict[mol.smiles]['traj'] = traj

    return gen_samples_dict


@torch.no_grad()
def test(gen_samples_dict, threshold=1.25, num_workers=1, test_starting_point=False):
    from multiprocess import Manager, Process

    def calc_metrics_and_append(value, cov_rs, mat_rs, cov_ps, mat_ps):
        gen_confs, ref_confs, rdmol = value['gen_confs'], value['ref_confs'], value['rdmol']
        starting_point = value['starting_point']
        if test_starting_point:
            gen_confs = starting_point

        if 'rdmol_org' in value.keys():
            rdmol_org = value['rdmol_org']
            cov_r, mat_r = calc_metrics(gen_confs, ref_confs, rdmol, threshold, rdmol_org=rdmol_org)
            cov_p, mat_p = calc_metrics(ref_confs, gen_confs, rdmol, threshold, rdmol_org=rdmol_org)
        else:
            cov_r, mat_r = calc_metrics(gen_confs, ref_confs, rdmol, threshold)
            cov_p, mat_p = calc_metrics(ref_confs, gen_confs, rdmol, threshold)
        cov_rs.append(cov_r)
        mat_rs.append(mat_r)
        cov_ps.append(cov_p)
        mat_ps.append(mat_p)

    with Manager() as manager:
        cov_rs, mat_rs = manager.list(), manager.list()
        cov_ps, mat_ps = manager.list(), manager.list()
        
        for step in tqdm(range(0, len(gen_samples_dict), num_workers)):
            process_list = []
            for value in list(gen_samples_dict.values())[step:step+num_workers]:
                p = Process(target=calc_metrics_and_append, args=(value, cov_rs, mat_rs, cov_ps, mat_ps))
                p.start()
                process_list.append(p)
            for p in process_list:
                p.join()

        cov_rs = list(cov_rs)
        mat_rs = list(mat_rs)
        cov_ps = list(cov_ps)
        mat_ps = list(mat_ps)
    
    cov_r_mean, cov_r_median = np.mean(cov_rs), np.median(cov_rs)
    mat_r_mean, mat_r_median = np.mean(mat_rs), np.median(mat_rs)
    cov_p_mean, cov_p_median = np.mean(cov_ps), np.median(cov_ps)
    mat_p_mean, mat_p_median = np.mean(mat_ps), np.median(mat_ps)

    return (cov_r_mean, mat_r_mean, cov_p_mean, mat_p_mean,
             cov_r_median, mat_r_median, cov_p_median, mat_p_median)


@hydra.main(version_base='1.3', config_path='configs', config_name='main')
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    
    gen_sample_path = cfg.ckpt_path.split('/')[:2]
    gen_sample_path = os.path.join(*gen_sample_path)

    if cfg.train.return_traj:
        gen_sample_path = os.path.join(gen_sample_path, 
                                       f'gen_samples_traj.pkl')
    elif cfg.train.test_mmff:
        gen_sample_path = os.path.join(gen_sample_path, 
                                       f'gen_samples_mmff{cfg.train.test_mmff_iters}.pkl')
    else:
        gen_sample_path = os.path.join(gen_sample_path, 
                                       f'gen_samples.pkl')
    print(f'gen_sample_path: {gen_sample_path}')
    
    if os.path.exists(gen_sample_path):
        print('Load generated samples')
        with open(gen_sample_path, 'rb') as f:
            gen_samples_dict = pickle.load(f)
    else:
        device = torch.device(f'cuda:{cfg.gpu}')

        if cfg.dataset.from_torsion:
            from disco.datasets.dataset_torsion import ConformerDataset, LMDBDataset, collate_fn
            #test_data = ConformerDataset(cfg.dataset.test_path, cfg.dataset.torsion_test_path)
            test_data = LMDBDataset(cfg.dataset.test_path, cfg.dataset.torsion_test_path)
            test_data = DataLoader(test_data, batch_size=1, collate_fn=collate_fn)
        else:
            from disco.datasets.dataset import ConformerDataset, LMDBDataset, collate_fn
            #test_data = ConformerDataset(cfg.dataset.test_path, mmff=cfg.dataset.mmff)
            test_data = LMDBDataset(cfg.dataset.test_path, mmff=cfg.dataset.mmff)
            test_data = DataLoader(test_data, batch_size=1, collate_fn=collate_fn)

        beta = call(cfg.diffusion.noise_schedule)
        diffusion = instantiate(cfg.diffusion.diffusion)(beta=beta)

        model = instantiate(cfg.model)
        model.load_state_dict(torch.load(cfg.ckpt_path))

        model = model.to(device)
        diffusion = diffusion.to(device)
        gen_samples_dict = generate(model, diffusion, test_data, device,
                                    cfg.train.test_batch_size, cfg.train.n_test_samples,
                                    cfg.train.return_traj, cfg.train.num_workers,
                                    cfg.dataset.from_torsion,
                                    cfg.train.test_mmff, cfg.train.test_mmff_iters,
                                    )
        with open(gen_sample_path, 'wb') as f:
            pickle.dump(gen_samples_dict, f)
    
    if 'drugs' in cfg.dataset.train_path:
        threshold = 1.25
    elif 'qm9' in cfg.dataset.train_path:
        threshold = 0.5

    (cov_r_mean, mat_r_mean, 
     cov_p_mean, mat_p_mean, 
     cov_r_median, mat_r_median, 
     cov_p_median, mat_p_median) = test(gen_samples_dict, threshold, cfg.train.num_workers, test_starting_point=True)
    print('=================== Results from baseline method ===================')
    print(f'cov_r_mean: {cov_r_mean}, mat_r_mean: {mat_r_mean}')
    print(f'cov_p_mean: {cov_p_mean}, mat_p_mean: {mat_p_mean}')
    print(f'cov_r_median: {cov_r_median}, mat_r_median: {mat_r_median}')
    print(f'cov_p_median: {cov_p_median}, mat_p_median: {mat_p_median}')

    (cov_r_mean, mat_r_mean, 
     cov_p_mean, mat_p_mean, 
     cov_r_median, mat_r_median, 
     cov_p_median, mat_p_median) = test(gen_samples_dict, threshold, cfg.train.num_workers, test_starting_point=False)
    print('=================== Results from baseline method + DiSCO ===================')
    print(f'cov_r_mean: {cov_r_mean}, mat_r_mean: {mat_r_mean}')
    print(f'cov_p_mean: {cov_p_mean}, mat_p_mean: {mat_p_mean}')
    print(f'cov_r_median: {cov_r_median}, mat_r_median: {mat_r_median}')
    print(f'cov_p_median: {cov_p_median}, mat_p_median: {mat_p_median}')
        
if __name__ == '__main__':
    main()
    

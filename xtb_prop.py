import pickle
import numpy as np
from tqdm import tqdm
from src.utils.chem import set_rdmol_positions
from src.utils.xtb import xtb_energy, xtb_optimize
import os
from rdkit import Chem
from rdkit.Chem import AllChem

os.environ['OMP_NUM_THREADS'] = '16'


def get_props(rdmol, confs, optimize=True, mmff=False):
    energies = []
    gaps = []
    dipoles = []

    for conf in confs:
        conf_mol = set_rdmol_positions(rdmol, conf)
        if mmff:
            AllChem.MMFFOptimizeMolecule(conf_mol)
        if optimize:
            status = xtb_optimize(conf_mol, level='crude')
            if status is None:
                continue
        res = xtb_energy(conf_mol, dipole=True)
        energy, gap = res['energy'], res['gap']
        dipole = res['dipole']
        energies.append(energy)
        gaps.append(gap)
        dipoles.append(dipole)
    
    mean_eng = np.mean(energies)
    min_eng = np.min(energies)
    mean_gap = np.mean(gaps)
    min_gap = np.min(gaps)
    max_gap = np.max(gaps)
    mean_dipole = np.mean(dipoles)

    return np.array([mean_eng, min_eng, mean_gap, min_gap, max_gap, mean_dipole])

def main(args):
    with open(args.file, 'rb') as f:
        samples = pickle.load(f)

    mine_diff_list = []
    base_diff_list = []

    for value in tqdm(samples.values(), desc='Computing energy for each molecule'):
        rdmol = value['rdmol']
        mine_gen_confs = value['gen_confs']
        starting_gen_confs = value['starting_point']

        ref_confs = value['ref_confs']

        ref_prop = get_props(rdmol, ref_confs, optimize=False, mmff=args.mmff)
        mine_prop = get_props(rdmol, mine_gen_confs, optimize=True, mmff=args.mmff)
        base_prop = get_props(rdmol, starting_gen_confs, optimize=True, mmff=args.mmff)

        mine_diff = np.abs(ref_prop - mine_prop)
        base_diff = np.abs(ref_prop - base_prop)

        print('Base diff: ', base_diff)
        print('DiSCO diff: ', mine_diff)
        
        mine_diff_list.append(mine_diff)
        base_diff_list.append(base_diff)

    mine_diff_list = np.array(mine_diff_list)
    mine_mean_diff = np.mean(mine_diff_list, axis=0)
    mine_median_diff = np.median(mine_diff_list, axis=0)

    base_diff_list = np.array(base_diff_list)
    base_mean_diff = np.mean(base_diff_list, axis=0)
    base_median_diff = np.median(base_diff_list, axis=0)

    print('Base ---------------------')
    print('Mean diff: ', base_mean_diff)
    print('Median diff: ', base_median_diff)

    print('DiSCO ---------------------')
    print('Mean diff: ', mine_mean_diff)
    print('Median diff: ', mine_median_diff)

    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='/root/conformer-bridge/outputs/230709-fromtorsion-qm9-7steps-final/gen_samples_prop.pkl')
    parser.add_argument('--mmff', action='store_true')
    args = parser.parse_args()
    main(args)
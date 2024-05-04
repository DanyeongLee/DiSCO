import pickle
import numpy as np
from tqdm import tqdm
from src.utils.chem import set_rdmol_positions
from src.utils.xtb import xtb_energy, xtb_optimize
import os

os.environ['OMP_NUM_THREADS'] = '16'


with open('outputs/230709-fromtorsion-drugs-7steps-final/gen_samples_traj.pkl', 'rb') as f:
    samples = pickle.load(f)


def get_energy_mean(rdmol, confs, optimize=False):
    energies = []
    for conf in confs:
        m = set_rdmol_positions(rdmol, conf)
        if optimize:
            optimize_success = xtb_optimize(m, level='crude')
            if optimize_success is None:
                print('Optimization failed.. skipping')
                continue
        energies.append(xtb_energy(m)['energy'])
    return np.mean(energies)


energy_per_mol = {}

smiles_list = list(samples.keys())


for smi in tqdm(smiles_list):
    try:
        value = samples[smi]

        rdmol = value['rdmol']
        traj = value['traj']
        ref_confs = value['ref_confs']
        gen_confs = value['gen_confs']
        starting_point = value['starting_point'][:30]

        energy_gap_list = []

        ref_energy = get_energy_mean(rdmol, ref_confs, optimize=False)
        base_energy = get_energy_mean(rdmol, starting_point, optimize=True)
        energy_gap = np.abs(base_energy - ref_energy)

        energy_gap_list.append(energy_gap)
        print(energy_gap)

        for i in range(len(traj)):
            confs = traj[i][:30]
            energy_mean = get_energy_mean(rdmol, confs, optimize=True)
            energy_gap = np.abs(energy_mean - ref_energy)
            energy_gap_list.append(energy_gap)
            print(energy_gap)
        
        energy_per_mol[smi] = energy_gap_list
    except:
        print('Error with {}'.format(smi))
        continue

with open('notebooks/traj_energy.pkl', 'wb') as f:
    pickle.dump(energy_per_mol, f)
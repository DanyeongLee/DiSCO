import pickle
from src.utils.chem import set_rdmol_positions
import numpy as np
from rdkit.Chem.rdMolTransforms import GetBondLength, GetAngleDeg

with open('/root/conformer-bridge/outputs/230709-fromtorsion-drugs-7steps-final/gen_samples_traj.pkl', 'rb') as f:
    samples = pickle.load(f)




def get_all_bond_lengths(mol):
    conf = mol.GetConformer()
    bond_lengths = []

    for i in mol.GetAtoms():
        for j in mol.GetAtoms():
            if i.GetIdx() < j.GetIdx():
                bond_lengths.append(GetBondLength(conf, i.GetIdx(), j.GetIdx()))

    return np.array(bond_lengths)


def get_all_bond_angles(mol):
    conf = mol.GetConformer()
    bond_angles = []

    for i in mol.GetAtoms():
        for j in mol.GetAtoms():
            for k in mol.GetAtoms():
                if i.GetIdx() < j.GetIdx() < k.GetIdx():
                    bond_angles.append(GetAngleDeg(conf, i.GetIdx(), j.GetIdx(), k.GetIdx()))

    return np.array(bond_angles)


from tqdm import tqdm
from rdkit import Chem

my_length_diff_list = []
starting_length_diff_list = []

my_angle_diff_list = []
starting_angle_diff_list = []

for smi, value in tqdm(samples.items()):
    rdmol = value['rdmol']
    ref_confs = value['ref_confs']
    gen_confs = value['gen_confs']
    starting_point = value['starting_point']

    ref_confs = ref_confs
    gen_confs = gen_confs
    starting_point = starting_point

    ref_mols = [set_rdmol_positions(rdmol, conf) for conf in ref_confs]
    gen_mols = [set_rdmol_positions(rdmol, conf) for conf in gen_confs]
    starting_mols = [set_rdmol_positions(rdmol, conf) for conf in starting_point]

    ref_lengths = []
    for ref_mol in ref_mols:
        #ref_mol = Chem.RemoveHs(ref_mol)
        ref_lengths.append(get_all_bond_lengths(ref_mol))
    ref_lengths = np.mean(ref_lengths, axis=0)

    gen_lengths = []
    for gen_mol in gen_mols:
        #gen_mol = Chem.RemoveHs(gen_mol)
        gen_lengths.append(get_all_bond_lengths(gen_mol))
    gen_lengths = np.mean(gen_lengths, axis=0)

    starting_lengths = []
    for starting_mol in starting_mols:
        #starting_mol = Chem.RemoveHs(starting_mol)
        starting_lengths.append(get_all_bond_lengths(starting_mol))
    starting_lengths = np.mean(starting_lengths, axis=0)

    ref_angles = []
    for ref_mol in ref_mols:
        #ref_mol = Chem.RemoveHs(ref_mol)
        ref_angles.append(get_all_bond_angles(ref_mol))
    ref_angles = np.mean(ref_angles, axis=0)

    gen_angles = []
    for gen_mol in gen_mols:
        #gen_mol = Chem.RemoveHs(gen_mol)
        gen_angles.append(get_all_bond_angles(gen_mol))
    gen_angles = np.mean(gen_angles, axis=0)

    starting_angles = []
    for starting_mol in starting_mols:
        #starting_mol = Chem.RemoveHs(starting_mol)
        starting_angles.append(get_all_bond_angles(starting_mol))
    starting_angles = np.mean(starting_angles, axis=0)


    my_length_diff = ref_lengths - gen_lengths
    starting_length_diff = ref_lengths - starting_lengths

    my_angle_diff = ref_angles - gen_angles
    starting_angle_diff = ref_angles - starting_angles

    my_length_diff_list.append(my_length_diff)
    starting_length_diff_list.append(starting_length_diff)
    my_angle_diff_list.append(my_angle_diff)
    starting_angle_diff_list.append(starting_angle_diff)


pkl_dict = {
    'my_length_diff_list': my_length_diff_list,
    'starting_length_diff_list': starting_length_diff_list,
    'my_angle_diff_list': my_angle_diff_list,
    'starting_angle_diff_list': starting_angle_diff_list
}

with open('notebooks/local_structure_error_H.pkl', 'wb') as f:
    pickle.dump(pkl_dict, f)
import copy
from tqdm import tqdm
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from .chem import get_best_rmsd, set_rdmol_positions
from .geometry import to_zero_com
from rdkit.Chem import AllChem, rdDistGeom



def dataset_to_each_mol(dataset):
    mols = {}
    positions = defaultdict(list)
    idx_set = set()
    print(len(dataset))

    for data in tqdm(dataset, desc='dataset_to_each_mol'):
        if data is None:
            continue
        idx = data.idx.item()
        data['rdmol'] = data['rdmol'][0]
        data['smiles'] = data['smiles'][0]
        if idx not in idx_set:
            idx_set.add(idx)
            mols[idx] = data
            if hasattr(data, 'pos_org'):
                positions[idx].append(data.pos_org)
            else:
                positions[idx].append(data.pos)
        else:
            if hasattr(data, 'pos_org'):
                positions[idx].append(data.pos_org)
            else:
                positions[idx].append(data.pos)

    return mols, positions


def get_pos_from_mol(mol, mmff=False, mmff_iters=1000):
    status = AllChem.EmbedMolecule(mol, rdDistGeom.ETKDGv3())
    if status == -1:
        print('Failed to generate conformers.')
        return None

    if mmff:
        AllChem.MMFFOptimizeMolecule(mol, mmffVariant='MMFF94s', maxIters=mmff_iters)

    conf = mol.GetConformer()
    return conf.GetPositions()


class InferenceDataset2(Dataset):
    def __init__(self, data, n_samples, torsion_mol_list, etkdg_scale=None, mmff=False, mmff_iters=1000):
        self.data = data
        self.n_samples = n_samples
        self.torsion_mol_list = torsion_mol_list
        self.etkdg_scale = etkdg_scale
        self.mmff = mmff
        self.mmff_iters = mmff_iters

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        data_ = self.data.clone()
        torsion_rdmol = copy.deepcopy(self.torsion_mol_list[idx])
        pos1 = torch.tensor(torsion_rdmol.GetConformer().GetPositions(), dtype=torch.float)
        pos1 = to_zero_com(pos1)
        data_.pos1 = pos1

        if self.mmff:
            AllChem.MMFFOptimizeMolecule(torsion_rdmol, mmffVariant='MMFF94s', maxIters=self.mmff_iters)
            pos1 = torch.tensor(torsion_rdmol.GetConformer().GetPositions(), dtype=torch.float)
            pos1 = to_zero_com(pos1)
            data_.pos1 = pos1
        
        return data_
    

class InferenceDataset(Dataset):
    def __init__(self, data, n_samples, mmff=False, mmff_iters=1000):
        self.data = data
        self.n_samples = n_samples
        self.mmff = mmff
        self.mmff_iters = mmff_iters

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        data_ = self.data.clone()
        try:
            new_pos1 = torch.tensor(
                get_pos_from_mol(data_.rdmol,
                                 mmff=self.mmff, mmff_iters=self.mmff_iters),
                                 dtype=torch.float)
        except:
            try:
                new_pos1 = torch.tensor(
                    get_pos_from_mol(data_.rdmol, etkdg_version=self.etkdg_version, 
                                    mmff=self.mmff, mmff_iters=self.mmff_iters),
                                    dtype=torch.float)
            except:
                try:
                    new_pos1 = torch.tensor(
                        get_pos_from_mol(data_.rdmol, etkdg_version=self.etkdg_version, 
                                        mmff=self.mmff, mmff_iters=self.mmff_iters),
                                        dtype=torch.float)
                except:
                    print('Failed to generate conformers.')
                    return None
        data_.pos1 = new_pos1
        return data_

def collate_fn(data_list):
    data_list = [data for data in data_list if data is not None]
    return Batch.from_data_list(data_list)


@torch.no_grad()
def gen_samples(model, diffusion, data, positions, 
                device, bsz=32,
                return_traj=False, num_workers=1,
                n_to_generate=None,
                from_torsion=False,  torsion_mol_list=None,
                mmff=False, mmff_iters=1000
                ):
    n_confs = len(positions)
    if n_to_generate is None:
        n_to_generate = n_confs * 2

    starting_pos_list = []

    if from_torsion:
        inference_dataset = InferenceDataset2(data, n_to_generate, torsion_mol_list, mmff, mmff_iters)
    else:
        inference_dataset = InferenceDataset(data, n_to_generate, mmff, mmff_iters)
    inference_loader = DataLoader(inference_dataset, batch_size=bsz, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    for batch in inference_loader:
        batch_size = batch.batch[-1].item() + 1
        starting_pos = batch.pos1.cpu().numpy()  # (N * bsz, 3)
        # to list of (N, 3)
        n_atoms = starting_pos.shape[0] // batch_size
        starting_pos = [starting_pos[i:i+n_atoms] for i in range(0, len(starting_pos), n_atoms)]
        starting_pos_list = starting_pos_list + starting_pos

        batch = batch.to(device)
        sample_traj = diffusion.sample(model, batch, return_traj=True)

        if return_traj:
            return sample_traj, starting_pos_list
        else:
            return sample_traj[-1], starting_pos_list


def calc_metrics(gen_positions, ref_positions, rdmol, threshold=1.25, rdmol_org=None):
    cover_count = 0
    mat_r = 0.

    for ref_pos in ref_positions:
        rmsd_list = []
        ref_pos = ref_pos
        if rdmol_org is not None:
            ref_mol = set_rdmol_positions(rdmol_org, ref_pos)
        else:
            ref_mol = set_rdmol_positions(rdmol, ref_pos)

        for gen_pos in gen_positions:
            gen_pos = gen_pos
            gen_mol = set_rdmol_positions(rdmol, gen_pos)
            rmsd = get_best_rmsd(gen_mol, ref_mol)
            rmsd_list.append(rmsd)

        mat_r += min(rmsd_list)
        if min(rmsd_list) < threshold:
            cover_count += 1
        
    cov_r = cover_count / len(ref_positions)
    mat_r = mat_r / len(ref_positions)

    return cov_r, mat_r

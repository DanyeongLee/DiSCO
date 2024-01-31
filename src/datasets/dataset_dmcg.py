import random
import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from src.utils.featurization import featurize_mol
from src.utils.chem import set_rdmol_positions
from src.utils.geometry import to_zero_com, align
from rdkit import Chem
from copy import deepcopy


class MyDataset(Dataset):
    def __init__(self,  data_path, dmcg_data_path, transform=None):
        with open(data_path, 'rb') as f:
            self.data_list = pickle.load(f)

        with open(dmcg_data_path, 'rb') as f:
            self.dmcg_data_dict = pickle.load(f)
        
        if 'H' in dmcg_data_path:
            self.removeHs = False
        else:
            self.removeHs = True

        self.transform = transform
        if 'drugs' in data_path:
            self.data_type = 'drugs'
        elif 'qm9' in data_path:
            self.data_type = 'qm9'

    def __len__(self):
        return len(self.data_list)
    
    def get_item_H(self, idx):
        data = self.data_list[idx].clone()
        try:
            canon_smi = Chem.MolToSmiles(data['rdmol'])
            dmcg_conf_list = self.dmcg_data_dict[canon_smi]['gen_confs']
        except:
            return None
        dmcg_conf = random.choice(dmcg_conf_list)
        rdmol = deepcopy(data.rdmol)
        pos = rdmol.GetConformer().GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)
        data.pos_org = pos
        pos = to_zero_com(pos)
        pos1 = set_rdmol_positions(rdmol, dmcg_conf).GetConformer().GetPositions()
        pos1 = torch.tensor(pos1, dtype=torch.float)
        if pos1.shape[0] != pos.shape[0]:
            return None
        pos1 = to_zero_com(pos1)

        aligned_pos = align(pos1, pos)
        data.pos = aligned_pos
        data.pos1 = pos1

        featured_data = featurize_mol(data.rdmol, types=self.data_type)
        data.x = featured_data.x
        data.edge_attr = featured_data.edge_attr
        data.edge_index = featured_data.edge_index

        data.atom_type = None
        data.edge_type = None

        if self.transform:
            data = self.transform(data)
        return data

    def get_item_noH(self, idx):
        data = self.data_list[idx].clone()
        try:
            canon_smi = Chem.MolToSmiles(Chem.RemoveHs(data['rdmol']))
            dmcg_conf_list = self.dmcg_data_dict[canon_smi]['gen_confs']
        except:
            return None
        dmcg_conf = random.choice(dmcg_conf_list)
        new_rdmol = deepcopy(data.rdmol)

        new_rdmol = Chem.RemoveHs(new_rdmol)
        data.rdmol_org = new_rdmol
        pos = new_rdmol.GetConformer().GetPositions()
        pos = torch.tensor(pos, dtype=torch.float)
        data.pos_org = pos
        pos = to_zero_com(pos)
        pos1 = set_rdmol_positions(new_rdmol, dmcg_conf).GetConformer().GetPositions()
        pos1 = torch.tensor(pos1, dtype=torch.float)
        if pos1.shape[0] != pos.shape[0]:
            return None
        pos1 = to_zero_com(pos1)

        aligned_pos = align(pos1, pos)
        data.pos = aligned_pos
        data.pos1 = pos1
        data.rdmol = deepcopy(new_rdmol)

        featured_data = featurize_mol(data.rdmol, types=self.data_type)
        data.x = featured_data.x
        data.edge_attr = featured_data.edge_attr
        data.edge_index = featured_data.edge_index

        data.atom_type = None
        data.edge_type = None

        if self.transform:
            data = self.transform(data)
        return data
    
    def __getitem__(self, idx):
        if self.removeHs:
            return self.get_item_noH(idx)
        else:
            return self.get_item_H(idx)


def collate_fn(data_list):
    data_list = [data for data in data_list if data is not None]
    return Batch.from_data_list(data_list)
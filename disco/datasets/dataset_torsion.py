import copy
import random
import pickle
import lmdb
from io import BytesIO

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch

from src.utils.featurization import featurize_mol
from src.utils.geometry import to_zero_com, align


class ConformerDataset(Dataset):
    def __init__(self,  data_path, torsion_data_path):
        with open(data_path, 'rb') as f:
            self.data_list = pickle.load(f)

        with open(torsion_data_path, 'rb') as f:
            self.torsion_data_dict = pickle.load(f)
        
        if 'drugs' in data_path:
            self.data_type = 'drugs'
        elif 'qm9' in data_path:
            self.data_type = 'qm9'

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx].clone()
        data.pos = to_zero_com(data.pos)
        featured_data = featurize_mol(data.rdmol, types=self.data_type)
        data.x = featured_data.x
        data.edge_attr = featured_data.edge_attr

        try:
            torsion_rdmol_list = self.torsion_data_dict[data.smiles]
        except:
            return None
        
        torsion_rdmol = copy.deepcopy(random.choice(torsion_rdmol_list))
        pos1 = torch.tensor(torsion_rdmol.GetConformer().GetPositions(), dtype=torch.float)

        if data.pos.shape != pos1.shape:
            return None

        pos1 = to_zero_com(pos1)
        data.pos1 = pos1
        
        data.pos_org = data.pos.clone()
        aligned_pos = align(data.pos1, data.pos)
        data.pos = aligned_pos

        data.atom_type = None
        data.edge_type = None

        return data
    

class LMDBDataset(Dataset):
    def __init__(self, data_path, torsion_data_path):
        self.env = lmdb.open(data_path, readonly=True, lock=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        
        with open(torsion_data_path, 'rb') as f:
            self.torsion_data_dict = pickle.load(f)

        if 'drugs' in data_path:
            self.data_type = 'drugs'
        elif 'qm9' in data_path:
            self.data_type = 'qm9'

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with self.env.begin(write=False) as txn:
            byte_data = txn.get(str(idx).encode())
            if byte_data is not None:
                data = self.deserialize_data(byte_data)
            else:
             return None
            
        data.pos = to_zero_com(data.pos)
        featured_data = featurize_mol(data.rdmol, types=self.data_type)
        data.x = featured_data.x
        data.edge_attr = featured_data.edge_attr
        try:
            torsion_rdmol_list = self.torsion_data_dict[data.smiles]
        except:
            return None
        
        torsion_rdmol = copy.deepcopy(random.choice(torsion_rdmol_list))
        pos1 = torch.tensor(torsion_rdmol.GetConformer().GetPositions(), dtype=torch.float)

        if data.pos.shape != pos1.shape:
            return None

        pos1 = to_zero_com(pos1)
        data.pos1 = pos1
        
        data.pos_org = data.pos.clone()
        aligned_pos = align(data.pos1, data.pos)
        data.pos = aligned_pos

        data.atom_type = None
        data.edge_type = None

        return data

    @staticmethod
    def deserialize_data(byte_data):
        buffer = BytesIO(byte_data)
        buffer.seek(0)
        data = torch.load(buffer)
        return data


def collate_fn(data_list):
    data_list = [data for data in data_list if data is not None]
    return Batch.from_data_list(data_list)

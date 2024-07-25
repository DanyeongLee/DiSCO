import lmdb
import pickle
from io import BytesIO
from rdkit.Chem import AllChem, rdDistGeom

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Batch
from disco.utils import featurize_mol, to_zero_com, align


def embed_mol(mol, mmff=False):
    status = AllChem.EmbedMolecule(mol, rdDistGeom.ETKDGv3())
    if mmff:
        AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    return conf.GetPositions()


class ConformerDataset(Dataset):
    def __init__(self, data_path, mmff=False):
        with open(data_path, 'rb') as f:
            self.data_list = pickle.load(f)
        self.mmff = mmff
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
            pos1 = torch.tensor(embed_mol(data.rdmol, self.mmff), dtype=torch.float)
            if pos1.shape[0] != data.pos.shape[0]:
                return None
            
            pos1 = to_zero_com(pos1)
            data.pos1 = pos1
        except:
            return None
        
        data.pos_org = data.pos.clone()
        aligned_pos = align(data.pos1, data.pos)
        data.pos = aligned_pos

        data.atom_type = None
        data.edge_type = None

        return data
    

class LMDBDataset(Dataset):
    def __init__(self, data_path, mmff=False):
        self.env = lmdb.open(data_path, readonly=True, lock=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()['entries']
        
        self.mmff = mmff
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
            pos1 = torch.tensor(embed_mol(data.rdmol, self.mmff), dtype=torch.float)
            if pos1.shape[0] != data.pos.shape[0]:
                return None
            
            pos1 = to_zero_com(pos1)
            data.pos1 = pos1
        except:
            return None
        
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

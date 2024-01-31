import pickle
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from src.utils.featurization import featurize_mol
from src.utils.geometry import to_zero_com, align
from rdkit import Chem
from rdkit.Chem import AllChem, rdDistGeom



def get_pos_from_mol(mol, etkdg_version=3):
    if etkdg_version == 1:
        status = AllChem.EmbedMolecule(mol, rdDistGeom.ETKDG())
    elif etkdg_version == 2:
        status = AllChem.EmbedMolecule(mol, rdDistGeom.ETKDGv2())
    elif etkdg_version == 3:
        status = AllChem.EmbedMolecule(mol, rdDistGeom.ETKDGv3())
    else:
        raise NotImplementedError
    conf = mol.GetConformer()
    return conf.GetPositions()


class MyDataset(Dataset):
    def __init__(self,  data_path, etkdg=False, transform=None):
        with open(data_path, 'rb') as f:
            self.data_list = pickle.load(f)
        self.etkdg = etkdg
        self.transform = transform
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
        if self.etkdg:
            print 
            try:
                pos1 = torch.tensor(get_pos_from_mol(data.rdmol, self.etkdg), dtype=torch.float)
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

        if self.transform:
            data = self.transform(data)
        return data


def collate_fn(data_list):
    data_list = [data for data in data_list if data is not None]
    return Batch.from_data_list(data_list)

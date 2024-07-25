from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from rdkit.Geometry import Point3D

from disco.utils import featurize_mol, qm9_types, drugs_types

class DiSCO(nn.Module):
    def __init__(self, diffusion, score_net, dataset='qm9'):
        super().__init__()
        self.diffusion = diffusion
        self.score_net = score_net
        self.dataset = dataset
    
    def get_device(self):
        # Method to get the device of the model
        return next(self.parameters()).device

    def make_batch(self, mol):
        data = featurize_mol(mol, qm9_types if self.dataset == 'qm9' else drugs_types)
        data_list = []
        for conf in mol.GetConformers():
            d = data.clone()
            pos = conf.GetPositions()
            d.pos1 = torch.Tensor(pos)
            data_list.append(d)
        return Batch.from_data_list(data_list)
    
    def replace_conformers(self, mol, new_confs):
        for i, conf in enumerate(mol.GetConformers()):
            for j in range(mol.GetNumAtoms()):
                x, y, z = new_confs[i][j]
                conf.SetAtomPosition(j, Point3D(float(x), float(y), float(z)))

    def forward(self, mol):
        mol = deepcopy(mol)
        batch = self.make_batch(mol)
        batch = batch.to(self.get_device())
        confs = self.diffusion.sample(self.score_net, batch)
        confs = np.array(confs)
        self.replace_conformers(mol, confs)
        return mol

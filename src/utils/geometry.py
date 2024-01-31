import torch
from torch_scatter import scatter_mean
from sklearn.metrics import pairwise_distances
from scipy.spatial.transform import Rotation as R


def unbatch(x, batch):
    return torch.split(x, batch.bincount().tolist())

def match_com(pos1, pos2):
    # match COM of pos2 to COM of pos1

    com1 = pos1.mean(dim=0)
    com2 = pos2.mean(dim=0)
    return pos2 + com1 - com2


def align(pos1, pos2):
    # align pos2 to pos1
    
    r = R.align_vectors(pos1.cpu().numpy(), pos2.cpu().numpy())[0]
    pos2 = r.apply(pos2.cpu().numpy())
    return torch.tensor(pos2, dtype=pos1.dtype, device=pos1.device)


def align_batch(batched_pos1, batched_pos2, batch):
    pos1_list = unbatch(batched_pos1, batch)
    pos2_list = unbatch(batched_pos2, batch)
    pos_list = []
    for pos1, pos2 in zip(pos1_list, pos2_list):
        pos_list.append(align(pos1, pos2))
    return torch.cat(pos_list, dim=0)


def to_zero_com(pos):
    mean = pos.mean(dim=0)
    return pos - mean


def batch_to_zero_com(pos, batch):
    com = scatter_mean(pos, batch, dim=0)
    return pos - com[batch]


def get_3d_neigbors(pos, cutoff=5):
    dist = pairwise_distances(pos.cpu().numpy())
    dist = torch.tensor(dist, dtype=pos.dtype, device=pos.device)
    edge_index = torch.nonzero((dist < cutoff) & (dist > 0), as_tuple=False).T.to(pos.device)
    distance = dist[edge_index[0], edge_index[1]]
    return edge_index, distance


def get_3d_neigbors_batch(pos, batch, cutoff=5):
    pos_list = unbatch(pos, batch)
    edge_index_list = []
    distance_list = []

    n_nodes = 0
    for pos in pos_list:
        edge_index, distance = get_3d_neigbors(pos, cutoff)
        edge_index += n_nodes
        edge_index_list.append(edge_index)
        distance_list.append(distance)
        n_nodes += pos.shape[0]
    edge_index = torch.cat(edge_index_list, dim=1)
    distance = torch.cat(distance_list, dim=0)
    return edge_index, distance
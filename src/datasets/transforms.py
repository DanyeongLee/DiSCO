import torch
from src.utils.geometry import to_zero_com, align


class AddNoise:
    def __init__(self, scale):
        self.scale = scale
    
    def __call__(self, data):
        pos1 = data.pos1.clone()
        pos1 += torch.randn_like(pos1) * self.scale
        pos1 = to_zero_com(pos1)
        data.pos1 = pos1
        aligned_pos = align(data.pos1, data.pos)
        data.pos = aligned_pos

        return data
    
    
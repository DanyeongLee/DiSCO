from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from disco.utils import unbatch, batch_to_zero_com, match_com, align


def extract(coef, time, n_dims):
    coef_t = coef[time]
    while coef_t.dim() < n_dims:
        coef_t = coef_t.unsqueeze(-1)
    return coef_t


def get_gaussian_params(sigma1, sigma2):
    denom = sigma1 ** 2 + sigma2 ** 2
    coef1 = sigma2 ** 2 / denom
    coef2 = sigma1 ** 2 / denom
    var = (sigma1 ** 2 * sigma2 ** 2) / denom

    return coef1, coef2, var


class I2SBDiffusion(nn.Module):
    def __init__(self, beta):
        super().__init__()
        self.n_timesteps = beta.shape[0]

        std_fwd = beta.cumsum(0).sqrt()
        std_bwd = beta.flip(0).cumsum(0).flip(0).sqrt()

        mu_x0, mu_x1, var = get_gaussian_params(std_bwd, std_fwd)
        std_sb = var.sqrt()

        self.register_buffer('beta', beta)
        self.register_buffer('std_fwd', std_fwd)
        self.register_buffer('std_bwd', std_bwd)
        self.register_buffer('std_sb', std_sb)
        self.register_buffer('mu_x0', mu_x0)
        self.register_buffer('mu_x1', mu_x1)

    def q_sample(self, x0, x1, batch, time, noise=None):
        mu_x0_t = extract(self.mu_x0, time, x0.dim())[batch]
        mu_x1_t = extract(self.mu_x1, time, x0.dim())[batch]
        std_sb_t = extract(self.std_sb, time, x0.dim())[batch]
        if noise is None:
            noise = torch.randn_like(x0)

        x_t = mu_x0_t * x0 + mu_x1_t * x1 + std_sb_t * noise

        x0_pos_list = unbatch(x0, batch)
        x_t_pos_list = unbatch(x_t, batch)

        aligned_x0_pos = []
        for x0_pos, x_t_pos in zip(x0_pos_list, x_t_pos_list):
            x0_pos = match_com(x_t_pos, x0_pos)
            x0_pos = align(x_t_pos, x0_pos)
            aligned_x0_pos.append(x0_pos)
        aligned_x0 = torch.cat(aligned_x0_pos, dim=0)

        std_fwd_t = extract(self.std_fwd, time, x0.dim())[batch]
        target = (x_t - aligned_x0) / std_fwd_t

        return x_t, target
    
    def training_loss(self, model, batch, bsz):
        time = torch.randint(0, self.n_timesteps, (batch.pos1.shape[0],), device=batch.pos.device)
        batch.time = time
        pos_t, target = self.q_sample(batch.pos, batch.pos1, batch.batch, time)
        batch.pos = pos_t
        pred = model(batch)
        loss = F.mse_loss(pred, target)
        loss *= batch.num_graphs / bsz
        return loss
    
    @torch.no_grad()
    def p_sample_step(self, model, batch):
        pred = model(batch)
        std_fwd_t = extract(self.std_fwd, batch.time, batch.pos.dim())
        x0_pred = batch.pos - std_fwd_t * pred

        std_fwd_t = extract(self.std_fwd, batch.time, batch.pos.dim())
        std_fwd_t_1 = extract(self.std_fwd, batch.time - 1, batch.pos.dim())
        std_delta = (std_fwd_t ** 2 - std_fwd_t_1 ** 2).sqrt()

        mu_x0, mu_xt, var = get_gaussian_params(std_fwd_t_1, std_delta)

        new_pos = mu_x0 * x0_pred + mu_xt * batch.pos

        if batch.time[0].item() > 1:
            new_pos = new_pos + var.sqrt() * torch.randn_like(new_pos)
        
        return new_pos
        
    @torch.no_grad()
    def sample(self, model, batch, return_traj=False):
        pos = batch.pos1
        device = batch.pos1.device

        if return_traj:
            traj = []
            pos = batch_to_zero_com(pos, batch.batch)

            for t in tqdm(reversed(range(1, self.n_timesteps))):    
                time = torch.ones(pos.shape[0], dtype=torch.long, device=device) * t
                batch.pos = pos
                batch.time = time
                pos = self.p_sample_step(model, batch)
                pos = batch_to_zero_com(pos, batch.batch)

                traj_pos = pos.clone()
                traj_pos = unbatch(traj_pos, batch.batch)
                traj_pos = [s.detach().cpu().numpy() for s in traj_pos]
                traj.append(traj_pos)

            return list(traj)
        else:
            for t in tqdm(reversed(range(1, self.n_timesteps))):
                pos = batch_to_zero_com(pos, batch.batch)
                time = torch.ones(pos.shape[0], dtype=torch.long, device=device) * t
                batch.pos = pos
                batch.time = time
                pos = self.p_sample_step(model, batch)

            pos = batch_to_zero_com(pos, batch.batch)
            pos = unbatch(pos.cpu(), batch.batch)

            return list(pos)
        
    

#!/usr/bin/env python3
"""
Hayat: Open-Then-Express SCM with delta perturbation heads.

Two-stage training:
  Stage 1 (PBS init): train encoder + SCM baseline on PBS cells only
  Stage 2 (delta): freeze baseline, train delta heads + perturbation embeddings only

Input:
  x:         [B, G] log1p-normalized expression
  gene_emb:  [G, D] scGPT gene embeddings
  pert_info: {'perturbation_id': [B], 'perturbation_type': [B], 'is_perturbed': [B]}

Output:
  mu1 / mu0, delta_mu, latents
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Tuple, Optional, Dict


def make_hyper(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class Hayat(nn.Module):
    def __init__(
        self,
        chrom_boundaries: List[Tuple[int, int]],
        gene_emb_dim: int = 512,
        d_u: int = 16,
        K: int = 64,
        d_c: int = 32,
        d_v: int = 16,
        R: int = 16,
        hyper_hidden: int = 64,
        d_p: int = 32,
        d_q: int = 16,
        d_type: int = 16,
        n_perturbation_types: int = 5,
        n_perturbations: int = 92,
    ):
        super().__init__()
        self.num_segments = len(chrom_boundaries)
        self.chrom_boundaries = chrom_boundaries

        num_genes = max(end for _, end in chrom_boundaries)
        seg_idx = torch.zeros(num_genes, dtype=torch.long)
        for i, (start, end) in enumerate(chrom_boundaries):
            seg_idx[start:end] = i
        self.register_buffer('_seg_idx', seg_idx)

        # RL annealing (gate: Gumbel temp; drive: Gaussian noise)
        self.register_buffer('gate_temperature', torch.tensor(1.0))
        self.register_buffer('drive_noise_scale', torch.tensor(0.1))

        # ── Gene-conditioned hypernetworks (SCM baseline) ──
        self.hyper_alpha  = make_hyper(gene_emb_dim, hyper_hidden, 1)
        self.hyper_Gamma  = make_hyper(gene_emb_dim, hyper_hidden, d_u)
        self.hyper_beta   = make_hyper(gene_emb_dim, hyper_hidden, 1)
        self.hyper_W      = make_hyper(gene_emb_dim, hyper_hidden, K)
        self.hyper_Lambda = make_hyper(gene_emb_dim, hyper_hidden, d_c)
        self.hyper_rho    = make_hyper(gene_emb_dim, hyper_hidden, d_v)

        for hyper in [self.hyper_alpha, self.hyper_Gamma, self.hyper_beta,
                      self.hyper_W, self.hyper_Lambda, self.hyper_rho]:
            hyper[-1].weight.data.mul_(0.01)
            hyper[-1].bias.data.zero_()
        self.hyper_alpha[-1].weight.data.zero_()

        # ── Encoder ──
        self.mlp_u = nn.Sequential(
            nn.Linear(3, d_u * 2), nn.GELU(), nn.Linear(d_u * 2, d_u),
        )
        self.mlp_v = nn.Sequential(
            nn.Linear(3, d_v * 2), nn.GELU(), nn.Linear(d_v * 2, d_v),
        )
        self.mlp_z = nn.Sequential(
            nn.Linear(self.num_segments, 128), nn.GELU(), nn.Linear(128, K),
        )
        self.mlp_c = nn.Sequential(
            nn.Linear(3, d_c * 2), nn.GELU(), nn.Linear(d_c * 2, d_c),
        )

        # ── Bilinear gene×cell ──
        self.A = nn.Linear(gene_emb_dim, R, bias=False)
        self.B = nn.Linear(K + d_c, R, bias=False)
        nn.init.normal_(self.A.weight, std=0.01)
        nn.init.normal_(self.B.weight, std=0.01)

        # ── Perturbation embedding ──
        self.pert_emb = nn.Embedding(n_perturbations + 1, d_p, padding_idx=0)
        nn.init.normal_(self.pert_emb.weight, std=0.01)
        self.type_emb = nn.Embedding(n_perturbation_types, d_type)
        nn.init.normal_(self.type_emb.weight, std=0.01)
        self.p_combine = nn.Linear(d_p + d_type, d_p)
        nn.init.normal_(self.p_combine.weight, std=0.01)
        nn.init.zeros_(self.p_combine.bias)

        # ── Delta heads (only these + pert_emb get gradient in stage 2) ──
        d_h = K + d_c + d_p
        self.d_q = d_q
        self.d_p = d_p
        self.mlp_q = nn.Sequential(
            nn.Linear(d_u + d_v + d_p, d_q * 2), nn.GELU(), nn.Linear(d_q * 2, d_q),
        )
        self.hyper_w_pi = make_hyper(gene_emb_dim, hyper_hidden, d_h)
        self.hyper_b_pi = make_hyper(gene_emb_dim, hyper_hidden, d_q)
        self.hyper_w_mu = make_hyper(gene_emb_dim, hyper_hidden, d_h)
        self.hyper_b_mu = make_hyper(gene_emb_dim, hyper_hidden, d_q)

        for hyper in [self.hyper_w_pi, self.hyper_b_pi, self.hyper_w_mu, self.hyper_b_mu]:
            hyper[-1].weight.data.mul_(0.01)
            hyper[-1].bias.data.zero_()

    # ── Parameter generation ──
    def _gene_params(self, gene_emb: Tensor) -> Dict[str, Tensor]:
        return {
            'alpha':   self.hyper_alpha(gene_emb).squeeze(-1),
            'Gamma':   self.hyper_Gamma(gene_emb),
            'beta':    self.hyper_beta(gene_emb).squeeze(-1),
            'W':       self.hyper_W(gene_emb),
            'Lambda':  self.hyper_Lambda(gene_emb),
            'rho':     self.hyper_rho(gene_emb),
        }

    def _delta_params(self, gene_emb: Tensor) -> Dict[str, Tensor]:
        return {
            'w_pi': self.hyper_w_pi(gene_emb),
            'b_pi': self.hyper_b_pi(gene_emb),
            'w_mu': self.hyper_w_mu(gene_emb),
            'b_mu': self.hyper_b_mu(gene_emb),
        }

    def _cached_params(self, gene_emb: Tensor):
        """Cache hypernetwork outputs — gene_emb is fixed, avoid recomputing.
        Only caches SCM baseline params (frozen in stage 2). Delta params
        need gradients so they are NOT cached."""
        if not hasattr(self, '_param_cache') or self._param_cache.get('_ptr') != id(gene_emb):
            self._param_cache = {
                '_ptr': id(gene_emb),
                'gene': {k: v.detach() for k, v in self._gene_params(gene_emb).items()},
                'A': self.A(gene_emb).detach(),
            }
        return self._param_cache

    def _build_pert_vector(self, pert_id, pert_type, gene_emb):
        """Build unified perturbation vector [B, d_p]."""
        B, device = pert_id.shape[0], pert_id.device
        pert_vec = torch.zeros(B, self.d_p, device=device)

        # Type 1 (cytokine): learned embedding
        if (pert_type == 1).any():
            mask = (pert_type == 1).unsqueeze(-1)
            ids = pert_id * (pert_type == 1).long()
            pert_vec = torch.where(mask, self.pert_emb(ids), pert_vec)

        # Type 2+ (CRISPR): not used in cytokine data, handled by perturbation_id
        for t in range(2, pert_type.max().item() + 1):
            if (pert_type == t).any():
                mask = (pert_type == t).unsqueeze(-1)
                ids = (pert_id * (pert_type == t).long()).clamp(0, gene_emb.shape[0] - 1)
                pert_vec = torch.where(mask, self.pert_emb(ids), pert_vec)

        # Type 0 (control): zeros (already initialized)

        t_emb = self.type_emb(pert_type)
        return self.p_combine(torch.cat([pert_vec, t_emb], dim=-1))

    def _segment_features(self, x_log: Tensor) -> Tensor:
        B, G = x_log.shape

        seg_idx = self._seg_idx.unsqueeze(0).expand(B, -1)
        w = torch.ones(B, G, device=x_log.device)

        count = torch.zeros(B, self.num_segments, device=x_log.device)
        count.scatter_add_(1, seg_idx, w).clamp_(1)

        mean = torch.zeros(B, self.num_segments, device=x_log.device)
        mean.scatter_add_(1, seg_idx, x_log * w)
        mean = mean / count

        mean_per_gene = mean.gather(1, seg_idx)
        var = torch.zeros(B, self.num_segments, device=x_log.device)
        var.scatter_add_(1, seg_idx, (x_log - mean_per_gene) ** 2 * w)
        var = var / count

        nz = torch.zeros(B, self.num_segments, device=x_log.device)
        nz.scatter_add_(1, seg_idx, (x_log > 1e-6).float() * w)
        nz = nz / count

        return torch.stack([mean, var, nz], dim=-1)

    def _gumbel_sigmoid(self, logits: Tensor, tau: float) -> Tensor:
        """Gumbel-Sigmoid: discrete binary gate with differentiable relaxation."""
        u = torch.rand_like(logits)
        g = -torch.log(-torch.log(u.clamp(1e-10, 1 - 1e-10)))
        return torch.sigmoid((logits + g) / tau)

    def baseline_forward(self, x, gene_emb, return_latent=False):
        """
        Pure SCM baseline (no delta). Used during PBS pre-training.
        Returns mu0 = ℓ · o · r
        """
        B, G = x.shape
        x_log = torch.log1p(x.clamp(1e-8))
        gene_emb = gene_emb.to(device=x.device)

        seg_feats = self._segment_features(x_log)
        u = self.mlp_u(seg_feats)
        v = self.mlp_v(seg_feats)
        z = self.mlp_z(seg_feats[:, :, 0])
        c = self.mlp_c(torch.stack([
            x_log.mean(dim=-1), x_log.std(dim=-1),
            (x < 1e-6).float().mean(dim=-1),
        ], dim=-1))

        params = self._gene_params(gene_emb)
        u_g = u[:, self._seg_idx, :]
        pi = params['alpha'] + (params['Gamma'] * u_g).sum(dim=-1)
        o = torch.sigmoid(pi)

        v_g = v[:, self._seg_idx, :]
        local_drive = (params['rho'] * v_g).sum(dim=-1)
        h_cell = torch.cat([z, c], dim=-1)
        a_g = self.A(gene_emb)
        b_cell = self.B(h_cell)
        bilinear = (a_g.unsqueeze(0) * b_cell.unsqueeze(1)).sum(dim=-1)
        r_raw = F.softplus(params['beta'] + (z @ params['W'].T) + (c @ params['Lambda'].T)
                           + local_drive + bilinear)
        # Cap drive to enforce gate participation
        r = r_raw.clamp(max=10.0)

        ell = x.sum(dim=-1, keepdim=True)
        ell = ell / ell.mean().clamp(1e-8)
        eps_open = 0.01
        mu0 = ell * (eps_open + (1 - eps_open) * o) * r

        if return_latent:
            gamma_u = (params['Gamma'] * u_g).sum(dim=-1)
            return mu0, {
                'u': u, 'v': v, 'z': z, 'c': c,
                'pi': pi, 'o': o, 'r': r,
                'alpha': params['alpha'], 'Gamma': params['Gamma'],
                'beta': params['beta'], 'W': params['W'],
                'Lambda': params['Lambda'], 'rho': params['rho'],
                'gamma_u': gamma_u, 'a_g': a_g, 'b_cell': b_cell,
            }
        return mu0

    def forward(self, x, gene_emb, pert_info, return_latent=False, rl_training=False):
        """
        Full forward (baseline + delta). Used in stage 2.
        Returns mu1, delta_mu, latents.
        Baseline params are frozen — only delta heads + pert_emb train.
        """
        B, G = x.shape
        x_log = torch.log1p(x.clamp(1e-8))
        gene_emb = gene_emb.to(device=x.device)
        device = x.device

        # ── Encoder (frozen after stage 1) ──
        seg_feats = self._segment_features(x_log)
        u = self.mlp_u(seg_feats)
        v = self.mlp_v(seg_feats)
        z = self.mlp_z(seg_feats[:, :, 0])
        c = self.mlp_c(torch.stack([
            x_log.mean(dim=-1), x_log.std(dim=-1),
            (x < 1e-6).float().mean(dim=-1),
        ], dim=-1))

        # ── SCM baseline (frozen after stage 1) ──
        cache = self._cached_params(gene_emb)
        params = cache['gene']
        u_g = u[:, self._seg_idx, :]
        pi = params['alpha'] + (params['Gamma'] * u_g).sum(dim=-1)
        o = torch.sigmoid(pi)

        v_g = v[:, self._seg_idx, :]
        local_drive = (params['rho'] * v_g).sum(dim=-1)
        h_cell = torch.cat([z, c], dim=-1)
        a_g = cache['A']
        b_cell = self.B(h_cell)
        bilinear = (a_g.unsqueeze(0) * b_cell.unsqueeze(1)).sum(dim=-1)
        r_raw = F.softplus(params['beta'] + (z @ params['W'].T) + (c @ params['Lambda'].T)
                           + local_drive + bilinear)
        # Cap drive to enforce gate participation
        r = r_raw.clamp(max=10.0)

        ell = x.sum(dim=-1, keepdim=True)
        ell = ell / ell.mean().clamp(1e-8)
        eps_open = 0.01
        mu0 = ell * (eps_open + (1 - eps_open) * o) * r

        # ── Perturbation vector (trainable in stage 2) ──
        pert_id = pert_info.get('perturbation_id', torch.zeros(B, dtype=torch.long, device=device))
        pert_type = pert_info.get('perturbation_type', torch.zeros(B, dtype=torch.long, device=device))
        p = self._build_pert_vector(pert_id, pert_type, gene_emb)

        # ── Delta heads (trainable in stage 2) ──
        delta_params = self._delta_params(gene_emb)
        h = torch.cat([z, c, p], dim=-1)
        p_exp = p.unsqueeze(1).expand(-1, self.num_segments, -1)
        q = self.mlp_q(torch.cat([u, v, p_exp], dim=-1))
        q_g = q[:, self._seg_idx, :]

        mu_delta_pi = (h @ delta_params['w_pi'].T) + (q_g * delta_params['b_pi']).sum(dim=-1)
        mu_delta_mu = (h @ delta_params['w_mu'].T) + (q_g * delta_params['b_mu']).sum(dim=-1)

        # ── RL-sampled perturbation (discrete gate + continuous drive) ──
        if rl_training and self.training:
            tau = self.gate_temperature.item()
            pi1 = pi + mu_delta_pi
            o1_raw = self._gumbel_sigmoid(pi1, tau)
            sigma = self.drive_noise_scale.item()
            delta_mu = mu_delta_mu + sigma * torch.randn_like(mu_delta_mu)
            # Straight-Through: hard gate forward, soft gradient backward
            o1 = (o1_raw.detach() > 0.5).float() + o1_raw - o1_raw.detach()
        else:
            pi1 = pi + mu_delta_pi
            o1 = torch.sigmoid(pi1)
            delta_mu = mu_delta_mu

        r1 = (r * torch.exp(delta_mu)).clamp(max=10.0)
        mu1 = ell * (eps_open + (1 - eps_open) * o1) * r1

        if return_latent:
            gamma_u = (params['Gamma'] * u_g).sum(dim=-1)
            delta_pi = pi1 - pi  # actual delta after gumbel/gaussian sampling
            return mu1, delta_mu, {
                'z': z, 'c': c, 'u': u, 'v': v,
                'pi': pi, 'o': o, 'r': r, 'mu0': mu0,
                'gamma_u': gamma_u,
                'p': p, 'pert_id': pert_id, 'pert_type': pert_type,
                'delta_pi': delta_pi, 'delta_mu': delta_mu,
                'mu_delta_pi': mu_delta_pi, 'mu_delta_mu': mu_delta_mu,
                'pi1': pi1, 'o1': o1, 'r1': r1,
            }
        return mu1, delta_mu

    # ── Freeze/unfreeze for two-stage training ──
    def freeze_baseline(self):
        """Freeze encoder + SCM hypernets + A/B. Keep delta + pert_emb trainable."""
        delta_params = set()
        for name, p in self.named_parameters():
            if any(k in name for k in ['hyper_w_', 'hyper_b_', 'mlp_q.', 'pert_emb', 'type_emb', 'p_combine']):
                delta_params.add(name)
                p.requires_grad = True
            else:
                p.requires_grad = False
        print(f"[Freeze] Frozen all baseline params. Trainable only: delta heads + perturbation embeddings")

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        print(f"[Unfreeze] All params trainable")

    def set_gate_temp(self, tau: float, drive_noise: float = None):
        self.gate_temperature.fill_(max(0.1, tau))
        if drive_noise is not None:
            self.drive_noise_scale.fill_(max(0.0, drive_noise))

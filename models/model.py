#!/usr/bin/env python3
"""
Hayat: Open-Then-Express SCM (gene-set agnostic)

Encoder:
  x → segment features (mean, var, nz) → û   (cis openness, 3→d_u)
  x → segment pool → z                        (trans programs, K dims)
  cell_emb → c                                (cell state, d_c dims)

Decoder (gene-conditioned via hypernetworks):
  π = s_α·α_g + Γ_g·û_s(g)                   gate (α initially suppressed)
  r = softplus(β_g + W_g·z + Λ_g·c)          drive
  μ = ℓ · (ε+(1-ε)·o) · r                    hurdle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import List, Tuple, Optional, Dict


def make_hyper(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim))


class PerturbEmbedding(nn.Module):
    """p = E_cyto[cytokine] + E_dose[dose] + E_time[time]"""
    def __init__(self, n_cytokines: int, n_doses: int, n_times: int, d_p: int = 32):
        super().__init__()
        self.d_p = d_p
        self.cyto_emb = nn.Embedding(n_cytokines + 1, d_p, padding_idx=0)
        self.dose_emb = nn.Embedding(n_doses + 1, d_p, padding_idx=0)
        self.time_emb = nn.Embedding(n_times + 1, d_p, padding_idx=0)
        for emb in [self.cyto_emb, self.dose_emb, self.time_emb]:
            nn.init.normal_(emb.weight, std=0.01)

    def forward(self, pert_info: Dict[str, Tensor]) -> Tensor:
        B = pert_info[list(pert_info.keys())[0]].shape[0]
        device = pert_info[list(pert_info.keys())[0]].device
        p = torch.zeros(B, self.d_p, device=device)
        for key, emb in [('cytokine', self.cyto_emb), ('dose', self.dose_emb), ('time', self.time_emb)]:
            if key in pert_info:
                p = p + emb(pert_info[key])
        return p


class Hayat(nn.Module):
    def __init__(
        self,
        chrom_boundaries: List[Tuple[int, int]],
        gene_emb_dim: int = 512,
        cell_emb_dim: int = 0,
        d_u: int = 16,
        K: int = 64,
        d_c: int = 32,
        d_v: int = 16,
        R: int = 16,
        hyper_hidden: int = 64,
        d_p: int = 32,
        d_q: int = 16,
        n_cytokines: int = 100,
        n_doses: int = 10,
        n_times: int = 10,
    ):
        super().__init__()
        self.num_segments = len(chrom_boundaries)
        self.d_u = d_u
        self.K = K
        self.d_c = d_c
        self.d_v = d_v
        self.R = R
        self.gene_emb_dim = gene_emb_dim
        self.chrom_boundaries = chrom_boundaries

        # Segment index lookup [G] — for vectorized segment operations
        seg_idx = torch.zeros(0, dtype=torch.long)
        self.register_buffer('_seg_idx', seg_idx)

        # Annealing
        self.register_buffer('gate_tau', torch.tensor(1.0))
        self.register_buffer('epsilon', torch.tensor(0.5))
        self.register_buffer('alpha_scale', torch.tensor(0.0))

        # ── Gene-conditioned hypernetworks ──
        self.hyper_alpha  = make_hyper(gene_emb_dim, hyper_hidden, 1)    # → α_g
        self.hyper_Gamma  = make_hyper(gene_emb_dim, hyper_hidden, d_u)  # → Γ_g
        self.hyper_beta   = make_hyper(gene_emb_dim, hyper_hidden, 1)    # → β_g
        self.hyper_W      = make_hyper(gene_emb_dim, hyper_hidden, K)    # → W_g
        self.hyper_Lambda = make_hyper(gene_emb_dim, hyper_hidden, d_c)  # → Λ_g
        self.hyper_rho    = make_hyper(gene_emb_dim, hyper_hidden, d_v)  # → ρ_g (local drive)
        self.hyper_theta  = make_hyper(gene_emb_dim, hyper_hidden, 1)    # → log θ_g

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

        # ── Bilinear gene×cell interaction ──
        self.A = nn.Linear(gene_emb_dim, R, bias=False)    # gene embedding → R
        self.B = nn.Linear(K + d_c, R, bias=False)          # cell state [z;c] → R
        self.A.weight.data.mul_(0.01)
        self.B.weight.data.mul_(0.01)

        if cell_emb_dim > 0:
            self.cell_emb_proj = nn.Linear(cell_emb_dim, d_c)
            self.mlp_c_fallback = None
        else:
            self.cell_emb_proj = None
            self.mlp_c_fallback = nn.Sequential(
                nn.Linear(3, d_c), nn.GELU(), nn.Linear(d_c, d_c),
            )

        # ── Perturbation ──
        self.d_p = d_p
        self.d_q = d_q
        self.d_h = K + d_c + d_p
        self.pert_embed = PerturbEmbedding(n_cytokines, n_doses, n_times, d_p)
        self.mlp_q = nn.Sequential(
            nn.Linear(d_u + d_v + d_p, d_q * 2), nn.GELU(), nn.Linear(d_q * 2, d_q),
        )
        self.hyper_w_pi = make_hyper(gene_emb_dim, hyper_hidden, self.d_h)
        self.hyper_b_pi = make_hyper(gene_emb_dim, hyper_hidden, d_q)
        self.hyper_w_mu = make_hyper(gene_emb_dim, hyper_hidden, self.d_h)
        self.hyper_b_mu = make_hyper(gene_emb_dim, hyper_hidden, d_q)
        for hyper in [self.hyper_w_pi, self.hyper_b_pi, self.hyper_w_mu, self.hyper_b_mu]:
            hyper[-1].weight.data.mul_(0.01)
            hyper[-1].bias.data.zero_()

        self._param_cache = {}

    # ── Parameter generation ──
    def _gene_params(self, gene_emb: Tensor) -> Dict[str, Tensor]:
        return {
            'alpha':  self.hyper_alpha(gene_emb).squeeze(-1),
            'Gamma':  self.hyper_Gamma(gene_emb),
            'beta':   self.hyper_beta(gene_emb).squeeze(-1),
            'W':      self.hyper_W(gene_emb),
            'Lambda': self.hyper_Lambda(gene_emb),
            'rho':    self.hyper_rho(gene_emb),
            'log_theta': self.hyper_theta(gene_emb).squeeze(-1),
        }

    def _delta_params(self, gene_emb: Tensor) -> Dict[str, Tensor]:
        return {
            'w_pi': self.hyper_w_pi(gene_emb),
            'b_pi': self.hyper_b_pi(gene_emb),
            'w_mu': self.hyper_w_mu(gene_emb),
            'b_mu': self.hyper_b_mu(gene_emb),
        }

    def _segment_features(self, x_log: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Per-segment [mean, variance, nonzero_rate] → [B, S, 3].
        Vectorized via scatter_add — no Python loop over segments."""
        B, G = x_log.shape

        # Lazy init segment index
        if self._seg_idx.shape[0] != G:
            seg_idx = torch.zeros(G, dtype=torch.long)
            for i, (start, end) in enumerate(self.chrom_boundaries):
                seg_idx[start:end] = i
            self._seg_idx = seg_idx.to(x_log.device)

        seg_idx = self._seg_idx.unsqueeze(0).expand(B, -1)  # [B, G]
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            w = mask
        else:
            w = torch.ones(B, G, device=x_log.device)

        # Count per segment
        count = torch.zeros(B, self.num_segments, device=x_log.device)
        count.scatter_add_(1, seg_idx, w)
        count = count.clamp(1)

        # Mean
        mean = torch.zeros(B, self.num_segments, device=x_log.device)
        mean.scatter_add_(1, seg_idx, x_log * w)
        mean = mean / count

        # Variance = E[x²] - E[x]²
        mean_per_gene = mean.gather(1, seg_idx)  # [B, G]
        sq_diff = (x_log - mean_per_gene) ** 2
        var = torch.zeros(B, self.num_segments, device=x_log.device)
        var.scatter_add_(1, seg_idx, sq_diff * w)
        var = var / count

        # Nonzero rate
        nz = torch.zeros(B, self.num_segments, device=x_log.device)
        nz.scatter_add_(1, seg_idx, (x_log > 1e-6).float() * w)
        nz = nz / count

        return torch.stack([mean, var, nz], dim=-1)  # [B, S, 3]

    def _gumbel_sigmoid(self, logits: Tensor, tau: float) -> Tensor:
        u = torch.rand_like(logits)
        g = -torch.log(-torch.log(u.clamp(1e-10, 1 - 1e-10)))
        return torch.sigmoid((logits + g) / tau)

    # ── Forward ──
    def forward(
        self,
        x: Tensor,
        gene_emb: Tensor,
        cell_emb: Optional[Tensor] = None,
        observed_mask: Optional[Tensor] = None,
        gene_mask: Optional[Tensor] = None,
        pert_info: Optional[Dict[str, Tensor]] = None,
        return_latent: bool = False,
        ablation: str = "full",
    ) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        """
        Args:
            x:             [B, G] expression counts
            gene_emb:      [G, D] gene embeddings
            cell_emb:      [B, D_cell] cell embeddings (optional)
            observed_mask: [B, G] 1=measured, 0=missing
            gene_mask:     [B, G] 1=visible to encoder, 0=masked (prediction target)
            pert_info:     {'cytokine': [B], 'dose': [B], 'time': [B]} (optional)
            ablation:      "full" / "no_gate" / "no_drive" / "additive"
        """
        B, G = x.shape
        x_log = torch.log1p(x)

        gene_emb = gene_emb.to(device=x.device)
        params = self._gene_params(gene_emb)
        alpha, Gamma = params['alpha'], params['Gamma']
        beta, W = params['beta'], params['W']
        Lambda = params['Lambda']
        log_theta = params['log_theta']

        # ── Encoder (masked genes excluded from segment pooling) ──
        encoder_mask = observed_mask
        if gene_mask is not None:
            encoder_mask = gene_mask if observed_mask is None else observed_mask * gene_mask

        seg_feats = self._segment_features(x_log, encoder_mask)  # [B, S, 3]
        u = self.mlp_u(seg_feats)                                   # [B, S, d_u]
        v = self.mlp_v(seg_feats)                                   # [B, S, d_v]  local drive state
        z = self.mlp_z(seg_feats[:, :, 0])                          # [B, K]

        if self.cell_emb_proj is not None and cell_emb is not None:
            c = self.cell_emb_proj(cell_emb)
        elif self.mlp_c_fallback is not None:
            c = self.mlp_c_fallback(torch.stack([
                x_log.mean(dim=-1), x_log.std(dim=-1), (x == 0).float().mean(dim=-1),
            ], dim=-1))
        else:
            c = torch.zeros(B, self.d_c, device=x.device)

        # ── Gate (unchanged) ──
        u_g = u[:, self._seg_idx, :]                                # [B, G, d_u]
        as_ = self.alpha_scale.item()
        pi = as_ * alpha + (Gamma * u_g).sum(dim=-1)                # [B, G]

        if self.training:
            o_raw = self._gumbel_sigmoid(pi, self.gate_tau.item())
        else:
            o_raw = torch.sigmoid(pi)

        # ── Drive (with local segment path + bilinear gene×cell) ──
        rho = params['rho']                                          # [G, d_v]
        v_g = v[:, self._seg_idx, :]                                 # [B, G, d_v]
        local_drive = (rho * v_g).sum(dim=-1)                        # [B, G]

        # Bilinear gene×cell: (A·e_g) · (B·[z;c])
        h_cell = torch.cat([z, c], dim=-1)                           # [B, K+d_c]
        a_g = self.A(gene_emb)                                       # [G, R]
        b_cell = self.B(h_cell)                                       # [B, R]
        bilinear = (a_g.unsqueeze(0) * b_cell.unsqueeze(1)).sum(dim=-1)  # [B, G]

        r = F.softplus(
            beta + (z @ W.T) + (c @ Lambda.T) + local_drive + bilinear
        )  # [B, G]

        # ── Hurdle fusion (for unmasked genes, used in reconstruction) ──
        ell = x.sum(dim=-1, keepdim=True)
        ell = ell / ell.mean().clamp(1e-8)
        eps = self.epsilon.item() if self.training else 0.01
        o_eff = eps + (1 - eps) * o_raw

        if ablation == "no_gate":
            mu = ell * r
        elif ablation == "no_drive":
            mu = ell * o_eff
        elif ablation == "additive":
            mu = ell * (o_eff + r)
        else:
            mu = ell * o_eff * r

        theta = F.softplus(log_theta)

        # ── Delta (perturbation) ──
        delta_pi, delta_mu, pi_pert, mu_pert, o_pert = None, None, None, None, None
        if pert_info is not None:
            delta_params = self._delta_params(gene_emb)
            p = self.pert_embed(pert_info)                                # [B, d_p]
            h = torch.cat([z, c, p], dim=-1)                              # [B, d_h]

            # Segment response to perturbation
            p_exp = p.unsqueeze(1).expand(-1, self.num_segments, -1)     # [B, S, d_p]
            q = self.mlp_q(torch.cat([u, v, p_exp], dim=-1))              # [B, S, d_q]
            q_g = q[:, self._seg_idx, :]                                  # [B, G, d_q]

            w_pi, b_pi = delta_params['w_pi'], delta_params['b_pi']
            w_mu, b_mu = delta_params['w_mu'], delta_params['b_mu']

            delta_pi = (h @ w_pi.T) + (q_g * b_pi).sum(-1)               # [B, G]
            delta_mu = (h @ w_mu.T) + (q_g * b_mu).sum(-1)               # [B, G]

            pi_pert = pi + delta_pi                                       # [B, G]
            if self.training:
                o_pert = torch.sigmoid(pi_pert)  # no gumbel on delta head initially
            else:
                o_pert = torch.sigmoid(pi_pert)
            mu_pert = mu * torch.exp(delta_mu)                            # [B, G] log-space shift

        if return_latent:
            gamma_u = (Gamma * u_g).sum(dim=-1)
            return mu, theta, {
                'u': u, 'z': z, 'c': c,
                'o_raw': o_raw, 'o_eff': o_eff, 'r': r, 'pi': pi,
                'W': W, 'alpha': alpha, 'Gamma': Gamma, 'beta': beta, 'Lambda': Lambda,
                'gamma_u': gamma_u,
                'p': p if pert_info is not None else None,
                'delta_pi': delta_pi, 'delta_mu': delta_mu,
                'pi_pert': pi_pert, 'mu_pert': mu_pert, 'o_pert': o_pert,
            }
        return mu, theta

    # ── Annealing ──
    def set_annealing(self, gate_tau: float, epsilon: float, alpha_scale: float = None):
        self.gate_tau.fill_(gate_tau)
        self.epsilon.fill_(epsilon)
        if alpha_scale is not None:
            self.alpha_scale.fill_(alpha_scale)

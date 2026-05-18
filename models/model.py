#!/usr/bin/env python3
"""
Hayat: Open-Then-Express Structural Causal Model (gene-set agnostic)

Encoder:
  x, mask → segment pool → u     (cis openness)
  x, mask → segment pool → z     (trans programs)
  x → stats ⊕ cell_emb → c      (global state)

Decoder (gene-conditioned):
  e_g → hypernetwork → α_g, β_g, Γ_g, W_g, Λ_g, θ_g    per-gene params
  o = GumbelSigmoid(α + Γ·u)                             gate
  r = softplus(β + W·z + Λ·c)                            drive
  μ = ℓ · (ε+(1-ε)·o) · r                                hurdle
  x̂ ~ NB(μ, θ)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from typing import List, Tuple, Optional, Dict


def make_hyper(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.GELU(),
        nn.Linear(hidden, out_dim),
    )


class Hayat(nn.Module):
    def __init__(
        self,
        chrom_boundaries: List[Tuple[int, int]],
        gene_emb_dim: int = 512,
        cell_emb_dim: int = 0,
        d_u: int = 8,
        K: int = 16,
        d_c: int = 16,
        hyper_hidden: int = 64,
    ):
        super().__init__()
        self.num_segments = len(chrom_boundaries)
        self.d_u = d_u
        self.K = K
        self.d_c = d_c
        self.gene_emb_dim = gene_emb_dim
        self.chrom_boundaries = chrom_boundaries

        # Annealing
        self.register_buffer('gate_tau', torch.tensor(1.0))
        self.register_buffer('epsilon', torch.tensor(0.5))

        # ── Gene-conditioned parameter generators (hypernetworks) ──
        # These replace per-gene Parameter tables.
        # Given gene embedding e_g [D], produce per-gene scalar/vector params.
        self.hyper_alpha = make_hyper(gene_emb_dim, hyper_hidden, 1)       # → α_g
        self.hyper_Gamma = make_hyper(gene_emb_dim, hyper_hidden, d_u)     # → Γ_g
        self.hyper_beta  = make_hyper(gene_emb_dim, hyper_hidden, 1)       # → β_g
        self.hyper_W     = make_hyper(gene_emb_dim, hyper_hidden, K)       # → W_g
        self.hyper_Lambda = make_hyper(gene_emb_dim, hyper_hidden, d_c)    # → Λ_g
        self.hyper_theta = make_hyper(gene_emb_dim, hyper_hidden, 1)       # → log θ_g

        # Small initial output
        for hyper in [self.hyper_alpha, self.hyper_Gamma, self.hyper_beta,
                      self.hyper_W, self.hyper_Lambda]:
            hyper[-1].weight.data.mul_(0.01)
            hyper[-1].bias.data.zero_()

        # ── Encoder ──
        self.mlp_u = nn.Sequential(
            nn.Linear(1, d_u * 2), nn.GELU(), nn.Linear(d_u * 2, d_u),
        )
        self.mlp_z = nn.Sequential(
            nn.Linear(self.num_segments, 128), nn.GELU(), nn.Linear(128, K),
        )

        # ── Cell state c ──
        # Primary: scGPT cell embedding → d_c  (if available)
        # Fallback: expression stats → d_c  (mean, std, zero_frac)
        if cell_emb_dim > 0:
            self.cell_emb_proj = nn.Linear(cell_emb_dim, d_c)
            self.mlp_c_fallback = None
        else:
            self.cell_emb_proj = None
            self.mlp_c_fallback = nn.Sequential(
                nn.Linear(3, d_c), nn.GELU(), nn.Linear(d_c, d_c),
            )

        # Cache: gene_set_key → generated params
        self._param_cache = {}

    # ── Parameter generation ──

    def _gene_params(self, gene_emb: Tensor) -> Dict[str, Tensor]:
        """Generate per-gene parameters from gene embeddings. [G, D] → each [G, *]"""
        return {
            'alpha':  self.hyper_alpha(gene_emb).squeeze(-1),    # [G]
            'Gamma':  self.hyper_Gamma(gene_emb),                # [G, d_u]
            'beta':   self.hyper_beta(gene_emb).squeeze(-1),     # [G]
            'W':      self.hyper_W(gene_emb),                    # [G, K]
            'Lambda': self.hyper_Lambda(gene_emb),               # [G, d_c]
            'log_theta': self.hyper_theta(gene_emb).squeeze(-1), # [G]
        }

    def get_params(self, gene_emb: Tensor, cache_key=None) -> Dict[str, Tensor]:
        """Cached parameter generation."""
        if cache_key is not None and cache_key in self._param_cache:
            return self._param_cache[cache_key]
        params = self._gene_params(gene_emb)
        if cache_key is not None:
            self._param_cache[cache_key] = {k: v.detach() for k, v in params.items()}
        return params

    # ── Encoder helpers ──

    def _segment_pool(self, x_log: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Masked mean log1p expression per segment → [B, S].
        mask: [B, G] or [G], 1=observed, 0=unobserved.
        """
        B = x_log.shape[0]
        x_agg = torch.zeros(B, self.num_segments, device=x_log.device)
        if mask is None:
            for i, (start, end) in enumerate(self.chrom_boundaries):
                x_agg[:, i] = x_log[:, start:end].mean(dim=1)
        else:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)  # [G] → [1, G]
            for i, (start, end) in enumerate(self.chrom_boundaries):
                seg_x = x_log[:, start:end]
                seg_m = mask[:, start:end]
                denom = seg_m.sum(dim=1).clamp(1)
                x_agg[:, i] = (seg_x * seg_m).sum(dim=1) / denom
        return x_agg

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
        return_latent: bool = False,
        ablation: str = "full",
    ) -> Tuple[Tensor, Tensor, Optional[Dict[str, Tensor]]]:
        """
        Args:
            x:             [B, G] expression counts
            gene_emb:      [G, D] scGPT gene embeddings (required)
            cell_emb:      [B, D_cell] pre-computed cell embeddings (optional)
            observed_mask: [B, G] or [G] 1=observed, 0=missing (optional)
            ablation:      "full" / "no_gate" / "no_drive" / "additive"
        Returns:
            mu:    [B, G]
            theta: [G] or [1]
            latents: dict (if return_latent)
        """
        B, G = x.shape
        x_log = torch.log1p(x)

        # ── Gene-conditioned parameters ──
        gene_emb = gene_emb.to(device=x.device)
        params = self._gene_params(gene_emb)
        alpha, Gamma = params['alpha'], params['Gamma']      # [G], [G, d_u]
        beta, W = params['beta'], params['W']                 # [G], [G, K]
        Lambda = params['Lambda']                              # [G, d_c]
        log_theta = params['log_theta']                        # [G]

        # ── Encoder ──
        x_agg = self._segment_pool(x_log, observed_mask)      # [B, S]
        u = self.mlp_u(x_agg.unsqueeze(-1))                    # [B, S, d_u]
        z = self.mlp_z(x_agg)                                  # [B, K]

        if self.cell_emb_proj is not None and cell_emb is not None:
            c = self.cell_emb_proj(cell_emb)                    # [B, d_c]
        elif self.mlp_c_fallback is not None:
            c = self.mlp_c_fallback(torch.stack([
                x_log.mean(dim=-1), x_log.std(dim=-1), (x == 0).float().mean(dim=-1),
            ], dim=-1))                                          # [B, d_c]
        else:
            c = torch.zeros(B, self.d_c, device=x.device)

        # ── Decoder ──
        # Per-gene segment assignment
        seg_ids = torch.zeros(G, dtype=torch.long, device=x.device)
        for i, (start, end) in enumerate(self.chrom_boundaries):
            seg_ids[start:end] = i

        u_g = u[:, seg_ids, :]                                  # [B, G, d_u]
        pi = alpha + (Gamma * u_g).sum(dim=-1)                  # [B, G]

        if self.training:
            o_raw = self._gumbel_sigmoid(pi, self.gate_tau.item())
        else:
            o_raw = torch.sigmoid(pi)

        r = F.softplus(
            beta + (z @ W.T) + (c @ Lambda.T)
        )  # [B, G]

        ell = x.sum(dim=-1, keepdim=True)
        ell = ell / ell.mean().clamp(1e-8)
        eps = self.epsilon.item() if self.training else 0.01
        o_eff = eps + (1 - eps) * o_raw                         # [B, G]

        if ablation == "no_gate":
            mu = ell * r
        elif ablation == "no_drive":
            mu = ell * o_eff
        elif ablation == "additive":
            mu = ell * (o_eff + r)
        else:
            mu = ell * o_eff * r

        theta = F.softplus(log_theta)                            # [G]

        if return_latent:
            return mu, theta, {
                'u': u, 'z': z, 'c': c,
                'o_raw': o_raw, 'o_eff': o_eff, 'r': r, 'pi': pi,
                'W': W, 'alpha': alpha, 'beta': beta, 'Gamma': Gamma, 'Lambda': Lambda,
            }
        return mu, theta

    # ── Annealing ──

    def set_annealing(self, gate_tau: float, epsilon: float):
        self.gate_tau.fill_(gate_tau)
        self.epsilon.fill_(epsilon)

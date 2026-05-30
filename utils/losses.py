import torch
import torch.nn.functional as F


def delta_loss(mu1, x, delta_pi, delta_mu, is_perturbed, o1=None,
               r1=None, l1_weight=0.01, pbs_weight=0.5, gate_bimodal_weight=0.0,
               open_fraction_target=0.25, open_fraction_weight=0.0,
               r_o_couple_weight=0.0):
    """
    Hayat delta prediction loss with gate/drive decomposition.

    Args:
        mu1:          [B, G] predicted expression (baseline + delta)
        x:            [B, G] observed expression
        delta_pi:     [B, G] gate logit change
        delta_mu:     [B, G] drive log-fold-change
        is_perturbed: [B]    0=PBS control, 1=perturbed
        o1:           [B, G] perturbed gate (for bimodal/open-fraction penalties)
        r1:           [B, G] perturbed drive (for r-o coupling)
    """
    mse = F.mse_loss(mu1, x)
    l1_pi = delta_pi.abs().mean()
    l1_mu = delta_mu.abs().mean()

    # PBS zero
    pbs_mask = (is_perturbed == 0)
    if pbs_mask.sum() > 0:
        pbs_pi = (delta_pi[pbs_mask] ** 2).mean()
        pbs_mu = (delta_mu[pbs_mask] ** 2).mean()
        pbs_loss = pbs_pi + pbs_mu
    else:
        pbs_loss = torch.tensor(0.0, device=x.device)
        pbs_pi = pbs_loss
        pbs_mu = pbs_loss

    # Gate bimodal: push o towards 0 or 1
    bimodal = (o1 * (1 - o1)).mean() if o1 is not None else torch.tensor(0.0, device=x.device)

    # Target open fraction: mean(o) ≈ target
    open_loss = torch.tensor(0.0, device=x.device)
    if o1 is not None and open_fraction_weight > 0:
        open_loss = (o1.mean() - open_fraction_target) ** 2

    # r-o coupling: penalize high r when gate is closed
    r_o_loss = torch.tensor(0.0, device=x.device)
    if r1 is not None and o1 is not None and r_o_couple_weight > 0:
        r_o_loss = (r1 ** 2 * (1 - o1)).mean()

    total = (mse + l1_weight * (l1_pi + l1_mu) + pbs_weight * pbs_loss
             + gate_bimodal_weight * bimodal
             + open_fraction_weight * open_loss
             + r_o_couple_weight * r_o_loss)

    pbs_pi_v = pbs_pi.item() if isinstance(pbs_pi, torch.Tensor) and pbs_mask.sum() > 0 else 0.0
    pbs_mu_v = pbs_mu.item() if isinstance(pbs_mu, torch.Tensor) and pbs_mask.sum() > 0 else 0.0

    return total, {
        'mse': mse.item(), 'l1_pi': l1_pi.item(), 'l1_mu': l1_mu.item(),
        'pbs_pi': pbs_pi_v, 'pbs_mu': pbs_mu_v,
        'bimodal': bimodal.item(), 'open': open_loss.item(), 'r_o': r_o_loss.item(),
    }


# ── Config ──

config = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "grad_clip_value": 1.0,
    "batch_size": 2048,
    "max_train_cells": 0,
    "l1_weight": 0.01,
    "pbs_weight": 0.5,
    "gate_bimodal_weight": 0.1,
    "open_fraction_target": 0.25,
    "open_fraction_weight": 0.05,
    "r_o_couple_weight": 0.01,
    "tau_init": 0.5,
    "tau_final": 0.1,
    "drive_noise_init": 0.1,
    "drive_noise_final": 0.01,
    "noise_epochs": 30,
    "pbs_epochs": 20,
    "pbs_max_cells": 50000,
    "delta_epochs": 50,
    "delta_lr": 5e-5,
}


# ── Metrics ──

def _gene_pearson_subset(pred, tgt, mask):
    if mask.sum() < 5:
        return 0.0
    eps = 1e-8
    p, t = pred[:, mask], tgt[:, mask]
    t_mean = torch.mean(t, dim=0, keepdim=True)
    p_mean = torch.mean(p, dim=0, keepdim=True)
    cov = torch.mean((t - t_mean) * (p - p_mean), dim=0)
    t_std = torch.std(t, dim=0)
    p_std = torch.std(p, dim=0)
    return torch.mean(cov / (t_std * p_std + eps)).item()


def calculate_metrics(predictions, targets):
    eps = 1e-8
    G = predictions.shape[1]
    err = predictions - targets

    mse = torch.mean(err ** 2).item()
    mae = torch.mean(err.abs()).item()

    # R² = 1 - SS_res / SS_tot (across all cells × genes)
    ss_res = (err ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = (1 - ss_res / (ss_tot + eps)).item()

    t_mean_cell = torch.mean(targets, dim=1, keepdim=True)
    p_mean_cell = torch.mean(predictions, dim=1, keepdim=True)
    cov_cell = torch.mean((targets - t_mean_cell) * (predictions - p_mean_cell), dim=1)
    t_std_cell = torch.std(targets, dim=1)
    p_std_cell = torch.std(predictions, dim=1)
    cell_pearson = torch.mean(cov_cell / (t_std_cell * p_std_cell + eps)).item()

    t_mean_gene = torch.mean(targets, dim=0, keepdim=True)
    p_mean_gene = torch.mean(predictions, dim=0, keepdim=True)
    cov_gene = torch.mean((targets - t_mean_gene) * (predictions - p_mean_gene), dim=0)
    t_std_gene = torch.std(targets, dim=0)
    p_std_gene = torch.std(predictions, dim=0)
    gene_pearson = torch.mean(cov_gene / (t_std_gene * p_std_gene + eps)).item()

    # Spearman (rank) — compute on subset to keep memory bounded
    n_sample = min(predictions.shape[0], 2000)
    if predictions.shape[0] > n_sample:
        idx = torch.randperm(predictions.shape[0])[:n_sample]
        p_sub, t_sub = predictions[idx], targets[idx]
    else:
        p_sub, t_sub = predictions, targets
    p_rank = p_sub.argsort(dim=0).argsort(dim=0).float()
    t_rank = t_sub.argsort(dim=0).argsort(dim=0).float()
    r_mean = p_rank.mean(dim=0, keepdim=True)
    t_mean_r = t_rank.mean(dim=0, keepdim=True)
    cov_r = ((p_rank - r_mean) * (t_rank - t_mean_r)).mean(dim=0)
    sr = (cov_r / (p_rank.std(dim=0) * t_rank.std(dim=0) + eps)).mean().item()

    gene_var = targets.var(dim=0)
    det_rate = (targets > 0).float().mean(dim=0)
    hvg_mask = gene_var > gene_var.topk(min(2000, G)).values.min()
    mid_mask = (det_rate > 0.1) & (det_rate < 0.9)

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "spearman": sr,
        "cell_pearson": cell_pearson,
        "gene_pearson": gene_pearson,
        "gene_hvg": _gene_pearson_subset(predictions, targets, hvg_mask),
        "gene_mid": _gene_pearson_subset(predictions, targets, mid_mask),
    }


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))

import torch
import torch.nn.functional as F


def nb_loss(mu, x, theta, mask=None):
    """
    NB2 negative log-likelihood.
    mu: [B, G] > 0,  x: [B, G] counts,  theta: [G] or scalar.
    mask: [B, G] or [G], 1=observed, 0=unobserved (optional).
    """
    eps = 1e-8
    if theta.dim() == 0:
        theta = theta.unsqueeze(0)  # scalar → [1]
    # theta: [G] or [1], broadcast to [B, G]
    t1 = torch.lgamma(x + theta + eps) - torch.lgamma(theta + eps) - torch.lgamma(x + 1.0)
    t2 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
    t3 = x * (torch.log(mu + eps) - torch.log(theta + mu + eps))
    nll = -(t1 + t2 + t3)
    if mask is not None:
        nll = nll * mask
        return nll.sum() / mask.sum().clamp(1)
    return nll.mean()


def detection_loss(pi, x):
    """BCE: gate logit π predicts whether x>0. Only computed on masked genes."""
    is_expr = (x > 0).float()
    return F.binary_cross_entropy_with_logits(pi, is_expr, reduction='none')


def delta_loss(delta_mu, x_pert, mu_base):
    """Huber between predicted Δμ and log-fold-change from baseline."""
    target = torch.log1p(x_pert) - torch.log1p(mu_base.detach() + 1e-8)
    return F.huber_loss(delta_mu, target)


def positive_nb_loss(mu, x, theta):
    """NB loss computed only on positions where x>0."""
    mask_pos = (x > 0).float()
    total = mask_pos.sum().clamp(1)
    return (nb_loss(mu, x, theta, mask=mask_pos) * total / mask_pos.numel())


def compute_loss(mu, x, theta, pi, o_raw, o_eff, z, W,
                 gene_mask=None, gate_rate_target=0.15):
    """
    Hayat loss with masked detection supervision.

    Args:
        mu:        [B, G] predicted mean expression
        x:         [B, G] counts
        theta:     [G] NB dispersion
        pi:        [B, G] gate logits (raw, before sigmoid)
        o_raw:     [B, G] raw gate
        o_eff:     [B, G] effective gate
        z:         [B, K] trans programs
        W:         [G, K] gene-program loading
        gene_mask: [B, G] 1=visible, 0=masked (detection target)
        gate_rate_target: desired mean open rate

    Returns (total, components_dict, gate_stats_dict).
    """
    # Reconstruction: standard NB on unmasked genes
    if gene_mask is not None:
        nb = nb_loss(mu, x, theta, mask=gene_mask)
        # Detection: gate predicts 1[x>0] on MASKED genes
        mask_target = 1.0 - gene_mask
        det = detection_loss(pi, x)
        det = (det * mask_target).sum() / mask_target.sum().clamp(1)
        # Positive magnitude: NB on masked genes where x>0
        mag = positive_nb_loss(mu, x, theta)
        mag = (mag * mask_target).sum() / mask_target.sum().clamp(1)
    else:
        nb = nb_loss(mu, x, theta)
        det = torch.tensor(0.0, device=x.device)
        mag = torch.tensor(0.0, device=x.device)

    w_sparse = W.abs().mean()

    gate_bimodal = (o_raw * (1.0 - o_raw)).mean()
    gate_rate = o_raw.mean()
    gate_rate_penalty = (gate_rate - gate_rate_target) ** 2

    K = z.shape[-1]
    z_c = z - z.mean(dim=0, keepdim=True)
    z_std = z_c.std(dim=0, keepdim=True).clamp(1e-8)
    z_corr = (z_c / z_std).T @ (z_c / z_std) / (z_c.shape[0] - 1)
    eye = torch.eye(K, device=z.device)
    program_decorr = (z_corr.abs() * (1.0 - eye)).mean()

    total = (
        nb
        + 1.0 * det
        + 1.0 * mag
        + 0.01 * w_sparse
        + 0.1 * gate_bimodal
        + 0.1 * gate_rate_penalty
        + 0.05 * program_decorr
    )

    # Gate diagnostics
    p_closed = (o_raw < 0.1).float().mean()
    p_open = (o_raw > 0.9).float().mean()
    p_mid = 1.0 - p_closed - p_open
    o_std_across_genes = o_raw.mean(dim=0).std()

    gate_stats = {
        'o_raw_mean': o_raw.mean().item(),
        'o_raw_std': o_raw.std().item(),
        'o_eff_mean': o_eff.mean().item(),
        'o_eff_std': o_eff.std().item(),
        'p_closed': p_closed.item(),
        'p_open': p_open.item(),
        'p_mid': p_mid.item(),
        'o_gene_std': o_std_across_genes.item(),
        'gate_rate': gate_rate.item(),
    }

    # Store for epoch-level AUC computation
    if gene_mask is not None:
        gate_stats['det_pi'] = pi.detach()
        gate_stats['det_x'] = x
        gate_stats['det_mask'] = 1.0 - gene_mask
    else:
        gate_stats['det_pi'] = None

    return total, {
        'nb': nb.item(),
        'det': det.item(),
        'mag': mag.item(),
        'sparse': w_sparse.item(),
        'bimodal': gate_bimodal.item(),
        'rate_pen': gate_rate_penalty.item(),
        'decorr': program_decorr.item(),
    }, gate_stats


# ── Config ──

config = {
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "grad_clip_value": 1.0,
    "stage0_epochs": 30,
    "stage1_epochs": 40,
    "batch_size": 16,
    "max_train_cells": 0,
    "gate_rate_target": 0.15,
    "delta_weight": 0.5,
    "pert_batch_ratio": 0.5,
}

# ── Metrics & checkpointing ──

def _gene_pearson_subset(pred, tgt, mask):
    """Per-gene Pearson for a subset of genes (mask: [G] bool)."""
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
    """Per-cell/gene/global Pearson + stratified gene Pearson + MSE."""
    eps = 1e-8
    G = predictions.shape[1]
    mse = torch.mean((predictions - targets) ** 2).item()

    # Per-cell Pearson
    t_mean_cell = torch.mean(targets, dim=1, keepdim=True)
    p_mean_cell = torch.mean(predictions, dim=1, keepdim=True)
    cov_cell = torch.mean((targets - t_mean_cell) * (predictions - p_mean_cell), dim=1)
    t_std_cell = torch.std(targets, dim=1)
    p_std_cell = torch.std(predictions, dim=1)
    cell_pearson = torch.mean(cov_cell / (t_std_cell * p_std_cell + eps)).item()

    # Per-gene Pearson: for each gene, corr across cells
    t_mean_gene = torch.mean(targets, dim=0, keepdim=True)
    p_mean_gene = torch.mean(predictions, dim=0, keepdim=True)
    cov_gene = torch.mean((targets - t_mean_gene) * (predictions - p_mean_gene), dim=0)
    t_std_gene = torch.std(targets, dim=0)
    p_std_gene = torch.std(predictions, dim=0)
    gene_pearson = torch.mean(cov_gene / (t_std_gene * p_std_gene + eps)).item()

    # Global Pearson: all (cell, gene) pairs flattened
    t_flat = targets.flatten()
    p_flat = predictions.flatten()
    t_mean = torch.mean(t_flat)
    p_mean = torch.mean(p_flat)
    cov_global = torch.mean((t_flat - t_mean) * (p_flat - p_mean))
    global_pearson = (cov_global / (torch.std(t_flat) * torch.std(p_flat) + eps)).item()

    # Stratified gene Pearson
    det_rate = (targets > 0).float().mean(dim=0)  # [G]
    gene_var = targets.var(dim=0)                   # [G]

    hvg_mask = gene_var > gene_var.topk(min(2000, G)).values.min()
    mid_mask = (det_rate > 0.1) & (det_rate < 0.9)
    high_mask = det_rate > 0.9

    return {"mse": mse, "cell_pearson": cell_pearson,
            "gene_pearson": gene_pearson, "global_pearson": global_pearson,
            "gene_hvg": _gene_pearson_subset(predictions, targets, hvg_mask),
            "gene_mid": _gene_pearson_subset(predictions, targets, mid_mask),
            "gene_high": _gene_pearson_subset(predictions, targets, high_mask)}


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device))

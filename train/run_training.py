#!/usr/bin/env python3
"""
Hayat training with gene-conditioned decoder + 3-phase annealing.
Gene embeddings → hypernetworks → per-gene params (α, β, Γ, W, Λ, θ).
No fixed gene set — works with any genes that have scGPT embeddings.
"""
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from tqdm import tqdm

from models import Hayat
from utils.losses import compute_loss, delta_loss, config, calculate_metrics, save_checkpoint, load_checkpoint


class PertDataset(Dataset):
    def __init__(self, x, pert_labels=None):
        self.x = x
        self.pert_labels = pert_labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.pert_labels is None:
            return self.x[idx], None
        return self.x[idx], {k: v[idx] for k, v in self.pert_labels.items()}


def create_dataloaders(train_data, val_data, train_cell_emb=None, val_cell_emb=None,
                       train_pert=None, val_pert=None):
    if train_pert is not None:
        train_ds = PertDataset(train_data, train_pert)
        val_ds = PertDataset(val_data, val_pert)
    elif train_cell_emb is not None:
        train_ds = TensorDataset(train_data, train_cell_emb)
        val_ds = TensorDataset(val_data, val_cell_emb)
    else:
        train_ds = TensorDataset(train_data)
        val_ds = TensorDataset(val_data)

    def collate(batch):
        if isinstance(batch[0], tuple) and len(batch[0]) == 2:
            has_pert = batch[0][1] is not None
            x = torch.stack([s[0] for s in batch])
            if has_pert:
                pert = {k: torch.stack([s[1][k] for s in batch]) for k in batch[0][1]}
            else:
                pert = None
            return x, pert
        else:
            return torch.stack([s for s in batch]), None

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def get_annealing(epoch, num_epochs):
    """
    3-stage schedule:
      Stage 0 (drive warmup):   o≡1, alpha_scale=0     epochs 0–4
      Stage 1 (gate from u):    alpha_scale=0.1         epochs 5–14
      Stage 2 (full gate):      alpha_scale→1, harden   epochs 15+
    """
    stage0_end = max(1, int(num_epochs * 0.05))   # 5%
    stage1_end = max(1, int(num_epochs * 0.15))   # 15%

    if epoch < stage0_end:
        # Stage 0: gate disabled, train drive only
        return 1.0, 0.5, 0.0
    elif epoch < stage1_end:
        # Stage 1: Γ·u active, α suppressed
        return 1.0, 0.1, 0.1
    else:
        # Stage 2: α ramps up, hardening
        frac = (epoch - stage1_end) / max(num_epochs - stage1_end - 1, 1)
        alpha_scale = 0.1 + 0.9 * frac
        tau = 1.0 - 0.5 * frac
        eps = 0.1 - 0.09 * frac
        return tau, eps, alpha_scale


def train_epoch(model, dataloader, optimizer, device, gene_emb, gate_rate_target, stage=0):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    loss_components = {}
    gate_stats_sum = {}
    delta_total = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        x = batch[0].to(device)
        pert_info = batch[1] if len(batch) > 1 and batch[1] is not None else None
        if pert_info is not None:
            pert_info = {k: v.to(device) for k, v in pert_info.items()}

        gene_mask = torch.bernoulli(torch.full_like(x, 0.8)).to(device)

        optimizer.zero_grad()

        mu, theta, latents = model(x, gene_emb=gene_emb, pert_info=pert_info,
                                   gene_mask=gene_mask, return_latent=True)

        if stage == 0:
            # PBS pretrain: baseline only
            loss, comps, gs = compute_loss(
                mu, x, theta, latents['pi'], latents['o_raw'], latents['o_eff'],
                latents['z'], latents['W'], gene_mask=gene_mask,
                gate_rate_target=gate_rate_target)
            preds = mu
        else:
            # Joint: μ_pert for all cells; delta loss pushes Δ→0 for control cells
            mu_pert = latents['mu_pert']
            pi_pert = latents['pi_pert']
            o_pert = latents['o_pert']

            loss, comps, gs = compute_loss(
                mu_pert, x, theta, pi_pert, o_pert, latents['o_eff'],
                latents['z'], latents['W'], gene_mask=gene_mask,
                gate_rate_target=gate_rate_target)

            loss_delta = delta_loss(latents['delta_mu'], x, mu)
            loss = loss + config["delta_weight"] * loss_delta
            delta_total += loss_delta.item()
            preds = mu_pert

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_value"])
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(preds.detach().cpu())
        all_targets.append(x.cpu())
        for k, v in comps.items():
            loss_components[k] = loss_components.get(k, 0.0) + v
        for k, v in gs.items():
            if k not in ('det_pi', 'det_x', 'det_mask'):
                gate_stats_sum[k] = gate_stats_sum.get(k, 0.0) + v

    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_preds, all_targets)
    for k in loss_components:
        loss_components[k] /= len(dataloader)
    for k in gate_stats_sum:
        gate_stats_sum[k] /= len(dataloader)
    gate_stats_sum['delta'] = delta_total / len(dataloader)
    return avg_loss, loss_components, metrics, gate_stats_sum


def val_epoch(model, dataloader, device, gene_emb, gate_rate_target, stage=0):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    loss_components = {}
    last_latents = None
    gate_stats_sum = {}
    delta_total = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            x = batch[0].to(device)
            pert_info = batch[1] if len(batch) > 1 and batch[1] is not None else None
            if pert_info is not None:
                pert_info = {k: v.to(device) for k, v in pert_info.items()}

            gene_mask = torch.zeros_like(x)
            gene_mask[:, :int(0.8 * x.shape[1])] = 1.0

            mu, theta, latents = model(x, gene_emb=gene_emb, pert_info=pert_info,
                                       gene_mask=gene_mask, return_latent=True)

            if stage == 0:
                loss, comps, gs = compute_loss(
                    mu, x, theta, latents['pi'], latents['o_raw'], latents['o_eff'],
                    latents['z'], latents['W'], gene_mask=gene_mask,
                    gate_rate_target=gate_rate_target)
                preds = mu
            else:
                mu_pert = latents['mu_pert']
                pi_pert = latents['pi_pert']
                o_pert = latents['o_pert']

                loss, comps, gs = compute_loss(
                    mu_pert, x, theta, pi_pert, o_pert, latents['o_eff'],
                    latents['z'], latents['W'], gene_mask=gene_mask,
                    gate_rate_target=gate_rate_target)

                loss_delta = delta_loss(latents['delta_mu'], x, mu)
                loss = loss + config["delta_weight"] * loss_delta
                delta_total += loss_delta.item()
                preds = mu_pert

            total_loss += loss.item()
            all_preds.append(preds.cpu())
            all_targets.append(x.cpu())
            for k, v in comps.items():
                loss_components[k] = loss_components.get(k, 0.0) + v
            for k, v in gs.items():
                if k not in ('det_pi', 'det_x', 'det_mask'):
                    gate_stats_sum[k] = gate_stats_sum.get(k, 0.0) + v
            last_det = {k: gs[k] for k in ('det_pi', 'det_x', 'det_mask') if k in gs}
            last_latents = {k: v.cpu() for k, v in latents.items()}

    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_preds, all_targets)
    for k in loss_components:
        loss_components[k] /= len(dataloader)
    for k in gate_stats_sum:
        gate_stats_sum[k] /= len(dataloader)
    gate_stats_sum.update(last_det)
    gate_stats_sum['delta'] = delta_total / len(dataloader)
    return avg_loss, loss_components, metrics, last_latents, gate_stats_sum


def train_model(model, train_data, val_data, config, device, checkpoint_path,
                gene_emb=None, train_cell_emb=None, val_cell_emb=None,
                train_pert=None, val_pert=None):
    use_pert = train_pert is not None
    train_loader, val_loader = create_dataloaders(
        train_data, val_data, train_cell_emb, val_cell_emb, train_pert, val_pert)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best_val_loss = float("inf")

    total_epochs = config["stage0_epochs"] + config["stage1_epochs"]

    for epoch in range(total_epochs):
        if epoch < config["stage0_epochs"]:
            stage = 0
            tau, eps, alpha_scale = get_annealing(epoch, config["stage0_epochs"])
            stage_name = "DRIVE" if alpha_scale == 0 else ("GATE_Γu" if alpha_scale <= 0.11 else "FULL")
        else:
            stage = 1
            tau, eps, alpha_scale = 0.5, 0.01, 1.0
            stage_name = "DELTA"

        model.set_annealing(tau, eps, alpha_scale)

        print(f"\nEpoch {epoch + 1}/{total_epochs}  [Stage {stage}: {stage_name}]  "
              f"τ={tau:.2f} ε={eps:.3f} α_scale={alpha_scale:.2f}")
        train_loss, train_comp, train_metrics, train_gs = train_epoch(
            model, train_loader, optimizer, device, gene_emb, config["gate_rate_target"], stage=stage)
        val_loss, val_comp, val_metrics, latents, val_gs = val_epoch(
            model, val_loader, device, gene_emb, config["gate_rate_target"], stage=stage)

        # Alpha vs Gamma·u dominance
        alpha_mean = latents['alpha'].abs().mean().item()
        gamma_u_mean = latents['gamma_u'].abs().mean().item()
        ratio = gamma_u_mean / (alpha_mean + 1e-8)
        pi_var = latents['pi'].var(dim=0).mean().item()

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Pearson: cell={train_metrics['cell_pearson']:.4f} gene={train_metrics['gene_pearson']:.4f} gene_hvg={train_metrics['gene_hvg']:.4f} gene_mid={train_metrics['gene_mid']:.4f}")
        print(f"Val Pearson:   cell={val_metrics['cell_pearson']:.4f} gene={val_metrics['gene_pearson']:.4f} gene_hvg={val_metrics['gene_hvg']:.4f} gene_mid={val_metrics['gene_mid']:.4f}")
        print(f"Loss: { {k: f'{v:.4f}' for k, v in val_comp.items()} }")
        gs = val_gs
        print(f"Gate: raw={gs['o_raw_mean']:.3f} closed={gs['p_closed']:.3f} open={gs['p_open']:.3f} mid={gs['p_mid']:.3f} gene_std={gs['o_gene_std']:.3f}")
        if stage == 1:
            print(f"Delta loss: {gs.get('delta', 0.0):.4f}")

        try:
            from sklearn.metrics import roc_auc_score
            det_pi = gs.get('det_pi')
            if det_pi is not None:
                det_x = gs['det_x']
                det_mask = gs['det_mask'].bool()
                det_auc = roc_auc_score(
                    (det_x[det_mask] > 0).float().cpu().numpy(),
                    det_pi[det_mask].cpu().numpy(),
                )
                print(f"Det AUC: {det_auc:.4f} | Dominance: |α|={alpha_mean:.4f} |Γ·u|={gamma_u_mean:.4f} ratio={ratio:.3f} Var(π)={pi_var:.6f}")
            else:
                print(f"Dominance: |α|={alpha_mean:.4f} |Γ·u|={gamma_u_mean:.4f} ratio={ratio:.3f} Var(π)={pi_var:.6f}")
        except Exception:
            print(f"Dominance: |α|={alpha_mean:.4f} |Γ·u|={gamma_u_mean:.4f} ratio={ratio:.3f} Var(π)={pi_var:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, checkpoint_path)
            print(f"Best model saved to {checkpoint_path}")

    load_checkpoint(model, checkpoint_path, device)
    return model


if __name__ == "__main__":
    data_dir = os.environ.get("GENE_DATA_DIR", "../data")
    is_schmidt = (data_dir == "Schmidt" or data_dir.endswith("Schmidt"))
    if is_schmidt:
        data_file = f"../{data_dir}/schmidt_data.pt"
        gene_emb_file = f"../{data_dir}/schmidt_gene_embeddings.pt"
        chrom_file = f"../{data_dir}/schmidt_chrom_boundaries.pt"
        pert_file = f"../{data_dir}/schmidt_pert_labels.pt"
    else:
        data_file_100k = f"{data_dir}/processed_data_100k.pt"
        if os.path.exists(data_file_100k):
            data_file = data_file_100k
        else:
            data_file = f"{data_dir}/processed_data.pt"
        gene_emb_file = f"{data_dir}/gene_embeddings.pt"
        chrom_file = f"{data_dir}/chrom_boundaries.pt"
        pert_file = f"{data_dir}/pert_labels.pt"

    cell_emb_file = f"{data_dir}/cell_embeddings.pt"

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    data = torch.load(data_file)

    if is_schmidt:
        expr = data["expression"]
        max_cells = config.get("max_train_cells", 0)
        if max_cells > 0 and expr.shape[0] > max_cells:
            expr = expr[:max_cells]
        split = int(expr.shape[0] * 0.8)
        rng = torch.Generator()
        indices = torch.randperm(expr.shape[0], generator=rng)
        train_data = expr[indices[:split]]
        val_data = expr[indices[split:]]
    else:
        train_data = data["train"]
        val_data = data["val"]
        max_cells = config.get("max_train_cells", 0)
        if max_cells > 0:
            if train_data.shape[0] > max_cells:
                train_data = train_data[:max_cells]
            val_max = max(max_cells // 5, 200)
            if val_data.shape[0] > val_max:
                val_data = val_data[:val_max]

    # ── Gene embeddings (required: scGPT gene token embeddings) ──
    if os.path.exists(gene_emb_file):
        gene_emb = torch.load(gene_emb_file)
        gene_emb_dim = gene_emb.shape[1]
        print(f"Gene embeddings: {gene_emb.shape[0]} genes × {gene_emb_dim} dim")
    else:
        raise FileNotFoundError(f"Gene embeddings required: {gene_emb_file}")

    # ── Cell embeddings (optional: pre-computed scGPT cell embeddings) ──
    train_cell_emb, val_cell_emb, cell_emb_dim = None, None, 0
    if os.path.exists(cell_emb_file):
        cell_data = torch.load(cell_emb_file)
        if (cell_data["train_emb"].shape[0] >= train_data.shape[0] and
            cell_data["val_emb"].shape[0] >= val_data.shape[0]):
            train_cell_emb = cell_data["train_emb"][:train_data.shape[0]]
            val_cell_emb = cell_data["val_emb"][:val_data.shape[0]]
            cell_emb_dim = cell_data["embed_dim"]
            print(f"Cell embeddings: {cell_emb_dim}-dim")
        else:
            print(f"Cell embeddings too small ({cell_data['train_emb'].shape[0]}), skipping")

    # ── Perturbation labels (optional: cytokine/dose/time per cell) ──
    train_pert, val_pert, pert_meta = None, None, {}
    if os.path.exists(pert_file):
        pert_data = torch.load(pert_file)
        if is_schmidt:
            n_cells = expr.shape[0]
            split_pert = {k: v[indices] for k, v in pert_data.items() if isinstance(v, torch.Tensor)}
            train_pert = {k: v[:split] for k, v in split_pert.items()}
            val_pert = {k: v[split:] for k, v in split_pert.items()}
        else:
            train_pert = pert_data.get("train_pert", pert_data.get("pert_labels"))
            val_pert = pert_data.get("val_pert", {})
            if train_pert is not None:
                for k in train_pert:
                    train_pert[k] = train_pert[k][:train_data.shape[0]]
                for k in val_pert:
                    val_pert[k] = val_pert[k][:val_data.shape[0]]
        pert_meta = {
            'n_cytokines': pert_data.get('n_cytokines', 100),
            'n_doses': pert_data.get('n_doses', 10),
            'n_times': pert_data.get('n_times', 10),
        }
        print(f"Perturbation labels: {len(train_pert)} keys, "
              f"n_cytokines={pert_meta['n_cytokines']}")

    chrom_boundaries = torch.load(chrom_file)

    print(f"Training: {train_data.shape[0]} cells × {train_data.shape[1]} genes")
    print(f"Segments: {len(chrom_boundaries)}")

    model = Hayat(
        chrom_boundaries=chrom_boundaries,
        gene_emb_dim=gene_emb_dim,
        cell_emb_dim=cell_emb_dim,
        n_cytokines=pert_meta.get('n_cytokines', 100),
        n_doses=pert_meta.get('n_doses', 10),
        n_times=pert_meta.get('n_times', 10),
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total | {trainable:,} trainable | "
          f"{total * 4 / 1024 / 1024:.1f} MB")

    ckpt_name = f"hayat_{data_dir.replace('/', '_')}.pt"
    ckpt_path = f"../checkpoints/{ckpt_name}"

    train_model(model, train_data, val_data, config, device=device, checkpoint_path=ckpt_path,
                gene_emb=gene_emb, train_cell_emb=train_cell_emb, val_cell_emb=val_cell_emb,
                train_pert=train_pert, val_pert=val_pert)

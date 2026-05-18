#!/usr/bin/env python3
"""
Hayat training with gene-conditioned decoder + 3-phase annealing.
Gene embeddings → hypernetworks → per-gene params (α, β, Γ, W, Λ, θ).
No fixed gene set — works with any genes that have scGPT embeddings.
"""
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models import Hayat
from utils.losses import compute_loss, config, calculate_metrics, save_checkpoint, load_checkpoint


def create_dataloaders(train_data, val_data, train_cell_emb=None, val_cell_emb=None):
    if train_cell_emb is not None:
        train_ds = TensorDataset(train_data, train_cell_emb)
        val_ds = TensorDataset(val_data, val_cell_emb)
        train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,
            collate_fn=lambda b: (torch.stack([s[0] for s in b]),
                                  torch.stack([s[1] for s in b])))
        val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False,
            collate_fn=lambda b: (torch.stack([s[0] for s in b]),
                                  torch.stack([s[1] for s in b])))
    else:
        train_loader = DataLoader(TensorDataset(train_data), batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(TensorDataset(val_data), batch_size=config["batch_size"], shuffle=False)
    return train_loader, val_loader


def get_annealing(epoch, num_epochs):
    warmup_end = int(num_epochs * 0.1)
    harden_end = int(num_epochs * 0.3)
    if epoch < warmup_end:
        frac = epoch / max(warmup_end - 1, 1)
        return 1.0, 0.5 - 0.4 * frac
    elif epoch < harden_end:
        frac = (epoch - warmup_end) / max(harden_end - warmup_end - 1, 1)
        return 1.0 - 0.5 * frac, 0.1 - 0.09 * frac
    else:
        return 0.5, 0.01


def train_epoch(model, dataloader, optimizer, device, gene_emb, gate_rate_target):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    loss_components = {}
    gate_stats_sum = {}

    for batch in tqdm(dataloader, desc="Training"):
        x = batch[0].to(device)
        cell_emb = batch[1].to(device) if len(batch) > 1 and batch[1].dim() == 2 else None

        optimizer.zero_grad()

        mu, theta, latents = model(x, gene_emb=gene_emb, cell_emb=cell_emb, return_latent=True)
        loss, comps, gs = compute_loss(
            mu, x, theta, latents['o_raw'], latents['o_eff'],
            latents['z'], latents['W'], gate_rate_target=gate_rate_target,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_value"])
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(mu.detach().cpu())
        all_targets.append(x.cpu())
        for k, v in comps.items():
            loss_components[k] = loss_components.get(k, 0.0) + v
        for k, v in gs.items():
            gate_stats_sum[k] = gate_stats_sum.get(k, 0.0) + v

    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_preds, all_targets)
    for k in loss_components:
        loss_components[k] /= len(dataloader)
    for k in gate_stats_sum:
        gate_stats_sum[k] /= len(dataloader)
    return avg_loss, loss_components, metrics, gate_stats_sum


def val_epoch(model, dataloader, device, gene_emb, gate_rate_target):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    loss_components = {}
    last_latents = None
    gate_stats_sum = {}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            x = batch[0].to(device)
            cell_emb = batch[1].to(device) if len(batch) > 1 and batch[1].dim() == 2 else None

            mu, theta, latents = model(x, gene_emb=gene_emb, cell_emb=cell_emb, return_latent=True)
            loss, comps, gs = compute_loss(
                mu, x, theta, latents['o_raw'], latents['o_eff'],
                latents['z'], latents['W'], gate_rate_target=gate_rate_target,
            )

            total_loss += loss.item()
            all_preds.append(mu.cpu())
            all_targets.append(x.cpu())
            for k, v in comps.items():
                loss_components[k] = loss_components.get(k, 0.0) + v
            for k, v in gs.items():
                gate_stats_sum[k] = gate_stats_sum.get(k, 0.0) + v
            last_latents = {k: v.cpu() for k, v in latents.items()}

    avg_loss = total_loss / len(dataloader)
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = calculate_metrics(all_preds, all_targets)
    for k in loss_components:
        loss_components[k] /= len(dataloader)
    for k in gate_stats_sum:
        gate_stats_sum[k] /= len(dataloader)
    return avg_loss, loss_components, metrics, last_latents, gate_stats_sum


def train_model(model, train_data, val_data, config, device, checkpoint_path,
                gene_emb=None, train_cell_emb=None, val_cell_emb=None):
    train_loader, val_loader = create_dataloaders(train_data, val_data, train_cell_emb, val_cell_emb)
    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    best_val_loss = float("inf")

    for epoch in range(config["num_epochs"]):
        tau, eps = get_annealing(epoch, config["num_epochs"])
        model.set_annealing(tau, eps)

        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}  [τ={tau:.2f}, ε={eps:.3f}]")
        train_loss, train_comp, train_metrics, train_gs = train_epoch(
            model, train_loader, optimizer, device, gene_emb, config["gate_rate_target"])
        val_loss, val_comp, val_metrics, latents, val_gs = val_epoch(
            model, val_loader, device, gene_emb, config["gate_rate_target"])

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Pearson: cell={train_metrics['cell_pearson']:.4f} gene={train_metrics['gene_pearson']:.4f} global={train_metrics['global_pearson']:.4f}")
        print(f"Val Pearson:   cell={val_metrics['cell_pearson']:.4f} gene={val_metrics['gene_pearson']:.4f} global={val_metrics['global_pearson']:.4f}")
        print(f"Loss: { {k: f'{v:.4f}' for k, v in val_comp.items()} }")
        gs = val_gs
        print(f"Gate: raw={gs['o_raw_mean']:.3f} eff={gs['o_eff_mean']:.3f} "
              f"closed={gs['p_closed']:.3f} open={gs['p_open']:.3f} mid={gs['p_mid']:.3f} "
              f"gene_std={gs['o_gene_std']:.3f}")

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
    else:
        data_file_100k = f"{data_dir}/processed_data_100k.pt"
        if os.path.exists(data_file_100k):
            data_file = data_file_100k
        else:
            data_file = f"{data_dir}/processed_data.pt"
        gene_emb_file = f"{data_dir}/gene_embeddings.pt"
        chrom_file = f"{data_dir}/chrom_boundaries.pt"

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

    chrom_boundaries = torch.load(chrom_file)

    print(f"Training: {train_data.shape[0]} cells × {train_data.shape[1]} genes")
    print(f"Segments: {len(chrom_boundaries)}")

    model = Hayat(
        chrom_boundaries=chrom_boundaries,
        gene_emb_dim=gene_emb_dim,
        cell_emb_dim=cell_emb_dim,
    ).to(device)

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total:,} total | {trainable:,} trainable | "
          f"{total * 4 / 1024 / 1024:.1f} MB")

    ckpt_name = f"hayat_{data_dir.replace('/', '_')}.pt"
    ckpt_path = f"../checkpoints/{ckpt_name}"

    train_model(model, train_data, val_data, config, device=device, checkpoint_path=ckpt_path,
                gene_emb=gene_emb, train_cell_emb=train_cell_emb, val_cell_emb=val_cell_emb)

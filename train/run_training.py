#!/usr/bin/env python3
"""Hayat two-stage: PBS pre-train + delta RL — all dense in-memory."""
import os, torch, torch.optim as optim
from tqdm import tqdm
from models import Hayat
from utils.losses import delta_loss, config, calculate_metrics, save_checkpoint, load_checkpoint
from utils.stream_dataset import DenseDataset, dense_loader, n_batches

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_pbs(model, dataset, val_dataset, config, device, optimizer, gene_emb):
    epochs = config.get("pbs_epochs", 20)
    bm, of_w, of_t, ro = config["gate_bimodal_weight"], config["open_fraction_weight"], \
                         config["open_fraction_target"], config["r_o_couple_weight"]
    bs = config["batch_size"]

    print(f"\n{'='*60}")
    print(f"Stage 1: PBS pre-train (dense, {epochs} epochs, bs={bs})")
    print(f"{'='*60}")

    best_val = float("inf")
    for epoch in range(epochs):
        model.train()
        tl, n = 0.0, 0
        loader = dense_loader(dataset, bs, shuffle=True, seed=42 + epoch)
        for x, _ in tqdm(loader, desc=f"PBS {epoch+1}", total=n_batches(len(dataset), bs)):
            optimizer.zero_grad()
            mu0, lat = model.baseline_forward(x.to(device), gene_emb, return_latent=True)
            o, r = lat['o'], lat['r']
            mse = torch.nn.functional.mse_loss(mu0, x.to(device))
            loss = mse + bm*(o*(1-o)).mean() + of_w*(o.mean()-of_t)**2 + ro*(r**2*(1-o)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_value"])
            optimizer.step()
            tl += loss.item(); n += 1

        model.eval()
        vl, nv = 0.0, 0
        with torch.no_grad():
            for x, _ in tqdm(dense_loader(val_dataset, bs, shuffle=False),
                             desc="PBS Val", total=n_batches(len(val_dataset), bs)):
                mu0, _ = model.baseline_forward(x.to(device), gene_emb, return_latent=True)
                vl += torch.nn.functional.mse_loss(mu0, x.to(device)).item(); nv += 1
        print(f"  train={tl/n:.4f} val={vl/nv:.4f}")
        if vl/nv < best_val:
            best_val = vl/nv
            save_checkpoint(model, os.path.join(ROOT, "checkpoints", "hayat_pbs.pt"))
            print("  saved")

    load_checkpoint(model, os.path.join(ROOT, "checkpoints", "hayat_pbs.pt"), device)
    print(f"[PBS] done. best={best_val:.4f}")


def train_delta(model, dataset, val_dataset, config, device, gene_emb):
    model.freeze_baseline()
    delta_p = [p for p in model.parameters() if p.requires_grad]
    opt = optim.AdamW(delta_p, lr=config.get("delta_lr", 5e-5), weight_decay=config["weight_decay"])
    best_val = float("inf")

    ti, tf = config["tau_init"], config["tau_final"]
    di, df = config["drive_noise_init"], config["drive_noise_final"]
    ne = config["noise_epochs"]
    bw, ow, ot, rw = config["gate_bimodal_weight"], config["open_fraction_weight"], \
                     config["open_fraction_target"], config["r_o_couple_weight"]
    bs = config["batch_size"]

    print(f"\n{'='*60}")
    print(f"Stage 2: Delta RL (dense, {sum(p.numel() for p in delta_p):,} trainable, bs={bs})")
    print(f"{'='*60}")

    for ep in range(config["delta_epochs"]):
        f = min(ep / max(ne - 1, 1), 1.0)
        tau, dns = ti*(1-f)+tf*f, di*(1-f)+df*f if ep < ne else (tf, df)
        model.set_gate_temp(tau, dns)
        rl = tau > tf + 0.01 or dns > df + 0.001

        print(f"\nEpoch {ep+1}/{config['delta_epochs']}  [τ={tau:.3f} σ={dns:.3f}]")

        model.train()
        tl, ct = 0.0, 0
        ct_c = {}
        # Keep one batch on GPU for gate stats (avoids costly CPU transfer every iter)
        last_lat = None
        loader = dense_loader(dataset, bs, shuffle=True, seed=42 + ep)
        for x, pert_info in tqdm(loader, desc="Delta Train", total=n_batches(len(dataset), bs)):
            x = x.to(device)
            pert_info = {k: v.to(device) for k, v in pert_info.items()}
            opt.zero_grad()
            mu1, dmu, lat = model(x, gene_emb, pert_info, return_latent=True, rl_training=rl)
            loss, comps = delta_loss(mu1, x, lat['delta_pi'], dmu, pert_info['is_perturbed'],
                                     o1=lat['o1'], r1=lat['r1'],
                                     l1_weight=config["l1_weight"], pbs_weight=config["pbs_weight"],
                                     gate_bimodal_weight=bw, open_fraction_target=ot,
                                     open_fraction_weight=ow, r_o_couple_weight=rw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(delta_p, config["grad_clip_value"])
            opt.step()
            tl += loss.item(); ct += 1
            for k, v in comps.items():
                ct_c[k] = ct_c.get(k, 0.0) + v
            last_lat = {k: v.detach() for k, v in lat.items()}

        n = max(ct, 1)

        model.eval()
        vl, vt = 0.0, 0
        cv_c = {}
        # Val metrics accumulated on GPU in subsets to avoid full transfer
        vp_parts, vm_parts = [], []
        lat_sample = None
        max_val_cat = 500000  # Keep ~500K cells for metrics (enough for stable Pearson)
        with torch.no_grad():
            for x, pert_info in tqdm(dense_loader(val_dataset, bs, shuffle=False),
                                     desc="Delta Val", total=n_batches(len(val_dataset), bs)):
                x = x.to(device)
                pert_info = {k: v.to(device) for k, v in pert_info.items()}
                mu1, dmu, lat = model(x, gene_emb, pert_info, return_latent=True)
                loss, comps = delta_loss(mu1, x, lat['delta_pi'], dmu, pert_info['is_perturbed'],
                                         o1=lat['o1'], r1=lat['r1'],
                                         l1_weight=config["l1_weight"], pbs_weight=config["pbs_weight"],
                                         gate_bimodal_weight=bw, open_fraction_target=ot,
                                         open_fraction_weight=ow, r_o_couple_weight=rw)
                vl += loss.item(); vt += 1
                for k, v in comps.items():
                    cv_c[k] = cv_c.get(k, 0.0) + v
                # Subsample for metrics to keep memory bounded
                if sum(p.shape[0] for p in vp_parts) < max_val_cat:
                    vp_parts.append(mu1.cpu()); vm_parts.append(x.cpu())
                lat_sample = {k: v.cpu() for k, v in lat.items()}

        nv = max(vt, 1)
        val_met = calculate_metrics(torch.cat(vp_parts), torch.cat(vm_parts))
        o = last_lat['o'].cpu(); o1 = last_lat['o1'].cpu()
        print(f"  Train={tl/n:.4f} Val={vl/nv:.4f}")
        print(f"  cell_r={val_met['cell_pearson']:.4f} gene_r={val_met['gene_pearson']:.4f}")
        print(f"  Gate: c={(o<0.1).float().mean():.3f} o={(o>0.9).float().mean():.3f}")
        print(f"  ΔGate: c={(o1<0.1).float().mean():.3f} o={(o1>0.9).float().mean():.3f}")
        print(f"  Loss: { {k: f'{v/nv:.4f}' for k, v in cv_c.items()} }")
        print(f"  |δπ|={last_lat['delta_pi'].abs().mean():.4f} |δμ|={last_lat.get('delta_mu',torch.zeros(1)).abs().mean():.4f}")
        if vl/nv < best_val:
            best_val = vl/nv
            save_checkpoint(model, os.path.join(ROOT, "checkpoints", "hayat_delta.pt"))
            print("  saved")

    load_checkpoint(model, os.path.join(ROOT, "checkpoints", "hayat_delta.pt"), device)
    return model


if __name__ == "__main__":
    import sys, gc
    data_dir = os.environ.get("GENE_DATA_DIR", os.path.join(ROOT, "data"))

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    gene_emb = torch.load(os.path.join(data_dir, "gene_embeddings.pt"))
    chrom_boundaries = torch.load(os.path.join(data_dir, "chrom_boundaries.pt"))
    print(f"Genes: {gene_emb.shape[0]} x {gene_emb.shape[1]}  Segments: {len(chrom_boundaries)}")

    # Resolve n_perturbations — checkpoint takes priority for shape compatibility
    skip_pbs = "--skip-pbs" in sys.argv
    n_pert = None
    pbs_ckpt = os.path.join(ROOT, "checkpoints", "hayat_pbs.pt")
    if skip_pbs and os.path.exists(pbs_ckpt):
        n_pert = torch.load(pbs_ckpt, map_location="cpu", weights_only=True)["pert_emb.weight"].shape[0] - 1  # padding_idx=0
    if n_pert is None:
        n_pert = torch.load(os.path.join(data_dir, "pert_labels.pt"), weights_only=True)["n_perturbations"]

    model = Hayat(chrom_boundaries=chrom_boundaries,
                  gene_emb_dim=gene_emb.shape[1],
                  n_perturbations=n_pert).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    if skip_pbs and os.path.exists(pbs_ckpt):
        model.load_state_dict(torch.load(pbs_ckpt, map_location=device, weights_only=True))
        print(f"[PBS] skipped — loaded {pbs_ckpt} (n_pert={n_pert})")
    else:
        if skip_pbs:
            print(f"[PBS] checkpoint not found at {pbs_ckpt}, running PBS...")

        opt = optim.AdamW(model.parameters(), lr=config["learning_rate"],
                          weight_decay=config["weight_decay"])
        pbs_train = DenseDataset(os.path.join(data_dir, "dense_pbs.pt"),
                                 os.path.join(data_dir, "dense_pbs_pert.pt"))
        pbs_val = DenseDataset(os.path.join(data_dir, "dense_pbs_val.pt"),
                               os.path.join(data_dir, "dense_pbs_val_pert.pt"))
        print(f"PBS train: {len(pbs_train):,}  PBS val: {len(pbs_val):,}")
        train_pbs(model, pbs_train, pbs_val, config, device, opt, gene_emb)
        del pbs_train, pbs_val

    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    # Stage 2: Delta — load train into RAM as float16 (73GB fits in 128GB unified)
    print("Loading train into RAM (float16, ~73GB)...")
    train_ds = DenseDataset(os.path.join(data_dir, "dense_train.pt"),
                            os.path.join(data_dir, "dense_train_pert.pt"), mmap=False)
    val_ds = DenseDataset(os.path.join(data_dir, "dense_val.pt"),
                          os.path.join(data_dir, "dense_val_pert.pt"))
    print(f"Delta train: {len(train_ds):,}  val: {len(val_ds):,}")
    train_delta(model, train_ds, val_ds, config, device, gene_emb)

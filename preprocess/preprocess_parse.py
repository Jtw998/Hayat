#!/usr/bin/env python3
"""
Preprocess Parse 10M PBMC 90-cytokine dataset for Hayat streaming training.
Only computes metadata — expression stays in h5ad (backed mode).

Usage:
  python preprocess_parse.py --input-h5ad data/20260203_Parse_10M_PBMC_cytokines.h5ad
  python preprocess_parse.py --input-h5ad PATH --hvg-sample 200000 --n-hvg 5000
"""

import argparse, os, json, numpy as np, pandas as pd, torch
import scanpy as sc
from pathlib import Path

DEFAULT_H5AD = "data/20260203_Parse_10M_PBMC_cytokines.h5ad"
SCGPT_DIR = "/Users/jw/scgpt-embedding"
OUTPUT_DIR = "data"


# ── Step 1: Gene coords ──

def match_gene_coords(gene_list, output_dir):
    import mygene
    mg = mygene.MyGeneInfo()
    res = mg.querymany(gene_list, scopes="symbol", species="human",
                       fields="genomic_pos.chr,genomic_pos.start", returnall=False)
    gene_meta = []
    for entry in res:
        gene = entry["query"]
        if "genomic_pos" in entry:
            pos = entry["genomic_pos"]
            pos = pos[0] if isinstance(pos, list) else pos
            if pos.get("chr") and pos.get("start"):
                gene_meta.append({"gene_name": gene, "chr": str(pos["chr"]), "start": int(pos["start"])})
    df = pd.DataFrame(gene_meta)
    print(f"Coords matched: {len(df)} / {len(gene_list)} genes")

    chr_order = {str(i): i for i in range(1, 23)}
    chr_order.update({"X": 23, "Y": 24})
    df["chr_order"] = df["chr"].map(lambda x: chr_order.get(x, 999))
    df = df.sort_values(by=["chr_order", "start"]).drop("chr_order", axis=1).reset_index(drop=True)
    df.to_csv(f"{output_dir}/gene_meta.csv", index=False)

    chrom_boundaries = []
    current_chr, start_idx = df["chr"].iloc[0], 0
    for i, ch in enumerate(df["chr"]):
        if ch != current_chr:
            chrom_boundaries.append((start_idx, i))
            start_idx, current_chr = i, ch
    chrom_boundaries.append((start_idx, len(df)))
    torch.save(chrom_boundaries, f"{output_dir}/chrom_boundaries.pt")
    print(f"  {len(chrom_boundaries)} chromosome blocks")
    return df


# ── Step 2: HVG on full data ──

def compute_hvg(h5ad_path, output_dir, n_hvg=5000, hvg_sample=200000, seed=42):
    """
    Compute HVG from a stratified sample of the full dataset.
    Saves: hvg_mask.pt (bool [G_matched]), hvg_gene_names.json
    """
    print(f"Loading full obs from backed h5ad...")
    adata = sc.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs
    n_cells = adata.n_obs

    # Stratified sample
    cytokine_col = "cytokine"
    cytokines = sorted(obs[cytokine_col].unique())
    n_per_cyto = max(100, hvg_sample // len(cytokines))
    print(f"Sampling ~{n_per_cyto} cells per cytokine x {len(cytokines)} cytokines")

    rng = np.random.default_rng(seed)
    indices = []
    for cyto in cytokines:
        mask = np.where(obs[cytokine_col].values == cyto)[0]
        n_take = min(n_per_cyto, len(mask))
        indices.extend(rng.choice(mask, n_take, replace=False).tolist())

    print(f"Loading {len(indices):,} cells for HVG computation...")
    subset = adata[indices, :].to_memory()
    print(f"Loaded: {subset.n_obs} cells x {subset.n_vars} genes")

    # Compute HVG on raw counts
    sc.pp.normalize_total(subset, target_sum=10000)
    sc.pp.log1p(subset)
    sc.pp.highly_variable_genes(subset, n_top_genes=n_hvg, flavor="seurat_v3")

    hvg_mask = torch.tensor(subset.var["highly_variable"].values, dtype=torch.bool)
    hvg_names = list(subset.var_names[hvg_mask.numpy()])
    torch.save(hvg_mask, f"{output_dir}/hvg_mask_full.pt")
    with open(f"{output_dir}/hvg_gene_names.json", "w") as f:
        json.dump(hvg_names, f)
    print(f"HVG: {len(hvg_names)} genes selected from {subset.n_vars} total")
    return hvg_mask, hvg_names


# ── Step 3: scGPT embeddings (HVG-filtered) ──

def match_scGPT(gene_meta_df, hvg_mask, scgpt_dir, output_dir):
    """Filter gene_meta to HVG, extract scGPT embeddings."""
    gene_meta_hvg = gene_meta_df[hvg_mask.numpy()].reset_index(drop=True)
    gene_meta_hvg.to_csv(f"{output_dir}/gene_meta.csv", index=False)

    # Recompute chrom boundaries for HVG
    chrom_boundaries = []
    current_chr, start_idx = gene_meta_hvg["chr"].iloc[0], 0
    for i, ch in enumerate(gene_meta_hvg["chr"]):
        if ch != current_chr:
            chrom_boundaries.append((start_idx, i))
            start_idx, current_chr = i, ch
    chrom_boundaries.append((start_idx, len(gene_meta_hvg)))
    torch.save(chrom_boundaries, f"{output_dir}/chrom_boundaries.pt")

    scgpt_dir = Path(scgpt_dir)
    gene_to_idx = json.load(open(scgpt_dir / "vocab.json"))
    state = torch.load(scgpt_dir / "best_model.pt", map_location="cpu")
    emb_layer = state["encoder.embedding.weight"]
    emb_dim = emb_layer.shape[1]

    target_genes = gene_meta_hvg["gene_name"].tolist()
    matched_emb = torch.zeros(len(target_genes), emb_dim, dtype=torch.float32)
    matched = 0
    for i, g in enumerate(target_genes):
        if g in gene_to_idx:
            matched_emb[i] = emb_layer[gene_to_idx[g]]
            matched += 1

    print(f"scGPT matched: {matched}/{len(target_genes)} ({100*matched/len(target_genes):.1f}%)")
    torch.save(matched_emb, f"{output_dir}/gene_embeddings.pt")

    # Also save HVG indices mapping full_gene → hv_gene for streaming
    hv_genes = target_genes
    return hv_genes


# ── Step 4: Perturbation labels (full dataset, split masks) ──

def build_pert_labels_full(h5ad_path, output_dir):
    """
    Read all obs from backed h5ad, build perturbation labels
    for entire dataset with train/val split masks.
    """
    adata = sc.read_h5ad(h5ad_path, backed="r")
    obs = adata.obs
    n_cells = adata.n_obs

    # Perturbation → index
    pert_list = sorted([c for c in obs["cytokine"].unique() if str(c).upper() != "PBS"])
    pert_list = ["control"] + pert_list
    pert_to_idx = {c: i for i, c in enumerate(pert_list)}
    pert_id = torch.tensor([pert_to_idx.get(c, 0) for c in obs["cytokine"]], dtype=torch.long)

    is_pert = torch.tensor([0 if str(t).upper() == "PBS" else 1 for t in obs["treatment"]], dtype=torch.long)
    pert_type = is_pert.clone()

    # Stratified split: 80/20 per cytokine
    rng = np.random.default_rng(42)
    train_cells = []
    val_cells = []
    for cyto in pert_list:
        mask = np.where(obs["cytokine"].values == cyto)[0]
        split = int(len(mask) * 0.8)
        shuffled = rng.permutation(mask)
        train_cells.append(shuffled[:split])
        val_cells.append(shuffled[split:])

    train_idx = np.sort(np.concatenate(train_cells))
    val_idx = np.sort(np.concatenate(val_cells))

    train_mask = torch.zeros(n_cells, dtype=torch.bool)
    val_mask = torch.zeros(n_cells, dtype=torch.bool)
    train_mask[torch.tensor(train_idx)] = True
    val_mask[torch.tensor(val_idx)] = True

    pert_dict = {
        "perturbation_id": pert_id,
        "perturbation_type": pert_type,
        "is_perturbed": is_pert,
        "train_mask": train_mask,
        "val_mask": val_mask,
        "n_perturbations": len(pert_list),
        "perturbation_names": pert_list,
        "n_perturbation_types": 2,
    }
    torch.save(pert_dict, f"{output_dir}/pert_labels.pt")

    n_train = train_mask.sum().item()
    n_val = val_mask.sum().item()
    n_pert = is_pert.sum().item()
    print(f"Pert labels: {n_cells:,} cells, {len(pert_list)} perturbations")
    print(f"  Train={n_train:,}  Val={n_val:,}  Control={n_cells-n_pert:,}  Perturbed={n_pert:,}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--hvg-sample", type=int, default=200000)
    parser.add_argument("--n-hvg", type=int, default=5000)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--scgpt-dir", default=SCGPT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    # Step 1: Gene coordinates (from backed h5ad)
    print("=" * 60)
    print("Step 1: Gene coordinate matching (mygene)")
    print("=" * 60)
    adata = sc.read_h5ad(args.input_h5ad, backed="r")
    all_genes = list(adata.var_names)
    print(f"Full dataset: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    gene_meta_df = match_gene_coords(all_genes, args.output_dir)
    print()

    # Step 2: HVG on a representative sample
    print("=" * 60)
    print("Step 2: HVG selection")
    print("=" * 60)
    hvg_mask, hvg_names = compute_hvg(args.input_h5ad, args.output_dir,
                                      n_hvg=args.n_hvg, hvg_sample=args.hvg_sample)
    print()

    # Step 3: scGPT embeddings (HVG only)
    print("=" * 60)
    print("Step 3: scGPT gene embeddings (HVG subset)")
    print("=" * 60)
    match_scGPT(gene_meta_df, hvg_mask, args.scgpt_dir, args.output_dir)
    print()

    # Step 4: Perturbation labels (full dataset)
    print("=" * 60)
    print("Step 4: Perturbation labels (full dataset)")
    print("=" * 60)
    build_pert_labels_full(args.input_h5ad, args.output_dir)
    print()

    print("=" * 60)
    print("Done. Outputs:")
    print("=" * 60)
    for f in sorted(os.listdir(args.output_dir)):
        if not f.endswith(".h5ad") and not f.endswith(".zip") and f != ".DS_Store":
            path = f"{args.output_dir}/{f}"
            size = os.path.getsize(path)
            if size > 1e9:
                tag = f"{size/1e9:.1f} GB"
            elif size > 1e6:
                tag = f"{size/1e6:.1f} MB"
            else:
                tag = f"{size/1024:.1f} KB"
            print(f"  {f:30s} {tag}")


if __name__ == "__main__":
    main()

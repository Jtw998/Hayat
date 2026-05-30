#!/usr/bin/env python3
"""Pre-extract full train dense float16 matrix (~68GB, fits in 128GB RAM)."""
import h5py, numpy as np, torch, pandas as pd, os, sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_dir = os.environ.get("GENE_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
h5ad_path = os.path.join(data_dir, "20260203_Parse_10M_PBMC_cytokines.h5ad")
pert_file = os.path.join(data_dir, "pert_labels.pt")
gene_meta = os.path.join(data_dir, "gene_meta.csv")

gm = pd.read_csv(gene_meta)
hv_names = list(gm["gene_name"])
n_hv = len(hv_names)
name_to_hv = {g: i for i, g in enumerate(hv_names)}

pert_data = torch.load(pert_file)
mask = pert_data["train_mask"]
cell_idx = torch.where(mask)[0].numpy()
n_cells = len(cell_idx)
print(f"[train] {n_cells:,} cells × {n_hv} genes = {n_cells * n_hv * 2 / 1e9:.1f} GB float16")

f = h5py.File(h5ad_path, "r")
indptr = f["X/indptr"]
indices = f["X/indices"]
data_arr = f["X/data"]
all_names = [g.decode() if isinstance(g, bytes) else g for g in f["var/_index"][:]]
col_map = np.array([name_to_hv.get(g, -1) for g in all_names], dtype=np.int32)

chunk_size = 2000
vals = np.zeros((n_cells, n_hv), dtype=np.float16)

for start in tqdm(range(0, n_cells, chunk_size), desc="Extract train", unit="chunk"):
    end = min(start + chunk_size, n_cells)
    chunk_idx = cell_idx[start:end]
    chunk_vals = np.zeros((end - start, n_hv), dtype=np.float32)
    for i, gi in enumerate(chunk_idx):
        s, e = int(indptr[gi]), int(indptr[gi + 1])
        if s == e:
            continue
        hv_pos = col_map[indices[s:e]]
        keep = hv_pos >= 0
        if keep.any():
            chunk_vals[i, hv_pos[keep]] = data_arr[s:e][keep]
    totals = chunk_vals.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    chunk_vals = np.log1p(chunk_vals / totals * 10000)
    vals[start:end] = chunk_vals.astype(np.float16)

f.close()

out_x = os.path.join(data_dir, "dense_train.pt")
torch.save(torch.from_numpy(vals), out_x)
print(f"Saved {out_x} ({os.path.getsize(out_x)/1e9:.2f} GB)")

# Also save pert info
pert_id = pert_data["perturbation_id"][cell_idx]
pert_type = pert_data["perturbation_type"][cell_idx]
is_pert = pert_data["is_perturbed"][cell_idx]
out_p = os.path.join(data_dir, "dense_train_pert.pt")
torch.save({"perturbation_id": pert_id, "perturbation_type": pert_type, "is_perturbed": is_pert}, out_p)
print(f"Saved {out_p}")

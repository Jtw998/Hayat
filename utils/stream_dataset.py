"""Fast data loaders: dense in-memory (PBS/val) + chunked h5py (train delta)."""

import torch, numpy as np, h5py, pandas as pd, os
from tqdm import tqdm


class DenseDataset:
    """Dense float16 matrix — large files memory-mapped, small ones in RAM."""

    def __init__(self, dense_path, pert_path, pbs_only=False, mmap=False):
        self._mmap = mmap
        self.x = torch.load(dense_path, mmap=mmap, weights_only=True)  # [N, G] float16
        pert = torch.load(pert_path, weights_only=True)
        self.pert_id = pert["perturbation_id"]
        self.pert_type = pert["perturbation_type"]
        self.is_perturbed = pert["is_perturbed"]

        if pbs_only:
            mask = self.is_perturbed == 0
            self.x = self.x[mask]
            self.pert_id = self.pert_id[mask]
            self.pert_type = self.pert_type[mask]
            self.is_perturbed = self.is_perturbed[mask]

        self.n_cells = len(self.x)
        self.n_genes = self.x.shape[1]

    def __len__(self):
        return self.n_cells


class ChunkedReader:
    """Reads large chunks from h5py, caches in memory, serves sub-batches."""

    def __init__(self, h5ad_path, pert_dict, gene_meta_path, split="train",
                 target_sum=10000, seed=42, chunk_size=4096):
        self.target_sum = target_sum
        self.chunk_size = chunk_size
        self.h5ad_path = h5ad_path

        pert = torch.load(pert_dict) if isinstance(pert_dict, str) else pert_dict
        mask = pert[f"{split}_mask"]
        cell_idx = torch.where(mask)[0]
        if split == "train":
            rng = torch.Generator().manual_seed(seed)
            cell_idx = cell_idx[torch.randperm(len(cell_idx), generator=rng)]
        self.cell_idx = cell_idx.numpy()
        self.pert_id = pert["perturbation_id"][cell_idx]
        self.pert_type = pert["perturbation_type"][cell_idx]
        self.is_perturbed = pert["is_perturbed"][cell_idx]
        self.n_cells = len(cell_idx)
        self.n_perturbations = pert["n_perturbations"]
        print(f"[{split}] {self.n_cells:,} cells (chunked, chunk={chunk_size})")

        gm = pd.read_csv(gene_meta_path)
        hv_names = list(gm["gene_name"])
        self.n_genes = len(hv_names)
        name_to_hv = {g: i for i, g in enumerate(hv_names)}

        self.f = h5py.File(h5ad_path, "r")
        self.indptr = self.f["X/indptr"]
        self.indices = self.f["X/indices"]
        self.data_arr = self.f["X/data"]

        all_names = [g.decode() if isinstance(g, bytes) else g
                     for g in self.f["var/_index"][:]]
        self._col_map = np.array(
            [name_to_hv.get(g, -1) for g in all_names], dtype=np.int32
        )

    def _read_chunk(self, global_idx):
        B = len(global_idx)
        vals = np.zeros((B, self.n_genes), dtype=np.float32)
        for i, gi in enumerate(global_idx):
            s, e = int(self.indptr[gi]), int(self.indptr[gi + 1])
            if s == e:
                continue
            hv_pos = self._col_map[self.indices[s:e]]
            keep = hv_pos >= 0
            if keep.any():
                vals[i, hv_pos[keep]] = self.data_arr[s:e][keep]
        totals = vals.sum(axis=1, keepdims=True)
        totals[totals == 0] = 1
        vals = np.log1p(vals / totals * self.target_sum)
        return torch.from_numpy(vals.astype(np.float32))

    def __len__(self):
        return self.n_cells

    def close(self):
        self.f.close()


def dense_loader(dataset, batch_size, shuffle=True, seed=42, chunk_cells=2_000_000):
    """Data loader — chunked sequential mmap for large files, direct for in-RAM."""
    import numpy as np

    n = len(dataset)
    bs = batch_size

    if not dataset._mmap or n <= chunk_cells:
        # In-RAM / small dataset: direct indexing
        idx = (torch.randperm(n, generator=torch.Generator().manual_seed(seed))
               if shuffle else torch.arange(n))
        for start in range(0, n, bs):
            batch_idx = idx[start:start + bs]
            x = dataset.x[batch_idx]
            x = x.clone().float() if dataset._mmap else x.float()
            yield x, {
                'perturbation_id': dataset.pert_id[batch_idx].long(),
                'perturbation_type': dataset.pert_type[batch_idx].long(),
                'is_perturbed': dataset.is_perturbed[batch_idx].long(),
            }
        return

    # Large mmap: read big contiguous chunks (float16 to save RAM), serve from RAM
    rng = np.random.RandomState(seed)
    n_chunks = (n + chunk_cells - 1) // chunk_cells
    chunk_order = rng.permutation(n_chunks) if shuffle else np.arange(n_chunks)

    for ci in chunk_order:
        c_start = int(ci) * chunk_cells
        c_end = min(c_start + chunk_cells, n)
        c_size = c_end - c_start

        # Contiguous mmap read → RAM as float16 (half the memory)
        x_chunk = dataset.x[c_start:c_end].clone()

        # Shuffle within chunk, serve batches
        local_idx = (torch.randperm(c_size, generator=torch.Generator().manual_seed(seed + int(ci) + 1))
                     if shuffle else torch.arange(c_size))
        for bstart in range(0, c_size, bs):
            bend = min(bstart + bs, c_size)
            bi = local_idx[bstart:bend]
            gi = bi + c_start
            yield x_chunk[bi].float(), {
                'perturbation_id': dataset.pert_id[gi].long(),
                'perturbation_type': dataset.pert_type[gi].long(),
                'is_perturbed': dataset.is_perturbed[gi].long(),
            }


def chunked_loader(reader, batch_size, shuffle=True, seed=42):
    """Read big chunks from h5py → cache → serve small batches from memory."""
    bs = batch_size
    cs = reader.chunk_size
    n_cells = reader.n_cells
    order = (np.random.RandomState(seed).permutation(n_cells)
             if shuffle else np.arange(n_cells))

    for chunk_start in range(0, n_cells, cs):
        chunk_end = min(chunk_start + cs, n_cells)
        chunk_order = order[chunk_start:chunk_end]

        # Read one huge chunk (amortizes Python/h5py overhead)
        x_cache = reader._read_chunk(reader.cell_idx[chunk_order])

        # Serve batches from cache
        for bstart in range(0, chunk_end - chunk_start, bs):
            bend = min(bstart + bs, chunk_end - chunk_start)
            idx = chunk_order[bstart:bend]
            yield x_cache[bstart:bend], {
                'perturbation_id': reader.pert_id[torch.tensor(idx)].long(),
                'perturbation_type': reader.pert_type[torch.tensor(idx)].long(),
                'is_perturbed': reader.is_perturbed[torch.tensor(idx)].long(),
            }


def n_batches(n_cells, batch_size):
    return (n_cells + batch_size - 1) // batch_size

#!/usr/bin/env python3
"""
Pre-compute scGPT cell embeddings for Hayat.
Uses scGPT Transformer to encode each cell's (gene, expression) pairs → 512-dim vector.

Usage:
  python compute_cell_embeddings.py \
    --data ../data/processed_data_100k.pt \
    --out ../data/cell_embeddings.pt \
    --scgpt /Users/jw/scgpt-embedding
"""
import os, sys, argparse, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm


# ── scGPT model (minimal copy from scgpt_embedding.py) ──

class GeneEncoder(nn.Module):
    def __init__(self, ntoken, dim, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(ntoken, dim, padding_idx=padding_idx)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        return self.norm(self.embedding(x))


class ContinuousValueEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_value=512):
        super().__init__()
        self.linear1 = nn.Linear(1, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_value = max_value
    def forward(self, x):
        x = torch.clamp(x.unsqueeze(-1), max=self.max_value)
        return self.dropout(self.norm(self.linear2(F.relu(self.linear1(x)))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))
    def forward(self, x):
        return self.dropout(x + self.pe[:x.size(0)])


class TransformerModel(nn.Module):
    def __init__(self, ntoken, d_model=512, nhead=8, d_hid=2048, nlayers=12, dropout=0.2, pad_token_id=0):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.gene_encoder = GeneEncoder(ntoken, d_model, padding_idx=pad_token_id)
        self.value_encoder = ContinuousValueEncoder(d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.norm = nn.LayerNorm(d_model)

    def _encode(self, gene_ids, values, padding_mask=None):
        if padding_mask is None:
            padding_mask = (gene_ids == self.pad_token_id)
        x = self.gene_encoder(gene_ids) + self.value_encoder(values)
        x = self.pos_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        return self.norm(x)

    def forward(self, gene_ids, values):
        padding_mask = (gene_ids == self.pad_token_id)
        output = self._encode(gene_ids, values, padding_mask)
        return output[:, 0, :]  # CLS token → cell embedding


# ── Dataset ──

class CellDataset(Dataset):
    def __init__(self, expressions, gene_names, vocab, max_seq_len=1200):
        self.expressions = expressions  # [N, G]
        self.gene_names = gene_names    # [G]
        # Map gene names to vocab indices
        gene_ids = []
        self.valid_mask = []
        for g in gene_names:
            gid = vocab.get(g, -1)
            gene_ids.append(gid)
            self.valid_mask.append(gid >= 0)
        self.gene_ids = torch.tensor(gene_ids, dtype=torch.long)
        self.valid_mask = torch.tensor(self.valid_mask, dtype=torch.bool)
        self.cls_id = vocab.get("<cls>", 0)
        self.pad_id = vocab.get("<pad>", 0)
        self.max_seq_len = max_seq_len

        n_valid = self.valid_mask.sum().item()
        print(f"Genes in vocab: {n_valid}/{len(gene_names)}")

    def __len__(self):
        return len(self.expressions)

    def __getitem__(self, idx):
        expr = self.expressions[idx]
        if isinstance(expr, torch.Tensor):
            expr = expr.numpy()
        valid_expr = expr[self.valid_mask.numpy()]
        valid_gene_ids = self.gene_ids[self.valid_mask]

        selected_ids = valid_gene_ids
        selected_expr = torch.from_numpy(valid_expr.astype(np.float32))

        n = len(selected_ids)
        if n > self.max_seq_len - 1:
            perm = torch.randperm(n)[:self.max_seq_len - 1]
            selected_ids = selected_ids[perm]
            selected_expr = selected_expr[perm]

        genes = torch.cat([torch.tensor([self.cls_id]), selected_ids])
        exprs = torch.cat([torch.zeros(1), selected_expr])
        return genes.long(), exprs.float()


def collate_fn(batch, pad_id=0):
    max_len = max(len(b[0]) for b in batch)
    gene_batch, expr_batch = [], []
    for genes, exprs in batch:
        pad_len = max_len - len(genes)
        gene_batch.append(torch.cat([genes, torch.full((pad_len,), pad_id, dtype=torch.long)]))
        expr_batch.append(torch.cat([exprs, torch.zeros(pad_len)]))
    return torch.stack(gene_batch), torch.stack(expr_batch)


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to processed_data.pt")
    parser.add_argument("--out", required=True, help="Output path for cell_embeddings.pt")
    parser.add_argument("--scgpt", default="/Users/jw/scgpt-embedding")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-cells", type=int, default=0)
    args = parser.parse_args()

    # Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load scGPT
    scgpt_dir = Path(args.scgpt)
    with open(scgpt_dir / "args.json") as f:
        cfg = json.load(f)
    with open(scgpt_dir / "vocab.json") as f:
        vocab = json.load(f)

    pad_id = vocab.get("<pad>", 0)
    model = TransformerModel(
        ntoken=len(vocab), d_model=cfg.get("embsize", 512),
        nhead=cfg.get("nhead", 8), d_hid=cfg.get("d_hid", 2048),
        nlayers=cfg.get("nlayers", 12), dropout=cfg.get("dropout", 0.2),
        pad_token_id=pad_id,
    )

    state = torch.load(scgpt_dir / "best_model.pt", map_location=device)
    state = {k.replace("encoder.", "gene_encoder."): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith("decoder.")}
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Loaded scGPT: {cfg.get('nlayers', 12)} layers, {cfg.get('embsize', 512)} dim")

    # Load expression data
    data = torch.load(args.data)
    if "train" in data:
        train_data = data["train"]
        val_data = data["val"]
        if args.max_cells > 0:
            train_data = train_data[:args.max_cells]
            val_data = val_data[:max(args.max_cells // 5, 200)]
        all_expr = torch.cat([train_data, val_data], dim=0)
        n_train = train_data.shape[0]
    else:
        all_expr = data["expression"]
        n_train = int(all_expr.shape[0] * 0.8)

    if isinstance(all_expr, torch.Tensor):
        all_expr = all_expr.numpy()
    print(f"Data: {all_expr.shape[0]} cells × {all_expr.shape[1]} genes")

    # Gene names — try data dict first, then gene_meta.csv
    gene_names = None
    if "gene_names" in data and data["gene_names"] is not None:
        gene_names = data["gene_names"]
    else:
        meta_path = args.data.rsplit("/", 1)[0] + "/gene_meta.csv"
        if os.path.exists(meta_path):
            import pandas as pd
            gene_names = pd.read_csv(meta_path)["gene_name"].tolist()
            print(f"Loaded {len(gene_names)} gene names from {meta_path}")
    if gene_names is None:
        gene_names = [f"gene_{i}" for i in range(all_expr.shape[1])]

    # Compute embeddings
    dataset = CellDataset(all_expr, gene_names, vocab)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=SequentialSampler(dataset),
        collate_fn=lambda b: collate_fn(b, pad_id), drop_last=False,
        num_workers=0, pin_memory=False,
    )

    emb_dim = cfg.get("embsize", 512)
    embeddings = np.zeros((len(dataset), emb_dim), dtype=np.float32)

    idx = 0
    for gene_batch, expr_batch in tqdm(loader, desc="Computing cell embeddings"):
        gene_batch = gene_batch.to(device)
        expr_batch = expr_batch.to(device)
        with torch.no_grad():
            cell_emb = model(gene_batch, expr_batch)
        n = len(cell_emb)
        embeddings[idx:idx+n] = cell_emb.cpu().numpy()
        idx += n

    # L2 normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    # Save
    torch.save({
        "train_emb": torch.from_numpy(embeddings[:n_train]),
        "val_emb": torch.from_numpy(embeddings[n_train:]),
        "embed_dim": emb_dim,
    }, args.out)
    print(f"Saved {len(embeddings)} cell embeddings → {args.out}")
    print(f"Train: {n_train}, Val: {len(embeddings) - n_train}")


if __name__ == "__main__":
    main()

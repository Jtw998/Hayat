# Hayat: Open-Then-Express Structural Causal Model for Single-Cell Transcriptomics

Gene-set agnostic, no external prior. Learns a structured cis/trans decomposition from expression data alone.

```
Gate:   o = GumbelSigmoid(α + Γ·u)      "can this gene express?"
Drive:  r = softplus(β + W·z + Λ·c)     "if open, how much?"
Output: μ = ℓ · (ε+(1-ε)·o) · r         closed gate → expression ≈ 0
```

---

## Quick Start

### 1. Install

```bash
pip install torch einops pandas
```

### 2. Prepare gene embeddings (required)

scGPT gene token embeddings are needed for the hypernetwork that generates per-gene parameters:
```bash
python preprocess/match_gene_embeddings.py --scgpt_dir /path/to/scgpt-embedding/
```

### 3. Pre-compute cell embeddings (recommended)

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python preprocess/compute_cell_embeddings.py \
  --data data/processed_data_100k.pt \
  --out data/cell_embeddings.pt \
  --scgpt /path/to/scgpt-embedding
```

### 4. Train

```bash
python train.py --data_dir data
```

Expected output: ~44 it/s on MPS, ~250K params, 0.9 MB.

---

## Architecture

```
              expression x [B, G]
                    │
    ┌───────────────┼───────────────┐
    │               │               │
 segment pool   segment pool   cell_emb [B,512]
 [B, S]         [B, S]         (pre-computed scGPT)
    │               │               │
 MLP_u           MLP_z           Linear
    │               │               │
 u [B,S,8]      z [B,16]        c [B,16]
 cis state       trans programs   cell state
    │               │               │
    └───────┬───────┘               │
            │                       │
   ╔════════╧═══════════════╗  ╔════╧══════════════╗
   ║  GATE (permission)     ║  ║  DRIVE (strength) ║
   ║  o = GumbelSigmoid(    ║  ║  r = softplus(    ║
   ║    α + Γ · u_s(g)      ║  ║    β + W·z + Λ·c  ║
   ║  )                     ║  ║  )                 ║
   ╚════════════╤═══════════╝  ╚════════╤═══════════╝
                │                        │
                └────────┬───────────────┘
                         │
                  μ = ℓ · (ε+(1-ε)o) · r
                  x̂ ~ NB(μ, θ)
```

Per-gene parameters (α, β, Γ, W, Λ, θ) are **generated from scGPT gene embeddings** via small shared hypernetworks. The model has no fixed gene set — feed different gene embeddings, get different parameters.

### Forbidden edges (core model claims)

| Edge | Status | Claim |
|------|--------|-------|
| z → gate | forbidden | Trans programs don't control chromatin openness |
| u → drive | forbidden | Cis openness only controls permission |
| c → gate | forbidden | Cell state doesn't bypass cis |
| gene A → gene B | forbidden | Gene-gene dependence mediated by shared latents |

### Ablation modes

```python
model.forward(x, gene_emb, ablation="no_gate")    # drive only (o≡1)
model.forward(x, gene_emb, ablation="no_drive")   # gate only (r≡1)
model.forward(x, gene_emb, ablation="additive")   # o + r (gate bypassable)
```

---

## Training

3-phase annealing:

| Phase | Epochs | ε (gate floor) | τ (gate temperature) |
|-------|--------|----------------|----------------------|
| Warmup | 1–10 | 0.5 → 0.1 | 1.0 |
| Hardening | 11–30 | 0.1 → 0.01 | 1.0 → 0.5 |
| Stable | 31+ | 0.01 | 0.5 |

Loss: NB reconstruction + W sparsity (L1) + gate bimodality + gate-rate prior + program decorrelation.

---

## Data Requirements

Per dataset directory:
- `processed_data.pt` — dict with `train`/`val` tensors [cells, genes]
- `chrom_boundaries.pt` — list of `(start, end)` tuples per chromosome block
- `gene_embeddings.pt` — scGPT gene token embeddings [genes, 512] **(required)**
- `cell_embeddings.pt` — pre-computed scGPT cell embeddings [cells, 512] (optional, use `compute_cell_embeddings.py`)
- `gene_meta.csv` — gene names, chromosome, position

---

## Hyperparameters

In `utils/losses.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `learning_rate` | 1e-4 | AdamW |
| `weight_decay` | 1e-5 | AdamW |
| `batch_size` | 16 | Training batch size |
| `num_epochs` | 100 | Training epochs |
| `max_train_cells` | 0 | Cap on cells (0 = all) |
| `gate_rate_target` | 0.15 | Target mean open rate |

Model constructor: `d_u=8` (cis state dim), `K=16` (trans programs), `d_c=16` (cell state dim), `hyper_hidden=64`.

---

## Perturbation Evaluation (Schmidt)

```bash
cd Schmidt
python evaluate.py --checkpoint ../checkpoints/hayat_.._data.pt
```

Six metrics: MSE, E-distance, PCC-delta, Wasserstein, KL-divergence, Common-DEGs.

---

## Project Structure

```
Hayat/
├── train.py                              # Entry point
├── models/model.py                       # Hayat SCM
├── train/run_training.py                 # Training loop + annealing
├── utils/losses.py                       # NB loss, regularizers, metrics
├── preprocess/
│   ├── preprocess_data.py                # h5ad → .pt
│   ├── match_gene_embeddings.py          # scGPT gene embedding matching
│   ├── generate_chrom_boundaries.py      # Chromosome block boundaries
│   └── compute_cell_embeddings.py        # scGPT cell embedding pre-computation
├── Schmidt/
│   ├── preprocess.py                     # Schmidt dataset prep
│   └── evaluate.py                       # 6-metric evaluation
├── analysis/
│   └── RESEARCH_PLAN.md                  # 4-study research plan
├── docs/
│   └── ARCHITECTURE_v2.md                # SCM specification
├── data/                                 # Preprocessed data
└── checkpoints/                          # Saved models
```

---

## FAQ

### Gene count changes between datasets

Use the same scGPT gene embeddings file. The hypernetwork generates per-gene parameters from embeddings — different gene sets just need corresponding embeddings. No model architecture changes.

### No scGPT embeddings

Gene embeddings are required. Run `match_gene_embeddings.py` or provide your own [G, D] embedding tensor.

### Out of memory

Reduce `batch_size` or `max_train_cells`. Model is only 0.9 MB — memory use is dominated by data.

### MPS fallback

The scGPT cell embedding pre-computation needs `PYTORCH_ENABLE_MPS_FALLBACK=1` on Apple Silicon. Training does not.

---

## License

MIT

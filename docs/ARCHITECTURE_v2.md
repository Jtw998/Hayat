# Hayat v2 — Open-Then-Express

## Layer 1: Inference Chain (encoder computation)

How latent variables are computed from input. This is NOT the causal model — just the encoding pathway.

```
                       INPUT
                 x ∈ ℝᴮˣᴳ  (B cells × G genes = 21900)
                 e_scGPT ∈ ℝᴳˣᴰ  (D = 512, frozen pretrained gene embeddings)
                       │
     ┌─────────────────┼─────────────────┐
     │                 │                 │
  Segment Pool     Gene-Level       Global Pool
  (per block)      (per gene)       (all genes)       scGPT
     │                 │                 │               │
  x_agg[s]        log(1+x_g)       log(1+x_g)        e_scGPT
  [B, S]           [B, G]           [B, G]           [G, D]
     │                 │                 │               │
  ┌──┴──┐          ┌──┴──┐          ┌──┴──────────┐    │
  │MLP_u│          │MLP_h│          │   MLP_c      │    │
  │S→du │          │1→dh │          │              │    │
  └──┬──┘          └──┬──┘          │ c_expr =     │    │
     │                 │             │  MLP(log(1+x)│    │
  u [B,S,du]      h [B,G,dh]        │  pooled)     │    │
  segment cis     per-gene          │              │    │
  openness        local ctx         │ c_sc =       │    │
     │                 │             │  proj(e_scGPT│    │
     │                 │             │  · x)        │    │
     │                 │             │              │    │
     │                 │             │ c = c_expr   │    │
     │                 │             │   + c_sc     │────┘
     │                 │             └──────┬───────┘
     │                 │                    │
     │                 │             c [B, dc]
     │                 │             global state
     │                 │                    │
     │                 │    ┌───────────────┘
     │                 │    │
     └────────┬────────┘    │
              │             │
         x_agg reshaped     │
         [B, S]             │
              │             │
         ┌────┴────┐        │
         │  MLP_z  │        │
         │  S→K    │        │
         └────┬────┘        │
              │             │
         z [B, K]           │
         trans programs     │
              │             │
              └──────┬──────┘
                     │
     ╔═══════════════╧═══════════════════════════╗
     ║          STRUCTURAL DECODER               ║
     ║  (Layer 2: Causal chain — see below)      ║
     ╚═══════════════╤═══════════════════════════╝
                     │
                   OUTPUT
                 x̂ ∈ ℝᴮˣᴳ
```

**Key design constraints on the encoder:**

| Constraint | Reason |
|------------|--------|
| z sees only segment-level aggregated signal (S=326), not per-gene | Prevents z from memorizing per-gene expression; forces it to learn global programs |
| scGPT embedding → c only, NOT gate | Preserves ATAC-like gate semantics; scGPT carries cell-type/global info, not cis-openness |
| u sees only per-segment pooled expression | Segment-level resolution matches chromatin-domain scale |
| h (per-gene MLP) is 1→4, no sequence model | Provides gene self-signal without gene-to-gene dependencies |

---

## Layer 2: Structural Causal Model (decoder + generative semantics)

This is what the model claims about biology. The encoder is just computation; these equations are the hypothesis.

### Allowed Causal Graph

```
segment cis state  u_s  ──► gate o_g ──┐
                                        ├──► expression x_g
trans programs     z    ──► drive r_g ─┘
global state       c    ──► drive r_g
library size       ℓ    ───────────────►

scGPT embedding    e_sc ──► c ─────────┘   (only through c, only into drive)
```

### Forbidden Edges (core model claims)

These edges are deliberately absent. Each one is a testable hypothesis.

| Forbidden Edge | Claim | How to Test |
|----------------|-------|-------------|
| gene A → gene B | Gene-gene dependence is mediated by shared segment state and trans programs, not direct | Ablation: add direct gene-gene edges, check if recon improves |
| z → gate | Trans programs don't control chromatin openness | Ablation: allow z→gate, check if gate semantics degrade |
| u → drive | Cis-openness only controls permission, not expression magnitude | Ablation: allow u→drive, check if gate/drive distinction blurs |
| c → gate | Global state doesn't bypass cis to open genes | Ablation: allow c→gate, check if ATAC correlation drops |
| e_scGPT → gate | Pretrained gene embeddings don't carry cis-accessibility info | Same as c→gate |

### Structural Equations

**Equation A — Gate (permission):**

```
π_ig = α_g + Γ_g · u_{i,s(g)} + γ_g · h_ig
o_ig = GumbelSigmoid(π_ig, τ)     →  o_ig ∈ [0,1]
```

- `α_g` [G]: gene's prior openness (learned)
- `Γ_g` [G, d_u]: how gene reads its segment's cis state (learned)
- `γ_g` [G, d_h]: how gene reads its own expression level (learned)
- `τ`: temperature, annealed during training (τ↓ → o → 0/1)

**Equation B — Drive (magnitude):**

```
r_ig = softplus(β_g + W_g · z_i + Λ_g · c_i)     →  r_ig ∈ [0,∞)
```

- `β_g` [G]: gene's basal expression level (learned)
- `W_g` [G, K]: gene-program sensitivity matrix — **sparse** (learned, L1-regularized)
- `Λ_g` [G, d_c]: global state effect (learned)

**Equation C — Observation (fusion):**

```
μ_ig = ℓ_i · (ε + (1-ε) · o_ig) · r_ig
x̂_ig ~ NegativeBinomial(μ_ig, θ_g)
```

- `ℓ_i`: library size (computed from input, not learned)
- `ε`: small floor (e.g. 0.01), annealed during training
- `θ_g` [G]: per-gene dispersion (learned or shared)

**Semantics of the hurdle:**

```
o ≈ 0  (closed)  →  μ ≈ ℓ · ε · r  ≈ 0     "closed → must be silent"
o ≈ 1  (open)    →  μ ≈ ℓ · r              "open → express at drive level"
```

---

## SCM Formalization

For method-section presentation:

```
Exogenous:  ε_gate, ε_drive, ε_obs

Endogenous:
  u  = f_enc_u(x)                      segment cis openness
  z  = f_enc_z(x)                      trans program activities
  c  = f_enc_c(x, e_scGPT)            global cell state
  o  = f_gate(u, h(x); α, Γ, γ)       gate (Bernoulli/Gumbel)
  r  = f_drive(z, c; β, W, Λ)         expression drive
  x̂  = f_obs(o, r, ℓ; θ, ε)           observed expression

Structural equations (the model's claims):
  o   = Gate(α + Γ·u + γ·h)           SCE-1: cis → gate
  r   = softplus(β + W·z + Λ·c)       SCE-2: trans/global → drive
  x̂   ~ NB(ℓ · (ε+(1-ε)o) · r, θ)     SCE-3: gate·drive → expression

Valid interventions:
  do(u_s ← v)     segment s openness set to v
  do(z_k ← v)     program k activity set to v
  do(c ← v)       global state set to v
  do(o_g ← 0)     gene g gate forcibly closed
  do(o_g ← 1)     gene g gate forcibly opened
```

---

## Key Shapes

| Symbol | Shape | Meaning |
|--------|-------|---------|
| `x` | [B, G] | Input expression (G = 21900) |
| `e_scGPT` | [G, 512] | Frozen scGPT gene embeddings |
| `u` | [B, S, d_u] | Segment cis-openness (S = 326, d_u = 8) |
| `h` | [B, G, d_h] | Per-gene local context (d_h = 4) |
| `z` | [B, K] | Trans programs (K = 16–32) |
| `c` | [B, d_c] | Global state (d_c = 16) |
| `π` | [B, G] | Gate logits |
| `o` | [B, G] | Open probability ∈ [0,1] |
| `r` | [B, G] | Expression drive ∈ [0,∞) |
| `μ` | [B, G] | Expected expression |

## Learned Parameters

| Param | Shape | Role |
|-------|-------|------|
| `α_g` | [G] | Gate bias (prior openness) |
| `Γ_g` | [G, d_u] | Gene → segment-state readout |
| `γ_g` | [G, d_h] | Gene → local-context readout |
| `β_g` | [G] | Drive bias (basal expression) |
| `W_g` | [G, K] | Gene → program sensitivity (sparse) |
| `Λ_g` | [G, d_c] | Gene → global-state effect |
| `θ_g` | [G] | NB dispersion |

## Loss

```
ℒ = NB(x, x̂, θ)                    reconstruction
  + λ₁ · ‖W‖₁                       sparse gene-program loading
  + λ₂ · H(o)                       gate entropy (→ 0/1)
  + λ₃ · ‖corr(z) - I‖             program decorrelation
  + λ₄ · ‖softplus(−π)‖₁            dead-gate penalty
```

## Training Schedule

| Phase | ε | τ (gate temp) | Duration | Purpose |
|-------|---|---------------|----------|---------|
| Warmup | 0.5 → 0.1 | 1.0 | epochs 1–10 | Let gate learn meaningful patterns |
| Hardening | 0.1 → 0.01 | 1.0 → 0.5 | epochs 11–30 | Sharpen gate toward 0/1 |
| Stable | 0.01 | 0.5 | epochs 31–50 | Final convergence |

## Intervention Semantics

```
do(u_s ↓)     →  o_g ↓ for all g ∈ segment s  →  μ_g ↓   (even if z strong)
do(z_k ↑)     →  r_g ↑ for genes with W_{g,k} ≠ 0  →  μ_g ↑  (only if o_g open)
do(c ↑)       →  r_g ↑ broadly  →  μ_g ↑  (but o_g unchanged)
do(o_g = 0)   →  μ_g ≈ 0  (gene silenced regardless of drive)
do(o_g = 1)   →  μ_g ≈ ℓ · r_g  (gene fully permissive)
```

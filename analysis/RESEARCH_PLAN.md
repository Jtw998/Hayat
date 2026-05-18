# Hayat 完整研究方案

## Hayat：基于 cis/trans 分治的基因表达建模及公共配对多组学交叉验证

---

## 一、研究背景与科学问题

基因表达调控本质上并不是单一层面的数值拟合问题，而是由两类不同尺度的生物机制共同决定：

1. **顺式调控（cis）** — 基因所在染色体邻域内的局部染色质上下文，包括增强子-启动子作用、局部拓扑结构、同染色体近邻依赖等；
2. **反式调控（trans）** — 少量全局调控因子，尤其是转录因子和染色质调控蛋白，通过跨染色体、跨通路传播调控信号，影响大量下游基因表达。

现有很多表达建模方法通常把所有基因统一当作等价输入 token，容易忽略两点：
- 基因组在线性染色体上的**局部结构约束**
- 少数关键 regulator 对大规模表达变化的**稀疏支配作用**

因此，本研究提出 **Hayat**，一个 Open-Then-Express 结构因果模型（SCM）：

> **基因表达 = gate × drive**
> - **gate（cis 门控）**：该基因在此细胞中是否"开放"可表达
> - **drive（trans 驱动）**：在开放的前提下，表达被推到多高
>
> 模型不引入 TF motif、GRN 先验或 ATAC 数据——所有参数从表达数据中学习，仅通过结构正则（稀疏性、gate 二值化、program 去相关）推动 cis/trans 分化。

并通过公共表达数据、公共扰动数据、公共 TF 知识库以及公共 10x Multiome PBMC 配对 RNA+ATAC 数据，对该结构假设进行系统验证。

---

## 二、核心假设

### 总体生物学假设

> 基因表达由两个可分离的机制决定：cis 控制"能否表达"（gate），trans 控制"表达多少"（drive）。gate 关闭时，trans 再强也无法驱动表达。

### 模型假设（SCM 结构方程）

- **H1（Gate 假设）**：segment-level cis state `u_s` → per-gene gate `o_g` 能学到有意义的"开放/关闭"二值化模式，且 gate 与外部 promoter accessibility 一致；
- **H2（Drive 假设）**：无先验 trans programs `z` 通过稀疏 gene loading `W` 驱动表达，能自动收敛到少数可解释的调控程序；
- **H3（Hurdle 假设）**：`o * r` 的 hurdle 融合（gate 关闭 → 表达为零）比加法/软调制更符合"可及性是必要条件"的调控逻辑；
- **H4（禁止边假设）**：模型故意禁止的因果边（z→gate, u→drive, c→gate, geneA→geneB）如果被加入会损害 gate 的 ATAC 一致性或 trans 的可解释性，说明这些禁止边是正确约束。

---

## 三、研究目标

### 目标 1：验证 Hayat 架构的必要性
- 非线性建模是否必要？
- cis/trans 分治是否必要？
- 染色体分块是否必要？

### 目标 2：验证 RegulatorGate 的生物学真实性
- 高 gate 基因是否富集已知 TF？
- 是否富集转录调控相关功能？
- 是否与独立扰动数据中的强效 perturbation gene 一致？

### 目标 3：验证模型内部机制是否得到公共多组学支持
- gate 是否与 promoter accessibility 一致？
- trans programs 是否与已知 TF regulon 一致？
- gate × drive 的交互是否与独立扰动数据中的"可及性依赖的 perturbation 效应"一致？

---

## 四、模型总体设计：Open-Then-Express SCM

### 4.1 推断链（Encoder）

| 隐变量 | 输入 | 计算 | 语义 |
|--------|------|------|------|
| `u` [B,S,d_u] | per-segment pooled log(1+x) | MLP_u | segment cis-openness |
| `h` [B,G,d_h] | per-gene log(1+x_g) | MLP_h (1→4) | gene self-signal |
| `z` [B,K] | per-segment pooled log(1+x) | MLP_z | trans programs |
| `c` [B,d_c] | global log(1+x) + scGPT embedding | MLP_c + proj(e_scGPT · x) | global cell state |

关键约束：z 只看到 segment 级粗粒度信号（S=326），不接触 per-gene 数据。

### 4.2 因果链（Decoder — 结构方程）

**Gate（许可）**：
```
π_g = α_g + Γ_g · u_{s(g)} + γ_g · h_g
o_g = GumbelSigmoid(π_g, τ)      → o_g ∈ [0,1]
```

**Drive（强度）**：
```
r_g = softplus(β_g + W_g · z + Λ_g · c)    → r_g ∈ [0,∞)
```

**Fusion（Hurdle）**：
```
μ_g = ℓ · (ε + (1-ε)· o_g) · r_g
x̂_g ~ NB(μ_g, θ_g)
```

### 4.3 禁止边（模型核心声明）

这些因果边被故意排除。每一条都是一个可检验的假设：

| 禁止边 | 声明 | 检验方式 |
|--------|------|----------|
| z → gate | 调控程序不控制染色质开放 | 加 z→gate 边，检查 gate-ATAC 相关性是否下降 |
| u → drive | 可及性只控制"能不能"，不控制"表达多少" | 加 u→drive 边，检查 gate/drive 语义是否模糊 |
| c/scGPT → gate | 全局状态不绕过 cis 直接开门 | 加 c→gate 边，检查 gate 是否失去二值化 |
| gene A → gene B | 基因间依赖由共享隐变量中介 | 加 direct edges，检查是否仅过拟合共表达 |

### 4.4 结构正则（替代先验）

```
稀疏性：  ‖W‖₁                  → 每个基因只被少数 program 驱动
门控二值化：H(o)               → gate 接近 0 或 1
去相关：  ‖corr(z) - I‖       → 不同 program 学不同东西
防死亡：  ‖softplus(−π)‖₁      → 防止所有 gate 学成 0
```


---

## 五、数据资源

### 5.1 已就绪数据

| 文件 | 大小 | 用途 |
|------|------|------|
| `data/processed_data.pt` | 55 GB | PBS 表达矩阵 train/val |
| `data/gene_embeddings.pt` | 45 MB | scGPT 基因嵌入 |
| `data/gene_meta.csv` | 413 KB | 基因染色体坐标 |
| `data/chrom_boundaries.pt` | 4.5 KB | 染色体块边界 |
| `position_table.pt` | 26 MB | Fourier 位置编码 |
| `Schmidt/schmidt_data.pt` | 1.1 GB | 扰动表达矩阵 |
| `Schmidt/schmidt_perturb_labels.pt` | — | 扰动标签 |
| `Schmidt/schmidt_gene_meta.csv` | — | Schmidt 基因坐标 |
| `Schmidt/schmidt_chrom_boundaries.pt` | — | Schmidt 染色体边界 |
| `Schmidt/schmidt_gene_embeddings.pt` | — | Schmidt 基因嵌入 |

### 5.2 外部公共数据

| 数据 | 来源 | 用途 |
|------|------|------|
| Lambert 2018 human TF list | humantfs.ccbr.utoronto.ca | TF 富集验证 |
| **10x Multiome PBMC 公共 RNA+ATAC 数据** | 10x Genomics / GEO | 配对多组学验证 |
| TRRUST / DoRothEA / ChEA（任选 1–2 个） | 公共数据库 | regulator-target 外部验证 |

### 5.3 为什么用 10x Multiome PBMC 替代 ENCODE PBMC ATAC-seq

1. 同一样本体系下的配对多组学验证更强
2. 可进行细胞类型分层，而不是 bulk 平均
3. 更适合验证 cis/trans 机制与细胞类型特异调控
4. 在不使用 fragments 的前提下，也可通过 peak-by-cell matrix 低成本完成 promoter accessibility 分析

---

## 六、数据预处理与统一框架

### 6.1 基因集统一

所有分析基于统一交集基因集：PBS 表达基因 ∩ Schmidt 数据基因 ∩ 基因 embedding 基因 ∩ 有有效染色体坐标的基因 ∩ Multiome RNA 与 ATAC 可映射的基因。输出统一基因表 `gene_table.csv`。

### 6.2 染色体排序与分块

按染色体编号和 genomic coordinate 排序，构建最终 `chrom_boundaries.pt`，所有 cis 相关分析使用同一排序。

### 6.3 10x Multiome PBMC 预处理

- **输入**：RNA expression matrix, ATAC peak-by-cell matrix, peak coordinates, barcodes
- **Promoter accessibility 定义**：TSS ± 2 kb，将与 promoter 重叠的 peaks 在每个细胞中求和 → `log1p` → library-size normalization
- **细胞层面简化**：保留 5k–10k 高质量细胞，使用 RNA 标记基因做粗细胞类型标注（T cells / B cells / NK cells / Monocytes / others），主分析采用 pseudo-bulk / cell-type average

---

## 七、计算资源与降本策略

### 7.1 分阶段训练

| 阶段 | cells | epochs | 用途 | 设备 |
|------|------|------|------|------|
| Pilot | 2k–5k | 20–30 | 跑通流程、验证趋势 | MPS / 4090 |
| Main | 10k–20k | 30–50 | 生成主结果 | 4090 / A100 |
| Enhanced | 50k+ | 50–100 | 补强结果 | A100 |

主结果以 **10k–20k cells** 为核心，不强依赖 100k cells 全量训练。大多数 study 复用同一个 Hayat 主 checkpoint。

### 7.2 降本原则

1. 只训练两个模型：Hayat 主模型 + LinearAE baseline
2. Ablation 尽量推理级实现（full / cis_only / no_blocking）
3. Attention 不保存全量 tensor，只做在线聚合，输出 `chr_flow_matrix`、`cross_chr_ratio`、`regulator_attention_mass`
4. Multiome 主分析不从 fragments 重算，只使用 processed peak matrix
5. ATAC 主分析用 pseudo-bulk，减少稀疏性

### 7.3 推荐配置

- 设备：RTX 4090 或 A100，显存 ≥ 12 GB
- batch_size = 16，mixed precision
- Pilot 0.5–1 天，Main 1–2 天，分析 1–2 天，主结果约一周

---

## 八、训练方案

### 8.1 Hayat 主模型

- 训练细胞：10k–20k
- epochs：30–50
- batch size：16
- optimizer：AdamW，lr=1e-4
- mixed precision：开启
- early stopping：patience=10
- seeds：3 个独立随机种子

输出：`checkpoints/hayat_main_seed{1,2,3}.pt`

### 8.2 LinearAE baseline

结构：N_genes → 64 → N_genes。使用与 Hayat 相同的训练数据和优化配置，作为最简非结构化线性 baseline。

---

## 九、研究内容与实验设计

---

## Study 1：主性能与 SCM 结构必要性验证

### 研究目的

验证 gate/drive 分治是否必要、hurdle 融合是否必要、禁止边约束是否正确。

### 比较模型

**主模型**：Hayat Full（gate × drive hurdle）
**Ablation**：

```
ablation="full":            o * r hurdle            (主模型)
ablation="soft_gate":       (ε + (1-ε)o) * r       (无二值化正则，gate 糊在 0.5)
ablation="additive":        o + r                   (加法融合，gate 可被 bypass)
ablation="no_gate":         r                        (纯 trans drive，无 gate)
ablation="no_drive":        o                        (纯 gate，无 trans 调制)
ablation="z_to_gate":       z → gate 边加入         (违反禁止边)
ablation="u_to_drive":      u → drive 边加入        (违反禁止边)
ablation="c_to_gate":       c → gate 边加入         (违反禁止边)
ablation="linear_ae":       LinearAE baseline       (无结构)
```

### 数据

- 训练/验证：PBS
- 外部泛化：Schmidt perturbation

### 指标

**重建指标**：Global Pearson、MSE、Intra-segment Pearson、Inter-segment Pearson
**扰动指标**：PCC-delta、Common-DEGs、Perturbation-wise MSE
**结构指标**：gate entropy（二值化程度）、W sparsity（program 稀疏度）、program-gene loading 的模块化程度
**禁止边指标**：加入禁止边后 gate-ATAC 相关性变化（Study 4 配合）

### 预期结果

- Full hurdle 在跨 segment 建模和扰动泛化上最优
- Additive 模型的重建可能不差，但 gate 语义退化（gate 失去"开关"意义）
- No-gate（纯 drive）无法关掉基因，零表达基因预测差
- No-drive（纯 gate）只能二值输出，无法建模表达强度
- 加入禁止边 → gate 二值化程度下降 / gate-ATAC 相关性下降

### 论文图

- 柱状图：各 ablation × 重建指标 + 扰动指标
- Gate entropy vs ATAC correlation（展示禁止边如何破坏 gate 语义）
- perturbation-wise scatter：Full vs key ablations

---

## Study 2：Trans Programs 生物学真实性验证

### 研究目的

验证无先验学出的 trans programs 是否具有真实的转录调控意义，gene-program loading 是否稀疏且可解释。

### 2A. Gene-Program Loading 分析

提取稀疏 loading 矩阵 `W` [G, K]，对每个 program k 取 top loading genes。分析每个 program 的基因集在染色体上的分布（是否跨染色体）、功能富集。

### 2B. Program 与已知 TF 的关联

对每个 program k，将 top loading genes 与 Lambert 2018 TF list、TRRUST / DoRothEA regulon 做富集检验。注意：这不是"program = TF"，而是"program 可能被某 TF 介导"。统计每个 program 富集到的 TF，计算 overlap p-value。

### 2C. GO / Reactome 功能注释

对每个 program 的 top 200 loading genes 做 GO/Reactome 富集。预期不同 program 对应不同生物学过程（如 cell cycle、immune response、metabolism）。

### 2D. Gate 与 Trans 的交互模式

按 gate score 将基因分为 high-gate 和 low-gate 两组，比较两组的 program loading 模式。预期：high-gate 基因的 loading 集中在特定 program（被主动调控），low-gate 基因的 loading 更均匀（仅背景表达）。

### 预期结果

- K 个 program 收敛到不同生物学功能
- 部分 program 显著富集已知 TF target
- W 矩阵稀疏（每个基因仅 2-5 个非零 loading）
- 高 gate 基因的 program loading 更集中

### 论文图

- W 矩阵 heatmap（genes × programs，按染色体排序）
- Per-program GO enrichment dotplot
- Program-TF enrichment network
- Gate score vs loading sparsity 关系图

---

## Study 3：Gate 的 ATAC 样特性验证

### 研究目的

验证 learned gate `o_g` 是否具有类似 chromatin accessibility 的语义——即 gate 学到的"开放/关闭"模式是否与真实的染色质状态一致。

### 3A. Gate 二值化程度分析

统计 per-gene gate 的分布：理想情况下 gate 应是双峰的（接近 0 或 1）。计算 gate entropy、bimodality index（BCI），并与随机 baseline 比较。

### 3B. Gate 与基因表达的关系

分析 gate 与基因平均表达、表达方差、zero-inflation 的关系。预期：高 gate 基因表达更高、方差更大、zero 比例更低。

### 3C. Gate 跨细胞类型的一致性

在 PBS 数据的不同细胞类型（如有标注）中比较 gate 的稳定性。预期：管家基因的 gate 跨类型一致偏高，细胞类型特异基因的 gate 在对应类型中偏高。

### 3D. 与 Multiome ATAC 的外部验证（连接 Study 4）

对每个基因比较 learned gate 与 promoter accessibility（来自 Multiome ATAC）。预期显著正相关。

### 预期结果

- 存在显著跨染色体调控流
- 高 gate regulator 具有更强跨染色体影响
- 部分经典 TF 的预测 target 与已知 regulon 显著重叠

### 论文图

- 22×22 chr flow heatmap
- cross-chr ratio 分布图
- selected TF 的 predicted target enrichment 图

---

## Study 4：基于 10x Multiome PBMC 的配对多组学外部验证

### 研究目的

利用公共 10x Multiome PBMC 配对 RNA+ATAC 数据，验证 Hayat 的 learned gate `o_g` 是否与真实染色质可及性一致。这是整个项目的多组学机制验证核心。

### 核心假设

- **H4.1**：per-gene learned gate `o_g`（平均跨细胞）与 promoter accessibility 显著正相关
- **H4.2**：这一关系在不同 PBMC 细胞类型中均成立（cell-type-specific gate vs ATAC）
- **H4.3**：高 gate 基因更偏"cis 许可"模式（表达主要受 gate 控制），低 gate 基因更偏"trans 驱动"模式
- **H4.4**：高 trans program loading 的基因在 gate 开放的前提下，表达主要由对应 program 活性解释

### 模型内部表征定义

将 Multiome RNA 输入 Hayat，直接提取结构化 latent：

```
o_g(c)  = gate probability per gene per cell     (from Eq A)
r_g(c)  = drive per gene per cell                 (from Eq B)
z_k(c)  = trans program activity per cell         (from encoder)
u_s(c)  = segment cis state per cell              (from encoder)
```

主分析使用跨细胞平均值：`ō_g = E_c[o_g(c)]`，按细胞类型版本 `ō_g(t) = E_{c∈t}[o_g(c)]`。

### 分析内容

#### 4A. 全局基因层面：gate 与 promoter accessibility

对每个基因比较 `ō_g`（learned gate）与 `ATAC(g)`（promoter accessibility），做 Spearman correlation。预期显著正相关。与 Study 3D 互为验证。

#### 4B. 细胞类型分层验证

在各主要 PBMC 细胞类型内分别计算 `corr_t = Spearman(ō_g(·,t), ATAC(·,t))`，排除 bulk 平均伪相关。

#### 4C. 控制表达量后的偏相关

`partial_corr(ō_g, ATAC | RNA_mean)`，若控制表达后仍显著，说明 gate 学到的不只是表达强度，而是更接近染色质状态的信息。

#### 4D. 按 promoter openness 分层评估模型行为

根据 Multiome 的 promoter accessibility 将基因分为 Open / Mid / Closed，比较重建 Pearson、重建 MSE、Schmidt PCC-delta。预期 Open 基因重建更优（gate 对其约束更强）。

#### 4E. Gate × Drive 交互模式

将基因按 gate score 和 mean drive 分层，观察 gate/drive 的联合分布。预期：存在 high-gate + high-drive（活跃表达基因）、high-gate + low-drive（可表达但未被激活）、low-gate + *（沉默基因）。

#### 4F. Program-Target 跨模态验证

对 top program k，提取其 top loading genes，检查这些基因在相应细胞类型中的 ATAC accessibility 是否更高，与 TRRUST / DoRothEA / ChEA target set 比较。

#### 4F. regulator-target 跨模态验证

对 top gate regulator 或 top TF：提取其高 attention / 高影响 target genes，检查这些 target 在相应细胞类型中是否更开放，与 TRRUST / DoRothEA / ChEA target set 比较。形成 "cis × trans × 外部知识" 交互验证。

### 预期结果

- ō_g 与 promoter accessibility 在全局及细胞类型层面均显著正相关
- 控制表达量后相关性仍保留
- High-gate 基因更偏"cis 许可"模式
- Top program 的 target genes 兼具更高可及性和更强 regulon 支持

### 论文图

- ō_g vs promoter accessibility 散点图
- 细胞类型相关系数柱状图
- Open/Mid/Closed 分层 gate 分布箱线图
- Gate × Drive 联合分布 heatmap
- Selected program target accessibility 图

---

## Supplementary Study：训练规模敏感性分析

### 目的

验证主要结论不依赖超大规模训练。

### 方案

训练 3 个规模（2k / 10k / 50k cells），比较 Global Pearson、PCC-delta、program TF enrichment OR、gate-ATAC Spearman。证明 10k–20k cells 已可得到稳定结果。

---

## 十、统计分析方案

| 分析类型 | 方法 |
|----------|------|
| 富集检验 | Fisher exact test / Hypergeometric test |
| 相关分析 | Spearman correlation / Partial correlation（控制表达量） |
| 分组比较 | Wilcoxon rank-sum / Kruskal-Wallis / t-test / ANOVA |
| 多重检验校正 | Benjamini-Hochberg FDR |
| 结果报告 | effect size, 95% CI, p 值或 FDR, mean ± SD, ≥ 3 seeds |

---

## 十一、实现结构

```
models/
├── hayat_v2.py                   # Open-Then-Express SCM 模型
└── model.py                      # Hayat v1 (Mamba2, 保留)

train/
├── run_training_v2.py            # v2 训练（分阶段 annealing）
└── run_training.py               # v1 训练

analysis/
├── study1_benchmark.py           # Full + forbidden-edge ablations + LinearAE
├── study2_programs.py            # Trans program 分析 (W loading, TF enrichment, GO)
├── study3_gate.py                # Gate 分析 (bimodality, expression relation, ATAC)
├── study4_multiome.py            # 10x Multiome 配对验证
├── train_linear_ae.py            # 线性AE训练
├── summarize_metrics.py          # 统一指标汇总
└── utils_analysis.py             # 共享工具函数
```

### 模型接口

```python
model.forward(
    x,                              # [B, G] expression
    e_scGPT=None,                   # [G, 512] optional frozen embeddings
    ablation="full",                # full / soft_gate / additive / no_gate / no_drive
    return_latent=False,            # 返回 u, z, c, o, r
)
```

### 输出 summary

- `gate_scores.csv` — per-gene learned gate ō_g
- `drive_scores.csv` — per-gene mean drive r̄_g
- `program_loadings.csv` — W matrix [G, K]
- `program_activities.csv` — z per cell [B, K]
- `segment_states.csv` — u per cell [B, S, d_u]
- `multiome_celltype_scores.csv` — cell-type-stratified gate vs ATAC

---

## 十二、实施步骤

### 阶段 0：实现 v2 模型

1. 实现 `hayat_v2.py`（encoder + SCM decoder）
2. 实现分阶段训练（warmup → hardening → stable）
3. 2000 cells 快速验证 gate 二值化、program 稀疏化是否收敛

### 阶段 1：最小闭环验证

1. 训练 5k cells Hayat v2 pilot 模型
2. 训练 LinearAE
3. 完成 Study 1 主比较 + 禁止边 ablation
4. 完成 Study 2 的 program TF 富集
5. 完成 Study 4 的 gate vs ATAC 验证

### 阶段 2：主结果生成

形成论文主图：

1. 训练 10k–20k cells 主模型
2. 完成 Study 1 全部
3. 完成 Study 2 全部
4. 完成 Study 3 的 chr flow 与 target enrichment
5. 完成 Study 4 全部多组学分析

### 阶段 3：增强验证

1. 规模敏感性分析
2. 选定经典 TF 做 case study
3. 可选 50k+ 训练
4. 可选 genome browser 可视化

---

## 十三、潜在风险与替代方案

| 风险 | 替代方案 |
|------|----------|
| cis 与 ATAC 相关性不够强 | cell-type-specific 分析；控制表达量偏相关；调整 promoter 窗口比较稳健性 |
| gate 对 TF 富集不显著 | 使用 top-K 替代固定 threshold；增加 TRRUST / DoRothEA regulon 支持；连续 gate score 对扰动 effect size 做相关 |
| no-blocking 比较被质疑不公平 | 主文标记为 inference-time stress test；补训 no-blocking 模型做 supplementary |
| Multiome ATAC 稀疏 | pseudo-bulk；必要时 metacell；主分析不依赖单细胞逐点 ATAC |

---

## 十四、预期主要结论

1. Hayat 在表达重建和扰动泛化上优于线性和退化模型
2. cis 分支主要负责局部/同染色体依赖，trans 分支主要承担全局/跨染色体调控
3. RegulatorGate 能自动识别具有真实调控功能的关键基因
4. 模型的 cis 表征与配对 Multiome 中的 promoter accessibility 一致
5. 模型的 trans 表征与 TF regulon、扰动传播和跨染色体调控流一致
6. Hayat 不仅提升预测性能，还学习到了与基因组调控组织相一致的表达生成机制

---

## 十五、论文叙事线

1. **问题提出** — 现有表达模型将"能否表达"和"表达多少"混为一谈，缺少对 cis 许可/trans 驱动的显式分离
2. **方法提出** — Hayat 通过 Open-Then-Express SCM：segment cis state → gate（许可），trans programs → drive（强度），hurdle 融合保证 gate 关闭时表达为零。不引入任何外部调控先验
3. **结果一：性能与 SCM 必要性** — Hayat hurdle 优于 Soft-gate、Additive、No-gate、No-drive；禁止边破坏 gate 语义
4. **结果二：Trans programs 真实性** — 无先验 programs 富集已知 TF target，gene loading 稀疏且模块化
5. **结果三：Gate 的 ATAC 样特征** — Learned gate 与独立 Multiome ATAC 的 promoter accessibility 显著一致
6. **结论** — Hayat 通过结构化 SCM 而非更强 backbone，实现了兼具性能与机制解释力的表达建模

---

## 十六、最小可执行版本

若希望先做一个低成本但完整闭环的版本，只完成以下 4 项：

1. 训练 10k cells Hayat v2
2. 训练 LinearAE
3. Study 1：Full + 3 个关键 ablation（no_gate, additive, z_to_gate）
4. Study 2 program TF 富集 + Study 4 gate-ATAC 验证

这四部分已经足够形成一个完整、可报告、可投稿扩展的核心框架。

# Preprocessed Data Discrepancies & TGNN Applicability Report

**Paper Title Proposal:** *Auditing Data Discrepancies in Ecological Event Streams: When Can Temporal Graph Neural Networks Be Applied to Continuous-Time Plant–Pollinator Networks?*

---

## 1. Executive Summary of Core Findings

Our systematic audit of the preprocessed continuous-time event stream (`plant_pollinator_events.csv`, `events_train.csv`, `events_val.csv`, `events_test.csv`, `flw_conf.csv`, `plant_pollinator_negative_events.csv`) from Ghosh & Mukhopadhyay reveals **6 foundational discrepancies** between standard TGNN continuous-time assumptions and actual empirical dataset properties:

| Discrepancy Dimension | Empirical Finding | TGNN Failure Mode | Severity |
|---|---|---|---|
| **D1: Zero-Interval Collapse** | **98.95%** of transitions have $\Delta \tau = 0$ (210 active days total, up to 318 events/day) | Time encoding $\cos(\omega \Delta t)$ collapses; shuffling intra-day order swings test AP by **0.015** | **High** |
| **D2: Split Boundary Leakage** | Same-day partitioning: 2017-07-14 in Train & Val (**28 shared pairs**); 2018-07-03 in Val & Test (**31 shared pairs**) | Memory updated in morning and evaluated in afternoon with $\Delta t = 0$; breaks data independence | **Critical** |
| **D3: Multi-Year Hiatus Collapse** | 1,049-day gap between 2018 and 2021 | Test AP collapses from **0.876** (2018) to **0.596** (2021) — a **32.0% drop** | **Critical** |
| **D4: Inductive Turnover Breakdown** | **56.9%** of test pollinators never appeared in training; **28.7%** of test events are novel | Recurrent pairs achieve **0.937 AP**, while inductive pairs drop to **0.776 AP** | **High** |
| **D5: Negative Realism Gap** | `negative_events.csv` has null species codes; benchmark resorted to uniform random sampling | Evaluating on ecologically co-active species drops AP from **0.879** to **0.661** | **High** |
| **D6: Weight-Loss Disconnect** | $c \in [0, 1]$ is heavily skewed (median 0.053, 8.8% < 0.01) | Baseline uses unweighted BCE, treating low-confidence noise identically to verified interactions | **Medium** |

---

## 2. In-Depth Technical Evidence

### Discrepancy 1: Timestamp Degeneracy & Permutation Variance
- **Active Survey Days:** 210 days over 10 years.
- **Mean Events per Active Day:** 95.4 (Maximum on single day: 318).
- **Zero $\Delta \tau$ Fraction:** 98.96% of consecutive events occur at the exact same day $\tau$.
- **Permutation Spread:** Test AP across random intra-day orderings ranges from 0.8723 to 0.8873 (spread = 0.0151).

### Discrepancy 2: Split Boundary Data Leakage
- **Train/Val Boundary Collision (Date: ['2017-07-14']):**
  - Train events on collision date: 72
  - Val events on collision date: 159
  - Identical $(u, v)$ pairs leaking across split: **28**
- **Val/Test Boundary Collision (Date: ['2018-07-03']):**
  - Val events on collision date: 157
  - Test events on collision date: 92
  - Identical $(u, v)$ pairs leaking across split: **31**

### Discrepancy 3: Post-Hiatus Model Collapse across the 1,049-Day Gap
- **Pre-Hiatus Test (2018, 1740 events):** AP = 0.8759, AUC = 0.8796
- **Post-Hiatus Test (2021, 263 events):** AP = 0.5958, AUC = 0.6822
- **Degradation:** AP dropped by 0.2801 (32.0%), AUC dropped by 0.1974 (22.4%).

### Discrepancy 4: Graph Topology & Inductive Generalization
- **Test Set Composition:**
  - Recurrent pairs (seen in train): 1256 events (62.7%) -> AP: **0.9368**, AUC: **0.9491**
  - Semi-novel pairs: 172 events (8.6%) -> AP: **0.8618**, AUC: **0.8599**
  - Inductive pairs (novel species): 575 events (28.7%) -> AP: **0.7760**, AUC: **0.7504**

### Discrepancy 5: Negative Realism Gap (3-Tier Evaluation)
- **Tier 1 (Uniform Random Negatives):** AP = 0.8789, AUC = 0.8719
- **Tier 2 (Spatially & Seasonally Co-Active):** AP = 0.6613, AUC = 0.6707
- **Realism Gap:** AP drop = **0.2176**, AUC drop = **0.2012**

### Discrepancy 6: Confidence-Weighted vs Unweighted BCE
- **Unweighted BCE Test AP:** 0.8459 (Loss: 0.5091)
- **Confidence-Weighted BCE Test AP:** 0.7856 (Loss: 1.0895)
- **AP Delta:** -0.0603

---

## 3. Publication Figures Generated
All high-resolution figures are exported to `reports_and_figures/`:
1. `fig_hiatus_collapse.png` — Visualizes the 2018 vs 2021 post-hiatus collapse.
2. `fig_recurrent_vs_novel.png` — Breaks down performance across recurrent, semi-novel, and inductive topological links.
3. `fig_negative_realism.png` — Demonstrates the Negative Realism Gap between random and ecologically co-active species.
4. `fig_permutation_variance.png` — Shows the boxplot of metric variation under intra-day sequence permutations.

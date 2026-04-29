# Multi-Objective-Inventory-Optimization-nsga2
Multi-Objective Optimization using nsga-II algorithm and TOPSIS algorithm




---

## 📌 Overview

This project presents a **machine learning-enhanced, multi-objective optimization framework** for inventory policy determination across a multi-product supply chain system.

Instead of collapsing all inventory goals into a single number (like classical EOQ does), this framework **simultaneously optimizes three conflicting objectives** and produces a full **Pareto front** — a set of optimal trade-off solutions. The best policy is then selected automatically using **TOPSIS**.



---

## 🎯 Objectives Optimized Simultaneously

| # | Objective | Goal | Description |
|---|-----------|------|-------------|
| f₁ | **Total Cost** | Minimize ↓ | Holding + Ordering + Stockout costs |
| f₂ | **Stockout Rate** | Minimize ↓ | Fraction of weeks with unmet demand |
| f₃ | **Avg Inventory** | Minimize ↓ | Working capital tied up in stock |

These objectives **conflict** with each other — reducing stockouts requires more stock, which raises costs. NSGA-II maps out all possible trade-offs, and TOPSIS picks the best compromise.

---

## 🏗️ Project Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE OVERVIEW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. SYNTHETIC DATA          2. SIMULATION ENGINE               │
│  ─────────────────          ─────────────────────              │
│  10 products × 52 weeks     Discrete-event (s,Q) policy        │
│  • Seasonality              • Holding cost accumulation        │
│  • Trend component          • Ordering trigger + lead time     │
│  • Gaussian noise           • Stockout detection & cost        │
│         │                            │                         │
│         ▼                            ▼                         │
│  3. NSGA-II OPTIMIZER       4. PARETO FRONT                    │
│  ────────────────           ────────────────                   │
│  Pop=100, Gen=80            Non-dominated solutions            │
│  • SBX Crossover            • Well-spread trade-offs           │
│  • Polynomial Mutation      • 3-objective surface              │
│  • Crowding Distance                 │                         │
│         │                            ▼                         │
│         └──────────────►  5. TOPSIS SELECTION                  │
│                            ──────────────────                  │
│                            Best compromise policy              │
│                            per-product ROP + Q values          │
└─────────────────────────────────────────────────────────────────┘
```



---

## 🔬 Methodology

### Synthetic Data Generation
Weekly demand is modeled as:

```
d_i(t) = μ_i  +  A_i · sin(2πt/52)  +  β_i · t  +  ε_i(t)
```
where μ is base demand, A·sin(…) is seasonality, β·t is trend, and ε ~ N(0,σ²) is noise.

### NSGA-II Algorithm
The **Non-dominated Sorting Genetic Algorithm II** (Deb et al., 2002):
- **Chromosome:** `[ROP₀, Q₀, ROP₁, Q₁, ..., ROP₉, Q₉]` — 20 continuous variables
- **Crossover:** Simulated Binary Crossover (SBX, η=15)
- **Mutation:** Polynomial Mutation (η=20, prob=1/dim)
- **Selection:** Binary tournament by rank + crowding distance
- **Elitism:** Combined pool of 200 → select best 100 per generation

### TOPSIS Decision
Selects the best Pareto solution by minimizing distance to the ideal best and maximizing distance from the ideal worst, with user-specified weights:

| Objective | Weight |
|-----------|--------|
| Total Cost | 0.50 |
| Stockout Rate | 0.30 |
| Avg Inventory | 0.20 |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Pareto Front Size | 100 non-dominated solutions |
| Best Total Cost (avg/product) | $43,074 / year |
| Best Stockout Rate | 21.73% |
| Best Avg Inventory | 247.1 units |
| TOPSIS Score | 0.6550 |
| Generations | 80 |
| Population Size | 100 |

### Convergence
| Generation | Pareto Front Size | Best Cost | Best Stockout |
|------------|-------------------|-----------|---------------|
| 20 | 129 | $45,260 | 11.54% |
| 40 | 126 | $43,313 | 11.35% |
| 60 | 119 | $43,130 | 11.35% |
| **80** | **100** | **$41,674** | **10.77%** |

<img width="2436" height="2179" alt="pareto_results" src="https://github.com/user-attachments/assets/4330b292-0a59-4efe-89ca-9cc03c70479c" />

---

## 🧩 Key Concepts

| Concept | Explanation |
|---------|-------------|
| **Pareto Front** | Set of solutions where no objective can be improved without worsening another |
| **Dominance** | Solution A dominates B if A is ≤ B on all objectives and < B on at least one |
| **Reorder Point (ROP)** | Inventory level that triggers a new purchase order |
| **Order Quantity (Q)** | Fixed number of units ordered each time |
| **EOQ** | Classical Economic Order Quantity — single-objective benchmark |
| **TOPSIS** | Multi-criteria method to pick best solution from Pareto set |

---

## 📈 Visualization Dashboard

The output chart (`pareto_results.png`) contains 6 panels:

1. **3-D Pareto Surface** — All solutions colored by TOPSIS score, best marked with ★
2. **TOPSIS Score Ranking** — Bar chart of all Pareto solutions ranked
3. **Convergence History** — How f₁, f₂, f₃ improve over 80 generations
4. **Cost vs Stockout Trade-off** — 2-D projection of Pareto front
5. **Cost vs Inventory Trade-off** — 2-D projection of Pareto front
6. **Best Solution Summary Table** — Key metrics at a glance

---





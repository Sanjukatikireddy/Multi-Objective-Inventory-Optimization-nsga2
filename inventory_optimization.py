"""
============================================================
  Inventory Optimization using Multi-Objective
  Optimization (NSGA-II) with Pareto Front Analysis
============================================================
  Objectives:
    f1 → Minimize Total Cost (holding + ordering)
    f2 → Minimize Stockout Rate
    f3 → Minimize Average Inventory Level

  Best solution selected via TOPSIS (Technique for Order
  of Preference by Similarity to Ideal Solution)
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)
import sys
sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────────────────
# 1. SYNTHETIC DATA GENERATION
# ──────────────────────────────────────────────────────────

def generate_inventory_data(n_products=10, n_periods=52):
    """
    Synthesise weekly demand, costs, and lead-time data
    for n_products over n_periods (weeks).
    """
    products = []
    for i in range(n_products):
        base_demand  = np.random.uniform(50, 300)       # units/week
        seasonality  = np.random.uniform(0.1, 0.4)
        trend        = np.random.uniform(-0.5, 1.5)     # units/week growth
        noise_std    = base_demand * 0.15

        weeks        = np.arange(n_periods)
        seasonal_comp = seasonality * base_demand * np.sin(2 * np.pi * weeks / 52)
        trend_comp    = trend * weeks
        noise         = np.random.normal(0, noise_std, n_periods)
        demand        = np.maximum(0, base_demand + seasonal_comp + trend_comp + noise)

        products.append({
            "product_id"      : f"P{i+1:02d}",
            "demand"          : demand,                              # weekly demand array
            "mean_demand"     : demand.mean(),
            "std_demand"      : demand.std(),
            "holding_cost"    : np.random.uniform(0.5,  3.0),       # $/unit/week
            "ordering_cost"   : np.random.uniform(50,  300),        # $/order
            "unit_cost"       : np.random.uniform(10,  100),        # $/unit
            "stockout_cost"   : np.random.uniform(5,   25),         # $/unit shortage
            "lead_time"       : np.random.randint(1,   5),          # weeks
            "service_level"   : np.random.uniform(0.85, 0.99),
        })
    return products

# ──────────────────────────────────────────────────────────
# 2. INVENTORY SIMULATION (EOQ + ROP-based policy)
# ──────────────────────────────────────────────────────────

def simulate_inventory(product, reorder_point, order_qty, n_periods=52):
    """
    Simulate (s, Q) inventory policy over n_periods.
    Returns: total_cost, stockout_rate, avg_inventory
    """
    demand      = product["demand"]
    h           = product["holding_cost"]
    K           = product["ordering_cost"]
    s_cost      = product["stockout_cost"]
    lead_time   = product["lead_time"]

    inventory       = order_qty                # start with one order
    total_holding   = 0.0
    total_ordering  = 0.0
    total_stockout  = 0.0
    stockout_events = 0
    on_order        = 0                        # units on order
    delivery_schedule = {}

    for t in range(n_periods):
        # Receive pending deliveries
        if t in delivery_schedule:
            inventory += delivery_schedule[t]

        # Satisfy demand
        d = demand[t]
        if inventory >= d:
            inventory -= d
        else:
            shortage        = d - inventory
            total_stockout += shortage * s_cost
            stockout_events += 1
            inventory       = 0

        # Holding cost
        total_holding += inventory * h

        # Reorder check
        if inventory + on_order <= reorder_point:
            total_ordering  += K
            arrival          = t + lead_time + 1
            delivery_schedule[arrival] = delivery_schedule.get(arrival, 0) + order_qty
            on_order        += order_qty

        # Update on_order
        delivered_this_period = delivery_schedule.get(t, 0)
        on_order = max(0, on_order - delivered_this_period)

        total_holding = max(0, total_holding)

    total_cost      = total_holding + total_ordering + total_stockout
    stockout_rate   = stockout_events / n_periods
    avg_inventory   = total_holding / (h * n_periods + 1e-9)

    return total_cost, stockout_rate, avg_inventory

# ──────────────────────────────────────────────────────────
# 3. MULTI-OBJECTIVE EVALUATION
# ──────────────────────────────────────────────────────────

def evaluate_solution(chromosome, products):
    """
    chromosome: [ROP_0, Q_0, ROP_1, Q_1, ..., ROP_n, Q_n]
    Returns: [F1_total_cost, F2_stockout_rate, F3_avg_inventory]
    """
    n = len(products)
    f1, f2, f3 = 0.0, 0.0, 0.0
    for i, prod in enumerate(products):
        rop = chromosome[2*i]
        qty = chromosome[2*i + 1]
        tc, sr, ai = simulate_inventory(prod, rop, qty)
        f1 += tc
        f2 += sr
        f3 += ai
    return np.array([f1 / n, f2 / n, f3 / n])

# ──────────────────────────────────────────────────────────
# 4. NSGA-II ALGORITHM
# ──────────────────────────────────────────────────────────

def dominates(a, b):
    """True if solution a dominates solution b (minimisation)."""
    return np.all(a <= b) and np.any(a < b)

def fast_non_dominated_sort(objectives):
    """Returns list of Pareto fronts (indices)."""
    n = len(objectives)
    dom_count  = np.zeros(n, dtype=int)
    dom_set    = [[] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(objectives[i], objectives[j]):
                dom_set[i].append(j)
            elif dominates(objectives[j], objectives[i]):
                dom_count[i] += 1

    fronts = [[i for i in range(n) if dom_count[i] == 0]]

    k = 0
    while k < len(fronts) and fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dom_set[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        k += 1

    return fronts

def crowding_distance(objectives, front):
    """Crowding distance assignment for a front."""
    n   = len(front)
    dist = np.zeros(n)
    if n <= 2:
        dist[:] = np.inf
        return dist
    obj_vals = objectives[front]
    for m in range(obj_vals.shape[1]):
        idx    = np.argsort(obj_vals[:, m])
        dist[idx[0]]  = np.inf
        dist[idx[-1]] = np.inf
        f_range = obj_vals[idx[-1], m] - obj_vals[idx[0], m]
        if f_range == 0:
            continue
        for i in range(1, n-1):
            dist[idx[i]] += (obj_vals[idx[i+1], m] - obj_vals[idx[i-1], m]) / f_range
    return dist

def tournament_select(pop, objectives, fronts, crowd_dist, k=2):
    """Binary tournament selection."""
    idx = np.random.randint(0, len(pop), k)
    # Determine front rank of each
    rank = np.zeros(len(pop), dtype=int)
    for r, f in enumerate(fronts):
        for i in f:
            rank[i] = r

    best = idx[0]
    for i in idx[1:]:
        if rank[i] < rank[best]:
            best = i
        elif rank[i] == rank[best] and crowd_dist[i] > crowd_dist[best]:
            best = i
    return best

def crossover(p1, p2, eta=15):
    """Simulated Binary Crossover (SBX)."""
    child1, child2 = p1.copy(), p2.copy()
    for i in range(len(p1)):
        u = np.random.rand()
        if u <= 0.5:
            beta = (2*u)**(1/(eta+1))
        else:
            beta = (1/(2*(1-u)))**(1/(eta+1))
        child1[i] = 0.5*((1+beta)*p1[i] + (1-beta)*p2[i])
        child2[i] = 0.5*((1-beta)*p1[i] + (1+beta)*p2[i])
    return child1, child2

def mutate(individual, bounds, eta=20, prob=0.1):
    """Polynomial Mutation."""
    result = individual.copy()
    for i in range(len(result)):
        if np.random.rand() < prob:
            u      = np.random.rand()
            lo, hi = bounds[i]
            delta  = hi - lo
            if delta == 0:
                continue
            if u < 0.5:
                delta_q = (2*u + (1-2*u)*(1-(result[i]-lo)/delta)**(eta+1))**(1/(eta+1)) - 1
            else:
                delta_q = 1 - (2*(1-u) + 2*(u-0.5)*((hi-result[i])/delta)**(eta+1))**(1/(eta+1))
            result[i] = np.clip(result[i] + delta_q*delta, lo, hi)
    return result

def nsga2(products, pop_size=100, n_gen=80):
    """
    NSGA-II for inventory optimisation.
    Decision variables: [ROP_i, Q_i] for each product i.
    """
    n = len(products)
    dim = 2 * n

    # Variable bounds  [ROP, Q]
    bounds = []
    for prod in products:
        mu, sigma = prod["mean_demand"], prod["std_demand"]
        lt        = prod["lead_time"]
        rop_lo    = max(0, mu*lt - 2*sigma)
        rop_hi    = mu*lt + 3*sigma
        q_lo      = max(10, mu)
        q_hi      = mu * 8
        bounds.append((rop_lo, rop_hi))
        bounds.append((q_lo,   q_hi))
    bounds = np.array(bounds)

    # Initialise population
    pop = np.array([
        np.array([np.random.uniform(lo, hi) for lo, hi in bounds])
        for _ in range(pop_size)
    ])

    # Evaluate
    objectives = np.array([evaluate_solution(ind, products) for ind in pop])

    history = {"f1": [], "f2": [], "f3": []}

    print("=" * 60)
    print("  NSGA-II Multi-Objective Inventory Optimisation")
    print("=" * 60)

    for gen in range(n_gen):
        fronts    = fast_non_dominated_sort(objectives)
        crowd_d   = np.zeros(pop_size)
        for f in fronts:
            d = crowding_distance(objectives, f)
            for idx, val in zip(f, d):
                crowd_d[idx] = val

        # Generate offspring
        offspring     = []
        offspring_obj = []
        while len(offspring) < pop_size:
            p1_idx = tournament_select(pop, objectives, fronts, crowd_d)
            p2_idx = tournament_select(pop, objectives, fronts, crowd_d)
            c1, c2 = crossover(pop[p1_idx], pop[p2_idx])
            c1 = mutate(c1, bounds)
            c2 = mutate(c2, bounds)
            c1 = np.clip(c1, bounds[:, 0], bounds[:, 1])
            c2 = np.clip(c2, bounds[:, 0], bounds[:, 1])
            offspring.append(c1)
            offspring.append(c2)
            offspring_obj.append(evaluate_solution(c1, products))
            offspring_obj.append(evaluate_solution(c2, products))

        # Combine + select next generation
        combined      = np.vstack([pop, offspring])
        combined_obj  = np.vstack([objectives, offspring_obj])
        all_fronts    = fast_non_dominated_sort(combined_obj)
        all_crowd     = np.zeros(len(combined))
        for f in all_fronts:
            d = crowding_distance(combined_obj, f)
            for idx, val in zip(f, d):
                all_crowd[idx] = val

        selected = []
        for f in all_fronts:
            if len(selected) + len(f) <= pop_size:
                selected.extend(f)
            else:
                rem   = pop_size - len(selected)
                crowd_sorted = sorted(f, key=lambda x: all_crowd[x], reverse=True)
                selected.extend(crowd_sorted[:rem])
                break

        pop        = combined[selected]
        objectives = combined_obj[selected]

        pf_idx = all_fronts[0]
        history["f1"].append(combined_obj[pf_idx, 0].min())
        history["f2"].append(combined_obj[pf_idx, 1].min())
        history["f3"].append(combined_obj[pf_idx, 2].min())

        if (gen+1) % 20 == 0:
            print(f"  Gen {gen+1:3d}/{n_gen} | PF size: {len(all_fronts[0]):3d} | "
                  f"Best Cost: ${history['f1'][-1]:,.0f} | "
                  f"Best Stockout: {history['f2'][-1]*100:.2f}%")

    # Final Pareto front
    fronts    = fast_non_dominated_sort(objectives)
    pf_idx    = fronts[0]
    pf_pop    = pop[pf_idx]
    pf_obj    = objectives[pf_idx]

    print(f"\n  Pareto Front contains {len(pf_idx)} non-dominated solutions.")
    return pf_pop, pf_obj, objectives, history

# ──────────────────────────────────────────────────────────
# 5. BEST SOLUTION SELECTION — TOPSIS
# ──────────────────────────────────────────────────────────

def topsis(objectives, weights=None):
    """
    TOPSIS for selecting best solution from the Pareto front.
    All objectives are assumed to be minimisation.
    weights: list of importance weights (sums to 1)
    """
    if weights is None:
        weights = np.array([0.5, 0.3, 0.2])   # cost, stockout, inventory

    n, m  = objectives.shape
    norms = np.linalg.norm(objectives, axis=0)
    norms[norms == 0] = 1
    norm_obj = objectives / norms

    weighted = norm_obj * weights

    # Ideal best (min) and worst (max) for minimisation
    ideal_best  = weighted.min(axis=0)
    ideal_worst = weighted.max(axis=0)

    dist_best  = np.sqrt(((weighted - ideal_best)**2).sum(axis=1))
    dist_worst = np.sqrt(((weighted - ideal_worst)**2).sum(axis=1))

    score = dist_worst / (dist_best + dist_worst + 1e-9)
    best_idx = np.argmax(score)

    return best_idx, score

# ──────────────────────────────────────────────────────────
# 6. VISUALISATION
# ──────────────────────────────────────────────────────────

def plot_results(pf_obj, all_obj, best_idx, topsis_scores, history, products):

    plt.rcParams.update({
        "figure.facecolor" : "#0d1117",
        "axes.facecolor"   : "#161b22",
        "axes.edgecolor"   : "#30363d",
        "axes.labelcolor"  : "#e6edf3",
        "xtick.color"      : "#8b949e",
        "ytick.color"      : "#8b949e",
        "text.color"       : "#e6edf3",
        "grid.color"       : "#21262d",
        "grid.linestyle"   : "--",
        "grid.alpha"       : 0.6,
        "font.family"      : "DejaVu Sans",
    })

    ACCENT   = "#58a6ff"
    GREEN    = "#3fb950"
    ORANGE   = "#f0883e"
    RED      = "#f85149"
    PURPLE   = "#bc8cff"
    GOLD     = "#e3b341"

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Inventory Optimization — Multi-Objective Pareto Analysis",
                 fontsize=17, fontweight="bold", color="#e6edf3", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── A: 3-D Pareto Front ──────────────────────────────
    ax3d = fig.add_subplot(gs[0, :2], projection="3d")
    ax3d.set_facecolor("#161b22")

    # All dominated solutions (grey)
    ax3d.scatter(all_obj[:, 0]/1e3, all_obj[:, 1]*100, all_obj[:, 2],
                 c="#30363d", s=15, alpha=0.4, label="Dominated")

    # Pareto front
    sc = ax3d.scatter(pf_obj[:, 0]/1e3, pf_obj[:, 1]*100, pf_obj[:, 2],
                      c=topsis_scores, cmap="plasma", s=60, alpha=0.9,
                      edgecolors="none", label="Pareto Front")
    cbar = plt.colorbar(sc, ax=ax3d, pad=0.1, shrink=0.6)
    cbar.set_label("TOPSIS Score", color="#e6edf3", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#8b949e")

    # Best solution
    best = pf_obj[best_idx]
    ax3d.scatter([best[0]/1e3], [best[1]*100], [best[2]],
                 c=GOLD, s=200, marker="*", zorder=5, label="Best (TOPSIS)")

    ax3d.set_xlabel("Total Cost (k$)", labelpad=8, color="#8b949e", fontsize=9)
    ax3d.set_ylabel("Stockout Rate (%)", labelpad=8, color="#8b949e", fontsize=9)
    ax3d.set_zlabel("Avg Inventory (units)", labelpad=8, color="#8b949e", fontsize=9)
    ax3d.set_title("3-D Pareto Front", color="#e6edf3", fontsize=11, pad=10)
    ax3d.legend(loc="upper left", fontsize=8,
                facecolor="#0d1117", edgecolor="#30363d", labelcolor="#e6edf3")
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.tick_params(colors="#8b949e", labelsize=7)

    # ── B: TOPSIS Score Distribution ─────────────────────
    ax_t = fig.add_subplot(gs[0, 2])
    sorted_scores = np.sort(topsis_scores)[::-1]
    colors_bar = [GOLD if i == 0 else ACCENT for i in range(len(sorted_scores))]
    ax_t.bar(range(len(sorted_scores)), sorted_scores, color=colors_bar, width=0.8, alpha=0.85)
    ax_t.set_xlabel("Pareto Solution (ranked)", fontsize=9)
    ax_t.set_ylabel("TOPSIS Score", fontsize=9)
    ax_t.set_title("TOPSIS Score Ranking", fontsize=11, fontweight="bold")
    ax_t.axhline(sorted_scores[0], color=GOLD, lw=1.2, ls="--", alpha=0.7)
    ax_t.text(1, sorted_scores[0]+0.005, f"Best: {sorted_scores[0]:.3f}",
              color=GOLD, fontsize=8)
    ax_t.grid(True, axis="y")

    # ── C: Convergence History ───────────────────────────
    ax_conv = fig.add_subplot(gs[1, 0])
    gens = range(1, len(history["f1"])+1)
    f1_norm = np.array(history["f1"]) / max(history["f1"])
    f2_norm = np.array(history["f2"]) / max(history["f2"])
    f3_norm = np.array(history["f3"]) / max(history["f3"])
    ax_conv.plot(gens, f1_norm, color=ACCENT,  lw=1.8, label="Cost")
    ax_conv.plot(gens, f2_norm, color=RED,     lw=1.8, label="Stockout")
    ax_conv.plot(gens, f3_norm, color=GREEN,   lw=1.8, label="Inventory")
    ax_conv.set_xlabel("Generation", fontsize=9)
    ax_conv.set_ylabel("Normalised Min Value", fontsize=9)
    ax_conv.set_title("Convergence History", fontsize=11, fontweight="bold")
    ax_conv.legend(fontsize=8, facecolor="#0d1117",
                   edgecolor="#30363d", labelcolor="#e6edf3")
    ax_conv.grid(True)

    # ── D: Objective Pairplot (Cost vs Stockout) ─────────
    ax_p1 = fig.add_subplot(gs[1, 1])
    ax_p1.scatter(all_obj[:, 0]/1e3, all_obj[:, 1]*100,
                  c="#30363d", s=12, alpha=0.4)
    ax_p1.scatter(pf_obj[:, 0]/1e3,  pf_obj[:, 1]*100,
                  c=topsis_scores, cmap="plasma", s=50, alpha=0.9)
    ax_p1.scatter(best[0]/1e3, best[1]*100,
                  c=GOLD, s=180, marker="*", zorder=5)
    ax_p1.set_xlabel("Total Cost (k$)", fontsize=9)
    ax_p1.set_ylabel("Stockout Rate (%)", fontsize=9)
    ax_p1.set_title("Cost vs Stockout Trade-off", fontsize=11, fontweight="bold")
    ax_p1.grid(True)

    # ── E: Objective Pairplot (Cost vs Inventory) ────────
    ax_p2 = fig.add_subplot(gs[1, 2])
    ax_p2.scatter(all_obj[:, 0]/1e3, all_obj[:, 2],
                  c="#30363d", s=12, alpha=0.4)
    ax_p2.scatter(pf_obj[:, 0]/1e3,  pf_obj[:, 2],
                  c=topsis_scores, cmap="plasma", s=50, alpha=0.9)
    ax_p2.scatter(best[0]/1e3, best[2],
                  c=GOLD, s=180, marker="*", zorder=5)
    ax_p2.set_xlabel("Total Cost (k$)", fontsize=9)
    ax_p2.set_ylabel("Avg Inventory (units)", fontsize=9)
    ax_p2.set_title("Cost vs Inventory Trade-off", fontsize=11, fontweight="bold")
    ax_p2.grid(True)

    # ── F: Demand Profile of first 4 products ────────────
    ax_dem = fig.add_subplot(gs[2, :2])
    weeks   = np.arange(52)
    col_map = [ACCENT, GREEN, ORANGE, PURPLE]
    for i, (prod, col) in enumerate(zip(products[:4], col_map)):
        ax_dem.plot(weeks, prod["demand"], color=col, lw=1.5, alpha=0.85,
                    label=prod["product_id"])
    ax_dem.set_xlabel("Week", fontsize=9)
    ax_dem.set_ylabel("Demand (units)", fontsize=9)
    ax_dem.set_title("Synthetic Demand Profiles (sample products)", fontsize=11, fontweight="bold")
    ax_dem.legend(fontsize=8, facecolor="#0d1117",
                  edgecolor="#30363d", labelcolor="#e6edf3", ncol=4)
    ax_dem.grid(True)

    # ── G: Best Solution Summary ─────────────────────────
    ax_sum = fig.add_subplot(gs[2, 2])
    ax_sum.axis("off")
    rows = [
        ["Metric",            "Value"],
        ["Total Cost (avg)",  f"${best[0]:,.0f}"],
        ["Stockout Rate",     f"{best[1]*100:.2f}%"],
        ["Avg Inventory",     f"{best[2]:.1f} units"],
        ["TOPSIS Score",      f"{topsis_scores[best_idx]:.4f}"],
        ["Pareto Front Size", f"{len(pf_obj)} solutions"],
        ["NSGA-II Pop Size",  "100"],
        ["Generations",       "80"],
    ]
    tbl = ax_sum.table(cellText=rows[1:], colLabels=rows[0],
                       cellLoc="center", loc="center",
                       colWidths=[0.55, 0.45])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#161b22" if r > 0 else "#21262d")
        cell.set_text_props(color="#e6edf3" if r > 0 else GOLD)
        cell.set_edgecolor("#30363d")
        if r == 0:
            cell.set_text_props(fontweight="bold", color=GOLD)
    ax_sum.set_title("Best Solution (TOPSIS)", fontsize=11,
                     fontweight="bold", pad=12)

    plt.savefig("pareto_results.png", dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    print("\n  Chart saved → pareto_results.png")
    plt.close()

# ──────────────────────────────────────────────────────────
# 7. PRINT RESULTS
# ──────────────────────────────────────────────────────────

def print_results(pf_obj, best_idx, topsis_scores, pf_pop, products):
    best_obj = pf_obj[best_idx]
    best_sol = pf_pop[best_idx]

    print("\n" + "="*60)
    print("  PARETO FRONT — Top 10 Solutions")
    print("="*60)
    ranked = np.argsort(topsis_scores)[::-1]
    print(f"  {'Rank':4s}  {'Cost ($)':>12s}  {'Stockout%':>10s}  {'Avg Inv':>10s}  {'TOPSIS':>8s}")
    print("  " + "-"*54)
    for rank, idx in enumerate(ranked[:10]):
        marker = " ★" if idx == best_idx else ""
        print(f"  {rank+1:4d}  {pf_obj[idx,0]:>12,.0f}  "
              f"{pf_obj[idx,1]*100:>10.2f}  {pf_obj[idx,2]:>10.1f}  "
              f"{topsis_scores[idx]:>8.4f}{marker}")

    print("\n" + "="*60)
    print("  BEST SOLUTION DETAILS")
    print("="*60)
    print(f"  Total Cost (avg/product):  ${best_obj[0]:>10,.2f}")
    print(f"  Stockout Rate:              {best_obj[1]*100:>10.2f}%")
    print(f"  Average Inventory:          {best_obj[2]:>10.1f}  units")
    print(f"  TOPSIS Score:               {topsis_scores[best_idx]:>10.4f}")

    print("\n  Recommended Policy per Product:")
    print(f"  {'Product':8s}  {'ROP (units)':>12s}  {'Order Qty':>12s}  {'EOQ (calc)':>12s}")
    print("  " + "-"*50)
    for i, prod in enumerate(products):
        rop = best_sol[2*i]
        qty = best_sol[2*i + 1]
        # Economic Order Quantity benchmark
        D   = prod["mean_demand"] * 52
        eoq = np.sqrt(2 * D * prod["ordering_cost"] / prod["holding_cost"])
        print(f"  {prod['product_id']:8s}  {rop:>12.1f}  {qty:>12.1f}  {eoq:>12.1f}")
    print("="*60)

# ──────────────────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "="*60)
    print("  Step 1 — Generating Synthetic Inventory Data")
    print("="*60)
    products = generate_inventory_data(n_products=10, n_periods=52)
    print(f"  Generated {len(products)} products × 52 weeks of demand data.")
    for p in products:
        print(f"    {p['product_id']}  μ={p['mean_demand']:.1f}  σ={p['std_demand']:.1f}  "
              f"h=${p['holding_cost']:.2f}  K=${p['ordering_cost']:.0f}  "
              f"LT={p['lead_time']}w")

    print("\n" + "="*60)
    print("  Step 2 — Running NSGA-II (80 generations, pop=100)")
    print("="*60)
    pf_pop, pf_obj, all_obj, history = nsga2(products, pop_size=100, n_gen=80)

    print("\n" + "="*60)
    print("  Step 3 — TOPSIS Best-Solution Selection")
    print("="*60)
    weights   = np.array([0.5, 0.3, 0.2])   # cost > stockout > inventory
    best_idx, topsis_scores = topsis(pf_obj, weights)
    print(f"  Weights → Cost: {weights[0]}, Stockout: {weights[1]}, Inventory: {weights[2]}")

    print_results(pf_obj, best_idx, topsis_scores, pf_pop, products)

    print("\n  Step 4 — Generating Visualisations …")
    plot_results(pf_obj, all_obj, best_idx, topsis_scores, history, products)

    print("\n  ✓ All done. Results saved to pareto_results.png")

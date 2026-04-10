"""
fig4_outcome_subplots.py

Generates per-outcome subnetwork plots for EEG preprocessing pipeline analysis.
For each outcome measure (PSD, ERSP, ERD/ERS, CMC), creates a separate network
showing only the preprocessing steps and transitions that appear in studies
computing that outcome.

Output
------
fig4_ia.png       – PSD subnetwork (standalone)
fig4_ib.png       – ERSP subnetwork (standalone)
fig4_ic.png       – ERD/ERS subnetwork (standalone)
fig4_id.png       – CMC subnetwork (standalone)
fig4_combined.png – 2×2 panel of all four
"""

import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
from collections import Counter, defaultdict
from math import sqrt
from utils.config import dir_cleancsv, dir_plots

# Load & merge data
steps_df    = pd.read_csv(os.path.join(dir_cleancsv, "Step_Keywords_cleaned.csv"))
outcomes_df = pd.read_csv(os.path.join(dir_cleancsv, "Outcome_Keywords_cleaned.csv"))

steps_df.columns    = steps_df.columns.str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)
outcomes_df.columns = outcomes_df.columns.str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)

steps_grouped = (
    steps_df
    .groupby("citation")["step_keywords"]
    .apply(lambda x: ";".join(x.dropna()))
    .reset_index()
)
outcomes_grouped = (
    outcomes_df
    .groupby("citation")["outcome_keywords_script"]
    .apply(lambda x: ";".join(x.dropna()))
    .reset_index()
    .rename(columns={"outcome_keywords_script": "outcome_keywords"})
)

df = pd.merge(steps_grouped, outcomes_grouped, on="citation", how="outer").fillna("")

# Stage mapping

stage_map = {
    "Raw data":                     ["Raw data"],
    "Pre ICA - Signal Cleaning":    ["Channel removal", "High-pass filter", "Low-pass filter",
                                     "Bandpass filter", "Notch filter", "Downsample"],
    "Pre ICA - Data Preprocessing": ["Artifact Rejection", "Bad channel detection",
                                     "Re-reference", "Epoching"],
    "ICA":                          ["IC decomposition", "IC rejection"],
    "Post ICA":                     ["Clustering", "Baseline correction",
                                     "Dipole fitting", "Normalization", "Despiking"],
    "Outcome":                      ["PSD", "ERD/ERS", "ERSP", "CMC"],
}

node_stage  = {node: stage for stage, nodes in stage_map.items() for node in nodes}
layer_order = ["Raw data", "Pre ICA - Signal Cleaning", "Pre ICA - Data Preprocessing",
               "ICA", "Post ICA", "Outcome"]
stage_y     = {stage: -i for i, stage in enumerate(layer_order)}

# Colour config
COLOR_MAP = {
    "Raw data":                     "#A9A9A9",
    "Pre ICA - Signal Cleaning":    "#FF8C42",
    "Pre ICA - Data Preprocessing": "#20B2AA",
    "ICA":                          "#9370DB",
    "Post ICA":                     "#D9534F",
    "Outcome":                      "#3CB371",
}

OUTCOMES       = ["PSD", "ERSP", "ERD/ERS", "CMC"]
SUBPLOT_LABELS = {o: lbl for o, lbl in zip(OUTCOMES, ["Bi", "Bii", "Biii", "Biv"])}
FNAME_STEMS = {
    "PSD":     "figBi_psd",
    "ERSP":    "figBii_ersp",
    "ERD/ERS": "figBiii_erd_ers",
    "CMC":     "figBiv_cmc",
}


# Build per-outcome transition counts
def build_outcome_transitions(df: pd.DataFrame, target_outcome: str):
    """Return (step_counts, transition_counts, n_studies) for one outcome."""
    step_counts       = Counter()
    transition_counts = Counter()
    n_studies         = 0

    for _, row in df.iterrows():
        outcomes = [o.strip() for o in row["outcome_keywords"].split(";") if o.strip()]
        if target_outcome not in outcomes:
            continue
        n_studies += 1
        steps = [s.strip() for s in row["step_keywords"].split(";") if s.strip()]
        step_counts.update(steps)
        for i in range(len(steps) - 1):
            transition_counts[(steps[i], steps[i + 1])] += 1
        if steps:
            transition_counts[(steps[-1], target_outcome)] += 1

    return step_counts, transition_counts, n_studies


# Node layout
#    y_scale=1.2  → tighter vertical gaps between stages
#    h_spacing=1.8 → narrower horizontal spread within a stage row

def get_node_positions(G: nx.DiGraph, h_spacing: float = 1.8, y_scale: float = 1.2) -> dict:
    x_counts   = defaultdict(int)
    x_counters = defaultdict(int)

    for node in G.nodes():
        y = stage_y.get(node_stage.get(node, "Raw data"), -10)
        x_counts[y] += 1

    positions = {}
    for node in G.nodes():
        y   = stage_y.get(node_stage.get(node, "Raw data"), -10)
        cnt = x_counts[y]
        idx = x_counters[y]
        x   = (idx - (cnt - 1) / 2.0) * h_spacing
        positions[node] = (x, y * y_scale)
        x_counters[y]  += 1

    return positions


# Edge style
def edge_style(weight: int):
    if   weight <= 2:  return "#C0C0C0",  3.0, 0.60
    elif weight <= 5:  return "#B0B0B0",  5.5, 0.70
    elif weight <= 10: return "#808080",  8.5, 0.75
    elif weight <= 15: return "#6B4C9A", 11.0, 0.80
    elif weight <= 20: return "#1E5A9E", 14.0, 0.85
    elif weight <= 30: return "#5C2D91", 17.0, 0.90
    else:              return "#000000", 21.0, 1.00

# Draw a single subnetwork panel
def draw_subnetwork(
    ax: plt.Axes,
    transition_counts: Counter,
    step_counts: Counter,
    n_studies: int,
    outcome: str,
    panel_label: str,
    fig_subplot_w_in: float,
    show_legend: bool = False,
) -> None:
    """Draw an outcome-specific preprocessing subnetwork onto *ax*."""

    if not transition_counts:
        ax.text(0.5, 0.5, f"No studies found\nfor {outcome}",
                ha="center", va="center", fontsize=20,
                transform=ax.transAxes, color="gray")
        ax.set_title(f"({panel_label})  {outcome}  (n = {n_studies})",
                     fontsize=24, weight="bold", pad=14)
        ax.axis("off")
        return

    # Build graph
    G = nx.DiGraph()
    for (src, dst), weight in transition_counts.items():
        G.add_edge(src, dst, weight=weight)

    node_colors = [COLOR_MAP.get(node_stage.get(n, "Raw data"), "#A9A9A9") for n in G.nodes()]

    base_size         = 3200
    degree_multiplier = 900
    node_sizes = [base_size + degree_multiplier * G.degree(n) for n in G.nodes()]

    pos = get_node_positions(G, h_spacing=1.8, y_scale=1.2)

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        ax=ax,
        edgecolors="black",
        linewidths=2.0,
    )

    # Node labels 
    labels = {}
    for node in G.nodes():
        if node == "Bad channel detection":
            labels[node] = "Bad channel\ndetection"
        elif " " in node and len(node.split()) == 2:
            labels[node] = node.replace(" ", "\n")
        else:
            labels[node] = node

    nx.draw_networkx_labels(
        G, pos, labels=labels,
        font_size=18,         
        font_weight="bold",
        ax=ax,
    )

    # Draw edges using FancyArrowPatch (node-boundary aware)
    xs     = [p[0] for p in pos.values()]
    data_w = (max(xs) - min(xs)) if max(xs) != min(xs) else 1.0
    scale  = fig_subplot_w_in / data_w

    for u, v in G.edges():
        w                = G[u][v]["weight"]
        color, lw, alpha = edge_style(w)
        sx, sy           = pos[u]
        ex, ey           = pos[v]
        dx, dy           = ex - sx, ey - sy
        dist             = sqrt(dx**2 + dy**2)
        if dist == 0:
            continue

        u_sz = base_size + degree_multiplier * G.degree(u)
        v_sz = base_size + degree_multiplier * G.degree(v)

        pts_to_in = 1.0 / 72.0
        u_rad     = sqrt(u_sz / 3.14159) * pts_to_in * scale * 0.15
        v_rad     = sqrt(v_sz / 3.14159) * pts_to_in * scale * 0.15

        a_start = (sx + (dx / dist) * u_rad, sy + (dy / dist) * u_rad)
        a_end   = (ex - (dx / dist) * v_rad, ey - (dy / dist) * v_rad)

        ax.add_patch(FancyArrowPatch(
            posA=a_start, posB=a_end,
            arrowstyle="-|>",
            mutation_scale=20 + lw * 2.8,
            color=color,
            linewidth=lw,
            alpha=alpha,
            zorder=1,
            shrinkA=0, shrinkB=0,
        ))

    # Panel title
    ax.set_title(
        f"{outcome}  (n = {n_studies} studies)",
        fontsize=24, weight="bold", pad=14,
    )
    ax.axis("off")

# Compute per-outcome data + print summary statistics
outcome_data = {}
for outcome in OUTCOMES:
    sc, tc, ns = build_outcome_transitions(df, outcome)
    outcome_data[outcome] = (sc, tc, ns)

    print(f"\n{'='*60}")
    print(f"Outcome: {outcome}  |  Studies: {ns}")
    print(f"{'='*60}")
    print("Preprocessing steps (count, %):")
    for step, cnt in sc.most_common():
        pct = (cnt / ns * 100) if ns else 0
        print(f"  {step:<30} {cnt:>3}  ({pct:.1f}%)")
    print("Top transitions:")
    total_t = sum(tc.values())
    for (src, dst), cnt in tc.most_common(10):
        pct = (cnt / total_t * 100) if total_t else 0
        print(f"  {src} -> {dst:<25} {cnt:>3}  ({pct:.1f}%)")

# Outcome specific subnetworks
subnet_W, subnet_H = 22, 24   # inches

for outcome in OUTCOMES:
    lbl        = SUBPLOT_LABELS[outcome]
    sc, tc, ns = outcome_data[outcome]

    fig_s, ax_s = plt.subplots(figsize=(subnet_W, subnet_H),
                                dpi=300, facecolor="white")
    draw_subnetwork(
        ax=ax_s,
        transition_counts=tc,
        step_counts=sc,
        n_studies=ns,
        outcome=outcome,
        panel_label=lbl,
        fig_subplot_w_in=subnet_W,
        show_legend=True,
    )

    # Two side-by-side legends at figure level
    node_h = [Patch(facecolor=c, edgecolor="black", linewidth=1.5, label=s)
              for s, c in COLOR_MAP.items()]
    edge_h = [
        Line2D([0], [0], color="#C0C0C0", lw=4,  label="1–2 studies"),
        Line2D([0], [0], color="#B0B0B0", lw=5,  label="3–5 studies"),
        Line2D([0], [0], color="#808080", lw=6,  label="6–10 studies"),
        Line2D([0], [0], color="#6B4C9A", lw=7,  label="11–15 studies"),
        Line2D([0], [0], color="#1E5A9E", lw=8,  label="16–20 studies"),
        Line2D([0], [0], color="#5C2D91", lw=9,  label="21–30 studies"),
        Line2D([0], [0], color="#000000", lw=10, label="31+ studies"),
    ]

    # Left box — preprocessing stages
    leg_stages = fig_s.legend(
        handles=node_h,
        title="Preprocessing stage",
        title_fontsize=20,
        fontsize=18,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=3,
        frameon=True,
        fancybox=True,
        edgecolor="black",
        handlelength=2.0,
        columnspacing=1.4,
    )
    leg_stages.get_title().set_weight("bold")
    fig_s.add_artist(leg_stages)

    # Right box — transition frequency
    leg_edges = fig_s.legend(
        handles=edge_h,
        title="Transition frequency",
        title_fontsize=20,
        fontsize=18,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=4,
        frameon=True,
        fancybox=True,
        edgecolor="black",
        handlelength=2.8,
        columnspacing=1.4,
    )
    leg_edges.get_title().set_weight("bold")

    plt.tight_layout(rect=[0, 0.12, 1, 1])
    save_path = dir_plots / f"{FNAME_STEMS[outcome]}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved → {save_path}")


# Combined 2×2 figure  
COMBINED_W, COMBINED_H = 44, 52    # inches
SUBPLOT_W_IN           = COMBINED_W / 2

fig, axes = plt.subplots(2, 2, figsize=(COMBINED_W, COMBINED_H),
                         dpi=300, facecolor="white")

for ax, outcome in zip(axes.flat, OUTCOMES):
    lbl        = SUBPLOT_LABELS[outcome]
    sc, tc, ns = outcome_data[outcome]
    draw_subnetwork(
        ax=ax,
        transition_counts=tc,
        step_counts=sc,
        n_studies=ns,
        outcome=outcome,
        panel_label=lbl,
        fig_subplot_w_in=SUBPLOT_W_IN,
        show_legend=False,
    )

# Shared stage legend
node_handles = [
    Patch(facecolor=c, edgecolor="black", linewidth=1.5, label=s)
    for s, c in COLOR_MAP.items()
]
edge_handles = [
    Line2D([0], [0], color="#C0C0C0", lw=5,  label="1–2 studies"),
    Line2D([0], [0], color="#B0B0B0", lw=6,  label="3–5 studies"),
    Line2D([0], [0], color="#808080", lw=7,  label="6–10 studies"),
    Line2D([0], [0], color="#6B4C9A", lw=8,  label="11–15 studies"),
    Line2D([0], [0], color="#1E5A9E", lw=9,  label="16–20 studies"),
    Line2D([0], [0], color="#5C2D91", lw=10, label="21–30 studies"),
    Line2D([0], [0], color="#000000", lw=11, label="31+ studies"),
]

leg1 = fig.legend(
    handles=node_handles,
    title="Preprocessing stage",
    title_fontsize=28,
    fontsize=26,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.01),
    ncol=3,
    frameon=True,
    fancybox=True,
    columnspacing=1.8,
    handlelength=2.2,
)
leg1.get_title().set_weight("bold")
fig.add_artist(leg1)

leg2 = fig.legend(
    handles=edge_handles,
    title="Transition frequency",
    title_fontsize=28,
    fontsize=26,
    loc="lower center",
    bbox_to_anchor=(0.5, -0.025),
    ncol=4,
    frameon=True,
    fancybox=True,
    columnspacing=1.8,
    handlelength=3.0,
)
leg2.get_title().set_weight("bold")

fig.suptitle(
    "EEG Preprocessing Pipelines Stratified by Neural Outcome Measure",
    fontsize=32, weight="bold", y=1.004,
)

plt.tight_layout(rect=[0, 0.06, 1, 1])
save_combined = dir_plots / "figB_combined.png"
plt.savefig(save_combined, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"\nCombined figure saved → {save_combined}")
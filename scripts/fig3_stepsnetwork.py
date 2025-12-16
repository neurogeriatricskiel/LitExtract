import os
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyArrowPatch
from matplotlib.lines import Line2D
from collections import Counter, defaultdict
from math import sqrt
from utils.config import dir_cleancsv, dir_plots

save_path = dir_plots / "fig3_stepsnetwork.png"

# Load cleaned CSVs
steps_df = pd.read_csv(os.path.join(dir_cleancsv, "Step_Keywords_cleaned.csv"))
outcomes_df = pd.read_csv(os.path.join(dir_cleancsv, "Outcome_Keywords_cleaned.csv"))

# Standardize column names
steps_df.columns = steps_df.columns.str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)
outcomes_df.columns = outcomes_df.columns.str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)

# Merge keywords by study
steps_grouped = steps_df.groupby('citation')['step_keywords'].apply(lambda x: ';'.join(x.dropna())).reset_index()
outcomes_grouped = outcomes_df.groupby('citation')['outcome_keywords_script'].apply(lambda x: ';'.join(x.dropna())).reset_index()
outcomes_grouped.rename(columns={'outcome_keywords_script': 'outcome_keywords'}, inplace=True)

# Merge using lowercase column name 
df = pd.merge(
    steps_grouped,
    outcomes_grouped,
    on='citation',
    how='outer'
).fillna("")

# Preprocessing stages
stage_map = {
    "Raw data": ["Raw data"],
    "Pre ICA - Signal Cleaning": ["Channel removal", "High-pass filter", "Low-pass filter",
                                  "Bandpass filter", "Notch filter", "Downsample"],
    "Pre ICA - Data Preprocessing": ["Artifact Rejection", "Bad channel detection", "Re-reference", "Epoching"],
    "ICA": ["IC decomposition", "IC rejection"],
    "Post ICA": ["Clustering", "Baseline correction", "Dipole fitting", "Normalization", "Despiking"],
    "Outcome": ["PSD", "ERD/ERS", "ERSP", "CMC"]
}

# Count steps and transitions 
step_counts = Counter()
transition_counts = Counter()

for _, row in df.iterrows():
    steps = [s for s in row["step_keywords"].split(";") if s]
    outcomes = [o for o in row["outcome_keywords"].split(";") if o]

    step_counts.update(steps)

    # Step-to-step transitions
    for i in range(len(steps)-1):
        transition_counts[(steps[i], steps[i+1])] += 1
    # Step-to-outcome transitions
    if steps and outcomes:
        last_step = steps[-1]
        for out in outcomes:
            transition_counts[(last_step, out)] += 1

total_studies = len(df)

# Print descriptive statistics with percentages 
print("=== Preprocessing Steps (Count & %) ===")
for step, count in step_counts.most_common():
    pct = (count / total_studies) * 100
    print(f"{step}: {count} ({pct:.1f}%)")

print("\n=== Top 20 Step Transitions (Count & %) ===")
for (src, dst), count in transition_counts.most_common(20):
    pct = (count / sum(transition_counts.values())) * 100
    print(f"{src} -> {dst}: {count} ({pct:.1f}%)")

# Node stage mapping 
node_stage = {key: stage for stage, keys in stage_map.items() for key in keys}
layer_order = ["Raw data", "Pre ICA - Signal Cleaning", "Pre ICA - Data Preprocessing", "ICA", "Post ICA", "Outcome"]
stage_y = {stage: -i for i, stage in enumerate(layer_order)}

def get_node_positions(G, node_stage):
    """Arrange nodes by stage with tighter horizontal spacing."""
    positions = {}
    x_coords = defaultdict(int)
    
    # First pass: count nodes per layer
    for node in G.nodes():
        stage = node_stage.get(node, "Raw data")
        y = stage_y.get(stage, -10)
        x_coords[y] += 1
    
    # Second pass: position nodes with reduced spacing
    x_counters = defaultdict(int)
    horizontal_spacing = 2.5
    
    for node in G.nodes():
        stage = node_stage.get(node, "Raw data")
        y = stage_y.get(stage, -10)
        count = x_coords[y]
        idx = x_counters[y]
        x = (idx - (count - 1)/2.0) * horizontal_spacing
        positions[node] = (x, y * 1.8)
        x_counters[y] += 1
    
    return positions

# Plotting
def plot_preprocessing_flow(transition_counts, node_stage_map, title="EEG Preprocessing Flow Across Studies"):
    G = nx.DiGraph()
    for (src, dst), weight in transition_counts.items():
        G.add_edge(src, dst, weight=weight)

    # Node colors - professional palette
    color_map = {
        "Raw data": "#A9A9A9",
        "Pre ICA - Signal Cleaning": "#FF8C42",
        "Pre ICA - Data Preprocessing": "#20B2AA",
        "ICA": "#9370DB",
        "Post ICA": "#D9534F",
        "Outcome": "#3CB371"
    }

    node_colors = [color_map.get(node_stage_map.get(node, "Raw data"), "gray") for node in G.nodes()]
    
    # Increased node sizes
    base_size = 3500
    degree_multiplier = 1200
    node_sizes = [base_size + degree_multiplier * G.degree(n) for n in G.nodes()]
    
    pos = get_node_positions(G, node_stage_map)

    # Larger figure for better quality with extra space at bottom for legends
    fig, ax = plt.subplots(figsize=(65, 43), dpi=500)
    
    # Draw nodes with edge for better definition
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, 
                          ax=ax, edgecolors='black', linewidths=3)
    
    # Much larger, bold labels with multi-word wrapping
    labels = {}
    for node in G.nodes():
        # Special case for three-word "Bad channel detection"
        if node == "Bad channel detection":
            labels[node] = "Bad channel\ndetection"
        # Split two-word labels into two lines
        elif ' ' in node and len(node.split()) == 2:
            labels[node] = node.replace(' ', '\n')
        else:
            labels[node] = node
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=65, font_weight="bold", 
                           font_family='sans-serif', ax=ax)

    # Edge styling with significantly increased thickness, bolder for highest frequencies
    def edge_style(weight):
        if weight <= 5:
            return "#B0B0B0", 7.0, 0.7
        elif weight <= 10:
            return "#808080", 10.0, 0.75
        elif weight <= 15:
            return "#6B4C9A", 13.0, 0.8
        elif weight <= 20:
            return "#1E5A9E", 16.0, 0.85
        elif weight <= 30:
            return "#5C2D91", 20.0, 0.9
        elif weight <= 35:
            return "#2C1810", 24.0, 0.95
        else:
            return "#000000", 30.0, 1.0

    # Draw edges with arrows that connect directly node-to-node
    for u, v in G.edges():
        w = G[u][v]['weight']
        color, width, alpha = edge_style(w)
        start, end = pos[u], pos[v]
        
        # Calculate direction vector
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = sqrt(dx**2 + dy**2)
        
        if dist == 0:
            continue
        
        # Get node sizes
        u_size = base_size + degree_multiplier * G.degree(u)
        v_size = base_size + degree_multiplier * G.degree(v)
        
        # Calculate radius from node size (size is in points squared)
        # Use figure DPI to convert properly
        points_to_inches = 1.0 / 72.0
        fig_width_inches = 55
        data_width = max([p[0] for p in pos.values()]) - min([p[0] for p in pos.values()])
        if data_width == 0:
            data_width = 1
        scale = fig_width_inches / data_width
        
        u_radius = sqrt(u_size / 3.14159) * points_to_inches * scale * 0.15
        v_radius = sqrt(v_size / 3.14159) * points_to_inches * scale * 0.15
        
        # Start arrow from edge of source node
        arrow_start = (start[0] + (dx/dist) * u_radius, start[1] + (dy/dist) * u_radius)
        # End arrow at edge of target node
        arrow_end = (end[0] - (dx/dist) * v_radius, end[1] - (dy/dist) * v_radius)
        
        # Arrow head size proportional to edge width
        mutation_scale = 25 + width * 3.0
        
        arrow = FancyArrowPatch(
            posA=arrow_start, 
            posB=arrow_end,
            arrowstyle="-|>",
            mutation_scale=mutation_scale, 
            color=color, 
            linewidth=width, 
            alpha=alpha,
            zorder=1,
            shrinkA=0,
            shrinkB=0
        )
        ax.add_patch(arrow)

    # Enhanced legends with increased font sizes
    node_legend_handles = [Patch(facecolor=c, edgecolor="black", linewidth=2, label=stage) 
                          for stage, c in color_map.items()]
    
    edge_legend_handles = [
        Line2D([0], [0], color="#B0B0B0", lw=8, label="1–5 articles"),
        Line2D([0], [0], color="#808080", lw=9, label="6–10 articles"),
        Line2D([0], [0], color="#6B4C9A", lw=10, label="11–15 articles"),
        Line2D([0], [0], color="#1E5A9E", lw=11, label="16–20 articles"),
        Line2D([0], [0], color="#5C2D91", lw=12, label="21–30 articles"),
        Line2D([0], [0], color="#2C1810", lw=13, label="31–35 articles"),
        Line2D([0], [0], color="#000000", lw=14, label="36+ articles")
    ]
    
    # Create two legends in separate rows at the bottom
    legend1 = ax.legend(
        handles=node_legend_handles, 
        title="Preprocessing Stages", 
        title_fontsize=58,
        fontsize=56, 
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.03),
        ncol=6,
        frameon=True,
        fancybox=True,
        shadow=True,
        columnspacing=2.0,
        handlelength=2.5,
        handleheight=2.0
    )
    legend1.get_title().set_weight('bold')
    
    ax.add_artist(legend1)
    
    legend2 = ax.legend(
        handles=edge_legend_handles, 
        title="Step Transition Frequency", 
        title_fontsize=58,
        fontsize=56, 
        loc="upper center", 
        bbox_to_anchor=(0.5, -0.15),
        ncol=7,
        frameon=True,
        fancybox=True,
        shadow=True,
        columnspacing=2.0,
        handlelength=3.0,
        handleheight=1.5
    )
    legend2.get_title().set_weight('bold')

    plt.title(title, fontsize=70, weight="bold", pad=40)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight", facecolor='white')
    plt.show()
    print(f"\nHigh-quality plot saved to: {save_path}")

# Run plot
plot_preprocessing_flow(transition_counts, node_stage)

print(f"\nPlot saved to: {save_path}")
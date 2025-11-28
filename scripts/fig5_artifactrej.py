import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import colorsys
from utils.config import dir_cleancsv, dir_plots

# Load data
data_path = os.path.join(dir_cleancsv, "Artifact_Methods_cleaned.csv")
save_path = dir_plots / "fig5_artifactrej.png"
df_artifact = pd.read_csv(data_path)

# Fill missing citation and strip whitespace
df_artifact["Citation"] = df_artifact["citation"].fillna("Unknown Study").astype(str).str.strip()

# Create short labels: first 4 letters of last name + year
def short_label(citation):
    parts = citation.split(",")
    last_name = parts[0][:4] if len(parts) > 0 else "Unkn"
    year = parts[-1].strip()[-4:] if len(parts) > 1 else "0000"
    return f"{last_name}{year}"

df_artifact["ShortLabel"] = df_artifact["Citation"].apply(short_label)

# Order studies by year
if "Year" in df_artifact.columns:
    df_artifact = df_artifact.sort_values("Year")

studies = df_artifact["ShortLabel"].unique()

# Prepare pivot table
methods = df_artifact["artifactrej_methods"].dropna().unique()
pivot = pd.DataFrame(0, index=studies, columns=methods)
for key, group in df_artifact.groupby("ShortLabel"):
    for m in group["artifactrej_methods"]:
        if pd.notna(m):
            pivot.loc[key, m] = 1


# Descriptive statistics 
method_counts = df_artifact["artifactrej_methods"].value_counts()
avg_methods_per_study = pivot.sum(axis=1).mean()
multi_method_studies = (pivot.sum(axis=1) > 1).sum()

print("Descriptive Statistics")
print(f"Total studies with artifact rejection analyzed: {len(studies)}")
print(f"Total unique artifact rejection methods: {len(methods)}\n")
print("Most common methods:")
print(method_counts.head(10).to_string())
print(f"\nAverage number of methods per study: {avg_methods_per_study:.2f}")
print(f"Number of studies using multiple methods: {multi_method_studies}")

# Assign colors
palette_base = sns.color_palette("tab20", n_colors=len(methods))
method_base_colors = {method: palette_base[i % len(palette_base)] for i, method in enumerate(methods)}

# Map frequency to intensity (darker = more common)
max_count = method_counts.max()
method_colors = {}
min_light, max_light = 0.3, 0.9
for method in methods:
    r, g, b = method_base_colors[method]
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    freq = method_counts.get(method, 0)
    l_new = max_light - (freq / max_count) * (max_light - min_light)
    r_new, g_new, b_new = colorsys.hls_to_rgb(h, l_new, s)
    method_colors[method] = (r_new, g_new, b_new)

# Plot 
fig_width = max(24, len(pivot) * 0.25)
fig_height = 15
plt.figure(figsize=(fig_width, fig_height), dpi=600)

bottoms = np.zeros(len(pivot))
for method in pivot.columns:
    plt.bar(
        pivot.index,
        pivot[method],
        bottom=bottoms,
        color=method_colors[method],
        width=0.9
    )
    bottoms += pivot[method].values

plt.xlabel("Studies", fontsize=22, weight="bold")
plt.ylabel("Artifact Rejection Methods used (n per study)", fontsize=22, weight="bold")
plt.title("Artifact Rejection Methods Across Studies", fontsize=26, weight="bold", pad=15)

# Rotate x-axis labels
plt.xticks(rotation=55, ha='right', fontsize=20)
plt.yticks(fontsize=22)

plt.grid(axis="y", linestyle="--", alpha=0.4)

# Legend below
sorted_methods = method_counts.sort_values(ascending=False).index.tolist()
legend_patches = [
    Patch(color=method_colors[m], label=f"{m} ({method_counts[m]})") for m in sorted_methods
]
plt.legend(
    handles=legend_patches,
    loc='upper center',
    bbox_to_anchor=(0.5, -0.25),
    ncol=4,
    fontsize=20,
    title="Artifact Rejection Methods (Total Count)",
    title_fontsize=22,
    frameon=False
)
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig(save_path, dpi=600, bbox_inches="tight")
plt.show()

print(f"\nPlot saved to: {save_path}")
